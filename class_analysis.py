import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.models as models
from captum.attr import IntegratedGradients, Saliency
from tqdm import tqdm

from utils import (
    set_seeds,
    get_imagenet_dataloader as get_dataloader,
    compute_sparsity_patch_wise,
    probability_drop_patch_wise,
    compute_sparsity_pixel_wise,
    probability_drop_pixel_wise,
)

# ==========================================================================
# 1. Load neuron mask

def build_mask_and_block(model: torch.nn.Module, stats_dir: Path):
    manifest = json.loads((stats_dir / "manifest.json").read_text())
    selected_block_name = manifest["selected_block"]
    cluster_results = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)
    labels = cluster_results["labels"]

    mask = np.ones(len(labels), dtype=np.float32) # zero out the cluster later in main()

    block_map = {
        "block_1": model.features.denseblock1,
        "block_2": model.features.denseblock2,
        "block_3": model.features.denseblock3,
        "block_4": model.features.denseblock4,
    }
    target_block = block_map[selected_block_name]
    return torch.tensor(mask), target_block, selected_block_name

# ==========================================================================
# 2. Filtered model wrapper

class FilteredDenseNet(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mask: torch.Tensor, target_block: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mask", mask.view(1, -1, 1, 1))
        self.target_block = target_block

    def forward(self, x):
        # hook defined inside to ensure fresh registration per forward
        def hook_fn(_module, _input, output):
            keep = output * self.mask  # gradients flow
            supp = (output * (1 - self.mask)).detach()  # blocked gradients
            return keep + supp

        handle = self.target_block.register_forward_hook(hook_fn)
        out = self.model(x)
        handle.remove()
        return out


# ==========================================================================
# MAIN

def main():
    # Load ImageNet mapping for readable class names
    try:
        with open("data/imagenet_class_index.json", "r") as fp:
            idx2name = {int(k): v[1] for k, v in json.load(fp).items()}
    except FileNotFoundError:
        idx2name = {}
    set_seeds(42)  # SEED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.densenet121(pretrained=True).to(device).eval()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_idx", type=int, default=0)
    parser.add_argument("--class_start", type=int, default=600, help="Starting class index (inclusive)")
    parser.add_argument("--class_end", type=int, default=800, help="Ending class index (exclusive)")
    parser.add_argument("--images_per_class", type=int, default=50, help="50 is full class (max)")
    parser.add_argument("--stats_dir", type=str, default="outputs/neuron_stats/block_1_k3")
    args = parser.parse_args()

    # Build mask
    stats_dir = Path(args.stats_dir)
    if not stats_dir.exists():
        raise FileNotFoundError(f"Neuron stats directory {stats_dir} not found. Run neuron_stats.py first.")
    full_mask, target_block, block_name = build_mask_and_block(base_model, stats_dir)

    # cluster choice is always explicit now
    chosen_cluster = args.cluster_idx

    mask = np.ones_like(full_mask.cpu().numpy())
    cluster_results = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)
    labels = cluster_results["labels"]

    # Select all channels in the chosen cluster
    cluster_indices = np.where(labels == chosen_cluster)[0]

    mask[cluster_indices] = 0.0  # set to suppress
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    print(f"Masking cluster {chosen_cluster} in {block_name}")

    filt_model = FilteredDenseNet(base_model, mask_tensor.to(device), target_block).to(device).eval()

    # Data loading 
    dl = get_dataloader(batch_size=32, num_workers=4, shuffle=False)
    dataset_full = dl.dataset

    # Select image indices for the requested classes
    targets = dataset_full.targets 
    
    # Use direct class range
    selected_class_ids = list(range(args.class_start, min(args.class_end, 1000)))
    
    print(f"Processing classes {args.class_start} to {args.class_end-1} ({len(selected_class_ids)} classes total)")

    class_to_indices: Dict[int, List[int]] = {c: [] for c in selected_class_ids}

    for idx, lbl in enumerate(targets):
        if lbl in class_to_indices and len(class_to_indices[lbl]) < args.images_per_class:
            class_to_indices[lbl].append(idx)
        # exit early if all filled
        if all(len(v) == args.images_per_class for v in class_to_indices.values()):
            break

    # Pre-load all selected images to CPU RAM   
    selected_indices = [idx for inds in class_to_indices.values() for idx in inds]
    images_flat: List[torch.Tensor] = []
    labels_flat: List[int] = []
    for idx in tqdm(selected_indices, desc="Pre-loading images"):
        img_cpu, lbl = dataset_full[idx]  
        images_flat.append(img_cpu)
        labels_flat.append(lbl)
    # Map original dataset index → position in the flat list (O(1) lookup later)
    idx_to_flatpos = {orig_idx: pos for pos, orig_idx in enumerate(selected_indices)}

    # Attribution methods (objects):
    attr_methods_filtered = {
        "ig": IntegratedGradients(filt_model),
        "saliency": Saliency(filt_model),
    }
    attr_methods_base = {
        "ig": IntegratedGradients(base_model),
        "saliency": Saliency(base_model),
    }

    # Attribution cache directories, add class range to avoid conflicts
    filtered_attr_dir = Path(f"outputs/class_analysis_metrics_{args.class_start}_{args.class_end}.csv").parent / f"attributions_filtered_{args.class_start}_{args.class_end}"
    filtered_attr_dir.mkdir(parents=True, exist_ok=True)

    baseline_attr_dir = Path("outputs/baseline/attributions")
    baseline_attr_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-class metric aggregation
    
    rows = []  # list of dicts to be written as CSV

    for class_id in selected_class_ids:
        if class_id not in class_to_indices:
            continue
            
        indices = class_to_indices[class_id]
        class_name = idx2name.get(class_id, str(class_id))
        print(f"Processing class {class_id} ({class_name}) with {len(indices)} images …")

        # Collect per-image metrics
        sparsity_scores: Dict[str, List[float]] = {"ig": [], "saliency": []}  
        faithfulness_scores_filtered: Dict[str, List[float]] = {"ig": [], "saliency": []}
        baseline_faithfulness_scores: Dict[str, List[float]] = {"ig": [], "saliency": []}
        baseline_sparsity_scores: Dict[str, List[float]] = {"ig": [], "saliency": []}

        for dataset_idx in tqdm(indices, desc=f"Class {class_id}"):
            flat_pos = idx_to_flatpos[dataset_idx]
            img_cpu = images_flat[flat_pos]  # cached CPU tensor
            tgt_label = labels_flat[flat_pos]
            img = img_cpu.to(device, non_blocking=True)

            for method_name in ["ig", "saliency"]:
                # Filtered attribution
                filt_attr_path = filtered_attr_dir / f"{method_name}_{dataset_idx}.npy"
                if filt_attr_path.exists(): # cache
                    attr_filtered = np.load(filt_attr_path)
                else:
                    attr_filtered_tensor = attr_methods_filtered[method_name].attribute(
                        img.unsqueeze(0), target=int(tgt_label)
                    )
                    attr_filtered = attr_filtered_tensor.squeeze().cpu().numpy()
                    np.save(filt_attr_path, attr_filtered)

                sparsity_scores[method_name].append(
                    compute_sparsity_pixel_wise(attr_filtered)
                )
                faithfulness_scores_filtered[method_name].append(
                    probability_drop_pixel_wise(
                        filt_model, img, attr_filtered, drop_fraction=0.1
                    )
                )

                # Baseline attribution
                base_attr_path = baseline_attr_dir / f"{method_name}_{dataset_idx}.npy"
                if base_attr_path.exists(): # cache
                    attr_base = np.load(base_attr_path)
                else:
                    attr_base_tensor = attr_methods_base[method_name].attribute(
                        img.unsqueeze(0), target=int(tgt_label)
                    )
                    attr_base = attr_base_tensor.squeeze().cpu().numpy()
                    np.save(base_attr_path, attr_base)
                baseline_faithfulness_scores[method_name].append(
                    probability_drop_pixel_wise(
                        base_model, img, attr_base, drop_fraction=0.1
                    )
                )
                baseline_sparsity_scores[method_name].append(
                    compute_sparsity_pixel_wise(attr_base)
                )

        # Aggregate for this class
        for method_name in ["ig", "saliency"]:
            diff_arr = (np.array(baseline_faithfulness_scores[method_name]) - np.array(faithfulness_scores_filtered[method_name]))
            mean_diff = float(np.mean(diff_arr))

            row = {
                "class_id": class_id,
                "class_name": class_name,
                "method": method_name,
                # Filtered metrics
                "filtered_sparsity_mean": float(np.mean(sparsity_scores[method_name])),
                "filtered_sparsity_std": float(np.std(sparsity_scores[method_name])),
                "filtered_faithfulness_mean": float(np.mean(faithfulness_scores_filtered[method_name])),
                "filtered_faithfulness_std": float(np.std(faithfulness_scores_filtered[method_name])),
                # Baseline metrics
                "baseline_sparsity_mean": float(np.mean(baseline_sparsity_scores[method_name])),
                "baseline_sparsity_std": float(np.std(baseline_sparsity_scores[method_name])),
                "baseline_faithfulness_mean": float(np.mean(baseline_faithfulness_scores[method_name])),
                "baseline_faithfulness_std": float(np.std(baseline_faithfulness_scores[method_name])),
                # Paired difference
                "mean_diff": mean_diff,
            }
            rows.append(row)

    # ------------------------------------------------------------------
    # Write CSV
    
    output_path = Path(f"outputs/class_analysis_metrics_{args.class_start}_{args.class_end}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "class_id",
        "class_name",
        "method",
        "filtered_sparsity_mean",
        "filtered_sparsity_std",
        "filtered_faithfulness_mean",
        "filtered_faithfulness_std",
        "baseline_sparsity_mean",
        "baseline_sparsity_std",
        "baseline_faithfulness_mean",
        "baseline_faithfulness_std",
        "mean_diff",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Class-wise metrics saved to {output_path}")
    print(f"Processed {len(selected_class_ids)} classes from {args.class_start} to {args.class_end-1}")


if __name__ == "__main__":
    main()
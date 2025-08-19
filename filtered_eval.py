import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from captum.attr import IntegratedGradients, Saliency
from tqdm import tqdm

from utils import (
    set_seeds,
    get_imagenet_dataloader as get_dataloader,
    compute_sparsity_pixel_wise,
    compute_sparsity_patch_wise,
    probability_drop_pixel_wise,
    probability_drop_patch_wise,
)


# ============================================================================
# 1. Load neuron mask

def build_mask_and_block(model: torch.nn.Module, stats_dir: Path):
    manifest = json.loads((stats_dir / "manifest.json").read_text())
    selected_block_name = manifest["selected_block"]
    cluster_results = np.load(stats_dir / "cluster_results.npz", allow_pickle=True) 
    labels = cluster_results["labels"]
    n_clusters = int(cluster_results["n_clusters"])

    # initialise an all-ones mask; we zero out the chosen cluster later in main()
    mask = np.ones(len(labels), dtype=np.float32)

    block_map = {
        "block_1": model.features.denseblock1,
        "block_2": model.features.denseblock2,
        "block_3": model.features.denseblock3,
        "block_4": model.features.denseblock4,
    }
    target_block = block_map[selected_block_name]
    return torch.tensor(mask), target_block, selected_block_name


# ============================================================================
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
            supp = (output * (1 - self.mask)).detach() # blocked gradients
            return keep + supp

        handle = self.target_block.register_forward_hook(hook_fn)
        out = self.model(x)
        handle.remove()
        return out


# ============================================================================
# MAIN

def main():
    set_seeds(42) # SEED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.densenet121(pretrained=True).to(device).eval()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_idx", type=int, default=0) # index of the cluster to filter
    parser.add_argument("--metric", choices=["patch", "pixel"], default="patch")
    parser.add_argument("--images", type=int, default=10000)
    parser.add_argument("--stats_dir", type=str, default="outputs/neuron_stats/block_1_k3")
    parser.add_argument("--output_dir", type=str, default="outputs/filtered_block_1_cluster_0_k3")
    
    
    args = parser.parse_args()

    # Build mask
    stats_dir = Path(args.stats_dir)
    if not stats_dir.exists():
        raise FileNotFoundError(f"Neuron stats directory {stats_dir} not found. Run neuron_stats.py first.")
    full_mask, target_block, block_name = build_mask_and_block(base_model, stats_dir)

    # select cluster by explicit index only
    chosen_cluster = args.cluster_idx

    mask = np.ones_like(full_mask.cpu().numpy())
    cluster_results = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)
    labels = cluster_results["labels"]

    # ----------------------------------------------------------------------
    # 3. Determine which channel indices to mask
  
    cluster_indices = np.where(labels == chosen_cluster)[0]

    mask[cluster_indices] = 0.0  # set to suppress
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    print(f"Masking cluster {chosen_cluster} in {block_name}")

    filt_model = FilteredDenseNet(base_model, mask_tensor.to(device), target_block).to(device).eval()

    # Attribution generation
    dl = get_dataloader(batch_size=32, num_workers=4)  # full loader for attribution subset
    out_root = Path(args.output_dir)
    out_attr_dir = out_root / "attributions"
    out_attr_dir.mkdir(parents=True, exist_ok=True)

    attr_methods = {
        "ig": IntegratedGradients(filt_model),
        "saliency": Saliency(filt_model),
    }

    collected = {m: 0 for m in attr_methods}
    max_images = args.images

    dataset_full = dl.dataset
    stride = max(1, len(dataset_full) // max_images)
    indices = [i * stride for i in range(max_images)]
    images_flat: List[torch.Tensor] = [] 
    labels_flat: List[int] = []
    for i in indices:
        img, lbl = dataset_full[i]
        images_flat.append(img)  # keep cpu tensor
        labels_flat.append(lbl)

    for idx, (img_cpu, tgt_label) in tqdm(list(enumerate(zip(images_flat, labels_flat))), desc="Attributions"):
        img = img_cpu.to(device, non_blocking=True)
        for name, method in attr_methods.items():
            attr = method.attribute(img.unsqueeze(0), target=int(tgt_label))
            np.save(out_attr_dir / f"{name}_{idx}.npy", attr.squeeze().cpu().numpy())
            collected[name] += 1

    # Metric computation
    sparsity_scores: Dict[str, list] = {m: [] for m in attr_methods}
    faithfulness_scores: Dict[str, list] = {m: [] for m in attr_methods}

    # choose functions per metric type
    if args.metric == "pixel":
        sparsity_fn = compute_sparsity_pixel_wise
        faith_fn = probability_drop_pixel_wise
    else:  # patch-wise
        sparsity_fn = compute_sparsity_patch_wise
        faith_fn = probability_drop_patch_wise

    # accuracy counters
    correct_filtered = 0
    for idx, img_cpu in tqdm(list(enumerate(images_flat)), desc="Metrics"):
        img = img_cpu.to(device, non_blocking=True)
        # accuracy check
        with torch.no_grad():
            pred_filt = filt_model(img.unsqueeze(0)).argmax(1).item()
        if pred_filt == labels_flat[idx]:
            correct_filtered += 1
        for method in attr_methods.keys():
            attr = np.load(out_attr_dir / f"{method}_{idx}.npy")
            sparsity_scores[method].append(sparsity_fn(attr))
            faithfulness_scores[method].append(
                faith_fn(filt_model, img, attr, drop_fraction=0.1)
            )

    # ------------------------------------------------------------------
    # Aggregate summary dictionary (mean & std only)
    
    accuracy_filtered = correct_filtered / len(images_flat) if images_flat else 0.0

    results = {}
    for method in attr_methods.keys():
        res = {
            "sparsity_mean": float(np.mean(sparsity_scores[method])),
            "sparsity_std": float(np.std(sparsity_scores[method])),
            "faithfulness_mean": float(np.mean(faithfulness_scores[method])),
            "faithfulness_std": float(np.std(faithfulness_scores[method])),
        }
        results[method] = res

    # add overall accuracy
    results["accuracy_top1"] = accuracy_filtered

    out_metrics = out_root / "metrics.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(results, indent=2))
    print("Filtered metrics saved to", out_metrics)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main() 
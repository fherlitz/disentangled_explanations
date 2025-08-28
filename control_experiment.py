import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
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
# 1. Random neuron mask generation

def build_random_mask_and_block(model: torch.nn.Module, block_name: str, n_random: int, seed: int = 42):
    block_map = {
        "block_1": model.features.denseblock1,
        "block_2": model.features.denseblock2,
        "block_3": model.features.denseblock3,
        "block_4": model.features.denseblock4,
    }
    
    if block_name not in block_map:
        raise ValueError(f"Unknown block name: {block_name}")
    
    target_block = block_map[block_name]
    
    block_channel_counts = {
        "block_1": 256,
        "block_2": 512,
        "block_3": 1024,
        "block_4": 1024,
    }
    
    total_neurons = block_channel_counts[block_name]
    
    # Create random mask
    np.random.seed(seed)
    mask = np.ones(total_neurons, dtype=np.float32)
    
    # Randomly select neurons to filter
    random_indices = np.random.choice(total_neurons, n_random, replace=False)
    mask[random_indices] = 0.0 
    
    print(f"Random filtering in {block_name}:")
    print(f"Total neurons: {total_neurons}")
    print(f"Neurons to filter: {n_random}")
    print(f"Neurons to keep: {total_neurons - n_random}")
    print(f"Random indices (first 10): {sorted(random_indices)[:10]}")
    
    return torch.tensor(mask), target_block, block_name

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
    parser.add_argument("--block", choices=["block_1", "block_2", "block_3", "block_4"], default="block_1")
    parser.add_argument("--n_random", type=int, default=214)
    parser.add_argument("--metric", choices=["patch", "pixel"], default="patch")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)    
    args = parser.parse_args()

    output_dir = f"outputs/control_random_{args.block}_{args.n_random}_seed{args.seed}"

    # Build random mask
    mask_tensor, target_block, block_name = build_random_mask_and_block(
        base_model, args.block, args.n_random, args.seed
    )

    print(f"Control experiment: Random filtering {args.n_random} neurons from {block_name}")

    filt_model = FilteredDenseNet(base_model, mask_tensor.to(device), target_block).to(device).eval()

    # Attribution generation
    dl = get_dataloader(batch_size=32, num_workers=4)
    out_root = Path(output_dir)
    out_attr_dir = out_root / "attributions"
    out_attr_dir.mkdir(parents=True, exist_ok=True)

    attr_methods = {
        "ig": IntegratedGradients(filt_model),
        "saliency": Saliency(filt_model),
    }

    max_images = args.images
    dataset_full = dl.dataset
    
    # Use same sampling strategy as main experiments for consistency
    stride = max(1, len(dataset_full) // max_images)
    indices = [i * stride for i in range(max_images)]
    images_flat: List[torch.Tensor] = [] 
    labels_flat: List[int] = []
    for i in indices:
        img, lbl = dataset_full[i]
        images_flat.append(img)  # keep cpu tensor
        labels_flat.append(lbl)

    print(f"Generating attributions for {len(images_flat)} images...")
    
    for idx, (img_cpu, tgt_label) in tqdm(list(enumerate(zip(images_flat, labels_flat))), desc="Attributions"):
        img = img_cpu.to(device, non_blocking=True)
        # use top predicted class from filtered model as attribution target
        with torch.no_grad():
            pred_class = filt_model(img.unsqueeze(0)).argmax(1).item()
        for name, method in attr_methods.items():
            attr = method.attribute(img.unsqueeze(0), target=int(pred_class))
            np.save(out_attr_dir / f"{name}_{idx}.npy", attr.squeeze().cpu().numpy())

    # Metric computation
    sparsity_scores: Dict[str, list] = {m: [] for m in attr_methods}
    faithfulness_scores: Dict[str, list] = {m: [] for m in attr_methods}

    # Choose functions per metric type
    if args.metric == "pixel":
        sparsity_fn = compute_sparsity_pixel_wise
        faith_fn = probability_drop_pixel_wise
    else:  # patch-wise
        sparsity_fn = compute_sparsity_patch_wise
        faith_fn = probability_drop_patch_wise

    print(f"Computing metrics using {args.metric}-wise evaluation...")
    for idx, img_cpu in tqdm(list(enumerate(images_flat)), desc="Metrics"):
        img = img_cpu.to(device, non_blocking=True)
        # Compute metrics for each attribution method
        for method in attr_methods.keys():
            attr = np.load(out_attr_dir / f"{method}_{idx}.npy")
            sparsity_scores[method].append(sparsity_fn(attr))
            faithfulness_scores[method].append(
                faith_fn(filt_model, img, attr, drop_fraction=0.1)
            )

    # Aggregate results
    results = {}
    for method in attr_methods.keys():
        results[method] = {
            "sparsity_mean": float(np.mean(sparsity_scores[method])),
            "sparsity_std": float(np.std(sparsity_scores[method])),
            "faithfulness_mean": float(np.mean(faithfulness_scores[method])),
            "faithfulness_std": float(np.std(faithfulness_scores[method])),
        }

    # Add experiment metadata
    results["experiment_type"] = "control_random"
    results["block_filtered"] = block_name
    results["neurons_filtered"] = args.n_random
    results["neurons_total"] = int(mask_tensor.numel())
    results["random_seed"] = args.seed
    results["metric_type"] = args.metric

    # Save results
    out_metrics = out_root / "metrics.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print("CONTROL EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Block: {block_name}")
    print(f"Random neurons filtered: {args.n_random}")
    print(f"Metric type: {args.metric}-wise")
    print()
    for method in attr_methods.keys():
        method_results = results[method]
        print(f"{method.upper()}:")
        print(f"  Faithfulness: {method_results['faithfulness_mean']:.2f}% ± {method_results['faithfulness_std']:.2f}%")
        print(f"  Sparsity:     {method_results['sparsity_mean']:.2f}% ± {method_results['sparsity_std']:.2f}%")
        print()
    print("Control experiment metrics saved to", out_metrics)
    print(f"\nTo compare with main results:")
    print(f"  Main (Block 1, C0): IG Faith ~43.60%, Saliency Faith ~36.98%")
    print(f"  Control (Random):   IG Faith {results['ig']['faithfulness_mean']:.2f}%, Saliency Faith {results['saliency']['faithfulness_mean']:.2f}%")

if __name__ == "__main__":
    main()
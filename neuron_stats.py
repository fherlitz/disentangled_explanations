import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.cluster import KMeans
from tqdm import tqdm
from utils import set_seeds, get_imagenet_dataloader as get_dataloader

# calculate activation rate, mean activation, and variance
def calculate_stats(activations: np.ndarray, threshold: float = 0.1):
    activation_rate = (activations > threshold).mean(axis=0) # activation rate
    mean_activation = activations.mean(axis=0)
    variance = activations.var(axis=0)
    return activation_rate, mean_activation, variance

# Collect spatially averaged activations for max_samples images
def analyze_block_activations(
    model: torch.nn.Module,
    block: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> np.ndarray:
    
    activations_list: List[np.ndarray] = []

    def hook_f(_module, _input, output):
        batch_act = output.mean(dim=[2, 3]).detach().cpu().numpy() # spatially averaged activations
        activations_list.append(batch_act)

    hook = block.register_forward_hook(hook_f)
    collected = 0
    model.eval()
    with torch.no_grad():
        # collect activations for 1000 images
        for imgs, lbls in tqdm(dataloader, desc="Collecting activations"):
            imgs = imgs.to(device)
            hook_f.batch_labels = lbls.numpy() 
            model(imgs)
            collected += imgs.size(0) # size(0) is the number of images in the batch
            if collected >= max_samples:
                break

    hook.remove()
    activations = np.concatenate(activations_list, axis=0)[:max_samples]
    return activations # 2D array of shape (max_samples, num_neurons)

# ============================================================================
# MAIN

def main():
    set_seeds(42) # SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=["block_1", "block_2", "block_3", "block_4"], default="block_1")
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument( "--output_dir", type=str, default=None,)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True).to(device)

    blocks_to_analyze = {
        "block_1": model.features.denseblock1,
        "block_2": model.features.denseblock2,
        "block_3": model.features.denseblock3,
        "block_4": model.features.denseblock4,
    }

    dataloader = get_dataloader(batch_size=32, num_workers=4)

    block_stats: Dict[str, Dict[str, np.ndarray]] = {} # block name -> {activation_rate, mean_activation, variance}

    # determine which blocks to process for activations
    blocks_for_collection = blocks_to_analyze.keys() if args.block == "auto" else [args.block]

    for name in blocks_for_collection:
        block = blocks_to_analyze[name]
        activations = analyze_block_activations(model, block, dataloader, device, max_samples=1000) 
        activation_rate, mean_act, var_act = calculate_stats(activations)
        block_stats[name] = {
            "activation_rate": activation_rate,
            "mean_activation": mean_act,
            "variance": var_act,
        }

    selected_block = args.block

    # clustering
    props = np.stack(
        [
            block_stats[selected_block]["activation_rate"],
            block_stats[selected_block]["mean_activation"],
            block_stats[selected_block]["variance"],
        ],
        axis=1,
    )

    props = (props - props.mean(axis=0)) / props.std(axis=0)

    n_clusters = args.n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(props)
    labels = kmeans.labels_

    # ================================================
    # Save everything

    # Determine target output directory
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/neuron_stats") / f"{selected_block}_k{n_clusters}"
    # Ensure directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # save per-block stats
    for name, stats in block_stats.items():
        np.savez(
            out_dir / f"stats_{name}.npz",
            activation_rate=stats["activation_rate"],
            mean_activation=stats["mean_activation"],
            variance=stats["variance"],
        )

    # save clustering results
    np.savez(
        out_dir / "cluster_results.npz",
        selected_block=selected_block,
        labels=labels,
        n_clusters=n_clusters,
    )

    # store a JSON manifest for convenience
    manifest = {
        "selected_block": selected_block,
        "n_clusters": n_clusters,
        "num_neurons": len(labels),
        "stats_files": {name: f"stats_{name}.npz" for name in block_stats.keys()},
    }
    with open(out_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("Neuron statistics and cluster assignments saved to", out_dir)


if __name__ == "__main__":
    main() 

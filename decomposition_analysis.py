import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm
from captum.attr import IntegratedGradients, Saliency
from pathlib import Path
from typing import Dict, List, Tuple
from filtered_eval import build_mask_and_block, FilteredDenseNet

# ============================================================================
# Helpers

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def attribution_overlay(img_t: torch.Tensor, attr: np.ndarray) -> np.ndarray:
    # Convert attr to tensor if it's numpy
    if isinstance(attr, np.ndarray):
        attr = torch.from_numpy(attr)
    
    # Sum over channels and take absolute value
    heat = attr.abs().sum(0)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8) # Normalize
    heat = heat.cpu().numpy()
    
    # Get original image
    img = (img_t * STD.to(img_t.device) + MEAN.to(img_t.device)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    
    # Apply colormap
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[..., :3]
    
    # Overlay with 60% original image, 40% heatmap
    overlay = 0.5 * img + 0.5 * heat_rgb
    return overlay

# ============================================================================
# Per-cluster attribution extraction

class ClusterAttributionExtractor:
    
    def __init__(self, model: torch.nn.Module, stats_dir: Path, device: torch.device):
        self.device = device
        self.base_model = model
        self.stats_dir = stats_dir
        
        # Load clustering information
        self.full_mask, self.target_block, self.block_name = build_mask_and_block(self.base_model, self.stats_dir)
        self.labels = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)["labels"]
        self.n_clusters = int(np.load(stats_dir / "cluster_results.npz", allow_pickle=True)["n_clusters"])
        
        # Build models for each cluster (keeping only that cluster active)
        self.cluster_models = self._build_cluster_models()
        
    def _build_cluster_models(self) -> Dict[int, torch.nn.Module]:
        models = {}
        
        for cluster_id in range(self.n_clusters):
            # Create mask where only the current cluster is active (1 = active, 0 = filtered)
            mask = np.zeros_like(self.full_mask.cpu().numpy())
            cluster_indices = np.where(self.labels == cluster_id)[0]
            mask[cluster_indices] = 1.0  # Only this cluster is active
            
            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(self.device)
            
            # Create filtered model
            filtered_model = FilteredDenseNet(
                self.base_model, 
                mask_tensor, 
                self.target_block
            ).to(self.device).eval()
            
            models[cluster_id] = filtered_model
            
        return models
    
    def get_cluster_attributions(
                self, 
                img: torch.Tensor, 
                target_class: int,
                method: str = "ig"
            ) -> Dict[int, np.ndarray]:
        attributions = {}
        
        for cluster_id, model in self.cluster_models.items():
            if method.lower() == "ig":
                attr_method = IntegratedGradients(model)
            elif method.lower() == "saliency":
                attr_method = Saliency(model)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get attribution for this cluster
            # Use top predicted class of the cluster-specific model
            with torch.no_grad():
                top_pred = model(img.unsqueeze(0)).argmax(1).item()
            attr = attr_method.attribute(img.unsqueeze(0), target=int(top_pred))
            attributions[cluster_id] = attr.squeeze().cpu().numpy()
            
        return attributions
    
    def get_combined_attribution(
                self,
                img: torch.Tensor,
                target_class: int,
                method: str = "ig",
                excluded_clusters: List[int] = None
            ) -> np.ndarray:
        if excluded_clusters is None:
            excluded_clusters = []
        
        # Create mask excluding specified clusters
        mask = np.ones_like(self.full_mask.cpu().numpy())
        for cluster_id in excluded_clusters:
            cluster_indices = np.where(self.labels == cluster_id)[0]
            mask[cluster_indices] = 0.0
        
        mask_tensor = torch.tensor(mask, dtype=torch.float32).to(self.device)
        
        # Create model with specified clusters filtered
        model = FilteredDenseNet(
            self.base_model,
            mask_tensor,
            self.target_block
        ).to(self.device).eval()
        
        # Get attribution
        if method.lower() == "ig":
            attr_method = IntegratedGradients(model)
        elif method.lower() == "saliency":
            attr_method = Saliency(model)
        
        with torch.no_grad():
            top_pred = model(img.unsqueeze(0)).argmax(1).item()
        attr = attr_method.attribute(img.unsqueeze(0), target=int(top_pred))
        return attr.squeeze().cpu().numpy()

# ============================================================================
# Analysis functions

def analyze_cluster_contributions(
            extractor: ClusterAttributionExtractor,
            img: torch.Tensor,
            label: int,
            method: str = "ig"
        ) -> Dict:
    
    # Get per-cluster attributions
    cluster_attrs = extractor.get_cluster_attributions(img, label, method)
    
    # Get baseline (all clusters active)
    baseline_attr = extractor.get_combined_attribution(img, label, method, excluded_clusters=[])
    
    # Get filtered (cluster 0 removed) - your best configuration
    filtered_attr = extractor.get_combined_attribution(img, label, method, excluded_clusters=[0])
    
    # Compute sum of all cluster attributions
    sum_attr = np.zeros_like(baseline_attr)
    for cluster_id, attr in cluster_attrs.items():
        sum_attr += attr
    
    # Compute cluster magnitudes
    cluster_magnitudes = {}
    for cluster_id, attr in cluster_attrs.items():
        cluster_magnitudes[cluster_id] = float(np.linalg.norm(attr))
    
    # Compute percentage contributions
    total_magnitude = sum(cluster_magnitudes.values())
    cluster_percentages = {k: v/total_magnitude*100 for k, v in cluster_magnitudes.items()}
    
    # Compute sparsity for each cluster (mass-based, like utils.py)
    cluster_sparsity = {}
    for cluster_id, attr in cluster_attrs.items():
        # Use mass-based sparsity: higher = more focused
        flat = np.abs(attr).flatten()
        total = flat.sum()
        if total == 0:
            cluster_sparsity[cluster_id] = 0.0
        else:
            idx_desc = np.argsort(flat)[::-1]  # largest to smallest
            cumulative_mass = np.cumsum(flat[idx_desc])
            k = np.searchsorted(cumulative_mass, 0.9 * total) + 1
            cluster_sparsity[cluster_id] = float(1.0 - (k / flat.size))
    
    # Compute cluster independence (cross-cluster correlations)
    cluster_independence = compute_cluster_independence(cluster_attrs)
    
    # Compute statistics
    stats = {
        'cluster_magnitudes': cluster_magnitudes,
        'cluster_percentages': cluster_percentages,
        'cluster_sparsity': cluster_sparsity,
        'reconstruction_error': float(np.mean(np.abs(baseline_attr - sum_attr))),
        'cluster_independence': cluster_independence
    }
    
    return {
        'attributions': cluster_attrs,
        'baseline': baseline_attr,
        'filtered': filtered_attr,
        'sum': sum_attr,
        'stats': stats
    }

def compute_cluster_independence(cluster_attrs: Dict[int, np.ndarray]) -> float:
    correlations = []
    cluster_list = list(cluster_attrs.values())
    
    for i in range(len(cluster_list)):
        for j in range(i+1, len(cluster_list)):
            corr = np.corrcoef(cluster_list[i].flatten(), cluster_list[j].flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    return float(np.mean(correlations)) if correlations else 0.0

def visualize_cluster_analysis(
            img: torch.Tensor,
            analysis: Dict,
            class_name: str,
            method: str,
            save_path: Path = None
        ):
    
    cluster_attrs = analysis['attributions']
    n_clusters = len(cluster_attrs)
    
    # Create figure with smaller size to reduce memory usage
    fig, axes = plt.subplots(2, 5, figsize=(16, 6)) 
    
    # Title
    fig.suptitle(f'Cluster Attribution Analysis - {method.upper()}\nClass: {class_name}', fontsize=12)  
    
    # Column headers
    column_titles = ['C0 Contribution', 'C1 Contribution', 'C2 Contribution', 'SUM (Baseline)', 'Filtered (C0 removed)']
    
    # Row 0: Attribution overlays
    for col in range(5):
        if col < n_clusters:
            # Individual cluster contributions
            cluster_id = col
            attr = cluster_attrs[cluster_id]
            overlay = attribution_overlay(img, attr)
            axes[0, col].imshow(overlay)
            axes[0, col].set_title(f'{column_titles[col]}', fontsize=10)  
        elif col == 3:
            # SUM (Baseline)
            sum_overlay = attribution_overlay(img, analysis['sum'])
            axes[0, col].imshow(sum_overlay)
            axes[0, col].set_title(f'{column_titles[col]}', fontsize=10)  
        elif col == 4:
            # Filtered (C0 removed)
            filtered_overlay = attribution_overlay(img, analysis['filtered'])
            axes[0, col].imshow(filtered_overlay)
            axes[0, col].set_title(f'{column_titles[col]}', fontsize=10)  
        
        axes[0, col].axis('off')
    
    # Row 1: Raw attribution maps
    for col in range(5):
        if col < n_clusters:
            # Individual cluster raw attributions
            cluster_id = col
            attr = cluster_attrs[cluster_id]
            attr_tensor = torch.from_numpy(attr) if isinstance(attr, np.ndarray) else attr
            heat = attr_tensor.abs().sum(0)
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            axes[1, col].imshow(heat.cpu().numpy(), cmap='jet')
            axes[1, col].set_title(f'{column_titles[col]} (Raw)', fontsize=10)  
        elif col == 3:
            # SUM (Baseline) raw
            baseline_tensor = torch.from_numpy(analysis['baseline']) if isinstance(analysis['baseline'], np.ndarray) else analysis['baseline']
            heat_baseline = baseline_tensor.abs().sum(0)
            heat_baseline = (heat_baseline - heat_baseline.min()) / (heat_baseline.max() - heat_baseline.min() + 1e-8)
            axes[1, col].imshow(heat_baseline.cpu().numpy(), cmap='jet')
            axes[1, col].set_title(f'{column_titles[col]} (Raw)', fontsize=10)  
        elif col == 4:
            # Filtered raw
            filtered_tensor = torch.from_numpy(analysis['filtered']) if isinstance(analysis['filtered'], np.ndarray) else analysis['filtered']
            heat_filtered = filtered_tensor.abs().sum(0)
            heat_filtered = (heat_filtered - heat_filtered.min()) / (heat_filtered.max() - heat_filtered.min() + 1e-8)
            axes[1, col].imshow(heat_filtered.cpu().numpy(), cmap='jet')
            axes[1, col].set_title(f'{column_titles[col]} (Raw)', fontsize=10)  
        
        axes[1, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')  
            print(f"Saved visualization to {save_path}")
        except Exception as e:
            print(f"Failed to save visualization: {e}")
            raise e
    
    return fig

# ============================================================================
# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_idx", type=int, default=1050)
    parser.add_argument("--num_images", type=int, default=20) # 1 for img_idx, >1 for stride sampling
    parser.add_argument("--stats_dir", type=str, default="outputs/neuron_stats/block_1_k3")
    parser.add_argument("--method", choices=["ig", "saliency"], default="ig")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    base_model = models.densenet121(pretrained=True).to(device).eval()
    
    # Create extractor
    extractor = ClusterAttributionExtractor(base_model, Path(args.stats_dir), device)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.squeeze().tolist(), std=STD.squeeze().tolist()),
    ])
    dataset = datasets.ImageFolder(root="./data/ILSVRC2012_img_val", transform=transform)
    
    # Debug: Print dataset information
    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Dataset classes: {len(dataset.classes)}")
    print(f"Dataset class_to_idx: {len(dataset.class_to_idx)}")
    print(f"First few class_to_idx entries: {list(dataset.class_to_idx.items())[:5]}")
    
    # Load ImageNet class names
    with open("data/imagenet_class_index.json", "r") as fp:
        idx2name = {int(k): v[1] for k, v in json.load(fp).items()}
    print(f"Loaded {len(idx2name)} ImageNet class names")
    
    # Create output directory
    save_dir = Path("outputs/cluster_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze multiple images
    aggregate_stats = {
        'cluster_magnitudes': {i: [] for i in range(extractor.n_clusters)},
        'cluster_percentages': {i: [] for i in range(extractor.n_clusters)},
        'cluster_sparsity': {i: [] for i in range(extractor.n_clusters)},
        'reconstruction_errors': [],
        'cluster_independence': []
    }
    
    # Handle single image analysis vs multiple images
    if args.num_images == 1:
        # Analyze the specific image index provided
        indices = [args.img_idx]
        print(f"\nAnalyzing specific image (index {args.img_idx})...")
    else:
        # Sample images evenly across the dataset
        indices = np.linspace(0, len(dataset)-1, args.num_images, dtype=int)
        print(f"\nAnalyzing {args.num_images} images evenly distributed across dataset...")
        
    
    for idx_num, img_idx in tqdm(enumerate(indices)):
        print(f"\nAnalyzing image {idx_num+1}/{len(indices)} (index {img_idx})...")
        
        img_t, label = dataset[img_idx]
        img_t = img_t.to(device)
        
        # Get class name
        class_name = idx2name[label]
        
        # Run analysis
        analysis = analyze_cluster_contributions(extractor, img_t, label, args.method)
        
        # Aggregate statistics
        for cluster_id in range(extractor.n_clusters):
            aggregate_stats['cluster_magnitudes'][cluster_id].append(
                analysis['stats']['cluster_magnitudes'][cluster_id]
            )
            aggregate_stats['cluster_percentages'][cluster_id].append(
                analysis['stats']['cluster_percentages'][cluster_id]
            )
            aggregate_stats['cluster_sparsity'][cluster_id].append(
                analysis['stats']['cluster_sparsity'][cluster_id]
            )
        aggregate_stats['reconstruction_errors'].append(analysis['stats']['reconstruction_error'])
        aggregate_stats['cluster_independence'].append(analysis['stats']['cluster_independence'])
        
        # Save visualization
        save_path = save_dir / f"cluster_analysis_{args.method}_{img_idx}.png"
        visualize_cluster_analysis(img_t, analysis, class_name, args.method, save_path)
        
        # Clear matplotlib memory
        plt.close('all')
        
        # Save numerical results
        results_path = save_dir / f"cluster_analysis_{args.method}_{img_idx}.json"
        stats_to_save = {}
        for key, value in analysis['stats'].items():
            if isinstance(value, dict):
                stats_to_save[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                        for k, v in value.items()}
            else:
                stats_to_save[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        with open(results_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    # Print aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS ACROSS ALL IMAGES")
    print("="*60)
    
    for cluster_id in range(extractor.n_clusters):
        print(f"\nCluster {cluster_id}:")
        print(f"  Avg Magnitude: {np.mean(aggregate_stats['cluster_magnitudes'][cluster_id]):.3f} "
              f"(± {np.std(aggregate_stats['cluster_magnitudes'][cluster_id]):.3f})")
        print(f"  Avg Percentage: {np.mean(aggregate_stats['cluster_percentages'][cluster_id]):.1f}% "
              f"(± {np.std(aggregate_stats['cluster_percentages'][cluster_id]):.1f}%)")
        print(f"  Avg Sparsity: {np.mean(aggregate_stats['cluster_sparsity'][cluster_id]):.3f} "
              f"(± {np.std(aggregate_stats['cluster_sparsity'][cluster_id]):.3f})")
    
    print(f"\nAvg Reconstruction Error: {np.mean(aggregate_stats['reconstruction_errors']):.5f} "
          f"(± {np.std(aggregate_stats['reconstruction_errors']):.5f})")
    print(f"Avg Cluster Independence: {np.mean(aggregate_stats['cluster_independence']):.3f} "
          f"(± {np.std(aggregate_stats['cluster_independence']):.3f})")
    
    # Save aggregate statistics
    agg_path = save_dir / f"aggregate_stats_{args.method}.json"
    with open(agg_path, 'w') as f:
        # Convert lists to stats
        agg_save = {}
        for key in aggregate_stats:
            if isinstance(aggregate_stats[key], dict):
                agg_save[key] = {}
                for cluster_id in aggregate_stats[key]:
                    agg_save[key][cluster_id] = {
                        'mean': float(np.mean(aggregate_stats[key][cluster_id])),
                        'std': float(np.std(aggregate_stats[key][cluster_id]))
                    }
            else:
                agg_save[key] = {
                    'mean': float(np.mean(aggregate_stats[key])),
                    'std': float(np.std(aggregate_stats[key]))
                }
        json.dump(agg_save, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")

if __name__ == "__main__":
    main()
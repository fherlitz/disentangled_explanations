import argparse
import json
import numpy as np
import torch
import torchvision.models as models
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ttest_rel
import warnings
import pickle
import gc
warnings.filterwarnings('ignore')
from utils import (
    set_seeds,
    get_imagenet_dataloader as get_dataloader,
    probability_drop_patch_wise,
    probability_drop_pixel_wise,
)
from filtered_eval import build_mask_and_block, FilteredDenseNet

def compute_faithfulness_batch(model, dataloader, attr_dir, method, indices, drop_fraction=0.1):
    device = next(model.parameters()).device
    faithfulness_scores = []
    
    for idx in indices:
        # Load attribution
        attr_path = attr_dir / f"{method}_{idx}.npy"
        if not attr_path.exists():
            continue
            
        attr = np.load(attr_path)
        
        # Get image from dataset
        img, true_label = dataloader.dataset[idx]
        img = img.to(device)
        
        # Compute faithfulness
        faith = probability_drop_patch_wise(model, img, attr, drop_fraction=drop_fraction, true_label=true_label)
        faithfulness_scores.append(faith)
        
        # Clear GPU memory periodically
        if len(faithfulness_scores) % 100 == 0:
            torch.cuda.empty_cache()
    
    return np.array(faithfulness_scores)

def process_in_batches(base_model, filt_model, dataloader, baseline_dir, filtered_dir, num_images, batch_size, drop_fraction, cache_dir):
    cache_file = cache_dir / 'faithfulness_scores.pkl'
    
    start_batch = 0
    results = {
        'ig': {'baseline': [], 'filtered': []},
        'saliency': {'baseline': [], 'filtered': []}
    }
    
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(start_batch, num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        indices = list(range(start_idx, end_idx))
        
        for method in ['ig', 'saliency']:
            # Compute baseline faithfulness
            baseline_scores = compute_faithfulness_batch(
                base_model, dataloader, baseline_dir, method, indices, drop_fraction
            )
            results[method]['baseline'].extend(baseline_scores) # Store baseline scores
            
            # Compute filtered faithfulness
            filtered_scores = compute_faithfulness_batch(
                filt_model, dataloader, filtered_dir, method, indices, drop_fraction
            )
            results[method]['filtered'].extend(filtered_scores) # Store filtered scores
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    for method in results:
        results[method]['baseline'] = np.array(results[method]['baseline'])
        results[method]['filtered'] = np.array(results[method]['filtered'])
    
    # Save final results
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved final results to {cache_file}")
    
    return results

def cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def bootstrap_ci(data, n_bootstrap=10000, ci=95, statistic=np.mean):
    bootstrap_stats = []
    n = len(data)
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        sample_indices = rng.choice(n, size=n, replace=True)
        sample = data[sample_indices]
        bootstrap_stats.append(statistic(sample))
    
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha)
    upper = np.percentile(bootstrap_stats, 100 - alpha)
    
    return lower, upper, bootstrap_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, default='outputs/patch_wise_evaluation/baseline/attributions')
    parser.add_argument('--filtered_dir', type=str, default='outputs/patch_wise_evaluation/filtered')
    parser.add_argument('--num_images', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='outputs/statistical_analysis_patch_wise')
    parser.add_argument('--drop_fraction', type=float, default=0.1)
    parser.add_argument('--stats_dir', type=str, default='outputs/neuron_stats/block_1_k3')
    args = parser.parse_args()
    
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output and cache directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    print("Loading models...")
    # Load models
    base_model = models.densenet121(pretrained=True).to(device).eval()
    
    # Build filtered model (Block 1, Cluster 0, k=3 default from stats_dir)
    stats_dir = Path(args.stats_dir)
    full_mask, target_block, _ = build_mask_and_block(base_model, stats_dir)
    labels = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)["labels"]
    
    mask_np = np.ones_like(full_mask.cpu().numpy())
    cluster_0_indices = np.where(labels == 0)[0]
    mask_np[cluster_0_indices] = 0.0
    mask_tensor = torch.tensor(mask_np, dtype=torch.float32)
    
    filt_model = FilteredDenseNet(base_model, mask_tensor.to(device), target_block).to(device).eval()
    
    # Get dataloader
    dl = get_dataloader(batch_size=1, num_workers=4, shuffle=False)
    
    # Process in batches from scratch
    baseline_dir = Path(args.baseline_dir)
    filtered_dir = Path(args.filtered_dir)
    
    results = process_in_batches(
        base_model, filt_model, dl,
        baseline_dir, filtered_dir,
        args.num_images, args.batch_size,
        args.drop_fraction, cache_dir
    )
    
    # Statistical analysis for each method
    statistical_results = {}
    
    for method in ['ig', 'saliency']:
        print(f"\n{'='*60}")
        print(f"Statistical Analysis for {method.upper()}")
        print('='*60)
        
        baseline = results[method]['baseline']
        filtered = results[method]['filtered']
        
        # Remove any NaN values
        valid_idx = ~(np.isnan(baseline) | np.isnan(filtered))
        baseline = baseline[valid_idx]
        filtered = filtered[valid_idx]
        
        differences = filtered - baseline
        
        method_results = {}
        
        # Basic statistics
        method_results['n_samples'] = len(baseline)
        method_results['baseline_mean'] = float(np.mean(baseline))
        method_results['baseline_std'] = float(np.std(baseline))
        method_results['filtered_mean'] = float(np.mean(filtered))
        method_results['filtered_std'] = float(np.std(filtered))
        method_results['mean_difference'] = float(np.mean(differences))
        method_results['std_difference'] = float(np.std(differences))
        
        print(f"\nBasic Statistics:")
        print(f"  N samples: {method_results['n_samples']}")
        print(f"  Baseline: {method_results['baseline_mean']:.4f} ± {method_results['baseline_std']:.4f}")
        print(f"  Filtered: {method_results['filtered_mean']:.4f} ± {method_results['filtered_std']:.4f}")
        print(f"  Mean Difference: {method_results['mean_difference']:.4f} ± {method_results['std_difference']:.4f}")
        
        # Paired t-test
        print(f"\nParametric Test:")
        t_stat, t_pvalue = ttest_rel(filtered, baseline)
        method_results['ttest_statistic'] = float(t_stat)
        method_results['ttest_pvalue'] = float(t_pvalue)
        print(f"  Paired t-test: t={t_stat:.4f}, p={t_pvalue:.4e}")
        
        # Effect sizes
        print(f"\nEffect Sizes:")
        cohen_d = cohens_d(filtered, baseline)
        method_results['cohens_d'] = float(cohen_d)
        print(f"  Cohen's d: {cohen_d:.4f} ({'small' if abs(cohen_d) < 0.5 else 'medium' if abs(cohen_d) < 0.8 else 'large'} effect)")
        
        # Bootstrap confidence intervals
        print(f"\nBootstrap Confidence Intervals (95%, 10000 iterations):")
        lower_ci, upper_ci, _ = bootstrap_ci(differences, n_bootstrap=10000)
        method_results['bootstrap_ci_lower'] = float(lower_ci)
        method_results['bootstrap_ci_upper'] = float(upper_ci)
        print(f"  Mean difference: [{lower_ci:.4f}, {upper_ci:.4f}]")
        
        # Percentage of improvements
        n_improved = np.sum(differences > 0)
        pct_improved = n_improved / len(differences) * 100
        method_results['percent_improved'] = float(pct_improved)
        print(f"\nImprovement Rate:")
        print(f"  Images with improvement: {n_improved}/{len(differences)} ({pct_improved:.1f}%)")
        
        # Median and percentiles
        print(f"\nPercentiles of Differences:")
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            val = np.percentile(differences, p)
            method_results[f'percentile_{p}'] = float(val)
            print(f"  {p}th percentile: {val:.4f}")
        
        statistical_results[method] = method_results
        
        # Clear memory after each method
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results to JSON
    json_path = output_dir / 'statistical_results.json'
    with open(json_path, 'w') as f:
        json.dump(statistical_results, f, indent=2)
    print(f"\nStatistical results saved to: {json_path}")
    
    # Create summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    for method in ['ig', 'saliency']:
        res = statistical_results[method]
        print(f"\n{method.upper()}:")
        print(f"  Improvement: {res['mean_difference']:.4f} ({res['mean_difference']/res['baseline_mean']*100:.1f}% relative)")
        print(f"  95% CI: [{res['bootstrap_ci_lower']:.4f}, {res['bootstrap_ci_upper']:.4f}]")
        print(f"  Paired t-test: t={res['ttest_statistic']:.4f}, p={res['ttest_pvalue']:.4e}")
        print(f"  Effect Size: d={res['cohens_d']:.3f} (Cohen's d)")
        print(f"  Success Rate: {res['percent_improved']:.1f}% of images improved")
    
    print("\n" + "="*60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
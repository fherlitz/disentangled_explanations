import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from utils import set_seeds

def build_random_mask_and_block(model: torch.nn.Module, block_name: str, n_random: int, seed: int = 42):
    block_map = {
        "block_1": model.features.denseblock1,
        "block_2": model.features.denseblock2,
        "block_3": model.features.denseblock3,
        "block_4": model.features.denseblock4,
    }
    
    target_block = block_map[block_name]
    
    # Get the number of channels for this block
    block_channel_counts = {
        "block_1": 256,
        "block_2": 512,
        "block_3": 1024,
        "block_4": 1024,
    }
    
    total_neurons = block_channel_counts[block_name]
    
    if n_random > total_neurons:
        raise ValueError(f"Cannot filter {n_random} neurons from block with {total_neurons} channels")
    
    # Create random mask
    np.random.seed(seed)
    mask = np.ones(total_neurons, dtype=np.float32)
    
    # Randomly select neurons to filter
    random_indices = np.random.choice(total_neurons, n_random, replace=False)
    mask[random_indices] = 0.0  # 0 = filter out, 1 = keep
    
    return torch.tensor(mask), target_block, block_name, random_indices

def load_neuron_stats(block_name: str, k: int = 3):
    stats_dir = Path(f"outputs/neuron_stats/{block_name}_k{k}")
    if not stats_dir.exists():
        raise FileNotFoundError(f"{stats_dir} not found, run neuron_stats.py first.")
    
    stats = np.load(stats_dir / f"stats_{block_name}.npz")
    labels = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)["labels"]
    
    return stats, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_random", type=int, default=214)
    parser.add_argument("--block", choices=["block_1", "block_2", "block_3", "block_4"], default="block_1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set up model and get random indices
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.densenet121(pretrained=True).to(device).eval()
    
    mask_tensor, target_block, block_name, random_indices = build_random_mask_and_block(
        base_model, args.block, args.n_random
    )
    
    # Load neuron statistics
    stats, labels = load_neuron_stats(args.block, 3)
    
    # Create DataFrame with neuron statistics
    df = pd.DataFrame({
        "neuron_id": range(len(stats["mean_activation"])),
        "mean": stats["mean_activation"],
        "rate": stats["activation_rate"],
        "variance": stats["variance"],
        "cluster": labels,
    })
    
    # Mark which neurons are randomly selected for filtering
    df["is_filtered"] = df["neuron_id"].isin(random_indices)
    df["neuron_type"] = df["is_filtered"].map({True: "Filtered", False: "Kept"})
    
    # Create output directory
    output_dir = Path(f"outputs/control_random/{args.block}_n{args.n_random}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # 3-D scatter of neurons coloured by filtering status (static)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot filtered neurons (red)
    filtered_df = df[df["is_filtered"]]
    ax.scatter(filtered_df["mean"], filtered_df["rate"], filtered_df["variance"], 
               c='red', s=20, alpha=0.4, label=f'Filtered ({len(filtered_df)})')
    
    # Plot kept neurons (blue)
    kept_df = df[~df["is_filtered"]]
    ax.scatter(kept_df["mean"], kept_df["rate"], kept_df["variance"], 
               c='blue', s=8, alpha=0.7, label=f'Kept ({len(kept_df)})')
    
    ax.set_xlabel("Mean activation")
    ax.set_ylabel("Activation rate")
    ax.set_zlabel("Variance")
    ax.legend()
    plt.tight_layout()
    
    out3d = output_dir / "control_scatter_3d.png"
    plt.savefig(out3d, dpi=300)
    print("Saved", out3d)
    
    # ------------------------------------------------------------------
    # Interactive 3-D scatter (HTML)
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Create interactive 3D scatter
        fig_3d = px.scatter_3d(
            df, x='mean', y='rate', z='variance', 
            color='neuron_type',
            labels={'mean': 'Mean activation', 'rate': 'Activation rate', 'variance': 'Variance'},
            opacity=0.7,
            size_max=15,
            color_discrete_map={'Filtered': 'red', 'Kept': 'blue'},
            category_orders={'neuron_type': ['Filtered', 'Kept']}
        )
        
        # Add hover information
        fig_3d.update_traces(
            hovertemplate="<b>Neuron %{customdata}</b><br>" +
                         "Mean: %{x:.3f}<br>" +
                         "Rate: %{y:.3f}<br>" +
                         "Variance: %{z:.3f}<br>" +
                         "Type: %{marker.color}<extra></extra>",
            customdata=df["neuron_id"]
        )
        
        # Position legend
        fig_3d.update_layout(
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        fig_3d.write_html(output_dir / "control_scatter_3d_interactive.html")
        print("Saved", output_dir / "control_scatter_3d_interactive.html")
        
    except ImportError:
        print("plotly not installed, skipping interactive 3D plot")
    
    # ------------------------------------------------------------------
    # 2-D t-SNE visualization
    
    try:
        from sklearn.manifold import TSNE
        
        X = df[["mean", "rate", "variance"]].values
        ts = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        # Plot filtered neurons
        filtered_mask = df["is_filtered"]
        plt.scatter(ts[filtered_mask, 0], ts[filtered_mask, 1], 
                   c='red', s=20, alpha=0.7, label=f'Filtered ({filtered_mask.sum()})')
        
        # Plot kept neurons
        kept_mask = ~df["is_filtered"]
        plt.scatter(ts[kept_mask, 0], ts[kept_mask, 1], 
                   c='blue', s=8, alpha=0.5, label=f'Kept ({kept_mask.sum()})')
        
        plt.xlabel("t-SNE-1")
        plt.ylabel("t-SNE-2")
        plt.title(f"{args.block}: t-SNE of neuron stats with random filtering")
        plt.legend()
        plt.tight_layout()
        
        out_tsne = output_dir / "control_tsne.png"
        plt.savefig(out_tsne, dpi=300)
        print("Saved", out_tsne)
        
    except ImportError:
        print("scikit-learn not installed, skipping t-SNE plot")
    
    # ------------------------------------------------------------------
    # Summary statistics
    
    summary_stats = {
        "block": args.block,
        "total_neurons": len(df),
        "filtered_neurons": len(filtered_df),
        "kept_neurons": len(kept_df),
        "filtered_percentage": len(filtered_df) / len(df) * 100,
        "n_random": args.n_random,
    }
    
    # Add cluster distribution
    cluster_dist = df[df["is_filtered"]]["cluster"].value_counts().to_dict()
    summary_stats["filtered_by_cluster"] = cluster_dist
    
    # Save summary
    summary_file = output_dir / "control_summary.json"
    import json
    summary_file.write_text(json.dumps(summary_stats, indent=2))
    print("Saved", summary_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONTROL EXPERIMENT VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Block: {args.block}")
    print(f"Total neurons: {len(df)}")
    print(f"Filtered neurons: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.1f}%)")
    print(f"Kept neurons: {len(kept_df)} ({len(kept_df)/len(df)*100:.1f}%)")
    print(f"n_random: {args.n_random}")
    print()
    print("Filtered neurons by cluster:")
    for cluster, count in sorted(cluster_dist.items()):
        print(f"  Cluster {cluster}: {count} neurons")
    print()
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main() 
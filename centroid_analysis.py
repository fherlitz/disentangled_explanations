import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="block_1", choices=["block_1", "block_2", "block_3", "block_4"])
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    stats_dir = Path(f"outputs/neuron_stats/{args.block}_k{args.k}")
    if not stats_dir.exists():
        raise FileNotFoundError(f"{stats_dir} not found, run neuron_stats.py first.")

    stats = np.load(stats_dir / f"stats_{args.block}.npz")
    labels = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)["labels"]

    # ------------------------------------------------------------------
    # Centroid analysis (csv)

    df = pd.DataFrame({
        "mean":      stats["mean_activation"],
        "sparsity":  stats["sparsity"],
        "variance":  stats["variance"],
        "cluster":   labels,
    })

    centroids = df.groupby("cluster").agg(
        mean=("mean", "mean"),
        sparsity=("sparsity", "mean"),
        variance=("variance", "mean"),
        n_neurons=("cluster", "size"),
    )
    centroids.to_csv(stats_dir / "cluster_centroids.csv")
    print("Saved", stats_dir / "cluster_centroids.csv")

    # ------------------------------------------------------------------
    # 3-D scatter of neurons 

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for c, grp in df.groupby("cluster"):
        ax.scatter(grp["mean"], grp["sparsity"], grp["variance"], s=8, label=f"C{c}")
    ax.set_xlabel("Mean activation")
    ax.set_ylabel("Sparsity")
    ax.set_zlabel("Variance")
    ax.legend(title="Cluster")
    plt.tight_layout()
    out3d = stats_dir / "cluster_scatter_3d.png"
    plt.savefig(out3d, dpi=300)
    print("Saved", out3d)

    # ------------------------------------------------------------------
    # Interactive 3-D scatter of neurons coloured by cluster (HTML)
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import colorsys
        
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        df['cluster'] = pd.Categorical(df['cluster'], categories=sorted(df['cluster'].unique()), ordered=True)
        
        # 3D scatter 
        fig_3d = px.scatter_3d(
            df, x='mean', y='sparsity', z='variance', 
            color='cluster', 
            title=f"{args.block}: neurons in 3-D stat space (k={args.k})",
            labels={'mean': 'Mean activation', 'sparsity': 'Sparsity', 'variance': 'Variance'},
            opacity=0.5, 
            size_max=10, 
            color_discrete_sequence=cluster_colors[:args.k],  
            category_orders={'cluster': sorted(df['cluster'].unique())} 
        )
        
        for i, trace in enumerate(fig_3d.data):
            trace.legendgroup = 'clusters'
        
        for c in range(args.k):
            centroid = centroids.iloc[c]
            base_color = cluster_colors[c]
            color_hex = base_color

            r = int(color_hex[1:3], 16) / 255.0
            g = int(color_hex[3:5], 16) / 255.0
            b = int(color_hex[5:7], 16) / 255.0
            
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            s = min(1.0, s * 2.0)  
            v = min(1.0, v * 1.3) 
            r_sat, g_sat, b_sat = colorsys.hsv_to_rgb(h, s, v)
            
            saturated_color = f'#{int(r_sat*255):02x}{int(g_sat*255):02x}{int(b_sat*255):02x}'
            
            fig_3d.add_trace(go.Scatter3d(
                x=[centroid['mean']], y=[centroid['sparsity']], z=[centroid['variance']],
                mode='markers',
                marker=dict(size=8, symbol='diamond', color=saturated_color),
                name=f'Centroid C{c}',
                showlegend=False,  
                legendgroup='centroids',
                legendgrouptitle=dict(text='Centroids')
            ))
        
        fig_3d.update_layout(
            legend=dict(
                x=0.02,  
                y=0.98,  
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1,
                groupclick="toggleitem"
            )
        )
        
        fig_3d.write_html(stats_dir / "cluster_scatter_3d_interactive.html")
        print("Saved", stats_dir / "cluster_scatter_3d_interactive.html")
        
    except ImportError:
        print("plotly not installed, skipping interactive 3D plot")

    # ------------------------------------------------------------------
    # 2-D t-SNE

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed, skipping t-SNE plot")
        return

    X = df[["mean", "sparsity", "variance"]].values
    ts = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X)
    plt.figure(figsize=(6, 5))
    for c in range(args.k):
        plt.scatter(ts[labels == c, 0], ts[labels == c, 1], s=8, label=f"C{c}")
    plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
    plt.title(f"{args.block}: t-SNE of neuron stats (k={args.k})")
    plt.legend()
    plt.tight_layout()
    out_tsne = stats_dir / "cluster_tsne.png"
    plt.savefig(out_tsne, dpi=300)
    print("Saved", out_tsne)


if __name__ == "__main__":
    main()

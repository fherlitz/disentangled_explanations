import subprocess
import sys
from pathlib import Path
from itertools import product
import json
import numpy as np

# ============================================================================
# CONFIGURATION 

BLOCKS = ["block_1", "block_2", "block_3", "block_4"]  # DenseNet blocks to analyse 
K_VALUES = [2, 3, 4, 5]                                # K numbers of clusters
IMAGES = 200                                           # Images per run (kept equal to baseline)
METRIC = "patch"                                       # Metric granularity for evaluation (patch, pixel)


# Helper wrapper to run commands
def run(cmd: list[str]):
    print("\n>", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("Command failed, aborting …", file=sys.stderr)
        sys.exit(res.returncode)


# ============================================================================
# 1. Ensure baseline exists

def ensure_baseline():
    base_metrics = Path("outputs/baseline/metrics.json")
    if base_metrics.exists():
        print("[baseline] metrics.json already present, skipping baseline generation.")
        return
    run([sys.executable, 
         "baseline_eval.py", 
         "--images", str(IMAGES),
         "--metric", METRIC, # generates attribution maps and metrics
         ])


# ============================================================================
# 2. Full grid sweep

def run_grid():
    for block, k in product(BLOCKS, K_VALUES):
        # Compute stats & clustering for this (block, k)
        stats_dir = Path(f"outputs/neuron_stats/{block}_k{k}")
        if stats_dir.exists():
            print(f"[stats] {stats_dir} already exists, skipping.")
        else:
            run([
                sys.executable,
                "neuron_stats.py",
                "--block", block,
                "--n_clusters", str(k),
                "--output_dir", str(stats_dir)
            ])

        # Run centroid analysis for this (block, k) combination
        print(f"[centroid] Running centroid analysis for {block}_k{k}")
        run([
            sys.executable,
            "centroid_analysis.py",
            "--block", block,
            "--k", str(k)
        ])

        # Load clustering info to know the number of clusters (k=5)
        cluster_results = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)
        n_clusters = int(cluster_results["n_clusters"])

        # Masking experiments: iterate over each cluster index directly
        for cid in range(n_clusters):
            cfg_name = f"{block}_cluster_{cid}_k{k}"
            out_dir = Path(f"outputs/filtered_{cfg_name}")

            # Skip run if metrics already exist to avoid recomputation
            if (out_dir / "metrics.json").exists():
                print(f"[skip] metrics for {cfg_name} already exist.")
                continue

            # Construct command including mandatory --output_dir flag
            cmd = [
                sys.executable, "filtered_eval.py",
                "--cluster_idx", str(cid),
                "--images", str(IMAGES),
                "--stats_dir", str(stats_dir),
                "--output_dir", str(out_dir),
                "--metric", METRIC,
            ]
            run(cmd)


# ============================================================================
# 3. Consolidate into master CSV

def aggregate():
    run([sys.executable, "collect_metrics.py"])


# ============================================================================
# MAIN

if __name__ == "__main__":
    ensure_baseline()
    run_grid()
    aggregate()
    print("\nAll experiments finished and results aggregated -> outputs/mask_grid_summary.csv")
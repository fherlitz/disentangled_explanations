## Disentangled explanations of neural network predictions based on activations statistics

This repository implements an end-to-end pipeline to analyze, filter, and visualize neuron clusters, using DenseNet-121 on ImageNet. This analysis is performed on explanation methods like Integrated Gradients, and Saliency. You can:

- Compute baseline accuracy and attribution metrics (sparsity, faithfulness)
- Collect neuron activation statistics and cluster neurons per DenseNet block
- Block gradient flow from an entire cluster during inference via a differentiable forward-hook
- Measure how filtering affects sparsity, faithfulness, and accuracy
- Visualize neuron clusters (t-SNE, 3D), attribution overlays, and masked images
- Run a random-neuron control experiment 
- Analyse per-cluster decomposition

### Contents
- `utils.py`: seeds, dataloader, metrics (sparsity, faithfulness)
- `baseline_eval.py`: baseline accuracy and metrics for IG & Saliency
- `neuron_stats.py`: activation capture (1k images), per-neuron stats, K-means
- `centroid_analysis.py`: t-SNE, 3D plots, centroid stats
- `filtered_eval.py`: mask a cluster (with detached gradients), recompute metrics
- `class_analysis.py` → `merge_classes.py`: per-class metrics and merge
- `collect_metrics.py`: aggregate all runs + baseline into a CSV
- `run_experiments.py`: run the full grid (blocks × k) and aggregate
- `accuracy.py`: top-1/top-5 accuracy for filtered vs baseline
- `statistic.py`: statistical significance of faithfulness improvements
- `control_experiment.py` + `control_visualization.py`: random-neuron control
- `decomposition_analysis.py`: per-cluster attribution decomposition and overlays
- `visualize_attribution.py` + `run_images_attribution.py`: export figure panels

---

## 1) Setup
### Dependencies
This project needs the following dependencies:

- torch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- plotly
- tqdm
- captum
- scipy


### ImageNet validation data
See `data/README.md` for detailed instructions. In short, place the validation set at `data/ILSVRC2012_img_val/` and run `data/subfolders.py` to organize images into class subfolders.

---

## 2) Quick start: run the full pipeline

This executes baseline + all blocks (`block_1..4`) × clusters (`k=2..5`), runs centroid analyses, filtered evaluations for every cluster, and aggregates results.

```bash
python run_experiments.py
```

Outputs are written under `outputs/` (see `outputs/README.md`).

---

## 3) Step-by-step tutorial

Below are the individual steps if you prefer to run components manually.

### 3.1 Baseline accuracy, attributions, and metrics
```bash
python baseline_eval.py --images 200 --metric patch    # or --metric pixel
```
- Saves: `outputs/baseline/accuracy.json`, `outputs/baseline/attributions/`, `outputs/baseline/metrics.json`

### 3.2 Neuron stats and clustering (per block)
```bash
python neuron_stats.py --block block_1 --n_clusters 3
```
- Saves: `stats_block_1.npz`, `cluster_results.npz`, `manifest.json`

### 3.3 Centroid analysis and visualizations
```bash
python centroid_analysis.py --block block_1 --k 3
```
- Saves: `cluster_centroids.csv`, `cluster_scatter_3d.png`, `cluster_scatter_3d_interactive.html`, `cluster_tsne.png`

### 3.4 Filter a specific cluster and recompute metrics
```bash
python filtered_eval.py \
  --cluster_idx 0 \
  --images 200 \
  --stats_dir outputs/neuron_stats/block_1_k3 \
  --output_dir outputs/filtered_block_1_cluster_0_k3 \
  --metric patch    # or pixel
```
- Saves: `outputs/filtered_block_1_cluster_0_k3/attributions/` and `metrics.json`

### 3.5 Per-class analysis (optional)
Compute class-wise metrics for a range, then merge.
```bash
python class_analysis.py \
  --cluster_idx 0 \
  --class_start 0 --class_end 200 \
  --images_per_class 50 \
  --stats_dir outputs/neuron_stats/block_1_k3

# Repeat for ranges [200,400), [400,600), [600,800), [800,1000)
# In our case a range of 200 classes takes ~13 hours to finish

# Merge the produced CSVs
python merge_classes.py \
  --input_files \
    outputs/class_analysis_metrics_0_200.csv \
    outputs/class_analysis_metrics_200_400.csv \
    outputs/class_analysis_metrics_400_600.csv \
    outputs/class_analysis_metrics_600_800.csv \
    outputs/class_analysis_metrics_800_1000.csv \
  --output outputs/patch_wise_evaluation/class_analysis_metrics_complete.csv
```

### 3.6 Aggregate all filtered runs + baseline into one CSV
```bash
python collect_metrics.py
```
- Saves: `outputs/mask_grid_summary.csv`

### 3.7 Accuracy comparison (top-1/top-5)
```bash
python accuracy.py --block block_1 --cluster 0 --k 3 --batches 1000
```
- Saves: `outputs/accuracy/acc_block_1_c0_k3.json`

### 3.8 Statistical significance of faithfulness improvements (optional)
```bash
python statistic.py \
  --baseline_dir outputs/baseline/attributions \
  --filtered_dir outputs/filtered_block_1_cluster_0_k3/attributions \
  --num_images 50000 \
  --batch_size 200 \
  --stats_dir outputs/neuron_stats/block_1_k3 \
  --output_dir outputs/statistical_analysis_patch_wise
```
- Saves: `statistical_results.json`

### 3.9 Control experiment (random neuron filtering)
```bash
python control_experiment.py --block block_1 --n_random 214 --metric patch --images 200

python control_visualization.py --block block_1 --n_random 214
```
- Saves: `outputs/control_random/.../attributions/`, `metrics.json`, and visualizations

### 3.10 Per-cluster decomposition analysis and overlays
```bash
python decomposition_analysis.py \
  --img_idx 1050 \
  --num_images 1 \   # --num_images 2-50000 for images distributed evenly and aggregate_stats_*.json (ignores --img_idx)
  --stats_dir outputs/neuron_stats/block_1_k3 \
  --method ig    # or saliency
```
- Saves: `outputs/decomposition_analysis/cluster_analysis_*.json|.png` and `aggregate_stats_*.json`

### 3.11 Attribution visualization panels
Single image:
```bash
python visualize_attribution.py --img_idx 1500 --cluster_idx 0 --block block_1 --k 3 --drop_fraction 0.1 --metric patch
```
Batch of 20 evenly spaced images:
```bash
python run_images_attribution.py
```
- Saves to: `outputs/visualizations/`

---

## 4) Metrics definition
- **Faithfulness (drop 10%)**: confidence drop when masking the top 10% most important pixels/patches by attribution mass
- **Sparsity (mass 90%)**: fraction of pixels/patches not needed to accumulate 90% of attribution mass

Pixel-wise metrics operate at spatial resolution; patch-wise metrics use a 14×14 non-overlapping grid.

---

## 5) Tips and troubleshooting
- Ensure `data/ILSVRC2012_img_val` is organized into class subfolders (run `data/subfolders.py` once)
- Expect large outputs (NPY attributions); keep `outputs/` on a drive with sufficient space
- Reproducibility: seeding is centralized in `utils.set_seeds`


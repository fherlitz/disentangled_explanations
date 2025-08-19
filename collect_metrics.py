import json
import re
from pathlib import Path
import csv

# ============================================================================
# CSV generation of all experiments

out_root = Path("outputs")

# capture groups: block, clusterId, k
pattern = re.compile(r"filtered_(?P<block>block_[^_]+)_cluster_(?P<clusterId>\d+)_k(?P<k>\d+)")

rows = []
for d in out_root.glob("filtered_*"):
    m = pattern.match(d.name)
    if not m:
        continue
    metrics_file = d / "metrics.json"
    if not metrics_file.exists():
        continue
    with open(metrics_file) as fp:
        metrics = json.load(fp)
    row_base = {
        "block": m.group("block"),
        "clusterId": int(m.group("clusterId")),
        "k_clusters": int(m.group("k")),
    }

    # flatten metrics
    for method in (m for m in metrics if isinstance(metrics[m], dict)):
        for key, val in metrics[method].items():
            row_base[f"{method}_{key}"] = val
    # copy any top-level scalar (e.g. accuracy_top1)
    for key, val in metrics.items():
        if not isinstance(val, dict):
            row_base[key] = val
    rows.append(row_base)

# add baseline metrics
baseline_file = out_root / "baseline" / "metrics.json"
if baseline_file.exists():
    # load baseline metrics
    with open(baseline_file) as fp:
        base_metrics = json.load(fp)
    # load baseline accuracy
    acc_path = out_root / "baseline" / "accuracy.json"
    acc_val = None
    if acc_path.exists():
        with open(acc_path) as fp:
            acc_val = json.load(fp).get("top1")

    row_base = {
        "block": "baseline",
        "clusterId": 0,
        "k_clusters": 0,
        "accuracy_top1": acc_val if acc_val is not None else "",
    }
    for method in base_metrics:
        for key, val in base_metrics[method].items():
            row_base[f"{method}_{key}"] = val
    rows.append(row_base)

# write CSV
csv_path = out_root / "mask_grid_summary.csv"
with open(csv_path, "w", newline="") as fp:
    headers = [
        "block",
        "k_clusters",
        "clusterId",
        "accuracy_top1",
        "ig_faithfulness_mean",
        "ig_faithfulness_std",
        "ig_sparsity_mean",
        "ig_sparsity_std",
        "saliency_faithfulness_mean",
        "saliency_faithfulness_std",
        "saliency_sparsity_mean",
        "saliency_sparsity_std",
    ]
    writer = csv.DictWriter(fp, fieldnames=headers)
    writer.writeheader()
    # Keep only keys in headers to avoid errors from extra metrics (mean_diff, ci95, etc.)
    filtered_rows = [{k: row.get(k, "") for k in headers} for row in rows]
    writer.writerows(filtered_rows)

print(f"Summary written to {csv_path} with {len(rows)} rows.") 
import argparse, numpy as np, torch, torchvision.models as models
import json
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from baseline_eval import compute_accuracy, get_dataloader
from filtered_eval import build_mask_and_block, FilteredDenseNet

# -----------------------------------------------------------------------------
# top-5 accuracy

def compute_accuracy_topk(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        k: int = 5, # top-5
        max_batches: int | None = None,
    ) -> float:
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"Top-{k} accuracy")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # (B, num_classes)
            # top-k indices along class dimension
            _, topk = outputs.topk(k, dim=1, largest=True, sorted=True)
            # check if ground truth label is within top-k for each sample
            correct += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            if max_batches is not None and i + 1 >= max_batches:
                break

    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="block_1")
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--batches", type=int, default=1000) # 50000 / 50 = 1000 batches
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.densenet121(weights="IMAGENET1K_V1").to(device).eval()

    stats_dir = Path(f"outputs/neuron_stats/{args.block}_k{args.k}")
    full_mask, target_block, _ = build_mask_and_block(base_model, stats_dir)
    labels = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)["labels"]

    mask_np = np.ones_like(full_mask.cpu().numpy())
    sel_idx = np.where(labels == args.cluster)[0]
    mask_np[sel_idx] = 0.0
    mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

    filt_model = FilteredDenseNet(base_model, mask_tensor.to(device), target_block).to(device).eval()

    dl = get_dataloader(batch_size=50, num_workers=5)

    print("Computing accuracies …")
    # Filtered model accuracies
    acc_filt_top1 = compute_accuracy(filt_model, dl, device, max_batches=args.batches)
    acc_filt_top5 = compute_accuracy_topk(filt_model, dl, device, k=5, max_batches=args.batches)
    print(f"Filtered accuracy  -  Top-1: {acc_filt_top1*100:.2f}% | Top-5: {acc_filt_top5*100:.2f}%")

    # Baseline model accuracies
    acc_base_top1 = compute_accuracy(base_model, dl, device, max_batches=args.batches)
    acc_base_top5 = compute_accuracy_topk(base_model, dl, device, k=5, max_batches=args.batches)
    print(
        f"Baseline accuracy - Top-1: {acc_base_top1*100:.2f}% | Top-5: {acc_base_top5*100:.2f}%  "
        f"(Δ Top-1 = {acc_filt_top1-acc_base_top1:+.4f}, Δ Top-5 = {acc_filt_top5-acc_base_top5:+.4f})"
    )

    # ----------------------------------------------------------------------
    # Save results to JSON
    
    results = {
        "block": args.block,
        "cluster": args.cluster,
                "top1_baseline": acc_base_top1,
        "top5_baseline": acc_base_top5,
        "top1_filtered": acc_filt_top1,
        "top5_filtered": acc_filt_top5,
    }

    out_dir = Path("outputs/accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (
        f"acc_{args.block}_c{args.cluster}_k{args.k}.json"
    )
    with out_file.open("w") as fp:
        json.dump(results, fp, indent=2)
    print("Accuracy JSON saved to", out_file)

    
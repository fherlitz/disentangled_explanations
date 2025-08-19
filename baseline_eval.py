import os
import argparse
import json
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from captum.attr import IntegratedGradients, Saliency
from tqdm import tqdm 
from pathlib import Path
from utils import (
    set_seeds,
    get_imagenet_dataloader as get_dataloader,
    compute_sparsity_pixel_wise,
    compute_sparsity_patch_wise,
    probability_drop_pixel_wise,
    probability_drop_patch_wise
)

# ============================================================================
# 1. Accuracy computation

# computes the accuracy of the model on the validation subset,
# top-1 accuracy is the percentage of images that are correctly classified
# so if the "best" output is correct, we count it as correct
def compute_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device, 
    max_batches: Optional[int] = None,
) -> float:

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Accuracy")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item() # item() converts the tensor to a scalar
            total += labels.size(0)

            if max_batches is not None and i + 1 >= max_batches:
                break

    return correct / total

# ============================================================================
# 2. Attribution collection

# Attribution for 200 images for each XAI method
def collect_attributions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    method: str = "ig",
    num_images: int = 200,
    save_dir: str = "outputs/baseline/attributions",
):

    os.makedirs(save_dir, exist_ok=True)

    if method.lower() == "ig":
        attr_method = IntegratedGradients(model)
    elif method.lower() == "saliency":
        attr_method = Saliency(model)

    total = len(dataset)
    stride = max(1, total // num_images)
    indices = [i * stride for i in range(num_images)]

    model.eval()
    for i, idx in enumerate(tqdm(indices, desc=f"Attributions-{method}")):
        img, label = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        attrs = attr_method.attribute(img_batch, target=label)
        np.save(os.path.join(save_dir, f"{method}_{i}.npy"), attrs.squeeze().cpu().numpy())

# ============================================================================
# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["patch", "pixel"], default="patch", help="Evaluation metric granularity")
    parser.add_argument("--images", type=int, default=200)
    args = parser.parse_args()

    set_seeds(42) # SEED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True).to(device)

    dataloader = get_dataloader(batch_size=32, num_workers=4)
    dataset = dataloader.dataset

    # Baseline accuracy
    acc = compute_accuracy(model, dataloader, device)
    print(f"Baseline Top-1 accuracy: {acc * 100:.2f}%")
    os.makedirs("outputs/baseline", exist_ok=True)
    with open("outputs/baseline/accuracy.json", "w") as f:
        json.dump({"top1": acc}, f, indent=2)

    # ------------------------------------------------------------------
    # Baseline attributions
    
    NUM_IMAGES = args.images  # used for both attribution and metric steps
    collect_attributions(model, dataset, device, method="ig", num_images=NUM_IMAGES)
    collect_attributions(model, dataset, device, method="saliency", num_images=NUM_IMAGES)

    # ------------------------------------------------------------------
    # Metrics (sparsity & faithfulness)

    ATTR_DIR = Path("outputs/baseline/attributions")
    SAVE_PATH = Path("outputs/baseline/metrics.json")

    sparsity_scores = {"ig": [], "saliency": []}
    faithfulness_scores = {"ig": [], "saliency": []}

    stride = max(1, len(dataset) // NUM_IMAGES)
    indices = [i * stride for i in range(NUM_IMAGES)]
    images_flat = [dataset[i][0].to(device) for i in indices]

    for idx, img in enumerate(tqdm(images_flat, desc="Metrics")):
        for method in ["ig", "saliency"]:
            attr_path = ATTR_DIR / f"{method}_{idx}.npy"
            attr = np.load(attr_path)

            if args.metric == "pixel":
                sparsity_val = compute_sparsity_pixel_wise(attr, mass_fraction=0.9)
                faith_val = probability_drop_pixel_wise(model, img, attr, drop_fraction=0.1)
            else:
                sparsity_val = compute_sparsity_patch_wise(attr, mass_fraction=0.9)
                faith_val = probability_drop_patch_wise(model, img, attr, drop_fraction=0.1)
            sparsity_scores[method].append(sparsity_val)
            faithfulness_scores[method].append(faith_val)

    results = {
        m: {
            "sparsity_mean": float(np.mean(sparsity_scores[m])),
            "sparsity_std": float(np.std(sparsity_scores[m])),
            "faithfulness_mean": float(np.mean(faithfulness_scores[m])),
            "faithfulness_std": float(np.std(faithfulness_scores[m])),
        }
        for m in ["ig", "saliency"]
    }

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as fp:
        json.dump(results, fp, indent=2)

    print("Baseline metrics saved to", SAVE_PATH)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 
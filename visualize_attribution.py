import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from captum.attr import IntegratedGradients, Saliency
from pathlib import Path
import random 
from filtered_eval import build_mask_and_block, FilteredDenseNet 
from utils import (
    compute_sparsity_pixel_wise,
    compute_sparsity_patch_wise
)

# ============================================================================
# Helpers

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def attribution_overlay(img_t: torch.Tensor, attr: torch.Tensor) -> np.ndarray:
    # sum over channels, normalize
    heat = attr.abs().sum(0)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat = heat.cpu().numpy()
    img = (img_t * STD.to(img_t.device) + MEAN.to(img_t.device)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[..., :3]
    overlay = 0.2 * img + 0.8 * heat_rgb
    return overlay

def make_masked_image(
            img_t: torch.Tensor,
            attr: torch.Tensor,
            drop_fraction: float = 0.1,
            grid_size: int = 14,
        ) -> torch.Tensor:
    attr_abs = attr.abs()
    if attr_abs.ndim == 3:
        attr_abs = attr_abs.sum(0)  # (H, W)

    H, W = attr_abs.shape
    patch_h, patch_w = H // grid_size, W // grid_size

    # aggregate attribution mass per patch
    patch_scores = torch.zeros((grid_size, grid_size), device=attr_abs.device)
    for i in range(grid_size):
        for j in range(grid_size):
            h0, h1 = i * patch_h, (i + 1) * patch_h
            w0, w1 = j * patch_w, (j + 1) * patch_w
            patch_scores[i, j] = attr_abs[h0:h1, w0:w1].sum()

    # select fixed top-k patches (drop_fraction of total patches)
    num_patches = grid_size * grid_size
    k = max(1, int(round(drop_fraction * num_patches)))
    flat = patch_scores.flatten()
    order = torch.argsort(flat, descending=True)
    selected = order[:k]

    mask2d = torch.ones((H, W), device=img_t.device)
    for idx in selected:
        r = idx // grid_size
        c = idx % grid_size
        h0, h1 = r * patch_h, (r + 1) * patch_h
        w0, w1 = c * patch_w, (c + 1) * patch_w
        mask2d[h0:h1, w0:w1] = 0.0 # h0:h1, w0:w1 is the patch

    mask = mask2d.unsqueeze(0)  # shape (1, H, W)
    return img_t * mask

# ---------------------------------------------------------------------------
# Pixel-wise masking helper 

def make_masked_image_pixel(
            img_t: torch.Tensor,
            attr: torch.Tensor,
            drop_fraction: float = 0.1,
        ) -> torch.Tensor:

    attr_abs = attr.abs()
    if attr_abs.ndim == 3:
        attr_abs = attr_abs.sum(0)  # (H, W)

    H, W = attr_abs.shape
    k = max(1, int(round(drop_fraction * H * W)))
    flat = attr_abs.flatten()
    idx_sorted = torch.argsort(flat, descending=True)[:k]

    mask2d = torch.ones((H, W), device=img_t.device)
    mask2d.view(-1)[idx_sorted] = 0.0
    mask = mask2d.unsqueeze(0).expand_as(img_t)
    return img_t * mask

# ============================================================================
# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_idx", type=int, default=1500) # -1 for random image visualization
    parser.add_argument("--cluster_idx", type=int, default=0)
    parser.add_argument("--block", choices=["block_1", "block_2", "block_3", "block_4"], default="block_1")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--drop_fraction", type=float, default=0.1)
    parser.add_argument("--metric", choices=["patch", "pixel"], default="patch")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.squeeze().tolist(), std=STD.squeeze().tolist()),
    ])
    dataset = datasets.ImageFolder(root="./data/ILSVRC2012_img_val", transform=transform)

    if args.img_idx < 0:
        rng = random.Random(42)
        idx_sel = rng.randrange(len(dataset))
    else:
        idx_sel = args.img_idx

    img_t, label = dataset[idx_sel]
    img_t = img_t.to(device)

    # baseline model
    base_model = models.densenet121(pretrained=True).to(device).eval()

    # build mask according to CLI options
    stats_dir = Path(f"outputs/neuron_stats/{args.block}_k{args.k}")
    full_mask, target_block, _ = build_mask_and_block(base_model, stats_dir)

    # select cluster index directly
    chosen_cluster = args.cluster_idx

    labels = np.load(stats_dir / "cluster_results.npz", allow_pickle=True)["labels"]
    mask_np = np.ones_like(full_mask.cpu().numpy())
    idxs = np.where(labels == chosen_cluster)[0]
    mask_np[idxs] = 0.0
    mask = torch.tensor(mask_np, dtype=torch.float32)

    # filtered model
    filt_model = FilteredDenseNet(base_model, mask.to(device), target_block).to(device).eval()

    # attribution methods
    # Load ImageNet class index mapping for readable names
    try:
        with open("data/imagenet_class_index.json", "r") as fp:
            _idx2name = {int(k): v[1] for k, v in json.load(fp).items()}
    except FileNotFoundError:
        _idx2name = {}

    methods = {
        "IG": IntegratedGradients,
        "Saliency": Saliency,
    }

    n_cols = 7
    fig, axes = plt.subplots(len(methods), n_cols, figsize=(3*n_cols, 6))
    class_name = _idx2name.get(label, str(label))
    fig.suptitle(f"Class: {class_name} (label {label})", fontsize=14)
    summary_lines = []
    for row, (name, ctor) in enumerate(methods.items()):
        # baseline
        baser = ctor(base_model)
        attr_base = baser.attribute(img_t.unsqueeze(0), target=label)

        # filtered
        filtr = ctor(filt_model)
        attr_filt = filtr.attribute(img_t.unsqueeze(0), target=label)

        # overlays
        overlay_base = attribution_overlay(img_t, attr_base.squeeze())
        overlay_filt = attribution_overlay(img_t, attr_filt.squeeze())

        # masked images (patch-wise & pixel-wise)
        masked_patch_base = make_masked_image(img_t, attr_base.squeeze(), drop_fraction=args.drop_fraction)
        masked_patch_filt = make_masked_image(img_t, attr_filt.squeeze(), drop_fraction=args.drop_fraction)

        masked_pixel_base = make_masked_image_pixel(img_t, attr_base.squeeze(), drop_fraction=args.drop_fraction)
        masked_pixel_filt = make_masked_image_pixel(img_t, attr_filt.squeeze(), drop_fraction=args.drop_fraction)

        # Compute sparsity metrics
        attr_base_np = attr_base.squeeze().cpu().numpy()
        attr_filt_np = attr_filt.squeeze().cpu().numpy()
        
        if args.metric == "patch":
            sparsity_base = compute_sparsity_patch_wise(attr_base_np)
            sparsity_filt = compute_sparsity_patch_wise(attr_filt_np)
        else:  # pixel-wise
            sparsity_base = compute_sparsity_pixel_wise(attr_base_np)
            sparsity_filt = compute_sparsity_pixel_wise(attr_filt_np)

        # faithfulness score
        with torch.no_grad():
            # Baseline predictions
            base_output = base_model(img_t.unsqueeze(0))
            base_pred = base_output.argmax(1).item()
            base_conf = torch.softmax(base_output, dim=1)[0, label].item()
            prob_after_mask_base = torch.softmax(base_model(masked_patch_base.unsqueeze(0)), dim=1)[0, label].item()
            abs_drop_base = base_conf - prob_after_mask_base
            rel_drop_base = abs_drop_base / base_conf if base_conf > 0 else 0.0

            # Filtered predictions
            filt_output = filt_model(img_t.unsqueeze(0))
            filt_pred = filt_output.argmax(1).item()
            filt_conf = torch.softmax(filt_output, dim=1)[0, label].item()
            prob_after_mask_filt = torch.softmax(filt_model(masked_patch_filt.unsqueeze(0)), dim=1)[0, label].item()
            abs_drop_filt = filt_conf - prob_after_mask_filt
            rel_drop_filt = abs_drop_filt / filt_conf if filt_conf > 0 else 0.0

        # Prediction accuracy information
        base_correct = base_pred == label
        filt_correct = filt_pred == label
        
        base_pred_name = _idx2name.get(base_pred, str(base_pred)) if not base_correct else "✓"
        filt_pred_name = _idx2name.get(filt_pred, str(filt_pred)) if not filt_correct else "✓"

        faithfulness_diff = abs_drop_filt - abs_drop_base
        sparsity_diff = sparsity_filt - sparsity_base
        
        summary = (f"{name}:  conf_base={base_conf:.3f}  conf_filt={filt_conf:.3f} | "
                   f"faithfulness_base={abs_drop_base:.3f} faithfulness_filt={abs_drop_filt:.3f} (diff={faithfulness_diff:+.3f}) | "
                   f"sparsity_base={sparsity_base:.3f} sparsity_filt={sparsity_filt:.3f} (diff={sparsity_diff:+.3f}) | "
                   f"pred_base={'✓' if base_correct else base_pred_name} "
                   f"pred_filt={'✓' if filt_correct else filt_pred_name}")
        summary_lines.append(summary)
        print(summary)

        # plot
        titles = [
            "Original image",
            f"Baseline {name}",
            f"Filtered {name}",
            "Baseline pixel-mask",
            "Filtered pixel-mask",
            "Baseline patch-mask",
            "Filtered patch-mask",
        ]

        imgs_to_show = [
            (img_t * STD.to(img_t.device) + MEAN.to(img_t.device)).clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
            overlay_base,
            overlay_filt,
            (masked_pixel_base * STD.to(masked_pixel_base.device) + MEAN.to(masked_pixel_base.device)).clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
            (masked_pixel_filt * STD.to(masked_pixel_filt.device) + MEAN.to(masked_pixel_filt.device)).clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
            (masked_patch_base * STD.to(masked_patch_base.device) + MEAN.to(masked_patch_base.device)).clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
            (masked_patch_filt * STD.to(masked_patch_filt.device) + MEAN.to(masked_patch_filt.device)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        ]

        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(imgs_to_show[col])
            ax.set_title(titles[col], fontsize=8)
            ax.axis('off')

    for i, line in enumerate(summary_lines):
        fig.text(0.01, 0.05 - i*0.03, line, fontsize=8, ha='left')

    plt.tight_layout(rect=[0,0.1,1,1])
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = f"idx_{chosen_cluster}_k{args.k}"
    out_path = out_dir / f"attr_example_{idx_sel}_{cfg_name}.png"
    plt.savefig(out_path, dpi=300)
    print("Saved visualization to", out_path)

if __name__ == "__main__":
    main() 
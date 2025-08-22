from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

__all__ = [
    "set_seeds",
    "get_imagenet_dataloader",
    "compute_sparsity_pixel_wise",
    "compute_sparsity_patch_wise",
    "probability_drop_pixel_wise",
    "probability_drop_patch_wise",
]

# ===========================================================
# REPRODUCIBILITY

# Set seeds for deterministic behaviour
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===========================================================
# DATA SETUP

# Load the ImageNet validation subset
def get_imagenet_dataloader(root: str | Path = "./data/ILSVRC2012_img_val", batch_size: int = 32, num_workers: int = 4, shuffle: bool = False, n_samples: Optional[int] = None) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_full = datasets.ImageFolder(root=str(root), transform=transform)

    if n_samples is not None and n_samples < len(dataset_full):
        # deterministic subsample, we keep ordering to preserve label alignment
        rng = np.random.default_rng(0)
        subset_idx = rng.choice(len(dataset_full), size=n_samples, replace=False)
        dataset_full = torch.utils.data.Subset(dataset_full, subset_idx.tolist())

    return DataLoader(
        dataset_full,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ) 

# For baseline_eval.py and filtered_eval.py

# ============================================================================
# METRICS

# - How concentrated is the explanation in a few pixels?
# Mass-based sparsity: fraction of pixels NOT needed to accumulate 
# mass_fraction of the total attribution. Higher means more focused explanation
def compute_sparsity_pixel_wise(attr: np.ndarray, mass_fraction: float = 0.9) -> float:
    flat = np.abs(attr).flatten()
    total = flat.sum() # total attribution mass (scalar)
    if total == 0:
        return 0.0

    idx_desc = np.argsort(flat)[::-1] # largest to smallest
    cumulative_mass = np.cumsum(flat[idx_desc])
    k = np.searchsorted(cumulative_mass, mass_fraction * total) + 1  # +1 because index, k pixels account for mass_fraction
    return 1.0 - (k / flat.size)

# - How concentrated is the explanation in a few patches?
#   Same idea as pixel-wise sparsity but on non-overlapping patches.
def compute_sparsity_patch_wise(
    attr: np.ndarray,
    mass_fraction: float = 0.9,
    grid_size: int = 14,
) -> float:

    attr_abs = np.abs(attr)
    if attr_abs.ndim == 3:
        # collapse channel dimension
        attr_abs = attr_abs.sum(axis=0)

    H, W = attr_abs.shape
    patch_h, patch_w = H // grid_size, W // grid_size
    if patch_h == 0 or patch_w == 0:
        raise ValueError(
            f"grid_size {grid_size} too large for attribution map of size {H}x{W}."
        )

    # Aggregate attribution mass per patch
    patch_scores = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        for j in range(grid_size):
            h0, h1 = i * patch_h, (i + 1) * patch_h
            w0, w1 = j * patch_w, (j + 1) * patch_w
            patch_scores[i, j] = attr_abs[h0:h1, w0:w1].sum()

    total_mass = patch_scores.sum()
    if total_mass == 0:
        return 0.0

    flat = patch_scores.flatten()
    idx_sorted = np.argsort(-flat)  # descending
    cumulative_mass = np.cumsum(flat[idx_sorted])
    k = np.searchsorted(cumulative_mass, mass_fraction * total_mass) + 1

    return 1.0 - (k / flat.size)

# ============================================================================
# EVALUATION


# Drop in confidence when pixels covering *mass_fraction* of attribution mass are zeroed
def probability_drop_pixel_wise(
    model: torch.nn.Module,
    img: torch.Tensor,
    attr: np.ndarray,
    drop_fraction: float = 0.1,
    true_label: Optional[int] = None,
) -> float:
    with torch.no_grad():
        logits = model(img.unsqueeze(0))
        if true_label is not None:
            true_class = true_label
        else:
            true_class = logits.argmax(1).item()
        prob_orig = torch.softmax(logits, dim=1)[0, true_class].item()

    # collapse channel dimension so we rank by spatial importance
    attr_abs = np.abs(attr)
    if attr_abs.ndim == 3:
        attr_abs = attr_abs.sum(axis=0)  # (H, W)

    flat = attr_abs.flatten()
    if flat.size == 0:
        return 0.0

    k = max(1, int(round(drop_fraction * flat.size)))
    sorted_idx = np.argsort(-flat)[:k]  # top-k spatial positions

    # build 2-D mask and broadcast to all channels
    H, W = attr_abs.shape
    mask_2d = np.ones((H, W), dtype=np.float32)
    mask_2d[np.unravel_index(sorted_idx, (H, W))] = 0.0
    if img.dim() == 3:
        mask = np.broadcast_to(mask_2d, img.cpu().numpy().shape)
    else:
        mask = mask_2d  # grayscale

    masked_np = img.cpu().numpy() * mask
    masked = torch.from_numpy(masked_np).to(img.device)

    with torch.no_grad():
        logits_masked = model(masked.unsqueeze(0))
        prob_masked = torch.softmax(logits_masked, dim=1)[0, true_class].item()

    return prob_orig - prob_masked


# The image is partitioned into grid_size × grid_size non-overlapping patches
# Patches (16x16 pixels) whose aggregated attribution mass accounts for mass_fraction of the total
# are zeroed out and the resulting drop in the true-class probability is returned.
def probability_drop_patch_wise(
    model: torch.nn.Module,
    img: torch.Tensor,
    attr: np.ndarray,
    drop_fraction: float = 0.1,
    grid_size: int = 14,
    true_label: Optional[int] = None,
) -> float:

    # 1. Original probability
    with torch.no_grad():
        logits = model(img.unsqueeze(0))
        if true_label is not None:
            true_class = true_label
        else:
            true_class = logits.argmax(1).item()
        prob_orig = torch.softmax(logits, dim=1)[0, true_class].item()

    # 2. Prepare attribution map
    attr_abs = np.abs(attr) 
    if attr_abs.ndim == 3:
        # Sum over channels -> (H, W)
        attr_abs = attr_abs.sum(axis=0)
    H, W = attr_abs.shape
    patch_h, patch_w = H // grid_size, W // grid_size

    # 3. Aggregate attribution mass per patch
    patch_scores = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        for j in range(grid_size):
            h0, h1 = i * patch_h, (i + 1) * patch_h
            w0, w1 = j * patch_w, (j + 1) * patch_w
            patch_scores[i, j] = attr_abs[h0:h1, w0:w1].sum() # sum over the patch

    # 4. Select fixed top-k patches (fraction of total patches)
    flat_scores = patch_scores.flatten()
    idx_sorted = np.argsort(-flat_scores)  # descending
    k = max(1, int(round(drop_fraction * flat_scores.size)))
    selected_flat = idx_sorted[:k]

    # Convert flat indices back to 2-D grid coordinates
    selected_coords = [(idx // grid_size, idx % grid_size) for idx in selected_flat]

    # 5. Build mask that zeros the selected patches
    mask_2d = np.ones((H, W), dtype=np.float32)
    for i, j in selected_coords:
        h0, h1 = i * patch_h, (i + 1) * patch_h
        w0, w1 = j * patch_w, (j + 1) * patch_w
        mask_2d[h0:h1, w0:w1] = 0.0 # h0:h1, w0:w1 is the patch

    # Broadcast to all channels of the input tensor
    if img.dim() == 2: # grayscale image
        mask = mask_2d
    else: # RGB image
        mask = np.broadcast_to(mask_2d, img.cpu().numpy().shape) 

    # 6. Apply mask and compute probability drop
    img_np = img.cpu().numpy()
    masked_np = img_np * mask
    masked = torch.from_numpy(masked_np).to(img.device)

    with torch.no_grad(): 
        logits_masked = model(masked.unsqueeze(0)) 
        prob_masked = torch.softmax(logits_masked, dim=1)[0, true_class].item() 

    return prob_orig - prob_masked # positive if masking important patches lowers prob

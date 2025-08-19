import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
NUM_IMAGES = 20 # Number of images to visualize

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.squeeze().tolist(), std=STD.squeeze().tolist()),
    ])
dataset = datasets.ImageFolder(root="./data/ILSVRC2012_img_val", transform=transform)
indices = np.linspace(0, len(dataset)-1, NUM_IMAGES, dtype=int)

for i in indices:
    os.system(
        f"python visualize_attribution.py "
        f"--img_idx {i} "
        f"--cluster_idx 0 "
        f"--drop_fraction 0.1"
    )
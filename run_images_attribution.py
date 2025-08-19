import os

class_id = 404

for i in range(0,50):
    os.system(
        f"python visualize_attribution.py "
        f"--img_idx {i + class_id*50} "
        f"--cluster_idx 0 "
        f"--drop_fraction 0.1"
    )
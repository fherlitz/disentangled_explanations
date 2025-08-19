import os
import shutil

# Paths
val_dir = 'data/ILSVRC2012_img_val'
gt_path = 'data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
classes_path = 'data/imagenet_classes.txt'

# Read class list
with open(classes_path) as f:
    classes = [line.strip() for line in f.readlines()]

# Read ground truth labels
with open(gt_path) as f:
    labels = [int(line.strip()) for line in f.readlines()]

# Make a mapping from index to class
idx_to_class = {i: classes[i] for i in range(len(classes))}

# Get all image filenames (sorted to match order)
img_files = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])

for img, label in zip(img_files, labels): 
    class_name = idx_to_class[label - 1]  # labels are 1-based
    class_dir = os.path.join(val_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    src = os.path.join(val_dir, img)
    dst = os.path.join(class_dir, img)
    shutil.move(src, dst)

print("Validation images organized into class subfolders.") 
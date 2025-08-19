## data/

Prepare the ImageNet validation dataset and class mappings in this directory.

Expected structure after preparation:

```
data/
  ILSVRC2012_img_val/
    n01440764/
      ILSVRC2012_val_00000001.JPEG
      ...
    n01443537/
      ...
    ... (1000 class folders)
  imagenet_classes.txt
  imagenet_class_index.json
  subfolders.py
```

### 1) Place validation images
- Put all ImageNet validation images under `data/ILSVRC2012_img_val/`.

### 2) Organize into class subfolders
Run once to move images into 1000 class directories using the provided ground-truth mapping.

```bash
python data/subfolders.py
```

The script expects:
- `data/ILSVRC2012_img_val/` with the raw validation images
- `data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt` (from the devkit)

After running, `ILSVRC2012_img_val` will contain 1000 subfolders named by class (e.g., `n01440764`).

### 3) Class name mappings
- `imagenet_class_index.json`: mapping used to display human-readable class names in figures; obtain from common sources (e.g., Keras/torch releases).

No dataset files are provided in this repository. Only the folder structure and scripts are included.



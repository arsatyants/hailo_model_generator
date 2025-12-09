# STEP 1: Data Preparation & Annotation

Prepare captured images and create YOLO format annotations for training.

## Prerequisites

- Python 3.12+
- OpenCV: `pip install opencv-python`
- PyYAML: `pip install pyyaml`

## Workflow

### 1. Add Captured Images to Dataset

Copy images from `hailo-compile/captured_images/` to the dataset:

```bash
python3 add_images_to_dataset.py
```

**What it does:**
- Copies image files (jpg, png) to `images/` directory
- Creates empty `.txt` label files for each image in `labels/` directory
- Skips `result_*` images (inference outputs)
- Skips images that already exist

**Default paths:**
- Source: `../../captured_images`
- Target: `../../datasets/yolo_dataset`

**Custom paths:**
```bash
python3 add_images_to_dataset.py \
    --captured-dir /path/to/captured \
    --dataset-dir /path/to/dataset
```

---

### 2. Annotate Images

Interactive annotation tool with bounding box drawing:

```bash
python3 annotate_drones.py
```

**Controls:**
- **Mouse**: Click and drag to draw bounding box
- **Number keys (0-1)**: Select class before drawing
  - `0` = drone (regular)
  - `1` = IR-Drone (infrared)
- **ENTER**: Save current annotation and move to next image
- **SPACE**: Skip image (useful for negative samples)
- **D**: Delete last drawn box
- **Q**: Quit tool

**What it does:**
- Loads images from dataset sequentially
- Shows existing annotations (if any)
- Saves annotations in YOLO format to `.txt` files

**YOLO Format:**
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalized to [0, 1]

**Example label file:**
```
0 0.512 0.456 0.123 0.187
1 0.712 0.623 0.098 0.145
```

**Custom dataset path:**
```bash
python3 annotate_drones.py --dataset-dir /path/to/dataset
```

---

### 3. Verify Annotations (Optional)

Check dataset for errors before training:

```bash
python3 verify_annotations.py
```

**Checks performed:**
- Missing label files
- Invalid coordinate ranges (must be [0, 1])
- Invalid class IDs
- Malformed label format
- Statistics summary

**Output example:**
```
VERIFICATION RESULTS
==================================================
Total images: 85
  With labels: 72 (84.7%)
  Empty labels (negative samples): 13
  Missing labels: 0

Total bounding boxes: 98

✅ All annotations are valid!
   Average boxes per image: 1.36
```

---

## Dataset Structure

After completing these steps, your dataset should look like:

```
MODEL-GEN/datasets/yolo_dataset/
├── images/
│   ├── capture_001.jpg
│   ├── capture_002.jpg
│   └── ...
├── labels/
│   ├── capture_001.txt
│   ├── capture_002.txt
│   └── ...
└── data.yaml
```

---

## Creating data.yaml

If `data.yaml` doesn't exist, create it:

```yaml
# Absolute paths required
train: /absolute/path/to/MODEL-GEN/datasets/yolo_dataset/images
val: /absolute/path/to/MODEL-GEN/datasets/yolo_dataset/images

# Number of classes
nc: 2

# Class names (index matches class_id in labels)
names: ['drone', 'IR-Drone']
```

**Important:** Use absolute paths, not relative paths!

---

## Tips for Good Annotations

### Quality Guidelines
1. **Tight boxes**: Bbox should closely fit the object
2. **Complete objects**: Include entire visible object
3. **Difficult cases**: Still annotate partially occluded objects
4. **Negative samples**: Empty label files are valid (no objects in image)

### Class Selection
- **Class 0 (drone)**: Regular RGB visible drones
- **Class 1 (IR-Drone)**: Thermal/infrared visible drones

### Hard Negatives
Include images without any objects (empty labels) to reduce false positives:
- Sky-only images
- Indoor scenes
- Various backgrounds

### Dataset Balance
Aim for:
- 50+ images per class minimum
- 100+ images per class recommended
- 10-20% negative samples (empty labels)

---

## Troubleshooting

### Issue: No images found
**Solution:** Check captured_images path exists and contains image files

### Issue: Annotation tool won't start
**Solution:** 
```bash
pip install opencv-python pyyaml numpy
```

### Issue: Can't see bounding box while drawing
**Solution:** Click and hold, then drag - release when box is correct size

### Issue: Wrong class annotated
**Solution:** Press `D` to delete last box, select correct class with number key, redraw

---

## Next Step

Once you have annotated images, proceed to **STEP 2: Training**:

```bash
cd ../step2_training
python3 train_yolov8.py
```

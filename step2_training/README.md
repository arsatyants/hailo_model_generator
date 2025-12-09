# STEP 2: Model Training

Train YOLOv8 model on annotated drone dataset.

## Prerequisites

- Python 3.12+
- Ultralytics: `pip install ultralytics`
- PyTorch (installed automatically with ultralytics)
- CUDA (optional, for GPU acceleration - highly recommended)
- **Dataset split into train/val sets** (see below)

## Before Training: Split Dataset

If you haven't already split your dataset into train/val sets, run:

```bash
cd ../step1_data_preparation
python3 split_train_val.py --dataset-dir ../datasets --val-ratio 0.15
```

**What it does:**
- Randomly moves 15% of images from train/ to val/ directory
- Maintains corresponding label files
- Uses seed=42 for reproducibility

**Expected output:**
```
Split Train/Validation Dataset
Total images: 61
Split: 52 train, 9 validation
✅ Split complete!
   Train: 52 images
   Validation: 9 images
```

**Note:** Only run this once! Running again will re-shuffle your dataset.

---

## Quick Start

```bash
# Train with default settings (200 epochs, batch=32)
python3 train_yolov8.py ../datasets/data.yaml
```

This will:
1. Load YOLOv8s pretrained weights
2. Train on your annotated dataset (train/ directory)
3. Validate on validation set (val/ directory)
4. Save best model to `runs/detect/train_drone_ir/weights/best.pt`
5. Generate training plots and metrics

---

## Training Parameters

### Basic Training
```bash
# Default training (recommended for first run)
python3 train_yolov8.py --epochs 200 --batch 32

# Quick test (fast, lower accuracy)
python3 train_yolov8.py --epochs 50 --batch 16

# High quality (slow, better accuracy)
python3 train_yolov8.py --epochs 300 --batch 64
```

### Custom Dataset Path
```bash
python3 train_yolov8.py --data /path/to/your/data.yaml
```

### Memory Issues
If you get "CUDA out of memory" errors:
```bash
# Reduce batch size
python3 train_yolov8.py --batch 16

# Or even smaller
python3 train_yolov8.py --batch 8
```

### Custom Run Name
```bash
python3 train_yolov8.py --name my_training_run
```

---

## Understanding Training Output

### Real-time Metrics
During training, you'll see:
```
Epoch    GPU_mem    box_loss    cls_loss    dfl_loss    Instances    Size
  1/200     3.2G       1.234       0.567       1.123         45      640
```

**Key metrics:**
- `box_loss`: Bounding box localization error (lower is better)
- `cls_loss`: Classification error (lower is better)  
- `dfl_loss`: Distribution focal loss (lower is better)
- `Instances`: Number of objects in batch
- `Size`: Input image size (640x640)

### Validation Metrics
Every epoch shows validation performance:
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all         13         17      0.501      0.481      0.652      0.412
                drone         13          8      0.501      0.481      0.501      0.324
             IR-Drone         13          9      1.000      1.000      0.803      0.501
```

**Key metrics:**
- `P` (Precision): Accuracy of detections
- `R` (Recall): Percentage of objects found
- `mAP50`: Mean Average Precision at IoU=0.5 (main metric)
- `mAP50-95`: mAP averaged across IoU thresholds

**Good results:**
- mAP50 > 0.7 = Excellent
- mAP50 > 0.5 = Good
- mAP50 < 0.3 = Needs more training/data

### Early Stopping
Training stops automatically if mAP doesn't improve for 15 epochs (patience).

Example:
```
Epoch 19/200: Early stopping - no improvement in 15 epochs
```

---

## Training Results

### Output Structure
```
runs/detect/train_drone_ir/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Last epoch model
├── confusion_matrix.png  # Class prediction accuracy
├── F1_curve.png         # F1 score vs confidence
├── P_curve.png          # Precision curve
├── PR_curve.png         # Precision-Recall curve
├── R_curve.png          # Recall curve
├── results.png          # Training metrics over time
└── args.yaml            # Training configuration

```

### Key Files
- **best.pt**: Use this for inference and ONNX export
- **results.png**: Visual summary of training progress
- **confusion_matrix.png**: Shows misclassifications

---

## Monitoring Training

### TensorBoard (Optional)
View training in real-time:
```bash
tensorboard --logdir runs/detect
```
Then open http://localhost:6006 in browser

### Watch Results Live
```bash
# In another terminal
watch -n 1 ls -lh runs/detect/train_drone_ir/weights/
```

---

## Training Tips

### Dataset Size Guidelines
- **Minimum**: 30 images per class
- **Recommended**: 100+ images per class
- **Ideal**: 500+ images per class

### Improve Accuracy

**1. Add more data:**
```bash
cd ../step1_data_preparation
python3 add_images_to_dataset.py
python3 annotate_drones.py
cd ../step2_training
python3 train_yolov8.py  # Retrain
```

**2. Include hard negatives:**
- Images without objects (empty labels)
- Similar backgrounds
- Different lighting conditions

**3. Data augmentation (automatic):**
- YOLOv8 applies: rotation, scaling, color jitter, mosaic
- Controlled by ultralytics internally

**4. Longer training:**
```bash
python3 train_yolov8.py --epochs 300 --patience 30
```

---

## Hardware Requirements

| Hardware | Batch Size | Training Time (100 images, 200 epochs) |
|----------|------------|----------------------------------------|
| CPU only | 8 | ~6 hours |
| GPU 4GB | 16 | ~2 hours |
| GPU 8GB+ | 32 | ~1 hour |
| GPU 16GB+ | 64 | ~45 min |

**GPU highly recommended** for reasonable training times.

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
python3 train_yolov8.py --batch 8
```

### Issue: "No module named 'ultralytics'"
**Solution:** Install ultralytics
```bash
pip install ultralytics
```

### Issue: Training loss not decreasing
**Solutions:**
1. Check annotations are correct: `cd ../step1_data_preparation && python3 verify_annotations.py`
2. Ensure sufficient data (50+ images per class)
3. Try longer training: `--epochs 300`

### Issue: Very low mAP (<0.2)
**Solutions:**
1. Verify label format is correct (YOLO format)
2. Check data.yaml paths are absolute
3. Ensure annotations match images (same filenames)
4. Add more varied training data

### Issue: Training stops immediately
**Solution:** Check data.yaml exists and paths are correct:
```bash
cat ../datasets/data.yaml
```

---

## Next Step

After training completes successfully:

```bash
cd ../step3_onnx_export
python3 export_onnx_with_nms.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt
```

This exports your trained model to ONNX format with embedded NMS for Hailo compilation.

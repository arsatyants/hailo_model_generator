# YOLOv8 to Hailo-8 HEF Model Generation Pipeline

Complete workflow for preparing, training, and compiling YOLOv8 models for Hailo-8 AI accelerator deployment.

## Overview

This pipeline converts drone detection images into production-ready `.hef` files for Raspberry Pi 5 with Hailo-8.

**Input:** Raw captured images from camera  
**Output:** Optimized `.hef` model file for Hailo-8 inference

## Directory Structure

```
MODEL-GEN/
├── README.md                       # This file - pipeline overview
├── SETUP.md                        # Complete setup instructions
├── LICENSE                         # MIT license
├── run_pipeline.sh                 # Master script (runs all 5 steps)
├── data.yaml.template              # Dataset configuration template
├── .gitignore                      # Git ignore rules
│
├── datasets/                       # YOUR TRAINING DATA GOES HERE
│   ├── train/
│   │   ├── images/                 # Training images
│   │   └── labels/                 # YOLO format labels
│   ├── val/
│   │   ├── images/                 # Validation images
│   │   └── labels/                 # Validation labels
│   └── data.yaml                   # Dataset configuration
│
├── captured_images/                # Raw images from camera (before annotation)
│
├── models/                         # Final HEF models (for deployment)
│   └── model.hef                   # Ready for Raspberry Pi
│
├── step1_data_preparation/
│   ├── README.md                   # Annotation guide
│   ├── requirements.txt
│   ├── add_images_to_dataset.py    # Import images to dataset
│   ├── annotate_drones.py          # Interactive annotation tool
│   └── verify_annotations.py       # Dataset validation
│
├── step2_training/
│   ├── README.md                   # Training guide
│   ├── requirements.txt
│   ├── train_yolov8.py             # YOLOv8 training script
│   └── runs/detect/train*/         # Training outputs (auto-generated)
│
├── step3_onnx_export/
│   ├── README.md                   # ONNX export guide
│   ├── requirements.txt
│   ├── export_onnx_with_nms.py     # PT → ONNX with NMS
│   └── verify_onnx_export.py       # Quality verification
│
├── step4_hef_compilation/
│   ├── README.md                   # HEF compilation guide
│   ├── requirements.txt
│   ├── compile_to_hef.py           # ONNX → HEF compiler
│   ├── prepare_calibration.py      # Calibration data prep
│   ├── hailo_optimization.mscript  # Compiler settings
│   ├── .venv_hailo_full/           # Hailo SDK (install separately)
│   └── calibration_data/           # Calibration images (auto-generated)
│
└── step5_raspberry_pi_testing/
    ├── README.md                   # Testing guide
    ├── requirements.txt
    ├── hailo_detect_live.py        # Real-time inference script
    ├── deploy_to_pi.sh             # Automated deployment
    └── test_results/               # Saved detection results
```

**Key Folders:**
- `datasets/` - Your training data with images and YOLO labels
- `captured_images/` - Raw images before annotation
- `models/` - Final HEF files ready for deployment
- `step*/` - Individual pipeline steps with scripts and documentation

## Prerequisites

### Hardware
- Development machine (x86_64) with GPU recommended for training
- Raspberry Pi 5 with Hailo-8 module (for deployment only)

### Software
- Python 3.12+
- CUDA (optional, for faster training)
- Hailo Dataflow Compiler SDK 3.33.0+
- Git

### Python Packages
```bash
# Main environment (.venv)
pip install ultralytics opencv-python pillow pyyaml numpy

# Hailo environment (.venv_hailo_full) - installed by Hailo SDK
# DO NOT modify this environment manually
```

## Pipeline Steps

### STEP 1: Data Preparation & Annotation
**Location:** `step1_data_preparation/`

Prepare images and create YOLO format annotations:

```bash
cd step1_data_preparation

# 1. Copy captured images to dataset
python3 add_images_to_dataset.py

# 2. Annotate images interactively
python3 annotate_drones.py

# 3. Verify annotations (optional)
python3 verify_annotations.py
```

**Output:** Dataset at `../datasets/` with:
- `train/images/` - Training images
- `train/labels/` - YOLO format labels (`.txt` files)
- `val/images/` - Validation images
- `val/labels/` - Validation labels
- `data.yaml` - Dataset configuration

---

### STEP 2: Model Training
**Location:** `step2_training/`

Train YOLOv8 model on annotated dataset:

```bash
cd step2_training

# Train with default parameters (100 epochs, batch=16)
python3 train_yolov8.py ../datasets/data.yaml

# Or use custom parameters
python3 train_yolov8.py ../datasets/data.yaml --epochs 200 --batch 32 --imgsz 640
```

**Output:** Trained model at `runs/detect/train*/weights/best.pt`

**Training Tips:**
- Monitor training: Check `runs/detect/train*/` for plots
- Early stopping: Uses patience=15 (stops if no improvement after 15 epochs)
- Validation: Automatically uses val/ split from dataset

---

### STEP 3: ONNX Export with NMS
**Location:** `step3_onnx_export/`

Export PyTorch model to ONNX format **with embedded NMS** (critical for Hailo compatibility):

```bash
cd step3_onnx_export

# Export with NMS enabled
python3 export_onnx_with_nms.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt

# Verify export (tests for false positives)
python3 verify_onnx_export.py best_nms.onnx
```

**Output:** `best_nms.onnx` (42-43 MB)

**Critical Settings:**
- `nms=True` - Embeds NMS in ONNX graph
- `opset=11` - Compatible with Hailo compiler
- `simplify=True` - Optimizes graph structure
- Output format: `(1, 300, 6)` instead of raw `(1, 6, 8400)`

**Verification:** Black image test should produce **0 detections** (not 4 false positives).

---

### STEP 4: HEF Compilation for Hailo-8
**Location:** `step4_hef_compilation/`

Compile ONNX to Hailo Executable Format (HEF):

```bash
cd step4_hef_compilation

# Prepare calibration data
python3 prepare_calibration.py ../datasets --num-samples 64

# Compile ONNX to HEF
python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx --output drone_detector
```

**Output:** `drone_detector.hef` (~9 MB) in `step4_hef_compilation/`

**Compilation Process:**
1. Parses ONNX model (strips NMS layer - Hailo doesn't support it)
2. Quantizes model using calibration images
3. Optimizes for Hailo-8 neural core
4. Generates HEF binary

**Important Notes:**
- NMS is stripped during compilation (must be applied in Python inference code)
- HEF expects **UINT8 input** format `(640, 640, 3)` in HWC layout
- Calibration uses 64 representative images from training set
- Input normalization: [0-255] UINT8, NOT [0-1] float32

---

### STEP 5: Raspberry Pi Testing
**Location:** `step5_raspberry_pi_testing/`

Deploy and test HEF model on Raspberry Pi 5 with Hailo-8:

```bash
cd step5_raspberry_pi_testing

# Automated deployment
./deploy_to_pi.sh 192.168.3.205 pi

# Then on Raspberry Pi (after SSH):
cd ~/MODEL-GEN/scripts
python3 hailo_detect_live.py --model ../models/model.hef
```

**Features:**
- Real-time inference at 40-50 FPS (YOLOv8s)
- Support for USB cameras and Raspberry Pi Camera Module
- Headless mode for remote operation
- Auto-save detections for validation
- Performance metrics and statistics

**Testing Modes:**
- **Live display**: Visual feedback with bounding boxes
- **Headless**: Save frames with detections only
- **Recording**: Auto-save all detection frames
- **Picamera**: Use Raspberry Pi Camera Module (recommended)

---

## Complete Workflow Example

```bash
# Fresh start - complete pipeline
cd MODEL-GEN

# 1. Prepare data
cd step1_data_preparation
python3 add_images_to_dataset.py
python3 annotate_drones.py
cd ..

# 2. Train model
cd step2_training
python3 train_yolov8.py --epochs 200
cd ..

# 3. Export to ONNX with NMS
cd step3_onnx_export
python3 export_onnx_with_nms.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt
python3 verify_onnx_export.py best_nms.onnx
cd ..

# 4. Compile to HEF
cd step4_hef_compilation
./compile_to_hef.sh ../step3_onnx_export/best_nms.onnx --name my_drone_detector
cd ..

# Result: hailo_models/my_drone_detector.hef ready for deployment!
```

---

## Troubleshooting

### Issue: ONNX Export Has False Positives
**Symptom:** `verify_onnx_export.py` shows 4 detections on black image  
**Solution:** Ensure `nms=True` in export command:
```python
model.export(format='onnx', nms=True, imgsz=640, opset=11, simplify=True)
```

### Issue: HEF Compilation Fails
**Symptom:** "Unsupported operation" errors  
**Solution:** Verify ONNX was exported with `opset=11` and `simplify=True`

### Issue: Pi Inference Fails with "Invalid argument"
**Symptom:** HEF loads but inference crashes  
**Solution:** Check preprocessing - HEF expects UINT8 [0-255], not float32 [0-1]:
```python
# CORRECT
rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)  # Keep UINT8
input_tensor = rgb  # Shape: (640, 640, 3) uint8

# WRONG
normalized = rgb.astype(np.float32) / 255.0  # Don't normalize!
```

### Issue: Low Detection Accuracy
**Solutions:**
1. Add more training images (aim for 100+ per class)
2. Include hard negatives (images without objects)
3. Increase training epochs
4. Verify label quality with `verify_annotations.py`

---

## File Format Reference

### YOLO Label Format (.txt)
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalized to [0, 1]. Example:
```
0 0.5 0.5 0.2 0.3
1 0.7 0.3 0.15 0.2
```

### data.yaml Format
```yaml
train: /absolute/path/to/dataset/images
val: /absolute/path/to/dataset/images  # Same as train for auto-split
nc: 2
names: ['drone', 'IR-Drone']
```

### HEF Input/Output Specification
```python
# Input
shape: (640, 640, 3)  # HWC format
dtype: uint8
range: [0, 255]

# Output (6 tensors - raw YOLO format, NMS stripped)
output_0: (1, 80, 80, 64)  # reg stride 8
output_1: (1, 20, 20, 64)  # reg stride 32
output_2: (1, 20, 20, 2)   # cls stride 32
output_3: (1, 40, 40, 2)   # cls stride 16
output_4: (1, 80, 80, 2)   # cls stride 8
output_5: (1, 40, 40, 64)  # reg stride 16
```

---

## Performance Expectations

| Stage | Time | Hardware |
|-------|------|----------|
| Annotation | ~30 sec/image | Any |
| Training (200 epochs) | 1-3 hours | GPU recommended |
| ONNX Export | 30 seconds | CPU |
| HEF Compilation | 2-3 minutes | x86_64 required |
| Inference (Pi5+Hailo) | 10-15ms/frame | 60-100 FPS |

---

## Version History

**v1.0** (Dec 2025)
- Initial release
- YOLOv8s base model
- 2-class detection (drone, IR-Drone)
- Hailo SDK 3.33.0 compatibility
- Complete UINT8 input pipeline

---

## Credits

Based on the computer-vision project by arsatyants.  
Hailo-8 integration and optimization December 2025.

---

## License

Follow the parent project license terms.

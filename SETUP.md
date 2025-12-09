# MODEL-GEN Quick Setup Guide

Complete setup instructions to get started with the MODEL-GEN pipeline.

## Prerequisites

### Hardware Requirements

**Development Machine** (for training and compilation):
- **CPU**: 4+ cores recommended (8+ for faster training)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but highly recommended for training)
  - GTX 1060 6GB or better
  - CUDA 11.8+ and cuDNN installed
- **Storage**: 20GB free space
- **OS**: Ubuntu 20.04+ or similar Linux distribution

**Deployment Target** (for inference):
- **Device**: Raspberry Pi 5 (8GB recommended)
- **Accelerator**: Hailo-8 AI module (M.2 or HAT+ form factor)
- **Camera**: Raspberry Pi Camera Module 3 or USB camera
- **Power**: 5V 5A power supply (for Pi + Hailo)

### Software Requirements

#### Python Environment

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Create virtual environment
cd MODEL-GEN
python3 -m venv .venv
source .venv/bin/activate
```

#### CUDA Setup (Optional - for GPU training)

```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA Toolkit (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads
# Follow instructions for your OS

# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Hailo Dataflow Compiler SDK

**Required for STEP 4 (HEF compilation)**

1. **Register** at [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free)
2. **Download** Hailo Dataflow Compiler SDK v3.33.0+
3. **Extract** to `step4_hef_compilation/.venv_hailo_full/`

```bash
# After downloading
cd step4_hef_compilation
tar -xzf hailo_dataflow_compiler_sdk_v3.33.0.tar.gz
# Creates .venv_hailo_full/ directory in step4_hef_compilation/
cd ..
```

**Verify installation**:
```bash
ls step4_hef_compilation/.venv_hailo_full/bin/activate
# Should exist
```

## Installation

### Step-by-Step Setup

#### 1. Clone or Extract MODEL-GEN

```bash
# If part of larger repo
cd computer-vision/MODEL-GEN

# Or extract standalone package
tar -xzf MODEL-GEN.tar.gz
cd MODEL-GEN
```

#### 2. Install Dependencies for Each Step

```bash
# Activate virtual environment
source .venv/bin/activate

# STEP 1: Data preparation
cd step1_data_preparation
pip install -r requirements.txt
cd ..

# STEP 2: Training
cd step2_training
pip install -r requirements.txt
cd ..

# STEP 3: ONNX export
cd step3_onnx_export
pip install -r requirements.txt
cd ..

# STEP 4: HEF compilation
cd step4_hef_compilation
pip install -r requirements.txt
cd ..
```

**Note**: STEP 4 also requires Hailo SDK (installed separately, see above).

#### 3. Verify Installation

```bash
# Check Python packages
python3 -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"

# Check CUDA (if using GPU)
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Quick Start

### Option 1: Run Complete Pipeline

Run all 4 steps automatically:

```bash
source .venv/bin/activate
./run_pipeline.sh /path/to/your/dataset output_model_name
```

**Example**:
```bash
./run_pipeline.sh ../lesson_18/drone_dataset/yolo_dataset drone_detector
```

This will:
1. Verify dataset annotations
2. Train YOLOv8 model (20-30 minutes)
3. Export to ONNX with NMS (30 seconds)
4. Compile to HEF for Hailo (10 minutes)

**Total time**: ~30-45 minutes

### Option 2: Run Steps Individually

#### STEP 1: Prepare and Annotate Data

```bash
cd step1_data_preparation

# Add new images to dataset
python3 add_images_to_dataset.py

# Annotate images interactively
python3 annotate_drones.py

# Verify annotations
python3 verify_annotations.py /path/to/dataset
```

See `step1_data_preparation/README.md` for detailed instructions.

#### STEP 2: Train YOLOv8 Model

```bash
cd step2_training

# Train with default settings (100 epochs)
python3 train_yolov8.py /path/to/dataset/data.yaml

# Custom training
python3 train_yolov8.py /path/to/dataset/data.yaml \
    --name my_drone_model \
    --epochs 200 \
    --batch 16 \
    --imgsz 640
```

See `step2_training/README.md` for parameter tuning.

#### STEP 3: Export to ONNX

```bash
cd step3_onnx_export

# Export trained model
python3 export_onnx_with_nms.py ../step2_training/runs/detect/train/weights/best.pt

# Verify export quality (black image test)
python3 verify_onnx_export.py best_nms.onnx
```

See `step3_onnx_export/README.md` for ONNX format details.

#### STEP 4: Compile to HEF

```bash
cd step4_hef_compilation

# Prepare calibration data
python3 prepare_calibration.py /path/to/dataset --num-samples 64

# Compile to HEF
python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx --output drone_detector
```

See `step4_hef_compilation/README.md` for Hailo SDK setup.

## Dataset Setup

### Create Your Own Dataset

#### Directory Structure

```
your_dataset/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
└── val/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

#### data.yaml Configuration

Copy template and edit:

```bash
cp data.yaml.template your_dataset/data.yaml
nano your_dataset/data.yaml
```

**Critical**: Use absolute paths:

```yaml
train: /home/user/MODEL-GEN/your_dataset/train/images
val: /home/user/MODEL-GEN/your_dataset/val/images
nc: 2
names:
  0: drone
  1: IR-Drone
```

#### Label Format

YOLO format (normalized coordinates):

```
class_id x_center y_center width height
```

Example `img_001.txt`:
```
0 0.5 0.5 0.3 0.2
1 0.7 0.3 0.15 0.1
```

Values are normalized to [0, 1]:
- `x_center`, `y_center`: Center of bounding box
- `width`, `height`: Box dimensions
- All values relative to image size

#### Annotation Tool

Use the included interactive annotator:

```bash
cd step1_data_preparation
python3 annotate_drones.py /path/to/images/
```

**Controls**:
- **Mouse**: Click and drag to draw bounding box
- **0-9**: Select class (0=drone, 1=IR-Drone, etc.)
- **ENTER**: Save and move to next image
- **SPACE**: Skip image
- **ESC**: Exit

### Use Existing Dataset

If you have a dataset from Roboflow or similar:

1. **Export as YOLOv8 format** (not YOLOv5)
2. **Extract** to your working directory
3. **Update** `data.yaml` with absolute paths
4. **Verify** with:
   ```bash
   cd step1_data_preparation
   python3 verify_annotations.py /path/to/dataset
   ```

## Troubleshooting Setup

### "No module named 'ultralytics'"

**Solution**:
```bash
source .venv/bin/activate
pip install ultralytics
```

### "CUDA out of memory" during training

**Solutions**:
1. **Reduce batch size**:
   ```bash
   python3 train_yolov8.py data.yaml --batch 8
   ```

2. **Use smaller model**:
   ```bash
   python3 train_yolov8.py data.yaml --model yolov8n.pt
   ```

3. **Train on CPU** (slower):
   ```bash
   python3 train_yolov8.py data.yaml --device cpu
   ```

### "Hailo SDK not found"

**Solution**: Install Hailo Dataflow Compiler SDK (see Prerequisites section above)

### Permission denied when running scripts

**Solution**:
```bash
chmod +x run_pipeline.sh
chmod +x step1_data_preparation/*.py
chmod +x step2_training/*.py
chmod +x step3_onnx_export/*.py
chmod +x step4_hef_compilation/*.py
```

### Import errors with cv2 or numpy

**Solution**:
```bash
pip install --upgrade opencv-python numpy
```

## Next Steps

After successful setup:

1. **Read the main README.md** for pipeline overview
2. **Prepare your dataset** (STEP 1)
3. **Run pipeline** with `./run_pipeline.sh`
4. **Deploy HEF file** to Raspberry Pi with Hailo-8

## Support

For detailed documentation:
- Main pipeline: `README.md`
- Data preparation: `step1_data_preparation/README.md`
- Training: `step2_training/README.md`
- ONNX export: `step3_onnx_export/README.md`
- HEF compilation: `step4_hef_compilation/README.md`

For Hailo-specific issues:
- [Hailo Community Forum](https://community.hailo.ai/)
- [Hailo Developer Documentation](https://hailo.ai/developer-zone/documentation/)

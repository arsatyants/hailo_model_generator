# Copilot Instructions: Hailo Model Generator

## Project Overview

This is a **5-step ML pipeline** that converts raw images into hardware-optimized `.hef` models for the Hailo-8 AI accelerator. The workflow: capture → annotate → train YOLOv8 → export ONNX → compile HEF → deploy to Raspberry Pi 5.

**Critical context**: This is NOT a standard ONNX export. Hailo-8 requires specific ONNX configurations (NMS embedded, opset=11, simplified graphs) that differ from typical deployment targets.

## Architecture & Data Flow

```
captured_images/ → datasets/ → step2_training/runs/ → step3_onnx_export/ → step4_hef_compilation/ → models/
                   (YOLO labels)  (best.pt model)    (ONNX with NMS)   (HEF binary)           (deployment)
```

**Key directories:**
- `datasets/`: Training data in YOLO format (train/val splits, `data.yaml` uses ABSOLUTE paths only)
- `step*/`: Self-contained pipeline stages with dedicated READMEs and requirements.txt
- `models/`: Final HEF files (9MB typical) ready for Pi deployment
- `step4_hef_compilation/.venv_hailo_full/`: **Hailo Dataflow Compiler SDK 3.33.0+** (SEPARATE from main `.venv`, do not modify)

**External Dependencies** (may reference parent directories):
- Hailo SDK install location: Code may reference `../../hailo-compile/.venv_hailo_full/` or similar paths outside project root
- Training datasets: Some scripts default to paths like `../../lesson_18/drone_dataset/yolo_dataset/` from external projects
- Always verify and update paths in scripts/configs to match your actual directory structure

## Critical Hailo-Specific Patterns

### ONNX Export Requirements (Step 3)
```python
# CRITICAL: nms=True is required for Hailo compatibility
model.export(
    format='onnx',
    nms=True,        # Embeds NMS → output (1, 300, 6) not (1, 6, 8400)
    opset=11,        # Hailo compiler requirement
    simplify=True    # Graph optimization for edge deployment
)
```

**Why this matters**: Without `nms=True`, ONNX outputs raw anchor predictions that Hailo's compiler cannot optimize. The black-image verification test (`verify_onnx_export.py`) checks for false positives—if it fails, re-export is required.

### Dual Virtual Environment Pattern + Version Split
**Development (Steps 1-4) on Linux workstation:**
- **Main `.venv`**: Ultralytics, OpenCV, PyTorch (training/annotation)
- **`.venv_hailo_full`**: **Hailo Dataflow Compiler SDK 3.33.0+** with `hailo_sdk_client` (compilation only)

**Deployment (Step 5) on Raspberry Pi:**
- **HailoRT 4.23+**: Runtime library only (no SDK needed on Pi)
- Installed via `python3-hailort` package or HailoRT installer

```bash
# Workstation: Steps 2-3 use main venv
source .venv/bin/activate
python3 train_yolov8.py

# Workstation: Step 4 uses Hailo SDK venv internally (3.33.0+)
python3 compile_to_hef.py  # Script switches to .venv_hailo_full automatically

# Raspberry Pi: Step 5 uses HailoRT 4.23+ (different from SDK)
python3 hailo_detect_live.py model.hef  # Uses installed HailoRT runtime
```

**Never run `pip install` in `.venv_hailo_full`**—it's managed by Hailo SDK installer.

### Calibration Data Format
```python
# For step4 quantization (INT8 conversion)
# Format: UINT8 [0-255], NOT normalized float32
image = cv2.imread(img_path)  # Already UINT8
image = cv2.resize(image, (640, 640))
cv2.imwrite(output_path, image)  # Preserve UINT8 encoding
```

Hailo's quantization process expects **raw pixel values**, not normalized tensors. Using float32 causes compilation to fail silently with poor accuracy.

## Build & Test Workflows

### Complete Pipeline
```bash
./run_pipeline.sh datasets/ drone_detector
# Runs all 5 steps sequentially (~30-60 min)
# Outputs: models/drone_detector.hef
```

### Individual Steps
```bash
# Step 1: Annotation (interactive OpenCV tool)
cd step1_data_preparation
python3 annotate_drones.py  # Keys: 0-1 select class, click-drag to draw bbox

# Step 2: Training (uses early stopping patience=15)
cd step2_training
python3 train_yolov8.py ../datasets/data.yaml --epochs 200 --batch 32
# Output: runs/detect/train*/weights/best.pt

# Step 3: ONNX export + verification
cd step3_onnx_export
python3 export_onnx_with_nms.py ../step2_training/runs/detect/train/weights/best.pt
python3 verify_onnx_export.py best_nms.onnx  # Must pass: 0 false positives on black image

# Step 4: HEF compilation (requires Hailo SDK)
cd step4_hef_compilation
python3 prepare_calibration.py ../datasets --num-samples 64
python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx --output drone_detector
# Output: drone_detector.hef (~9MB)

# Step 5: Deploy to Pi (automated)
cd step5_raspberry_pi_testing
./deploy_to_pi.sh 192.168.3.205 pi  # SSH into Pi, copies files, installs HailoRT
```

### Verification Scripts
- `step1/verify_annotations.py`: Checks YOLO label format, coordinate bounds, missing files
- `step3/verify_onnx_export.py`: Black-image test (detects false positives from bad NMS config)
- All scripts use emoji-based status indicators (✅ ❌ 📊) for output formatting

## Project-Specific Conventions

### YOLO Label Format
```
# datasets/train/labels/image_001.txt
0 0.5 0.5 0.2 0.3  # class x_center y_center width height (all normalized 0-1)
```

Empty `.txt` files are **valid** (negative samples with no objects).

### data.yaml Requirements
```yaml
# CRITICAL: Use ABSOLUTE paths only
train: /home/user/hailo_model_generator/datasets/train/images  # NOT ./train/images
val: /home/user/hailo_model_generator/datasets/val/images

nc: 2
names:
  0: Aircraft
  1: IR-Drone
```

Ultralytics fails silently with relative paths in multi-step pipelines.

### hailo_optimization.mscript
```python
enable_loose_mode  # Suppresses shape warnings (safe for YOLOv8 with NMS)
optimization_level(0)  # 0=fast compile, 2=best accuracy
```

Applied during Step 4 compilation. `enable_loose_mode` is REQUIRED for models with NMS layers—without it, compilation fails with shape mismatch errors.

## Raspberry Pi Deployment

### HailoRT API Usage Pattern
```python
from hailo_platform import VDevice, HEF, InputVStreamParams, OutputVStreamParams, FormatType

# Load HEF
hef = HEF("model.hef")
vdevice = VDevice()
network_group = vdevice.configure(hef)[0]

# Configure streams
input_params = InputVStreamParams.make_from_network_group(
    network_group, quantized=True, format_type=FormatType.UINT8  # Input: UINT8
)
output_params = OutputVStreamParams.make_from_network_group(
    network_group, quantized=False, format_type=FormatType.FLOAT32  # Output: dequantized
)
```

**Key difference**: HailoRT 4.23+ uses `FormatType.UINT8/FLOAT32` (not earlier versions' quantized boolean only). This is the **runtime library** on Pi, distinct from the **Dataflow Compiler SDK 3.33.0+** used on the workstation.

### Camera Integration
```python
# Raspberry Pi Camera Module 3
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 640)}))
```

`hailo_detect_live.py` supports both Picamera2 and USB webcams with fallback detection.

## Common Pitfalls

1. **"data.yaml path error"**: Always use absolute paths in `data.yaml`, not `./train/images`
2. **"ONNX has 4 false positives"**: Re-export with `nms=True` explicitly set
3. **"Hailo SDK not found"**: Check `.venv_hailo_full/` exists at `step4_hef_compilation/.venv_hailo_full/` (Dataflow Compiler SDK 3.33.0+)
4. **"Calibration error"**: Verify images are UINT8, not float32 (use `cv2.imwrite`, not `np.save`)
5. **"HEF inference crash on Pi"**: Ensure HailoRT 4.23+ is installed on Pi (different from workstation's SDK 3.33.0+)
6. **"External path errors"**: Scripts may reference `../../hailo-compile/` or `../../lesson_18/`—update paths to match your directory structure

## Testing & Validation

Run verification after each critical step:
```bash
cd step1_data_preparation && python3 verify_annotations.py ../datasets
cd step3_onnx_export && python3 verify_onnx_export.py best_nms.onnx
```

For full pipeline validation, use `run_pipeline.sh` which includes automated checks between stages.

# Hailo Model Generator Wiki

Complete documentation for the YOLOv8 to Hailo-8 HEF Model Generation Pipeline.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Advanced Topics](#advanced-topics)
5. [Troubleshooting](#troubleshooting)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

---

## Getting Started

### What is This Project?

This pipeline converts raw camera images into optimized neural network models that run on the Hailo-8 AI accelerator. The end result is a `.hef` file that can perform real-time object detection on a Raspberry Pi 5 at 60-100 FPS.

### Quick Start (5 Minutes)

```bash
# Clone the repository
git clone <repository-url>
cd hailo_model_generator

# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r step1_data_preparation/requirements.txt

# Add some images to captured_images/ folder
# Run the complete pipeline
./run_pipeline.sh datasets/ my_model
```

### System Requirements

**Development Machine:**
- OS: Ubuntu 20.04+ or similar Linux distribution
- CPU: x86_64 architecture (required for Hailo SDK)
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU with CUDA support (optional, for faster training)
- Storage: 10GB free space

**Deployment Target:**
- Raspberry Pi 5
- Hailo-8 AI Accelerator Module
- Raspberry Pi Camera Module 3 or USB webcam
- Raspberry Pi OS (64-bit)

---

## Architecture Overview

### Pipeline Stages

```
┌─────────────────────┐
│ Captured Images     │ Raw camera images
│ (PNG/JPG)          │
└──────────┬──────────┘
           │ Step 1: Annotation
           ↓
┌─────────────────────┐
│ YOLO Dataset        │ Images + labels
│ (train/val splits) │
└──────────┬──────────┘
           │ Step 2: Training
           ↓
┌─────────────────────┐
│ PyTorch Model       │ best.pt (~20MB)
│ (YOLOv8)           │
└──────────┬──────────┘
           │ Step 3: ONNX Export
           ↓
┌─────────────────────┐
│ ONNX Model          │ With/without NMS
│ (opset 11)         │
└──────────┬──────────┘
           │ Step 4: HEF Compilation
           ↓
┌─────────────────────┐
│ HEF Binary          │ Optimized for Hailo-8
│ (INT8 quantized)   │ (~9MB)
└──────────┬──────────┘
           │ Step 5: Deployment
           ↓
┌─────────────────────┐
│ Raspberry Pi 5      │ Real-time inference
│ + Hailo-8          │ 60-100 FPS
└─────────────────────┘
```

### Data Flow

**Training Phase (Development Machine):**
1. Raw images → YOLO annotations
2. Annotated dataset → Trained PyTorch model
3. PyTorch model → ONNX model
4. ONNX model → HEF binary (with quantization)

**Inference Phase (Raspberry Pi):**
1. Camera frame (RGB) → Preprocessing (resize, pad)
2. UINT8 input → Hailo-8 inference
3. Raw outputs → Python NMS (if needed)
4. Filtered detections → Visualization/logging

### Key Technologies

- **YOLOv8**: State-of-the-art object detection architecture
- **Hailo-8**: Neural processing unit with 26 TOPS performance
- **ONNX**: Intermediate format for model portability
- **HailoRT**: Runtime library for inference on Hailo-8
- **Dataflow Compiler**: Hailo's tool for model optimization

---

## Step-by-Step Guide

### Step 1: Data Preparation & Annotation

**Goal:** Create a labeled dataset in YOLO format.

#### 1.1 Capture Images

Place raw images in `captured_images/` directory:
```bash
captured_images/
├── drone_001.jpg
├── drone_002.jpg
└── ...
```

**Best practices:**
- Capture diverse scenes (different lighting, backgrounds, angles)
- Include images without objects (negative samples)
- Aim for 100+ images per class
- Use consistent resolution (640x640 recommended)

#### 1.2 Run Annotation Tool

```bash
cd step1_data_preparation
python3 annotate_drones.py --dataset ../datasets/train
```

**Interactive controls:**
- **Mouse**: Click and drag to draw bounding boxes
- **Keys 0-9**: Select object class
- **Space**: Save current annotation and move to next image
- **Backspace**: Delete last box
- **Escape**: Exit without saving

**Output format:**
```
datasets/
├── train/
│   ├── images/
│   │   └── drone_001.jpg
│   └── labels/
│       └── drone_001.txt  # YOLO format: class x y w h
└── data.yaml
```

#### 1.3 Verify Dataset

```bash
python3 verify_annotations.py ../datasets
```

This checks for:
- Missing label files
- Invalid coordinates (outside [0,1] range)
- Empty annotations
- Mismatched image/label counts

---

### Step 2: Model Training

**Goal:** Train a YOLOv8 model on your annotated dataset.

#### 2.1 Prepare data.yaml

Ensure absolute paths in `datasets/data.yaml`:
```yaml
train: /home/user/hailo_model_generator/datasets/train/images
val: /home/user/hailo_model_generator/datasets/val/images
nc: 2
names:
  0: drone
  1: IR-Drone
```

#### 2.2 Start Training

```bash
cd step2_training
python3 train_yolov8.py ../datasets/data.yaml --epochs 200 --batch 32
```

**Parameters explained:**
- `--epochs`: Number of training iterations (default: 200)
- `--batch`: Batch size (adjust based on GPU memory)
- `--imgsz`: Input image size (default: 640)
- `--patience`: Early stopping patience (default: 15)
- `--name`: Training run name (default: train_drone_ir)

#### 2.3 Monitor Training

Training outputs are saved to `runs/detect/<name>/`:
```
runs/detect/train_drone_ir/
├── weights/
│   ├── best.pt       # Best model checkpoint
│   └── last.pt       # Last epoch checkpoint
├── results.csv       # Training metrics
├── confusion_matrix.png
├── PR_curve.png
└── results.png       # Loss/mAP plots
```

**Key metrics:**
- **mAP50**: Mean average precision at 50% IoU (aim for >0.8)
- **mAP50-95**: Stricter metric (aim for >0.5)
- **Box loss**: Should decrease steadily
- **Class loss**: Should converge to low value

#### 2.4 Training Tips

**If accuracy is low:**
1. Add more training images (especially hard examples)
2. Increase epochs
3. Verify label quality
4. Balance class distribution

**If training is slow:**
1. Reduce batch size
2. Use smaller model (YOLOv8n instead of YOLOv8s)
3. Enable GPU acceleration (CUDA)

**If overfitting occurs:**
1. Add more diverse images
2. Enable data augmentation
3. Reduce epochs
4. Use early stopping

---

### Step 3: ONNX Export

**Goal:** Convert PyTorch model to ONNX format for Hailo compilation.

#### 3.1 Export Model (Two Options)

**Option A: Without NMS (Recommended)**
```bash
cd step3_onnx_export
python3 export_onnx_for_hailo.py ../step2_training/runs/detect/train/weights/best.pt
```

Output: `best.onnx` with 6 raw output tensors. Python NMS required in inference.

**Option B: With NMS Embedded**
```bash
python3 export_onnx_for_hailo.py ../step2_training/runs/detect/train/weights/best.pt --nms
```

Output: `best_nms.onnx` with NMS embedded. No Python NMS needed, but less flexible.

#### 3.2 Why Two Options?

| Feature | Without NMS | With NMS |
|---------|-------------|----------|
| Output format | Raw predictions (6 tensors) | Final detections (1 tensor) |
| Python NMS needed | Yes | No |
| Flexibility | High (adjust thresholds at runtime) | Low (fixed at export) |
| HEF size | ~9MB | ~9MB |
| Compilation success | Always works | May fail for some models |

**Recommendation:** Start with `nms=False` (default), use Python NMS in inference script.

#### 3.3 Verify Export

```bash
python3 verify_onnx_export.py best.onnx
```

This runs inference on a black image and checks for false positives.

---

### Step 4: HEF Compilation

**Goal:** Compile ONNX to Hailo Executable Format (HEF).

#### 4.1 Install Hailo Dataflow Compiler

Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/):
1. Get Hailo Dataflow Compiler SDK 3.33.0+
2. Extract to `step4_hef_compilation/.venv_hailo_full/`
3. Verify installation:
```bash
source step4_hef_compilation/.venv_hailo_full/bin/activate
python -c "import hailo_sdk_client; print('OK')"
deactivate
```

#### 4.2 Prepare Calibration Data

Calibration is used for INT8 quantization:
```bash
cd step4_hef_compilation
python3 prepare_calibration.py ../datasets --num-samples 64
```

This creates `calibration_data/` with 64 preprocessed images (UINT8, 640x640).

**Important:** Images must be:
- Same preprocessing as training (resize, letterbox)
- UINT8 format [0-255], NOT float32
- Representative of inference data

#### 4.3 Compile to HEF

```bash
python3 compile_to_hef.py ../step3_onnx_export/best.onnx --output my_model
```

**Compilation process:**
1. Parse ONNX (2-3 seconds)
2. Optimize graph (10-15 seconds)
3. Quantize to INT8 using calibration data (60-90 seconds)
4. Compile for Hailo-8 architecture (30-60 seconds)
5. Generate HEF binary

**Output:** `models/my_model.hef` (~9MB)

#### 4.4 Understanding hailo_optimization.mscript

```python
# Enable loose mode - suppresses shape warnings
enable_loose_mode

# Optimization level (0=fast, 2=best accuracy)
optimization_level(0)

# Batch size (1 for real-time inference)
batch_size(1)
```

**When to change:**
- Set `optimization_level(2)` for better accuracy (slower compile)
- Disable `enable_loose_mode` if you see runtime errors
- Increase `batch_size` for batch inference (not common on Pi)

---

### Step 5: Raspberry Pi Deployment

**Goal:** Deploy HEF model to Raspberry Pi 5 and test inference.

#### 5.1 Install HailoRT on Raspberry Pi

On the Raspberry Pi:
```bash
# Install HailoRT runtime (not the SDK)
sudo apt update
sudo apt install python3-hailort

# Verify installation
python3 -c "import hailo_platform; print('HailoRT OK')"
```

#### 5.2 Automated Deployment

From your development machine:
```bash
cd step5_raspberry_pi_testing
./deploy_to_pi.sh
```

The script will:
1. Prompt for Pi IP address and credentials
2. Copy HEF file and inference scripts
3. Install dependencies
4. Verify HailoRT installation
5. Provide SSH command for testing

**Manual password entry required** for SSH/SCP operations.

#### 5.3 Run Inference on Pi

SSH to the Pi and run:
```bash
cd ~/MODEL-GEN/scripts
python3 hailo_detect_live.py --model ../models/my_model.hef --headless
```

**Inference modes:**
- `--headless`: No display, save detections to files (recommended for SSH)
- `--picamera`: Use Raspberry Pi Camera Module
- `--camera 0`: Use USB webcam at /dev/video0
- `--conf 0.35`: Confidence threshold
- `--iou 0.5`: NMS IoU threshold

#### 5.4 Understanding Inference Pipeline

```python
# 1. Preprocess
image = cv2.resize(frame, (640, 640))  # Keep UINT8
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Hailo inference
outputs = hailo_device.infer(rgb)  # 6 output tensors

# 3. Decode outputs (DFL format)
boxes, scores, class_ids = decode_yolov8_outputs(outputs)

# 4. Apply Python NMS
boxes, scores, class_ids = nms(boxes, scores, class_ids)

# 5. Visualize or save
draw_boxes(frame, boxes, scores, class_ids)
```

---

## Advanced Topics

### Dual NMS Workflow

The pipeline supports both ONNX-embedded NMS and Python NMS:

#### When ONNX has NMS (nms=True)
```
ONNX model → Hailo-8 → Final detections (1, 300, 6)
                         ↓
                   Direct use (no Python NMS)
```

#### When ONNX has no NMS (nms=False)
```
ONNX model → Hailo-8 → Raw predictions (6 tensors)
                         ↓
                   Python NMS → Final detections
```

**Python NMS Script:**
`step5_raspberry_pi_testing/nms_postprocess.py` can be used standalone:
```bash
python3 nms_postprocess.py --input raw_output.npy --output detections.npz
```

### Custom Classes

To add or modify classes:

1. Update `data.yaml`:
```yaml
nc: 3  # Number of classes
names:
  0: drone
  1: IR-Drone
  2: bird  # New class
```

2. Re-annotate images with new class
3. Retrain model
4. Update inference script class names

### Model Optimization

**For better accuracy:**
- Use YOLOv8m or YOLOv8l (larger models)
- Increase training epochs
- Add more training data
- Use test-time augmentation

**For better speed:**
- Use YOLOv8n (nano model)
- Reduce input size to 320x320
- Set `optimization_level(0)` in compilation

**For smaller model size:**
- Use YOLOv8n
- Reduce number of classes
- Prune unnecessary layers

### Batch Processing

For processing multiple images (not real-time):

```python
# Load HEF with batch_size > 1
network_group_params.batch_size = 4

# Prepare batch
batch_images = np.stack([img1, img2, img3, img4])

# Infer
outputs = pipeline.infer({input_name: batch_images})
```

### Multi-Camera Setup

Run multiple inference instances:
```bash
# Terminal 1
python3 hailo_detect_live.py --model model.hef --camera 0 --headless

# Terminal 2
python3 hailo_detect_live.py --model model.hef --camera 1 --headless
```

Note: Hailo-8 can handle multiple streams, but CPU/memory may be bottleneck.

---

## Troubleshooting

### Common Issues

#### Issue: "No HEF file found in models/"

**Cause:** HEF wasn't compiled or is in wrong directory.

**Solution:**
```bash
# Check if HEF exists
ls -lh step4_hef_compilation/*.hef
ls -lh models/*.hef

# Move if needed
mv step4_hef_compilation/*.hef models/
```

#### Issue: "Cannot connect to Raspberry Pi"

**Cause:** SSH connection failed.

**Solution:**
1. Verify Pi is powered on: `ping <pi_ip>`
2. Check SSH is enabled: `sudo systemctl status ssh`
3. Test manual SSH: `ssh user@<pi_ip>`
4. Check firewall: `sudo ufw status`

#### Issue: "sshpass not installed"

**Cause:** Deployment script needs sshpass for password handling.

**Solution:**
```bash
sudo apt-get install sshpass
```

#### Issue: "ONNX export has false positives"

**Cause:** Model exported without NMS or with wrong settings.

**Solution:**
```bash
# Re-export with --nms flag
python3 export_onnx_for_hailo.py best.pt --nms

# Or use default (no NMS) and apply Python NMS
python3 export_onnx_for_hailo.py best.pt
```

#### Issue: "Hailo SDK not found"

**Cause:** `.venv_hailo_full` not installed.

**Solution:**
1. Download Hailo Dataflow Compiler from hailo.ai
2. Extract to `step4_hef_compilation/.venv_hailo_full/`
3. Verify: `source .venv_hailo_full/bin/activate && python -c "import hailo_sdk_client"`

#### Issue: "Calibration error: Invalid image format"

**Cause:** Calibration images are float32 instead of UINT8.

**Solution:**
```bash
# Regenerate calibration data
rm -rf step4_hef_compilation/calibration_data
python3 step4_hef_compilation/prepare_calibration.py datasets --num-samples 64
```

#### Issue: "Low FPS on Raspberry Pi"

**Causes and solutions:**
1. **CPU bottleneck**: Optimize preprocessing, use smaller image size
2. **USB camera latency**: Use Picamera2 instead
3. **Display overhead**: Use `--headless` mode
4. **Memory swap**: Add more RAM or reduce batch size

#### Issue: "Detections are wrong/missing"

**Debug steps:**
1. Verify model accuracy in training metrics
2. Test ONNX model with verify script
3. Check preprocessing matches training
4. Adjust confidence threshold: `--conf 0.2`
5. Verify NMS is applied (for non-NMS models)

---

## API Reference

### Training API

```python
from step2_training.train_yolov8 import train_yolov8

# Train with custom parameters
train_yolov8(
    data_yaml='datasets/data.yaml',
    epochs=200,
    batch=32,
    imgsz=640,
    patience=15,
    name='my_training_run'
)
```

### Export API

```python
from step3_onnx_export.export_onnx_for_hailo import export_onnx_for_hailo

# Export without NMS
export_onnx_for_hailo(
    pt_path='best.pt',
    output_dir='exports',
    imgsz=640,
    nms=False  # Python NMS required
)

# Export with NMS
export_onnx_for_hailo(
    pt_path='best.pt',
    output_dir='exports',
    imgsz=640,
    nms=True  # No Python NMS needed
)
```

### Compilation API

```python
from step4_hef_compilation.compile_to_hef import compile_onnx_to_hef

# Compile ONNX to HEF
hef_path = compile_onnx_to_hef(
    onnx_path='best.onnx',
    output_name='my_model',
    calib_dir='calibration_data',
    hailo_venv='.venv_hailo_full'
)
```

### Inference API

```python
from step5_raspberry_pi_testing.hailo_detect_live import HailoDetector

# Initialize detector
detector = HailoDetector(
    hef_path='model.hef',
    conf_thresh=0.25,
    iou_thresh=0.45
)

# Run detection
boxes, scores, class_ids = detector.detect(frame)

# Cleanup
detector.cleanup()
```

---

## Best Practices

### Dataset Creation

1. **Quality over quantity**: 100 good annotations > 1000 poor ones
2. **Balance classes**: Equal samples per class when possible
3. **Include edge cases**: Occluded objects, different scales, lighting
4. **Negative samples**: Images without objects prevent false positives
5. **Consistent annotation**: Same person annotates all, or use guidelines

### Training

1. **Start small**: Use YOLOv8n for quick iterations
2. **Monitor validation**: Watch for overfitting
3. **Early stopping**: Let patience=15 prevent wasted epochs
4. **Save checkpoints**: Keep multiple model versions
5. **Document experiments**: Track hyperparameters and results

### Model Export

1. **Verify ONNX**: Always run verification script
2. **Test inference**: Try ONNX model before HEF compilation
3. **Document NMS choice**: Note whether model uses embedded or Python NMS
4. **Keep ONNX files**: Don't delete after HEF compilation (for debugging)

### Deployment

1. **Test locally first**: Run inference on development machine
2. **Profile performance**: Measure FPS, latency, memory usage
3. **Log errors**: Capture stdout/stderr for debugging
4. **Version models**: Name HEF files with version/date
5. **Backup configurations**: Save model configs and scripts

### Security

1. **Use SSH keys**: Set up key-based authentication for Pi
2. **Update regularly**: Keep Raspberry Pi OS and packages updated
3. **Restrict access**: Firewall rules for Pi network access
4. **Secure models**: Don't expose HEF files publicly (proprietary)

---

## FAQ

**Q: Can I use this with other YOLO versions (YOLOv5, YOLOv10)?**  
A: Yes, but you'll need to modify export scripts. YOLOv8 is recommended for best Hailo compatibility.

**Q: Does this work on Raspberry Pi 4?**  
A: No, Hailo-8 module requires Raspberry Pi 5 M.2 slot.

**Q: Can I train on Windows?**  
A: Training works on Windows, but HEF compilation requires Linux (x86_64).

**Q: How many images do I need for training?**  
A: Minimum 50 per class, recommended 100+, ideal 500+.

**Q: What's the difference between Hailo SDK and HailoRT?**  
A: SDK is for development (model compilation), HailoRT is runtime library (inference only).

**Q: Can I use custom backbones (ResNet, EfficientNet)?**  
A: Yes, but you'll need to export them to ONNX and ensure Hailo compatibility.

**Q: How do I improve model accuracy?**  
A: More data, longer training, better annotations, or larger model (YOLOv8m/l).

**Q: Can I deploy to other edge devices (Jetson, Coral)?**  
A: Not with HEF format. You'll need to export to TensorRT (Jetson) or TFLite (Coral).

**Q: Is GPU required for training?**  
A: No, but highly recommended. GPU training is 10-50x faster than CPU.

**Q: How do I update to a new model version?**  
A: Retrain with new data, export ONNX, compile new HEF, deploy to Pi.

---

## Additional Resources

### Official Documentation
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [ONNX Documentation](https://onnx.ai/onnx/)

### Community Resources
- [Hailo Community Forum](https://community.hailo.ai/)
- [Raspberry Pi Forums](https://forums.raspberrypi.com/)
- [Ultralytics Discord](https://discord.com/invite/ultralytics)

### Related Projects
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
- [Ultralytics Hub](https://hub.ultralytics.com/)
- [ONNX Model Zoo](https://github.com/onnx/models)

---

## Contributing

Contributions are welcome! Please:
1. Test changes thoroughly
2. Update documentation
3. Follow existing code style
4. Add examples for new features

---

## License

This project follows the parent project license terms.

---

**Last Updated:** December 11, 2025  
**Version:** 1.0.0  
**Maintainer:** arsatyants

# STEP 4: Compile to HEF for Hailo-8

Compile your ONNX model to HEF (Hailo Executable Format) for deployment on Hailo-8 AI accelerator.

## Overview

This step converts the ONNX model from STEP 3 into a hardware-optimized binary that runs on the Hailo-8 neural processing unit. The process includes:

1. **Parse**: Load ONNX and convert to Hailo internal representation
2. **Optimize**: Apply graph optimizations for Hailo-8 architecture
3. **Quantize**: Convert FP32 to INT8 using calibration dataset
4. **Compile**: Generate HEF binary for runtime deployment

**Expected time**: 5-15 minutes depending on model size

## Prerequisites

### Hailo Dataflow Compiler SDK

**Critical**: This step requires the Hailo Dataflow Compiler SDK, which is separate from HailoRT (the runtime library).

- **SDK Version**: 3.33.0 or later recommended
- **Installation**: Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/)
- **Location**: Extract to `.venv_hailo_full/` (in this directory)
- **License**: Requires Hailo developer account (free registration)

**Check installation**:
```bash
ls .venv_hailo_full/bin/activate
# Should exist
```

### Calibration Dataset

Required for INT8 quantization. Use 50-100 representative images from your training set.

**Format requirements**:
- **Type**: UINT8 [0-255] (NOT normalized float32)
- **Size**: 640x640 pixels
- **Color**: RGB order
- **Format**: JPG or PNG

## Quick Start

### 1. Prepare Calibration Data

```bash
# Create calibration dataset from training images
python3 prepare_calibration.py ../../datasets/yolo_dataset/

# With custom number of samples
python3 prepare_calibration.py ../../datasets/yolo_dataset/ --num-samples 100
```

**Expected output:**
```
==============================================================
Prepare Calibration Dataset for Hailo
==============================================================
Source: ../../datasets/yolo_dataset/
Output: ./calibration_data
Samples: 64
Image size: 640x640
==============================================================

📊 Found 85 images
📸 Processing 64 calibration images...
   Processed 10/64
   Processed 20/64
   ...
   Processed 64/64

✅ Prepared 64 calibration images
   Output directory: ./calibration_data
   Format: UINT8 RGB [0-255]
   Size: 640x640x3

🔍 Verification:
   Sample image shape: (640, 640, 3)
   Data type: uint8
   Value range: [0, 255]

==============================================================
✅ CALIBRATION DATASET READY
==============================================================
```

### 2. Compile ONNX to HEF

```bash
# Basic compilation (uses ./calibration_data by default)
python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx

# With custom output name
python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx --output drone_detector

# With custom calibration data
python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx \
    --output drone_detector \
    --calib ./my_calibration_data/
```

**Expected output:**
```
==============================================================
Compile ONNX to HEF for Hailo-8
==============================================================
✅ Hailo SDK found
✅ ONNX file: ../step3_onnx_export/best_nms.onnx (43.2 MB)
📊 Calibration dataset: 64 images from ./calibration_data
✅ Calibration data ready

==============================================================
🔧 Running Hailo Dataflow Compiler...
   (This may take 5-15 minutes)

==============================================================
Hailo Dataflow Compiler
==============================================================
ONNX: /path/to/best_nms.onnx
Calibration: /path/to/calibration_data
Output: /path/to/model.hef
==============================================================

[1/4] Parsing ONNX model...
   ✅ ONNX parsed

[2/4] Optimizing model...
Hailo optimization configuration applied:
  - enable_loose_mode: True
  - optimization_level: 0
   ✅ Applied optimization script: hailo_optimization.mscript

[3/4] Quantizing model (calibration)...
   Using 64 calibration images
   ✅ Quantization complete

[4/4] Compiling to HEF...
==============================================================
✅ COMPILATION SUCCESSFUL
==============================================================
HEF file: /path/to/model.hef
Size: 9.3 MB
==============================================================

==============================================================
🎉 HEF COMPILATION COMPLETE
==============================================================

📦 Output file: /path/to/model.hef
   Size: 9.3 MB

💡 Next steps:
   1. Copy HEF to Raspberry Pi:
      scp model.hef pi@192.168.3.205:~/models/
   2. Run inference:
      python3 hailo_inference.py --model ~/models/model.hef
```

## Compilation Process Details

### Stage 1: ONNX Parsing

The compiler loads the ONNX model and identifies network structure:

- **Input layer**: `images` (1, 3, 640, 640)
- **Output layer**: `output0` (1, 300, 6) - post-NMS detections
- **Network type**: YOLOv8 with embedded NMS
- **Parameters**: ~11M for YOLOv8s

**Critical**: The ONNX must have NMS embedded (from STEP 3 with `nms=True`). Without this, compilation will fail or produce incorrect results.

### Stage 2: Graph Optimization

Applies Hailo-specific optimizations:

- **enable_loose_mode**: Required for YOLOv8 with NMS (suppresses strict shape checking)
- **optimization_level=0**: Fastest compilation, good accuracy balance
- **Graph fusion**: Combines operations for efficiency
- **Dead code elimination**: Removes unused branches

**Warning**: You may see `enable_loose_mode is deprecated` - this is safe to ignore and the setting is still required for YOLOv8.

### Stage 3: Quantization

Converts FP32 model to INT8 using calibration data:

- **Method**: Post-Training Quantization (PTQ)
- **Precision**: 8-bit integer weights and activations
- **Calibration**: Uses 64 images to determine optimal quantization ranges
- **Output**: Quantized Hailo model with <2% accuracy loss

**Format Critical**: Calibration images must be UINT8 [0-255], matching the runtime input format. Using normalized float32 [0-1] will cause inference failures.

### Stage 4: HEF Generation

Final binary compilation for Hailo-8 neural core:

- **Target**: Hailo-8 AI processor (26 TOPS)
- **Memory layout**: Optimized for on-chip SRAM access
- **Instruction scheduling**: Parallel execution on tensor processors
- **Output format**: Raw YOLO tensors (NMS layer is stripped)

**Important**: Hailo strips the NMS layer during compilation. The HEF outputs 6 raw YOLO tensors that require CPU-side NMS post-processing during inference.

## Output Files

### HEF Binary (`model.hef`)

- **Size**: 4-10 MB (compressed from 40+ MB ONNX)
- **Format**: Hailo Executable Format (binary)
- **Contains**: 
  - Quantized INT8 weights
  - Hailo-8 instructions
  - Network metadata
  - Input/output specifications

### Model Information

After compilation, the HEF contains:

```python
# Runtime information (example)
Input:
  - Name: images
  - Shape: (640, 640, 3) HWC
  - Type: UINT8
  - Range: [0, 255]
  - Color: RGB

Outputs (6 raw YOLO tensors):
  - output1: (80, 80, 64)
  - output2: (40, 40, 64)
  - output3: (20, 20, 64)
  - output1_cls: (80, 80, 2)
  - output2_cls: (40, 40, 2)
  - output3_cls: (20, 20, 2)
```

**Note**: Output format is raw YOLO, not the (1, 300, 6) post-NMS format from ONNX. You must apply NMS in your inference code.

## Calibration Dataset Best Practices

### Number of Images

- **Minimum**: 32 images
- **Recommended**: 64-100 images
- **Maximum**: 200+ images (diminishing returns)

More calibration images = better quantization accuracy, but longer compilation time.

### Image Selection

Choose diverse samples that represent real-world inference conditions:

```python
# Good calibration set:
✅ Different lighting conditions (day/night)
✅ Various drone sizes and distances
✅ Different backgrounds (sky, buildings, trees)
✅ Edge cases (partial occlusion, motion blur)

# Poor calibration set:
❌ All similar images (same scene)
❌ Only perfect conditions (bright daylight)
❌ Missing edge cases
```

### Preprocessing

**Critical**: Calibration images must use identical preprocessing as inference:

```python
# Calibration preparation:
image = cv2.imread('image.jpg')
image = cv2.resize(image, (640, 640))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Save as UINT8 [0-255] - NO NORMALIZATION

# Runtime inference:
frame = picamera.capture_array()  # Returns RGB uint8
frame_resized = cv2.resize(frame, (640, 640))
# Input directly as UINT8 [0-255] - NO NORMALIZATION
```

**Common mistake**: Normalizing calibration data to [0-1] but using [0-255] at inference (or vice versa). This causes poor accuracy.

## Performance Expectations

### Compilation Time

| Model | ONNX Size | Compilation Time | HEF Size |
|-------|-----------|------------------|----------|
| YOLOv8n | 12 MB | 3-5 min | 2.5 MB |
| YOLOv8s | 44 MB | 5-10 min | 4.5 MB |
| YOLOv8m | 100 MB | 10-20 min | 9.0 MB |

**Factors affecting time:**
- Model size (parameters)
- Optimization level (0 is fastest)
- Number of calibration images
- CPU performance

### Inference Performance (Hailo-8)

After compilation, expect these speeds on Raspberry Pi 5:

| Model | HEF Size | FPS | Latency | Power |
|-------|----------|-----|---------|-------|
| YOLOv8n | 2.5 MB | 60-80 | 12-16ms | 2W |
| YOLOv8s | 4.5 MB | 40-50 | 20-25ms | 2.5W |
| YOLOv8m | 9.0 MB | 25-35 | 28-40ms | 3W |

**Note**: Times include Hailo inference only, not CPU post-processing (NMS, drawing).

### Accuracy After Quantization

Typical mAP50 retention:

- **YOLOv8n**: 95-98% of FP32 accuracy
- **YOLOv8s**: 96-99% of FP32 accuracy
- **YOLOv8m**: 97-99% of FP32 accuracy

Example: If training achieved mAP50=0.652, expect mAP50=0.64-0.65 after quantization.

## Deployment to Raspberry Pi

### 1. Copy HEF File

```bash
# SCP transfer
scp model.hef pi@192.168.3.205:~/models/

# Or with rsync
rsync -avz model.hef pi@192.168.3.205:~/models/
```

### 2. Install HailoRT on Pi

```bash
# On Raspberry Pi 5
sudo apt update
sudo apt install python3-hailort

# Verify installation
python3 -c "from hailo_platform import HEF; print('HailoRT OK')"
```

### 3. Run Inference

```python
#!/usr/bin/env python3
"""Basic Hailo inference example"""

import cv2
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams

# Load HEF
hef = HEF('model.hef')
device = VDevice()

# Configure
params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = device.configure(hef, params)[0]
network_group_params = network_group.create_params()

# Input VStream
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_infos = hef.get_output_vstream_infos()

# Inference loop
with network_group.activate(network_group_params):
    with network_group.create_input_vstream(input_vstream_info) as input_vstream:
        with network_group.create_output_vstreams(output_vstream_infos) as output_vstreams:
            
            # Capture frame
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            
            # Preprocess
            frame_resized = cv2.resize(frame, (640, 640))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Run inference (UINT8 input)
            input_vstream.send(frame_rgb)
            
            # Get outputs
            outputs = [stream.recv() for stream in output_vstreams]
            
            # Post-process (apply NMS on CPU)
            # ... NMS implementation here ...
```

See `working_hailo_detector.py` in project root for complete example.

## Troubleshooting

### Compilation fails: "Hailo SDK not found"

**Problem**: `.venv_hailo_full` directory not found

**Solution**: Install Hailo Dataflow Compiler SDK:
1. Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/)
2. Extract to `.venv_hailo_full/` (within step4_hef_compilation/)
3. Verify: `ls .venv_hailo_full/bin/activate`

### Compilation fails: "No module named 'hailo_sdk_client'"

**Problem**: Hailo SDK not properly installed

**Solution**: Reinstall SDK or check Python path:
```bash
source .venv_hailo_full/bin/activate
python3 -c "import hailo_sdk_client; print('OK')"
```

### Compilation fails: "Input shape mismatch"

**Problem**: ONNX input shape doesn't match expected (1, 3, 640, 640)

**Solution**: Re-export ONNX with correct `imgsz` parameter:
```bash
cd ../step3_onnx_export
python3 export_onnx_with_nms.py <model.pt> --imgsz 640
```

### Compilation succeeds but inference fails: "Shape mismatch"

**Problem**: Calibration data format doesn't match inference format

**Root cause**: Calibration used float32 [0-1], inference uses uint8 [0-255]

**Solution**: Recreate calibration data with `prepare_calibration.py` (ensures UINT8):
```bash
rm -rf calibration_data/
python3 prepare_calibration.py ../../datasets/yolo_dataset/
python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx
```

### Low accuracy after quantization (>10% mAP drop)

**Problem**: Poor calibration dataset

**Solutions**:
1. **Increase calibration samples**:
   ```bash
   python3 prepare_calibration.py <source> --num-samples 100
   ```

2. **Use more diverse images** (different lighting, distances, backgrounds)

3. **Try optimization_level=1** (edit `hailo_optimization.mscript`):
   ```python
   optimization_level(1)  # Slower compilation, better accuracy
   ```

### Compilation takes >30 minutes

**Problem**: Large model or high optimization level

**Solutions**:
1. **Check optimization level** in `hailo_optimization.mscript` (should be 0)
2. **Reduce calibration samples**:
   ```bash
   python3 compile_to_hef.py <onnx> --num-samples 32
   ```
3. **Use smaller model** (YOLOv8n instead of YOLOv8s)

### HEF file not created after compilation

**Problem**: Compilation script failed silently

**Check logs**: Look for Python traceback in terminal output

**Common causes**:
- Out of memory (need 8GB+ RAM)
- Disk space (need 5GB+ free)
- ONNX file corrupted (re-export from STEP 3)

### Warning: "enable_loose_mode is deprecated"

**Status**: ⚠️ Safe to ignore

**Explanation**: The parameter is deprecated but still required for YOLOv8 with NMS. Hailo team recommends keeping it enabled despite the warning.

**No action needed**: Compilation will succeed with this warning.

### Runtime error: "NMS not found in outputs"

**Expected behavior**: Hailo strips NMS layer during compilation

**Explanation**: The HEF outputs raw YOLO tensors, not post-NMS detections. You must implement CPU-side NMS in your inference code.

**Solution**: See `working_hailo_detector.py` for NMS implementation with OpenCV.

## Advanced Configuration

### Custom Optimization Script

Edit `hailo_optimization.mscript` for advanced tuning:

```python
# hailo_optimization.mscript

# Disable loose mode (stricter checking)
# enable_loose_mode  # Comment out to disable

# Higher optimization level (slower, better accuracy)
optimization_level(1)  # 0=fast, 1=balanced, 2=best

# Custom quantization parameters
quantization_param(calibration_iterations=3)

# Resource allocation
allocate_clusters(num_clusters=8)
```

### Multi-batch Compilation

For processing multiple frames in parallel:

```python
# In compilation script (_compile_temp.py), change:
net_input_shapes={'images': [4, 3, 640, 640]}  # Batch size 4

# Note: Hailo-8 has limited memory, batch > 8 may fail
```

### Mixed Precision

Keep certain layers in FP16 for better accuracy:

```python
# In optimization script
mixed_precision(
    layers=['model.22.cv2.0', 'model.22.cv3.0'],
    precision='float16'
)
```

## Validation

After compilation, validate HEF quality:

### 1. Check HEF metadata

```bash
# Install hailortcli
sudo apt install hailort-tools

# Inspect HEF
hailortcli parse-hef model.hef
```

**Expected output:**
```
Network group: yolov8s
Input streams:
  - images: 640x640x3 UINT8
Output streams:
  - output1: 80x80x64 UINT8
  - output2: 40x40x64 UINT8
  - output3: 20x20x64 UINT8
  - output1_cls: 80x80x2 UINT8
  - output2_cls: 40x40x2 UINT8
  - output3_cls: 20x20x2 UINT8
```

### 2. Test on sample image

Run inference on validation images and compare with PyTorch results:

```python
# Compare PT vs HEF
# Expected: >95% IoU overlap for detected boxes
```

### 3. Measure performance

```bash
# Benchmark on Raspberry Pi 5
hailortcli benchmark model.hef
```

**Expected**: 40-50 FPS for YOLOv8s

## Summary Checklist

- [ ] Hailo Dataflow Compiler SDK installed (`.venv_hailo_full`)
- [ ] ONNX model with NMS from STEP 3
- [ ] Calibration dataset prepared (64+ images, UINT8 format)
- [ ] `prepare_calibration.py` executed successfully
- [ ] `compile_to_hef.py` completed without errors
- [ ] HEF file generated (4-10 MB)
- [ ] "enable_loose_mode deprecated" warning (safe to ignore)
- [ ] HEF file copied to Raspberry Pi
- [ ] HailoRT installed on Raspberry Pi (`python3-hailort`)
- [ ] Test inference on sample image

**Ready for deployment**: When all checks pass, your model is ready for edge AI inference on Raspberry Pi 5 with Hailo-8 accelerator.

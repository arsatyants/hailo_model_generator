
# STEP 3: ONNX Export for Hailo (with or without NMS)

Export your trained YOLOv8 PyTorch model (`.pt`) to ONNX format for Hailo-8 compilation. **You can now export ONNX with or without embedded Non-Maximum Suppression (NMS)** for maximum flexibility.


## NMS Options: Flexible Export

You can export ONNX **with or without NMS embedded**:

- **WITHOUT NMS (recommended for Hailo-8):**
   - `nms=False` (default)
   - Model outputs raw predictions `(1, 6, 8400)`
   - **Python NMS is performed in the inference script on the Pi**
   - Maximum flexibility, always compiles for Hailo-8

- **WITH NMS (optional/legacy):**
   - Use `--nms` flag
   - Model outputs final detections `(1, 300, 6)` (NMS embedded in ONNX)
   - No Python NMS needed on Pi
   - Less flexible, may not be supported for all YOLOv8 models


## Input Requirements

Before starting this step, you must have:

1. **Trained Model**: `best.pt` file from STEP 2
   - Location: `../step2_training/runs/detect/<training_name>/weights/best.pt`
   - Typical size: 20-25 MB for YOLOv8s
2. **Virtual Environment**: The `.venv` from MODEL-GEN root with ultralytics installed
   ```bash
   source ../.venv/bin/activate
   pip install ultralytics
   ```


## Quick Start

### 1. Export to ONNX (with or without NMS)

```bash
# Activate environment
source ../.venv/bin/activate

# Export ONNX WITHOUT NMS (recommended)
python3 export_onnx_for_hailo.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt

# Export ONNX WITH NMS embedded
python3 export_onnx_for_hailo.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt --nms

# Export (with custom output name)
python3 export_onnx_for_hailo.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt --nms --output drone_detector_nms.onnx
```


**File size check**: ONNX with NMS is ~2x larger than PT file (40-50 MB vs 20-25 MB) because it includes the NMS operations. ONNX without NMS is smaller and outputs raw predictions.


### 2. Verify Export Quality

**Critical test:** Run inference on a pure black image. A correct export (with NMS) should produce **0 detections** (no false positives).

```bash
python3 verify_onnx_export.py best_nms.onnx
```

**Expected output (PASS):**
```
Output shape: (1, 300, 6)
Detections found: 0
✅ VERIFICATION PASSED
```


**If verification fails (detections found on black image):**
```
❌ VERIFICATION FAILED
   Found false positives on black image
   This indicates the ONNX export is incorrect

🔧 Solution:
   Re-export with correct NMS setting:
   python3 export_onnx_for_hailo.py <model.pt> --nms
```

## Output Format Details

### ONNX Input
- **Format**: NCHW (batch, channels, height, width)
- **Shape**: `(1, 3, 640, 640)`
- **Type**: `float32`
- **Range**: `[0.0, 1.0]` (normalized)
- **Color**: RGB order


### ONNX Output (with NMS)
- **Format**: `(batch, max_detections, 6)`
- **Shape**: `(1, 300, 6)`
- **Type**: `float32`
- **Channels**:
   - `[0]`: x1 (left)
   - `[1]`: y1 (top)
   - `[2]`: x2 (right)
   - `[3]`: y2 (bottom)
   - `[4]`: confidence (0-1)
   - `[5]`: class_id (integer)

**Coordinates**: Absolute pixel values (0-640 range for 640x640 image)

### ONNX Output (without NMS)
- **Format**: `(batch, channels, anchors)`
- **Shape**: `(1, 6, 8400)`
- **Note**: Hailo can now compile this format, but you **must use Python NMS in the inference script** on the Pi.

## Export Parameters Explained

### Required Settings


#### `nms` (configurable)
- **Purpose**: Embed Non-Maximum Suppression in ONNX graph
- **Effect**: Converts raw YOLO output to post-processed detections
- **Why**: Hailo compilation works with both, but Python NMS is recommended for flexibility
- **Default**: `False` (must explicitly set to `True` for embedded NMS)

#### `opset=11`
- **Purpose**: ONNX operator set version
- **Why**: Stable version compatible with Hailo Dataflow Compiler
- **Range**: 11-13 work, but 11 is most tested

#### `simplify=True`
- **Purpose**: Optimize ONNX graph (removes unnecessary nodes)
- **Effect**: Smaller file size, faster compilation
- **Requirement**: Requires `onnxsim` package

#### `imgsz=640`
- **Purpose**: Input image size for inference
- **Why**: Must match training size
- **Standard**: 640 is default for YOLOv8

### Optional Settings

#### `half=False`
- **Default**: `False` (FP32)
- **Hailo**: Will quantize to INT8 during compilation anyway
- **Recommendation**: Keep as FP32 for better calibration

#### `dynamic=False`
- **Default**: `False` (fixed batch size)
- **Hailo**: Does not support dynamic shapes
- **Recommendation**: Keep as fixed

## Files Generated

After successful export:

```
step3_onnx_export/
├── best_nms.onnx              # Exported ONNX model (~40-50 MB)
└── best_nms.onnx.json         # (optional) metadata file
```


**Keep the ONNX file** - you'll need it for STEP 4 (HEF compilation). Name it clearly to indicate if NMS is embedded or not.

## Troubleshooting

### Export Fails: "No module named 'ultralytics'"

**Problem**: Ultralytics not installed in current environment

**Solution**:
```bash
source ../.venv/bin/activate
pip install ultralytics
```

### Export Fails: "No module named 'onnxsim'"

**Problem**: ONNX simplifier not installed (needed for `simplify=True`)

**Solution**:
```bash
pip install onnx-simplifier
```

Or disable simplification:
```python
# In export_onnx_with_nms.py, change:
simplify=False
```

### Verification Fails: "No module named 'onnxruntime'"

**Problem**: ONNX Runtime not installed

**Solution**:
```bash
pip install onnxruntime
```

### Verification Finds False Positives (detections > 0)

**Problem**: Model exported without NMS or with incorrect settings


**Root Cause**: Output shape is `(1, 6, 8400)` (no NMS) or `(1, 300, 6)` (with NMS). If you want embedded NMS, re-export with `--nms`.

Verify the export script has:
```python
model.export(
    format='onnx',
    nms=True,        # ← MUST BE True
    opset=11,
    simplify=True,
    imgsz=640
)
```

### ONNX File Too Small (<10 MB)

**Problem**: NMS not embedded (only base model exported)

**Indication**: File size should be ~2x the PT file size

**Solution**: Check export logs for errors, re-run with `--verbose`:
```bash
python3 export_onnx_with_nms.py <model.pt> --verbose
```

### Test Inference Produces Wrong Output Shape

**Expected**: `(1, 300, 6)` - post-processed detections  
**Wrong**: `(1, 6, 8400)` - raw YOLO anchors

**Solution**: Delete ONNX file and re-export:
```bash
rm best_nms.onnx
python3 export_onnx_with_nms.py <model.pt>
```


Verify output shape with:
```bash
python3 -c "import onnxruntime as ort; sess = ort.InferenceSession('best_nms.onnx'); print('Output shape:', sess.get_outputs()[0].shape)"
```

## Performance Expectations

### Export Speed
- **YOLOv8n**: ~10 seconds
- **YOLOv8s**: ~15 seconds
- **YOLOv8m**: ~25 seconds

### File Sizes
| Model | PT Size | ONNX Size (no NMS) | ONNX Size (with NMS) |
|-------|---------|--------------------|-----------------------|
| YOLOv8n | 6 MB | 12 MB | 22 MB |
| YOLOv8s | 22 MB | 44 MB | 88 MB |
| YOLOv8m | 50 MB | 100 MB | 200 MB |

**Note**: ONNX with NMS is ~4x larger than PT because it includes post-processing ops.

### Verification Speed
- **Black image test**: <1 second
- **ONNX Runtime**: Uses CPU inference
- **Memory**: ~500 MB for YOLOv8s


## Next Steps

After successful verification:

1. **Copy ONNX file path** for next step:
   ```bash
   realpath best_nms.onnx
   # Example: /home/user/MODEL-GEN/step3_onnx_export/best_nms.onnx
   ```

2. **Proceed to STEP 4** (HEF Compilation):
   ```bash
   cd ../step4_hef_compilation
   ```

3. **Prepare calibration data** (covered in STEP 4):
   - 50-100 representative images
   - Same preprocessing as training
   - UINT8 format [0-255]

## Advanced Usage

### Test ONNX Inference on Real Image

```python
import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession('best_nms.onnx')

# Load image
image = cv2.imread('test_drone.jpg')
image = cv2.resize(image, (640, 640))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess (normalize to 0-1)
image_float = image_rgb.astype(np.float32) / 255.0
image_nchw = image_float.transpose(2, 0, 1)[np.newaxis, ...]  # HWC → NCHW

# Inference
outputs = session.run(None, {'images': image_nchw})
detections = outputs[0][0]  # (300, 6)

# Filter by confidence
for det in detections:
    if det[4] > 0.5:  # confidence > 50%
        x1, y1, x2, y2, conf, cls = det
        print(f"Class {int(cls)}: {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

### Batch Export Multiple Models

```bash
# Export all trained models
for pt_file in ../step2_training/runs/detect/*/weights/best.pt; do
    echo "Exporting: $pt_file"
    python3 export_onnx_with_nms.py "$pt_file"
done
```

### Check ONNX Graph Structure

```python
import onnx

model = onnx.load('best_nms.onnx')
print("Graph nodes:")
for node in model.graph.node[:10]:  # First 10 nodes
    print(f"  {node.op_type}: {node.name}")

print(f"\nTotal nodes: {len(model.graph.node)}")
print(f"Inputs: {[inp.name for inp in model.graph.input]}")
print(f"Outputs: {[out.name for out in model.graph.output]}")
```


## Summary Checklist

- [ ] Virtual environment activated (`.venv`)
- [ ] `ultralytics` package installed
- [ ] `onnxruntime` package installed (for verification)
- [ ] Trained `.pt` model available from STEP 2
- [ ] Export script run with correct NMS setting (default: no NMS, or use `--nms`)
- [ ] ONNX file size matches expectations (with NMS: ~2x PT file, without NMS: ~same as PT file)
- [ ] Verification test passed (0 detections on black image for NMS models)
- [ ] ONNX file path copied for STEP 4

**Ready for STEP 4**: When verification passes, you can compile to HEF format for Hailo-8 deployment.

---

## Inference and Python NMS

- If you export **without NMS** (`nms=False`), the inference script (`hailo_detect_live.py`) will automatically apply Python NMS to the raw model outputs on the Pi.
- If you export **with NMS** (`--nms`), the inference script will detect this and **skip Python NMS**, using the model's output directly.

See project Copilot instructions for more details and best practices.

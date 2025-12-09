# STEP 3: Export ONNX with NMS

Export your trained YOLOv8 PyTorch model (`.pt`) to ONNX format with embedded Non-Maximum Suppression (NMS).

## Critical Settings

**The most important parameter for Hailo deployment is `nms=True`**. This enables post-processing inside the ONNX model, producing a clean output format that Hailo can properly compile.

### What NMS Does

- **Without NMS** (`nms=False`): Output is raw YOLO format `(1, 6, 8400)` - 8400 anchor predictions
- **With NMS** (`nms=True`): Output is post-processed format `(1, 300, 6)` - up to 300 filtered detections

The second format is what we need for Hailo HEF compilation.

## Input Requirements

Before starting this step, you must have:

1. **Trained Model**: `best.pt` file from STEP 2
   - Location: `../step2_training/runs/detect/<training_name>/weights/best.pt`
   - Typical size: 20-25 MB for YOLOv8s

2. **Virtual Environment**: The `.venv` from project root with ultralytics installed
   ```bash
   source ../../.venv/bin/activate
   pip install ultralytics
   ```

## Quick Start

### 1. Export to ONNX

```bash
# Activate environment
source ../../.venv/bin/activate

# Export (basic)
python3 export_onnx_with_nms.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt

# Export (with custom output name)
python3 export_onnx_with_nms.py ../step2_training/runs/detect/train_drone_ir/weights/best.pt \
    --output drone_detector_nms.onnx
```

**Expected output:**
```
Exporting PyTorch model to ONNX with NMS...
Model: ../step2_training/runs/detect/train_drone_ir/weights/best.pt
Output: best_nms.onnx
ONNX export settings:
   format=onnx
   nms=True (CRITICAL)
   opset=11
   simplify=True
   imgsz=640
Starting export...
✅ Export successful!
Output file: best_nms.onnx (43.2 MB)
```

**File size check**: ONNX with NMS is ~2x larger than PT file (40-50 MB vs 20-25 MB) because it includes the NMS operations.

### 2. Verify Export Quality

**Critical test**: Run inference on a pure black image. A correct export should produce **0 detections** (no false positives).

```bash
python3 verify_onnx_export.py best_nms.onnx
```

**Expected output (PASS):**
```
==============================================================
Verify ONNX Export - Black Image Test
==============================================================
Model: best_nms.onnx
Size: 43.2 MB
==============================================================

📦 Loading ONNX model...
   Input name: images
   Input shape: [1, 3, 640, 640]
   Outputs: 1
      [0] output0: [1, 300, 6]

🖤 Creating pure black test image (640x640)...
🔍 Running inference on black image...

📊 Results:
   Output shape: (1, 300, 6)
   Detections found: 0

==============================================================
✅ VERIFICATION PASSED
==============================================================
   ONNX model correctly produces 0 detections on black image
   Export with nms=True is working correctly

💡 Next step: Compile to HEF
   cd ../step4_hef_compilation
   ./compile_to_hef.sh /absolute/path/to/best_nms.onnx
```

**If verification fails (detections found on black image):**
```
❌ VERIFICATION FAILED
   Found 4 false positive(s) on black image
   This indicates the ONNX export is incorrect

🔧 Solution:
   Re-export with nms=True:
   python3 export_onnx_with_nms.py <model.pt>
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

### ONNX Output (without NMS - broken)
- **Format**: `(batch, channels, anchors)`
- **Shape**: `(1, 6, 8400)`
- **Problem**: Hailo cannot properly compile this format

## Export Parameters Explained

### Required Settings

#### `nms=True` (CRITICAL)
- **Purpose**: Embed Non-Maximum Suppression in ONNX graph
- **Effect**: Converts raw YOLO output to post-processed detections
- **Why**: Hailo compilation works correctly with this format
- **Default**: `False` (must explicitly set to `True`)

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

**Keep the ONNX file** - you'll need it for STEP 4 (HEF compilation).

## Troubleshooting

### Export Fails: "No module named 'ultralytics'"

**Problem**: Ultralytics not installed in current environment

**Solution**:
```bash
source ../../.venv/bin/activate
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

**Root Cause**: Output shape is `(1, 6, 8400)` instead of `(1, 300, 6)`

**Solution**: Re-export with correct parameters:
```bash
python3 export_onnx_with_nms.py <model.pt>
```

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
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('best_nms.onnx')
print('Output shape:', sess.get_outputs()[0].shape)
"
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
- [ ] Export script run with `nms=True`
- [ ] ONNX file size ~2x larger than PT file
- [ ] Verification test passed (0 detections on black image)
- [ ] ONNX file path copied for STEP 4

**Ready for STEP 4**: When verification passes, you can compile to HEF format for Hailo-8 deployment.

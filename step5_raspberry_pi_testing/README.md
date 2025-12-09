# STEP 5: Raspberry Pi Testing with Hailo-8

Deploy and test your compiled HEF model on Raspberry Pi 5 with Hailo-8 AI accelerator.

## Overview

This final step deploys your trained and compiled model to the Raspberry Pi 5 for real-time inference testing. The Hailo-8 AI accelerator provides hardware-accelerated inference at 40-50 FPS for YOLOv8s models.

**What you'll do:**
1. Deploy HEF model to Raspberry Pi
2. Install HailoRT runtime
3. Run real-time detection with camera
4. Collect performance metrics
5. Save test results

## Prerequisites

### Hardware Requirements

**Raspberry Pi 5 Setup:**
- **Board**: Raspberry Pi 5 (4GB or 8GB RAM)
- **OS**: Raspberry Pi OS (Debian 12 Bookworm) 64-bit
- **Hailo**: Hailo-8 AI accelerator (M.2 HAT+ or PCIe module)
- **Camera**: Raspberry Pi Camera Module 3 OR USB webcam
- **Power**: 27W USB-C power supply (Hailo requires extra power)
- **Network**: Ethernet or WiFi connection
- **Storage**: 16GB+ microSD card (32GB recommended)

**Development Machine:**
- SSH client (built-in on Linux/Mac, PuTTY on Windows)
- Network access to Raspberry Pi
- HEF file from STEP 4

### Software Requirements (Raspberry Pi)

**Pre-installed on Raspberry Pi OS:**
- Python 3.11+
- Picamera2 (for Raspberry Pi Camera)
- OpenCV dependencies

**To be installed:**
- HailoRT (`python3-hailort` package)
- OpenCV Python bindings
- NumPy

## Quick Start

### Option 1: Automated Deployment

```bash
# From your development machine
cd step5_raspberry_pi_testing

# Deploy everything automatically
./deploy_to_pi.sh <pi_ip_address> <username>

# Example:
./deploy_to_pi.sh 192.168.3.205 pi
```

**What it does:**
1. Tests SSH connection to Pi
2. Creates directory structure on Pi
3. Copies HEF model from `step4_hef_compilation/`
4. Copies inference script
5. Installs HailoRT and dependencies
6. Sets up execution permissions

**Expected output:**
```
==================================================================
Deploy and Test Hailo Model on Raspberry Pi 5
==================================================================

Target: pi@192.168.3.205

✅ Found HEF file: drone_detector.hef

🔌 Testing connection to Raspberry Pi...
✅ SSH connection OK

📁 Creating directories on Pi...
✅ Directories created

📦 Copying HEF model to Pi...
✅ HEF file copied

📄 Copying inference script to Pi...
✅ Script copied

🔍 Checking Hailo installation on Pi...
✅ HailoRT is installed

📦 Installing Python dependencies on Pi...

==================================================================
🎉 DEPLOYMENT COMPLETE
==================================================================

Files on Raspberry Pi:
  Model:  ~/MODEL-GEN/models/drone_detector.hef
  Script: ~/MODEL-GEN/scripts/hailo_detect_live.py
  Output: ~/MODEL-GEN/test_results/
```

### Option 2: Manual Deployment

If automated script fails, deploy manually:

```bash
# 1. Copy HEF file
scp ../step4_hef_compilation/model.hef pi@192.168.3.205:~/models/

# 2. Copy inference script
scp hailo_detect_live.py pi@192.168.3.205:~/scripts/

# 3. SSH to Pi
ssh pi@192.168.3.205

# 4. Install HailoRT
sudo apt update
sudo apt install python3-hailort

# 5. Install Python packages
pip3 install opencv-python numpy --break-system-packages
```

## Running Inference on Raspberry Pi

### 1. SSH to Raspberry Pi

```bash
ssh pi@192.168.3.205
cd ~/MODEL-GEN/scripts
```

### 2. Basic Inference (USB Camera)

```bash
# Run with default settings
python3 hailo_detect_live.py --model ../models/model.hef

# Expected output:
# ==============================================================
# 🚀 HAILO Real-time Detection - Raspberry Pi 5
# ==============================================================
# 📦 Loading ../models/model.hef...
# ✅ Model loaded: images
# 📹 Opening USB camera 0...
# ✅ USB camera ready
# 
# 💾 Saving results to: ../test_results
# 
# Controls:
#   'q' - Quit
#   's' - Save current frame
#   'r' - Toggle recording mode (auto-save all frames)
# ==============================================================
```

### 3. Raspberry Pi Camera Module

```bash
# Use Pi Camera instead of USB
python3 hailo_detect_live.py --model ../models/model.hef --picamera
```

**Note**: Raspberry Pi Camera Module 3 provides better quality and lower latency than USB cameras.

### 4. Headless Mode (No Display)

For remote operation without monitor:

```bash
# Run headless - saves frames with detections only
python3 hailo_detect_live.py --model ../models/model.hef --headless

# Check saved frames
ls ../test_results/
```

### 5. Adjust Detection Thresholds

```bash
# Lower confidence threshold (more detections, more false positives)
python3 hailo_detect_live.py --model ../models/model.hef --conf 0.15 --iou 0.45

# Higher confidence threshold (fewer false positives, may miss objects)
python3 hailo_detect_live.py --model ../models/model.hef --conf 0.40 --iou 0.50
```

**Threshold guidelines:**
- **conf 0.15-0.20**: Maximum recall (find everything)
- **conf 0.25-0.30**: Balanced (default)
- **conf 0.35-0.50**: High precision (fewer false positives)

### 6. Custom Camera Device

```bash
# Use different USB camera
python3 hailo_detect_live.py --model ../models/model.hef --camera 1

# List available cameras on Pi:
v4l2-ctl --list-devices
```

## Command-Line Options

```bash
python3 hailo_detect_live.py [OPTIONS]

Options:
  --model PATH         Path to HEF model (default: ../models/model.hef)
  --conf FLOAT         Confidence threshold 0.0-1.0 (default: 0.25)
  --iou FLOAT          NMS IOU threshold 0.0-1.0 (default: 0.45)
  --camera INT         USB camera device ID (default: 0)
  --picamera           Use Raspberry Pi Camera Module
  --save-dir PATH      Directory for saved frames (default: ../test_results)
  --headless           Run without display (save frames only)
  -h, --help           Show help message
```

## Interactive Controls

While inference is running:

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame to file |
| `r` | Toggle recording mode (auto-save all frames with detections) |

**Recording mode**: When enabled (press `r`), automatically saves every frame that contains detections to `test_results/` directory.

## Performance Metrics

### Expected Performance (Raspberry Pi 5 + Hailo-8)

| Model | HEF Size | FPS | Latency | CPU Usage | Power |
|-------|----------|-----|---------|-----------|-------|
| YOLOv8n | 2.5 MB | 60-80 | 12-16ms | <5% | 2.0W |
| YOLOv8s | 4.5 MB | 40-50 | 20-25ms | <10% | 2.5W |
| YOLOv8m | 9.0 MB | 25-35 | 28-40ms | <15% | 3.0W |

**Breakdown:**
- **Hailo inference**: 8-15ms (neural core)
- **Preprocessing**: 2-5ms (CPU)
- **NMS post-processing**: 1-3ms (CPU)
- **Rendering/display**: 5-10ms (CPU)

### Real-time Performance Example

```
Frame 30: Found 2 detections | FPS: 42.3
  #1: drone score=0.874 size=145.2x98.7
  #2: IR-Drone score=0.652 size=67.3x45.1

Frame 60: Found 1 detections | FPS: 43.1
  #1: drone score=0.923 size=187.5x132.4

...

==============================================================
📊 Final Statistics
==============================================================
   Frames processed: 1247
   Total detections: 3582
   Runtime: 29.4s
   Average FPS: 42.4
   Avg detections/frame: 2.87
   Results saved to: ../test_results
==============================================================
```

## Output Files

### Directory Structure

After running tests:

```
MODEL-GEN/
├── models/
│   └── model.hef                  # Deployed HEF model
├── scripts/
│   └── hailo_detect_live.py       # Inference script
└── test_results/
    ├── manual_1733759821.jpg      # Manually saved frames (press 's')
    ├── detect_000042.jpg          # Auto-saved detections (recording mode)
    ├── detect_000087.jpg
    └── ...
```

### Frame Naming

- **manual_<timestamp>.jpg**: Frames saved with 's' key
- **detect_<frame_number>.jpg**: Auto-saved frames in recording mode (only frames with detections)

## Testing Workflow

### 1. Basic Functionality Test

**Goal**: Verify model loads and runs

```bash
python3 hailo_detect_live.py --model ../models/model.hef

# Check:
# ✅ Model loads without errors
# ✅ Camera initializes
# ✅ Inference runs (FPS > 0)
# ✅ Press 'q' to quit works
```

### 2. Detection Accuracy Test

**Goal**: Verify model detects objects correctly

```bash
# Run with known test objects
python3 hailo_detect_live.py --model ../models/model.hef

# Test scenarios:
# 1. Point camera at drone image/model
# 2. Move object closer/farther
# 3. Try different angles
# 4. Test with multiple objects

# Check:
# ✅ Correct class labels (drone vs IR-Drone)
# ✅ Reasonable confidence scores (>0.25)
# ✅ Bounding boxes fit objects
# ✅ No false positives on empty scenes
```

### 3. Performance Test

**Goal**: Measure FPS and latency

```bash
# Run for 60 seconds
python3 hailo_detect_live.py --model ../models/model.hef

# Wait for 60+ seconds, then press 'q'

# Check final stats:
# ✅ Average FPS > 40 (for YOLOv8s)
# ✅ Detection time < 25ms
# ✅ No frame drops or freezes
```

### 4. Threshold Tuning Test

**Goal**: Find optimal confidence threshold

```bash
# Test different thresholds
python3 hailo_detect_live.py --model ../models/model.hef --conf 0.15  # Low
python3 hailo_detect_live.py --model ../models/model.hef --conf 0.25  # Default
python3 hailo_detect_live.py --model ../models/model.hef --conf 0.35  # High

# Compare:
# - Number of detections
# - False positive rate
# - Missing detections

# Choose best balance
```

### 5. Long-term Stability Test

**Goal**: Test for memory leaks and overheating

```bash
# Run for 10+ minutes
python3 hailo_detect_live.py --model ../models/model.hef --headless

# Monitor with:
# - htop (CPU/memory usage)
# - vcgencmd measure_temp (temperature)

# Check:
# ✅ Memory usage stable (not increasing)
# ✅ CPU temperature < 70°C
# ✅ No crashes or errors
```

## Troubleshooting

### Deployment Issues

#### SSH connection fails

**Problem**: `Cannot connect to pi@<ip>`

**Solutions**:
1. **Verify Pi is on network**: `ping <pi_ip>`
2. **Check SSH is enabled**: 
   ```bash
   # On Pi (if you have monitor/keyboard):
   sudo systemctl enable ssh
   sudo systemctl start ssh
   ```
3. **Try different IP**: Check router for Pi's actual IP
4. **Use hostname**: `ssh pi@raspberrypi.local`

#### File copy fails

**Problem**: `Permission denied` during scp

**Solution**:
```bash
# On Pi, ensure directories exist and have correct permissions:
ssh pi@<ip>
mkdir -p ~/MODEL-GEN/models ~/MODEL-GEN/scripts
chmod 755 ~/MODEL-GEN ~/MODEL-GEN/models ~/MODEL-GEN/scripts
```

### Runtime Issues

#### "No module named 'hailo_platform'"

**Problem**: HailoRT not installed

**Solution**:
```bash
sudo apt update
sudo apt install python3-hailort

# Verify:
python3 -c "from hailo_platform import HEF; print('OK')"
```

#### "Cannot open camera"

**Problem**: Camera device not found

**Solutions**:

1. **For USB camera**:
   ```bash
   # List cameras
   ls -l /dev/video*
   
   # Try different device IDs
   python3 hailo_detect_live.py --model ../models/model.hef --camera 0
   python3 hailo_detect_live.py --model ../models/model.hef --camera 2
   ```

2. **For Pi Camera**:
   ```bash
   # Check camera is enabled
   sudo raspi-config
   # Navigate to: Interface Options → Camera → Enable
   
   # Reboot
   sudo reboot
   
   # Test camera
   rpicam-hello
   ```

3. **Permission issue**:
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

#### Low FPS (<20)

**Problem**: Performance below expectations

**Diagnostic**:
```bash
# Check Hailo device
lspci | grep Hailo
# Should show: "Hailo Technologies Ltd. Hailo-8 AI Processor"

# Check CPU usage
htop
# Hailo process should use <10% CPU

# Check throttling
vcgencmd get_throttled
# Should return: throttled=0x0
```

**Solutions**:

1. **Insufficient power**:
   ```bash
   # Check voltage
   vcgencmd measure_volts
   # Should be ~5.1V, not <4.8V
   
   # Solution: Use 27W official power supply
   ```

2. **CPU throttling (overheating)**:
   ```bash
   # Check temperature
   vcgencmd measure_temp
   # Should be <70°C
   
   # Solutions:
   # - Add heatsink/fan to Pi
   # - Improve airflow
   # - Reduce room temperature
   ```

3. **Wrong camera mode**:
   ```bash
   # USB cameras have higher latency
   # Use Pi Camera Module for best performance:
   python3 hailo_detect_live.py --model ../models/model.hef --picamera
   ```

#### "HEF file not found"

**Problem**: Model file missing or wrong path

**Solution**:
```bash
# Check file exists
ls -lh ~/MODEL-GEN/models/*.hef

# If missing, copy from development machine:
scp ../step4_hef_compilation/model.hef pi@<ip>:~/MODEL-GEN/models/

# Use absolute path:
python3 hailo_detect_live.py --model ~/MODEL-GEN/models/model.hef
```

#### False positives (detections on empty scenes)

**Problem**: Model detects objects that aren't there

**Solutions**:

1. **Increase confidence threshold**:
   ```bash
   python3 hailo_detect_live.py --model ../models/model.hef --conf 0.35
   ```

2. **Check calibration**: Model may need retraining with better calibration data (return to STEP 4)

3. **Verify ONNX export**: Ensure STEP 3 verification passed (0 detections on black image)

#### No detections (objects not detected)

**Problem**: Model doesn't detect visible objects

**Solutions**:

1. **Lower confidence threshold**:
   ```bash
   python3 hailo_detect_live.py --model ../models/model.hef --conf 0.15
   ```

2. **Check object size**: Model trained on 640x640 may miss very small/large objects

3. **Check lighting**: Extreme lighting conditions may affect detection

4. **Verify training**: Check STEP 2 training mAP50 (should be >0.5)

#### Display window not showing (with X forwarding)

**Problem**: Running over SSH without display

**Solution**:
```bash
# Use headless mode
python3 hailo_detect_live.py --model ../models/model.hef --headless

# Or enable X forwarding (slow over network):
ssh -X pi@<ip>
export DISPLAY=:0
python3 hailo_detect_live.py --model ../models/model.hef
```

## Performance Optimization

### 1. Maximize FPS

```bash
# Use Pi Camera (lower latency than USB)
python3 hailo_detect_live.py --model ../models/model.hef --picamera

# Run headless (no rendering overhead)
python3 hailo_detect_live.py --model ../models/model.hef --headless

# Use smaller model (YOLOv8n instead of YOLOv8s)
# Retrain with yolov8n.pt in STEP 2
```

### 2. Reduce Power Consumption

```bash
# Increase confidence threshold (fewer CPU NMS operations)
python3 hailo_detect_live.py --model ../models/model.hef --conf 0.35

# Disable recording mode (don't save every frame)
# Don't press 'r' key

# Lower camera resolution in script (edit hailo_detect_live.py):
# main={"size": (320, 240), "format": "RGB888"}
```

### 3. Improve Accuracy

```bash
# Lower confidence to catch more objects
python3 hailo_detect_live.py --model ../models/model.hef --conf 0.20

# Adjust IOU for overlapping objects
python3 hailo_detect_live.py --model ../models/model.hef --iou 0.35

# Better lighting conditions (well-lit scenes)
```

## Collecting Test Results

### 1. Save Representative Frames

```bash
# Run inference
python3 hailo_detect_live.py --model ../models/model.hef

# When good detection appears, press 's' to save
# Repeat for various scenarios:
# - Close objects
# - Distant objects  
# - Multiple objects
# - Different angles
# - Edge cases
```

### 2. Recording Mode for Dataset

```bash
# Enable auto-save
python3 hailo_detect_live.py --model ../models/model.hef

# Press 'r' to start recording
# All frames with detections will be auto-saved
# Press 'r' again to stop

# Collect frames to:
# - Verify model works in real conditions
# - Gather hard negatives for retraining
# - Create demo videos
```

### 3. Copy Results Back to Development Machine

```bash
# From development machine:
scp -r pi@192.168.3.205:~/MODEL-GEN/test_results ./test_results_pi/

# Review results locally
ls test_results_pi/
```

### 4. Performance Logs

The script prints statistics at the end:

```bash
# Run test, press 'q' when done

# Final output shows:
# - Total frames processed
# - Total detections
# - Average FPS
# - Avg detections per frame
```

Copy this output for documentation:
```bash
# Redirect output to log file
python3 hailo_detect_live.py --model ../models/model.hef 2>&1 | tee test_log.txt

# Copy log back
scp pi@<ip>:~/MODEL-GEN/scripts/test_log.txt ./
```

## Next Steps After Testing

### Model Performs Well

If you achieve target FPS (>40) and good detection accuracy:

1. **Document performance**: Save test logs and example frames
2. **Deploy to production**: Use model in final application
3. **Consider retraining**: Add edge cases found during testing

### Model Needs Improvement

If performance or accuracy is poor:

**Low FPS (<30)**:
- Use smaller model (YOLOv8n)
- Check hardware (power, cooling, Hailo connection)

**Poor accuracy**:
- Return to STEP 1: Add more training data
- Return to STEP 2: Train longer (more epochs)
- Return to STEP 4: Use more calibration samples

**False positives**:
- Increase confidence threshold (`--conf 0.35`)
- Add negative samples to training dataset
- Return to STEP 3: Verify ONNX export quality

**Missing detections**:
- Lower confidence threshold (`--conf 0.15`)
- Check training dataset covers test scenarios
- Verify object sizes match training data

## Summary Checklist

- [ ] Raspberry Pi 5 with Hailo-8 set up and powered
- [ ] HEF model deployed to Pi (`~/MODEL-GEN/models/`)
- [ ] HailoRT installed (`sudo apt install python3-hailort`)
- [ ] Camera connected and working (Pi Camera or USB)
- [ ] Inference script runs without errors
- [ ] FPS meets target (>40 for YOLOv8s)
- [ ] Detections are accurate (correct classes, good bounding boxes)
- [ ] No false positives on empty scenes
- [ ] Test results saved (`~/MODEL-GEN/test_results/`)
- [ ] Performance logs documented

**Production ready**: When all checks pass, your model is deployed and validated for real-world use.

## Resources

**Hailo Documentation**:
- [HailoRT Python API](https://hailo.ai/developer-zone/documentation/)
- [Raspberry Pi 5 Setup Guide](https://www.raspberrypi.com/documentation/)

**Community Support**:
- [Hailo Community Forum](https://community.hailo.ai/)
- [Raspberry Pi Forums](https://forums.raspberrypi.com/)

**Troubleshooting Tools**:
```bash
# Check Hailo device
lspci | grep Hailo
dmesg | grep hailo

# Monitor resources
htop
vcgencmd measure_temp
vcgencmd get_throttled

# Test camera
v4l2-ctl --list-devices
rpicam-hello  # For Pi Camera
```

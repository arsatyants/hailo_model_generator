# Test Results Directory

This directory stores inference results from Raspberry Pi testing.

## Files Saved Here

### Manual Captures (press 's' during inference)
- `manual_<timestamp>.jpg` - Frames saved manually during testing
- Used for documentation, demos, and quality verification

### Auto-Saved Detections (recording mode - press 'r')
- `detect_<frame_number>.jpg` - Frames with detections (auto-saved)
- Only frames containing detected objects are saved
- Useful for collecting edge cases and validation data

## Usage

Results are automatically created when running:
```bash
python3 hailo_detect_live.py --model ../models/model.hef
```

Press:
- `s` to save current frame
- `r` to toggle recording mode (auto-save detections)

## Transfer Results

Copy results back to development machine:
```bash
# From development machine:
scp -r pi@192.168.3.205:~/MODEL-GEN/test_results ./test_results_from_pi/
```

## Analysis

Use saved frames to:
- Verify detection accuracy
- Document model performance
- Identify failure cases for retraining
- Create demo videos/presentations
- Compare different model versions

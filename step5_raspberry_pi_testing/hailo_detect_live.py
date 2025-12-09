#!/usr/bin/env python3
"""
HAILO Real-time Detection with Visualization - HailoRT 4.23
YOLOv8 detection with proper coordinate decoding and NMS

Optimized for Raspberry Pi 5 with Hailo-8 AI accelerator
"""
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Tuple, List

# Check for Picamera2 (Raspberry Pi Camera)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

from hailo_platform import (VDevice, HEF, InputVStreamParams, OutputVStreamParams, 
                            FormatType, InferVStreams)

class HailoDetector:
    def __init__(self, hef_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        self.hef_path = hef_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Load model
        print(f"📦 Loading {hef_path}...")
        self.hef = HEF(hef_path)
        
        # Configure device
        self.vdevice = VDevice()
        self.network_group = self.vdevice.configure(self.hef)[0]
        
        # Create VStream params - use FLOAT32 for outputs (dequantized)
        self.input_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        self.output_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.input_name = list(self.input_params.keys())[0]
        self.network_group_params = self.network_group.create_params()
        
        print(f"✅ Model loaded: {self.input_name}")
        
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess with letterbox padding"""
        h, w = image.shape[:2]
        scale = min(640 / h, 640 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Letterbox padding
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_h = (640 - new_h) // 2
        pad_w = (640 - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert to RGB and add batch dimension
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        batched = np.expand_dims(rgb, axis=0)
        
        return batched, scale, (pad_w, pad_h)
    
    def decode_outputs(self, outputs: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode YOLOv8 outputs to boxes, scores, class_ids"""
        # Extract tensors (remove batch dimension)
        tensors = [v[0] for v in outputs.values()]
        
        # Organize by resolution and type
        reg_outputs = {}  # bbox regression
        cls_outputs = {}  # classification
        
        for tensor in tensors:
            h, w, c = tensor.shape
            if c == 64:  # regression (4 directions * 16 bins)
                reg_outputs[h] = tensor
            elif c == 2 or c == 3:  # classification (2 or 3 classes depending on model)
                cls_outputs[h] = tensor
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        # Process each scale (80x80, 40x40, 20x20)
        for size in sorted(reg_outputs.keys(), reverse=True):
            stride = 640 // size  # 8, 16, or 32
            reg = reg_outputs[size]
            cls = cls_outputs[size]
            
            # Create grid
            grid_y, grid_x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            
            # Decode DFL (Distribution Focal Loss) format - already dequantized floats
            reg = reg.reshape(size, size, 4, 16)
            # Softmax with numerical stability
            reg_max = np.max(reg, axis=-1, keepdims=True)
            reg_exp = np.exp(np.clip(reg - reg_max, -20, 20))  # Clip to prevent overflow
            reg_softmax = reg_exp / (reg_exp.sum(axis=-1, keepdims=True) + 1e-8)
            reg_distances = (reg_softmax * np.arange(16)).sum(axis=-1)  # [H, W, 4]
            
            # Decode to xyxy format
            # DFL outputs distances from center, convert to absolute coordinates
            center_x = (grid_x + 0.5) * stride
            center_y = (grid_y + 0.5) * stride
            
            x1 = center_x - reg_distances[:, :, 0] * stride
            y1 = center_y - reg_distances[:, :, 1] * stride
            x2 = center_x + reg_distances[:, :, 2] * stride
            y2 = center_y + reg_distances[:, :, 3] * stride
            
            boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
            
            # Decode classification scores - apply sigmoid to logits
            cls_clipped = np.clip(cls, -20, 20)  # Prevent overflow
            scores = 1 / (1 + np.exp(-cls_clipped))  # Sigmoid
            
            num_classes = cls.shape[-1]
            scores = scores.reshape(-1, num_classes)
            
            # YOLOv8: if 3 classes, only use first 2 (drone, IR-Drone), last is background/padding
            if num_classes == 3:
                scores = scores[:, :2]  # Only keep first 2 real classes
            
            max_scores = scores.max(axis=1)
            max_class_ids = scores.argmax(axis=1)
            
            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_class_ids.append(max_class_ids)
        
        # Concatenate all scales
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)
        
        return boxes, scores, class_ids
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> dict:
        """Non-Maximum Suppression"""
        # Filter by confidence
        conf_mask = scores > self.conf_thresh
        
        # Filter by valid boxes (positive area, within image bounds)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        valid_mask = (x2 > x1) & (y2 > y1) & (x1 >= 0) & (y1 >= 0) & (x2 < 640) & (y2 < 640)
        
        # Combine masks and filter valid classes (0=drone, 1=IR-Drone only)
        class_mask = (class_ids >= 0) & (class_ids <= 1)
        mask = conf_mask & valid_mask & class_mask
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return {'boxes': np.array([]), 'scores': np.array([]), 'class_ids': np.array([])}
        
        # NMS per class
        keep_indices = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]
            
            # OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                class_boxes.tolist(),
                class_scores.tolist(),
                self.conf_thresh,
                self.iou_thresh
            )
            
            if len(indices) > 0:
                keep_indices.extend(class_indices[indices.flatten()].tolist())
        
        # Return the filtered arrays directly
        if len(keep_indices) == 0:
            return {'boxes': np.array([]), 'scores': np.array([]), 'class_ids': np.array([])}
        
        return {
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'class_ids': class_ids[keep_indices]
        }
    
    def postprocess(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                   scale: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]) -> Tuple:
        """Convert boxes back to original image coordinates"""
        # Apply NMS - returns filtered arrays
        result = self.nms(boxes, scores, class_ids)
        
        boxes = result['boxes']
        scores = result['scores']
        class_ids = result['class_ids']
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Remove padding
        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        
        # Scale back to original image
        boxes /= scale
        
        # Clip to image bounds
        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        
        return boxes, scores, class_ids
    
    def detect(self, image: np.ndarray) -> Tuple:
        """Run full detection pipeline"""
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        input_data, scale, pad = self.preprocess(image)
        
        # Inference
        with self.network_group.activate(self.network_group_params):
            with InferVStreams(self.network_group, self.input_params, self.output_params) as pipeline:
                outputs = pipeline.infer({self.input_name: input_data})
        
        # Decode
        boxes, scores, class_ids = self.decode_outputs(outputs)
        
        # Postprocess
        boxes, scores, class_ids = self.postprocess(boxes, scores, class_ids, scale, pad, (orig_h, orig_w))
        
        return boxes, scores, class_ids
    
    def cleanup(self):
        self.vdevice.release()


def init_camera(use_picamera: bool = False, camera_id: int = 0):
    """Initialize camera (Picamera2 or USB)"""
    if use_picamera and PICAMERA2_AVAILABLE:
        print("📹 Initializing Raspberry Pi Camera...")
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(2)  # Warmup
        print("✅ Picamera2 ready")
        return picam, True
    else:
        print(f"📹 Opening USB camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return None, False
        print("✅ USB camera ready")
        return cap, False


def capture_frame(camera, is_picamera: bool):
    """Capture frame from camera"""
    if is_picamera:
        frame = camera.capture_array()
        # Picamera2 returns RGB, convert to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame
    else:
        return camera.read()


def main():
    parser = argparse.ArgumentParser(
        description='Hailo-8 Real-time Detection on Raspberry Pi 5',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        default='../models/model.hef',
        help='Path to HEF model file (default: ../models/model.hef)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS (default: 0.45)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID for USB camera (default: 0)'
    )
    
    parser.add_argument(
        '--picamera',
        action='store_true',
        help='Use Raspberry Pi Camera (Picamera2)'
    )
    
    parser.add_argument(
        '--save-dir',
        default='../test_results',
        help='Directory to save captured frames (default: ../test_results)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display (save frames only)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 HAILO Real-time Detection - Raspberry Pi 5")
    print("="*60)
    
    # Check model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print("\n💡 Tip: Copy your HEF file from step4_hef_compilation/")
        print(f"   cp ../step4_hef_compilation/model.hef {model_path}")
        return 1
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = HailoDetector(
        hef_path=str(model_path),
        conf_thresh=args.conf,
        iou_thresh=args.iou
    )
    
    # Initialize camera
    camera, is_picamera = init_camera(args.picamera, args.camera)
    if camera is None:
        return 1
    
    print(f"\n💾 Saving results to: {save_dir}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Toggle recording mode (auto-save all frames)")
    print("="*60 + "\n")
    
    # Class names and colors
    class_names = ['drone', 'IR-Drone']
    colors = [(0, 255, 0), (0, 165, 255)]  # Green, Orange
    
    # Stats
    frame_count = 0
    detection_count = 0
    fps_start = time.time()
    fps_display = 0
    recording = False
    
    try:
        while True:
            ret, frame = capture_frame(camera, is_picamera)
            if not ret:
                print("⚠️  Failed to capture frame")
                break
            
            # Run detection
            detect_start = time.time()
            boxes, scores, class_ids = detector.detect(frame)
            detect_time = time.time() - detect_start
            
            # Update stats
            frame_count += 1
            detection_count += len(boxes)
            
            # Draw detections
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                class_idx = int(class_id)
                
                # Skip invalid class IDs
                if class_idx < 0 or class_idx >= len(class_names):
                    continue
                
                color = colors[class_idx]
                label = f"{class_names[class_idx]} {score:.2f}"
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 4), (x1 + label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate FPS
            if frame_count % 10 == 0:
                fps_display = frame_count / (time.time() - fps_start)
            
            # Draw info overlay
            info_text = f"FPS: {fps_display:.1f} | Detections: {len(boxes)} | Time: {detect_time*1000:.1f}ms"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if recording:
                cv2.putText(frame, "REC", (frame.shape[1] - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Print detection info every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Found {len(boxes)} detections | FPS: {fps_display:.1f}")
                if len(boxes) > 0:
                    for i, (box, score, class_id) in enumerate(zip(boxes[:3], scores[:3], class_ids[:3])):
                        box_w, box_h = box[2] - box[0], box[3] - box[1]
                        class_idx = int(class_id)
                        if 0 <= class_idx < len(class_names):
                            print(f"  #{i+1}: {class_names[class_idx]} score={score:.3f} size={box_w:.1f}x{box_h:.1f}")
            
            # Auto-save if recording
            if recording and len(boxes) > 0:
                filename = save_dir / f"detect_{frame_count:06d}.jpg"
                cv2.imwrite(str(filename), frame)
            
            # Display (if not headless)
            if not args.headless:
                cv2.imshow('HAILO Detection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = save_dir / f"manual_{int(time.time())}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"💾 Saved: {filename}")
                elif key == ord('r'):
                    recording = not recording
                    print(f"🔴 Recording: {'ON' if recording else 'OFF'}")
            else:
                # Headless mode - save frames with detections
                if len(boxes) > 0:
                    filename = save_dir / f"detect_{frame_count:06d}.jpg"
                    cv2.imwrite(str(filename), frame)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopped by user")
    
    finally:
        # Final stats
        elapsed = time.time() - fps_start
        print(f"\n{'='*60}")
        print("📊 Final Statistics")
        print(f"{'='*60}")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {detection_count}")
        print(f"   Runtime: {elapsed:.1f}s")
        print(f"   Average FPS: {frame_count/elapsed:.1f}")
        print(f"   Avg detections/frame: {detection_count/max(frame_count, 1):.2f}")
        print(f"   Results saved to: {save_dir}")
        print(f"{'='*60}")
        
        # Cleanup
        if is_picamera:
            camera.stop()
        else:
            camera.release()
        
        if not args.headless:
            cv2.destroyAllWindows()
        
        detector.cleanup()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Multi-Class Drone Annotation Tool
Interactive tool to annotate drones and IR-drones in images with bounding boxes
Supports multiple classes for YOLO format
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml

class DroneAnnotator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        
        # Load class names from data.yaml
        data_yaml = self.dataset_path / "data.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', ['Aircraft', 'IR-Drone'])
        else:
            self.class_names = ['Aircraft', 'IR-Drone']
        
        print(f"📋 Available classes:")
        for i, name in enumerate(self.class_names):
            print(f"   {i}: {name}")
        
        # State
        self.current_image_idx = 0
        # Get all image files (jpg, jpeg, png)
        self.images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.images.extend(self.images_dir.glob(ext))
        self.images = sorted(self.images)
        self.drawing = False
        self.start_point = None
        self.current_bbox = None
        self.current_class = 0  # Default to first class
        self.temp_image = None
        
        # Colors for different classes
        self.class_colors = [
            (0, 255, 0),    # Green for Aircraft
            (255, 0, 255),  # Magenta for IR-Drone
            (0, 255, 255),  # Yellow for class 2
            (255, 128, 0),  # Orange for class 3
        ]
        
        print(f"\n✓ Found {len(self.images)} images")
        print(f"✓ Dataset: {self.dataset_path}")
    
    def load_annotations(self, image_file):
        """Load existing annotations for an image"""
        label_file = self.labels_dir / f"{image_file.stem}.txt"
        annotations = []
        
        if label_file.exists() and label_file.stat().st_size > 0:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
        
        return annotations
    
    def save_annotations(self, image_file, annotations):
        """Save annotations to label file"""
        label_file = self.labels_dir / f"{image_file.stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def draw_annotations(self, image, annotations, img_height, img_width):
        """Draw existing annotations on image"""
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            
            # Convert normalized to pixel coordinates
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)
            
            # Get color for class
            color = self.class_colors[class_id % len(self.class_colors)]
            
            # Draw bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[class_id]}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_image = self.current_image.copy()
                color = self.class_colors[self.current_class % len(self.class_colors)]
                cv2.rectangle(self.temp_image, self.start_point, (x, y), color, 2)
                
                # Show class name
                label = f"Drawing: {self.class_names[self.current_class]}"
                cv2.putText(self.temp_image, label, (self.start_point[0], self.start_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.current_bbox = (self.start_point[0], self.start_point[1], x, y)
    
    def bbox_to_yolo(self, bbox, img_height, img_width):
        """Convert pixel bbox to YOLO format (normalized)"""
        x1, y1, x2, y2 = bbox
        
        # Ensure correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return (self.current_class, x_center, y_center, width, height)
    
    def run(self):
        """Main annotation loop"""
        if not self.images:
            print("❌ No images found")
            return
        
        cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotation Tool', 1280, 720)
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        print("\n" + "="*70)
        print("🎨 ANNOTATION CONTROLS")
        print("="*70)
        print("Mouse:")
        print("  - Click & drag: Draw bounding box")
        print("\nKeyboard:")
        print("  - 0-9: Switch class (0=Aircraft, 1=IR-Drone, etc.)")
        print("  - SPACE: Add bbox to current image (continue annotating)")
        print("  - ENTER: Save and move to next image")
        print("  - 'n': Next image (without saving bbox)")
        print("  - 'p': Previous image")
        print("  - 'd': Delete last annotation")
        print("  - 's': Save current annotations")
        print("  - 'q': Quit")
        print("="*70 + "\n")
        
        while self.current_image_idx < len(self.images):
            image_file = self.images[self.current_image_idx]
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"⚠️  Failed to load: {image_file}")
                self.current_image_idx += 1
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Load existing annotations
            annotations = self.load_annotations(image_file)
            
            # Reset state for new image
            self.current_image = image.copy()
            self.temp_image = None
            self.current_bbox = None
            
            while True:
                # Display image
                display = self.temp_image if self.temp_image is not None else self.current_image.copy()
                
                # Draw existing annotations
                self.draw_annotations(display, annotations, img_height, img_width)
                
                # Draw info overlay
                overlay_h = 120
                cv2.rectangle(display, (5, 5), (500, overlay_h), (0, 0, 0), -1)
                cv2.rectangle(display, (5, 5), (500, overlay_h), (0, 255, 0), 2)
                
                info_text = [
                    f"Image: {self.current_image_idx + 1}/{len(self.images)} - {image_file.name}",
                    f"Annotations: {len(annotations)}",
                    f"Current class: {self.current_class} - {self.class_names[self.current_class]}",
                    "SPACE=add bbox | ENTER=next image"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(display, text, (10, 25 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                cv2.imshow('Annotation Tool', display)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Class selection (0-9)
                if ord('0') <= key <= ord('9'):
                    class_num = key - ord('0')
                    if class_num < len(self.class_names):
                        self.current_class = class_num
                        print(f"✓ Switched to class {self.current_class}: {self.class_names[self.current_class]}")
                
                # Add bbox to current image (continue annotating)
                elif key == ord(' '):
                    if self.current_bbox:
                        yolo_bbox = self.bbox_to_yolo(self.current_bbox, img_height, img_width)
                        annotations.append(yolo_bbox)
                        print(f"✓ Added {self.class_names[self.current_class]} bbox (total: {len(annotations)})")
                        self.current_bbox = None
                        # Reset the current image to show updated annotations
                        self.current_image = image.copy()
                
                # Next image (ENTER key)
                elif key == 13:  # Enter key
                    self.save_annotations(image_file, annotations)
                    print(f"💾 Saved: {image_file.stem}.txt ({len(annotations)} annotations)")
                    self.current_image_idx += 1
                    break
                
                # Next without saving bbox
                elif key == ord('n'):
                    self.save_annotations(image_file, annotations)
                    print(f"⏭️  Next: {image_file.name}")
                    self.current_image_idx += 1
                    break
                
                # Previous
                elif key == ord('p'):
                    if self.current_image_idx > 0:
                        self.save_annotations(image_file, annotations)
                        self.current_image_idx -= 1
                        print(f"⏮️  Previous")
                    break
                
                # Delete last annotation
                elif key == ord('d'):
                    if annotations:
                        removed = annotations.pop()
                        print(f"🗑️  Deleted last annotation (class {removed[0]})")
                
                # Save
                elif key == ord('s'):
                    self.save_annotations(image_file, annotations)
                    print(f"💾 Saved: {len(annotations)} annotations")
                
                # Quit
                elif key == ord('q'):
                    self.save_annotations(image_file, annotations)
                    print("\n👋 Quitting...")
                    cv2.destroyAllWindows()
                    return
        
        print("\n✅ All images annotated!")
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Multi-class Drone Annotation Tool')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory (should contain images/ and labels/)')
    
    args = parser.parse_args()
    
    annotator = DroneAnnotator(args.dataset)
    annotator.run()


if __name__ == "__main__":
    main()

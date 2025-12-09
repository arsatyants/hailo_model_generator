#!/usr/bin/env python3
"""
Verify dataset annotations
Check for issues like missing labels, incorrect format, out-of-bounds coordinates
"""

import os
from pathlib import Path
import argparse

def verify_annotations(dataset_dir):
    """
    Verify YOLO format annotations in dataset
    
    Checks:
    - Missing label files
    - Empty label files (valid for negative samples)
    - Label format correctness
    - Coordinate bounds [0, 1]
    - Class ID validity
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(images_dir.glob(ext))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return False
    
    print(f"\n📊 Verifying {len(image_files)} images...")
    
    issues = []
    stats = {
        'total': len(image_files),
        'with_labels': 0,
        'empty_labels': 0,
        'missing_labels': 0,
        'total_boxes': 0,
        'invalid_format': 0,
        'out_of_bounds': 0,
        'invalid_class': 0
    }
    
    for img_file in sorted(image_files):
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        # Check if label exists
        if not label_file.exists():
            issues.append(f"Missing label: {img_file.name}")
            stats['missing_labels'] += 1
            continue
        
        # Check if empty (valid for negative samples)
        if label_file.stat().st_size == 0:
            stats['empty_labels'] += 1
            continue
        
        stats['with_labels'] += 1
        
        # Verify label format
        try:
            with open(label_file, 'r') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"{img_file.name}: Line {line_no} - Expected 5 values, got {len(parts)}")
                        stats['invalid_format'] += 1
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        stats['total_boxes'] += 1
                        
                        # Check class ID (assuming 2 classes: 0=drone, 1=IR-Drone)
                        if class_id < 0 or class_id > 1:
                            issues.append(f"{img_file.name}: Line {line_no} - Invalid class_id {class_id}")
                            stats['invalid_class'] += 1
                        
                        # Check bounds [0, 1]
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 <= width <= 1 and 0 <= height <= 1):
                            issues.append(f"{img_file.name}: Line {line_no} - Coordinates out of bounds [0,1]")
                            stats['out_of_bounds'] += 1
                        
                    except ValueError as e:
                        issues.append(f"{img_file.name}: Line {line_no} - Invalid number format: {e}")
                        stats['invalid_format'] += 1
                        
        except Exception as e:
            issues.append(f"{img_file.name}: Error reading file - {e}")
            stats['invalid_format'] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(f"Total images: {stats['total']}")
    print(f"  With labels: {stats['with_labels']} ({stats['with_labels']/stats['total']*100:.1f}%)")
    print(f"  Empty labels (negative samples): {stats['empty_labels']}")
    print(f"  Missing labels: {stats['missing_labels']}")
    print(f"\nTotal bounding boxes: {stats['total_boxes']}")
    
    if stats['invalid_format'] + stats['out_of_bounds'] + stats['invalid_class'] > 0:
        print(f"\n⚠️  ISSUES FOUND:")
        print(f"  Invalid format: {stats['invalid_format']}")
        print(f"  Out of bounds: {stats['out_of_bounds']}")
        print(f"  Invalid class ID: {stats['invalid_class']}")
        
        if issues:
            print(f"\n📋 First 20 issues:")
            for issue in issues[:20]:
                print(f"   - {issue}")
            if len(issues) > 20:
                print(f"   ... and {len(issues)-20} more issues")
        
        return False
    else:
        print(f"\n✅ All annotations are valid!")
        avg_boxes = stats['total_boxes'] / max(stats['with_labels'], 1)
        print(f"   Average boxes per image: {avg_boxes:.2f}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Verify YOLO dataset annotations')
    parser.add_argument(
        '--dataset-dir',
        default='../../lesson_18/drone_dataset/yolo_dataset',
        help='YOLO dataset directory (default: ../../lesson_18/drone_dataset/yolo_dataset)'
    )
    
    args = parser.parse_args()
    
    # Resolve path
    script_dir = Path(__file__).parent
    dataset_dir = (script_dir / args.dataset_dir).resolve()
    
    print("="*60)
    print("Verify Dataset Annotations")
    print("="*60)
    print(f"Dataset: {dataset_dir}")
    
    success = verify_annotations(dataset_dir)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
Add captured images to YOLO dataset
Copies images from captured_images directory to dataset and creates empty label files
"""

import os
import shutil
from pathlib import Path
import argparse

def add_images_to_dataset(captured_dir, dataset_dir):
    """
    Copy images from captured directory to dataset
    Create empty label files for new images (for later annotation)
    
    Args:
        captured_dir: Source directory with captured images
        dataset_dir: Target YOLO dataset directory
    """
    captured_path = Path(captured_dir)
    dataset_path = Path(dataset_dir)
    
    # Create dataset directories if they don't exist
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(captured_path.glob(ext))
    
    if not image_files:
        print(f"⚠️  No images found in {captured_dir}")
        return 0
    
    print(f"\n📸 Found {len(image_files)} images in {captured_dir}")
    
    copied = 0
    skipped = 0
    
    for img_file in image_files:
        # Skip result images (inference outputs)
        if 'result_' in img_file.name:
            skipped += 1
            continue
        
        # Copy image
        dest_img = images_dir / img_file.name
        if dest_img.exists():
            print(f"   ⏭️  Skip (exists): {img_file.name}")
            skipped += 1
            continue
        
        shutil.copy2(img_file, dest_img)
        
        # Create empty label file (for annotation)
        label_file = labels_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            label_file.touch()  # Create empty file
        
        print(f"   ✓ Added: {img_file.name}")
        copied += 1
    
    print(f"\n✅ Added {copied} images to dataset")
    print(f"   Skipped: {skipped} (already exist or result images)")
    print(f"\n📁 Dataset location: {dataset_path}")
    print(f"   Images: {len(list(images_dir.glob('*')))} files")
    print(f"   Labels: {len(list(labels_dir.glob('*.txt')))} files")
    
    return copied


def main():
    parser = argparse.ArgumentParser(description='Add captured images to YOLO dataset')
    parser.add_argument(
        '--captured-dir',
        default='../../captured_images',
        help='Directory with captured images (default: ../../captured_images)'
    )
    parser.add_argument(
        '--dataset-dir',
        default='../../datasets',
        help='Target YOLO dataset directory (default: ../../datasets)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    captured_dir = (script_dir / args.captured_dir).resolve()
    dataset_dir = (script_dir / args.dataset_dir).resolve()
    
    print("="*60)
    print("Add Images to Dataset")
    print("="*60)
    print(f"Source: {captured_dir}")
    print(f"Target: {dataset_dir}")
    
    if not captured_dir.exists():
        print(f"\n❌ Error: Captured images directory not found: {captured_dir}")
        print("   Create it or specify correct path with --captured-dir")
        return 1
    
    # Add images
    count = add_images_to_dataset(captured_dir, dataset_dir)
    
    if count > 0:
        print(f"\n💡 Next step: Annotate images")
        print(f"   python3 annotate_drones.py --dataset-dir {dataset_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())

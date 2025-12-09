#!/usr/bin/env python3
"""
Split dataset into train and validation sets
Randomly splits images and their labels into train/val directories
"""

import shutil
from pathlib import Path
import argparse
import random


def split_dataset(dataset_dir, val_ratio=0.15, seed=42):
    """
    Split dataset into train and validation sets
    
    Args:
        dataset_dir: Path to dataset directory (should contain train/ subdirectory)
        val_ratio: Ratio of validation images (default 0.15 = 15%)
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_dir)
    train_img = dataset_path / "train" / "images"
    train_lbl = dataset_path / "train" / "labels"
    val_img = dataset_path / "val" / "images"
    val_lbl = dataset_path / "val" / "labels"
    
    # Check if train directory exists
    if not train_img.exists():
        print(f"❌ Error: Training images directory not found: {train_img}")
        return False
    
    # Create validation directories
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)
    
    # Get all images from train
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in image_extensions:
        images.extend(train_img.glob(ext))
    
    if not images:
        print(f"❌ Error: No images found in {train_img}")
        return False
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio))
    
    print("="*60)
    print("Split Train/Validation Dataset")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Total images: {len(images)}")
    print(f"Validation ratio: {val_ratio:.0%}")
    print(f"Split: {len(images) - val_count} train, {val_count} validation")
    print("="*60)
    
    # Move validation images
    moved = 0
    for img in images[:val_count]:
        lbl = train_lbl / f"{img.stem}.txt"
        
        try:
            # Move image
            shutil.move(str(img), str(val_img / img.name))
            
            # Move label if exists
            if lbl.exists():
                shutil.move(str(lbl), str(val_lbl / lbl.name))
            
            moved += 1
            if moved % 5 == 0 or moved == val_count:
                print(f"   Moved {moved}/{val_count} images to validation...")
        except Exception as e:
            print(f"⚠️  Failed to move {img.name}: {e}")
    
    # Count final results
    train_count = len(list(train_img.glob('*.*')))
    val_count_final = len(list(val_img.glob('*.*')))
    
    print(f"\n✅ Split complete!")
    print(f"   Train: {train_count} images")
    print(f"   Validation: {val_count_final} images")
    print(f"   Ratio: {train_count/(train_count+val_count_final):.1%} / {val_count_final/(train_count+val_count_final):.1%}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train and validation sets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dataset-dir',
        default='../datasets',
        help='Dataset directory (default: ../datasets)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation ratio (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Resolve path
    script_dir = Path(__file__).parent
    dataset_dir = (script_dir / args.dataset_dir).resolve()
    
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    # Split dataset
    success = split_dataset(dataset_dir, args.val_ratio, args.seed)
    
    if success:
        print(f"\n💡 Next step: Train model")
        print(f"   cd ../step2_training")
        print(f"   python3 train_yolov8.py {dataset_dir}/data.yaml")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
Prepare calibration dataset for Hailo quantization

Converts training images to UINT8 format [0-255] for calibration.
Hailo expects UINT8 numpy arrays, not normalized float32.

Critical: Calibration format must match inference format:
- Input: UINT8 [0-255]
- Shape: (640, 640, 3) HWC
- Color: RGB order
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil


def prepare_calibration_images(source_dir, output_dir, num_samples=64, imgsz=640):
    """
    Prepare calibration dataset from training images
    
    Args:
        source_dir: Path to training images directory
        output_dir: Path to save calibration images
        num_samples: Number of images to use (default 64)
        imgsz: Target image size (default 640)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Prepare Calibration Dataset for Hailo")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
    print(f"Image size: {imgsz}x{imgsz}")
    print("="*60)
    
    # Find image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(source_dir.glob(ext))
    
    # Also search in subdirectories (train/images/, val/images/)
    for subdir in ['train/images', 'val/images', 'images']:
        subdir_path = source_dir / subdir
        if subdir_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(subdir_path.glob(ext))
    
    if len(image_files) == 0:
        print(f"❌ Error: No images found in {source_dir}")
        print("\n📁 Expected structure:")
        print("   source_dir/")
        print("   ├── image_001.jpg")
        print("   ├── image_002.jpg")
        print("   └── ...")
        print("   OR")
        print("   source_dir/train/images/")
        print("   ├── image_001.jpg")
        print("   └── ...")
        return False
    
    print(f"\n📊 Found {len(image_files)} images")
    
    # Select samples (evenly distributed)
    if len(image_files) > num_samples:
        step = len(image_files) // num_samples
        selected = [image_files[i * step] for i in range(num_samples)]
    else:
        selected = image_files
        print(f"⚠️  Using all {len(selected)} images (less than requested {num_samples})")
    
    print(f"📸 Processing {len(selected)} calibration images...")
    
    # Process images
    success_count = 0
    for i, img_path in enumerate(selected):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ⚠️  Failed to read: {img_path.name}")
                continue
            
            # Resize to target size
            img_resized = cv2.resize(img, (imgsz, imgsz))
            
            # Convert BGR to RGB (Hailo expects RGB)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Save as NPY file (UINT8 [0-255] - NO normalization)
            output_path = output_dir / f"calib_{i:04d}.npy"
            np.save(str(output_path), img_rgb)
            
            success_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(selected)}")
                
        except Exception as e:
            print(f"   ❌ Error processing {img_path.name}: {e}")
            continue
    
    print(f"\n✅ Prepared {success_count} calibration images")
    print(f"   Output directory: {output_dir}")
    print(f"   Format: NPY (UINT8 RGB [0-255])")
    print(f"   Size: {imgsz}x{imgsz}x3")
    
    # Verify output
    sample_files = list(output_dir.glob('calib_*.npy'))
    if len(sample_files) > 0:
        sample_img = np.load(str(sample_files[0]))
        print(f"\n🔍 Verification:")
        print(f"   Sample image shape: {sample_img.shape}")
        print(f"   Data type: {sample_img.dtype}")
        print(f"   Value range: [{sample_img.min()}, {sample_img.max()}]")
        
        if sample_img.dtype != np.uint8:
            print(f"   ⚠️  Warning: Expected uint8, got {sample_img.dtype}")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description='Prepare calibration dataset for Hailo quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # From training dataset
  python3 prepare_calibration.py ../datasets/

  # Custom number of samples
  python3 prepare_calibration.py ../datasets/ --num-samples 100

  # Custom output directory
  python3 prepare_calibration.py /path/to/images/ --output ./my_calib_data/
        '''
    )
    
    parser.add_argument(
        'source_dir',
        help='Path to source images directory (training dataset)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./calibration_data',
        help='Output directory for calibration images (default: ./calibration_data)'
    )
    
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=64,
        help='Number of calibration samples (default: 64, recommended: 50-100)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Target image size (default: 640)'
    )
    
    args = parser.parse_args()
    
    # Prepare calibration data
    success = prepare_calibration_images(
        args.source_dir,
        args.output,
        args.num_samples,
        args.imgsz
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ CALIBRATION DATASET READY")
        print("="*60)
        print(f"\n💡 Next step: Compile to HEF")
        print(f"   python3 compile_to_hef.py <onnx_file> --calib {args.output}")
        print()
        return 0
    else:
        print("\n❌ Failed to prepare calibration dataset")
        return 1


if __name__ == '__main__':
    exit(main())

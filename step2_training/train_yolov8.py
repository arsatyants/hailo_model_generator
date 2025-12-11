#!/usr/bin/env python3
"""
Train YOLOv8 model for drone detection
Supports both drone and IR-Drone classes
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO  # type: ignore

def train_yolov8(data_yaml, epochs=200, batch=32, imgsz=640, patience=15, name='train_drone_ir'):
    """
    Train YOLOv8 model on drone dataset
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Maximum number of training epochs
        batch: Batch size
        imgsz: Input image size
        patience: Early stopping patience (epochs without improvement)
        name: Training run name
    """
    print("="*60)
    print("YOLOv8 Training - Drone Detection")
    print("="*60)
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Patience: {patience}")
    print(f"Run name: {name}")
    print("="*60)
    
    # Check if data.yaml exists
    if not Path(data_yaml).exists():
        print(f"\n❌ Error: data.yaml not found: {data_yaml}")
        print("   Make sure the dataset is properly set up")
        return False
    
    # Load YOLOv8s model (small variant - good balance of speed/accuracy)
    print("\n📦 Loading YOLOv8s base model...")
    model = YOLO('yolov8s.pt')
    
    # Train the model
    print("\n🚀 Starting training...")
    print("   This may take 1-3 hours depending on dataset size and hardware")
    print("   Training will stop early if no improvement after 15 epochs")
    print("   Press Ctrl+C to stop manually\n")
    
    # Set project directory to keep training output in project root runs/ folder
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    project_dir = project_root / "runs" / "detect"
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=patience,
            name=name,
            project=str(project_dir),
            plots=True,  # Generate training plots
            save=True,   # Save checkpoints
            device=0,    # Use GPU if available, else CPU
        )
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        print(f"Best weights: {project_dir}/{name}/weights/best.pt")
        print(f"Last weights: {project_dir}/{name}/weights/last.pt")
        print(f"\nTraining plots saved to: {project_dir}/{name}/")
        print("\n💡 Next step: Export to ONNX")
        print(f"   cd ../step3_onnx_export")
        print(f"   python3 export_onnx_with_nms.py {project_dir}/{name}/weights/best.pt")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"   Partial weights may be available in {project_dir}/{name}/weights/")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for drone detection')
    parser.add_argument(
        '--data',
        default='../datasets/data.yaml',
        help='Path to data.yaml (default: ../datasets/data.yaml)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size (default: 32, reduce if out of memory)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience in epochs (default: 15)'
    )
    parser.add_argument(
        '--name',
        default='train_drone_ir',
        help='Training run name (default: train_drone_ir)'
    )
    
    args = parser.parse_args()
    
    # Resolve data.yaml path
    script_dir = Path(__file__).parent
    data_yaml = (script_dir / args.data).resolve()
    
    # Train model
    success = train_yolov8(
        data_yaml=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        name=args.name
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

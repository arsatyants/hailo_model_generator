#!/usr/bin/env python3
"""
Export YOLOv8 PyTorch model to ONNX format for Hailo compilation

CRITICAL: For Hailo-8, we export WITHOUT NMS (nms=False)
NMS will be handled in post-processing on Raspberry Pi
"""

import argparse
from pathlib import Path
from ultralytics import YOLO  # type: ignore
import re

def export_onnx_for_hailo(pt_path, output_dir=None, imgsz=640, nms=False):
    """
    Export PyTorch model to ONNX WITHOUT NMS for Hailo compilation
    
    Args:
        pt_path: Path to .pt model file
        output_dir: Output directory (default: same as model)
        imgsz: Input image size
        
    Returns:
        Path to exported ONNX file
    """
    pt_path = Path(pt_path)
    
    if not pt_path.exists():
        print(f"❌ Error: Model file not found: {pt_path}")
        return None
    
    print("="*60)
    print("Export YOLOv8 to ONNX for Hailo")
    print("="*60)
    print(f"Model: {pt_path}")
    print(f"Image size: {imgsz}")
    if nms:
        print("NMS: ENABLED (embedded in ONNX, no Python NMS needed)")
    else:
        print("NMS: DISABLED (post-processing on Pi)")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model = YOLO(str(pt_path))
    
    print("\n🚀 Exporting to ONNX...")
    print("   Settings:")
    print(f"   - nms={nms} ({'embedded in ONNX' if nms else 'NMS in post-processing'})")
    print("   - opset=11 (Hailo compatible)")
    print("   - simplify=True (graph optimization)")
    print()
    try:
        result = model.export(
            format='onnx',
            imgsz=imgsz,
            nms=nms,         # User-controlled
            opset=11,        # Hailo requires opset 11
            simplify=True    # Simplify graph
        )
        
        # Find ONNX file
        onnx_path = None
        if isinstance(result, str) and result.endswith('.onnx'):
            onnx_path = Path(result)
        elif isinstance(result, (list, tuple)):
            for item in result:
                if str(item).endswith('.onnx'):
                    onnx_path = Path(str(item))
                    break
        
        if not onnx_path or not onnx_path.exists():
            # Search for generated ONNX
            onnx_path = pt_path.parent / f"{pt_path.stem}.onnx"
            if not onnx_path.exists():
                print(f"❌ Error: Could not find exported ONNX file")
                return None
        
        # Move to output dir if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            final_path = output_dir / f"{pt_path.stem}_hailo.onnx"
            import shutil
            shutil.copy2(onnx_path, final_path)
            onnx_path = final_path
        
        print("\n" + "="*60)
        print("✅ Export Successful!")
        print("="*60)
        print(f"ONNX file: {onnx_path}")
        print(f"Size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        print("📊 Expected output format:")
        print("   Shape: (1, 84, 8400) for YOLOv8")
        print("   Format: Raw predictions [x, y, w, h, class_scores...]")
        print()
        print("💡 Next step: Compile to HEF")
        print(f"   cd ../step4_hef_compilation")
        print(f"   python3 compile_to_hef.py {onnx_path}")
        
        return onnx_path
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        return None


def find_latest_model(runs_dir=None):
    """
    Find the latest trained model by looking at numbered training directories
    
    Args:
        runs_dir: Path to runs/detect directory (default: ../runs/detect)
        
    Returns:
        Path to best.pt or None if not found
    """
    if runs_dir is None:
        # Default to project root runs/detect
        script_dir = Path(__file__).parent
        runs_dir = script_dir.parent / "runs" / "detect"
    else:
        runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        return None
    
    # Find all directories matching pattern (e.g., drone_detector, drone_detector2, etc.)
    model_dirs = []
    for dir_path in runs_dir.iterdir():
        if dir_path.is_dir():
            # Extract number from directory name if it exists
            match = re.search(r'(\d+)$', dir_path.name)
            number = int(match.group(1)) if match else 0
            model_dirs.append((number, dir_path))
    
    if not model_dirs:
        return None
    
    # Sort by number (highest first)
    model_dirs.sort(key=lambda x: x[0], reverse=True)
    
    # Find best.pt in the latest directory
    for _, dir_path in model_dirs:
        best_pt = dir_path / "weights" / "best.pt"
        if best_pt.exists():
            return best_pt
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 to ONNX for Hailo compilation')
    parser.add_argument(
        'model',
        nargs='?',
        help='Path to .pt model file (optional: auto-finds latest if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory (default: same as model)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--runs-dir',
        help='Custom runs directory to search (default: ../runs/detect)'
    )
    
    parser.add_argument(
        '--nms',
        action='store_true',
        help='Export ONNX with NMS embedded (default: False, disables NMS for Hailo best practice)'
    )

    args = parser.parse_args()

    # Auto-find latest model if not provided
    model_path = args.model
    if not model_path:
        print("🔍 Searching for latest trained model...")
        model_path = find_latest_model(args.runs_dir)
        if model_path:
            print(f"✅ Found latest model: {model_path}")
        else:
            print("❌ Error: No trained model found")
            print("   Please provide model path explicitly or train a model first")
            return 1

    # Export
    onnx_path = export_onnx_for_hailo(
        pt_path=model_path,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        nms=args.nms
    )

    return 0 if onnx_path else 1


if __name__ == '__main__':
    exit(main())

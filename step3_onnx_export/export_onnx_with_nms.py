#!/usr/bin/env python3
"""
Export YOLOv8 PyTorch model to ONNX format WITH NMS
Critical: nms=True ensures embedded NMS for Hailo compatibility
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def export_onnx_with_nms(pt_path, output_dir=None, imgsz=640):
    """
    Export PyTorch model to ONNX with embedded NMS
    
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
    print("Export YOLOv8 to ONNX with NMS")
    print("="*60)
    print(f"Model: {pt_path}")
    print(f"Image size: {imgsz}")
    print("NMS: ENABLED (critical for Hailo)")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model = YOLO(str(pt_path))
    
    # Export with NMS enabled
    print("\n🚀 Exporting to ONNX...")
    print("   Settings:")
    print("   - nms=True (embeds NMS in graph)")
    print("   - opset=11 (Hailo compatible)")
    print("   - simplify=True (graph optimization)")
    print()
    
    try:
        result = model.export(
            format='onnx',
            imgsz=imgsz,
            nms=True,        # CRITICAL: Embed NMS in ONNX
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
            final_path = output_dir / f"{pt_path.stem}_nms.onnx"
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
        print("   Shape: (1, 300, 6)")
        print("   Format: [batch, detections, (x1, y1, x2, y2, conf, class)]")
        print()
        print("💡 Next step: Verify export")
        print(f"   python3 verify_onnx_export.py {onnx_path}")
        
        return onnx_path
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 to ONNX with NMS')
    parser.add_argument(
        'model',
        help='Path to .pt model file'
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
    
    args = parser.parse_args()
    
    # Export
    onnx_path = export_onnx_with_nms(
        pt_path=args.model,
        output_dir=args.output_dir,
        imgsz=args.imgsz
    )
    
    return 0 if onnx_path else 1


if __name__ == '__main__':
    exit(main())

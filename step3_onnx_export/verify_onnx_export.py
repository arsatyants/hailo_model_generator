#!/usr/bin/env python3
"""
Verify ONNX export quality
Tests for false positives on black images - critical quality check
"""

import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path

def verify_onnx_export(onnx_path):
    """
    Verify ONNX model produces correct output
    
    Test: Run inference on pure black image
    Expected: 0 detections (no false positives)
    
    Args:
        onnx_path: Path to ONNX file
        
    Returns:
        True if verification passed, False otherwise
    """
    onnx_path = Path(onnx_path)
    
    if not onnx_path.exists():
        print(f"❌ Error: ONNX file not found: {onnx_path}")
        return False
    
    print("="*60)
    print("Verify ONNX Export - Black Image Test")
    print("="*60)
    print(f"Model: {onnx_path}")
    print(f"Size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("="*60)
    
    try:
        # Load ONNX model
        print("\n📦 Loading ONNX model...")
        session = ort.InferenceSession(str(onnx_path))
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"   Input name: {input_name}")
        print(f"   Input shape: {input_shape}")
        
        # Get output details
        outputs = session.get_outputs()
        print(f"   Outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"      [{i}] {output.name}: {output.shape}")
        
        # Create black test image
        print("\n🖤 Creating pure black test image (640x640)...")
        # ONNX expects NCHW format: (batch, channels, height, width)
        black_image = np.zeros((1, 3, 640, 640), dtype=np.float32)
        
        # Run inference
        print("🔍 Running inference on black image...")
        result = session.run(None, {input_name: black_image})
        
        # Analyze results
        print("\n📊 Results:")
        
        # Check output format
        if len(result) == 1:
            output = result[0]
            print(f"   Output shape: {output.shape}")
            
            # Expected format with NMS: (1, 300, 6)
            if len(output.shape) == 3 and output.shape[1] <= 300:
                # Post-NMS format (batch, max_detections, 6)
                detections = output[0]  # Remove batch dimension
                
                # Count valid detections (confidence > 0)
                valid_detections = []
                for det in detections:
                    if len(det) >= 5 and det[4] > 0.01:  # confidence > 1%
                        valid_detections.append(det)
                
                num_detections = len(valid_detections)
                print(f"   Detections found: {num_detections}")
                
                if num_detections > 0:
                    print(f"\n   ⚠️  Detection details:")
                    for i, det in enumerate(valid_detections[:5]):  # Show first 5
                        x1, y1, x2, y2, conf = det[:5]
                        class_id = int(det[5]) if len(det) > 5 else -1
                        print(f"      [{i}] x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, conf={conf:.3f}, class={class_id}")
                
                # Verdict
                if num_detections == 0:
                    print("\n" + "="*60)
                    print("✅ VERIFICATION PASSED")
                    print("="*60)
                    print("   ONNX model correctly produces 0 detections on black image")
                    print("   Export with nms=True is working correctly")
                    print("\n💡 Next step: Compile to HEF")
                    print(f"   cd ../step4_hef_compilation")
                    print(f"   ./compile_to_hef.sh {onnx_path.resolve()}")
                    return True
                else:
                    print("\n" + "="*60)
                    print("❌ VERIFICATION FAILED")
                    print("="*60)
                    print(f"   Found {num_detections} false positive(s) on black image")
                    print("   This indicates the ONNX export is incorrect")
                    print("\n🔧 Solution:")
                    print("   Re-export with nms=True:")
                    print(f"   python3 export_onnx_with_nms.py <model.pt>")
                    return False
                    
            else:
                # Raw format (1, 6, 8400) - NMS not embedded
                print(f"\n⚠️  Output format indicates NMS is NOT embedded")
                print(f"   Expected: (1, 300, 6)")
                print(f"   Got: {output.shape}")
                print("\n❌ VERIFICATION FAILED")
                print("\n🔧 Solution:")
                print("   Re-export with nms=True parameter:")
                print(f"   python3 export_onnx_with_nms.py <model.pt>")
                return False
        else:
            print(f"\n⚠️  Unexpected number of outputs: {len(result)}")
            print("   Expected 1 output tensor")
            return False
            
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify ONNX export quality')
    parser.add_argument(
        'onnx_file',
        help='Path to ONNX file to verify'
    )
    
    args = parser.parse_args()
    
    # Verify
    success = verify_onnx_export(args.onnx_file)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

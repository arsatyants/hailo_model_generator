#!/usr/bin/env python3
"""
Compile ONNX to HEF for Hailo-8 AI Accelerator

Complete compilation pipeline:
1. ONNX → Hailo Model (parse, optimize, quantize)
2. Apply optimization script (enable_loose_mode, optimization_level=0)
3. Compile to HEF binary

Requirements:
- Hailo Dataflow Compiler SDK installed (.venv_hailo_full)
- ONNX model with NMS from STEP 3
- Calibration dataset (50-100 images, UINT8 format)
"""

import sys
import argparse
from pathlib import Path
import subprocess
import os

# Critical: Use Hailo venv for compilation
# This path should point to your Hailo SDK installation
HAILO_VENV = Path(__file__).parent / '.venv_hailo_full'


def check_hailo_sdk():
    """Verify Hailo SDK installation"""
    if not HAILO_VENV.exists():
        print(f"❌ Error: Hailo SDK venv not found: {HAILO_VENV}")
        print("\n🔧 Solution:")
        print("   Install Hailo Dataflow Compiler SDK:")
        print("   https://hailo.ai/developer-zone/software-downloads/")
        print("   Extract to: hailo-compile/.venv_hailo_full/")
        return False
    
    activate_script = HAILO_VENV / 'bin' / 'activate'
    if not activate_script.exists():
        print(f"❌ Error: Hailo venv activation script not found")
        return False
    
    return True


def prepare_calibration_data(calib_dir, num_samples=64):
    """
    Prepare calibration dataset in correct format
    
    Args:
        calib_dir: Path to calibration images directory
        num_samples: Number of images to use (default 64)
        
    Returns:
        Path to prepared calibration data, or None on error
    """
    calib_dir = Path(calib_dir)
    
    if not calib_dir.exists():
        print(f"❌ Error: Calibration directory not found: {calib_dir}")
        print("\n📁 Expected structure:")
        print("   calibration_data/")
        print("   ├── image_001.jpg")
        print("   ├── image_002.jpg")
        print("   └── ...")
        return None
    
    # Find calibration files (NPY or image files)
    calib_files = list(calib_dir.glob('*.npy'))
    if not calib_files:
        # Fall back to image files
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            calib_files.extend(calib_dir.glob(ext))
    
    if len(calib_files) < num_samples:
        print(f"⚠️  Warning: Found only {len(calib_files)} calibration files, need {num_samples}")
        print(f"   Using {len(calib_files)} files for calibration")
        num_samples = len(calib_files)
    
    print(f"📊 Calibration dataset: {num_samples} files from {calib_dir}")
    
    return calib_dir


def compile_onnx_to_hef(onnx_path, output_name, calib_dir, hailo_venv):
    """
    Compile ONNX to HEF using Hailo SDK
    
    This uses the Hailo Dataflow Compiler which must be run
    in the .venv_hailo_full environment (separate from main .venv)
    
    Args:
        onnx_path: Path to ONNX file
        output_name: Name for output HEF file
        calib_dir: Path to calibration images
        hailo_venv: Path to Hailo SDK venv
        
    Returns:
        Path to generated HEF file, or None on error
    """
    onnx_path = Path(onnx_path).resolve()
    # Save HEF to models directory
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    output_hef = (models_dir / f"{output_name}.hef").resolve()
    optimization_script = Path(__file__).parent / 'hailo_optimization.mscript'
    
    # Create Python script for Hailo SDK
    compile_script = Path(__file__).parent / '_compile_temp.py'
    
    compile_code = f'''
import sys
from pathlib import Path
from hailo_sdk_client import ClientRunner

# Paths (all as Path objects to avoid AttributeError)
onnx_path = Path(r"{onnx_path}")
calib_dir = Path(r"{calib_dir}")
output_hef = Path(r"{output_hef}")
optimization_script = Path(r"{optimization_script}")

print("="*60)
print("Hailo Dataflow Compiler")
print("="*60)
print(f"ONNX: {{onnx_path}}")
print(f"Calibration: {{calib_dir}}")
print(f"Output: {{output_hef}}")
print("="*60)

try:
    # Initialize runner
    runner = ClientRunner(hw_arch='hailo8')
    
    # Step 1: Parse ONNX
    print("\\n[1/4] Parsing ONNX model...")
    # Use Conv output nodes recommended by Hailo SDK for HailoRT post-processing
    end_nodes = ['/model.22/cv2.0/cv2.0.2/Conv', '/model.22/cv3.0/cv3.0.2/Conv', \\
                 '/model.22/cv2.1/cv2.1.2/Conv', '/model.22/cv3.1/cv3.1.2/Conv', \\
                 '/model.22/cv2.2/cv2.2.2/Conv', '/model.22/cv3.2/cv3.2.2/Conv']
    hn = runner.translate_onnx_model(
        str(onnx_path),
        'yolov8',
        start_node_names=['images'],
        end_node_names=end_nodes,
        net_input_shapes={{'images': [1, 3, 640, 640]}}
    )
    print("   ✅ ONNX parsed")
    
    # Step 2: Optimize
    print("\\n[2/4] Optimizing model...")
    # Skip optimization script for now - use SDK defaults
    print("   ✅ Using SDK default optimization settings")
    
    # Step 3: Quantize (calibration)
    print("\\n[3/4] Quantizing model (calibration)...")
    calib_files = list(calib_dir.glob('*.npy'))
    if not calib_files:
        calib_files = list(calib_dir.glob('*.jpg'))
    print(f"   Using {{len(calib_files)}} calibration files")
    runner.optimize(str(calib_dir))
    print("   ✅ Quantization complete")
    
    # Step 4: Compile to HEF
    print("\\n[4/4] Compiling to HEF...")
    hef = runner.compile()
    
    # Save HEF
    with open(str(output_hef), 'wb') as f:
        f.write(hef)
    
    print("="*60)
    print("✅ COMPILATION SUCCESSFUL")
    print("="*60)
    print(f"HEF file: {{output_hef}}")
    print(f"Size: {{output_hef.stat().st_size / 1024 / 1024:.1f}} MB")
    print("="*60)
    
    sys.exit(0)
    
except Exception as e:
    print(f"\\n❌ Compilation failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    with open(compile_script, 'w') as f:
        f.write(compile_code)
    
    print("\n🔧 Running Hailo Dataflow Compiler...")
    print("   (This may take 5-15 minutes)")
    print()
    
    # Run in Hailo venv
    activate_cmd = f"source {hailo_venv}/bin/activate"
    python_cmd = f"{hailo_venv}/bin/python3 {compile_script}"
    
    result = subprocess.run(
        f"{activate_cmd} && {python_cmd}",
        shell=True,
        executable='/bin/bash',
        capture_output=False
    )
    
    # Cleanup temp script
    if compile_script.exists():
        compile_script.unlink()
    
    if result.returncode != 0:
        print("\n❌ Compilation failed")
        return None
    
    if not output_hef.exists():
        print(f"\n❌ HEF file not created: {output_hef}")
        return None
    
    return output_hef


def main():
    parser = argparse.ArgumentParser(
        description='Compile ONNX to HEF for Hailo-8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic compilation
  python3 compile_to_hef.py ../step3_onnx_export/best_nms.onnx

  # With custom output name
  python3 compile_to_hef.py best_nms.onnx --output drone_detector

  # With custom calibration data
  python3 compile_to_hef.py best_nms.onnx --calib ./my_calib_data/
        '''
    )
    
    parser.add_argument(
        'onnx_file',
        help='Path to ONNX file from STEP 3'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='model',
        help='Output HEF name (default: model)'
    )
    
    parser.add_argument(
        '--calib',
        default='./calibration_data',
        help='Path to calibration images directory (default: ./calibration_data)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=64,
        help='Number of calibration samples (default: 64)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Compile ONNX to HEF for Hailo-8")
    print("="*60)
    
    # Check Hailo SDK
    if not check_hailo_sdk():
        return 1
    
    print("✅ Hailo SDK found")
    
    # Check ONNX file
    onnx_path = Path(args.onnx_file)
    if not onnx_path.exists():
        print(f"❌ Error: ONNX file not found: {onnx_path}")
        return 1
    
    print(f"✅ ONNX file: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Prepare calibration data
    calib_dir = prepare_calibration_data(args.calib, args.num_samples)
    if calib_dir is None:
        return 1
    
    print("✅ Calibration data ready")
    
    # Compile
    print("\n" + "="*60)
    hef_path = compile_onnx_to_hef(
        onnx_path,
        args.output,
        calib_dir,
        HAILO_VENV
    )
    
    if hef_path is None:
        return 1
    
    # Success
    print("\n" + "="*60)
    print("🎉 HEF COMPILATION COMPLETE")
    print("="*60)
    print(f"\n📦 Output file: {hef_path}")
    print(f"   Size: {hef_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n💡 Next steps:")
    print("   1. Copy HEF to Raspberry Pi:")
    print(f"      scp {hef_path} pi@<pi_ip>:~/models/")
    print("   2. Run inference:")
    print("      python3 hailo_inference.py --model ~/models/model.hef")
    print()
    
    return 0


if __name__ == '__main__':
    exit(main())

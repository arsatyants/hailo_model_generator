#!/usr/bin/env bash
#
# Complete MODEL-GEN Pipeline
# Runs all 5 steps to generate and test HEF model
#
# Usage:
#   ./run_pipeline.sh [dataset_dir] [output_name]
#
# Example:
#   ./run_pipeline.sh datasets drone_detector
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Banner
echo "=================================================================="
echo "MODEL-GEN: Complete YOLOv8 → Hailo HEF Pipeline"
echo "=================================================================="
echo ""

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [dataset_dir] [output_name]"
    echo ""
    echo "Arguments:"
    echo "  dataset_dir   Path to YOLO dataset (default: datasets/)"
    echo "  output_name   Name for output HEF file (default: model)"
    echo ""
    echo "Example:"
    echo "  $0 datasets drone_detector"
    echo "  $0  # Uses defaults: datasets/, model"
    exit 1
fi

DATASET_DIR="${1:-datasets}"
OUTPUT_NAME="${2:-model}"

# Verify dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}❌ Error: Dataset directory not found: $DATASET_DIR${NC}"
    exit 1
fi

# Check for data.yaml
if [ ! -f "$DATASET_DIR/data.yaml" ]; then
    echo -e "${RED}❌ Error: data.yaml not found in $DATASET_DIR${NC}"
    echo "Expected structure:"
    echo "  $DATASET_DIR/"
    echo "  ├── data.yaml"
    echo "  ├── train/images/"
    echo "  ├── train/labels/"
    echo "  ├── val/images/"
    echo "  └── val/labels/"
    exit 1
fi

echo -e "${GREEN}✅ Dataset found: $DATASET_DIR${NC}"
echo -e "${GREEN}✅ Output name: $OUTPUT_NAME${NC}"
echo ""

# Ask for confirmation
read -p "Start complete pipeline? This may take 30-60 minutes. [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

echo ""
echo "=================================================================="
echo "PIPELINE OVERVIEW"
echo "=================================================================="
echo "STEP 1: Data preparation (skip if already annotated)"
echo "STEP 2: Training YOLOv8 model (~20-30 minutes)"
echo "STEP 3: Export to ONNX with NMS (~30 seconds)"
echo "STEP 4: Compile to HEF for Hailo (~10 minutes)"
echo "STEP 5: Copy to models/ directory for deployment"
echo "=================================================================="
echo ""

# Step 1: Data Preparation (optional - skip if already annotated)
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 1: Data Preparation${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
read -p "Run data verification? (check annotations) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd step1_data_preparation
    python3 verify_annotations.py "$DATASET_DIR"
    cd ..
fi
echo ""

# Step 2: Training
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 2: Training YOLOv8 Model${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "This will take 20-30 minutes..."
echo ""

cd step2_training
python3 train_yolov8.py "$DATASET_DIR/data.yaml" --name "$OUTPUT_NAME" --epochs 100

# Find the best.pt file
BEST_PT=$(find runs/detect -name "best.pt" | tail -n 1)
if [ ! -f "$BEST_PT" ]; then
    echo -e "${RED}❌ Error: Training failed - best.pt not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Training complete: $BEST_PT${NC}"
cd ..
echo ""

# Step 3: ONNX Export
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 3: Export to ONNX with NMS${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd step3_onnx_export
python3 export_onnx_with_nms.py "../step2_training/$BEST_PT" --output "${OUTPUT_NAME}_nms.onnx"

ONNX_FILE="${OUTPUT_NAME}_nms.onnx"
if [ ! -f "$ONNX_FILE" ]; then
    echo -e "${RED}❌ Error: ONNX export failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ ONNX export complete: $ONNX_FILE${NC}"

# Verify ONNX
echo ""
echo "Verifying ONNX quality (black image test)..."
python3 verify_onnx_export.py "$ONNX_FILE"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: ONNX verification failed${NC}"
    echo "Re-run export manually and check for false positives."
    exit 1
fi

cd ..
echo ""

# Step 4: HEF Compilation
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 4: Compile to HEF for Hailo-8${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd step4_hef_compilation

# Prepare calibration data
if [ ! -d "calibration_data" ] || [ -z "$(ls -A calibration_data)" ]; then
    echo "Preparing calibration dataset..."
    python3 prepare_calibration.py "$DATASET_DIR" --num-samples 64
fi

# Compile
echo ""
echo "Compiling to HEF (this will take 10-15 minutes)..."
python3 compile_to_hef.py "../step3_onnx_export/$ONNX_FILE" --output "$OUTPUT_NAME"

HEF_FILE="${OUTPUT_NAME}.hef"
if [ ! -f "$HEF_FILE" ]; then
    echo -e "${RED}❌ Error: HEF compilation failed${NC}"
    exit 1
fi

cd ..

# Step 5: Copy to models directory
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 5: Copy HEF to models/ directory${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

mkdir -p models
cp "step4_hef_compilation/$HEF_FILE" "models/$HEF_FILE"
echo -e "${GREEN}✅ HEF copied to: models/$HEF_FILE${NC}"

# Success
echo ""
echo "=================================================================="
echo -e "${GREEN}🎉 PIPELINE COMPLETE - Model Ready for Deployment${NC}"
echo "=================================================================="
echo ""
echo "📦 Generated files:"
echo "   PyTorch:  step2_training/$BEST_PT"
echo "   ONNX:     step3_onnx_export/$ONNX_FILE"
echo "   HEF:      step4_hef_compilation/$HEF_FILE"
echo "   DEPLOY:   models/$HEF_FILE  ← Use this for deployment"
echo ""
echo "📊 Model info:"
HEF_SIZE=$(stat -f%z "models/$HEF_FILE" 2>/dev/null || stat -c%s "models/$HEF_FILE")
HEF_SIZE_MB=$(echo "scale=1; $HEF_SIZE / 1024 / 1024" | bc)
echo "   HEF size: ${HEF_SIZE_MB} MB"
echo ""
echo "💡 Deploy to Raspberry Pi 5:"
echo "   cd step5_raspberry_pi_testing"
echo "   ./deploy_to_pi.sh <pi_ip> pi"
echo ""
echo "   Example:"
echo "   cd step5_raspberry_pi_testing"
echo "   ./deploy_to_pi.sh 192.168.3.205 pi"
echo ""
echo "=================================================================="

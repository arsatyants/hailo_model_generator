#!/usr/bin/env bash
#
# Deploy HEF model to Raspberry Pi and run tests
#
# Usage:
#   ./deploy_to_pi.sh <pi_ip> <pi_user>
#
# Example:
#   ./deploy_to_pi.sh 192.168.3.205 pi
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=================================================================="
echo "Deploy and Test Hailo Model on Raspberry Pi 5"
echo "=================================================================="
echo ""

# Interactive mode if no arguments provided
if [ "$#" -lt 2 ]; then
    echo -e "${CYAN}Interactive Deployment Mode${NC}"
    echo ""
    
    # Prompt for IP address
    echo -n "Enter Raspberry Pi IP address (e.g., 192.168.1.100): "
    read PI_IP
    
    if [ -z "$PI_IP" ]; then
        echo -e "${RED}❌ Error: IP address cannot be empty${NC}"
        exit 1
    fi
    
    # Prompt for username with default
    echo -n "Enter username (default: pi): "
    read PI_USER
    PI_USER=${PI_USER:-pi}
    
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "  IP Address: $PI_IP"
    echo "  Username:   $PI_USER"
    echo ""
    echo -n "Continue with deployment? (y/n): "
    read CONFIRM
    
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
    echo ""
else
    PI_IP="$1"
    PI_USER="$2"
fi

PI_HOST="${PI_USER}@${PI_IP}"

echo -e "${GREEN}Target: ${PI_HOST}${NC}"
echo ""

# Find HEF file
HEF_FILE=$(find ../step4_hef_compilation -name "*.hef" -type f | head -n 1)

if [ -z "$HEF_FILE" ]; then
    echo -e "${RED}❌ Error: No HEF file found in step4_hef_compilation/${NC}"
    echo ""
    echo "Please compile your model first:"
    echo "  cd ../step4_hef_compilation"
    echo "  python3 compile_to_hef.py ../step3_onnx_export/your_model.onnx"
    exit 1
fi

HEF_NAME=$(basename "$HEF_FILE")
echo -e "${GREEN}✅ Found HEF file: $HEF_NAME${NC}"
echo ""

# Test SSH connection
echo "🔌 Testing connection to Raspberry Pi..."
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${PI_HOST}" "echo 'SSH OK'" > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Cannot connect to ${PI_HOST}${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check Pi is powered on and connected to network"
    echo "  2. Verify IP address: ping ${PI_IP}"
    echo "  3. Check SSH is enabled on Pi"
    echo "  4. Try manual SSH: ssh ${PI_HOST}"
    exit 1
fi
echo -e "${GREEN}✅ SSH connection OK${NC}"
echo ""

# Create directories on Pi
echo "📁 Creating directories on Pi..."
ssh "${PI_HOST}" "mkdir -p ~/MODEL-GEN/models ~/MODEL-GEN/test_results ~/MODEL-GEN/scripts"
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Copy HEF file
echo "📦 Copying HEF model to Pi..."
scp "$HEF_FILE" "${PI_HOST}:~/MODEL-GEN/models/"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ HEF file copied${NC}"
else
    echo -e "${RED}❌ Failed to copy HEF file${NC}"
    exit 1
fi
echo ""

# Copy inference script
echo "📄 Copying inference script to Pi..."
scp hailo_detect_live.py "${PI_HOST}:~/MODEL-GEN/scripts/"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Script copied${NC}"
else
    echo -e "${RED}❌ Failed to copy script${NC}"
    exit 1
fi
echo ""

# Copy requirements
echo "📄 Copying requirements.txt..."
scp requirements.txt "${PI_HOST}:~/MODEL-GEN/scripts/"
echo ""

# Check Hailo installation on Pi
echo "🔍 Checking Hailo installation on Pi..."
HAILO_CHECK=$(ssh "${PI_HOST}" "python3 -c 'import hailo_platform; print(\"OK\")' 2>/dev/null || echo 'FAIL'")

if [ "$HAILO_CHECK" = "OK" ]; then
    echo -e "${GREEN}✅ HailoRT is installed${NC}"
else
    echo -e "${YELLOW}⚠️  HailoRT not found, installing...${NC}"
    ssh "${PI_HOST}" "sudo apt update && sudo apt install -y python3-hailort"
fi
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies on Pi..."
ssh "${PI_HOST}" "cd ~/MODEL-GEN/scripts && pip3 install -r requirements.txt --break-system-packages" 2>/dev/null || \
ssh "${PI_HOST}" "cd ~/MODEL-GEN/scripts && pip3 install opencv-python numpy --break-system-packages"
echo ""

# Make script executable
ssh "${PI_HOST}" "chmod +x ~/MODEL-GEN/scripts/hailo_detect_live.py"

# Print usage instructions
echo "=================================================================="
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE${NC}"
echo "=================================================================="
echo ""
echo "Files on Raspberry Pi:"
echo "  Model:  ~/MODEL-GEN/models/${HEF_NAME}"
echo "  Script: ~/MODEL-GEN/scripts/hailo_detect_live.py"
echo "  Output: ~/MODEL-GEN/test_results/"
echo ""
echo -e "${CYAN}Quick Start - Test with Webcam via SSH:${NC}"
echo ""
echo "  1. SSH to Pi and run headless mode (recommended for SSH):"
echo -e "     ${YELLOW}ssh ${PI_HOST}${NC}"
echo -e "     ${YELLOW}cd ~/MODEL-GEN/scripts${NC}"
echo -e "     ${YELLOW}python3 hailo_detect_live.py --model ../models/${HEF_NAME} --headless${NC}"
echo ""
echo -e "${BLUE}Other Options:${NC}"
echo ""
echo "  2. With USB webcam (interactive, requires X11 forwarding):"
echo "     python3 hailo_detect_live.py --model ../models/${HEF_NAME}"
echo ""
echo "  3. With Raspberry Pi Camera Module:"
echo "     python3 hailo_detect_live.py --model ../models/${HEF_NAME} --picamera --headless"
echo ""
echo "  4. Adjust detection thresholds:"
echo "     python3 hailo_detect_live.py --model ../models/${HEF_NAME} --headless --conf 0.35 --iou 0.5"
echo ""
echo -e "${CYAN}Controls during inference:${NC}"
echo "  Ctrl+C     - Stop and show statistics"
echo "  's' key    - Save current frame (if display enabled)"
echo "  'r' key    - Toggle recording mode"
echo "  'q' key    - Quit"
echo ""
echo -e "${GREEN}Headless mode benefits:${NC}"
echo "  • Works perfectly over SSH (no display needed)"
echo "  • Auto-saves detection frames to ~/MODEL-GEN/test_results/"
echo "  • Shows FPS and detection stats in terminal"
echo "  • Uses /dev/video0 webcam automatically"
echo ""
echo "=================================================================="
echo ""
echo -n "Would you like to SSH to Pi now? (y/n): "
read SSH_NOW

if [[ "$SSH_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${CYAN}Connecting to ${PI_HOST}...${NC}"
    ssh -t "${PI_HOST}" "cd ~/MODEL-GEN/scripts && bash -l"
fi

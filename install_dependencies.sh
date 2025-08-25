#!/bin/bash
# Installation script for wolverines pelage sorting project
# This script handles complex dependencies that are difficult to specify in requirements.txt

set -e  # Exit on error

echo "=== Installing dependencies for wolverines pelage sorting ==="

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch"
    CUDA_VERSION="cpu"
fi

# Install PyTorch with appropriate CUDA version
echo "=== Installing PyTorch ==="
if [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "cpu" ]]; then
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing default PyTorch (should work for most CUDA versions)"
    pip install torch==2.7.1 torchvision==0.22.1
fi

# Install other requirements
echo "=== Installing other dependencies ==="
pip install -r requirements.txt

# Install transformers from git (required for DINOv3)
echo "=== Installing transformers from source ==="
pip install git+https://github.com/huggingface/transformers.git

# Verify installation
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo "=== Installation completed successfully! ==="
echo ""
echo "Usage:"
echo "  1. Activate your environment: mamba activate wolverines"
echo "  2. Run installation: bash install_dependencies.sh"
echo "  3. Test installation: python -c 'import torch; print(torch.cuda.is_available())'"
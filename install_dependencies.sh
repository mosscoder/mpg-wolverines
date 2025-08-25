#!/bin/bash
set -e  # Exit on error

echo "=== Installing wolverines pelage sorting workflow ==="
echo "This installation requires GPU/CUDA support"

# Check if mamba is available
if ! command -v mamba &> /dev/null; then
    echo "ERROR: mamba not found. Please install mamba/conda first."
    exit 1
fi

# Create or update mamba environment
echo "=== Creating mamba environment ==="
if mamba env list | grep -q "wolverines"; then
    echo "Environment 'wolverines' already exists. Removing and recreating..."
    mamba env remove -n wolverines -y
fi

echo "Creating new mamba environment with Python 3.12..."
mamba create -n wolverines python=3.12 -y

# Note: Using 'mamba run' to execute commands in the wolverines environment
echo "Environment created. Installing dependencies in wolverines environment..."

# Install PyTorch with CUDA 12.1 (standard for modern clusters)
echo "=== Installing PyTorch with CUDA 12.1 ==="
mamba run -n wolverines pip install torch==2.7.1 torchvision==0.22.1

# Install other requirements
echo "=== Installing other dependencies ==="
mamba run -n wolverines pip install -r requirements.txt

# Install transformers from git (required for DINOv3)
echo "=== Installing transformers from source ==="
mamba run -n wolverines pip install git+https://github.com/huggingface/transformers.git

# Verify installation
echo "=== Verifying installation ==="
mamba run -n wolverines python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

mamba run -n wolverines python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
mamba run -n wolverines python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo "=== Installation completed successfully! ==="
#!/bin/bash
# Installation script for wolverines pelage sorting project
# This script handles environment creation and complex dependencies
# REQUIRES GPU/CUDA - designed for compute clusters with guaranteed GPU access

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

# Activate environment
echo "Activating wolverines environment..."
eval "$(mamba shell.bash hook)"
mamba activate wolverines

# Verify CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This installation requires CUDA/GPU support."
    echo "This script is designed for compute clusters with guaranteed GPU access."
    exit 1
fi

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
echo "Detected CUDA version: $CUDA_VERSION"

# Install PyTorch with appropriate CUDA version
echo "=== Installing PyTorch with CUDA support ==="
if [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.0" ]]; then
    # Fallback to 12.1 for CUDA 12.0
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
else
    # Default to CUDA 12.1 for other/newer CUDA versions
    echo "Installing PyTorch with CUDA 12.1 (compatible with most modern clusters)"
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
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

# Verify CUDA is actually available in PyTorch
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✓ CUDA is available in PyTorch"
    python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
else
    echo "ERROR: CUDA is not available in PyTorch. Installation failed."
    echo "This could indicate a PyTorch/CUDA version mismatch."
    exit 1
fi

python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo "=== Installation completed successfully! ==="
echo "✓ Mamba environment 'wolverines' created with Python 3.12"
echo "✓ GPU/CUDA support verified"
echo ""
echo "Usage:"
echo "  1. Activate your environment: mamba activate wolverines"
echo "  2. Verify GPU access: python -c 'import torch; print(f\"GPUs: {torch.cuda.device_count()}\")'"
echo "  3. Run experiments: cd /path/to/pelage_sorting && sbatch sbatch/00_sweep_image_size.sbatch"
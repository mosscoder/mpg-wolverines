# Installation Troubleshooting Guide

## Common Issues and Solutions

### 1. PyTorch Installation Fails

**Error**: `Could not find a version that satisfies the requirement torch==2.7.1`

**Solutions**:
```bash
# Check available PyTorch versions
pip index versions torch

# Install latest stable instead
pip install torch torchvision

# Or specify CUDA version explicitly
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
```

### 2. NumPy Version Conflicts

**Error**: `numpy==2.3.2 not found` or version conflicts

**Solution**:
```bash
# Install compatible NumPy version
pip install "numpy>=1.24.0,<2.0.0"
```

### 3. Scikit-learn Version Issues

**Error**: `scikit-learn==1.7.1 not found`

**Solution**:
```bash
# Install latest stable version
pip install "scikit-learn>=1.3.0"
```

### 4. Transformers Installation from Git Fails

**Error**: Git clone fails or network issues

**Solutions**:
```bash
# Try with different protocol
pip install git+https://github.com/huggingface/transformers.git

# Or install stable version first, then upgrade
pip install transformers
pip install --upgrade git+https://github.com/huggingface/transformers.git

# If behind corporate firewall
pip install transformers  # Use stable version instead
```

### 5. CUDA Version Detection Issues

**Error**: Wrong CUDA version detected or PyTorch can't find GPUs

**Solutions for GPU Clusters**:
```bash
# Check your CUDA version
nvidia-smi
nvcc --version

# Install PyTorch for your specific CUDA version
# For CUDA 11.8:
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (most common):
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# Verify GPU access:
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### 6. Minimal Installation for GPU Clusters

Install only essential packages with GPU support:
```bash
# Essential with CUDA 12.1 (adjust CUDA version as needed)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
pip install datasets transformers  # Core ML packages
pip install numpy pandas matplotlib scikit-learn  # Basic data science
pip install pillow  # Image processing
```

Then install additional packages as needed:
```bash
pip install scipy seaborn  # For figure generation
```

### 7. Environment Issues

**Problem**: Package conflicts or broken environment on GPU cluster

**Solution**: Start fresh with GPU support
```bash
# Remove old environment
mamba env remove -n wolverines

# Create new environment
mamba create -n wolverines python=3.12
mamba activate wolverines

# Install with GPU support (adjust CUDA version as needed)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
pip install datasets transformers numpy pandas matplotlib scikit-learn pillow
```

### 8. Check Installation on GPU Cluster

Verify everything works with GPU support:
```bash
python -c "
import torch
import torchvision
import transformers
import datasets
import numpy as np
print('✓ All core packages imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Transformers: {transformers.__version__}')

# Test GPU access
if torch.cuda.is_available():
    print('✓ GPU access verified')
    print(f'Current GPU: {torch.cuda.get_device_name()}')
else:
    print('✗ ERROR: GPU access failed')
    exit(1)
"
```

### 9. HuggingFace Authentication (if needed)

```bash
# If you get authentication errors when loading datasets
pip install huggingface_hub
huggingface-cli login
# Then enter your HF token
```

## Quick Fix Commands for GPU Clusters

If you just need to get running quickly on a GPU cluster:

```bash
# Minimal working installation with GPU support
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets numpy pandas matplotlib scikit-learn pillow

# Test GPU access works
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# If that fails, verify GPU is detected at system level
nvidia-smi
```
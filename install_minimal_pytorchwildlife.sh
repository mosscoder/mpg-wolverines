#!/bin/bash
# Minimal installation script for PytorchWildlife with MegaDetectorV6
# Only installs what hugging_face_dataset/v2/scripts/01_run_megadetector.py needs.
# Run inside a fresh environment (see README.md, Environment).
# Uses MIT-licensed YOLOv9 models to avoid YOLOv5 dependencies

set -e

echo "Installing minimal PytorchWildlife environment..."

# Core PytorchWildlife (without dependencies to avoid conflicts)
echo "Installing PytorchWildlife from GitHub..."
pip install --no-deps git+https://github.com/microsoft/CameraTraps.git

# ALL required dependencies for PytorchWildlife
echo "Installing ALL PytorchWildlife dependencies..."
pip install supervision==0.23.0
pip install torch torchvision
pip install ultralytics
pip install Pillow numpy pandas tqdm
pip install wget chardet timm

# For the wolverines dataset
echo "Installing dataset dependencies..."
pip install datasets

echo "Installation complete!"
echo "Run: python hugging_face_dataset/v2/scripts/01_run_megadetector.py --help"
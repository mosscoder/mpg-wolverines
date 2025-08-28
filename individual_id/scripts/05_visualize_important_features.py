#!/usr/bin/env python3
"""
Script 05: Visualize Important Features
Create visualization showing most important features for individual ID classification.
Shows spatial activation of top 3 discriminative features for each wolverine.
"""

import sys
import os
import json
import torch
import numpy as np
from PIL import Image
import random

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(wolverines_root)

from utils.dataset import load_wolverines_dataset, set_all_seeds
from utils.preprocessing import get_height_crop_and_resize_transform
from utils.models import create_model
from utils.individual_id import get_qualified_individuals, select_test_images_per_individual, load_model_checkpoint
from utils.feature_viz import (
    extract_top_features, 
    extract_dinov3_patch_features, 
    create_feature_visualization_grid
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize important features for individual ID')
    parser.add_argument('--model_path', type=str,
                       default='models/individual_id_final.pth',
                       help='Path to trained model')
    parser.add_argument('--output_path', type=str, 
                       default='figures/feature_importance_grid.png',
                       help='Output path for visualization')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for test image selection')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Individual ID Feature Importance Visualization")
    print("=" * 80)
    
    # Set device
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seed
    set_all_seeds(args.seed)
    
    # Load trained model using shared utility
    print("Loading trained model...")
    checkpoint, individual_ids, linear_weights = load_model_checkpoint(args.model_path, device)
    
    # Create model for feature extraction
    num_classes = len(individual_ids)
    model = create_model(device=device, num_classes=num_classes)
    
    # Load the linear layer weights into the model
    linear_layer = model.classifier[-1]
    linear_layer.weight.data = linear_weights.to(device)
    if checkpoint['linear_bias'] is not None:
        linear_layer.bias.data = checkpoint['linear_bias'].to(device)
    
    model.eval()
    
    # Extract top features using shared utility
    print("Extracting most important features...")
    top_feature_indices, top_feature_values = extract_top_features(linear_weights, top_k=3)
    
    # Load test dataset
    print("Loading test dataset...")
    _, test_dataset = load_wolverines_dataset()
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Select test images using shared utility
    print("Selecting test images...")
    selected_images = select_test_images_per_individual(test_dataset, individual_ids, seed=args.seed)
    
    if not selected_images:
        print("Error: No test images selected")
        return
    
    print(f"Selected {len(selected_images)} test images")
    
    # Create transform
    transform = get_height_crop_and_resize_transform(height=1280, resize=728)
    
    # Create visualization using shared utility
    print("Creating visualization grid...")
    create_feature_visualization_grid(
        selected_images=selected_images,
        individual_ids=individual_ids,
        top_feature_indices=top_feature_indices,
        model=model,
        transform=transform,
        device=device,
        output_path=args.output_path,
        patch_grid_size=(45, 45),  # For 728x728 images with 16x16 patches
        target_size=(728, 728)
    )
    
    # Print summary
    print("\nVisualization Summary:")
    print(f"Individuals: {len(individual_ids)}")
    print(f"Top features: {top_feature_indices}")
    print(f"Output: {args.output_path}")
    
    print("\nFeature importance values:")
    for i, (idx, val) in enumerate(zip(top_feature_indices, top_feature_values)):
        print(f"  Feature {idx}: {val:.4f}")

if __name__ == "__main__":
    main()
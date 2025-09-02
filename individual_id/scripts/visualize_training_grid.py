#!/usr/bin/env python3
"""
Script: Visualize Training Grid
Creates a 2x5 grid visualization of 10 random training images with center crop transform.
"""

import sys
import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Assume script is run from wolverines root directory
# Add current directory to path for utils
sys.path.append('.')

from utils.dataset import load_wolverines_dataset, set_all_seeds
from utils.preprocessing import get_center_crop_transform
from utils.individual_id import create_temporal_sweep_dataset


def visualize_training_grid(args):
    """Create and save 2x5 grid of random training images"""
    
    # Verify we're in the wolverines root directory
    if not os.path.exists('utils/dataset.py'):
        raise RuntimeError("Please run this script from the wolverines root directory")
    
    # Set seed for reproducibility
    set_all_seeds(args.seed)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, test_dataset = load_wolverines_dataset()
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Load top 3 individuals
    import json
    config_path = 'individual_id/results/feasible_individuals.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    individuals_sorted = config['individuals_sorted_by_pelage']
    feasible_individuals = random.sample(individuals_sorted, min(3, len(individuals_sorted)))
    print(f"Randomly selected 3 individuals: {', '.join(feasible_individuals)}")
    
    # Create temporal datasets for both approaches
    pelage_train_dataset, _, _, _ = create_temporal_sweep_dataset(
        train_dataset, test_dataset, feasible_individuals, 
        sample_size=16, approach='pelage', seed=args.seed
    )
    
    pelage_abs_train_dataset, _, _, _ = create_temporal_sweep_dataset(
        train_dataset, test_dataset, feasible_individuals, 
        sample_size=16, approach='pelage_abs', seed=args.seed
    )
    
    print(f"Pelage visible training samples: {len(pelage_train_dataset)}")
    print(f"Pelage invisible training samples: {len(pelage_abs_train_dataset)}")
    
    # Create transform
    transform = get_center_crop_transform(width=1480, height=1480)
    
    # Randomly select 10 images from each dataset
    n_images_per_type = 10
    
    # Select pelage visible images
    pelage_n = min(n_images_per_type, len(pelage_train_dataset))
    pelage_indices = random.sample(range(len(pelage_train_dataset)), pelage_n)
    
    # Select pelage invisible images
    pelage_abs_n = min(n_images_per_type, len(pelage_abs_train_dataset))
    pelage_abs_indices = random.sample(range(len(pelage_abs_train_dataset)), pelage_abs_n)
    
    print(f"Visualizing {pelage_n} pelage visible + {pelage_abs_n} pelage invisible images")
    
    # Create figure with 4 rows, 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Training Images: Pelage Visible (Top) vs Not Visible (Bottom)', fontsize=16)
    
    # Display pelage visible images (rows 0-1)
    for i, idx in enumerate(pelage_indices):
        row = i // 5
        col = i % 5
        
        # Get sample and apply transform
        sample = pelage_train_dataset[idx]
        image = sample['image']
        individual_id = sample.get('id', 'Unknown')
        pelage = sample.get('pelage', 'Unknown')
        
        # Apply transform
        transformed_image = transform(image)
        
        # Convert to numpy for display
        img_np = transformed_image.permute(1, 2, 0).numpy()
        
        # Denormalize for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Display image
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'{individual_id}\nVisible: {pelage}', fontsize=10)
        axes[row, col].axis('off')
    
    # Display pelage invisible images (rows 2-3)
    for i, idx in enumerate(pelage_abs_indices):
        row = (i // 5) + 2  # Start from row 2
        col = i % 5
        
        # Get sample and apply transform
        sample = pelage_abs_train_dataset[idx]
        image = sample['image']
        individual_id = sample.get('id', 'Unknown')
        pelage = sample.get('pelage', 'Unknown')
        
        # Apply transform
        transformed_image = transform(image)
        
        # Convert to numpy for display
        img_np = transformed_image.permute(1, 2, 0).numpy()
        
        # Denormalize for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Display image
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'{individual_id}\nInvisible: {pelage}', fontsize=10)
        axes[row, col].axis('off')
    
    # Hide any unused subplots
    for i in range(pelage_n, 10):
        row = i // 5
        col = i % 5
        axes[row, col].axis('off')
    
    for i in range(pelage_abs_n, 10):
        row = (i // 5) + 2
        col = i % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'individual_id/results/training_grid_visualization.png'
    os.makedirs('individual_id/results', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Training grid visualization saved to: {output_path}")
    print(f"  Used {pelage_n} pelage visible + {pelage_abs_n} pelage invisible images")
    print(f"  Transform: 1480x1480 center crop, no resize")


def main():
    parser = argparse.ArgumentParser(description='Visualize training images in grid')
    parser.add_argument('--output_dir', type=str,
                       default='individual_id/results',
                       help='Output directory for results (relative to wolverines root)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for image selection')
    
    args = parser.parse_args()
    
    try:
        visualize_training_grid(args)
        print("\n🎯 Training grid visualization completed successfully!")
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Visualize center crop bounding boxes on a pelage image.
Shows 64, 128, and 256 pixel width crops emanating from center of image.
Only crops along x-axis, keeps full height.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(script_dir)
sys.path.append(wolverines_root)

from utils.dataset import load_wolverines_dataset

def find_pelage_image(dataset, max_search=100):
    """Find first image with label==1 (pelage visible)"""
    print("Searching for pelage image (label==1)...")
    
    for i in range(min(len(dataset), max_search)):
        item = dataset[i]
        if item['label'] == 1:
            print(f"Found pelage image at index {i}")
            print(f"  ID: {item['id']}")
            print(f"  Station: {item.get('station', 'unknown')}")
            print(f"  Year: {item.get('year', 'unknown')}")
            return item, i
    
    print(f"No pelage image found in first {max_search} samples")
    return None, -1

def visualize_center_crops(image, crop_widths, output_path):
    """Visualize center crop bounding boxes on image"""
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    img_height, img_width = img_array.shape[:2]
    
    print(f"Image dimensions: {img_width} x {img_height}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Colors for different crop widths
    colors = ['blue', 'cyan', 'lightblue']
    
    # Draw bounding boxes for each crop width
    for i, crop_width in enumerate(crop_widths):
        # Calculate center crop coordinates
        center_x = img_width // 2
        
        # X coordinates for center crop
        x_left = max(0, center_x - crop_width // 2)
        x_right = min(img_width, center_x + crop_width // 2)
        actual_width = x_right - x_left
        
        # Y coordinates (full height)
        y_top = 0
        y_bottom = img_height
        actual_height = y_bottom - y_top
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_left, y_top), 
            actual_width, 
            actual_height,
            linewidth=3,
            edgecolor=colors[i],
            facecolor='none',
            alpha=0.8
        )
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add label
        ax.text(x_left + 10, y_top + 30 + (i * 25), 
                f'{crop_width}px width', 
                color=colors[i], 
                fontsize=12, 
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        print(f"Crop {crop_width}px: x=[{x_left}:{x_right}], actual_width={actual_width}")
    
    # Draw horizontal red lines for height crops from center (728, 1024, and 1280 pixels)
    height_crops = [728, 1024, 1280]
    center_y = img_height // 2
    
    for i, crop_height in enumerate(height_crops):
        # Calculate top and bottom lines from center
        y_top = max(0, center_y - crop_height // 2)
        y_bottom = min(img_height, center_y + crop_height // 2)
        
        # Draw both lines for this height
        colors = ['red', 'darkred', 'maroon']
        line_color = colors[i]
        ax.axhline(y=y_top, color=line_color, linewidth=2, linestyle='--', alpha=0.8)
        ax.axhline(y=y_bottom, color=line_color, linewidth=2, linestyle='--', alpha=0.8)
        
        # Add label at center
        label_x = img_width - 150 - (i * 20)  # Offset labels so they don't overlap
        ax.text(label_x, center_y + (i * 30), f'{crop_height}px height', 
                color=line_color, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        print(f"Height crop {crop_height}px: y=[{y_top}:{y_bottom}], actual_height={y_bottom - y_top}")
    
    # Set title and remove axes
    ax.set_title('Center Crop Visualization - Pelage Frame Detection\n'
                'Blue boxes: widths (256, 512, 728px) | Red lines: heights (728, 1024, 1280px)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    
    # Also save with transparent background
    transparent_path = output_path.replace('.png', '_transparent.png')
    plt.savefig(transparent_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Transparent version saved to: {transparent_path}")
    
    plt.close()

def main():
    print("="*60)
    print("Wolverine Pelage Center Crop Visualization")
    print("="*60)
    
    # Load dataset
    print("Loading wolverines dataset...")
    train_dataset, _ = load_wolverines_dataset()
    print(f"Loaded {len(train_dataset)} training samples")
    
    # Find pelage image
    pelage_item, item_idx = find_pelage_image(train_dataset)
    
    if pelage_item is None:
        print("ERROR: No pelage image found!")
        return
    
    # Get image
    image = pelage_item['image']  # PIL Image from HuggingFace dataset
    
    print(f"\nSelected image details:")
    print(f"  Index: {item_idx}")
    print(f"  Individual ID: {pelage_item['id']}")
    print(f"  Label: {pelage_item['label']} (pelage visible)")
    print(f"  Image size: {image.size}")
    print(f"  Image mode: {image.mode}")
    
    # Define crop widths to visualize
    crop_widths = [256, 512, 728]
    
    # Output path
    output_dir = os.path.join(script_dir, 'results')
    output_path = os.path.join(output_dir, 'bb_viz.png')
    
    print(f"\nVisualizing center crops with widths: {crop_widths}")
    
    # Create visualization
    visualize_center_crops(image, crop_widths, output_path)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Check the output at: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
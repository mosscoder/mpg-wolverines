#!/usr/bin/env python3
"""
Model Comparison Grid Generator
Creates a visual comparison grid showing detection crops from all 5 MegaDetector models.
"""

# Fix OpenMP conflict on macOS before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import json
import tempfile
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Use HuggingFace datasets directly
from datasets import load_dataset


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def crop_image_from_bbox(image, bbox):
    """
    Crop image using bounding box coordinates
    
    Args:
        image: PIL Image (RGB)
        bbox: Bounding box [x, y, width, height]
    
    Returns:
        PIL Image cropped to bounding box or None if invalid
    """
    x, y, width, height = bbox
    
    # Ensure coordinates are within image bounds
    x = max(0, int(x))
    y = max(0, int(y))
    width = min(int(width), image.width - x)
    height = min(int(height), image.height - y)
    
    # Ensure we have valid dimensions
    if width <= 0 or height <= 0:
        return None
        
    return image.crop((x, y, x + width, y + height))


def detect_with_model(image, model_version, confidence_threshold=0.8):
    """
    Run detection on single image with specified model
    
    Args:
        image: PIL Image
        model_version: Model version string
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        dict: {'bbox': [x,y,w,h], 'confidence': float} or None if no detection
    """
    from PytorchWildlife.models import detection as pw_detection
    from PytorchWildlife.data import transforms as pw_transforms
    
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Initialize model
    detection_model = pw_detection.MegaDetectorV6(
        device=device,
        pretrained=True,
        version=model_version
    )
    
    # Set appropriate image size based on model
    if model_version in ['MDV6-yolov9-e', 'MDV6-yolov10-e']:
        # Enhanced models use 1280x1280
        detection_model.IMAGE_SIZE = 1280
        detection_model.predictor.args.imgsz = 1280
        detection_model.transform = pw_transforms.MegaDetector_v5_Transform(
            target_size=1280, stride=32
        )
    else:
        # Compact models use 640x640 (default)
        detection_model.IMAGE_SIZE = 640
        detection_model.predictor.args.imgsz = 640
    
    # Save image temporarily and run detection
    with tempfile.TemporaryDirectory(prefix='model_comparison_') as temp_dir:
        image_path = os.path.join(temp_dir, 'test_image.jpg')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path, 'JPEG', quality=95)
        
        try:
            # Run detection
            result = detection_model.single_image_detection(image_path, det_conf_thres=confidence_threshold)
            
            if 'detections' in result and len(result['detections']) > 0:
                sv_detections = result['detections']
                
                # Find highest confidence animal detection
                best_detection = None
                best_confidence = 0
                
                for j in range(len(sv_detections)):
                    x1, y1, x2, y2 = sv_detections.xyxy[j]
                    conf = float(sv_detections.confidence[j])
                    class_id = int(sv_detections.class_id[j])
                    
                    # Keep only animals (class_id=0) with highest confidence
                    if class_id == 0 and conf > best_confidence:
                        best_confidence = conf
                        best_detection = {
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'confidence': conf
                        }
                
                return best_detection
                
        except Exception as e:
            print(f"Error with model {model_version}: {e}")
            return None
    
    return None


def create_comparison_grid(images, model_versions, confidence_threshold=0.8, crop_size=(300, 300)):
    """
    Create comparison grid showing crops from all models
    
    Args:
        images: List of PIL Images to test
        model_versions: List of model version strings
        confidence_threshold: Detection confidence threshold
        crop_size: Target size for displayed crops
    
    Returns:
        PIL Image: Grid comparison image
    """
    num_images = len(images)
    num_models = len(model_versions)
    
    # Grid dimensions: +1 col for original, +1 row for headers
    grid_cols = num_models + 1  # original + models
    grid_rows = num_images + 1  # header + images
    
    # Create figure
    fig_width = grid_cols * 4
    fig_height = grid_rows * 4
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))
    
    # Handle single row case
    if grid_rows == 1:
        axes = axes.reshape(1, -1)
    if grid_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Header row
    axes[0, 0].text(0.5, 0.5, 'Original\nImage', ha='center', va='center', fontsize=12, weight='bold')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    for col, model_version in enumerate(model_versions):
        axes[0, col + 1].text(0.5, 0.5, model_version.replace('MDV6-', ''), 
                             ha='center', va='center', fontsize=10, weight='bold')
        axes[0, col + 1].set_xticks([])
        axes[0, col + 1].set_yticks([])
    
    # Process each image
    detection_stats = {model: {'detections': 0, 'total_confidence': 0} for model in model_versions}
    
    for row, image in enumerate(tqdm(images, desc="Processing images")):
        row_idx = row + 1  # +1 for header
        
        # Show original image
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"Image {row}", fontsize=10)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        
        # Test each model
        for col, model_version in enumerate(model_versions):
            col_idx = col + 1  # +1 for original
            
            print(f"Testing {model_version} on image {row}...")
            detection = detect_with_model(image, model_version, confidence_threshold)
            
            if detection:
                # Crop and display detection
                crop = crop_image_from_bbox(image, detection['bbox'])
                if crop:
                    # Resize crop for consistent display
                    crop_resized = crop.resize(crop_size, Image.Resampling.LANCZOS)
                    axes[row_idx, col_idx].imshow(crop_resized)
                    axes[row_idx, col_idx].set_title(f"Conf: {detection['confidence']:.3f}", fontsize=9)
                    
                    # Update stats
                    detection_stats[model_version]['detections'] += 1
                    detection_stats[model_version]['total_confidence'] += detection['confidence']
                else:
                    # Invalid bbox
                    axes[row_idx, col_idx].text(0.5, 0.5, 'Invalid\nBBox', ha='center', va='center')
                    axes[row_idx, col_idx].set_facecolor('lightcoral')
            else:
                # No detection
                axes[row_idx, col_idx].text(0.5, 0.5, 'No\nDetection', ha='center', va='center')
                axes[row_idx, col_idx].set_facecolor('lightgray')
            
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
    
    plt.tight_layout()
    return fig, detection_stats


def print_model_statistics(detection_stats, num_images):
    """Print summary statistics for each model"""
    print(f"\n" + "=" * 80)
    print("MODEL COMPARISON STATISTICS")
    print("=" * 80)
    
    for model, stats in detection_stats.items():
        detection_rate = stats['detections'] / num_images
        avg_confidence = stats['total_confidence'] / max(1, stats['detections'])
        
        print(f"{model:20s}: {stats['detections']}/{num_images} detections ({detection_rate:.1%}), "
              f"avg confidence: {avg_confidence:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Compare MegaDetector models with visual grid')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to compare (default: 5)')
    parser.add_argument('--confidence', type=float, default=0.8,
                       help='Confidence threshold for detections (default: 0.8)')
    parser.add_argument('--output', type=str, default='process_masks/results/model_comparison_grid.png',
                       help='Output grid image path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for image selection (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MegaDetector Model Comparison Grid Generator")
    print("=" * 80)
    print(f"Confidence threshold: {args.confidence}")
    print(f"Selection strategy: Highest confidence bounding box")
    
    # Set random seed
    set_all_seeds(args.seed)
    
    # Load dataset
    print("Loading wolverines dataset from HuggingFace...")
    dataset = load_dataset("kdoherty/wolverines", split="train")
    
    # Filter for label 1 images only (wolverine present)
    label_1_indices = [i for i in range(len(dataset)) if dataset[i]['label'] == 1]
    print(f"Found {len(label_1_indices)} images with label=1 (wolverine present)")
    
    # Select random sample from label 1 images
    if args.num_images > len(label_1_indices):
        args.num_images = len(label_1_indices)
    
    indices = random.sample(label_1_indices, args.num_images)
    selected_images = [dataset[i]['image'] for i in indices]
    
    print(f"Selected {len(selected_images)} random images with label=1 (wolverine present)")
    
    # Model versions to compare
    model_versions = [
        'MDV6-yolov9-c',
        'MDV6-yolov9-e', 
        'MDV6-yolov10-c',
        'MDV6-yolov10-e',
        'MDV6-rtdetr-c'
    ]
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate comparison grid
    print(f"\nGenerating comparison grid...")
    start_time = time.time()
    
    fig, stats = create_comparison_grid(selected_images, model_versions, args.confidence)
    
    processing_time = time.time() - start_time
    
    # Save results
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save statistics
    stats_file = args.output.replace('.png', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print_model_statistics(stats, args.num_images)
    
    print(f"\nComparison completed in {processing_time/60:.1f} minutes")
    print(f"Grid saved to: {args.output}")
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
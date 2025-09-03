#!/usr/bin/env python3
"""
Script 02: Generate Segmentation Masks with SAM
Uses Segment Anything Model to create precise wildlife masks from MegaDetector bounding boxes.
"""

# Fix OpenMP conflict on macOS before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import load_wolverines_dataset, set_all_seeds


def load_sam_model(model_type='vit_b', device='cpu'):
    """Load Segment Anything Model"""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
    
    print(f"Loading SAM model ({model_type})...")
    
    # Model checkpoints
    model_urls = {
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    }
    
    model_path = f"process_masks/models/sam_{model_type}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Download model if not exists
    if not os.path.exists(model_path):
        print(f"Downloading SAM {model_type} model...")
        import requests
        response = requests.get(model_urls[model_type], stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        print(f"Model downloaded to: {model_path}")
    
    # Load model
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    print(f"SAM {model_type} loaded on device: {device}")
    return predictor


def generate_mask_from_bbox(predictor, image, bbox,
                           pred_iou_thresh=0.8,
                           stability_score_thresh=0.9,
                           min_mask_region_area=500):
    """
    Generate segmentation mask from bounding box using SAM
    
    Args:
        predictor: SAM predictor object
        image: PIL Image
        bbox: Bounding box [x, y, width, height]
        pred_iou_thresh: IoU threshold for mask quality filtering (lower = more liberal)
        stability_score_thresh: Stability threshold for mask filtering (lower = more liberal)
        min_mask_region_area: Minimum area for mask regions (suppresses small holes)
    
    Returns:
        Binary mask as numpy array (same size as image)
    """
    # Convert PIL to numpy
    image_array = np.array(image)
    
    # Set image for SAM
    predictor.set_image(image_array)
    
    # Use bounding box directly without buffer for precise masks
    x, y, width, height = bbox
    
    # Convert to SAM format (x_min, y_min, x_max, y_max)
    x_min = x
    y_min = y
    x_max = x + width
    y_max = y + height
    
    input_box = np.array([x_min, y_min, x_max, y_max])
    
    # Generate mask with quality scores
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True  # Get multiple masks with scores
    )
    
    # Filter masks by predicted IoU threshold
    if np.max(scores) < pred_iou_thresh:
        print(f"  Warning: Best mask score {np.max(scores):.3f} below threshold {pred_iou_thresh}")
    
    # Select best scoring mask
    best_mask_idx = np.argmax(scores)
    selected_mask = masks[best_mask_idx]
    best_score = scores[best_mask_idx]
    
    # Remove small disconnected regions if specified
    if min_mask_region_area > 0:
        import cv2
        mask_uint8 = selected_mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        
        # Keep only components larger than min_mask_region_area
        filtered_mask = np.zeros_like(selected_mask, dtype=bool)
        for label_id in range(1, num_labels):
            component_mask = labels == label_id
            if np.sum(component_mask) >= min_mask_region_area:
                filtered_mask |= component_mask
        
        selected_mask = filtered_mask
    
    return selected_mask  # Shape: (H, W) boolean array


def apply_mask_to_image(image, mask):
    """
    Apply binary mask to image, setting background to transparent
    
    Args:
        image: PIL Image (RGB)
        mask: Binary numpy array (H, W)
    
    Returns:
        PIL Image with RGBA format (masked background transparent)
    """
    # Convert image to RGBA
    image_rgba = image.convert('RGBA')
    image_array = np.array(image_rgba)
    
    # Apply mask to alpha channel
    image_array[:, :, 3] = mask.astype(np.uint8) * 255
    
    # Create new PIL image
    masked_image = Image.fromarray(image_array, 'RGBA')
    
    return masked_image


def process_detections(dataset, predictor, detections_file, output_dir):
    """Process all detections to generate masks"""
    
    # Load detections
    print(f"Loading detections from: {detections_file}")
    with open(detections_file, 'r') as f:
        detection_data = json.load(f)
    
    detections = detection_data['detections']
    
    # Create output directories
    masks_dir = os.path.join(output_dir, 'masks')
    masked_images_dir = os.path.join(output_dir, 'masked_images')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    
    # Process each image with detections
    mask_results = {}
    processing_stats = {'images_processed': 0, 'masks_generated': 0, 'failed_masks': 0}
    
    print(f"Generating masks for images with wildlife detections...")
    
    # Process all images with detections from the JSON
    images_with_detections = [(k, v) for k, v in detections.items() if v['detections']]
    print(f"Processing {len(images_with_detections)} images with detections")
    
    for image_id, image_data in tqdm(images_with_detections):
        if not image_data['detections']:
            continue  # Skip images without detections
        
        # Get original image
        original_index = image_data['original_index']
        image = dataset[original_index]['image']
        
        # Use the highest confidence detection for this image
        if not image_data['detections']:
            continue
            
        # Sort detections by confidence and use the best one
        best_detection = max(image_data['detections'], key=lambda x: x['confidence'])
        
        try:
            # Generate mask for best detection with quality filtering
            mask = generate_mask_from_bbox(
                predictor, 
                image, 
                best_detection['bbox'],
                pred_iou_thresh=0.8,
                stability_score_thresh=0.9,
                min_mask_region_area=500
            )
            
            # Apply mask to create masked image
            masked_image = apply_mask_to_image(image, mask)
            
            # Save with simple naming that maps to dataset index
            masked_image_filename = f"{image_id}.png"  # Maps directly to dataset row
            mask_filename = f"{image_id}_mask.npy"
            
            # Ensure PNG format with alpha channel (RGBA)
            masked_image.save(os.path.join(masked_images_dir, masked_image_filename), 'PNG')
            np.save(os.path.join(masks_dir, mask_filename), mask)
            
            # Store mask info
            image_masks = [{
                'bbox': best_detection['bbox'],
                'confidence': best_detection['confidence'],
                'mask_file': mask_filename,
                'masked_image_file': masked_image_filename,
                'mask_area': int(np.sum(mask)),
                'image_area': mask.shape[0] * mask.shape[1],
                'coverage_ratio': float(np.sum(mask)) / (mask.shape[0] * mask.shape[1])
            }]
            
            processing_stats['masks_generated'] += 1
            
        except Exception as e:
            print(f"Error generating mask for {image_id}: {e}")
            processing_stats['failed_masks'] += 1
            image_masks = []
        
        # Store results for this image
        mask_results[image_id] = {
            'original_index': original_index,
            'individual_id': image_data['individual_id'],
            'label': image_data['label'],
            'date': image_data['date'],
            'masks': image_masks,
            'has_mask': len(image_masks) > 0
        }
        
        processing_stats['images_processed'] += 1
    
    # Save mask results
    mask_output_file = os.path.join(output_dir, 'mask_results.json')
    final_mask_results = {
        'mask_data': mask_results,
        'processing_stats': processing_stats,
        'model_info': {
            'sam_model': 'vit_b',  # Default model type
            'mask_format': 'numpy_binary_array'
        },
        'file_info': {
            'masks_directory': masks_dir,
            'masked_images_directory': masked_images_dir,
            'timestamp': time.time()
        }
    }
    
    with open(mask_output_file, 'w') as f:
        json.dump(final_mask_results, f, indent=2)
    
    return final_mask_results, mask_output_file


def print_mask_summary(results):
    """Print summary of mask generation results"""
    stats = results['processing_stats']
    mask_data = results['mask_data']
    
    print(f"\n" + "=" * 60)
    print("SAM MASK GENERATION SUMMARY")
    print("=" * 60)
    print(f"Images processed: {stats['images_processed']}")
    print(f"Masks generated: {stats['masks_generated']}")
    print(f"Failed masks: {stats['failed_masks']}")
    
    if stats['masks_generated'] > 0:
        # Calculate coverage statistics
        coverage_ratios = []
        for image_data in mask_data.values():
            for mask_info in image_data['masks']:
                coverage_ratios.append(mask_info['coverage_ratio'])
        
        if coverage_ratios:
            avg_coverage = np.mean(coverage_ratios)
            min_coverage = np.min(coverage_ratios)
            max_coverage = np.max(coverage_ratios)
            
            print(f"Mask coverage statistics:")
            print(f"  Average coverage: {avg_coverage:.3f} ({100*avg_coverage:.1f}% of image)")
            print(f"  Min coverage: {min_coverage:.3f} ({100*min_coverage:.1f}% of image)")
            print(f"  Max coverage: {max_coverage:.3f} ({100*max_coverage:.1f}% of image)")


def main():
    parser = argparse.ArgumentParser(description='Generate segmentation masks using SAM')
    parser.add_argument('--detections_file', type=str, 
                       default='process_masks/results/detections/wildlife_detections.json',
                       help='Path to MegaDetector detection results')
    parser.add_argument('--output_dir', type=str, default='process_masks/results/masks',
                       help='Directory to save mask results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu', 'mps'],
                       default='mps', help='Device to use for SAM')
    parser.add_argument('--model_type', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
                       default='vit_b', help='SAM model size (vit_b is fastest)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SAM Mask Generation Pipeline")
    print("=" * 80)
    
    # Check if detections file exists
    if not os.path.exists(args.detections_file):
        print(f"Error: Detections file not found: {args.detections_file}")
        print("Please run 01_detect_wildlife.py first")
        return
    
    # Set random seed
    set_all_seeds(42)
    
    # Load dataset
    print("Loading wolverines dataset...")
    train_dataset, test_dataset = load_wolverines_dataset()
    dataset = train_dataset  # Use same dataset as detection step
    
    # Setup device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif args.device == "gpu" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load SAM model
    predictor = load_sam_model(args.model_type, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process detections to generate masks
    print(f"\nStarting mask generation...")
    start_time = time.time()
    
    results, output_file = process_detections(dataset, predictor, args.detections_file, args.output_dir)
    
    processing_time = time.time() - start_time
    
    # Print summary
    print_mask_summary(results)
    
    print(f"\nMask generation completed in {processing_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print(f"Next step: Run 03_update_dataset.py to add masks to HuggingFace dataset")


if __name__ == "__main__":
    main()
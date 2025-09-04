#!/usr/bin/env python3
"""
Script 01: Wildlife Detection with MegaDetector
Uses MegaDetector v6 to detect wildlife in images.
Saves bounding box coordinates and optional cropped images.
Includes pre-detection proportional cropping.
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
from PIL import Image
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


def crop_image_proportional(image, top_crop, bottom_crop, left_crop, right_crop):
    """
    Crops an image by a proportion of its width and height.

    Args:
        image (PIL.Image): The input image.
        top_crop (float): Proportion to crop from the top (0.0 to 1.0).
        bottom_crop (float): Proportion to crop from the bottom (0.0 to 1.0).
        left_crop (float): Proportion to crop from the left (0.0 to 1.0).
        right_crop (float): Proportion to crop from the right (0.0 to 1.0).

    Returns:
        tuple: A tuple containing:
        - PIL.Image: The cropped image.
        - tuple: The (x_offset, y_offset) in pixels of the crop.
    """
    width, height = image.size
    left = int(width * left_crop)
    top = int(height * top_crop)
    right = int(width * (1 - right_crop))
    bottom = int(height * (1 - bottom_crop))
    # Ensure crop coordinates are valid
    if left >= right or top >= bottom:
        print("Warning: Invalid crop dimensions, returning original image.")
        return image, (0, 0)
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image, (left, top)


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


def init_detection_model(model_version, device):
    """
    Initialize MegaDetector model once
    """
    from PytorchWildlife.models import detection as pw_detection
    from PytorchWildlife.data import transforms as pw_transforms
    
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
    
    return detection_model


def detect_image(image, detection_model, confidence_threshold=0.5):
    """
    Run detection on single image with pre-initialized model
    
    Args:
        image: PIL Image
        detection_model: Pre-initialized MegaDetector model
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        dict: {'bbox': [x,y,w,h], 'confidence': float} or None if no detection
    """
    # Save image temporarily and run detection
    with tempfile.TemporaryDirectory(prefix='detection_') as temp_dir:
        image_path = os.path.join(temp_dir, 'image.jpg')
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
                            'confidence': conf,
                            'category': 'animal',
                            'class_name': 'animal'
                        }
                
                return best_detection
                
        except Exception as e:
            print(f"Error with detection: {e}")
            return None
    
    return None


def detect_batch(images, detection_model, confidence_threshold=0.5):
    """
    Run detection on batch of images efficiently
    
    Args:
        images: List of PIL Images
        detection_model: Pre-initialized MegaDetector model
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detection results (same order as input images)
    """
    results = []
    
    # Process each image in the batch
    for image in images:
        detection = detect_image(image, detection_model, confidence_threshold)
        results.append(detection)
    
    return results


def process_dataset(dataset, model_version, output_dir, confidence_threshold=0.5, save_crops=True,
                   top_crop=0.0, bottom_crop=0.0, left_crop=0.0, right_crop=0.0, batch_size=16):
    """Process dataset with single model, applying pre-detection crops"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Initializing {model_version} model on {device}...")
    detection_model = init_detection_model(model_version, device)
    
    # Create output directories
    cropped_images_dir = os.path.join(output_dir, 'cropped_images') if save_crops else None
    if save_crops:
        os.makedirs(cropped_images_dir, exist_ok=True)
    
    # Process images
    all_detections = {}
    processing_stats = {'crops_created': 0, 'failed_crops': 0, 'detections': 0}
    
    print(f"Processing {len(dataset)} images with {model_version} (batch size: {batch_size})...")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_indices = list(range(batch_start, batch_end))
        
        # Prepare batch images
        batch_images = []
        batch_samples = []
        batch_ids = []
        
        for idx in batch_indices:
            sample = dataset[idx]
            original_image = sample['image']
            image_id = f"image_{idx:06d}"
            
            # Apply proportional crop if specified before detection
            has_pre_crop = top_crop > 0 or bottom_crop > 0 or left_crop > 0 or right_crop > 0
            if has_pre_crop:
                image_to_process, (x_offset, y_offset) = crop_image_proportional(
                    original_image, top_crop, bottom_crop, left_crop, right_crop
                )
            else:
                image_to_process = original_image
                x_offset, y_offset = 0, 0
            
            batch_images.append(image_to_process)
            batch_samples.append((sample, original_image, x_offset, y_offset, has_pre_crop))
            batch_ids.append(image_id)
        
        # Run detection on batch
        batch_detections = detect_batch(batch_images, detection_model, confidence_threshold)
        
        # Process batch results
        for i, detection in enumerate(batch_detections):
            idx = batch_indices[i]
            sample, original_image, x_offset, y_offset, has_pre_crop = batch_samples[i]
            image_id = batch_ids[i]
            
            # Adjust bounding box coordinates if a pre-detection crop was applied
            if detection and has_pre_crop:
                detection['bbox'][0] += x_offset
                detection['bbox'][1] += y_offset
            
            # Create cropped image from the ORIGINAL image using the adjusted bbox
            cropped_image_file = None
            if detection and save_crops:
                try:
                    # Use original_image here to ensure final crop is from the full source
                    cropped_image = crop_image_from_bbox(original_image, detection['bbox'])
                    if cropped_image is not None:
                        cropped_image_file = f"{image_id}_crop.jpg"
                        crop_path = os.path.join(cropped_images_dir, cropped_image_file)
                        cropped_image.save(crop_path, 'JPEG', quality=95)
                        processing_stats['crops_created'] += 1
                    else:
                        processing_stats['failed_crops'] += 1
                except Exception as e:
                    print(f"Error cropping image {image_id}: {e}")
                    processing_stats['failed_crops'] += 1
            
            # Store result
            all_detections[image_id] = {
                'original_index': idx,
                'individual_id': sample.get('id', 'Unknown'),
                'label': sample.get('label', 0),
                'date': sample.get('date', 'Unknown'),
                'detections': [detection] if detection else [],
                'cropped_image_file': cropped_image_file
            }
            
            if detection:
                processing_stats['detections'] += 1
    
    # Calculate statistics
    detection_stats = {
        'total_images': len(dataset),
        'images_with_detections': processing_stats['detections'],
        'total_detections': processing_stats['detections'],
        'crops_created': processing_stats['crops_created'],
        'failed_crops': processing_stats['failed_crops']
    }
    
    # Save results
    output_file = os.path.join(output_dir, 'wildlife_detections.json')
    final_results = {
        'detections': all_detections,
        'stats': detection_stats,
        'model_info': {
            'model': f'MegaDetectorV6_{model_version}',
            'detection_mode': 'highest_confidence_per_image',
            'confidence_threshold': confidence_threshold,
            'categories': ['animal']
        },
        'processing_info': {
            'timestamp': time.time(),
            'total_images_processed': len(dataset),
            'cropped_images_directory': cropped_images_dir if save_crops else None,
            'pre_detection_crop': {
                'top_crop': top_crop,
                'bottom_crop': bottom_crop,
                'left_crop': left_crop,
                'right_crop': right_crop
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results, output_file


def print_detection_summary(results, model_version):
    """Print summary of detection results"""
    stats = results['stats']
    detections = results['detections']
    
    print(f"\n" + "=" * 60)
    print(f"DETECTION SUMMARY - {model_version}")
    print("=" * 60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']} ({100*stats['images_with_detections']/stats['total_images']:.1f}%)")
    print(f"Total animal detections: {stats['total_detections']}")
    if 'crops_created' in stats:
        print(f"Cropped images created: {stats['crops_created']}")
        if stats['failed_crops'] > 0:
            print(f"Failed crops: {stats['failed_crops']}")
    print(f"Detection strategy: Highest confidence per image")
    
    # Individual-level statistics
    individual_stats = {}
    for image_data in detections.values():
        individual_id = image_data['individual_id']
        if individual_id not in individual_stats:
            individual_stats[individual_id] = {'images': 0, 'detections': 0}
        
        individual_stats[individual_id]['images'] += 1
        individual_stats[individual_id]['detections'] += len(image_data['detections'])
    
    print(f"\nTop individuals by detection count:")
    sorted_individuals = sorted(individual_stats.items(),
                              key=lambda x: x[1]['detections'], reverse=True)
    
    for i, (individual_id, stats) in enumerate(sorted_individuals[:10]):
        detection_rate = stats['detections'] / stats['images']
        print(f"  {i+1:2d}. {individual_id}: {stats['detections']} detections from {stats['images']} images ({detection_rate:.2f} per image)")


def main():
    parser = argparse.ArgumentParser(description='Detect wildlife using MegaDetector models')
    
    # Model and processing arguments
    parser.add_argument('--model', type=str, default='MDV6-rtdetr-c',
                       choices=['MDV6-yolov9-c', 'MDV6-yolov9-e', 'MDV6-yolov10-c', 'MDV6-yolov10-e', 'MDV6-rtdetr-c'],
                       help='Model version to use (default: MDV6-rtdetr-c)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process (default: None for all)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--save_crops', type=bool, default=True,
                       help='Save cropped images of detections (default: True)')
    parser.add_argument('--output_dir', type=str, default='process_masks/results/detections',
                       help='Directory to save detection results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing images')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Pre-detection cropping arguments
    parser.add_argument('--top_crop', type=float, default=0.0,
                       help='Proportion of image height to crop from the top before detection (0.0 to 1.0)')
    parser.add_argument('--bottom_crop', type=float, default=0.0,
                       help='Proportion of image height to crop from the bottom before detection (0.0 to 1.0)')
    parser.add_argument('--left_crop', type=float, default=0.0,
                       help='Proportion of image width to crop from the left before detection (0.0 to 1.0)')
    parser.add_argument('--right_crop', type=float, default=0.0,
                       help='Proportion of image width to crop from the right before detection (0.0 to 1.0)')

    args = parser.parse_args()
    
    print("=" * 80)
    print("MegaDetector Wildlife Detection Pipeline")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save crops: {args.save_crops}")
    
    if args.top_crop > 0 or args.bottom_crop > 0 or args.left_crop > 0 or args.right_crop > 0:
        print("Pre-detection crop settings:")
        print(f"  Top: {args.top_crop*100:.1f}%, Bottom: {args.bottom_crop*100:.1f}%, "
              f"Left: {args.left_crop*100:.1f}%, Right: {args.right_crop*100:.1f}%")
    
    # Set random seed
    set_all_seeds(args.seed)
    
    # Load dataset
    print("Loading wolverines dataset from HuggingFace...")
    dataset = load_dataset("kdoherty/wolverines", split="train")
    if args.max_images:
        dataset = dataset.select(range(args.max_images))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset
    print(f"\nStarting detection...")
    start_time = time.time()
    
    results, output_file = process_dataset(
        dataset, args.model, args.output_dir, args.confidence, args.save_crops,
        top_crop=args.top_crop, bottom_crop=args.bottom_crop,
        left_crop=args.left_crop, right_crop=args.right_crop,
        batch_size=args.batch_size
    )
    
    processing_time = time.time() - start_time
    
    # Print summary
    print_detection_summary(results, args.model)
    
    print(f"\nProcessing completed in {processing_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    if args.save_crops:
        print(f"Cropped images saved to: {results['processing_info']['cropped_images_directory']}")


if __name__ == "__main__":
    main()
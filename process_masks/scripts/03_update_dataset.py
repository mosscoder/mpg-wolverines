#!/usr/bin/env python3
"""
Script 03: Update HuggingFace Dataset with MegaDetector and SAM Results
Adds MegaDetector detection results and SAM segmentation masks to the wolverines dataset.
Includes confidence scores, bounding boxes, centroids, areas, and status flags.
Renames 'label' to 'pelage' and drops original image column.
"""

import sys
import os
import argparse
import json
import tempfile
import numpy as np
from datasets import Dataset, Features, Image, Value
from tqdm import tqdm
from PIL import Image as PILImage

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import load_wolverines_dataset, set_all_seeds


def load_detection_results(detection_file):
    """Load MegaDetector detection results"""
    print(f"Loading detection results from: {detection_file}")
    
    with open(detection_file, 'r') as f:
        detection_results = json.load(f)
    
    detections = detection_results['detections']
    stats = detection_results['stats']
    
    print(f"Loaded results for {len(detections)} images")
    print(f"Total detections: {stats['total_detections']}")
    
    return detection_results


def calculate_mean_bbox_dimensions(detections):
    """Calculate mean bounding box dimensions from all detections"""
    widths = []
    heights = []
    
    for detection_data in detections.values():
        if detection_data['detections']:
            bbox = detection_data['detections'][0]['bbox']  # [x, y, width, height]
            widths.append(bbox[2])
            heights.append(bbox[3])
    
    if widths and heights:
        mean_width = int(np.mean(widths))
        mean_height = int(np.mean(heights))
        print(f"Mean bbox dimensions: {mean_width}x{mean_height} (from {len(widths)} detections)")
        return mean_width, mean_height
    else:
        # Fallback if no detections
        print("No detections found, using default crop dimensions")
        return 256, 256


def create_center_crop(image, crop_width, crop_height):
    """Create a center crop of specified dimensions from PIL Image"""
    img_width, img_height = image.size
    
    # Calculate center crop coordinates
    left = max(0, (img_width - crop_width) // 2)
    top = max(0, (img_height - crop_height) // 2)
    right = min(img_width, left + crop_width)
    bottom = min(img_height, top + crop_height)
    
    return image.crop((left, top, right, bottom))


def create_detection_and_mask_columns(dataset, detection_results, detections_dir, masks_dir):
    """Create new dataset columns with MegaDetector and SAM results"""
    
    detections = detection_results['detections']
    
    # Calculate mean bbox dimensions for center cropping
    mean_width, mean_height = calculate_mean_bbox_dimensions(detections)
    
    # Create temporary directory for fallback crops
    temp_dir = tempfile.mkdtemp(prefix='wolverines_fallback_crops_')
    print(f"Using temporary directory for fallback crops: {temp_dir}")
    
    # Prepare new column data
    megadetector_confidence = []
    megadetector_bbox_xmin = []
    megadetector_bbox_ymin = []
    megadetector_bbox_xmax = []
    megadetector_bbox_ymax = []
    megadetector_bbox_x_center = []
    megadetector_bbox_y_center = []
    megadetector_bbox_area = []
    megadetector_status = []
    megadetector_image = []
    sam_status = []
    sam_mask = []
    
    print(f"Processing detection and mask data for {len(dataset)} samples...")
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        image_id = f"image_{i:06d}"
        
        if image_id in detections and detections[image_id]['detections']:
            # Get the best (first) detection
            detection = detections[image_id]['detections'][0]
            bbox = detection['bbox']  # [x, y, width, height]
            confidence = detection['confidence']
            
            # Calculate bounding box coordinates and metrics
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            x_center = xmin + bbox[2] / 2
            y_center = ymin + bbox[3] / 2
            area = bbox[2] * bbox[3]
            
            # MegaDetector columns
            megadetector_confidence.append(float(confidence))
            megadetector_bbox_xmin.append(float(xmin))
            megadetector_bbox_ymin.append(float(ymin))
            megadetector_bbox_xmax.append(float(xmax))
            megadetector_bbox_ymax.append(float(ymax))
            megadetector_bbox_x_center.append(float(x_center))
            megadetector_bbox_y_center.append(float(y_center))
            megadetector_bbox_area.append(float(area))
            megadetector_status.append(1)
            
            # MegaDetector cropped image path
            crop_image_path = os.path.join(detections_dir, 'cropped_images', f"{image_id}_crop.jpg")
            if os.path.exists(crop_image_path):
                megadetector_image.append(crop_image_path)
            else:
                megadetector_image.append(None)
        else:
            # No detection found - create center crop with mean dimensions
            original_image = sample['image']
            center_crop = create_center_crop(original_image, mean_width, mean_height)
            
            # Save center crop to temporary directory
            center_crop_path = os.path.join(temp_dir, f"{image_id}_center_crop.jpg")
            center_crop.save(center_crop_path, 'JPEG', quality=95)
            
            # Calculate center crop bbox for metadata
            img_width, img_height = original_image.size
            crop_x = max(0, (img_width - mean_width) // 2)
            crop_y = max(0, (img_height - mean_height) // 2)
            actual_width = min(mean_width, img_width - crop_x)
            actual_height = min(mean_height, img_height - crop_y)
            
            megadetector_confidence.append(None)
            megadetector_bbox_xmin.append(float(crop_x))
            megadetector_bbox_ymin.append(float(crop_y))
            megadetector_bbox_xmax.append(float(crop_x + actual_width))
            megadetector_bbox_ymax.append(float(crop_y + actual_height))
            megadetector_bbox_x_center.append(float(crop_x + actual_width / 2))
            megadetector_bbox_y_center.append(float(crop_y + actual_height / 2))
            megadetector_bbox_area.append(float(actual_width * actual_height))
            megadetector_status.append(0)  # Still 0 because no actual detection
            megadetector_image.append(center_crop_path)
        
        # SAM mask - check by file existence
        sam_mask_path = os.path.join(masks_dir, f"{image_id}_sam_crop.png")
        if os.path.exists(sam_mask_path):
            sam_status.append(1)
            sam_mask.append(sam_mask_path)
        else:
            # No SAM mask - use the crop image (either detection crop or center crop)
            sam_status.append(0)
            if megadetector_image[-1] is not None:  # Use the crop we just created/found
                sam_mask.append(megadetector_image[-1])
            else:
                sam_mask.append(None)
    
    return {
        'megadetector_confidence': megadetector_confidence,
        'megadetector_bbox_xmin': megadetector_bbox_xmin,
        'megadetector_bbox_ymin': megadetector_bbox_ymin,
        'megadetector_bbox_xmax': megadetector_bbox_xmax,
        'megadetector_bbox_ymax': megadetector_bbox_ymax,
        'megadetector_bbox_x_center': megadetector_bbox_x_center,
        'megadetector_bbox_y_center': megadetector_bbox_y_center,
        'megadetector_bbox_area': megadetector_bbox_area,
        'megadetector_status': megadetector_status,
        'megadetector_image': megadetector_image,
        'sam_status': sam_status,
        'sam_mask': sam_mask
    }


def create_updated_dataset(original_dataset, detection_columns):
    """Create new dataset with MegaDetector and SAM columns"""
    
    print("Creating updated dataset with detection and mask data...")
    
    # Prepare core columns (dropping original image, renaming label to pelage)
    updated_data = {
        'id': [sample.get('id', 'Unknown') for sample in original_dataset],
        'pelage': [sample.get('label', 0) for sample in original_dataset],  # Renamed from 'label'
        'date': [sample.get('date', 'Unknown') for sample in original_dataset],
    }
    
    # Add all detection and mask columns
    updated_data.update(detection_columns)
    
    # Define features schema with explicit dtypes
    features = Features({
        'id': Value('string'),
        'pelage': Value('int32'),
        'date': Value('string'),
        'megadetector_confidence': Value('float32'),
        'megadetector_bbox_xmin': Value('float32'),
        'megadetector_bbox_ymin': Value('float32'),
        'megadetector_bbox_xmax': Value('float32'),
        'megadetector_bbox_ymax': Value('float32'),
        'megadetector_bbox_x_center': Value('float32'),
        'megadetector_bbox_y_center': Value('float32'),
        'megadetector_bbox_area': Value('float32'),
        'megadetector_status': Value('int32'),
        'megadetector_image': Image(),
        'sam_status': Value('int32'),
        'sam_mask': Image(),
    })
    
    # Create new dataset with explicit features
    updated_dataset = Dataset.from_dict(updated_data, features=features)
    
    print(f"Updated dataset created with {len(updated_dataset)} samples")
    print(f"Columns: id, pelage (renamed from label), date, MegaDetector columns, SAM columns")
    print(f"Image columns: megadetector_image, sam_mask (with Image dtype)")
    
    return updated_dataset


def save_dataset_locally(dataset, output_path):
    """Save dataset locally before pushing to HuggingFace"""
    print(f"Saving dataset locally to: {output_path}")
    
    # Save as parquet for efficiency
    dataset.save_to_disk(output_path)
    
    print(f"Dataset saved locally")


def print_dataset_summary(dataset, detection_columns):
    """Print summary of updated dataset"""
    
    n_total = len(dataset)
    n_with_detections = sum(detection_columns['megadetector_status'])
    n_with_masks = sum(detection_columns['sam_status'])
    n_without_detections = n_total - n_with_detections
    
    print(f"\n" + "=" * 60)
    print("UPDATED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {n_total}")
    print(f"Samples with MegaDetector detections: {n_with_detections} ({100*n_with_detections/n_total:.1f}%)")
    print(f"Samples with SAM masks: {n_with_masks} ({100*n_with_masks/n_total:.1f}%)")
    print(f"Samples without detections: {n_without_detections} ({100*n_without_detections/n_total:.1f}%)")
    
    if n_with_detections > 0:
        # Filter out None values for statistics
        confidences = [c for c in detection_columns['megadetector_confidence'] if c is not None]
        areas = [a for a in detection_columns['megadetector_bbox_area'] if a is not None]
        
        print(f"\nMegaDetector statistics:")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Min/Max confidence: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
        print(f"  Average bbox area: {np.mean(areas):.1f} pixels")
        print(f"  Min/Max bbox area: {np.min(areas):.1f} - {np.max(areas):.1f} pixels")
    
    # Count samples with crops/masks
    n_with_crops = sum(1 for path in detection_columns['megadetector_image'] if path is not None)
    n_with_mask_images = sum(1 for path in detection_columns['sam_mask'] if path is not None)
    
    print(f"\nImage availability:")
    print(f"  Samples with crop images: {n_with_crops} ({100*n_with_crops/n_total:.1f}%)")
    print(f"  Samples with mask images: {n_with_mask_images} ({100*n_with_mask_images/n_total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Update HuggingFace dataset with MegaDetector and SAM results')
    parser.add_argument('--detection_file', type=str,
                       default='process_masks/results/detections/wildlife_detections.json',
                       help='Path to MegaDetector detection results file')
    parser.add_argument('--detections_dir', type=str, default='process_masks/results/detections',
                       help='Directory containing MegaDetector cropped images')
    parser.add_argument('--masks_dir', type=str, default='process_masks/results/masked_and_cropped_wolverines',
                       help='Directory containing SAM mask images')
    parser.add_argument('--output_dir', type=str, default='process_masks/results/dataset',
                       help='Directory to save updated dataset locally')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push updated dataset to HuggingFace Hub')
    parser.add_argument('--hub_dataset_name', type=str, default='kdoherty/wolverines',
                       help='HuggingFace dataset name for upload')
    parser.add_argument('--config_name', type=str, default='crops-masks',
                       help='Configuration name for HuggingFace dataset')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HuggingFace Dataset Update Pipeline")
    print("=" * 80)
    
    # Check if detection results file exists
    if not os.path.exists(args.detection_file):
        print(f"Error: Detection results file not found: {args.detection_file}")
        print("Please run 01_detect_wildlife.py first")
        return
    
    # Set random seed (not needed for this deterministic script, but kept for consistency)
    set_all_seeds(42)
    
    # Load original dataset
    print("Loading original wolverines dataset...")
    train_dataset, _ = load_wolverines_dataset()
    
    # Load detection results
    detection_results = load_detection_results(args.detection_file)
    
    # Process training dataset
    print(f"\nProcessing training dataset ({len(train_dataset)} samples)...")
    
    detection_columns = create_detection_and_mask_columns(
        train_dataset, detection_results, args.detections_dir, args.masks_dir
    )
    
    # Create updated dataset
    updated_train_dataset = create_updated_dataset(train_dataset, detection_columns)
    
    # Print summary
    print_dataset_summary(updated_train_dataset, detection_columns)
    
    # Save locally
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, 'train')
    save_dataset_locally(updated_train_dataset, train_output_path)
    
    # Optional: Push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing dataset to HuggingFace Hub: {args.hub_dataset_name}")
        print(f"Config: {args.config_name}")
        
        # Push only the updated train split to the specified config
        updated_train_dataset.push_to_hub(
            args.hub_dataset_name,
            config_name=args.config_name,
            split='train'
        )
        print(f"Dataset successfully pushed to: {args.hub_dataset_name} (config: {args.config_name})")
    
    print(f"\nDataset update completed!")
    print(f"Local dataset saved to: {train_output_path}")
    
    if args.push_to_hub:
        print(f"HuggingFace dataset: {args.hub_dataset_name} (config: {args.config_name})")
    else:
        print("Use --push_to_hub to upload to HuggingFace Hub")


if __name__ == "__main__":
    main()
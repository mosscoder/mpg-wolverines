#!/usr/bin/env python3
"""
Script 03: Update HuggingFace Dataset with Masked Images
Adds a new 'masked_image' column to the wolverines dataset containing 
only the segmented wildlife pixels with transparent background.
"""

import sys
import os
import argparse
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import load_wolverines_dataset, set_all_seeds


def load_mask_results(mask_results_file):
    """Load mask generation results"""
    print(f"Loading mask results from: {mask_results_file}")
    
    with open(mask_results_file, 'r') as f:
        mask_results = json.load(f)
    
    mask_data = mask_results['mask_data']
    stats = mask_results['processing_stats']
    
    print(f"Loaded results for {len(mask_data)} images")
    print(f"Total masks available: {stats['masks_generated']}")
    
    return mask_results


def create_masked_image_column(dataset, mask_results, masks_base_dir):
    """Create new dataset column with masked images"""
    
    mask_data = mask_results['mask_data']
    
    # Prepare new column data
    masked_images = []
    mask_metadata = []
    
    print(f"Creating masked image column for {len(dataset)} samples...")
    
    for i, sample in enumerate(tqdm(dataset)):
        image_id = f"image_{i:06d}"
        
        if image_id in mask_data and mask_data[image_id]['masks']:
            # Use the first (best) mask if multiple detections
            mask_info = mask_data[image_id]['masks'][0]
            
            # Load the masked image
            masked_image_path = os.path.join(
                masks_base_dir, 'masked_images', mask_info['masked_image_file']
            )
            
            if os.path.exists(masked_image_path):
                masked_image = Image.open(masked_image_path)
                masked_images.append(masked_image)
                
                mask_metadata.append({
                    'has_mask': True,
                    'bbox': mask_info['bbox'],
                    'confidence': mask_info['confidence'],
                    'coverage_ratio': mask_info['coverage_ratio'],
                    'mask_area': mask_info['mask_area']
                })
            else:
                # No mask file found - use original image
                masked_images.append(sample['image'])
                mask_metadata.append({
                    'has_mask': False,
                    'bbox': None,
                    'confidence': 0.0,
                    'coverage_ratio': 1.0,  # Full image
                    'mask_area': 0
                })
        else:
            # No detection found - use original image
            masked_images.append(sample['image'])
            mask_metadata.append({
                'has_mask': False,
                'bbox': None,
                'confidence': 0.0,
                'coverage_ratio': 1.0,  # Full image
                'mask_area': 0
            })
    
    return masked_images, mask_metadata


def create_updated_dataset(original_dataset, masked_images, mask_metadata):
    """Create new dataset with masked images and metadata"""
    
    print("Creating updated dataset with masked images...")
    
    # Prepare all columns
    updated_data = {
        'image': [sample['image'] for sample in original_dataset],  # Original images
        'masked_image': masked_images,  # New masked images
        'id': [sample.get('id', 'Unknown') for sample in original_dataset],
        'label': [sample.get('label', 0) for sample in original_dataset],
        'date': [sample.get('date', 'Unknown') for sample in original_dataset],
    }
    
    # Add mask metadata
    updated_data['mask_bbox'] = [meta['bbox'] for meta in mask_metadata]
    updated_data['mask_confidence'] = [meta['confidence'] for meta in mask_metadata]
    updated_data['mask_coverage_ratio'] = [meta['coverage_ratio'] for meta in mask_metadata]
    updated_data['has_wildlife_mask'] = [meta['has_mask'] for meta in mask_metadata]
    
    # Create new dataset
    updated_dataset = Dataset.from_dict(updated_data)
    
    print(f"Updated dataset created with {len(updated_dataset)} samples")
    print(f"New columns: masked_image, mask_bbox, mask_confidence, mask_coverage_ratio, has_wildlife_mask")
    
    return updated_dataset


def save_dataset_locally(dataset, output_path):
    """Save dataset locally before pushing to HuggingFace"""
    print(f"Saving dataset locally to: {output_path}")
    
    # Save as parquet for efficiency
    dataset.save_to_disk(output_path)
    
    print(f"Dataset saved locally")


def print_dataset_summary(dataset, mask_metadata):
    """Print summary of updated dataset"""
    
    n_total = len(dataset)
    n_with_masks = sum(meta['has_mask'] for meta in mask_metadata)
    n_without_masks = n_total - n_with_masks
    
    print(f"\n" + "=" * 60)
    print("UPDATED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {n_total}")
    print(f"Samples with wildlife masks: {n_with_masks} ({100*n_with_masks/n_total:.1f}%)")
    print(f"Samples using original images: {n_without_masks} ({100*n_without_masks/n_total:.1f}%)")
    
    if n_with_masks > 0:
        coverages = [meta['coverage_ratio'] for meta in mask_metadata if meta['has_mask']]
        confidences = [meta['confidence'] for meta in mask_metadata if meta['has_mask']]
        
        print(f"\nMask statistics:")
        print(f"  Average coverage: {np.mean(coverages):.3f} ({100*np.mean(coverages):.1f}% of image)")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Min/Max coverage: {np.min(coverages):.3f} - {np.max(coverages):.3f}")
        print(f"  Min/Max confidence: {np.min(confidences):.3f} - {np.max(confidences):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Update HuggingFace dataset with wildlife masks')
    parser.add_argument('--mask_results_file', type=str,
                       default='process_masks/results/masks/mask_results.json',
                       help='Path to SAM mask results file')
    parser.add_argument('--masks_base_dir', type=str, default='process_masks/results/masks',
                       help='Base directory containing mask files')
    parser.add_argument('--output_dir', type=str, default='process_masks/results/dataset',
                       help='Directory to save updated dataset locally')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push updated dataset to HuggingFace Hub')
    parser.add_argument('--hub_dataset_name', type=str, default='kdoherty/wolverines-masked',
                       help='HuggingFace dataset name for upload')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HuggingFace Dataset Update Pipeline")
    print("=" * 80)
    
    # Check if mask results file exists
    if not os.path.exists(args.mask_results_file):
        print(f"Error: Mask results file not found: {args.mask_results_file}")
        print("Please run 02_generate_masks.py first")
        return
    
    # Set random seed
    set_all_seeds(42)
    
    # Load original dataset
    print("Loading original wolverines dataset...")
    train_dataset, test_dataset = load_wolverines_dataset()
    
    # Load mask results
    mask_results = load_mask_results(args.mask_results_file)
    
    # Process training dataset (extend to test later)
    print(f"\nProcessing training dataset ({len(train_dataset)} samples)...")
    
    masked_images, mask_metadata = create_masked_image_column(
        train_dataset, mask_results, args.masks_base_dir
    )
    
    # Create updated dataset
    updated_train_dataset = create_updated_dataset(train_dataset, masked_images, mask_metadata)
    
    # Print summary
    print_dataset_summary(updated_train_dataset, mask_metadata)
    
    # Save locally
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, 'train')
    save_dataset_locally(updated_train_dataset, train_output_path)
    
    # Optional: Push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing dataset to HuggingFace Hub: {args.hub_dataset_name}")
        
        # Create DatasetDict for train/test splits
        dataset_dict = DatasetDict({
            'train': updated_train_dataset,
            'test': test_dataset  # Keep original test set for now
        })
        
        # Push to hub
        dataset_dict.push_to_hub(args.hub_dataset_name)
        print(f"Dataset successfully pushed to: {args.hub_dataset_name}")
    
    print(f"\nDataset update completed!")
    print(f"Local dataset saved to: {train_output_path}")
    
    if args.push_to_hub:
        print(f"HuggingFace dataset: {args.hub_dataset_name}")
    else:
        print("Use --push_to_hub to upload to HuggingFace Hub")


if __name__ == "__main__":
    main()
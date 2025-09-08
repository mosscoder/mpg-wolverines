#!/usr/bin/env python3
"""
Script 01: Run MegaDetector on Robust Pelage Images
Uses MegaDetector v6 to detect and crop wolverines from all robust pelage directories.
Saves crops with trackable filenames and maintains complete metadata mapping.
"""

# Fix OpenMP conflict on macOS before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import json
import csv
import tempfile
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_robust_directories(robust_dirs_file):
    """Load list of robust directories to process"""
    print(f"Loading robust directories from: {robust_dirs_file}")
    
    if not os.path.exists(robust_dirs_file):
        raise FileNotFoundError(f"Robust directories file not found: {robust_dirs_file}")
    
    with open(robust_dirs_file, 'r') as f:
        data = json.load(f)
    
    directories = data.get('directories_to_process', [])
    print(f"Found {len(directories)} directories to process")
    
    return directories


def load_inventory_metadata(db_path, drive_path):
    """Load inventory metadata for mapping directory info to metadata"""
    print(f"Loading inventory metadata from: {db_path}")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    # Load and preprocess inventory data (same as 00_summarize_robust.py)
    db = pd.read_excel(db_path, sheet_name=1)
    db['Folder path'] = db['Folder path'].str.replace('D:', drive_path).str.replace('\\', '/')
    db['Marks'] = db['Marks'].str.replace(' marks', '')
    db = db.rename(columns=str.lower)
    db['ymdh'] = db['year'].astype(str) + db['date-time'].str.replace('.', '').str.replace('-', '')
    db.drop(columns=['date-time', 'link'], inplace=True)
    db = db.rename(columns={'type': 'color', 'folder path': 'folder_path'})
    db['color'] = db['color'].apply(lambda x: 1 if x == 'COLOR' else 0)
    
    # Filter for robust only
    robust_db = db[db['marks'] == 'Robust'].copy()
    print(f"Loaded metadata for {len(robust_db)} robust records")
    
    return robust_db


def init_detection_model(model_version, device):
    """Initialize MegaDetector model"""
    from PytorchWildlife.models import detection as pw_detection
    from PytorchWildlife.data import transforms as pw_transforms
    
    print(f"Initializing {model_version} model on {device}...")
    
    # Initialize model
    detection_model = pw_detection.MegaDetectorV6(
        device=device,
        pretrained=True,
        version=model_version
    )
    
    # Set appropriate image size based on model
    if model_version in ['MDV6-yolov9-e', 'MDV6-yolov10-e']:
        detection_model.IMAGE_SIZE = 1280
        detection_model.predictor.args.imgsz = 1280
        detection_model.transform = pw_transforms.MegaDetector_v5_Transform(
            target_size=1280, stride=32
        )
    else:
        detection_model.IMAGE_SIZE = 640
        detection_model.predictor.args.imgsz = 640
    
    return detection_model


def detect_image_single(image_path, detection_model, confidence_threshold=0.8):
    """Run detection on single image file (fallback)"""
    try:
        # Load image
        with Image.open(image_path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Run detection directly on file path
            result = detection_model.single_image_detection(image_path, det_conf_thres=confidence_threshold)
            
            if 'detections' in result and len(result['detections']) > 0:
                sv_detections = result['detections']
                
                # Find all animal detections above threshold
                detections = []
                for j in range(len(sv_detections)):
                    x1, y1, x2, y2 = sv_detections.xyxy[j]
                    conf = float(sv_detections.confidence[j])
                    class_id = int(sv_detections.class_id[j])
                    
                    # Keep only animals (class_id=0) above confidence threshold
                    if class_id == 0 and conf >= confidence_threshold:
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'confidence': conf,
                            'category': 'animal',
                            'class_name': 'animal'
                        })
                
                # Sort by confidence (highest first)
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                return detections, image.copy()
            
    except Exception as e:
        print(f"Error detecting image {image_path}: {e}")
        return [], None
    
    return [], None


def crop_image_from_bbox(image, bbox):
    """Crop image using bounding box coordinates"""
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


def generate_crop_filename(metadata, original_basename):
    """Generate standardized crop filename"""
    # Format: {id}_{year}_{ymdh}_{original_basename}.jpg
    base_name = os.path.splitext(original_basename)[0]  # Remove extension
    return f"{metadata['id']}_{metadata['year']}_{metadata['ymdh']}_{base_name}.jpg"



def process_directory(directory_path, metadata_df, detection_model, crops_dir, confidence_threshold=0.8, batch_size=8, num_workers=4):
    """Process all images in a directory using batch processing"""
    print(f"Processing directory: {directory_path}")
    
    # Get directory metadata
    dir_metadata = metadata_df[metadata_df['folder_path'] == directory_path]
    if dir_metadata.empty:
        print(f"  Warning: No metadata found for directory {directory_path}")
        return []
    
    # Use first row for directory-level metadata (all should be same id/year/station)
    dir_info = dir_metadata.iloc[0]
    
    # Find all image files
    image_patterns = ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob(os.path.join(directory_path, pattern)))
    
    if not image_files:
        print(f"  No image files found")
        return []
    
    print(f"  Found {len(image_files)} images")
    
    crop_metadata = []
    crops_created = 0
    crops_skipped = 0
    
    # Process images directly
    for image_path in tqdm(image_files, desc="  Processing images", leave=False):
        try:
            original_basename = os.path.basename(image_path)
            
            # Generate expected crop filename
            crop_filename = generate_crop_filename(dir_info, original_basename)
            crop_path = os.path.join(crops_dir, crop_filename)
            
            # Skip if crop already exists (resume functionality)
            if os.path.exists(crop_path):
                crops_skipped += 1
                continue
            
            # Load image
            with Image.open(image_path) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Run detection
                result = detection_model.single_image_detection(
                    image_path, 
                    det_conf_thres=confidence_threshold
                )
                
                # Only process if we have detections
                if 'detections' in result and len(result['detections']) > 0:
                    sv_detections = result['detections']
                    
                    # Find highest confidence animal detection
                    best_detection = None
                    for j in range(len(sv_detections)):
                        x1, y1, x2, y2 = sv_detections.xyxy[j]
                        conf = float(sv_detections.confidence[j])
                        class_id = int(sv_detections.class_id[j])
                        
                        # Keep only animals (class_id=0) above confidence threshold
                        if class_id == 0 and conf >= confidence_threshold:
                            best_detection = {
                                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                'confidence': conf,
                                'category': 'animal',
                                'class_name': 'animal'
                            }
                            break  # Take first (highest confidence) detection only
                    
                    # Process the best detection if found
                    if best_detection:
                        # Create crop
                        cropped_image = crop_image_from_bbox(image, best_detection['bbox'])
                        
                        if cropped_image is not None:
                            # Save crop immediately
                            try:
                                cropped_image.save(crop_path, 'JPEG', quality=95)
                                crops_created += 1
                                
                                # Prepare and collect metadata
                                metadata_entry = {
                                    'id': dir_info['id'],
                                    'year': dir_info['year'],
                                    'ymdh': dir_info['ymdh'],
                                    'station': dir_info['station'],
                                    'color': dir_info['color'],
                                    'original_path': image_path,
                                    'original_filename': original_basename,
                                    'crop_filename': crop_filename,
                                    'confidence': best_detection['confidence'],
                                    'bbox_x': best_detection['bbox'][0],
                                    'bbox_y': best_detection['bbox'][1],
                                    'bbox_width': best_detection['bbox'][2],
                                    'bbox_height': best_detection['bbox'][3],
                                    'image_width': image.width,
                                    'image_height': image.height,
                                    'crop_width': cropped_image.width,
                                    'crop_height': cropped_image.height
                                }
                                crop_metadata.append(metadata_entry)
                                
                            except Exception as e:
                                print(f"    Error saving crop {crop_path}: {e}")
                                
        except Exception as e:
            print(f"  Error processing image {image_path}: {e}")
    
    if crops_skipped > 0:
        print(f"  Created {crops_created} crops, skipped {crops_skipped} existing crops from {len(image_files)} images")
    else:
        print(f"  Created {crops_created} crops from {len(image_files)} images")
    return crop_metadata


def save_metadata(all_metadata, metadata_file):
    """Save metadata to CSV file"""
    print(f"Saving metadata to: {metadata_file}")
    
    if not all_metadata:
        print("No metadata to save")
        return
    
    # Convert to DataFrame for easy CSV writing
    df = pd.DataFrame(all_metadata)
    df.to_csv(metadata_file, index=False)
    
    print(f"Saved metadata for {len(all_metadata)} crops")


def generate_summary_stats(all_metadata, summary_file):
    """Generate and save summary statistics"""
    if not all_metadata:
        print("No data for summary statistics")
        return
    
    df = pd.DataFrame(all_metadata)
    
    # Calculate statistics
    stats = {
        'total_crops': len(df),
        'unique_individuals': df['id'].nunique(),
        'unique_years': sorted(df['year'].unique().tolist()),
        'unique_stations': sorted(df['station'].unique().tolist()),
        'confidence_stats': {
            'min': float(df['confidence'].min()),
            'max': float(df['confidence'].max()),
            'mean': float(df['confidence'].mean()),
            'std': float(df['confidence'].std())
        },
        'crops_per_individual': df['id'].value_counts().to_dict(),
        'crops_per_year': df['year'].value_counts().to_dict(),
        'color_distribution': {
            'color_images': int(df[df['color'] == 1]['color'].count()),
            'bw_images': int(df[df['color'] == 0]['color'].count())
        }
    }
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total crops created: {stats['total_crops']:,}")
    print(f"Unique individuals: {stats['unique_individuals']}")
    print(f"Years covered: {stats['unique_years']}")
    print(f"Stations covered: {len(stats['unique_stations'])}")
    print(f"Color images: {stats['color_distribution']['color_images']:,}")
    print(f"B&W images: {stats['color_distribution']['bw_images']:,}")
    print(f"Confidence range: {stats['confidence_stats']['min']:.3f} - {stats['confidence_stats']['max']:.3f}")
    print(f"Mean confidence: {stats['confidence_stats']['mean']:.3f}")
    
    # Top individuals
    print("\nTop 10 individuals by crop count:")
    top_individuals = sorted(stats['crops_per_individual'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
    for i, (individual, count) in enumerate(top_individuals, 1):
        print(f"  {i:2d}. {individual}: {count:,} crops")


def main():
    parser = argparse.ArgumentParser(description='Run MegaDetector on robust pelage images')
    parser.add_argument('--robust_dirs', type=str,
                       default='hugging_face_dataset/v2/data/robust_directories.json',
                       help='JSON file with robust directories to process')
    parser.add_argument('--drive_path', type=str,
                       default='/Volumes/Seagate Portable Drive',
                       help='Path to external drive')
    parser.add_argument('--model', type=str, default='MDV6-rtdetr-c',
                       choices=['MDV6-yolov9-c', 'MDV6-yolov9-e', 'MDV6-yolov10-c', 'MDV6-yolov10-e', 'MDV6-rtdetr-c'],
                       help='MegaDetector model version')
    parser.add_argument('--confidence', type=float, default=0.8,
                       help='Confidence threshold for detections')
    parser.add_argument('--output_dir', type=str,
                       default='hugging_face_dataset/v2',
                       help='Output directory for crops and metadata')
    parser.add_argument('--max_dirs', type=int, default=None,
                       help='Maximum directories to process (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing images')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for parallel I/O operations')
    
    args = parser.parse_args()
    
    # Set random seed
    set_all_seeds(args.seed)
    
    # Setup paths
    db_path = os.path.join(args.drive_path, 'KYLE_AI_DATABASE/Detections_inventory/Detections inventory_2023.08.29.xlsx')
    crops_dir = os.path.join(args.output_dir, 'init_crops')
    metadata_dir = os.path.join(args.output_dir, 'metadata')
    metadata_file = os.path.join(metadata_dir, 'crop_metadata.csv')
    summary_file = os.path.join(metadata_dir, 'detection_summary.json')
    
    # Create output directories
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    print("="*80)
    print("MegaDetector Processing for Robust Pelage Images (Optimized)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Batch size: {args.batch_size}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Drive path: {args.drive_path}")
    print(f"Output crops: {crops_dir}")
    print(f"Output metadata: {metadata_file}")
    
    try:
        # Load directories and metadata
        directories = load_robust_directories(args.robust_dirs)
        metadata_df = load_inventory_metadata(db_path, args.drive_path)
        
        # Limit directories for testing
        if args.max_dirs:
            directories = directories[:args.max_dirs]
            print(f"Limited to first {args.max_dirs} directories for testing")
        
        # Initialize detection model
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        detection_model = init_detection_model(args.model, device)
        
        # Process all directories
        print(f"\nProcessing {len(directories)} directories...")
        start_time = time.time()
        
        all_metadata = []
        for i, directory in enumerate(directories, 1):
            print(f"\n[{i}/{len(directories)}] {directory}")
            
            if not os.path.exists(directory):
                print(f"  Directory not found, skipping: {directory}")
                continue
            
            crop_metadata = process_directory(
                directory, metadata_df, detection_model, crops_dir, 
                args.confidence, args.batch_size, args.num_workers
            )
            
            # Save metadata for this directory immediately
            if crop_metadata:
                dir_metadata_file = os.path.join(metadata_dir, f'crop_metadata_dir{i:04d}.csv')
                save_metadata(crop_metadata, dir_metadata_file)
                print(f"  Saved metadata to: crop_metadata_dir{i:04d}.csv")
            
            all_metadata.extend(crop_metadata)
        
        processing_time = time.time() - start_time
        
        # Concatenate all individual metadata files into final file
        print(f"\nConcatenating metadata files...")
        metadata_files = glob(os.path.join(metadata_dir, 'crop_metadata_dir*.csv'))
        if metadata_files:
            # Read and concatenate all CSV files
            all_dataframes = []
            for file in sorted(metadata_files):
                df = pd.read_csv(file)
                all_dataframes.append(df)
            
            if all_dataframes:
                final_df = pd.concat(all_dataframes, ignore_index=True)
                final_df.to_csv(metadata_file, index=False)
                print(f"Final metadata saved to: {os.path.basename(metadata_file)}")
            
            # Generate summary from final concatenated data
            all_metadata = final_df.to_dict('records')
        
        generate_summary_stats(all_metadata, summary_file)
        
        print(f"\n" + "="*60)
        print(f"PROCESSING COMPLETE")
        print(f"Total processing time: {processing_time/60:.1f} minutes")
        print(f"Crops saved to: {crops_dir}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"Summary saved to: {summary_file}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
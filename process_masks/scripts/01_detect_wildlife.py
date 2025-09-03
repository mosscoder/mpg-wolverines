#!/usr/bin/env python3
"""
Script 01: Wildlife Detection with MegaDetector
Uses MegaDetector v5 to detect wildlife in images with confidence > 0.95.
Saves bounding box coordinates for subsequent mask generation.
"""

# Fix OpenMP conflict on macOS before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import load_wolverines_dataset, set_all_seeds


def run_pytorchwildlife_subprocess(image_paths):
    """Run PytorchWildlife MegaDetectorV6 via subprocess to avoid utils module conflicts"""
    
    # Create temporary files for communication
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        image_paths_file = f.name
        json.dump(image_paths, f)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name
    
    try:
        # Run PytorchWildlife in subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'run_pytorchwildlife.py')
        cmd = [sys.executable, script_path, image_paths_file, output_file]
        
        print("Running PytorchWildlife MegaDetectorV6-RTDetr (highest confidence detection) in isolated subprocess...")
        print("(This may take several minutes for large datasets)")
        
        # Use Popen for real-time output streaming
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                            text=True, bufsize=1, universal_newlines=True) as process:
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
            
            # Wait for completion and check return code
            process.wait()
            
            if process.returncode != 0:
                print(f"Error running PytorchWildlife:")
                stderr_output = process.stderr.read()
                if stderr_output:
                    print(stderr_output)
                raise RuntimeError(f"PytorchWildlife subprocess failed with code {process.returncode}")
        
        # Load and return results
        with open(output_file, 'r') as f:
            return json.load(f)
    
    finally:
        # Clean up temp files
        if os.path.exists(image_paths_file):
            os.unlink(image_paths_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def save_dataset_images_temporarily(dataset, temp_dir):
    """Save dataset images to temporary directory for MegaDetector processing"""
    image_paths = []
    image_mapping = {}  # Maps file path to dataset index
    
    print(f"Saving {len(dataset)} images to temporary directory...")
    
    for i, sample in enumerate(tqdm(dataset)):
        image = sample['image']
        image_id = f"image_{i:06d}"
        image_path = os.path.join(temp_dir, f"{image_id}.jpg")
        
        # Save image
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(image_path, 'JPEG', quality=95)
        
        image_paths.append(image_path)
        image_mapping[image_path] = i
    
    return image_paths, image_mapping

def parse_detection_results(detection_results, image_mapping, dataset):
    """Parse PytorchWildlife detection results to our format"""
    all_detections = {}
    
    # Handle the case where detection_results might be a list
    if not isinstance(detection_results, list):
        print(f"Warning: Unexpected result format: {type(detection_results)}")
        return all_detections
    
    for result in detection_results:
        if not result or not isinstance(result, dict):
            continue
            
        image_path = result.get('file')
        if not image_path or image_path not in image_mapping:
            continue
            
        dataset_index = image_mapping[image_path]
        sample = dataset[dataset_index]
        image_id = f"image_{dataset_index:06d}"
        
        # Parse detections from MegaDetector format
        detections = []
        for detection in result.get('detections', []):
            conf = detection['conf']
            category_id = detection['category']
            
            # PytorchWildlife categories: '1'=animal, '2'=person, '3'=vehicle  
            # Note: Confidence already filtered by adaptive thresholding in subprocess
            if category_id == '1':  # Only animals
                bbox = detection['bbox']
                # PytorchWildlife bbox format: [x_min, y_min, width_normalized, height_normalized]
                # Convert to absolute coordinates
                img_width, img_height = sample['image'].size
                x = bbox[0] * img_width
                y = bbox[1] * img_height
                width = bbox[2] * img_width
                height = bbox[3] * img_height
                
                detections.append({
                    'bbox': [float(x), float(y), float(width), float(height)],
                    'confidence': float(conf),
                    'category': 'animal',
                    'class_name': 'animal'
                })
        
        # Store results
        all_detections[image_id] = {
            'original_index': dataset_index,
            'individual_id': sample.get('id', 'Unknown'),
            'label': sample.get('label', 0),
            'date': sample.get('date', 'Unknown'),
            'detections': detections
        }
    
    return all_detections


def process_dataset(dataset, output_dir):
    """Process entire dataset through PytorchWildlife MegaDetectorV6 using subprocess isolation"""
    
    # Create temporary directory for images
    with tempfile.TemporaryDirectory(prefix='megadetector_') as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Save all dataset images to temporary files
        image_paths, image_mapping = save_dataset_images_temporarily(dataset, temp_dir)
        
        # Run PytorchWildlife via subprocess to avoid utils conflicts
        print(f"Processing {len(image_paths)} images through PytorchWildlife MegaDetectorV6...")
        detection_results = run_pytorchwildlife_subprocess(image_paths)
        
        # Parse results to our format
        all_detections = parse_detection_results(detection_results, image_mapping, dataset)
        
        # Calculate statistics
        detection_stats = {
            'total_images': len(dataset),
            'images_with_detections': sum(1 for data in all_detections.values() if data['detections']),
            'total_detections': sum(len(data['detections']) for data in all_detections.values())
        }
        
        # Save final results
        output_file = os.path.join(output_dir, 'wildlife_detections.json')
        final_results = {
            'detections': all_detections,
            'stats': detection_stats,
            'model_info': {
                'model': 'MegaDetectorV6_RTDetr',
                'detection_mode': 'highest_confidence_per_image',
                'categories': ['animal']
            },
            'processing_info': {
                'timestamp': time.time(),
                'total_images_processed': len(dataset)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results, output_file


def print_detection_summary(results):
    """Print summary of detection results"""
    stats = results['stats']
    detections = results['detections']
    
    print(f"\n" + "=" * 60)
    print("PYTORCHWILDLIFE DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']} ({100*stats['images_with_detections']/stats['total_images']:.1f}%)")
    print(f"Total animal detections: {stats['total_detections']}")
    print(f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}")
    print(f"Average detections per positive image: {stats['total_detections']/max(1, stats['images_with_detections']):.2f}")
    
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
    parser = argparse.ArgumentParser(description='Detect wildlife in wolverine images using PytorchWildlife MegaDetectorV6')
    parser.add_argument('--output_dir', type=str, default='process_masks/results/detections',
                       help='Directory to save detection results')
    # Removed confidence argument as we now return highest confidence detection per image
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process (None = process full dataset)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PytorchWildlife MegaDetectorV6 Detection Pipeline")
    print("=" * 80)
    
    # Set random seed
    set_all_seeds(42)
    
    # Load dataset
    print("Loading wolverines dataset...")
    train_dataset, test_dataset = load_wolverines_dataset()
    
    # Combine datasets for processing
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Use training dataset for now (can extend to both later)
    dataset = train_dataset
    print(f"DEBUG: Before limiting - dataset has {len(dataset)} images")
    
    if args.max_images:
        print(f"Limiting to first {args.max_images} images for testing")
        dataset = dataset.select(range(min(args.max_images, len(dataset))))
        print(f"DEBUG: After limiting - dataset has {len(dataset)} images")
    
    print(f"Processing {len(dataset)} images")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset
    print(f"\nStarting detection (highest confidence mode)")
    start_time = time.time()
    
    results, output_file = process_dataset(dataset, args.output_dir)
    
    processing_time = time.time() - start_time
    
    # Print summary
    print_detection_summary(results)
    
    print(f"\nProcessing completed in {processing_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print(f"Next step: Run 02_generate_masks.py to create segmentation masks")


if __name__ == "__main__":
    main()
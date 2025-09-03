#!/usr/bin/env python3
"""
Standalone MegaDetector runner that avoids utils module conflicts.
This script is called as a subprocess to isolate the MegaDetector environment.
"""

import sys
import json
import os

def run_megadetector(image_paths_file, output_file, confidence_threshold=0.95):
    """Run MegaDetector on a list of image paths"""
    
    # Set MPS fallback for torchvision NMS operation
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Remove current directory from path to avoid utils conflict
    if '.' in sys.path:
        sys.path.remove('.')
    if '' in sys.path:
        sys.path.remove('')
    
    # Now import MegaDetector after cleaning path
    from megadetector.detection.run_detector_batch import load_and_run_detector_batch
    
    # Load image paths
    with open(image_paths_file, 'r') as f:
        image_paths = json.load(f)
    
    print(f"Running MegaDetector on {len(image_paths)} images...")
    sys.stdout.flush()
    
    # Run detection with verbose output
    results = load_and_run_detector_batch(
        model_file='MDV5A',
        image_file_names=image_paths,
        confidence_threshold=confidence_threshold,
        quiet=False
    )
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: run_megadetector.py <image_paths.json> <output.json> <confidence_threshold>")
        sys.exit(1)
    
    image_paths_file = sys.argv[1]
    output_file = sys.argv[2]
    confidence_threshold = float(sys.argv[3])
    
    sys.exit(run_megadetector(image_paths_file, output_file, confidence_threshold))
#!/usr/bin/env python3
"""
Standalone MegaDetectorV6 runner using PytorchWildlife.
This script is called as a subprocess to isolate the detection environment.
"""

import sys
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def run_pytorchwildlife(image_paths_file, output_file):
    """Run MegaDetectorV6-Apache-RTDetr-Extra on a list of image paths, returning highest confidence detection"""
    
    # Set MPS fallback for torchvision NMS operation
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Remove current directory from path to avoid utils conflict
    if '.' in sys.path:
        sys.path.remove('.')
    if '' in sys.path:
        sys.path.remove('')
    
    # Import PytorchWildlife MegaDetectorV6
    from PytorchWildlife.models import detection as pw_detection
    print("Using MegaDetectorV6 RTDetr model")
    
    import torch
    
    # Load image paths
    with open(image_paths_file, 'r') as f:
        image_paths = json.load(f)
    
    print(f"Running wildlife detection on {len(image_paths)} images...")
    sys.stdout.flush()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Initialize MegaDetectorV6 with RTDetr model
    print("Loading MegaDetectorV6 RTDetr model (this may download weights on first run)...")
    
    detection_model = pw_detection.MegaDetectorV6(
        device=device,
        pretrained=True,
        version="MDV6-rtdetr-c"  # RTDetr model
    )
    print("Using MegaDetectorV6 RTDetr")
    
    # Process images and collect results
    results = []
    print(f"Processing {len(image_paths)} images (returning highest confidence detection per image)...")
    
    for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        # Print progress every 10 images
        if idx % 10 == 0 and idx > 0:
            print(f"\nProcessed {idx}/{len(image_paths)} images...")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Run detection at low confidence to get all possible detections
            detection_result = detection_model.single_image_detection(image_array, det_conf_thres=0.1)
            
            # Convert supervision.Detections format to MegaDetector-compatible format
            all_detections = []
            
            if 'detections' in detection_result and len(detection_result['detections']) > 0:
                sv_detections = detection_result['detections']
                image_height, image_width = image_array.shape[:2]
                
                # Iterate through detections using supervision format
                for i in range(len(sv_detections)):
                    # Get bounding box in absolute coordinates [x1, y1, x2, y2]
                    x1, y1, x2, y2 = sv_detections.xyxy[i]
                    
                    # Convert to MegaDetector format [x, y, width, height] normalized
                    x = float(x1 / image_width)
                    y = float(y1 / image_height)
                    width = float((x2 - x1) / image_width)
                    height = float((y2 - y1) / image_height)
                    
                    # Get confidence and class
                    conf = float(sv_detections.confidence[i]) if sv_detections.confidence is not None else 1.0
                    class_id = int(sv_detections.class_id[i]) if sv_detections.class_id is not None else 0
                    
                    # MegaDetectorV6: 0=animal, 1=person, 2=vehicle
                    # Convert to MegaDetector v5 format: '1'=animal, '2'=person, '3'=vehicle
                    if class_id == 0:  # animal
                        category = '1'
                        all_detections.append({
                            'category': category,
                            'conf': conf,
                            'bbox': [x, y, width, height]
                        })
            
            # Select only the highest confidence detection
            detections = []
            if all_detections:
                highest_conf_detection = max(all_detections, key=lambda x: x['conf'])
                detections = [highest_conf_detection]
            
            # Add result in MegaDetector format for compatibility
            results.append({
                'file': image_path,
                'detections': detections,
                'failure': None
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'file': image_path,
                'detections': [],
                'failure': str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run_pytorchwildlife.py <image_paths.json> <output.json>")
        sys.exit(1)
    
    image_paths_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sys.exit(run_pytorchwildlife(image_paths_file, output_file))
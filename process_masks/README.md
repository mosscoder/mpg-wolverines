# Wildlife Masking Pipeline

This pipeline uses **MegaDetector** and **Segment Anything Model (SAM)** to create precise wildlife masks for the wolverines dataset.

## Overview

1. **MegaDetector** detects wildlife with confidence > 0.95
2. **SAM** generates precise segmentation masks from bounding boxes  
3. **Dataset Update** adds masked images to HuggingFace dataset

## Pipeline Steps

### Step 1: Detect Wildlife
```bash
python process_masks/scripts/01_detect_wildlife.py [--max_images 100]
```
- Downloads MegaDetector v5 automatically
- Processes all images in training dataset
- Saves detections to `process_masks/results/detections/wildlife_detections.json`

### Step 2: Generate Masks
```bash
python process_masks/scripts/02_generate_masks.py
```
- Downloads SAM model automatically (vit_b by default)
- Creates segmentation masks for each detection
- Saves masks and masked images to `process_masks/results/masks/`

### Step 3: Update Dataset
```bash
python process_masks/scripts/03_update_dataset.py [--push_to_hub]
```
- Adds `masked_image` column to HuggingFace dataset
- Preserves original images in `image` column
- Optionally pushes to HuggingFace Hub as new dataset

## Dependencies

Install additional dependencies:
```bash
pip install ultralytics  # For MegaDetector
pip install git+https://github.com/facebookresearch/segment-anything.git  # For SAM
```

## Output

**New dataset columns:**
- `masked_image`: Wildlife-only pixels with transparent background
- `mask_bbox`: Bounding box coordinates [x, y, width, height]
- `mask_confidence`: MegaDetector confidence score
- `mask_coverage_ratio`: Fraction of image covered by mask
- `has_wildlife_mask`: Boolean indicating if masking was applied

## Usage

This masked dataset can improve individual ID classification by:
- Removing background distractions
- Focusing on wolverine-specific features
- Reducing domain shift between different camera locations
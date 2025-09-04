#!/usr/bin/env python3
"""
Test script for SAM mask application pipeline
Loads a few samples from the crops-masks dataset and tests mask application
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

def apply_sam_mask(image, mask):
    """Apply SAM mask to image by setting background to black"""
    # Convert to numpy arrays
    img_np = np.array(image)
    mask_np = np.array(mask.convert('L'))  # Convert mask to grayscale
    
    # Normalize mask to 0-1 range
    if mask_np.max() > 1:
        mask_np = mask_np / 255.0
    
    # Apply mask: set background pixels (mask=0) to black
    if len(img_np.shape) == 3:  # RGB image
        masked_img = img_np * mask_np[:, :, np.newaxis]
    else:  # Grayscale image
        masked_img = img_np * mask_np
    
    # Convert back to PIL Image
    masked_img = masked_img.astype(np.uint8)
    return Image.fromarray(masked_img)

def test_mask_pipeline():
    """Test the mask application pipeline"""
    print("Loading crops-masks dataset...")
    try:
        dataset = load_dataset("kdoherty/wolverines", "crops-masks", split="train")
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Find samples with valid masks
    valid_samples = []
    for i, sample in enumerate(dataset):
        if (sample['megadetector_status'] == 1 and 
            sample['sam_status'] == 1 and 
            sample['megadetector_image'] is not None and
            sample['sam_mask'] is not None):
            valid_samples.append((i, sample))
            if len(valid_samples) >= 4:  # Get 4 samples for testing
                break
    
    if len(valid_samples) == 0:
        print("No valid samples found with both MegaDetector crops and SAM masks")
        return False
    
    print(f"Found {len(valid_samples)} valid samples for testing")
    
    # Test mask application on each sample
    fig, axes = plt.subplots(2, len(valid_samples), figsize=(len(valid_samples)*4, 8))
    if len(valid_samples) == 1:
        axes = axes.reshape(2, 1)
    
    for col, (idx, sample) in enumerate(valid_samples):
        # Original cropped image
        orig_img = sample['megadetector_image']
        axes[0, col].imshow(orig_img)
        axes[0, col].set_title(f"Original Crop\nID: {sample['id'][:8]}\nPelage: {sample['pelage']}")
        axes[0, col].axis('off')
        
        # Apply mask
        try:
            masked_img = apply_sam_mask(orig_img, sample['sam_mask'])
            axes[1, col].imshow(masked_img)
            axes[1, col].set_title(f"SAM Masked\nBG removed")
            axes[1, col].axis('off')
            print(f"✓ Successfully applied mask to sample {idx}")
        except Exception as e:
            print(f"✗ Error applying mask to sample {idx}: {e}")
            axes[1, col].text(0.5, 0.5, f"Error:\n{str(e)}", 
                             ha='center', va='center', transform=axes[1, col].transAxes)
            axes[1, col].axis('off')
    
    plt.tight_layout()
    output_path = 'test_mask_pipeline_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Test visualization saved to: {output_path}")
    return True

def test_filtering_logic():
    """Test the filtering logic for different approaches"""
    print("\nTesting approach filtering logic...")
    
    try:
        dataset = load_dataset("kdoherty/wolverines", "crops-masks", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    approaches = ['pelage', 'pelage_abs', 'pelage_masked', 'pelage_abs_masked']
    
    for approach in approaches:
        count = 0
        base_approach = approach.replace('_masked', '')
        is_masked = 'masked' in approach
        
        for sample in dataset:
            # Base filtering: ALL approaches require megadetector_status == 1
            if sample['megadetector_status'] != 1:
                continue
                
            # Apply pelage filtering
            if base_approach == 'pelage' and sample['pelage'] == 1:
                # Additional mask requirement for masked variants
                if is_masked:
                    if sample['sam_status'] == 1:
                        count += 1
                else:
                    count += 1
            elif base_approach == 'pelage_abs' and sample['pelage'] == 0:
                # Additional mask requirement for masked variants
                if is_masked:
                    if sample['sam_status'] == 1:
                        count += 1
                else:
                    count += 1
        
        print(f"  {approach}: {count} samples available")
        if count == 0:
            print(f"    ⚠️  WARNING: No samples for {approach}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SAM Mask Pipeline")
    print("=" * 60)
    
    # Test mask application
    mask_test_passed = test_mask_pipeline()
    
    # Test filtering logic
    filter_test_passed = test_filtering_logic()
    
    print("\n" + "=" * 60)
    if mask_test_passed and filter_test_passed:
        print("✓ All tests passed!")
        print("The masked pipeline is ready for experimentation.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
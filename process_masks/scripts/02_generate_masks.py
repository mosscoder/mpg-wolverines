#!/usr/bin/env python3
"""
Script 02: Generate Masks with SAM-2 (Hybrid Prompting)
Uses detections from a JSON file to generate segmented masks. This version
uses a hybrid approach, prompting the model with both the bounding box and
its center point to improve segmentation accuracy.
"""

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import Sam2Processor, Sam2Model, infer_device # UPDATED
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

def process_batch_masks(images, bboxes, model, processor, device, kernel):
    """
    Process multiple images in a single batch for efficiency
    
    Args:
        images: List of PIL Images
        bboxes: List of bounding boxes [x, y, w, h]
        model: SAM-2 model
        processor: SAM-2 processor
        device: torch device
        kernel: Pre-created morphological kernel
    
    Returns:
        List of processed masks (numpy arrays)
    """
    if not images:
        return []
    
    # --- 1. Prepare Batch Prompts ---
    batch_boxes = []
    batch_points = []
    batch_labels = []
    
    for bbox in bboxes:
        x, y, w, h = bbox
        # Bounding box prompt (XYXY format)
        batch_boxes.append([[x, y, x + w, y + h]])
        
        # Point prompt at center of box
        center_x, center_y = x + w / 2, y + h / 2
        batch_points.append([[[center_x, center_y]]])
        batch_labels.append([[1]])  # 1 = foreground point
    
    # --- 2. Batch Process ---
    inputs = processor(
        images=images,
        input_boxes=batch_boxes,
        input_points=batch_points,
        input_labels=batch_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process all masks at once
    all_masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )
    
    # --- 3. Process Each Mask ---
    processed_masks = []
    for i, masks in enumerate(all_masks):
        binary_mask = masks[0, 0].numpy().astype(np.uint8) * 255
        
        # Apply morphological closing with pre-created kernel
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_mask = np.zeros_like(binary_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        processed_masks.append(final_mask)
    
    return processed_masks


def apply_mask_to_crop(image, bbox_xywh, mask):
    """
    Memory-efficient cropping and mask application
    
    Args:
        image: PIL Image
        bbox_xywh: Bounding box [x, y, w, h]
        mask: Binary mask (numpy array)
    
    Returns:
        PIL Image (RGBA) - cropped and masked
    """
    x, y, w, h = bbox_xywh
    
    # Crop image first to reduce memory usage
    crop_box = (int(x), int(y), int(x + w), int(y + h))
    cropped_image = image.crop(crop_box)
    cropped_mask = mask[int(y):int(y+h), int(x):int(x+w)]
    
    # Convert to RGBA and apply mask
    cropped_np = np.array(cropped_image)
    rgba_image = np.concatenate([cropped_np, np.full((cropped_np.shape[0], cropped_np.shape[1], 1), 255, dtype=np.uint8)], axis=-1)
    rgba_image[:, :, 3] = cropped_mask
    
    return Image.fromarray(rgba_image, 'RGBA')


def main(json_path: str, output_dir: str, kernel_size: int, batch_size: int = 8):
    """
    Main function to set up models, load data, and orchestrate batch image processing.
    """
    print("Starting the SAM-2 mask generation pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to '{output_dir}/'")
    print(f"Batch size: {batch_size}")

    # Use infer_device for automatic device selection
    device = infer_device()
    print(f"Using device: {device}")

    print("Loading the 'kdoherty/wolverines' dataset...")
    try:
        dataset = load_dataset("kdoherty/wolverines", split="train", trust_remote_code=True)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    print("Loading the Segment Anything 2.1 Model...")
    try:
        model_id = "facebook/sam2.1-hiera-base-plus"
        model = Sam2Model.from_pretrained(model_id).to(device)
        processor = Sam2Processor.from_pretrained(model_id)
        print("SAM-2.1 model loaded.")
    except Exception as e:
        print(f"Failed to load SAM-2.1 model. Error: {e}")
        return

    print(f"Loading detection data from '{json_path}'...")
    try:
        with open(json_path, 'r') as f:
            detection_data = json.load(f)
        detections = detection_data.get("detections", {})
        print("Detection data loaded.")
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_path}' is not a valid JSON file.")
        return

    # Pre-create morphological kernel for efficiency
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Filter valid detections
    valid_items = [(k, v) for k, v in detections.items() 
                   if v.get("detections") and v["original_index"] < len(dataset)]
    
    print(f"Dataset contains {len(dataset)} images.")
    print(f"JSON contains {len(detections)} total entries, {len(valid_items)} valid for processing.")

    print("\nProcessing detections in batches...")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(valid_items), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(valid_items))
        
        # Prepare batch data
        batch_images = []
        batch_bboxes = []
        batch_keys = []
        batch_indices = []
        
        for i in range(batch_start, batch_end):
            image_key, details = valid_items[i]
            original_index = details["original_index"]
            
            try:
                image = dataset[original_index]["image"].convert("RGB")
                bbox = details["detections"][0]["bbox"]
                
                batch_images.append(image)
                batch_bboxes.append(bbox)
                batch_keys.append(image_key)
                batch_indices.append(i)
                
            except Exception as e:
                print(f"Error loading {image_key}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Process entire batch
        try:
            batch_masks = process_batch_masks(batch_images, batch_bboxes, model, processor, device, kernel)
            
            # Save results
            for j, mask in enumerate(batch_masks):
                if j < len(batch_keys):
                    image_key = batch_keys[j]
                    bbox = batch_bboxes[j]
                    image = batch_images[j]
                    
                    # Apply mask and save
                    output_path = f"{output_dir}/{image_key}_sam_crop.png"
                    masked_crop = apply_mask_to_crop(image, bbox, mask)
                    masked_crop.save(output_path)
                    
        except Exception as e:
            print(f"Error processing batch {batch_start}-{batch_end}: {e}")
            # Fallback to individual processing
            for j in range(len(batch_images)):
                try:
                    image_key = batch_keys[j]
                    bbox = batch_bboxes[j]
                    image = batch_images[j]
                    output_path = f"{output_dir}/{image_key}_sam_crop.png"
                    
                    # Process single image
                    single_mask = process_batch_masks([image], [bbox], model, processor, device, kernel)[0]
                    masked_crop = apply_mask_to_crop(image, bbox, single_mask)
                    masked_crop.save(output_path)
                    
                except Exception as e:
                    print(f"Error processing {batch_keys[j] if j < len(batch_keys) else 'unknown'}: {e}")

    print(f"\nProcessing complete! Check the '{output_dir}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images with SAM-2.1 using hybrid box-and-point prompts from a JSON file."
    )
    parser.add_argument(
        "json_path", type=str, help="Path to the input JSON file with wildlife detections."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="masked_and_cropped_wolverines",
        help="Directory to save output images."
    )
    parser.add_argument(
        "-k", "--kernel_size", type=int, default=50,
        help="Size of the kernel for morphological closing to clean the mask."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=8,
        help="Number of images to process in parallel (default: 8)."
    )
    args = parser.parse_args()
    main(args.json_path, args.output_dir, args.kernel_size, args.batch_size)
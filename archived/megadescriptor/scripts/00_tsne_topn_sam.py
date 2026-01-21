#!/usr/bin/env python3
"""
Script 00: t-SNE Visualization of MegaDescriptor Embeddings (Top-N Individuals - SAM-Masked Pelage)
Creates an interactive Bokeh plot showing the embedding space of SAM-masked pelage-visible images
from the top N individuals (or all individuals), with color indicating individual ID.
Uses SAM-cropped and masked images instead of original dataset images.
"""

import sys
import os
import argparse
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from PIL import Image
import base64
from io import BytesIO
import timm
from glob import glob

# Bokeh imports for interactive plotting
from bokeh.plotting import figure, save, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category20, Set3
from bokeh.layouts import column
from bokeh.models import Div

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import load_wolverines_dataset, set_all_seeds
from collections import defaultdict


def get_available_mask_indices(mask_dir):
    """
    Get set of dataset indices for which SAM masks are available
    
    Args:
        mask_dir: Directory containing SAM-masked images
    
    Returns:
        Set of integers representing available dataset indices
    """
    mask_files = glob(os.path.join(mask_dir, "image_*_sam_crop.png"))
    indices = set()
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        # Extract index from "image_XXXXXX_sam_crop.png"
        try:
            index_str = filename.split('_')[1]
            index = int(index_str)
            indices.add(index)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse index from {filename}")
            continue
    
    print(f"Found {len(indices)} SAM-masked images in {mask_dir}")
    return indices


def create_megadescriptor_transform_sam():
    """Create MegaDescriptor-specific transform pipeline for SAM-cropped images"""
    import torchvision.transforms as T
    from PIL import Image
    
    print("MegaDescriptor transform for SAM crops: RGBA→RGB conversion → 384x384 resize + normalize")
    
    return T.Compose([
        T.Resize(size=(384, 384), interpolation=Image.LANCZOS),
        T.ToTensor(), 
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def create_thumbnail_transform_sam():
    """Create transform for generating thumbnails from SAM-cropped images"""
    import torchvision.transforms as T
    from PIL import Image
    
    return T.Compose([
        T.Resize(size=(256, 256), interpolation=Image.LANCZOS),
    ])


def filter_dataset_for_sam_masks_and_individuals(train_dataset, test_dataset, individual_ids, available_indices):
    """Filter datasets to only include SAM-masked samples from specified individuals with pelage visible"""
    if individual_ids:
        print(f"Filtering datasets for individuals: {', '.join(individual_ids)} (SAM-masked pelage visible only)")
    else:
        print("Filtering datasets for all individuals (SAM-masked pelage visible only)")
    print("Note: SAM masks only available for training set")
    
    def is_sam_masked_target_individual_with_pelage(sample, dataset_idx):
        return (dataset_idx in available_indices and
                (individual_ids is None or sample.get('id', 'Unknown') in individual_ids) and 
                sample.get('label', 0) == 1)  # Only pelage visible
    
    # Filter training dataset (only training set has masks)
    filtered_train = []
    for i, sample in enumerate(train_dataset):
        if is_sam_masked_target_individual_with_pelage(sample, i):
            filtered_train.append((sample, i))
    
    # No test dataset filtering since masks only exist for training set
    filtered_test = []
    
    print(f"Filtered training dataset: {len(filtered_train)} SAM-masked pelage-visible samples")
    print(f"Test dataset: 0 samples (no SAM masks available for test set)")
    
    return filtered_train, filtered_test


def load_sam_masked_image(mask_dir, dataset_idx):
    """Load SAM-masked image for given dataset index"""
    mask_path = os.path.join(mask_dir, f"image_{dataset_idx:06d}_sam_crop.png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"SAM mask not found: {mask_path}")
    
    # Load RGBA image and convert to RGB
    rgba_image = Image.open(mask_path).convert('RGBA')
    # Create white background and composite
    white_bg = Image.new('RGB', rgba_image.size, (255, 255, 255))
    rgb_image = Image.alpha_composite(white_bg.convert('RGBA'), rgba_image).convert('RGB')
    
    return rgb_image


def extract_features_and_thumbnails_sam(filtered_train, filtered_test, mask_dir, model, transform, thumbnail_transform, device, batch_size=16):
    """Extract MegaDescriptor features and generate thumbnails using SAM-masked images"""
    model.eval()
    features = []
    labels = []
    individual_ids = []
    pelage_labels = []
    thumbnails = []
    dates = []
    dataset_splits = []
    
    # Combine filtered datasets
    all_samples = [(sample, idx, 'train') for sample, idx in filtered_train] + \
                  [(sample, idx, 'test') for sample, idx in filtered_test]
    
    print(f"Extracting features from {len(all_samples)} SAM-masked samples...")
    
    # Process in batches for memory efficiency
    for i in range(0, len(all_samples), batch_size):
        batch_end = min(i + batch_size, len(all_samples))
        batch_data = all_samples[i:batch_end]
        
        # Prepare batch
        batch_images = []
        batch_labels = []
        batch_ids = []
        batch_pelage = []
        batch_thumbnails = []
        batch_dates = []
        batch_splits = []
        
        for sample, dataset_idx, split in batch_data:
            try:
                # Load SAM-masked image
                sam_image = load_sam_masked_image(mask_dir, dataset_idx)
                
                # Extract features using MegaDescriptor transform
                img_tensor = transform(sam_image)
                batch_images.append(img_tensor)
                
                # Generate thumbnail
                thumbnail_img = thumbnail_transform(sam_image)
                
                # Convert thumbnail to base64
                buffer = BytesIO()
                thumbnail_img.save(buffer, format='PNG')
                thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                batch_thumbnails.append(f"data:image/png;base64,{thumbnail_b64}")
                
                batch_labels.append(sample['label'])
                batch_ids.append(sample.get('id', 'Unknown'))
                batch_pelage.append(sample.get('label', 0))
                batch_dates.append(sample.get('date', 'Unknown'))
                batch_splits.append(split)
                
            except Exception as e:
                print(f"Error loading SAM mask for index {dataset_idx}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack into batch tensor for feature extraction
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Extract features
        with torch.no_grad():
            batch_features = model(batch_tensor).cpu()
        
        features.append(batch_features)
        labels.extend(batch_labels)
        individual_ids.extend(batch_ids)
        pelage_labels.extend(batch_pelage)
        thumbnails.extend(batch_thumbnails)
        dates.extend(batch_dates)
        dataset_splits.extend(batch_splits)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_data)}/{len(all_samples)} samples")
    
    # Concatenate all features
    all_features = torch.cat(features, dim=0)
    print(f"Extracted features shape: {all_features.shape}")
    
    return (all_features, torch.tensor(labels), individual_ids, 
            torch.tensor(pelage_labels), thumbnails, dates, dataset_splits)


def perform_tsne(features, perplexity=30, max_iter=1000, random_state=42):
    """Perform t-SNE dimensionality reduction"""
    print(f"Performing t-SNE with perplexity={perplexity}, max_iter={max_iter}...")
    
    # Ensure features are contiguous in memory
    features_np = features.cpu().numpy().copy()
    
    # Use exact method for stability with smaller datasets
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, 
                random_state=random_state, verbose=1, method='exact', n_jobs=1)
    
    tsne_features = tsne.fit_transform(features_np)
    
    print(f"t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    return tsne_features


def main():
    parser = argparse.ArgumentParser(description='Create t-SNE visualization of MegaDescriptor embeddings for top-N individuals (SAM-masked pelage)')
    parser.add_argument('--output_dir', type=str, default='megadescriptor/figures',
                       help='Directory to save output HTML file')
    parser.add_argument('--mask_dir', type=str, 
                       default='process_masks/results/masked_and_cropped_wolverines',
                       help='Directory containing SAM-masked images')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu', 'mps'],
                       default='mps', help='Device to use for feature extraction')
    parser.add_argument('--perplexity', type=int, default=15,
                       help='t-SNE perplexity parameter (lower for fewer samples)')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='t-SNE number of iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for t-SNE')
    parser.add_argument('--top_n', type=int, default=None,
                       help='Number of top individuals to include (default: None = all individuals)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MegaDescriptor t-SNE Embedding Visualization (Top-N Individuals - SAM-Masked)")
    print("=" * 80)
    
    # Set random seed
    set_all_seeds(args.seed)
    
    # Get available SAM mask indices
    available_indices = get_available_mask_indices(args.mask_dir)
    if not available_indices:
        print(f"No SAM masks found in {args.mask_dir}")
        return
    
    # Determine which individuals to use
    if args.top_n is not None:
        print(f"Loading top {args.top_n} individuals configuration...")
        config_path = 'individual_id/results/feasible_individuals.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        individuals_sorted = config['individuals_sorted_by_pelage']
        feasible_individuals = individuals_sorted[:args.top_n]
        print(f"Using top {args.top_n} individuals: {', '.join(feasible_individuals)}")
    else:
        print("Using all available individuals with SAM masks")
        feasible_individuals = None  # Will be determined from available data
    
    # Load both training and test datasets
    print("\nLoading training and test datasets...")
    train_dataset, test_dataset = load_wolverines_dataset()
    print(f"Full training dataset: {len(train_dataset)} samples")
    print(f"Full test dataset: {len(test_dataset)} samples")
    
    # Filter for SAM-masked samples from specified individuals with pelage visible
    filtered_train, filtered_test = filter_dataset_for_sam_masks_and_individuals(
        train_dataset, test_dataset, feasible_individuals, available_indices)
    
    total_samples = len(filtered_train) + len(filtered_test)
    print(f"Total SAM-masked samples: {total_samples}")
    print(f"  Training: {len(filtered_train)} samples")
    print(f"  Test: {len(filtered_test)} samples")
    
    if total_samples == 0:
        print("No samples found matching criteria (SAM-masked + specified individuals + pelage visible)")
        return
    
    # Print dataset statistics
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for sample, _ in filtered_train:
        individual_id = sample.get('id', 'Unknown')
        train_counts[individual_id] += 1
    
    for sample, _ in filtered_test:
        individual_id = sample.get('id', 'Unknown')
        test_counts[individual_id] += 1
    
    # Get actual individuals found in data
    all_found_individuals = sorted(set(train_counts.keys()) | set(test_counts.keys()))
    
    # Update feasible_individuals if using all individuals
    if feasible_individuals is None:
        feasible_individuals = all_found_individuals
    
    print(f"\nDataset statistics for {len(feasible_individuals)} individuals (SAM-masked pelage visible):") 
    for individual_id in feasible_individuals:
        train_count = train_counts[individual_id]
        test_count = test_counts[individual_id]
        total_count = train_count + test_count
        if total_count > 0:
            print(f"  {individual_id}: {total_count} total ({train_count} train, {test_count} test)")
    
    # Create MegaDescriptor model
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif args.device == "gpu" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    model = model.to(device).eval()
    
    transform = create_megadescriptor_transform_sam()
    thumbnail_transform = create_thumbnail_transform_sam()
    
    print(f"\nUsing device: {device}")
    print(f"Model: MegaDescriptor-L-384")
    print(f"SAM mask directory: {args.mask_dir}")
    
    # Extract features and thumbnails from SAM-masked images
    start_time = time.time()
    
    (features, labels, individual_ids, pelage_labels, 
     thumbnails, dates, dataset_splits) = extract_features_and_thumbnails_sam(
        filtered_train, filtered_test, args.mask_dir, model, transform, thumbnail_transform, device, batch_size=16
    )
    
    feature_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_time/60:.1f} minutes")
    
    if len(features) == 0:
        print("No features extracted - no valid SAM masks found")
        return
    
    # Perform t-SNE
    print(f"\nPerforming t-SNE dimensionality reduction...")
    start_time = time.time()
    
    tsne_features = perform_tsne(
        features, 
        perplexity=args.perplexity, 
        max_iter=args.max_iter, 
        random_state=args.seed
    )
    
    tsne_time = time.time() - start_time
    print(f"t-SNE completed in {tsne_time/60:.1f} minutes")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Create output filename based on selection
    if args.top_n is not None:
        output_filename = f'tsne_top{args.top_n}_sam_masked.html'
    else:
        output_filename = 'tsne_all_sam_masked.html'
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Create interactive plot
    print(f"\nCreating interactive Bokeh visualization...")
    create_bokeh_plot_sam(tsne_features, individual_ids, pelage_labels, 
                         thumbnails, dates, dataset_splits, output_path, feasible_individuals, args.top_n)
    
    # Print summary statistics
    unique_individuals = len(set(individual_ids))
    n_total = len(pelage_labels)
    n_train = sum(1 for s in dataset_splits if s == 'train')
    n_test = sum(1 for s in dataset_splits if s == 'test')
    
    print(f"\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Individuals: {', '.join(feasible_individuals)}")
    print(f"Total SAM-masked samples: {n_total}")
    print(f"All samples: SAM-masked pelage visible only")
    print(f"Training samples: {n_train} ({100*n_train/n_total:.1f}%)")
    print(f"Test samples: {n_test} ({100*n_test/n_total:.1f}%)")
    print(f"Feature extraction time: {feature_time/60:.1f} minutes")
    print(f"t-SNE time: {tsne_time/60:.1f} minutes")
    print(f"Interactive plot: {output_path}")


def create_color_palette(individual_ids):
    """Create color palette for individual IDs"""
    unique_ids = sorted(list(set(individual_ids)))
    n_colors = len(unique_ids)
    
    # Use Category20 palette for up to 20 individuals
    if n_colors <= 20:
        colors = Category20[max(3, n_colors)][:n_colors]
    else:
        # Cycle through Category20 if more than 20 individuals
        base_colors = Category20[20]
        colors = [base_colors[i % 20] for i in range(n_colors)]
    
    # Create mapping from individual ID to color
    color_map = {uid: colors[i] for i, uid in enumerate(unique_ids)}
    
    return color_map, unique_ids


def create_bokeh_plot_sam(tsne_features, individual_ids, pelage_labels, thumbnails, dates, dataset_splits, output_path, feasible_individuals, top_n=None):
    """Create interactive Bokeh plot for SAM-masked data (training set only)"""
    
    # Create color mapping
    color_map, unique_ids = create_color_palette(individual_ids)
    colors = [color_map[uid] for uid in individual_ids]
    
    # All samples are from training set (since masks only exist for training)
    shapes = ['circle'] * len(individual_ids)  # All circles since all are training
    split_status = ['Train'] * len(individual_ids)
    
    # Create data source
    source = ColumnDataSource(data=dict(
        x=tsne_features[:, 0],
        y=tsne_features[:, 1],
        individual_id=individual_ids,
        split_status=split_status,
        color=colors,
        shape=shapes,
        thumbnail=thumbnails,
        date=dates
    ))
    
    # Create figure
    p = figure(
        width=1000, 
        height=800,
        title=f"MegaDescriptor Embedding Space ({'Top ' + str(top_n) if top_n else 'All'} Individuals - SAM-Masked Training Set)",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add training samples (all circles)
    p.circle('x', 'y', size=8, color='color', alpha=0.7, 
            source=source, legend_label="Training Set (SAM-Masked)")
    
    # Configure hover tool with thumbnail
    hover = HoverTool(tooltips="""
        <div>
            <div>
                <img src="@thumbnail" style="width:256px; height:256px; border:1px solid #ccc;">
            </div>
            <div style="margin-top:5px;">
                <span style="font-weight:bold;">Individual:</span> @individual_id<br>
                <span style="font-weight:bold;">Dataset:</span> @split_status<br>
                <span style="font-weight:bold;">Date:</span> @date<br>
                <span style="font-weight:bold;">Position:</span> (@x{0.00}, @y{0.00})
            </div>
        </div>
    """)
    
    p.add_tools(hover)
    
    # Styling
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label = "t-SNE Component 1"
    p.yaxis.axis_label = "t-SNE Component 2"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    # Create summary statistics
    n_individuals = len(unique_ids)
    n_total = len(pelage_labels)
    
    title_suffix = f"Top {top_n}" if top_n else "All"
    summary_text = f"""
    <h3>MegaDescriptor Embedding Visualization ({title_suffix} Individuals - SAM-Masked Training Set)</h3>
    <p><strong>Individuals:</strong> {', '.join(feasible_individuals)}</p>
    <p><strong>Total samples:</strong> {n_total} (training set only)</p>
    <p><strong>All samples:</strong> SAM-masked pelage visible only (detected climbing poses)</p>
    <p><strong>Note:</strong> Only training set shown - SAM masks not available for test set</p>
    <p><strong>Instructions:</strong> Hover over points to see 256x256 SAM-masked thumbnails. Colors indicate individuals.</p>
    """
    
    summary_div = Div(text=summary_text, width=1000)
    
    # Combine plot and summary
    layout = column(summary_div, p)
    
    # Save plot
    output_file(output_path)
    save(layout)
    
    print(f"Interactive plot saved to: {output_path}")
    print(f"Open in browser to explore {n_total} SAM-masked training samples from {n_individuals} individuals")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 00: t-SNE Visualization of MegaDescriptor Embeddings (Top 3 Individuals - Pelage Only)
Creates an interactive Bokeh plot showing the embedding space of pelage-visible images
from the top 3 individuals, with shape indicating train/test split and color indicating individual ID.
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


def create_megadescriptor_transform():
    """Create MegaDescriptor-specific transform pipeline"""
    import torchvision.transforms as T
    from utils.preprocessing import ProportionalCrop
    from PIL import Image
    
    print("MegaDescriptor transform: proportional crop (keep center 50% width, 90% height) → 384x384 resize + normalize")
    
    return T.Compose([
        ProportionalCrop(top_frac=0.05, bottom_frac=0.05, left_frac=0.25, right_frac=0.25),
        T.Resize(size=(384, 384), interpolation=Image.LANCZOS),
        T.ToTensor(), 
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def create_thumbnail_transform():
    """Create transform for generating thumbnails"""
    import torchvision.transforms as T
    from utils.preprocessing import ProportionalCrop
    from PIL import Image
    
    return T.Compose([
        ProportionalCrop(top_frac=0.05, bottom_frac=0.05, left_frac=0.25, right_frac=0.25),
        T.Resize(size=(256, 256), interpolation=Image.LANCZOS),
    ])


def filter_dataset_for_individuals_pelage_only(dataset, individual_ids):
    """Filter dataset to only include specified individuals with pelage visible (label==1)"""
    print(f"Filtering dataset for individuals: {', '.join(individual_ids)} (pelage visible only)")
    
    # Create filter function
    def is_target_individual_with_pelage(sample):
        return (sample.get('id', 'Unknown') in individual_ids and 
                sample.get('label', 0) == 1)  # Only pelage visible
    
    # Apply filter
    filtered_dataset = dataset.filter(is_target_individual_with_pelage)
    
    print(f"Filtered dataset: {len(filtered_dataset)} pelage-visible samples from {len(individual_ids)} individuals")
    
    return filtered_dataset


def extract_features_and_thumbnails(dataset, model, transform, thumbnail_transform, device, batch_size=16):
    """Extract MegaDescriptor features and generate thumbnails for all samples"""
    model.eval()
    features = []
    labels = []
    individual_ids = []
    pelage_labels = []
    thumbnails = []
    dates = []
    
    print(f"Extracting features and thumbnails from {len(dataset)} samples...")
    
    # Process in batches for memory efficiency
    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        batch_samples = [dataset[j] for j in range(i, batch_end)]
        
        # Prepare batch
        batch_images = []
        batch_labels = []
        batch_ids = []
        batch_pelage = []
        batch_thumbnails = []
        batch_dates = []
        
        for sample in batch_samples:
            # Extract features using MegaDescriptor transform
            img_tensor = transform(sample['image'])
            batch_images.append(img_tensor)
            
            # Generate thumbnail
            thumbnail_img = thumbnail_transform(sample['image'])
            
            # Convert thumbnail to base64
            buffer = BytesIO()
            thumbnail_img.save(buffer, format='PNG')
            thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            batch_thumbnails.append(f"data:image/png;base64,{thumbnail_b64}")
            
            batch_labels.append(sample['label'])
            batch_ids.append(sample.get('id', 'Unknown'))
            batch_pelage.append(sample.get('label', 0))  # All samples are pelage=1 now
            batch_dates.append(sample.get('date', 'Unknown'))
        
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
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_samples)}/{len(dataset)} samples")
    
    # Concatenate all features
    all_features = torch.cat(features, dim=0)
    print(f"Extracted features shape: {all_features.shape}")
    
    return (all_features, torch.tensor(labels), individual_ids, 
            torch.tensor(pelage_labels), thumbnails, dates)


def perform_tsne(features, perplexity=30, max_iter=1000, random_state=42):
    """Perform t-SNE dimensionality reduction"""
    print(f"Performing t-SNE with perplexity={perplexity}, max_iter={max_iter}...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, 
                random_state=random_state, verbose=1)
    
    tsne_features = tsne.fit_transform(features.numpy())
    
    print(f"t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    return tsne_features


def create_color_palette(individual_ids):
    """Create color palette for individual IDs"""
    unique_ids = sorted(list(set(individual_ids)))
    n_colors = len(unique_ids)
    
    # Use distinct colors for the 3 individuals
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    # Create mapping from individual ID to color
    color_map = {uid: colors[i % len(colors)] for i, uid in enumerate(unique_ids)}
    
    return color_map, unique_ids


def create_bokeh_plot(tsne_features, individual_ids, pelage_labels, thumbnails, dates, dataset_splits, output_path):
    """Create interactive Bokeh plot"""
    
    # Create color mapping
    color_map, unique_ids = create_color_palette(individual_ids)
    colors = [color_map[uid] for uid in individual_ids]
    
    # Create shape mapping (circle=train, square=test)
    shapes = ['circle' if split == 'train' else 'square' for split in dataset_splits]
    split_status = [split.title() for split in dataset_splits]
    
    # Create data source
    source = ColumnDataSource(data=dict(
        x=tsne_features[:, 0],
        y=tsne_features[:, 1],
        individual_id=individual_ids,
        split_status=split_status,
        dataset_split=dataset_splits,
        color=colors,
        shape=shapes,
        thumbnail=thumbnails,
        date=dates
    ))
    
    # Create figure
    p = figure(
        width=1000, 
        height=800,
        title="MegaDescriptor Embedding Space (Top 3 Individuals - Pelage Only)",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add points with different shapes for train/test split
    # Training samples (circles)
    train_source = ColumnDataSource(data={
        k: [v[i] for i in range(len(individual_ids)) if dataset_splits[i] == 'train']
        for k, v in source.data.items()
    })
    
    if len(train_source.data['x']) > 0:
        p.circle('x', 'y', size=8, color='color', alpha=0.7, 
                source=train_source, legend_label="Training Set")
    
    # Test samples (squares)
    test_source = ColumnDataSource(data={
        k: [v[i] for i in range(len(individual_ids)) if dataset_splits[i] == 'test']
        for k, v in source.data.items()
    })
    
    if len(test_source.data['x']) > 0:
        p.square('x', 'y', size=8, color='color', alpha=0.7,
                source=test_source, legend_label="Test Set")
    
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
    n_train = sum(1 for s in dataset_splits if s == 'train')
    n_test = sum(1 for s in dataset_splits if s == 'test')
    
    summary_text = f"""
    <h3>MegaDescriptor Embedding Visualization (Top 3 Individuals - Pelage Only)</h3>
    <p><strong>Individuals:</strong> {', '.join(unique_ids)}</p>
    <p><strong>Total samples:</strong> {n_total}</p>
    <p><strong>All samples:</strong> Pelage visible only (climbing poses)</p>
    <p><strong>Training samples:</strong> {n_train} ({100*n_train/n_total:.1f}%)</p>
    <p><strong>Test samples:</strong> {n_test} ({100*n_test/n_total:.1f}%)</p>
    <p><strong>Instructions:</strong> Hover over points to see 256x256 image thumbnails. Click legend items to hide/show groups.</p>
    <p><strong>Shape code:</strong> Circle = Training set, Square = Test set</p>
    """
    
    summary_div = Div(text=summary_text, width=1000)
    
    # Combine plot and summary
    layout = column(summary_div, p)
    
    # Save plot
    output_file(output_path)
    save(layout)
    
    print(f"Interactive plot saved to: {output_path}")
    print(f"Open in browser to explore {n_total} samples from {n_individuals} individuals")


def main():
    parser = argparse.ArgumentParser(description='Create t-SNE visualization of MegaDescriptor embeddings for top 3 individuals (pelage only)')
    parser.add_argument('--output_dir', type=str, default='megadescriptor/figures',
                       help='Directory to save output HTML file')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu', 'mps'],
                       default='mps', help='Device to use for feature extraction')
    parser.add_argument('--perplexity', type=int, default=15,
                       help='t-SNE perplexity parameter (lower for fewer samples)')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='t-SNE number of iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for t-SNE')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MegaDescriptor t-SNE Embedding Visualization (Top 3 Individuals - Pelage Only)")
    print("=" * 80)
    
    # Set random seed
    set_all_seeds(args.seed)
    
    # Load top 3 individuals (same as 01 scripts)
    print("Loading top 3 individuals configuration...")
    config_path = 'individual_id/results/feasible_individuals.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    individuals_sorted = config['individuals_sorted_by_pelage']
    feasible_individuals = individuals_sorted[:3]
    print(f"Using top 3 individuals: {', '.join(feasible_individuals)}")
    
    # Load both training and test datasets
    print("\nLoading training and test datasets...")
    train_dataset, test_dataset = load_wolverines_dataset()
    print(f"Full training dataset: {len(train_dataset)} samples")
    print(f"Full test dataset: {len(test_dataset)} samples")
    
    # Filter for top 3 individuals with pelage visible only
    filtered_train = filter_dataset_for_individuals_pelage_only(train_dataset, feasible_individuals)
    filtered_test = filter_dataset_for_individuals_pelage_only(test_dataset, feasible_individuals)
    
    # Combine datasets with dataset split labels
    combined_samples = []
    dataset_splits = []
    
    # Add training samples
    for sample in filtered_train:
        combined_samples.append(sample)
        dataset_splits.append('train')
    
    # Add test samples
    for sample in filtered_test:
        combined_samples.append(sample)
        dataset_splits.append('test')
    
    print(f"Combined dataset: {len(combined_samples)} pelage-visible samples")
    print(f"  Training: {len(filtered_train)} samples")
    print(f"  Test: {len(filtered_test)} samples")
    
    # Print dataset statistics
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for sample in filtered_train:
        individual_id = sample.get('id', 'Unknown')
        train_counts[individual_id] += 1
    
    for sample in filtered_test:
        individual_id = sample.get('id', 'Unknown')
        test_counts[individual_id] += 1
    
    print("\nDataset statistics for top 3 individuals (pelage visible only):")
    for individual_id in feasible_individuals:
        train_count = train_counts[individual_id]
        test_count = test_counts[individual_id]
        total_count = train_count + test_count
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
    
    transform = create_megadescriptor_transform()
    thumbnail_transform = create_thumbnail_transform()
    
    print(f"\nUsing device: {device}")
    print(f"Model: MegaDescriptor-L-384")
    
    # Create a temporary dataset-like object for feature extraction
    class CombinedDataset:
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    combined_dataset = CombinedDataset(combined_samples)
    
    # Extract features and thumbnails
    start_time = time.time()
    
    (features, labels, individual_ids, pelage_labels, 
     thumbnails, dates) = extract_features_and_thumbnails(
        combined_dataset, model, transform, thumbnail_transform, device, batch_size=16
    )
    
    feature_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_time/60:.1f} minutes")
    
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
    output_path = os.path.join(args.output_dir, 'tsne_top3_pelage_only.html')
    
    # Create interactive plot
    print(f"\nCreating interactive Bokeh visualization...")
    create_bokeh_plot(tsne_features, individual_ids, pelage_labels, 
                     thumbnails, dates, dataset_splits, output_path)
    
    # Print summary statistics
    unique_individuals = len(set(individual_ids))
    n_total = len(pelage_labels)
    n_train = sum(1 for s in dataset_splits if s == 'train')
    n_test = sum(1 for s in dataset_splits if s == 'test')
    
    print(f"\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Individuals: {', '.join(feasible_individuals)}")
    print(f"Total samples: {n_total}")
    print(f"All samples: Pelage visible only (climbing poses)")
    print(f"Training samples: {n_train} ({100*n_train/n_total:.1f}%)")
    print(f"Test samples: {n_test} ({100*n_test/n_total:.1f}%)")
    print(f"Feature extraction time: {feature_time/60:.1f} minutes")
    print(f"t-SNE time: {tsne_time/60:.1f} minutes")
    print(f"Interactive plot: {output_path}")


if __name__ == "__main__":
    main()
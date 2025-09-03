#!/usr/bin/env python3
"""
Script 00: t-SNE Visualization of MegaDescriptor Embeddings
Creates an interactive Bokeh plot showing the embedding space of the full training set
with shape indicating pelage visibility and color indicating individual ID.
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
            batch_pelage.append(sample.get('label', 0))
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
    
    # Use Category20 and Set3 palettes for more colors
    if n_colors <= 20:
        colors = Category20[20] if n_colors > 3 else Category20[max(3, n_colors)]
    else:
        # Extend with Set3 if we need more colors
        colors = Category20[20] + Set3[12]
    
    # Create mapping from individual ID to color
    color_map = {uid: colors[i % len(colors)] for i, uid in enumerate(unique_ids)}
    
    return color_map, unique_ids


def stratified_sample_by_individual(dataset, sample_fraction=0.1, seed=42):
    """Stratify sample by (individual_id, pelage) to get representative subset"""
    np.random.seed(seed)
    
    # Group samples by (individual_id, pelage) combinations
    strata_groups = defaultdict(list)
    for i, sample in enumerate(dataset):
        individual_id = sample.get('id', 'Unknown')
        pelage_status = sample.get('label', 0)
        key = (individual_id, pelage_status)
        strata_groups[key].append(i)
    
    sampled_indices = []
    sampling_stats = []
    
    # Sample from each stratum
    for (individual_id, pelage_status), indices in strata_groups.items():
        n_samples = len(indices)
        n_to_sample = max(1, int(n_samples * sample_fraction))  # At least 1 sample
        
        # Randomly sample from this stratum
        sampled = np.random.choice(indices, size=n_to_sample, replace=False)
        sampled_indices.extend(sampled)
        
        pelage_str = 'visible' if pelage_status == 1 else 'not_visible'
        sampling_stats.append({
            'individual_id': individual_id,
            'pelage_status': pelage_str,
            'total_samples': n_samples,
            'sampled': n_to_sample,
            'fraction': n_to_sample / n_samples
        })
    
    # Sort indices to maintain order
    sampled_indices = sorted(sampled_indices)
    
    # Print sampling summary
    unique_individuals = len(set([s['individual_id'] for s in sampling_stats]))
    visible_strata = len([s for s in sampling_stats if s['pelage_status'] == 'visible'])
    not_visible_strata = len([s for s in sampling_stats if s['pelage_status'] == 'not_visible'])
    
    print(f"Stratified sampling: {len(sampled_indices)} samples from {len(strata_groups)} strata")
    print(f"Individuals: {unique_individuals}")
    print(f"Pelage visible strata: {visible_strata}")
    print(f"Pelage not visible strata: {not_visible_strata}")
    print(f"Sample fraction: {sample_fraction} (min 1 sample per stratum)")
    
    # Create subset dataset
    sampled_dataset = dataset.select(sampled_indices)
    
    return sampled_dataset, sampling_stats


def create_bokeh_plot(tsne_features, individual_ids, pelage_labels, thumbnails, dates, output_path):
    """Create interactive Bokeh plot"""
    
    # Create color mapping
    color_map, unique_ids = create_color_palette(individual_ids)
    colors = [color_map[uid] for uid in individual_ids]
    
    # Create shape mapping (circle=visible, square=not visible)
    shapes = ['circle' if pelage == 1 else 'square' for pelage in pelage_labels]
    pelage_status = ['Visible' if pelage == 1 else 'Not Visible' for pelage in pelage_labels]
    
    # Create data source
    source = ColumnDataSource(data=dict(
        x=tsne_features[:, 0],
        y=tsne_features[:, 1],
        individual_id=individual_ids,
        pelage_status=pelage_status,
        pelage_label=pelage_labels.numpy(),
        color=colors,
        shape=shapes,
        thumbnail=thumbnails,
        date=dates
    ))
    
    # Create figure
    p = figure(
        width=1000, 
        height=800,
        title="MegaDescriptor Embedding Space (t-SNE)",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add points with different shapes for pelage visibility
    # Visible pelage (circles)
    visible_source = ColumnDataSource(data={
        k: [v[i] for i in range(len(individual_ids)) if pelage_labels[i] == 1]
        for k, v in source.data.items()
    })
    
    if len(visible_source.data['x']) > 0:
        p.circle('x', 'y', size=8, color='color', alpha=0.7, 
                source=visible_source, legend_label="Pelage Visible")
    
    # Not visible pelage (squares)
    not_visible_source = ColumnDataSource(data={
        k: [v[i] for i in range(len(individual_ids)) if pelage_labels[i] == 0]
        for k, v in source.data.items()
    })
    
    if len(not_visible_source.data['x']) > 0:
        p.square('x', 'y', size=8, color='color', alpha=0.7,
                source=not_visible_source, legend_label="Pelage Not Visible")
    
    # Configure hover tool with thumbnail
    hover = HoverTool(tooltips="""
        <div>
            <div>
                <img src="@thumbnail" style="width:256px; height:256px; border:1px solid #ccc;">
            </div>
            <div style="margin-top:5px;">
                <span style="font-weight:bold;">Individual:</span> @individual_id<br>
                <span style="font-weight:bold;">Pelage:</span> @pelage_status<br>
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
    n_visible = sum(pelage_labels.numpy())
    n_total = len(pelage_labels)
    
    summary_text = f"""
    <h3>MegaDescriptor Embedding Visualization</h3>
    <p><strong>Dataset:</strong> Stratified sample of training set</p>
    <p><strong>Total samples:</strong> {n_total}</p>
    <p><strong>Individuals:</strong> {n_individuals}</p>
    <p><strong>Pelage visible:</strong> {n_visible} ({100*n_visible/n_total:.1f}%)</p>
    <p><strong>Pelage not visible:</strong> {n_total - n_visible} ({100*(n_total-n_visible)/n_total:.1f}%)</p>
    <p><strong>Instructions:</strong> Hover over points to see 256x256 image thumbnails. Click legend items to hide/show groups.</p>
    <p><strong>Shape code:</strong> Circle = Pelage visible (climbing), Square = Pelage not visible (normal pose)</p>
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
    parser = argparse.ArgumentParser(description='Create t-SNE visualization of MegaDescriptor embeddings')
    parser.add_argument('--output_dir', type=str, default='megadescriptor/figures',
                       help='Directory to save output HTML file')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu', 'mps'],
                       default='mps', help='Device to use for feature extraction')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                       help='Fraction of samples to use per individual (default: 1.0)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='t-SNE number of iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling and t-SNE')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MegaDescriptor t-SNE Embedding Visualization")
    print("=" * 80)
    
    # Set random seed
    set_all_seeds(args.seed)
    
    # Load full training dataset
    print("Loading full training dataset...")
    train_dataset, _ = load_wolverines_dataset()
    print(f"Full training dataset: {len(train_dataset)} samples")
    
    # Stratified sampling by individual
    print(f"\nPerforming stratified sampling ({args.sample_fraction} per individual)...")
    sampled_dataset, sampling_stats = stratified_sample_by_individual(
        train_dataset, sample_fraction=args.sample_fraction, seed=args.seed
    )
    print(f"Sampled dataset: {len(sampled_dataset)} samples")
    
    # Print sampling statistics
    total_individuals = len(sampling_stats)
    avg_fraction = np.mean([s['fraction'] for s in sampling_stats])
    print(f"Individuals represented: {total_individuals}")
    print(f"Average sampling fraction: {avg_fraction:.3f}")
    
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
    
    print(f"Using device: {device}")
    print(f"Model: MegaDescriptor-L-384")
    
    # Extract features and generate thumbnails
    start_time = time.time()
    
    (features, labels, individual_ids, pelage_labels, 
     thumbnails, dates) = extract_features_and_thumbnails(
        sampled_dataset, model, transform, thumbnail_transform, device, batch_size=16
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
    output_path = os.path.join(args.output_dir, f'tsne_sampled_{args.sample_fraction:.1f}_training_set.html')
    
    # Create interactive plot
    print(f"\nCreating interactive Bokeh visualization...")
    create_bokeh_plot(tsne_features, individual_ids, pelage_labels, 
                     thumbnails, dates, output_path)
    
    # Print summary statistics
    unique_individuals = len(set(individual_ids))
    n_visible = sum(pelage_labels.numpy())
    n_total = len(pelage_labels)
    
    print(f"\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {n_total}")
    print(f"Unique individuals: {unique_individuals}")
    print(f"Pelage visible: {n_visible} ({100*n_visible/n_total:.1f}%)")
    print(f"Pelage not visible: {n_total - n_visible} ({100*(n_total-n_visible)/n_total:.1f}%)")
    print(f"Feature extraction time: {feature_time/60:.1f} minutes")
    print(f"t-SNE time: {tsne_time/60:.1f} minutes")
    print(f"Interactive plot: {output_path}")


if __name__ == "__main__":
    main()
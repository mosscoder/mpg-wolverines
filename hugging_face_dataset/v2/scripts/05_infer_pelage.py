#!/usr/bin/env python3
"""
Script 05: Infer Pelage on Out-of-Sample Crops
Runs inference on all crop images that were not used for training the pelage model.
Generates pelage probabilities (probability of clear markings vs none/partial).
"""

import sys
import os
import argparse
import time
import sqlite3
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path (from hugging_face_dataset/v2/scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
v2_dir = os.path.dirname(script_dir)  # Go up to v2
hugging_face_dir = os.path.dirname(v2_dir)  # Go up to hugging_face_dataset
wolverines_root = os.path.dirname(hugging_face_dir)  # Go up to wolverines root
sys.path.append(wolverines_root)

from utils.preprocessing import get_standard_transform
from utils.models import create_model


class InferenceDataset(torch.utils.data.Dataset):
    """Dataset class for inference on crop images"""
    def __init__(self, df, crops_dir, transform):
        self.df = df.reset_index(drop=True)
        self.crops_dir = crops_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.crops_dir, row['crop_filename'])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, idx  # Return index to match back to metadata


def get_best_device():
    """Get the best available device with cascade: CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name()})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"Using device: {device_name}")
    return device


def load_pelage_labeled_crops(db_path):
    """Load list of crop filenames that were used for training"""
    print(f"Loading training crop filenames from: {db_path}")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # Get all labeled crop filenames
    query = '''
    SELECT DISTINCT crop_filename as filename
    FROM pelage_labels 
    WHERE label IS NOT NULL
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    labeled_filenames = set(df['filename'].tolist())
    print(f"Found {len(labeled_filenames)} labeled crops used for training")
    
    return labeled_filenames


def load_all_crop_metadata(metadata_csv):
    """Load metadata for all crops"""
    print(f"Loading crop metadata from: {metadata_csv}")
    
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    
    df = pd.read_csv(metadata_csv)
    
    print(f"Loaded metadata for {len(df)} crops")
    print(f"Columns: {list(df.columns)}")
    
    return df


def filter_out_of_sample_crops(metadata_df, labeled_filenames):
    """Filter to only include crops that were NOT used for training"""
    print("Filtering to out-of-sample crops...")
    
    # Filter out crops that were used for training
    out_of_sample_df = metadata_df[~metadata_df['crop_filename'].isin(labeled_filenames)].copy()
    
    print(f"Out-of-sample crops: {len(out_of_sample_df)}")
    print(f"In-sample (training) crops: {len(metadata_df) - len(out_of_sample_df)}")
    
    return out_of_sample_df


def check_images_exist(df, crops_dir):
    """Check which images actually exist on disk"""
    print("Checking which crop images exist on disk...")
    
    existing_crops = []
    missing_crops = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        image_path = os.path.join(crops_dir, row['crop_filename'])
        if os.path.exists(image_path):
            existing_crops.append(idx)
        else:
            missing_crops.append(row['crop_filename'])
    
    existing_df = df.loc[existing_crops].copy()
    
    print(f"Found {len(existing_df)} existing crop images")
    if missing_crops:
        print(f"Missing {len(missing_crops)} crop images")
        
    return existing_df


def load_production_model(model_path, device):
    """Load the trained production model"""
    print(f"Loading production model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found: {model_path}")
    
    # Create model architecture (same as training)
    model = create_model(device=str(device))
    
    # Load only the classifier head weights
    classifier_state = torch.load(model_path, map_location=device)
    model.classifier.load_state_dict(classifier_state)
    
    # Ensure entire model is on the correct device
    model = model.to(device)
    
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    return model


def create_inference_dataloader(df, crops_dir, device, batch_size=32, resize_size=224):
    """Create dataloader for inference"""
    print(f"Creating dataloader with batch_size={batch_size}")
    
    # Use same transform as training (without augmentation)
    transform = get_standard_transform(resize_size=resize_size)
    
    # Create dataset using module-level class
    dataset = InferenceDataset(df, crops_dir, transform)
    
    # Set num_workers based on device to avoid hanging issues
    if device.type == 'mps':
        num_workers = 0  # Avoid multiprocessing issues on MPS
    elif device.type == 'cuda':
        num_workers = 4  # Use multiprocessing on CUDA
    else:
        num_workers = 0  # Use single process on CPU
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != torch.device('cpu'))
    )
    
    print(f"Created dataloader with {len(dataset)} samples")
    return dataloader


def run_inference(model, dataloader, device):
    """Run inference on all crops and return probabilities"""
    print("Running inference on out-of-sample crops...")
    
    all_probs = []
    all_indices = []
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, indices) in enumerate(tqdm(dataloader, desc="Inference")):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert to probabilities (softmax)
            probs = torch.softmax(outputs, dim=1)
            
            # Get probability of class 1 (clear pelage markings)
            pelage_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(pelage_probs)
            all_indices.extend(indices.cpu().numpy())
    
    print(f"Inference completed on {len(all_probs)} crops")
    
    # Sort by original indices to maintain order
    sorted_results = sorted(zip(all_indices, all_probs))
    _, sorted_probs = zip(*sorted_results)
    
    return np.array(sorted_probs)


def save_inference_results(df, pelage_probs, output_path):
    """Save inference results with metadata"""
    print(f"Saving inference results to: {output_path}")
    
    # Add pelage probabilities to dataframe
    results_df = df.copy()
    results_df['pelage_probability'] = pelage_probs
    
    # Round probabilities for readability
    results_df['pelage_probability'] = results_df['pelage_probability'].round(6)
    
    # Calculate bbox area if not present
    if 'bbox_area' not in results_df.columns:
        results_df['bbox_area'] = results_df['bbox_width'] * results_df['bbox_height']
    
    # Sort by id and ymdh for consistency
    results_df = results_df.sort_values(['id', 'ymdh']).reset_index(drop=True)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as CSV
    results_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print(f"\n" + "="*60)
    print("INFERENCE RESULTS SUMMARY")
    print("="*60)
    print(f"Total crops processed: {len(results_df):,}")
    print(f"Unique individuals: {results_df['id'].nunique()}")
    print(f"Date range: {results_df['ymdh'].min()} - {results_df['ymdh'].max()}")
    print(f"\nPelage probability statistics:")
    print(f"  Mean: {results_df['pelage_probability'].mean():.4f}")
    print(f"  Median: {results_df['pelage_probability'].median():.4f}")
    print(f"  Min: {results_df['pelage_probability'].min():.4f}")
    print(f"  Max: {results_df['pelage_probability'].max():.4f}")
    print(f"  Std: {results_df['pelage_probability'].std():.4f}")
    
    # Threshold analysis
    high_confidence_threshold = 0.8
    low_confidence_threshold = 0.2
    
    high_conf = (results_df['pelage_probability'] >= high_confidence_threshold).sum()
    low_conf = (results_df['pelage_probability'] <= low_confidence_threshold).sum()
    medium_conf = len(results_df) - high_conf - low_conf
    
    print(f"\nPelage confidence distribution:")
    print(f"  High confidence (≥{high_confidence_threshold}): {high_conf:,} ({100*high_conf/len(results_df):.1f}%)")
    print(f"  Medium confidence ({low_confidence_threshold}-{high_confidence_threshold}): {medium_conf:,} ({100*medium_conf/len(results_df):.1f}%)")
    print(f"  Low confidence (≤{low_confidence_threshold}): {low_conf:,} ({100*low_conf/len(results_df):.1f}%)")
    
    print(f"\nResults saved to: {output_path}")
    
    return results_df


def create_probability_histogram(probabilities, output_dir, filename_prefix="pelage_probabilities"):
    """Create and save histogram of pelage probabilities"""
    print("Creating probability histogram...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with density
    ax1.hist(probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax1.axvline(np.mean(probabilities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(probabilities):.3f}')
    ax1.axvline(np.median(probabilities), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(probabilities):.3f}')
    ax1.axvline(0.5, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Decision Threshold: 0.5')
    ax1.set_xlabel('Pelage Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Pelage Probabilities (Density)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram with counts
    counts, bins, patches = ax2.hist(probabilities, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(np.mean(probabilities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(probabilities):.3f}')
    ax2.axvline(np.median(probabilities), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(probabilities):.3f}')
    ax2.axvline(0.5, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Decision Threshold: 0.5')
    ax2.set_xlabel('Pelage Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Pelage Probabilities (Counts)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""
    Total Samples: {len(probabilities):,}
    Mean: {np.mean(probabilities):.4f}
    Median: {np.median(probabilities):.4f}
    Std: {np.std(probabilities):.4f}
    Min: {np.min(probabilities):.4f}
    Max: {np.max(probabilities):.4f}
    
    Confidence Levels:
    High (≥0.8): {(probabilities >= 0.8).sum():,} ({100*(probabilities >= 0.8).sum()/len(probabilities):.1f}%)
    Medium (0.2-0.8): {((probabilities >= 0.2) & (probabilities < 0.8)).sum():,} ({100*((probabilities >= 0.2) & (probabilities < 0.8)).sum()/len(probabilities):.1f}%)
    Low (≤0.2): {(probabilities <= 0.2).sum():,} ({100*(probabilities <= 0.2).sum()/len(probabilities):.1f}%)
    
    Binary Classification (threshold=0.5):
    Predicted Positive: {(probabilities >= 0.5).sum():,} ({100*(probabilities >= 0.5).sum()/len(probabilities):.1f}%)
    Predicted Negative: {(probabilities < 0.5).sum():,} ({100*(probabilities < 0.5).sum()/len(probabilities):.1f}%)
    """
    
    # Add text box to second subplot
    ax2.text(1.05, 0.5, stats_text.strip(), transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save histogram
    histogram_path = os.path.join(output_dir, f"{filename_prefix}_histogram.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to: {histogram_path}")
    
    # Also create a simple box plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    box_plot = ax.boxplot(probabilities, vert=False, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
    ax.set_xlabel('Pelage Probability')
    ax.set_title('Box Plot of Pelage Probabilities')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(np.mean(probabilities), 1.1, f'Mean: {np.mean(probabilities):.3f}', 
            ha='center', va='bottom', fontsize=12, color='blue')
    ax.text(np.median(probabilities), 0.9, f'Median: {np.median(probabilities):.3f}', 
            ha='center', va='top', fontsize=12, color='red')
    
    boxplot_path = os.path.join(output_dir, f"{filename_prefix}_boxplot.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plot saved to: {boxplot_path}")
    
    return histogram_path, boxplot_path


def main():
    parser = argparse.ArgumentParser(description='Infer pelage probabilities on out-of-sample crops')
    parser.add_argument('--model_path', type=str,
                       default='pelage_sorting/results/02_production/production_model_classifier.pth',
                       help='Path to trained production model')
    parser.add_argument('--db_path', type=str,
                       default='hugging_face_dataset/v2/data/labeling/pelage_labels.db',
                       help='Path to pelage labels database')
    parser.add_argument('--crops_dir', type=str,
                       default='hugging_face_dataset/v2/init_crops',
                       help='Directory containing crop images')
    parser.add_argument('--metadata_csv', type=str,
                       default='hugging_face_dataset/v2/metadata/crop_metadata.csv',
                       help='Path to crop metadata CSV')
    parser.add_argument('--output_path', type=str,
                       default='hugging_face_dataset/v2/data/inference/pelage_inference_results.csv',
                       help='Path to save inference results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--resize_size', type=int, default=224,
                       help='Image resize size (should match training)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PELAGE INFERENCE ON OUT-OF-SAMPLE CROPS")
    print("="*60)
    print(f"Production model: {args.model_path}")
    print(f"Labels database: {args.db_path}")
    print(f"Crops directory: {args.crops_dir}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Output path: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Resize size: {args.resize_size}")
    
    try:
        # Get best available device
        device = get_best_device()
        
        # Load labeled crop filenames (used for training)
        labeled_filenames = load_pelage_labeled_crops(args.db_path)
        
        # Load all crop metadata
        metadata_df = load_all_crop_metadata(args.metadata_csv)
        
        # Filter to out-of-sample crops only
        out_of_sample_df = filter_out_of_sample_crops(metadata_df, labeled_filenames)
        
        # Check which images actually exist
        existing_df = check_images_exist(out_of_sample_df, args.crops_dir)
        
        if len(existing_df) == 0:
            print("No out-of-sample crop images found!")
            return
        
        # Load production model
        model = load_production_model(args.model_path, device)
        
        # Create dataloader
        dataloader = create_inference_dataloader(
            existing_df, args.crops_dir, device,
            batch_size=args.batch_size, 
            resize_size=args.resize_size
        )
        
        # Run inference
        start_time = time.time()
        pelage_probs = run_inference(model, dataloader, device)
        inference_time = time.time() - start_time
        
        print(f"Inference completed in {inference_time/60:.1f} minutes")
        print(f"Processing rate: {len(pelage_probs)/inference_time:.1f} crops/second")
        
        # Save results
        results_df = save_inference_results(existing_df, pelage_probs, args.output_path)
        
        # Create probability histogram
        output_dir = os.path.dirname(args.output_path)
        if not output_dir:
            output_dir = "."
        histogram_path, boxplot_path = create_probability_histogram(
            pelage_probs, output_dir, "pelage_inference_probabilities"
        )
        
        print(f"\n✓ Pelage inference completed successfully!")
        print(f"Results saved to: {args.output_path}")
        print(f"Histogram saved to: {histogram_path}")
        print(f"Box plot saved to: {boxplot_path}")
        
    except Exception as e:
        print(f"\nError during pelage inference: {e}")
        raise


if __name__ == "__main__":
    main()
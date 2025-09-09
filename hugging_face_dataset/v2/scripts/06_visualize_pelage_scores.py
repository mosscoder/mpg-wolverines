#!/usr/bin/env python3
"""
Script 06: Visualize Pelage Scores Grid
Creates an image grid showing pelage scores by individual and score decile.
Rows represent individuals, columns represent pelage score deciles (0-0.1, 0.1-0.2, etc.)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
import seaborn as sns


def load_pelage_inference_results(csv_path):
    """Load pelage inference results from CSV and ensure unique images"""
    print(f"Loading pelage inference results from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Inference results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} inference records")
    
    # Remove duplicates based on crop_filename to ensure unique images
    original_count = len(df)
    df = df.drop_duplicates(subset=['crop_filename'], keep='first')
    duplicate_count = original_count - len(df)
    
    if duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicate images, keeping {len(df)} unique images")
    
    print(f"Unique individuals: {df['id'].nunique()}")
    print(f"Pelage score range: {df['pelage_probability'].min():.4f} - {df['pelage_probability'].max():.4f}")
    
    return df


def get_min_median_max(df):
    """Get min, median, and max pelage score images for each individual"""
    print("Finding min, median, and max pelage scores for each individual...")
    
    score_labels = ['Max', 'Median', 'Min']  # Top to bottom order
    selected_images = []
    
    for individual_id in df['id'].unique():
        individual_data = df[df['id'] == individual_id].copy()
        
        if len(individual_data) == 0:
            continue
            
        # Sort by pelage score
        individual_data = individual_data.sort_values('pelage_probability')
        
        # Get min, median, max images
        min_img = individual_data.iloc[0]  # Lowest score
        max_img = individual_data.iloc[-1]  # Highest score
        
        # Get median (middle image by score)
        median_idx = len(individual_data) // 2
        median_img = individual_data.iloc[median_idx]
        
        # Add score level labels
        min_img = min_img.copy()
        min_img['score_level'] = 'Min'
        
        median_img = median_img.copy()  
        median_img['score_level'] = 'Median'
        
        max_img = max_img.copy()
        max_img['score_level'] = 'Max'
        
        selected_images.extend([max_img, median_img, min_img])  # Max first (top row)
    
    selected_df = pd.DataFrame(selected_images).reset_index(drop=True)
    
    print(f"Selected {len(selected_df)} images ({len(selected_df)//3} individuals × 3 score levels)")
    
    # Print individual statistics
    individual_stats = df.groupby('id')['pelage_probability'].agg(['min', 'median', 'max'])
    print("Individual score ranges:")
    for individual_id in individual_stats.index[:5]:  # Show first 5
        stats = individual_stats.loc[individual_id]
        print(f"  {individual_id}: Min={stats['min']:.3f}, Median={stats['median']:.3f}, Max={stats['max']:.3f}")
    if len(individual_stats) > 5:
        print(f"  ... and {len(individual_stats)-5} more individuals")
    
    return selected_df, score_labels


def select_individuals_for_grid(selected_df, max_individuals=20):
    """Select top individuals by image count and filter the selected images"""
    print(f"Selecting top {max_individuals} individuals for visualization...")
    
    # Count images per individual and select top ones
    individual_counts = selected_df.groupby('id').size().sort_values(ascending=False)
    selected_ids = individual_counts.head(max_individuals).index.tolist()
    
    print(f"Selected {len(selected_ids)} individuals:")
    for idx, individual_id in enumerate(selected_ids[:5]):  # Show top 5
        total_images = individual_counts[individual_id]
        print(f"  {idx+1}. {individual_id}: {total_images} score levels")
    if len(selected_ids) > 5:
        print(f"  ... and {len(selected_ids)-5} more")
    
    # Filter to selected individuals
    final_df = selected_df[selected_df['id'].isin(selected_ids)].copy()
    
    print(f"Final selection: {len(final_df)} images for {len(selected_ids)} individuals")
    
    return final_df, selected_ids


def create_image_grid(sampled_df, selected_ids, score_labels, crops_dir, output_path, 
                     image_size=(80, 80), figsize_per_cell=(1, 1)):
    """Create and save the image grid visualization (rotated: bins as rows, individuals as columns)"""
    print("Creating image grid visualization...")
    
    n_individuals = len(selected_ids)
    n_levels = len(score_labels)
    
    # Calculate figure size (score levels are rows, individuals are columns)
    fig_width = n_individuals * figsize_per_cell[0]
    fig_height = n_levels * figsize_per_cell[1]
    
    fig, axes = plt.subplots(n_levels, n_individuals, 
                            figsize=(fig_width, fig_height),
                            facecolor='white')
    
    # Handle single row or column cases
    if n_levels == 1:
        axes = axes.reshape(1, -1)
    elif n_individuals == 1:
        axes = axes.reshape(-1, 1)
    
    # Set up the grid (i=score_level, j=individual)
    for i, score_level in enumerate(score_labels):
        for j, individual_id in enumerate(selected_ids):
            ax = axes[i, j]
            
            # Find image for this score_level-individual combination
            mask = (sampled_df['id'] == individual_id) & (sampled_df['score_level'] == score_level)
            matching_images = sampled_df[mask]
            
            if len(matching_images) > 0:
                # Load and display image
                row = matching_images.iloc[0]
                image_path = os.path.join(crops_dir, row['crop_filename'])
                
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_resized = img.resize(image_size, Image.Resampling.LANCZOS)
                    ax.imshow(img_resized)
                    
                    # Show the actual score value
                    score_value = row['pelage_probability']
                    ax.text(0.02, 0.98, f'{score_value:.3f}', 
                           transform=ax.transAxes, fontsize=6, color='white',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
                           va='top', ha='left')
                    
                except Exception as e:
                    # Show placeholder for missing/corrupt images
                    ax.text(0.5, 0.5, 'Missing\nImage', transform=ax.transAxes,
                           ha='center', va='center', fontsize=8, color='red')
                    ax.set_facecolor('lightgray')
            
            else:
                # Show placeholder for missing combinations
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=8, color='gray')
                ax.set_facecolor('whitesmoke')
            
            # Remove axis ticks and labels for individual cells
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add column headers (individual IDs) for top row
            if i == 0:
                ax.set_title(individual_id, fontsize=10, pad=10, weight='bold', rotation=45, ha='left')
            
            # Add row labels (score levels) for leftmost column
            if j == 0:
                ax.set_ylabel(score_level, fontsize=10, rotation=0, 
                             ha='right', va='center', weight='bold')
    
    # Add axis labels
    #fig.text(0.5, 0.02, 'Wolverine Individual', ha='center', fontsize=12, weight='bold')
    fig.text(0.08, 0.5, 'Pelage Score', va='center', rotation=90, fontsize=12, weight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.07, left=0.15, right=0.95)
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Image grid saved to: {output_path}")
    
    return output_path



def main():
    parser = argparse.ArgumentParser(description='Create pelage score visualization grid')
    parser.add_argument('--inference_csv', type=str,
                       default='hugging_face_dataset/v2/data/inference/pelage_inference_results.csv',
                       help='Path to pelage inference results CSV')
    parser.add_argument('--crops_dir', type=str,
                       default='hugging_face_dataset/v2/init_crops',
                       help='Directory containing crop images')
    parser.add_argument('--output_dir', type=str,
                       default='hugging_face_dataset/v2/data/visualization',
                       help='Directory to save visualization outputs')
    parser.add_argument('--max_individuals', type=int, default=20,
                       help='Maximum number of individuals to include in grid')
    parser.add_argument('--image_size', type=int, nargs=2, default=[120, 120],
                       help='Size (width, height) for each image in grid')
    parser.add_argument('--figsize_per_cell', type=float, nargs=2, default=[1.5, 1.5],
                       help='Figure size (width, height) per cell in inches')
    parser.add_argument('--random_state', type=int, default=123,
                       help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PELAGE SCORE VISUALIZATION GRID")
    print("="*70)
    print(f"Inference results: {args.inference_csv}")
    print(f"Crops directory: {args.crops_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max individuals: {args.max_individuals}")
    print(f"Image size: {args.image_size[0]}x{args.image_size[1]} pixels")
    print(f"Figure size per cell: {args.figsize_per_cell[0]}x{args.figsize_per_cell[1]} inches")
    
    try:
        # Load pelage inference results
        df = load_pelage_inference_results(args.inference_csv)
        
        # Get min/median/max for each individual
        selected_df, score_labels = get_min_median_max(df)
        
        # Select individuals for visualization
        sampled_df, selected_ids = select_individuals_for_grid(
            selected_df, max_individuals=args.max_individuals
        )
        
        # Create image grid
        grid_path = os.path.join(args.output_dir, 'pelage_score_grid.png')
        create_image_grid(
            sampled_df, selected_ids, score_labels, args.crops_dir, grid_path,
            image_size=tuple(args.image_size),
            figsize_per_cell=tuple(args.figsize_per_cell)
        )
        
        print(f"\n" + "="*70)
        print("VISUALIZATION COMPLETE")
        print("="*70)
        print(f"📊 Image grid: {grid_path}")
        print(f"\nGrid dimensions: {len(score_labels)} score levels × {len(selected_ids)} individuals")
        print(f"Total cells: {len(selected_ids) * len(score_labels)}")
        print(f"Cells with data: {len(sampled_df)} ({100*len(sampled_df)/(len(selected_ids)*len(score_labels)):.1f}%)")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 07: Create Reidentification Dataset Config
Uses pelage inference results to create a new HuggingFace dataset config.
Stratifies by individual ID and uses the last 10% of unique timestamps per individual for test set.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json



def load_inference_results(inference_csv):
    """Load pelage inference results"""
    print(f"Loading inference results from: {inference_csv}")
    
    if not os.path.exists(inference_csv):
        raise FileNotFoundError(f"Inference results not found: {inference_csv}")
    
    df = pd.read_csv(inference_csv)
    
    # Rename crop_filename to filename for consistency
    if 'crop_filename' in df.columns:
        df = df.rename(columns={'crop_filename': 'filename'})
    
    # Ensure ymdh is integer type
    df['ymdh'] = df['ymdh'].astype(int)
    
    # Rename columns for clarity
    if 'pelage_probability' in df.columns:
        df = df.rename(columns={'pelage_probability': 'pelage_score'})
    if 'confidence' in df.columns:
        df = df.rename(columns={'confidence': 'megadetector_confidence'})
    
    # Calculate bbox area if not present
    if 'bbox_area' not in df.columns and 'bbox_width' in df.columns and 'bbox_height' in df.columns:
        df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    
    # Add data source indicator
    df['data_source'] = 'inference'
    
    print(f"Loaded {len(df)} inference results")
    print(f"Pelage score range: {df['pelage_score'].min():.4f} - {df['pelage_score'].max():.4f}")
    
    return df


def load_megadetector_metadata(metadata_csv, filenames_needed):
    """Load MegaDetector metadata for specific filenames"""
    print(f"Loading MegaDetector metadata from: {metadata_csv}")
    
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    
    # Load full metadata
    df = pd.read_csv(metadata_csv)
    
    # Filter to only needed filenames based on crop_filename
    if 'crop_filename' in df.columns:
        needed_df = df[df['crop_filename'].isin(filenames_needed)].copy()
        # Rename crop_filename to filename for consistency
        needed_df = needed_df.rename(columns={'crop_filename': 'filename'})
    else:
        # If already renamed to filename
        needed_df = df[df['filename'].isin(filenames_needed)].copy()
    
    # Calculate bbox area if not present
    if 'bbox_area' not in needed_df.columns and 'bbox_width' in needed_df.columns and 'bbox_height' in needed_df.columns:
        needed_df['bbox_area'] = needed_df['bbox_width'] * needed_df['bbox_height']
    
    print(f"Loaded metadata for {len(needed_df)} crops")
    
    return needed_df


def merge_inference_with_metadata(inference_df, metadata_df):
    """Merge inference results with metadata"""
    print("Merging inference results with metadata...")
    
    # Ensure inference data has required columns
    if 'id' not in inference_df.columns:
        print("Warning: 'id' column missing from inference data")
    if 'color' not in inference_df.columns:
        print("Warning: 'color' column missing from inference data")
    
    # Select available metadata columns
    metadata_columns = ['filename']
    optional_columns = ['megadetector_confidence', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'bbox_area']
    
    for col in optional_columns:
        if col in metadata_df.columns:
            metadata_columns.append(col)
        else:
            print(f"Warning: '{col}' column not found in metadata")
    
    # Merge with metadata to get bounding box information
    merged_df = inference_df.merge(
        metadata_df[metadata_columns],
        on='filename',
        how='left'
    )
    
    print(f"Combined dataset: {len(merged_df)} total records (inference only)")
    
    # Check for missing metadata
    if 'megadetector_confidence' in merged_df.columns:
        missing_metadata = merged_df['megadetector_confidence'].isna().sum()
        if missing_metadata > 0:
            print(f"Warning: {missing_metadata} records missing MegaDetector metadata")
    
    return merged_df


def create_temporal_split_by_individual(df, test_proportion=0.1, random_state=42):
    """
    Create train/test split by taking the last 10% of unique timestamps per individual.
    This simulates a temporal holdout where we test on the most recent data per individual.
    """
    print(f"Creating temporal split with {test_proportion*100:.1f}% test data per individual...")
    
    np.random.seed(random_state)
    
    train_indices = []
    test_indices = []
    
    individual_stats = []
    
    for individual_id in tqdm(df['id'].unique(), desc="Processing individuals"):
        individual_data = df[df['id'] == individual_id].copy()
        
        # Sort by timestamp
        individual_data = individual_data.sort_values('ymdh')
        
        # Get unique timestamps for this individual
        unique_timestamps = individual_data['ymdh'].unique()
        unique_timestamps = np.sort(unique_timestamps)
        
        # Check if individual has sufficient timestamps for splitting
        min_timestamps_for_split = 2
        
        if len(unique_timestamps) < min_timestamps_for_split:
            # Put all data in training set if insufficient timestamps
            train_timestamps = unique_timestamps
            test_timestamps = np.array([])
            n_train_timestamps = len(unique_timestamps)
            n_test_timestamps = 0
        else:
            # Normal splitting for individuals with sufficient data
            n_test_timestamps = max(1, int(len(unique_timestamps) * test_proportion))
            n_train_timestamps = len(unique_timestamps) - n_test_timestamps
            
            # Take the last n_test_timestamps for test set
            test_timestamps = unique_timestamps[-n_test_timestamps:]
            train_timestamps = unique_timestamps[:-n_test_timestamps]
        
        # Split the data
        train_mask = individual_data['ymdh'].isin(train_timestamps)
        test_mask = individual_data['ymdh'].isin(test_timestamps)
        
        train_indices.extend(individual_data[train_mask].index.tolist())
        test_indices.extend(individual_data[test_mask].index.tolist())
        
        # Track statistics
        individual_stats.append({
            'id': individual_id,
            'total_images': len(individual_data),
            'total_timestamps': len(unique_timestamps),
            'train_timestamps': len(train_timestamps),
            'test_timestamps': len(test_timestamps),
            'train_images': train_mask.sum(),
            'test_images': test_mask.sum()
        })
    
    # Create split dataframes
    train_df = df.loc[train_indices].copy().reset_index(drop=True)
    test_df = df.loc[test_indices].copy().reset_index(drop=True)
    
    # Print split statistics
    print(f"\nTemporal split completed:")
    print(f"  Train: {len(train_df):,} images from {train_df['id'].nunique()} individuals")
    print(f"  Test: {len(test_df):,} images from {test_df['id'].nunique()} individuals")
    
    # Individual-level statistics
    stats_df = pd.DataFrame(individual_stats)
    print(f"\nPer-individual statistics:")
    print(f"  Average images per individual: {stats_df['total_images'].mean():.1f}")
    print(f"  Average timestamps per individual: {stats_df['total_timestamps'].mean():.1f}")
    print(f"  Average train images per individual: {stats_df['train_images'].mean():.1f}")
    print(f"  Average test images per individual: {stats_df['test_images'].mean():.1f}")
    
    return train_df, test_df, stats_df


def print_split_statistics(train_df, test_df):
    """Print detailed statistics about the train/test split"""
    print("\n" + "="*70)
    print("REIDENTIFICATION DATASET SPLIT STATISTICS")
    print("="*70)
    
    total = len(train_df) + len(test_df)
    print(f"Total samples: {total:,}")
    print(f"Train: {len(train_df):,} ({100*len(train_df)/total:.1f}%)")
    print(f"Test: {len(test_df):,} ({100*len(test_df)/total:.1f}%)")
    
    # Data source distribution
    print("\nData source distribution:")
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        print(f"\n{split_name}:")
        source_counts = split_df['data_source'].value_counts()
        for source in ['labeled', 'inference']:
            count = source_counts.get(source, 0)
            pct = 100*count/len(split_df) if len(split_df) > 0 else 0
            print(f"  {source.capitalize()}: {count:,} ({pct:.1f}%)")
    
    # Individual distribution
    print(f"\nUnique individuals:")
    print(f"  Train: {train_df['id'].nunique()}")
    print(f"  Test: {test_df['id'].nunique()}")
    overlap = len(set(train_df['id']) & set(test_df['id']))
    print(f"  Overlap: {overlap} (all individuals should appear in both sets)")
    
    # Color distribution
    print("\nColor distribution:")
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        print(f"\n{split_name}:")
        for color in [0, 1]:
            count = len(split_df[split_df['color'] == color])
            pct = 100*count/len(split_df) if len(split_df) > 0 else 0
            color_name = ['B&W', 'Color'][color]
            print(f"  {color_name}: {count:,} ({pct:.1f}%)")
    
    # Temporal distribution
    print("\nTemporal distribution:")
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        if len(split_df) > 0:
            print(f"\n{split_name}:")
            print(f"  Date range: {split_df['ymdh'].min()} - {split_df['ymdh'].max()}")
            print(f"  Unique timestamps: {split_df['ymdh'].nunique()}")


def validate_dataset(df, crops_dir):
    """Validate dataset completeness"""
    print("Validating dataset completeness...")
    
    required_fields = [
        'id', 'ymdh', 'color', 'filename', 'pelage_score',
        'megadetector_confidence', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'bbox_area'
    ]
    
    # Check for missing columns
    missing_cols = [field for field in required_fields if field not in df.columns]
    if missing_cols:
        raise ValueError(f"ERROR: Missing required columns: {missing_cols}")
    
    # Check for missing values in critical fields
    critical_fields = ['id', 'ymdh', 'color', 'filename', 'pelage_score']
    for field in critical_fields:
        missing_count = df[field].isna().sum()
        if missing_count > 0:
            missing_records = df[df[field].isna()][['id', 'filename']].head(10)
            raise ValueError(
                f"ERROR: {missing_count} records have missing {field}!\n"
                f"Examples:\n{missing_records.to_string(index=False)}"
            )
    
    # Validate pelage scores are in [0, 1]
    invalid_scores = df[(df['pelage_score'] < 0) | (df['pelage_score'] > 1)]
    if len(invalid_scores) > 0:
        raise ValueError(
            f"ERROR: {len(invalid_scores)} records have invalid pelage_score values!\n"
            f"Scores must be between 0 and 1"
        )
    
    # Check that image files exist (sample check for performance)
    print("Sampling image files to verify they exist...")
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    missing_images = []
    for idx, row in sample_df.iterrows():
        image_path = os.path.join(crops_dir, row['filename'])
        if not os.path.exists(image_path):
            missing_images.append({'id': row['id'], 'filename': row['filename']})
    
    if missing_images:
        missing_df = pd.DataFrame(missing_images)
        raise FileNotFoundError(
            f"ERROR: {len(missing_images)} sample image files are missing!\n"
            f"Examples:\n{missing_df.head(5).to_string(index=False)}\n"
            f"Please ensure all crop images exist."
        )
    
    print(f"✓ Successfully validated {len(df)} records")
    print(f"✓ All required fields are complete")
    print(f"✓ All pelage scores are in [0, 1]")
    print(f"✓ Sample of {sample_size} image files exist")


def create_dataset_dict(df, crops_dir):
    """Create dataset dictionary for HuggingFace Dataset"""
    print("Creating dataset dictionary...")
    
    data_dict = {
        'id': df['id'].astype(str).tolist(),
        'ymdh': df['ymdh'].astype(int).tolist(),
        'color': df['color'].astype(int).tolist(),
        'pelage_score': df['pelage_score'].astype(float).tolist(),
        'data_source': df['data_source'].astype(str).tolist(),
        'filename': df['filename'].astype(str).tolist(),
        'megadetector_confidence': df['megadetector_confidence'].astype(float).tolist(),
        'bbox_x': df['bbox_x'].astype(float).tolist(),
        'bbox_y': df['bbox_y'].astype(float).tolist(),
        'bbox_width': df['bbox_width'].astype(float).tolist(),
        'bbox_height': df['bbox_height'].astype(float).tolist(),
        'bbox_area': df['bbox_area'].astype(float).tolist(),
        'image': []
    }
    
    
    # Load images
    print("Loading images...")
    for filename in tqdm(df['filename'], desc="Loading images"):
        image_path = os.path.join(crops_dir, filename)
        data_dict['image'].append(image_path)
    
    return data_dict


def create_dataset(data_dict):
    """Create HuggingFace Dataset with proper schema"""
    print("Creating HuggingFace Dataset...")
    
    # Define features schema
    features = Features({
        'id': Value('string'),                         # Individual ID
        'ymdh': Value('int64'),                        # Date timestamp
        'color': Value('int32'),                       # 0=B&W, 1=Color
        'pelage_score': Value('float32'),              # AI pelage visibility score
        'data_source': Value('string'),                # 'inference'
        'image': HFImage(),                            # Cropped wolverine image
        'filename': Value('string'),                   # Original crop filename
        'megadetector_confidence': Value('float32'),   # MegaDetector wolverine confidence
        'bbox_x': Value('float32'),                    # Bounding box x coordinate
        'bbox_y': Value('float32'),                    # Bounding box y coordinate
        'bbox_width': Value('float32'),                # Bounding box width
        'bbox_height': Value('float32'),               # Bounding box height
        'bbox_area': Value('float32'),                 # Calculated bbox area
    })
    
    # Create dataset
    dataset = Dataset.from_dict(data_dict, features=features)
    
    print(f"Created dataset with {len(dataset)} samples")
    return dataset


def save_splits_locally(train_dataset, test_dataset, output_dir):
    """Save both train and test datasets locally"""
    print(f"Saving train/test splits locally to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train split
    train_dir = os.path.join(output_dir, 'train')
    train_dataset.save_to_disk(train_dir)
    print(f"Train dataset saved to: {train_dir}")
    
    # Save test split
    test_dir = os.path.join(output_dir, 'test')
    test_dataset.save_to_disk(test_dir)
    print(f"Test dataset saved to: {test_dir}")
    
    print("Both splits saved locally successfully")


def print_dataset_summary(dataset, split_name):
    """Print summary of created dataset"""
    print(f"\n{split_name.upper()} DATASET SUMMARY")
    print("="*60)
    
    n_total = len(dataset)
    
    print(f"Total samples: {n_total:,}")
    
    
    # Individual distribution
    ids = dataset['id']
    unique_ids = len(set(ids))
    print(f"Unique individuals: {unique_ids}")
    
    # Pelage score statistics
    scores = dataset['pelage_score']
    print(f"Pelage score statistics:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Min/Max: {np.min(scores):.4f} - {np.max(scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Create 'reidentification' config from pelage inference results")
    parser.add_argument('--inference_csv', type=str,
                       default='hugging_face_dataset/v2/data/inference/pelage_inference_results.csv',
                       help='Path to pelage inference results')
    parser.add_argument('--metadata_csv', type=str,
                       default='hugging_face_dataset/v2/metadata/crop_metadata.csv',
                       help='Path to crop metadata CSV')
    parser.add_argument('--crops_dir', type=str,
                       default='hugging_face_dataset/v2/init_crops',
                       help='Directory containing crop images')
    parser.add_argument('--output_dir', type=str,
                       default='hugging_face_dataset/v2/data/reidentification_dataset',
                       help='Directory to save dataset locally')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push dataset to HuggingFace Hub')
    parser.add_argument('--hub_dataset_name', type=str,
                       default='kdoherty/wolverines',
                       help='HuggingFace dataset name')
    parser.add_argument('--config_name', type=str,
                       default='reidentification',
                       help='Configuration name for HuggingFace dataset')
    parser.add_argument('--test_proportion', type=float, default=0.1,
                       help='Proportion of timestamps per individual for test set (default: 0.1)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Creating 'reidentification' Config for Wolverines Dataset")
    print("="*80)
    print(f"Inference results: {args.inference_csv}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Crops directory: {args.crops_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test proportion: {args.test_proportion}")
    
    try:
        # Load inference results (already contains metadata from script 05)
        inference_df = load_inference_results(args.inference_csv)
        
        # The inference results already contain all metadata, so use directly
        merged_df = inference_df
        
        # Validate dataset
        validate_dataset(merged_df, args.crops_dir)
        
        # Create temporal split by individual
        train_df, test_df, individual_stats = create_temporal_split_by_individual(
            merged_df,
            test_proportion=args.test_proportion,
            random_state=args.random_state
        )
        
        # Print split statistics
        print_split_statistics(train_df, test_df)
        
        # Create dataset dictionaries
        train_dict = create_dataset_dict(train_df, args.crops_dir)
        test_dict = create_dataset_dict(test_df, args.crops_dir)
        
        # Create HuggingFace datasets
        print("\nCreating train dataset...")
        train_dataset = create_dataset(train_dict)
        
        print("Creating test dataset...")
        test_dataset = create_dataset(test_dict)
        
        # Print summaries
        print_dataset_summary(train_dataset, "Train")
        print_dataset_summary(test_dataset, "Test")
        
        # Save locally
        save_splits_locally(train_dataset, test_dataset, args.output_dir)
        
        # Optional: Push to HuggingFace Hub
        if args.push_to_hub:
            print(f"\nPushing dataset to HuggingFace Hub: {args.hub_dataset_name}")
            print(f"Config: {args.config_name}")
            
            # Push train split
            train_dataset.push_to_hub(
                args.hub_dataset_name,
                config_name=args.config_name,
                split='train'
            )
            print(f"✓ Train split pushed successfully")
            
            # Push test split
            test_dataset.push_to_hub(
                args.hub_dataset_name,
                config_name=args.config_name,
                split='test'
            )
            print(f"✓ Test split pushed successfully")
            
            print(f"✓ Dataset successfully pushed to: {args.hub_dataset_name} (config: {args.config_name})")
        
        # Save individual statistics
        individual_stats_path = os.path.join(args.output_dir, 'individual_split_stats.csv')
        individual_stats.to_csv(individual_stats_path, index=False)
        print(f"✓ Individual statistics saved to: {individual_stats_path}")
        
        print(f"\n" + "="*70)
        print("REIDENTIFICATION DATASET CREATION COMPLETE")
        print(f"Local dataset: {args.output_dir}")
        print(f"  - Train split: {args.output_dir}/train")
        print(f"  - Test split: {args.output_dir}/test")
        if args.push_to_hub:
            print(f"HuggingFace: {args.hub_dataset_name} (config: {args.config_name})")
        else:
            print("Use --push_to_hub to upload to HuggingFace Hub")
        print("="*70)
        
    except Exception as e:
        print(f"\n" + "="*70)
        print("REIDENTIFICATION DATASET CREATION FAILED")
        print("="*70)
        print(f"Error: {e}")
        print("\nPlease fix the issues above and try again.")
        print("="*70)
        raise


if __name__ == "__main__":
    main()
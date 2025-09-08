#!/usr/bin/env python3
"""
Script 04: Create 'pelage' Config for Wolverines Dataset
Creates a specialized HuggingFace dataset config with pelage labels and MegaDetector bounding box data.
Only includes labeled images with strict validation - fails loudly if any data is missing.
"""

import os
import sys
import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path for imports
sys.path.append('.')


def load_pelage_labels(db_path):
    """Load pelage labels from SQLite database"""
    print(f"Loading pelage labels from: {db_path}")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # Query only labeled data
    query = '''
    SELECT 
        id, 
        color, 
        ymdh, 
        label, 
        crop_filename as filename,
        image_path
    FROM pelage_labels 
    WHERE label IS NOT NULL
    ORDER BY id, ymdh
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} labeled records")
    print(f"Label distribution:")
    for label_val in [0, 1, 2]:
        count = len(df[df['label'] == label_val])
        label_name = ['None', 'Partial', 'Full'][label_val]
        print(f"  {label_name} ({label_val}): {count} images")
    
    return df


def load_megadetector_metadata(metadata_csv):
    """Load MegaDetector metadata with bounding box data"""
    print(f"Loading MegaDetector metadata from: {metadata_csv}")
    
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"MegaDetector metadata not found: {metadata_csv}")
    
    df = pd.read_csv(metadata_csv)
    
    # Select only needed columns for bbox data
    bbox_columns = [
        'crop_filename', 'confidence', 'bbox_x', 'bbox_y', 
        'bbox_width', 'bbox_height', 'crop_width', 'crop_height'
    ]
    
    # Check if required columns exist
    missing_cols = [col for col in bbox_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in MegaDetector metadata: {missing_cols}")
    
    bbox_df = df[bbox_columns].copy()
    
    # Calculate bbox area
    bbox_df['bbox_area'] = bbox_df['bbox_width'] * bbox_df['bbox_height']
    
    print(f"Loaded MegaDetector metadata for {len(bbox_df)} crops")
    return bbox_df


def merge_pelage_and_bbox_data(pelage_df, bbox_df):
    """Merge pelage labels with MegaDetector bounding box data"""
    print("Merging pelage labels with MegaDetector bounding box data...")
    
    # Merge on crop_filename
    merged_df = pelage_df.merge(
        bbox_df,
        left_on='filename',
        right_on='crop_filename',
        how='left'
    )
    
    # Drop duplicate crop_filename column
    merged_df = merged_df.drop('crop_filename', axis=1)
    
    print(f"Merged dataset has {len(merged_df)} records")
    
    # Check for missing bbox data
    missing_bbox = merged_df['confidence'].isna().sum()
    if missing_bbox > 0:
        print(f"Warning: {missing_bbox} records missing MegaDetector data")
    
    return merged_df


def validate_dataset(df, crops_dir):
    """Validate all required fields exist - fail loudly if any data is missing"""
    print("Validating dataset completeness...")
    
    required_fields = [
        'id', 'ymdh', 'color', 'label', 'filename', 
        'confidence', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'bbox_area'
    ]
    
    # Check for missing columns
    missing_cols = [field for field in required_fields if field not in df.columns]
    if missing_cols:
        raise ValueError(f"ERROR: Missing required columns: {missing_cols}")
    
    # Check for missing values in each field
    for field in required_fields:
        missing_count = df[field].isna().sum()
        if missing_count > 0:
            # Show which records are missing data
            missing_records = df[df[field].isna()][['id', 'filename']].head(10)
            raise ValueError(
                f"ERROR: {missing_count} records have missing {field}!\n"
                f"Examples of missing records:\n{missing_records.to_string(index=False)}\n"
                f"Please fix the missing data and try again."
            )
    
    # Validate label values are in correct range
    invalid_labels = df[~df['label'].isin([0, 1, 2])]
    if len(invalid_labels) > 0:
        raise ValueError(
            f"ERROR: {len(invalid_labels)} records have invalid label values!\n"
            f"Valid labels are 0 (None), 1 (Partial), 2 (Full)\n"
            f"Invalid records:\n{invalid_labels[['id', 'filename', 'label']].to_string(index=False)}"
        )
    
    # Validate all image files exist
    print("Checking that all image files exist...")
    missing_images = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        image_path = os.path.join(crops_dir, row['filename'])
        if not os.path.exists(image_path):
            missing_images.append({'id': row['id'], 'filename': row['filename'], 'path': image_path})
    
    if missing_images:
        missing_df = pd.DataFrame(missing_images)
        raise FileNotFoundError(
            f"ERROR: {len(missing_images)} image files are missing!\n"
            f"Examples:\n{missing_df.head(10).to_string(index=False)}\n"
            f"Please ensure all crop images exist before creating dataset."
        )
    
    # Validate bounding box dimensions are positive
    invalid_bbox = df[(df['bbox_width'] <= 0) | (df['bbox_height'] <= 0)]
    if len(invalid_bbox) > 0:
        raise ValueError(
            f"ERROR: {len(invalid_bbox)} records have invalid bounding box dimensions!\n"
            f"Examples:\n{invalid_bbox[['id', 'filename', 'bbox_width', 'bbox_height']].head(10).to_string(index=False)}"
        )
    
    print(f"✓ Successfully validated {len(df)} records")
    print(f"✓ All required fields are complete")
    print(f"✓ All image files exist")
    print(f"✓ All labels are valid (0, 1, 2)")
    print(f"✓ All bounding boxes are valid")


def create_dataset_dict(df, crops_dir):
    """Create dataset dictionary for HuggingFace Dataset"""
    print("Creating dataset dictionary...")
    
    data_dict = {
        'id': df['id'].astype(str).tolist(),
        'ymdh': df['ymdh'].astype(int).tolist(),
        'color': df['color'].astype(int).tolist(), 
        'label': df['label'].astype(int).tolist(),
        'filename': df['filename'].astype(str).tolist(),
        'confidence': df['confidence'].astype(float).tolist(),
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


def create_stratified_split(df, test_size=0.2, random_state=42):
    """Create stratified train/test split based on id, color, and label"""
    print(f"Creating stratified split with test_size={test_size}...")
    
    # Create stratification key combining id, color, and label
    df['strata'] = df['id'].astype(str) + '_' + df['color'].astype(str) + '_' + df['label'].astype(str)
    
    # Count samples per strata
    strata_counts = df['strata'].value_counts()
    print(f"Found {len(strata_counts)} unique strata (id_color_label combinations)")
    
    # Handle strata with only 1 sample (can't split)
    single_sample_strata = strata_counts[strata_counts == 1].index
    if len(single_sample_strata) > 0:
        print(f"Warning: {len(single_sample_strata)} strata have only 1 sample, assigning to train set")
        single_sample_df = df[df['strata'].isin(single_sample_strata)]
        multi_sample_df = df[~df['strata'].isin(single_sample_strata)]
    else:
        single_sample_df = pd.DataFrame()
        multi_sample_df = df.copy()
    
    # Perform stratified split on multi-sample strata
    if len(multi_sample_df) > 0:
        train_df, test_df = train_test_split(
            multi_sample_df,
            test_size=test_size,
            stratify=multi_sample_df['strata'],
            random_state=random_state
        )
        
        # Add single-sample strata to train set
        if len(single_sample_df) > 0:
            train_df = pd.concat([train_df, single_sample_df], ignore_index=True)
    else:
        train_df = single_sample_df
        test_df = pd.DataFrame()
    
    # Drop temporary strata column
    train_df = train_df.drop('strata', axis=1) if 'strata' in train_df.columns else train_df
    test_df = test_df.drop('strata', axis=1) if len(test_df) > 0 and 'strata' in test_df.columns else test_df
    
    print(f"Split complete: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def print_split_statistics(train_df, test_df):
    """Print statistics about the train/test split"""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    # Overall split
    total = len(train_df) + len(test_df)
    print(f"Total samples: {total:,}")
    print(f"Train: {len(train_df):,} ({100*len(train_df)/total:.1f}%)")
    print(f"Test: {len(test_df):,} ({100*len(test_df)/total:.1f}%)")
    
    # Label distribution per split
    print("\nLabel distribution:")
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        if len(split_df) == 0:
            continue
        print(f"\n{split_name}:")
        for label in [0, 1, 2]:
            count = len(split_df[split_df['label'] == label])
            pct = 100*count/len(split_df) if len(split_df) > 0 else 0
            label_name = ['None', 'Partial', 'Full'][label]
            print(f"  {label_name}: {count:,} ({pct:.1f}%)")
    
    # Unique individuals per split
    print("\nUnique individuals:")
    print(f"  Train: {train_df['id'].nunique()}")
    print(f"  Test: {test_df['id'].nunique()}")
    overlap = len(set(train_df['id']) & set(test_df['id']))
    print(f"  Overlap: {overlap}")
    
    # Color distribution per split
    print("\nColor distribution:")
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        if len(split_df) == 0:
            continue
        print(f"\n{split_name}:")
        for color in [0, 1]:
            count = len(split_df[split_df['color'] == color])
            pct = 100*count/len(split_df) if len(split_df) > 0 else 0
            color_name = ['B&W', 'Color'][color]
            print(f"  {color_name}: {count:,} ({pct:.1f}%)")


def create_dataset(data_dict):
    """Create HuggingFace Dataset with proper schema"""
    print("Creating HuggingFace Dataset...")
    
    # Define features schema
    features = Features({
        'id': Value('string'),              # Individual ID (e.g., "BDF10-M6")
        'ymdh': Value('int64'),             # Date timestamp (YYYYMMDDHHMM)
        'color': Value('int32'),            # 0=B&W, 1=Color
        'label': Value('int32'),            # 0=None, 1=Partial, 2=Full
        'image': HFImage(),                 # Cropped wolverine image
        'filename': Value('string'),        # Original crop filename
        'confidence': Value('float32'),     # MegaDetector confidence score
        'bbox_x': Value('float32'),         # Bounding box x coordinate
        'bbox_y': Value('float32'),         # Bounding box y coordinate
        'bbox_width': Value('float32'),     # Bounding box width
        'bbox_height': Value('float32'),    # Bounding box height
        'bbox_area': Value('float32'),      # Calculated bbox area
    })
    
    # Create dataset
    dataset = Dataset.from_dict(data_dict, features=features)
    
    print(f"Created dataset with {len(dataset)} samples")
    return dataset


def save_dataset_locally(dataset, output_dir):
    """Save dataset locally before pushing to hub"""
    print(f"Saving dataset locally to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved locally successfully")


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


def print_dataset_summary(dataset):
    """Print summary of created dataset"""
    print("\n" + "="*60)
    print("PELAGE DATASET SUMMARY")
    print("="*60)
    
    n_total = len(dataset)
    
    # Label distribution
    labels = dataset['label']
    label_counts = {0: 0, 1: 0, 2: 0}
    for label in labels:
        label_counts[label] += 1
    
    print(f"Total samples: {n_total:,}")
    print(f"Label distribution:")
    print(f"  None (0):    {label_counts[0]:,} ({100*label_counts[0]/n_total:.1f}%)")
    print(f"  Partial (1): {label_counts[1]:,} ({100*label_counts[1]/n_total:.1f}%)")
    print(f"  Full (2):    {label_counts[2]:,} ({100*label_counts[2]/n_total:.1f}%)")
    
    # Individual distribution
    ids = dataset['id']
    unique_ids = len(set(ids))
    print(f"\nUnique individuals: {unique_ids}")
    
    # Color distribution
    colors = dataset['color']
    color_counts = {0: 0, 1: 0}
    for color in colors:
        color_counts[color] += 1
    
    print(f"Image types:")
    print(f"  B&W (0):   {color_counts[0]:,} ({100*color_counts[0]/n_total:.1f}%)")
    print(f"  Color (1): {color_counts[1]:,} ({100*color_counts[1]/n_total:.1f}%)")
    
    # MegaDetector statistics
    confidences = dataset['confidence']
    areas = dataset['bbox_area']
    
    print(f"\nMegaDetector statistics:")
    print(f"  Average confidence: {np.mean(confidences):.3f}")
    print(f"  Min/Max confidence: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
    print(f"  Average bbox area: {np.mean(areas):.1f} pixels")
    print(f"  Min/Max bbox area: {np.min(areas):.1f} - {np.max(areas):.1f} pixels")


def main():
    parser = argparse.ArgumentParser(description="Create 'pelage' config for wolverines dataset")
    parser.add_argument('--db_path', type=str,
                       default='hugging_face_dataset/v2/data/labeling/pelage_labels.db',
                       help='Path to pelage labels SQLite database')
    parser.add_argument('--crops_dir', type=str,
                       default='hugging_face_dataset/v2/init_crops',
                       help='Directory containing crop images')
    parser.add_argument('--metadata_csv', type=str,
                       default='hugging_face_dataset/v2/metadata/crop_metadata.csv',
                       help='Path to MegaDetector metadata CSV')
    parser.add_argument('--output_dir', type=str,
                       default='hugging_face_dataset/v2/data/pelage_dataset',
                       help='Directory to save dataset locally')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push dataset to HuggingFace Hub')
    parser.add_argument('--hub_dataset_name', type=str,
                       default='kdoherty/wolverines',
                       help='HuggingFace dataset name')
    parser.add_argument('--config_name', type=str,
                       default='pelage',
                       help='Configuration name for HuggingFace dataset')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for test split (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Creating 'pelage' Config for Wolverines Dataset")
    print("="*80)
    print(f"Database: {args.db_path}")
    print(f"Crops directory: {args.crops_dir}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Load pelage labels
        pelage_df = load_pelage_labels(args.db_path)
        
        # Load MegaDetector metadata
        bbox_df = load_megadetector_metadata(args.metadata_csv)
        
        # Merge datasets
        merged_df = merge_pelage_and_bbox_data(pelage_df, bbox_df)
        
        # Strict validation - fail loudly if any data is missing
        validate_dataset(merged_df, args.crops_dir)
        
        # Create stratified train/test split
        train_df, test_df = create_stratified_split(
            merged_df, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        
        # Print split statistics
        print_split_statistics(train_df, test_df)
        
        # Create dataset dictionaries for both splits
        train_dict = create_dataset_dict(train_df, args.crops_dir)
        test_dict = create_dataset_dict(test_df, args.crops_dir) if len(test_df) > 0 else None
        
        # Create HuggingFace datasets
        print("\nCreating train dataset...")
        train_dataset = create_dataset(train_dict)
        
        if test_dict is not None:
            print("Creating test dataset...")
            test_dataset = create_dataset(test_dict)
        else:
            test_dataset = None
        
        # Print summaries
        print("\n" + "="*60)
        print("TRAIN DATASET SUMMARY")
        print("="*60)
        print_dataset_summary(train_dataset)
        
        if test_dataset is not None:
            print("\n" + "="*60)
            print("TEST DATASET SUMMARY")
            print("="*60)
            print_dataset_summary(test_dataset)
        
        # Save locally
        if test_dataset is not None:
            save_splits_locally(train_dataset, test_dataset, args.output_dir)
        else:
            save_dataset_locally(train_dataset, args.output_dir)
        
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
            
            # Push test split if it exists
            if test_dataset is not None:
                test_dataset.push_to_hub(
                    args.hub_dataset_name,
                    config_name=args.config_name,
                    split='test'
                )
                print(f"✓ Test split pushed successfully")
            
            print(f"✓ Dataset successfully pushed to: {args.hub_dataset_name} (config: {args.config_name})")
        
        print(f"\n" + "="*60)
        print("DATASET CREATION COMPLETE")
        print(f"Local dataset: {args.output_dir}")
        if test_dataset is not None:
            print(f"  - Train split: {args.output_dir}/train")
            print(f"  - Test split: {args.output_dir}/test")
        if args.push_to_hub:
            print(f"HuggingFace: {args.hub_dataset_name} (config: {args.config_name})")
            if test_dataset is not None:
                print(f"  - Both train and test splits uploaded")
        else:
            print("Use --push_to_hub to upload to HuggingFace Hub")
        print("="*60)
        
    except Exception as e:
        print(f"\n" + "="*60)
        print("DATASET CREATION FAILED")
        print("="*60)
        print(f"Error: {e}")
        print("\nPlease fix the issues above and try again.")
        print("="*60)
        raise


if __name__ == "__main__":
    main()
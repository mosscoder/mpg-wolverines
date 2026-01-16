#!/usr/bin/env python3
"""
Script 08: Reveal Data Provenance
Creates a CSV file documenting the provenance of all files used in both pelage and reidentification datasets.
Extracts filename (full_path), task (pelage or reidentification), and split (train or test) for each sample.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm


def load_dataset_splits(dataset_dir):
    """Load train and test splits from a dataset directory"""
    print(f"Loading dataset from: {dataset_dir}")

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train split not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test split not found: {test_dir}")

    train_dataset = load_from_disk(train_dir)
    test_dataset = load_from_disk(test_dir)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_dataset, test_dataset


def load_crop_metadata(metadata_csv):
    """Load crop metadata and create mapping from crop_filename to original_path"""
    print(f"Loading crop metadata from: {metadata_csv}")

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Crop metadata not found: {metadata_csv}")

    metadata_df = pd.read_csv(metadata_csv)

    # Create mapping from crop_filename to original_path
    filename_mapping = {}
    for _, row in metadata_df.iterrows():
        crop_filename = row['crop_filename']
        original_path = row['original_path']
        filename_mapping[crop_filename] = original_path

    print(f"Loaded mapping for {len(filename_mapping):,} crop files")
    return filename_mapping


def extract_provenance_data(dataset, task_name, split_name, filename_mapping):
    """Extract provenance data from a dataset split using original source paths"""
    print(f"Extracting provenance for {task_name} {split_name} split...")

    provenance_records = []
    missing_mappings = 0

    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {task_name} {split_name}"):
        crop_filename = sample['filename']

        # Look up original source path
        if crop_filename in filename_mapping:
            original_path = filename_mapping[crop_filename]
        else:
            # Fallback if mapping not found
            print(f"Warning: No mapping found for {crop_filename}")
            original_path = f"MISSING_MAPPING_{crop_filename}"
            missing_mappings += 1

        provenance_records.append({
            'filename': original_path,
            'task': task_name,
            'split': split_name
        })

    if missing_mappings > 0:
        print(f"Warning: {missing_mappings} files had missing mappings in {task_name} {split_name}")

    return provenance_records


def create_provenance_csv(pelage_dir, reidentification_dir, metadata_csv, output_csv):
    """Create comprehensive provenance CSV for both datasets"""
    print("="*80)
    print("REVEALING DATA PROVENANCE")
    print("="*80)
    print(f"Pelage dataset: {pelage_dir}")
    print(f"Reidentification dataset: {reidentification_dir}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Output CSV: {output_csv}")

    # Load crop metadata mapping
    print("\n" + "="*60)
    print("LOADING CROP METADATA")
    print("="*60)
    filename_mapping = load_crop_metadata(metadata_csv)

    all_provenance_records = []

    # Load pelage dataset
    print("\n" + "="*60)
    print("PROCESSING PELAGE DATASET")
    print("="*60)
    pelage_train, pelage_test = load_dataset_splits(pelage_dir)

    # Extract pelage provenance
    pelage_train_records = extract_provenance_data(pelage_train, 'pelage', 'train', filename_mapping)
    pelage_test_records = extract_provenance_data(pelage_test, 'pelage', 'test', filename_mapping)

    all_provenance_records.extend(pelage_train_records)
    all_provenance_records.extend(pelage_test_records)

    # Load reidentification dataset
    print("\n" + "="*60)
    print("PROCESSING REIDENTIFICATION DATASET")
    print("="*60)
    reid_train, reid_test = load_dataset_splits(reidentification_dir)

    # Extract reidentification provenance
    reid_train_records = extract_provenance_data(reid_train, 'reidentification', 'train', filename_mapping)
    reid_test_records = extract_provenance_data(reid_test, 'reidentification', 'test', filename_mapping)

    all_provenance_records.extend(reid_train_records)
    all_provenance_records.extend(reid_test_records)

    # Create DataFrame and save
    print(f"\n" + "="*60)
    print("CREATING PROVENANCE CSV")
    print("="*60)
    provenance_df = pd.DataFrame(all_provenance_records)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    provenance_df.to_csv(output_csv, index=False)

    # Print summary statistics
    print(f"Total records: {len(provenance_df):,}")
    print(f"Unique files: {provenance_df['filename'].nunique():,}")
    print("\nBreakdown by task and split:")
    summary = provenance_df.groupby(['task', 'split']).size().reset_index(name='count')
    for _, row in summary.iterrows():
        print(f"  {row['task'].capitalize()} {row['split']}: {row['count']:,}")

    print(f"\nProvenance CSV saved to: {output_csv}")

    # Check for overlapping files between tasks
    print(f"\n" + "="*60)
    print("OVERLAP ANALYSIS")
    print("="*60)
    pelage_files = set(provenance_df[provenance_df['task'] == 'pelage']['filename'])
    reid_files = set(provenance_df[provenance_df['task'] == 'reidentification']['filename'])
    overlap = pelage_files & reid_files

    print(f"Files in pelage dataset: {len(pelage_files):,}")
    print(f"Files in reidentification dataset: {len(reid_files):,}")
    print(f"Files appearing in both datasets: {len(overlap):,}")

    if len(overlap) > 0:
        print(f"Overlap percentage: {100 * len(overlap) / len(pelage_files | reid_files):.1f}%")

    return provenance_df


def main():
    parser = argparse.ArgumentParser(description="Reveal data provenance for pelage and reidentification datasets")
    parser.add_argument('--pelage_dir', type=str,
                       default='hugging_face_dataset/v2/data/pelage_dataset',
                       help='Directory containing pelage dataset splits')
    parser.add_argument('--reidentification_dir', type=str,
                       default='hugging_face_dataset/v2/data/reidentification_dataset',
                       help='Directory containing reidentification dataset splits')
    parser.add_argument('--metadata_csv', type=str,
                       default='hugging_face_dataset/v2/metadata/crop_metadata.csv',
                       help='CSV file containing crop metadata with original paths')
    parser.add_argument('--output_csv', type=str,
                       default='hugging_face_dataset/v2/metadata/utilized_data_provenance.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    try:
        # Validate input directories and files
        if not os.path.exists(args.pelage_dir):
            raise FileNotFoundError(f"Pelage dataset directory not found: {args.pelage_dir}")
        if not os.path.exists(args.reidentification_dir):
            raise FileNotFoundError(f"Reidentification dataset directory not found: {args.reidentification_dir}")
        if not os.path.exists(args.metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")

        # Create provenance CSV
        provenance_df = create_provenance_csv(
            args.pelage_dir,
            args.reidentification_dir,
            args.metadata_csv,
            args.output_csv
        )

        print(f"\n" + "="*80)
        print("PROVENANCE EXTRACTION COMPLETE")
        print("="*80)
        print(f"Output: {args.output_csv}")
        print(f"Total records: {len(provenance_df):,}")
        print("="*80)

    except Exception as e:
        print(f"\n" + "="*80)
        print("PROVENANCE EXTRACTION FAILED")
        print("="*80)
        print(f"Error: {e}")
        print("\nPlease fix the issues above and try again.")
        print("="*80)
        raise


if __name__ == "__main__":
    main()
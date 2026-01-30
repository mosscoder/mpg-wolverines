#!/usr/bin/env python3
"""
Script 00: Summarize Robust Pelage Images
Analyzes the wolverine detection inventory to identify and count robust pelage images
by individual and year. Prepares data for MegaDetector processing and dataset creation.
"""

import os
import json
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
import argparse


def load_inventory_data(db_path, drive_path):
    """Load and preprocess the wolverine detection inventory"""
    print("Loading detection inventory...")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    # Load Excel file
    db = pd.read_excel(db_path, sheet_name=1)
    print(f"Loaded {len(db)} records from inventory")
    
    # Clean and preprocess
    db['Folder path'] = db['Folder path'].str.replace('D:', drive_path).str.replace('\\', '/')
    db['Marks'] = db['Marks'].str.replace(' marks', '')
    db = db.rename(columns=str.lower)
    db['ymdh'] = db['year'].astype(str) + db['date-time'].str.replace('.', '').str.replace('-', '')
    db.drop(columns=['date-time', 'link'], inplace=True)
    db = db.rename(columns={'type': 'color', 'folder path': 'folder_path'})
    db['color'] = db['color'].apply(lambda x: 1 if x == 'COLOR' else 0)
    
    print(f"Preprocessed data: {len(db)} records")
    return db


def filter_robust_samples(db):
    """Filter for robust pelage samples only"""
    print("\nFiltering for robust pelage samples...")
    
    # Count original marks distribution
    marks_counts = db['marks'].value_counts()
    print("Original marks distribution:")
    for mark, count in marks_counts.items():
        print(f"  {mark}: {count}")
    
    # Filter for robust only
    robust_db = db[db['marks'] == 'Robust'].copy()
    print(f"\nRobust samples: {len(robust_db)} records")
    
    return robust_db


def get_inventory_statistics(db):
    """Compute inventory statistics excluding unreviewed events"""
    # Exclude unsorted (unreviewed) events from statistics
    reviewed_db = db[db['marks'] != 'Unsorted'].copy()

    total_events = len(db)
    reviewed_events = len(reviewed_db)
    unreviewed_events = total_events - reviewed_events

    # Counts by category (reviewed only)
    marks_counts = reviewed_db['marks'].value_counts().to_dict()

    # Calculate percentages
    marks_percentages = {
        mark: round(count / reviewed_events * 100, 1)
        for mark, count in marks_counts.items()
    }

    return {
        'total_events': total_events,
        'reviewed_events': reviewed_events,
        'unreviewed_events': unreviewed_events,
        'counts_by_category': marks_counts,
        'percentages_by_category': marks_percentages
    }


def get_basic_statistics(robust_db):
    """Get basic statistics for console output"""
    print("\nGetting basic statistics...")
    
    unique_individuals = robust_db['id'].nunique()
    unique_folders = robust_db['folder_path'].nunique()
    year_range = f"{robust_db['year'].min()}-{robust_db['year'].max()}"
    
    print(f"Unique individuals: {unique_individuals}")
    print(f"Unique folders: {unique_folders}")
    print(f"Year range: {year_range}")
    
    return {
        'unique_individuals': unique_individuals,
        'unique_folders': unique_folders,
        'year_range': year_range
    }


def scan_directories_simple(robust_db, output_dir):
    """Simple directory scan - just check existence and create processing list"""
    print("\nScanning directories...")
    
    existing_dirs = []
    missing_dirs = []
    
    # Group by folder path to avoid duplicates
    unique_folders = robust_db['folder_path'].unique()
    print(f"Checking {len(unique_folders)} unique directories...")
    
    for i, folder_path in enumerate(unique_folders):
        if i % 100 == 0:
            print(f"  Checked {i}/{len(unique_folders)} directories...")
            
        if os.path.exists(folder_path):
            # Quick check for any image files
            jpg_patterns = ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg']
            has_images = False
            for pattern in jpg_patterns:
                if glob(os.path.join(folder_path, pattern)):
                    has_images = True
                    break
            
            if has_images:
                existing_dirs.append(folder_path)
        else:
            missing_dirs.append(folder_path)
    
    print(f"Scan complete!")
    print(f"  Directories with images: {len(existing_dirs)}")
    print(f"  Missing directories: {len(missing_dirs)}")
    
    # Save missing directories if any
    if missing_dirs:
        missing_file = os.path.join(output_dir, 'missing_directories.txt')
        with open(missing_file, 'w') as f:
            f.write("Missing Directories:\n")
            f.write("==================\n\n")
            for dir_path in missing_dirs:
                f.write(f"{dir_path}\n")
        print(f"Missing directories saved to: missing_directories.txt")
    
    return existing_dirs, missing_dirs


def save_outputs(existing_dirs, basic_stats, inventory_stats, output_dir):
    """Save essential output files"""
    print(f"\nSaving outputs to: {output_dir}")

    # Only essential file: Directories to process with MegaDetector
    megadetector_data = {
        'directories_to_process': existing_dirs,
        'total_directories': len(existing_dirs),
        'purpose': 'MegaDetector batch processing for robust pelage images',
        'generated_by': '00_summarize_robust.py'
    }
    with open(os.path.join(output_dir, 'robust_directories.json'), 'w') as f:
        json.dump(megadetector_data, f, indent=2)
    print("Saved: robust_directories.json")

    # Save inventory statistics
    inventory_data = {
        'counts_by_category': inventory_stats['counts_by_category'],
        'generated_by': '00_summarize_robust.py',
        'note': 'Percentages computed from reviewed events only (excludes Unsorted)',
        'percentages_by_category': inventory_stats['percentages_by_category'],
        'reviewed_events': inventory_stats['reviewed_events'],
        'total_capture_events': inventory_stats['total_events'],
        'unreviewed_events': inventory_stats['unreviewed_events']
    }
    with open(os.path.join(output_dir, 'inventory_summary.json'), 'w') as f:
        json.dump(inventory_data, f, indent=2, sort_keys=True)
    print("Saved: inventory_summary.json")

    print(f"\nSummary:")
    print(f"  Unique individuals: {basic_stats['unique_individuals']}")
    print(f"  Year range: {basic_stats['year_range']}")
    print(f"  Directories to process: {len(existing_dirs)}")


def main():
    parser = argparse.ArgumentParser(description='Summarize robust pelage images for MegaDetector processing')
    parser.add_argument('--output_dir', type=str,
                       default='hugging_face_dataset/v2/data',
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup paths
    db_path = 'data/Detections inventory_2023.08.29.xlsx'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Drive path for folder path replacement
    drive_path = '/Volumes/Seagate Portable Drive'

    print("="*80)
    print("Robust Pelage Image Analysis")
    print("="*80)
    print(f"Database path: {db_path}")
    print(f"Output directory: {args.output_dir}")

    try:
        # Load and filter data
        db = load_inventory_data(db_path, drive_path)
        inventory_stats = get_inventory_statistics(db)
        robust_db = filter_robust_samples(db)
        
        # Get basic statistics
        basic_stats = get_basic_statistics(robust_db)
        
        # Scan directories
        existing_dirs, missing_dirs = scan_directories_simple(robust_db, args.output_dir)
        
        # Save outputs
        save_outputs(existing_dirs, basic_stats, inventory_stats, args.output_dir)
        
        print("\n" + "="*80)
        print("Analysis complete!")
        print("Ready for MegaDetector processing.")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 02: Prepare Pelage Labeling Data
Filters crop metadata to earliest dates per ID/color combination and generates 
DINOv3 embeddings for pelage quality labeling.
"""

# Fix OpenMP conflict on macOS before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import sqlite3
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle

# DINOv3 and transformers imports
from transformers import AutoImageProcessor, AutoModel


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def init_dinov3_model(device):
    """Initialize DINOv3 base model for embedding generation"""
    print(f"Loading DINOv3 model on {device}...")
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m')
    model = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m')
    
    model = model.to(device)
    model.eval()
    
    print(f"DINOv3 model loaded successfully")
    return processor, model


def setup_database(db_path):
    """Set up SQLite database schema"""
    print(f"Setting up database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with proper schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pelage_labels (
            image_path TEXT PRIMARY KEY,
            crop_filename TEXT,
            id TEXT,
            color INTEGER,
            ymdh TEXT,
            embedding BLOB,
            embedding_dim INTEGER,
            label INTEGER,
            labeled_at TIMESTAMP,
            cluster_k INTEGER,
            cluster_id INTEGER
        )
    ''')
    
    # Create indices
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_id_color_ymdh ON pelage_labels(id, color, ymdh)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_label ON pelage_labels(label)')
    
    conn.commit()
    conn.close()
    
    print("Database schema created")


def filter_earliest_dates(metadata_df, crops_dir):
    """Filter to earliest date for each ID × color combination"""
    print("Filtering to earliest dates per ID/color combination...")
    
    # Add full crop path
    metadata_df['crop_path'] = metadata_df['crop_filename'].apply(
        lambda x: os.path.join(crops_dir, x)
    )
    
    # Filter to existing crops only
    existing_mask = metadata_df['crop_path'].apply(os.path.exists)
    metadata_df = metadata_df[existing_mask].copy()
    print(f"Found {len(metadata_df)} crops with existing files")
    
    # Group by ID and color, find earliest date
    earliest = metadata_df.groupby(['id', 'color'])['ymdh'].min().reset_index()
    earliest.columns = ['id', 'color', 'earliest_ymdh']
    
    # Merge back to get all crops from earliest dates
    filtered_df = metadata_df.merge(
        earliest, 
        on=['id', 'color'], 
        how='inner'
    )
    filtered_df = filtered_df[filtered_df['ymdh'] == filtered_df['earliest_ymdh']].copy()
    
    print(f"Filtered to {len(filtered_df)} crops from earliest capture dates")
    print(f"Covering {filtered_df.groupby(['id', 'color']).ngroups} unique ID/color combinations")
    
    return filtered_df


def generate_embeddings_batch(image_paths, processor, model, device, batch_size=16):
    """Generate DINOv3 embeddings for a batch of images"""
    embeddings = []
    
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches", total=num_batches):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load batch of images
        for path in batch_paths:
            try:
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_images.append(img.copy())  # Use copy to ensure data persists
            except Exception as e:
                print(f"Error loading {path}: {e}")
                batch_images.append(None)
        
        # Filter out None images
        valid_images = [img for img in batch_images if img is not None]
        if not valid_images:
            embeddings.extend([None] * len(batch_paths))
            continue
        
        # Process batch
        try:
            inputs = processor(valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Extract CLS token (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0]
                
                # L2 normalize
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                batch_embeddings = cls_embeddings.cpu().numpy()
            
            # Map back to original order
            valid_idx = 0
            for j, img in enumerate(batch_images):
                if img is not None:
                    embeddings.append(batch_embeddings[valid_idx])
                    valid_idx += 1
                else:
                    embeddings.append(None)
                    
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            embeddings.extend([None] * len(batch_paths))
    
    return embeddings


def process_embeddings(filtered_df, crops_dir, db_path, processor, model, device, batch_size=16):
    """Process and store embeddings for filtered crops"""
    print(f"Processing embeddings with batch size {batch_size}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check which crops already have embeddings
    existing_paths = set()
    cursor.execute("SELECT image_path FROM pelage_labels WHERE embedding IS NOT NULL")
    existing_paths = {row[0] for row in cursor.fetchall()}
    
    # Filter to unprocessed crops
    unprocessed_df = filtered_df[~filtered_df['crop_path'].isin(existing_paths)].copy()
    print(f"Found {len(existing_paths)} existing embeddings, processing {len(unprocessed_df)} new crops")
    
    if len(unprocessed_df) == 0:
        print("All crops already processed")
        conn.close()
        return
    
    # Generate embeddings in batches
    image_paths = unprocessed_df['crop_path'].tolist()
    embeddings = generate_embeddings_batch(image_paths, processor, model, device, batch_size)
    
    # Store embeddings in database
    print("Storing embeddings in database...")
    processed_count = 0
    
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Storing embeddings"):
        embedding_idx = unprocessed_df.index.get_loc(idx)
        embedding = embeddings[embedding_idx]
        
        if embedding is not None:
            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)
            
            cursor.execute('''
                INSERT OR REPLACE INTO pelage_labels 
                (image_path, crop_filename, id, color, ymdh, embedding, embedding_dim, label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['crop_path'],
                row['crop_filename'], 
                row['id'],
                row['color'],
                row['ymdh'],
                embedding_blob,
                embedding.shape[0],
                None  # label initially null
            ))
            processed_count += 1
        else:
            print(f"Skipping failed embedding for {row['crop_path']}")
    
    conn.commit()
    conn.close()
    
    print(f"Stored {processed_count} embeddings successfully")


def main():
    parser = argparse.ArgumentParser(description='Prepare pelage labeling data with DINOv3 embeddings')
    parser.add_argument('--metadata_csv', type=str,
                       default='hugging_face_dataset/v2/metadata/crop_metadata.csv',
                       help='Path to crop metadata CSV')
    parser.add_argument('--crops_dir', type=str,
                       default='hugging_face_dataset/v2/init_crops',
                       help='Directory containing crop images')
    parser.add_argument('--output_db', type=str,
                       default='hugging_face_dataset/v2/data/labeling/pelage_labels.db',
                       help='Output SQLite database')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for embedding generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_all_seeds(args.seed)
    
    print("="*80)
    print("Pelage Labeling Data Preparation")
    print("="*80)
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Crops directory: {args.crops_dir}")
    print(f"Output database: {args.output_db}")
    print(f"Batch size: {args.batch_size}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_db), exist_ok=True)
    
    try:
        # Setup database
        setup_database(args.output_db)
        
        # Load and filter metadata
        print(f"\nLoading metadata from: {args.metadata_csv}")
        metadata_df = pd.read_csv(args.metadata_csv)
        print(f"Loaded {len(metadata_df)} crop records")
        
        filtered_df = filter_earliest_dates(metadata_df, args.crops_dir)
        
        # Initialize DINOv3 model
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        processor, model = init_dinov3_model(device)
        
        # Process embeddings
        start_time = time.time()
        process_embeddings(filtered_df, args.crops_dir, args.output_db, processor, model, device, args.batch_size)
        processing_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print(f"PREPARATION COMPLETE")
        print(f"Processing time: {processing_time/60:.1f} minutes")
        print(f"Database ready at: {args.output_db}")
        print("Ready for labeling with Streamlit app")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 02: Prepare Pelage Labeling Data
Filters crop metadata to earliest dates per ID/color combination and generates 
DINOv3 embeddings for pelage quality labeling.
"""

# Fix OpenMP conflict on macOS before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

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
from sklearn.cluster import KMeans


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
    
    # Create precomputed clusters table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS precomputed_clusters (
            image_path TEXT,
            id TEXT,
            color INTEGER,
            ymdh TEXT,
            k_value INTEGER,
            cluster_id INTEGER,
            PRIMARY KEY (image_path, k_value)
        )
    ''')
    
    # Create indices for clusters table
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_clusters_group ON precomputed_clusters(id, color, ymdh, k_value)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_clusters_k ON precomputed_clusters(k_value)')
    
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


def get_unique_groups(db_path):
    """Get all unique ID/color/date groups that have embeddings"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT id, color, ymdh, COUNT(*) as image_count
        FROM pelage_labels 
        WHERE embedding IS NOT NULL
        GROUP BY id, color, ymdh
        ORDER BY id, color, ymdh
    ''')
    
    groups = cursor.fetchall()
    conn.close()
    
    return groups


def get_group_embeddings(db_path, id_val, color_val, ymdh_val):
    """Get embeddings for a specific group"""
    conn = sqlite3.connect(db_path)
    
    query = '''
        SELECT image_path, embedding
        FROM pelage_labels 
        WHERE id = ? AND color = ? AND ymdh = ? AND embedding IS NOT NULL
        ORDER BY image_path
    '''
    
    df = pd.read_sql_query(query, conn, params=[id_val, color_val, ymdh_val])
    conn.close()
    
    if len(df) == 0:
        return [], []
    
    # Extract embeddings
    embeddings = []
    image_paths = []
    
    for _, row in df.iterrows():
        embedding_blob = row['embedding']
        embedding = pickle.loads(embedding_blob)
        embeddings.append(embedding)
        image_paths.append(row['image_path'])
    
    return image_paths, np.array(embeddings)


def compute_group_clusters(image_paths, embeddings, k_values, id_val, color_val, ymdh_val):
    """Compute clusters for a group at multiple K values"""
    group_clusters = {}
    
    # Ensure embeddings are float32 for memory efficiency
    embeddings = embeddings.astype(np.float32)
    
    for k in k_values:
        # Adjust K if we have fewer images than clusters
        effective_k = min(k, len(embeddings))
        
        if effective_k < 2:
            # Can't cluster with less than 2 items
            cluster_labels = [0] * len(image_paths)
        else:
            try:
                # Use safer KMeans parameters
                kmeans = KMeans(
                    n_clusters=effective_k, 
                    random_state=42, 
                    n_init=3,  # Reduced from 10 to prevent seg faults
                    algorithm='lloyd'  # Explicitly use lloyd algorithm
                )
                cluster_labels = kmeans.fit_predict(embeddings)
            except Exception as e:
                print(f"Error clustering group {id_val}/{color_val}/{ymdh_val} at K={k}: {e}")
                # Fallback: assign sequential cluster IDs
                cluster_labels = [i % effective_k for i in range(len(image_paths))]
        
        # Store results for this K
        group_clusters[k] = []
        for i, image_path in enumerate(image_paths):
            group_clusters[k].append({
                'image_path': image_path,
                'id': id_val,
                'color': color_val,
                'ymdh': ymdh_val,
                'k_value': k,
                'cluster_id': int(cluster_labels[i])
            })
    
    return group_clusters


def store_precomputed_clusters(db_path, clusters_data):
    """Store precomputed clusters in database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for cluster_record in clusters_data:
        cursor.execute('''
            INSERT OR REPLACE INTO precomputed_clusters 
            (image_path, id, color, ymdh, k_value, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            cluster_record['image_path'],
            cluster_record['id'],
            cluster_record['color'],
            cluster_record['ymdh'],
            cluster_record['k_value'],
            cluster_record['cluster_id']
        ))
    
    conn.commit()
    conn.close()


def check_existing_clusters(db_path, id_val, color_val, ymdh_val, k_values):
    """Check which K values already have precomputed clusters for this group"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT k_value
        FROM precomputed_clusters 
        WHERE id = ? AND color = ? AND ymdh = ?
    ''', (id_val, color_val, ymdh_val))
    
    existing_k_values = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    return [k for k in k_values if k not in existing_k_values]


def clear_precomputed_clusters(db_path):
    """Clear all precomputed clusters from database"""
    print("Clearing existing precomputed clusters...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM precomputed_clusters")
    conn.commit()
    
    # Get count to confirm
    cursor.execute("SELECT COUNT(*) FROM precomputed_clusters")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Cleared all clusters. Database now has {count} cluster records.")


def precompute_all_clusters(db_path, k_values=[2, 4, 8, 16, 32, 64]):
    """Precompute clusters for all groups at specified K values"""
    print(f"Precomputing clusters for K values: {k_values}")
    
    # Get all unique groups
    groups = get_unique_groups(db_path)
    print(f"Found {len(groups)} unique groups to process")
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    for i, (id_val, color_val, ymdh_val, image_count) in enumerate(tqdm(groups, desc="Processing groups")):
        try:
            print(f"Processing group {i+1}/{len(groups)}: {id_val}/{color_val}/{ymdh_val} ({image_count} images)")
            
            # Check which K values are missing for this group
            missing_k_values = check_existing_clusters(db_path, id_val, color_val, ymdh_val, k_values)
            
            if not missing_k_values:
                total_skipped += 1
                continue
            
            # Get embeddings for this group
            image_paths, embeddings = get_group_embeddings(db_path, id_val, color_val, ymdh_val)
            
            if len(embeddings) == 0:
                print(f"Warning: No embeddings found for group {id_val}/{color_val}/{ymdh_val}")
                continue
            
            print(f"Computing clusters for K values: {missing_k_values}")
            
            # Compute clusters for missing K values only
            group_clusters = compute_group_clusters(
                image_paths, embeddings, missing_k_values, 
                id_val, color_val, ymdh_val
            )
            
            # Store all clusters for this group
            all_cluster_data = []
            for k, cluster_list in group_clusters.items():
                all_cluster_data.extend(cluster_list)
            
            if all_cluster_data:
                store_precomputed_clusters(db_path, all_cluster_data)
                total_processed += 1
                print(f"Successfully stored clusters for group {id_val}/{color_val}/{ymdh_val}")
            
            # Clear memory after each group
            del embeddings, group_clusters, all_cluster_data
            
        except Exception as e:
            print(f"Error processing group {id_val}/{color_val}/{ymdh_val}: {e}")
            total_errors += 1
            # Continue with next group rather than failing entirely
            continue
    
    print(f"Completed clustering: {total_processed} processed, {total_skipped} skipped, {total_errors} errors")
    return total_processed


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
    parser.add_argument('--force_recluster', action='store_true',
                       help='Force recomputation of all clusters (clears existing clusters)')
    
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
        embedding_time = time.time() - start_time
        
        # Precompute clusters
        print(f"\nStarting cluster precomputation...")
        cluster_start_time = time.time()
        
        # Clear existing clusters if force_recluster is set
        if args.force_recluster:
            clear_precomputed_clusters(args.output_db)
        
        processed_groups = precompute_all_clusters(args.output_db)
        cluster_time = time.time() - cluster_start_time
        
        total_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print(f"PREPARATION COMPLETE")
        print(f"Embedding processing time: {embedding_time/60:.1f} minutes")
        print(f"Cluster precomputation time: {cluster_time/60:.1f} minutes")
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Processed {processed_groups} groups for clustering")
        print(f"Database ready at: {args.output_db}")
        print("Ready for labeling with Streamlit app")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 03: Best Epochs Cross-Validation
Find optimal epoch count for individual ID classification using 5-fold CV.
Uses only label==1 (pelage visible) samples from individuals with 32+ such samples.
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to wolverines root
sys.path.append(wolverines_root)

from utils.dataset import load_wolverines_dataset, set_all_seeds, create_dataloaders, WolverinesDataset
from utils.preprocessing import get_height_crop_and_resize_transform
from utils.models import create_model
from utils.individual_id import get_qualified_individuals, create_cv_folds
from utils.training import CVTrainer


def run_cv_fold(fold_idx: int, args: argparse.Namespace) -> dict:
    """Run one cross-validation fold"""
    
    print(f"=== Cross-Validation Fold {fold_idx} ===")
    
    # Set seed for reproducibility
    set_all_seeds(42)
    
    # Load dataset - training set only
    print("Loading dataset...")
    train_dataset, _ = load_wolverines_dataset()
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Get qualified individuals
    individual_ids = get_qualified_individuals()
    print(f"Qualified individuals: {individual_ids}")
    
    # Filter to qualified individuals and label==1 only
    qualified_ids_set = set(individual_ids)
    filtered_dataset = train_dataset.filter(
        lambda x: x['id'] in qualified_ids_set and x['label'] == 1
    )
    print(f"Filtered to {len(filtered_dataset)} label==1 samples from qualified individuals")
    
    # Create CV folds
    folds = create_cv_folds(filtered_dataset, individual_ids, n_folds=5, seed=42)
    
    if fold_idx >= len(folds):
        print(f"Fold {fold_idx} not available")
        return None
    
    train_indices, val_indices = folds[fold_idx]
    
    # Create dataset views
    fold_train_dataset = filtered_dataset.select(train_indices)
    fold_val_dataset = filtered_dataset.select(val_indices)
    
    # Encode individual IDs as class labels
    fold_train_dataset = fold_train_dataset.class_encode_column('id')
    fold_train_dataset = fold_train_dataset.rename_column('id', 'class_label')
    
    fold_val_dataset = fold_val_dataset.class_encode_column('id') 
    fold_val_dataset = fold_val_dataset.rename_column('id', 'class_label')
    
    # Rename original label to pelage for tracking
    fold_train_dataset = fold_train_dataset.rename_column('label', 'pelage')
    fold_val_dataset = fold_val_dataset.rename_column('label', 'pelage')
    fold_train_dataset = fold_train_dataset.rename_column('class_label', 'label')
    fold_val_dataset = fold_val_dataset.rename_column('class_label', 'label')
    
    num_classes = len(individual_ids)
    print(f"CV Fold {fold_idx}: {len(fold_train_dataset)} train, {len(fold_val_dataset)} val ({num_classes} classes)")
    
    # Create transforms and data loaders
    batch_size = 16
    transform = get_height_crop_and_resize_transform(height=1280, resize=728)
    
    train_loader, val_loader = create_dataloaders(
        fold_train_dataset, fold_val_dataset,
        transform, transform, batch_size
    )
    
    # Create model
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device, num_classes=num_classes)
    
    print(f"Using device: {device}")
    print(f"Model classes: {num_classes}")
    
    # Create trainer using shared CVTrainer
    trainer = CVTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    # Train
    start_time = time.time()
    epochs = 50
    
    print(f"Training fold {fold_idx} for {epochs} epochs...")
    epoch_results = trainer.train_cv_fold(
        train_loader=train_loader,
        val_loader=val_loader, 
        epochs=epochs,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Find best epoch by validation F1
    best_epoch_idx = np.argmax([r['val_f1'] for r in epoch_results])
    best_epoch = epoch_results[best_epoch_idx]
    
    print(f"Best epoch: {best_epoch['epoch']} (F1: {best_epoch['val_f1']:.4f})")
    print(f"Training time: {training_time/60:.1f} minutes")
    
    # Prepare results
    results = {
        'fold_idx': fold_idx,
        'individual_ids': individual_ids,
        'num_classes': num_classes,
        'dataset_stats': {
            'train_samples': len(fold_train_dataset),
            'val_samples': len(fold_val_dataset),
            'total_filtered': len(filtered_dataset)
        },
        'training_params': {
            'epochs': epochs,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': batch_size
        },
        'best_epoch': {
            'epoch_num': best_epoch['epoch'],
            'val_f1': best_epoch['val_f1'],
            'val_accuracy': best_epoch['val_accuracy'],
            'val_loss': best_epoch['val_loss']
        },
        'epoch_history': epoch_results,
        'training_time': training_time,
        'timestamp': time.time()
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Cross-validation for best epochs')
    parser.add_argument('--fold', type=int, required=True,
                       help='Fold index (0-4)')
    parser.add_argument('--output_dir', type=str,
                       default='results/best_epoch_cv',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Individual ID Best Epochs CV - Fold {args.fold}")
    print("=" * 80)
    
    # Run CV fold
    try:
        results = run_cv_fold(args.fold, args)
        if results is None:
            return
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"fold_{args.fold}.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved results to: {output_path}")
        print(f"  Best epoch: {results['best_epoch']['epoch_num']}")
        print(f"  Best F1: {results['best_epoch']['val_f1']:.4f}")
        
    except Exception as e:
        print(f"Error in fold {args.fold}: {e}")
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 00: Find Best Epoch
Finds optimal number of training epochs using 5-fold cross-validation.
Fixed learning rate at 0.001, records performance at each epoch.
Uses fixed 224x224 resize and batch size 32.
"""

import sys
import os
import argparse
import time
import json
import torch
from pathlib import Path

# Add utils to path (assumes we cd to pelage_sorting in sbatch)
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to wolverines root
sys.path.append(wolverines_root)

from utils.dataset import (
    load_wolverines_dataset, create_kfold_splits, 
    create_dataloaders, set_all_seeds
)
from utils.preprocessing import get_standard_transform
from utils.models import create_model
from utils.training import (
    ModelTrainer, save_results, check_result_exists
)


def get_job_combinations(job_idx: int, max_jobs: int = 5) -> list:
    """Map job index to fold number - one job per fold"""
    
    # Simple: 5 folds, 5 jobs
    folds = [0, 1, 2, 3, 4]  # 5-fold CV
    
    # Handle case where job_idx exceeds available jobs
    if job_idx >= len(folds):
        return []
    
    # One fold per job
    return [folds[job_idx]]




def get_fixed_params():
    """Get fixed parameters for learning rate sweep"""
    
    params = {
        # Fixed parameters
        'resize_size': 224,  # Fixed 224x224 for DINOv3 efficiency
        'batch_size': 32,    # Updated batch size
        'epochs': 50,        # Longer training for LR sweep
        'weight_decay': 0.01
    }
    
    return params


def train_single_config(fold: int, args: argparse.Namespace) -> dict:
    """Train one fold configuration with fixed LR=0.001 and return results"""
    
    # Set seed (consistent for this fold)
    seed = fold  # Use fold as seed for consistency
    set_all_seeds(seed)
    
    # Get fixed parameters with locked learning rate
    params = get_fixed_params()
    lr = 0.001  # Fixed learning rate
    
    # Create output filename
    filename = f"fold={fold}.json"
    output_path = os.path.join(args.output_dir, filename)
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print(f"Training: fold={fold}, lr={lr:.3f} (fixed)")
    
    # Load dataset
    print("Loading dataset...")
    dataset, _ = load_wolverines_dataset()
    
    # Dataset will be processed lazily in transforms
    
    # Create k-fold splits (using preprocessed dataset)
    print("Creating 5-fold splits...")
    fold_datasets = create_kfold_splits(dataset, n_folds=5, seed=0)
    
    if fold >= len(fold_datasets):
        print(f"Error: Fold {fold} not available")
        return None
    
    train_dataset, val_dataset = fold_datasets[fold]
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create transforms with resize and normalization
    transform = get_standard_transform(resize_size=params['resize_size'])
    
    # Create dataloaders using shared utility
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        transform, transform, params['batch_size']
    )
    
    # Create model
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=lr,
        weight_decay=params['weight_decay']
    )
    
    # Train with detailed epoch tracking
    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=params['epochs'],
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Prepare detailed results with epoch-by-epoch metrics
    final_results = {
        'job_idx': args.idx,
        'learning_rate': lr,
        'fold': fold,
        'seed': seed,
        'params': params,
        'training_time': training_time,
        'epochs_trained': results['epochs_trained'],
        'best_val_f1': results['best_val_f1'],
        'final_val_f1': results['final_val_metrics']['f1_score'],
        'final_val_precision': results['final_val_metrics']['precision'],
        'final_val_recall': results['final_val_metrics']['recall'],
        'final_val_accuracy': results['final_val_metrics']['accuracy'],
        # Include epoch-by-epoch history
        'train_history': trainer.train_history,
        'val_history': trainer.val_history
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Saved results to: {filename}")
    print(f"  Best validation F1: {final_results['best_val_f1']:.4f}")
    print(f"  Final validation F1: {final_results['final_val_f1']:.4f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Find best epoch with 5-fold CV (fixed LR=0.001)')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (0-4, one per fold)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='results/00_best_epoch',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Best Epoch Search - Job {args.idx}")
    print("=" * 60)
    
    # Get fold for this job
    folds = get_job_combinations(args.idx)
    
    if not folds:
        print(f"No fold assigned for job {args.idx}")
        return
    
    fold = folds[0]  # One fold per job
    print(f"Processing fold {fold} with fixed LR=0.001")
    
    # Train this fold
    print(f"\n--- Training Fold {fold} ---")
    try:
        result = train_single_config(fold, args)
    except Exception as e:
        print(f"Error training Fold={fold}: {e}")
        return
    
    print(f"\n" + "=" * 60)
    print(f"Job {args.idx} completed!")
    
    if result:
        print(f"Fold {fold} - Best validation F1: {result['best_val_f1']:.4f}")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("No results generated")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 02: Learning Rate Sweep
Sweeps over different learning rates using 5-fold cross-validation.
Records performance at each epoch.
Uses best settings from scripts 00 and 01.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

# Add utils to path (assumes we cd to pelage_sorting in sbatch)
sys.path.append('.')

from utils.dataset import (
    load_wolverines_dataset, create_kfold_splits, 
    create_dataloaders, set_all_seeds
)
from utils.transforms import create_transform_from_params
from utils.models import create_model
from utils.training import (
    ModelTrainer, save_results, check_result_exists
)
from utils.preemption import CheckpointManager, ProgressTracker


def get_job_combinations(job_idx: int, max_jobs: int = 40) -> list:
    """Map job index to list of (lr, fold) tuples - supports up to 40 jobs"""
    
    # Parameters from CLAUDE.md
    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    folds = [0, 1, 2, 3, 4]  # 5-fold CV
    
    # Generate all combinations: 7 LRs × 5 folds = 35 total
    all_combinations = []
    for lr in learning_rates:
        for fold in folds:
            all_combinations.append((lr, fold))
    
    total_combinations = len(all_combinations)  # 35 total
    
    # Handle case where job_idx exceeds available work
    if job_idx >= total_combinations:
        return []
    
    # Simple 1:1 mapping for learning rate sweep (35 combinations, use jobs 0-34)
    # Jobs 35-39 will have no work
    return [all_combinations[job_idx]]


def get_best_params_from_previous_experiments():
    """
    Load best parameters from previous experiments.
    Analyzes actual results from scripts 00 and 01.
    """
    from utils.training import get_best_crop_size_from_results, get_best_augmentation_params
    
    # Get best crop size from script 00
    best_crop_size = get_best_crop_size_from_results()
    print(f"Best crop_size from script 00: {best_crop_size}")
    
    # Get best augmentation params from script 01
    best_aug_params = get_best_augmentation_params()
    print(f"Best augmentation params from script 01:")
    for key, value in best_aug_params.items():
        print(f"  {key}: {value}")
    
    best_params = {
        # From script 00
        'crop_size': best_crop_size,
        'resize_size': 256,
        
        # From script 01
        'max_zoom': best_aug_params.get('max_zoom', 1.25),
        'h_flip_p': best_aug_params.get('h_flip_p', 0.5),
        'grayscale_p': best_aug_params.get('grayscale_p', 0.25),
        'blur_type': best_aug_params.get('blur_type', 'moderate'),
        'blur_p': best_aug_params.get('blur_p', 0.25),
        'cutmix_p': best_aug_params.get('cutmix_p', 0.25),
        
        # Fixed for this experiment
        'batch_size': 32,
        'epochs': 50,  # Longer training for LR sweep
        'weight_decay': 0.01
    }
    
    return best_params


def train_single_config(lr: float, fold: int, args: argparse.Namespace) -> dict:
    """Train one LR/fold configuration and return results"""
    
    # Set seed (consistent across all LR experiments for this fold)
    seed = fold  # Use fold as seed for consistency
    set_all_seeds(seed)
    
    # Get best parameters from previous experiments
    best_params = get_best_params_from_previous_experiments()
    best_params['learning_rate'] = lr
    
    # Create output filename
    filename = f"lr={lr:.6f}_fold={fold}.json"
    output_path = os.path.join(args.output_dir, filename)
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print(f"Training: lr={lr}, fold={fold}")
    
    # Load dataset
    print("Loading dataset...")
    dataset, _ = load_wolverines_dataset()
    
    # Create k-fold splits (full training set)
    print("Creating 5-fold splits...")
    fold_data = create_kfold_splits(dataset, n_folds=5, seed=0)
    
    if fold >= len(fold_data):
        print(f"Error: Fold {fold} not available")
        return None
    
    train_images, train_labels, val_images, val_labels = fold_data[fold]
    print(f"Train samples: {len(train_images)}, Val samples: {len(val_images)}")
    
    # Create transforms
    train_transform = create_transform_from_params(best_params, is_train=True)
    val_transform = create_transform_from_params(best_params, is_train=False)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_images, train_labels, val_images, val_labels,
        train_transform, val_transform, best_params['batch_size']
    )
    
    # Create model
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=lr,
        weight_decay=best_params['weight_decay']
    )
    
    # Train with detailed epoch tracking
    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=best_params['epochs'],
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Prepare detailed results with epoch-by-epoch metrics
    final_results = {
        'job_idx': args.idx,
        'learning_rate': lr,
        'fold': fold,
        'seed': seed,
        'best_params': best_params,
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
    parser = argparse.ArgumentParser(description='Sweep over learning rates with 5-fold CV')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (0-39)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='results/02_learning_rate',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Learning Rate Sweep - Job {args.idx}")
    print("=" * 60)
    
    # Get combinations for this job
    combinations = get_job_combinations(args.idx)
    
    if not combinations:
        print(f"No combinations for job {args.idx}")
        return
    
    print(f"Processing {len(combinations)} configurations:")
    for lr, fold in combinations:
        print(f"  LR: {lr}, Fold: {fold}")
    
    # Train each combination
    results_summary = []
    for i, (lr, fold) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(lr, fold, args)
            if result:
                results_summary.append(result)
        except Exception as e:
            print(f"Error training LR={lr}, Fold={fold}: {e}")
            continue
    
    print(f"\n" + "=" * 60)
    print(f"Job {args.idx} completed!")
    print(f"Successfully processed {len(results_summary)} configurations")
    
    if results_summary:
        avg_f1 = sum(r['best_val_f1'] for r in results_summary) / len(results_summary)
        best_f1 = max(r['best_val_f1'] for r in results_summary)
        print(f"Average best validation F1: {avg_f1:.4f}")
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 02: Train Production Model
Trains final model on combined train+test sets using optimal hyperparameters.
Saves only the linear classifier head (frozen backbone not saved).
Uses optimal learning rate and epochs from script 00.
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
    load_wolverines_dataset, create_dataloaders, set_all_seeds
)
from utils.preprocessing import get_standard_transform
from utils.models import create_model
from utils.training import (
    ModelTrainer, check_result_exists
)


def get_optimal_params_from_lr_sweep():
    """
    Load optimal parameters from script 00 learning rate sweep using proper cross-validation.
    """
    import json
    import glob
    import numpy as np
    from collections import defaultdict
    
    # Get best learning rate and optimal epochs from script 00 (proper cross-validation)
    best_lr = 0.001  # Default fallback
    optimal_epochs = 30  # Default fallback
    results_pattern = "results/00_learning_rate/*.json"
    result_files = glob.glob(results_pattern)
    
    if result_files:
        # Group results by learning rate
        lr_groups = defaultdict(list)
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                lr = result.get('learning_rate')
                if lr is not None and 'val_history' in result:
                    lr_groups[lr].append(result)
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Find best learning rate using proper cross-validation
        best_cv_f1 = 0
        best_cv_epochs = 30
        
        for lr, lr_results in lr_groups.items():
            # Collect validation histories for this learning rate
            all_val_histories = []
            for result in lr_results:
                if 'val_history' in result and result['val_history']:
                    val_f1s = [epoch.get('f1_score', 0) for epoch in result['val_history']]
                    all_val_histories.append(val_f1s)
            
            if not all_val_histories:
                continue
            
            # Find max epochs across all folds
            max_epochs = max(len(history) for history in all_val_histories)
            
            # Compute mean F1 at each epoch across folds
            epoch_mean_f1s = []
            for epoch_idx in range(max_epochs):
                fold_f1s = []
                for history in all_val_histories:
                    if epoch_idx < len(history):
                        fold_f1s.append(history[epoch_idx])
                    else:
                        fold_f1s.append(history[-1])  # Use last value if shorter
                epoch_mean_f1s.append(np.mean(fold_f1s))
            
            # Find the epoch with best mean F1 across folds
            lr_best_f1 = max(epoch_mean_f1s)
            lr_best_epoch = epoch_mean_f1s.index(lr_best_f1) + 1  # Convert to 1-indexed
            
            # Track overall best across learning rates
            if lr_best_f1 > best_cv_f1:
                best_cv_f1 = lr_best_f1
                best_lr = lr
                optimal_epochs = lr_best_epoch
        
        print(f"Best learning rate from cross-validation: {best_lr} (F1: {best_cv_f1:.4f})")
        print(f"Optimal epochs from cross-validation: {optimal_epochs}")
    else:
        print("No learning rate results found, using defaults")
    
    optimal_params = {
        # Fixed parameters
        'resize_size': 224,
        'batch_size': 32,
        'weight_decay': 0.01,
        'seed': 0,
        
        # From script 00 (cross-validated)
        'learning_rate': best_lr,
        'optimal_epochs': optimal_epochs,
    }
    
    print("=" * 60)
    print("OPTIMAL PARAMETERS FOR PRODUCTION MODEL:")
    print(f"Resize size (fixed): 224")
    print(f"Batch size (fixed): 32") 
    print(f"Learning rate (from script 00, CV): {best_lr}")
    print(f"Optimal epochs (from CV): {optimal_epochs}")
    print("=" * 60)
    
    return optimal_params


def combine_train_test_datasets(train_dataset, test_dataset):
    """
    Combine train and test datasets for production training.
    
    Returns:
        combined_dataset: Combined dataset for training
        dataset_stats: Statistics about the combined dataset
    """
    from datasets import concatenate_datasets
    
    # Combine datasets
    combined_dataset = concatenate_datasets([train_dataset, test_dataset])
    
    # Calculate statistics
    train_labels = [item['label'] for item in train_dataset]
    test_labels = [item['label'] for item in test_dataset]
    combined_labels = [item['label'] for item in combined_dataset]
    
    dataset_stats = {
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'combined_samples': len(combined_dataset),
        'train_label_dist': [train_labels.count(0), train_labels.count(1)],
        'test_label_dist': [test_labels.count(0), test_labels.count(1)],
        'combined_label_dist': [combined_labels.count(0), combined_labels.count(1)]
    }
    
    return combined_dataset, dataset_stats


def train_production_model(args: argparse.Namespace) -> dict:
    """Train the production model with optimal parameters on combined data"""
    
    # Get optimal parameters from script 00
    optimal_params = get_optimal_params_from_lr_sweep()
    
    # Set seed
    set_all_seeds(optimal_params['seed'])
    
    # Create output filename
    filename = "production_model_results.json"
    output_path = os.path.join(args.output_dir, filename)
    model_path = os.path.join(args.output_dir, "production_model_classifier.pth")
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print("Training production model with optimal parameters:")
    for key, value in optimal_params.items():
        print(f"  {key}: {value}")
    
    # Load train and test datasets
    print("Loading train and test datasets...")
    train_dataset, test_dataset = load_wolverines_dataset()
    
    # Combine datasets for production training
    print("Combining train and test datasets...")
    combined_dataset, dataset_stats = combine_train_test_datasets(train_dataset, test_dataset)
    
    print(f"Combined dataset: {dataset_stats['combined_samples']} samples")
    print(f"Label distribution: {dataset_stats['combined_label_dist']}")
    
    # Create transforms with resize and normalization
    transform = get_standard_transform(resize_size=optimal_params['resize_size'])
    
    # Create dataloader for combined dataset (we'll use it as both train and val for final metrics)
    # Note: For production, we train on all data without validation split
    train_loader, val_loader = create_dataloaders(
        combined_dataset, combined_dataset,  # Same dataset for train and val
        transform, transform, optimal_params['batch_size']
    )
    
    # Create model
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=optimal_params['learning_rate'],
        weight_decay=optimal_params['weight_decay']
    )
    
    # Train production model
    start_time = time.time()
    print(f"\nStarting production training for {optimal_params['optimal_epochs']} epochs...")
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,  # Using same data for final metrics
        epochs=optimal_params['optimal_epochs'],
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Save only the classifier head (linear layer) using trainer method
    trainer.save_classifier_head(model_path)
    
    # Prepare comprehensive results
    final_results = {
        'job_idx': args.idx,
        'model_type': 'production',
        'optimal_params': optimal_params,
        'training_time': training_time,
        'dataset_stats': dataset_stats,
        'model_stats': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'classifier_saved_path': model_path
        },
        'final_performance': {
            'epochs_trained': results['epochs_trained'],
            'final_f1': results['final_val_metrics']['f1_score'],
            'final_precision': results['final_val_metrics']['precision'],
            'final_recall': results['final_val_metrics']['recall'],
            'final_accuracy': results['final_val_metrics']['accuracy']
        },
        # Include complete epoch-by-epoch history
        'train_history': trainer.train_history,
        'val_history': trainer.val_history,  # Same as train for production
        'timestamp': time.time()
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Production model training completed!")
    print(f"✓ Results saved to: {filename}")
    print(f"✓ Classifier model saved to: {model_path}")
    print(f"\n=== PRODUCTION MODEL PERFORMANCE ===")
    print(f"Final F1 score: {final_results['final_performance']['final_f1']:.4f}")
    print(f"Final precision: {final_results['final_performance']['final_precision']:.4f}")
    print(f"Final recall: {final_results['final_performance']['final_recall']:.4f}")
    print(f"Final accuracy: {final_results['final_performance']['final_accuracy']:.4f}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Trained on: {dataset_stats['combined_samples']} samples (train+test combined)")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Train production model on combined train+test data')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (should be 0)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='results/02_production',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 60)
    
    if args.idx != 0:
        print(f"Warning: This script should typically run with --idx 0, got {args.idx}")
    
    try:
        result = train_production_model(args)
        if result:
            print(f"\n🎯 Production model training completed successfully!")
            print(f"📁 Model and results saved to: {args.output_dir}")
        else:
            print("Skipped: Results already exist (use --overwrite to rerun)")
    
    except Exception as e:
        print(f"Error during production model training: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 03: Test Performance
Final training with optimal hyperparameters on full train/test split.
Records performance at each epoch.
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
    load_wolverines_dataset, create_dataloaders, set_all_seeds
)
from utils.preprocessing import get_best_resize_size, get_standard_transform, preprocess_dataset
from utils.models import create_model
from utils.training import (
    ModelTrainer, check_result_exists
)


def get_optimal_params_from_experiments():
    """
    Load optimal parameters from all previous experiments.
    """
    import json
    import glob
    
    # Get best resize size from script 00
    best_resize = get_best_resize_size()
    
    # Get best learning rate from script 01
    best_lr = 0.001  # Default fallback
    optimal_epochs = 30  # Default fallback
    results_pattern = "results/01_learning_rate/*.json"
    result_files = glob.glob(results_pattern)
    
    if result_files:
        best_f1 = 0
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                f1_score = result.get('best_val_f1', 0)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_lr = result['learning_rate']
                    # Find optimal epoch (where best F1 was achieved)
                    if 'val_history' in result:
                        val_f1s = [epoch.get('f1_score', 0) for epoch in result['val_history']]
                        optimal_epochs = val_f1s.index(max(val_f1s)) + 1 if val_f1s else 30
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"Best learning rate from script 01: {best_lr} (F1: {best_f1:.4f})")
        print(f"Optimal epochs: {optimal_epochs}")
    
    optimal_params = {
        # From script 00
        'resize_size': best_resize,
        
        # From script 01
        'learning_rate': best_lr,
        'optimal_epochs': optimal_epochs,
        
        # Fixed params
        'batch_size': 16,
        'weight_decay': 0.01,
        'seed': 0
    }
    
    print("=" * 60)
    print("OPTIMAL PARAMETERS FROM PREVIOUS EXPERIMENTS:")
    print(f"Resize size (from 00): {best_resize}")
    print(f"Learning rate (from 01): {best_lr}")
    print(f"Optimal epochs: {optimal_epochs}")
    print("=" * 60)
    
    return optimal_params


def convert_dataset_to_lists(dataset):
    """Convert HuggingFace dataset to lists of images and labels"""
    images = []
    labels = []
    
    for item in dataset:
        images.append(item['image'])
        labels.append(item['label'])
    
    return images, labels


def train_final_model(args: argparse.Namespace) -> dict:
    """Train the final model with optimal parameters"""
    
    # Get optimal parameters
    optimal_params = get_optimal_params_from_experiments()
    
    # Set seed
    set_all_seeds(optimal_params['seed'])
    
    # Create output filename
    filename = "final_test_performance.json"
    output_path = os.path.join(args.output_dir, filename)
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print("Training final model with optimal parameters:")
    for key, value in optimal_params.items():
        print(f"  {key}: {value}")
    
    # Load full datasets
    print("Loading full datasets...")
    train_dataset, test_dataset = load_wolverines_dataset()
    
    # Preprocess datasets (resize images)
    print(f"Preprocessing images to {optimal_params['resize_size']}x{optimal_params['resize_size']}...")
    train_dataset = preprocess_dataset(train_dataset, optimal_params['resize_size'])
    test_dataset = preprocess_dataset(test_dataset, optimal_params['resize_size'])
    
    # Convert to lists
    train_images, train_labels = convert_dataset_to_lists(train_dataset)
    test_images, test_labels = convert_dataset_to_lists(test_dataset)
    
    print(f"Train samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Train label distribution: {[train_labels.count(0), train_labels.count(1)]}")
    print(f"Test label distribution: {[test_labels.count(0), test_labels.count(1)]}")
    
    # Create transforms (simple: just normalize)
    transform = get_standard_transform()
    train_transform = transform
    test_transform = transform
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_images, train_labels, test_images, test_labels,
        train_transform, test_transform, optimal_params['batch_size']
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
    
    # Train with detailed tracking
    start_time = time.time()
    print(f"\nStarting training for {optimal_params['optimal_epochs']} epochs...")
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Use test set as validation for final evaluation
        epochs=optimal_params['optimal_epochs'],
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Prepare comprehensive results
    final_results = {
        'job_idx': args.idx,
        'optimal_params': optimal_params,
        'training_time': training_time,
        'dataset_stats': {
            'train_samples': len(train_images),
            'test_samples': len(test_images),
            'train_label_dist': [train_labels.count(0), train_labels.count(1)],
            'test_label_dist': [test_labels.count(0), test_labels.count(1)]
        },
        'model_stats': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'final_performance': {
            'epochs_trained': results['epochs_trained'],
            'best_test_f1': results['best_val_f1'],  # This is actually test F1
            'final_test_f1': results['final_val_metrics']['f1_score'],
            'final_test_precision': results['final_val_metrics']['precision'],
            'final_test_recall': results['final_val_metrics']['recall'],
            'final_test_accuracy': results['final_val_metrics']['accuracy']
        },
        # Include complete epoch-by-epoch history
        'train_history': trainer.train_history,
        'test_history': trainer.val_history,  # This is actually test history
        'timestamp': time.time()
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Final model training completed!")
    print(f"✓ Results saved to: {filename}")
    print(f"\n=== FINAL PERFORMANCE ===")
    print(f"Best test F1 score: {final_results['final_performance']['best_test_f1']:.4f}")
    print(f"Final test F1 score: {final_results['final_performance']['final_test_f1']:.4f}")
    print(f"Final test precision: {final_results['final_performance']['final_test_precision']:.4f}")
    print(f"Final test recall: {final_results['final_performance']['final_test_recall']:.4f}")
    print(f"Final test accuracy: {final_results['final_performance']['final_test_accuracy']:.4f}")
    print(f"Training time: {training_time/60:.1f} minutes")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Final test performance evaluation')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (should be 0)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='results/02_test',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINAL TEST PERFORMANCE EVALUATION")
    print("=" * 60)
    
    if args.idx != 0:
        print(f"Warning: This script should typically run with --idx 0, got {args.idx}")
    
    try:
        result = train_final_model(args)
        if result:
            print(f"\n🎯 Final test evaluation completed successfully!")
        else:
            print("Skipped: Results already exist (use --overwrite to rerun)")
    
    except Exception as e:
        print(f"Error during final test evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
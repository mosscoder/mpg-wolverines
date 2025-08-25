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
from utils.transforms import create_transform_from_params
from utils.models import create_model
from utils.training import (
    ModelTrainer, check_result_exists
)


def get_optimal_params_from_experiments():
    """
    Load optimal parameters from all previous experiments.
    """
    from utils.training import (
        get_best_crop_size_from_results, 
        get_best_augmentation_params,
        get_best_learning_rate_from_results
    )
    
    # Get best crop size from script 00
    best_crop_size = get_best_crop_size_from_results()
    
    # Get best augmentation params from script 01
    best_aug_params = get_best_augmentation_params()
    
    # Get best learning rate and optimal epochs from script 02
    lr_results = get_best_learning_rate_from_results()
    best_lr = lr_results.get('learning_rate', 0.001)
    optimal_epochs = lr_results.get('optimal_epochs', 30)
    
    optimal_params = {
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
        
        # From script 02
        'learning_rate': best_lr,
        'optimal_epochs': optimal_epochs,
        
        # Fixed params
        'batch_size': 32,
        'weight_decay': 0.01,
        'seed': 0
    }
    
    print("=" * 60)
    print("OPTIMAL PARAMETERS FROM PREVIOUS EXPERIMENTS:")
    print(f"Crop size (from 00): {best_crop_size}")
    print(f"Learning rate (from 02): {best_lr}")
    print(f"Optimal epochs: {optimal_epochs}")
    print("Augmentations (from 01):")
    for key, value in best_aug_params.items():
        print(f"  {key}: {value}")
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
    
    # Convert to lists
    train_images, train_labels = convert_dataset_to_lists(train_dataset)
    test_images, test_labels = convert_dataset_to_lists(test_dataset)
    
    print(f"Train samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Train label distribution: {[train_labels.count(0), train_labels.count(1)]}")
    print(f"Test label distribution: {[test_labels.count(0), test_labels.count(1)]}")
    
    # Create transforms
    train_transform = create_transform_from_params(optimal_params, is_train=True)
    test_transform = create_transform_from_params(optimal_params, is_train=False)
    
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
                       default='results/03_test',
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
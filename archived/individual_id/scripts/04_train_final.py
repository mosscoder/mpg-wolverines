#!/usr/bin/env python3
"""
Script 04: Train Final Model
Train individual ID model for optimal epochs and evaluate on test set.
Uses best epoch count from cross-validation results.
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import load_wolverines_dataset, set_all_seeds, create_dataloaders, WolverinesDataset
from utils.preprocessing import get_height_crop_and_resize_transform
from utils.models import create_model
from utils.individual_id import get_qualified_individuals, load_best_cv_epochs, create_individual_datasets, save_model_checkpoint
from utils.training import FinalTrainer


def main():
    parser = argparse.ArgumentParser(description='Train final individual ID model')
    parser.add_argument('--cv_results_dir', type=str, 
                       default='results/best_epoch_cv',
                       help='Directory containing CV results')
    parser.add_argument('--models_dir', type=str,
                       default='models', 
                       help='Directory to save trained model')
    parser.add_argument('--output_file', type=str,
                       default='results/test_performance.json',
                       help='Output file for test results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Individual ID Final Model Training")
    print("=" * 80)
    
    # Set seed
    set_all_seeds(args.seed)
    
    # Load best epochs from CV
    print("Loading cross-validation results...")
    optimal_epochs, mean_cv_f1, fold_results = load_best_cv_epochs(args.cv_results_dir)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, test_dataset = load_wolverines_dataset()
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Get qualified individuals
    individual_ids = get_qualified_individuals()
    print(f"Qualified individuals: {individual_ids}")
    
    # Create train/test datasets using shared utility
    train_encoded, test_encoded, train_counts, test_counts = create_individual_datasets(
        train_dataset, test_dataset, individual_ids, label_only=True
    )
    
    num_classes = len(individual_ids)
    print(f"Classes: {num_classes}")
    
    # Create transforms and data loaders
    batch_size = 16
    transform = get_height_crop_and_resize_transform(height=1280, resize=728)
    
    # For final training, we use the full training set as both train and "val"
    train_loader, _ = create_dataloaders(
        train_encoded, train_encoded,  # Use same dataset for both
        transform, transform, batch_size
    )
    
    # Create test loader
    test_pytorch_dataset = WolverinesDataset(test_encoded, transform)
    test_loader = torch.utils.data.DataLoader(
        test_pytorch_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Create model
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device, num_classes=num_classes)
    
    print(f"Using device: {device}")
    print(f"Model classes: {num_classes}")
    
    # Create trainer using shared FinalTrainer
    trainer = FinalTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    # Train final model
    start_time = time.time()
    print(f"Training final model for {optimal_epochs} epochs...")
    
    final_train_acc = trainer.train_final(
        train_loader=train_loader,
        epochs=optimal_epochs,
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save model using shared utility
    model_path = os.path.join(args.models_dir, "individual_id_final.pth")
    
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimal_epochs': optimal_epochs,
        'batch_size': batch_size
    }
    
    save_model_checkpoint(model, individual_ids, model_path, config)
    
    # Evaluate on test set using shared trainer method
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_loader, individual_ids)
    
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1 score: {test_results['f1_score']:.4f}")
    
    # Prepare comprehensive results
    final_results = {
        'individual_ids': individual_ids,
        'num_classes': num_classes,
        'training_params': {
            'optimal_epochs': optimal_epochs,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': batch_size,
            'seed': args.seed
        },
        'cross_validation': {
            'mean_cv_f1': mean_cv_f1,
            'fold_results_summary': [
                {
                    'fold': fold['fold_idx'],
                    'best_epoch': fold['best_epoch']['epoch_num'],
                    'best_f1': fold['best_epoch']['val_f1']
                }
                for fold in fold_results
            ]
        },
        'dataset_stats': {
            'train_total': len(train_encoded),
            'test_total': len(test_encoded),
            'train_per_individual': train_counts,
            'test_per_individual': test_counts
        },
        'training_results': {
            'final_train_accuracy': final_train_acc,
            'training_time': training_time,
            'train_history': trainer.train_history
        },
        'test_results': {
            'accuracy': test_results['accuracy'],
            'f1_score': test_results['f1_score'],
            'classification_report': test_results['classification_report'],
            'confusion_matrix': test_results['confusion_matrix']
        },
        'model_path': model_path,
        'timestamp': time.time()
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Saved results to: {args.output_file}")
    print(f"Model saved to: {model_path}")
    print(f"Final test F1: {test_results['f1_score']:.4f}")

if __name__ == "__main__":
    main()
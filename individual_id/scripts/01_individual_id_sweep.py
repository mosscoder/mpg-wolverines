#!/usr/bin/env python3
"""
Script 01: Individual ID Sweep
Multi-class classification experiment with varying samples per individual.
Tests pelage-only vs random sampling approaches with different sample sizes.
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict, Counter
import random

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to wolverines root
sys.path.append(wolverines_root)

from utils.dataset import load_wolverines_dataset, set_all_seeds, create_dataloaders, WolverinesDataset
from utils.preprocessing import get_standard_transform
from utils.models import create_model
from utils.training import check_result_exists

def get_job_combinations(job_idx: int, max_jobs: int = 24) -> list:
    """Map job index to list of (sample_size, approach, seed) tuples"""
    
    # Experimental parameters - focused on 3 individuals with max 32 samples
    sample_sizes = [2, 4, 8, 16, 32]
    approaches = ['pelage_only', 'random_sample']
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Generate all combinations: 5 sizes × 2 approaches × 8 seeds = 80 total
    all_combinations = []
    for sample_size in sample_sizes:
        for approach in approaches:
            for seed in seeds:
                all_combinations.append((sample_size, approach, seed))
    
    total_combinations = len(all_combinations)  # 80 total
    
    # Handle case where job_idx exceeds available jobs
    if job_idx >= max_jobs:
        return []
    
    # Distribute 80 combinations across 24 jobs
    # Jobs 0-7: 4 configs each (32 configs)
    # Jobs 8-23: 3 configs each (48 configs)
    if job_idx < 8:
        start_idx = job_idx * 4
        end_idx = start_idx + 4
    else:
        start_idx = 32 + (job_idx - 8) * 3
        end_idx = start_idx + 3
    
    if end_idx <= total_combinations:
        return all_combinations[start_idx:end_idx]
    else:
        return all_combinations[start_idx:total_combinations]

def get_feasible_individuals(dataset, min_sample_size=64):
    """Get the 3 selected individuals with sufficient samples for experiments"""
    
    # Use only the 3 selected individuals: BDF10-M6, HLC20-H3, and Turk
    selected_individuals = ['BDF10-M6', 'HLC20-H3', 'Turk']
    
    # Convert dataset to analyze individual counts
    individual_counts = defaultdict(lambda: {'total': 0, 'label_0': 0, 'label_1': 0})
    
    for item in dataset:
        ind_id = item['id']
        individual_counts[ind_id]['total'] += 1
        if item['label'] == 0:
            individual_counts[ind_id]['label_0'] += 1
        else:
            individual_counts[ind_id]['label_1'] += 1
    
    # Verify the selected individuals have sufficient samples
    feasible_individuals = []
    for ind_id in selected_individuals:
        if ind_id in individual_counts:
            counts = individual_counts[ind_id]
            # Need at least 64 total samples (32 train + 32 val) and sufficient pelage for experiments
            if counts['total'] >= 64 and counts['label_1'] >= 32:
                feasible_individuals.append(ind_id)
                print(f"{ind_id}: {counts['total']} total, {counts['label_1']} pelage - ✓ feasible")
            else:
                print(f"{ind_id}: {counts['total']} total, {counts['label_1']} pelage - ✗ insufficient")
        else:
            print(f"{ind_id}: not found in dataset")
    
    print(f"Selected {len(feasible_individuals)} individuals: {feasible_individuals}")
    
    return feasible_individuals, individual_counts

def create_individual_dataset(dataset, individual_ids, sample_size, approach, seed):
    """Create train/val dataset views for individual ID classification with 32+32 split"""
    
    set_all_seeds(seed)
    
    # Group indices by individual
    individual_indices = defaultdict(list)
    for i, item in enumerate(dataset):
        if item['id'] in individual_ids:
            individual_indices[item['id']].append(i)
    
    # Create train and val index lists
    train_indices = []
    val_indices = []
    individual_to_class = {ind_id: i for i, ind_id in enumerate(sorted(individual_ids))}
    
    for ind_id in individual_ids:
        all_indices = individual_indices[ind_id]
        
        if approach == 'pelage_only':
            # Use only samples with label=1 (pelage visible)
            pelage_indices = [i for i in all_indices if dataset[i]['label'] == 1]
            available_indices = pelage_indices
        else:  # random_sample
            # Use any samples regardless of label
            available_indices = all_indices
        
        # Need 32 for training + 32 for validation = 64 total per individual
        total_needed = 64
        if len(available_indices) < total_needed:
            print(f"Warning: {ind_id} has only {len(available_indices)} available samples, need {total_needed}")
            # Use all available indices
            shuffled_indices = random.sample(available_indices, len(available_indices))
        else:
            # Sample 64 total indices
            shuffled_indices = random.sample(available_indices, total_needed)
        
        # Split: first 32 for training, next 32 for validation
        train_indices_ind = shuffled_indices[:32]
        val_indices_ind = shuffled_indices[32:64] if len(shuffled_indices) >= 64 else shuffled_indices[len(train_indices_ind):]
        
        # Add to train/val index lists
        train_indices.extend(train_indices_ind)
        val_indices.extend(val_indices_ind)
        
        print(f"{ind_id}: {len(train_indices_ind)} train + {len(val_indices_ind)} val samples")
    
    print(f"Total: {len(train_indices)} train + {len(val_indices)} val samples")
    print(f"Classes: {len(individual_ids)} individuals")
    
    # Create lightweight dataset views - no .map() calls to avoid memory duplication
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    return train_dataset, val_dataset, individual_to_class

class IndividualDataset:
    """Memory-efficient dataset for individual identification that computes labels on-demand"""
    
    def __init__(self, hf_dataset, individual_to_class, transform=None):
        self.hf_dataset = hf_dataset
        self.individual_to_class = individual_to_class
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']  # PIL Image from HuggingFace dataset
        
        # Compute individual class from ID on-demand
        ind_id = item['id']
        individual_class = self.individual_to_class[ind_id]
        
        if self.transform:
            image = self.transform(image)
        
        return image, individual_class

class MultiClassModelTrainer:
    """Trainer for multi-class individual identification"""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=0.01):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_images)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, num_classes):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # Calculate per-class metrics
        report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
        
        return avg_loss, accuracy, all_predictions, all_labels, report
    
    def train(self, train_loader, val_loader, num_classes, epochs=30, verbose=True):
        """Train the model"""
        
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation  
            val_loss, val_acc, val_preds, val_labels, val_report = self.validate_epoch(val_loader, num_classes)
            
            # Save history
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'accuracy': train_acc
            })
            
            self.val_history.append({
                'epoch': epoch + 1,
                'loss': val_loss,
                'accuracy': val_acc,
                'classification_report': val_report
            })
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Final evaluation
        final_val_loss, final_val_acc, final_preds, final_labels, final_report = self.validate_epoch(val_loader, num_classes)
        
        results = {
            'epochs_trained': epochs,
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'final_val_accuracy': final_val_acc,
            'final_val_loss': final_val_loss,
            'final_predictions': final_preds,
            'final_labels': final_labels,
            'final_classification_report': final_report,
            'confusion_matrix': confusion_matrix(final_labels, final_preds).tolist()
        }
        
        return results

def train_single_config(sample_size: int, approach: str, seed: int, args: argparse.Namespace) -> dict:
    """Train one configuration and return results"""
    
    # Set seed
    set_all_seeds(seed)
    
    # Create output filename
    filename = f"samples={sample_size}_approach={approach}_seed={seed}.json"
    output_path = os.path.join(args.output_dir, filename)
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print(f"Training: samples={sample_size}, approach={approach}, seed={seed}")
    
    # Load dataset - TRAINING SET ONLY for individual ID experiments
    print("Loading dataset...")
    train_dataset, _ = load_wolverines_dataset()
    print(f"Training dataset: {len(train_dataset)} samples (training set only)")
    
    # Get feasible individuals
    feasible_individuals, individual_counts = get_feasible_individuals(train_dataset, sample_size)
    
    if len(feasible_individuals) < 3:
        print(f"Error: Only {len(feasible_individuals)} feasible individuals, need at least 3")
        return None
    
    # Create individual dataset with fixed 32+32 train/val split
    ind_train_dataset, ind_val_dataset, individual_to_class = create_individual_dataset(
        train_dataset, feasible_individuals, sample_size, approach, seed
    )
    
    num_classes = len(feasible_individuals)
    print(f"Multi-class problem: {num_classes} individuals (classes)")
    
    print(f"Train samples: {len(ind_train_dataset)}")
    print(f"Val samples: {len(ind_val_dataset)}")
    
    # Create transforms with resize for individual ID
    resize_size = 728  # Use fixed size for individual ID
    transform = get_standard_transform(resize_size=resize_size)
    batch_size = 16
    
    # Create memory-efficient datasets that compute individual classes on-demand
    train_pytorch_dataset = IndividualDataset(ind_train_dataset, individual_to_class, transform)
    val_pytorch_dataset = IndividualDataset(ind_val_dataset, individual_to_class, transform)
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_pytorch_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_pytorch_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Create model (modify for multi-class)
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device, num_classes=num_classes)
    
    print(f"Using device: {device}")
    print(f"Model classes: {num_classes}")
    
    # Create trainer
    trainer = MultiClassModelTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    # Train
    start_time = time.time()
    epochs = 30  # Reasonable for multi-class with limited samples
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Prepare comprehensive results
    final_results = {
        'job_idx': args.idx,
        'sample_size': sample_size,
        'approach': approach,
        'seed': seed,
        'num_classes': num_classes,
        'num_individuals': len(feasible_individuals),
        'individual_ids': feasible_individuals,
        'individual_to_class': individual_to_class,
        'individual_counts': {ind_id: individual_counts[ind_id] for ind_id in feasible_individuals},
        'dataset_stats': {
            'total_samples_used': len(ind_train_dataset) + len(ind_val_dataset),
            'train_samples': len(ind_train_dataset),
            'val_samples': len(ind_val_dataset),
            'samples_per_individual': sample_size,
            'actual_samples_per_class': (len(ind_train_dataset) + len(ind_val_dataset)) // len(feasible_individuals)
        },
        'training_time': training_time,
        'performance': results,
        'train_history': trainer.train_history,
        'val_history': trainer.val_history,
        'experimental_params': {
            'resize_size': resize_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': 0.001,
            'weight_decay': 0.01
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Saved results to: {filename}")
    print(f"  Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"  Final validation accuracy: {results['final_val_accuracy']:.4f}")
    print(f"  Training time: {training_time/60:.1f} minutes")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Individual ID classification sweep')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Individual ID Classification - Job {args.idx}")
    print("=" * 80)
    
    # Get combinations for this job
    combinations = get_job_combinations(args.idx)
    
    if not combinations:
        print(f"No combinations for job {args.idx}")
        return
    
    print(f"Processing {len(combinations)} configurations:")
    for sample_size, approach, seed in combinations:
        print(f"  Samples: {sample_size}, Approach: {approach}, Seed: {seed}")
    
    # Train each combination
    results_summary = []
    for i, (sample_size, approach, seed) in enumerate(combinations):
        print(f"\\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(sample_size, approach, seed, args)
            if result:
                results_summary.append(result)
        except Exception as e:
            print(f"Error training samples={sample_size}, approach={approach}, seed={seed}: {e}")
            continue
    
    print(f"\\n" + "=" * 80)
    print(f"Job {args.idx} completed!")
    print(f"Successfully processed {len(results_summary)} configurations")
    
    if results_summary:
        avg_acc = sum(r['performance']['best_val_accuracy'] for r in results_summary) / len(results_summary)
        best_acc = max(r['performance']['best_val_accuracy'] for r in results_summary)
        print(f"Average best validation accuracy: {avg_acc:.4f}")
        print(f"Best validation accuracy: {best_acc:.4f}")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
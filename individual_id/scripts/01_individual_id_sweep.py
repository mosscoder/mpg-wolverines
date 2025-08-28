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
from utils.preprocessing import get_height_crop_and_resize_transform
from utils.models import create_model
from utils.training import check_result_exists

def get_job_combinations(job_idx: int, max_jobs: int = 24) -> list:
    """Map job index to list of (sample_size, approach, seed) tuples"""
    
    # Experimental parameters - pelage vs pelage_abs experiments
    sample_sizes = [2, 4, 8, 16, 32]
    approaches = ['pelage', 'pelage_abs']
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
    # Jobs 0-7: 4 configs each (32 total)
    # Jobs 8-23: 3 configs each (48 total)
    if job_idx < 8:
        configs_per_job = 4
        start_idx = job_idx * 4
        end_idx = start_idx + 4
    else:
        configs_per_job = 3
        start_idx = 32 + (job_idx - 8) * 3
        end_idx = start_idx + 3
    
    if end_idx <= total_combinations:
        return all_combinations[start_idx:end_idx]
    else:
        return all_combinations[start_idx:total_combinations]


def load_feasible_individuals(results_dir='results'):
    """Load pre-computed feasible individuals from script 00"""
    import json
    
    config_path = os.path.join(results_dir, 'feasible_individuals.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Feasible individuals config not found at {config_path}. Run script 00 first.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    feasible_ids = config['feasible_individuals_pelage_64']
    print(f"Loaded feasible individuals: {', '.join(feasible_ids)}")
    
    return config

def create_individual_dataset(dataset, individual_ids, sample_size, approach, seed):
    """Create train/val dataset views for individual ID classification"""
    
    set_all_seeds(seed)
    
    # Group indices by individual and label
    individual_indices = defaultdict(lambda: {'visible': [], 'invisible': []})
    for i, item in enumerate(dataset):
        if item['id'] in individual_ids:
            if item['label'] == 1:
                individual_indices[item['id']]['visible'].append(i)
            else:
                individual_indices[item['id']]['invisible'].append(i)
    
    # Create train and val index lists
    train_indices = []
    val_indices = []
    individual_to_class = {ind_id: i for i, ind_id in enumerate(sorted(individual_ids))}
    
    for ind_id in individual_ids:
        visible_indices = individual_indices[ind_id]['visible']
        invisible_indices = individual_indices[ind_id]['invisible']
        
        if approach == 'pelage':
            # Training: use sample_size visible samples
            # Validation: use 32 visible samples
            train_needed = sample_size
            val_needed = 32
            total_needed = train_needed + val_needed
            
            if len(visible_indices) < total_needed:
                print(f"Error: {ind_id} has only {len(visible_indices)} visible, need {total_needed}")
                continue
            
            # Sample from visible indices only
            random.shuffle(visible_indices)
            val_indices_ind = visible_indices[:val_needed]  # First 32 for validation
            train_indices_ind = visible_indices[val_needed:val_needed + train_needed]  # Next sample_size for training
            
            print(f"{ind_id}: {len(train_indices_ind)} train (pelage only) + {len(val_indices_ind)} val (pelage only)")
            
        elif approach == 'pelage_abs':
            # Training: use sample_size invisible samples  
            # Validation: use 32 invisible samples
            train_needed = sample_size
            val_needed = 32
            total_needed = train_needed + val_needed
            
            if len(invisible_indices) < total_needed:
                print(f"Error: {ind_id} has only {len(invisible_indices)} invisible, need {total_needed}")
                continue
            
            # Sample from invisible indices only
            random.shuffle(invisible_indices)
            val_indices_ind = invisible_indices[:val_needed]  # First 32 for validation
            train_indices_ind = invisible_indices[val_needed:val_needed + train_needed]  # Next sample_size for training
            
            print(f"{ind_id}: {len(train_indices_ind)} train (invisible only) + {len(val_indices_ind)} val (invisible only)")
        
        # Add to master lists
        train_indices.extend(train_indices_ind)
        val_indices.extend(val_indices_ind)
    
    print(f"Total: {len(train_indices)} train + {len(val_indices)} val samples")
    print(f"Classes: {len(individual_ids)} individuals")
    
    # Create lightweight dataset views
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    # Preserve pelage visibility by renaming 'label' to 'pelage'
    train_dataset = train_dataset.rename_column('label', 'pelage')
    val_dataset = val_dataset.rename_column('label', 'pelage')
    
    # Use HuggingFace native tools: class_encode_column + rename
    # This automatically maps the unique IDs to integers
    train_dataset = train_dataset.class_encode_column('id')
    train_dataset = train_dataset.rename_column('id', 'label')
    
    val_dataset = val_dataset.class_encode_column('id')
    val_dataset = val_dataset.rename_column('id', 'label')
    
    # Extract the label mapping for reference
    # The class_encode_column creates a ClassLabel feature with the mapping
    label_names = train_dataset.features['label'].names
    label2id = {name: i for i, name in enumerate(label_names)}
    
    return train_dataset, val_dataset, label2id


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
        
        for batch_data in train_loader:
            if len(batch_data) == 3:  # Has pelage info
                batch_images, batch_labels, _ = batch_data  # Ignore pelage in training
            else:
                batch_images, batch_labels = batch_data
                
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
        """Validate for one epoch with separate pelage metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_pelage = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:  # Has pelage info
                    batch_images, batch_labels, batch_pelage = batch_data
                    all_pelage.extend(batch_pelage.cpu().numpy())
                else:
                    batch_images, batch_labels = batch_data
                    # Fill with -1 if no pelage info available
                    all_pelage.extend([-1] * len(batch_labels))
                
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Convert to numpy arrays for easier indexing
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_pelage = np.array(all_pelage)
        
        # Overall accuracy
        overall_accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # Overall classification report
        overall_report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
        
        # Separate metrics by pelage visibility if pelage info available
        pelage_metrics = {}
        if len(all_pelage) > 0 and not all(p == -1 for p in all_pelage):
            visible_mask = all_pelage == 1
            invisible_mask = all_pelage == 0
            
            if np.sum(visible_mask) > 0:
                visible_acc = sum(all_predictions[visible_mask] == all_labels[visible_mask]) / np.sum(visible_mask)
                visible_report = classification_report(all_labels[visible_mask], all_predictions[visible_mask], 
                                                     output_dict=True, zero_division=0)
                pelage_metrics['visible'] = {
                    'accuracy': visible_acc,
                    'count': int(np.sum(visible_mask)),
                    'classification_report': visible_report
                }
            
            if np.sum(invisible_mask) > 0:
                invisible_acc = sum(all_predictions[invisible_mask] == all_labels[invisible_mask]) / np.sum(invisible_mask)
                invisible_report = classification_report(all_labels[invisible_mask], all_predictions[invisible_mask],
                                                       output_dict=True, zero_division=0)
                pelage_metrics['invisible'] = {
                    'accuracy': invisible_acc,
                    'count': int(np.sum(invisible_mask)),
                    'classification_report': invisible_report
                }
        
        return avg_loss, overall_accuracy, all_predictions, all_labels, overall_report, pelage_metrics
    
    def train(self, train_loader, val_loader, num_classes, epochs=30, verbose=True):
        """Train the model"""
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation  
            val_loss, val_acc, val_preds, val_labels, val_report, pelage_metrics = self.validate_epoch(val_loader, num_classes)
            
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
                'classification_report': val_report,
                'pelage_metrics': pelage_metrics
            })
            
            if verbose:
                # Enhanced display with pelage metrics
                msg = f"Epoch {epoch+1:2d}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
                if pelage_metrics:
                    if 'visible' in pelage_metrics:
                        msg += f", Vis={pelage_metrics['visible']['accuracy']:.4f}"
                    if 'invisible' in pelage_metrics:
                        msg += f", Inv={pelage_metrics['invisible']['accuracy']:.4f}"
                print(msg)
        
        # Final evaluation
        final_val_loss, final_val_acc, final_preds, final_labels, final_report, final_pelage_metrics = self.validate_epoch(val_loader, num_classes)
        
        results = {
            'epochs_trained': epochs,
            'final_val_accuracy': final_val_acc,
            'final_val_loss': final_val_loss,
            # Remove large numpy arrays that cause JSON serialization issues
            # 'final_predictions': final_preds,  # Too large for JSON
            # 'final_labels': final_labels,      # Too large for JSON
            'final_classification_report': final_report,
            'final_pelage_metrics': final_pelage_metrics,
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

    # Load pre-computed feasible individuals
    try:
        feasible_config = load_feasible_individuals(args.output_dir)
        feasible_individuals = feasible_config['feasible_individuals_pelage_64']
        individual_counts = {
            ind_id: {'label_1': feasible_config['individual_pelage_counts'][ind_id],
                    'total': feasible_config['individual_total_counts'][ind_id]}
            for ind_id in feasible_individuals
        }
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 00_count_individuals.py first to generate feasible individuals.")
        return None
    
    # Filter dataset to only feasible individuals - MASSIVE memory reduction!
    feasible_ids_set = set(feasible_individuals)
    print(f"Filtering dataset to {len(feasible_individuals)} individuals...")
    
    # Filter to feasible individuals only (need both visible and invisible samples)
    filtered_dataset = train_dataset.filter(
        lambda x: x['id'] in feasible_ids_set
    )
    
    print(f"Filtered dataset: {len(filtered_dataset)} samples (down from {len(train_dataset)})")
    
    if len(feasible_individuals) < 3:
        print(f"Error: Only {len(feasible_individuals)} feasible individuals, need at least 3")
        return None
    
    # Create individual dataset with fixed 32+32 train/val split from filtered data
    ind_train_dataset, ind_val_dataset, individual_to_class = create_individual_dataset(
        filtered_dataset, feasible_individuals, sample_size, approach, seed
    )
    
    num_classes = len(feasible_individuals)
    print(f"Multi-class problem: {num_classes} individuals (classes)")
    
    print(f"Train samples: {len(ind_train_dataset)}")
    print(f"Val samples: {len(ind_val_dataset)}")
    
    # Create transforms - single pipeline for all approaches
    batch_size = 16
    transform = get_height_crop_and_resize_transform(height=1280, resize=728)
    resize_size = "728x728"
    
    # Use standard WolverinesDataset - same as pelage_sorting!
    train_loader, val_loader = create_dataloaders(
        ind_train_dataset, ind_val_dataset,
        transform, transform, batch_size
    )
    
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
    epochs = 50  # Reasonable for multi-class with limited samples
    
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
            'approach': approach,
            'approach_details': f'Approach: {approach} (1280px height crop → 728x728 square resize)',
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
    print(f"  Final validation accuracy: {results['final_val_accuracy']:.4f}")
    
    # Show pelage-specific performance if available
    if final_pelage_metrics:
        if 'visible' in final_pelage_metrics:
            vis_acc = final_pelage_metrics['visible']['accuracy']
            vis_count = final_pelage_metrics['visible']['count']
            print(f"  Final visible accuracy: {vis_acc:.4f} (n={vis_count})")
        if 'invisible' in final_pelage_metrics:
            inv_acc = final_pelage_metrics['invisible']['accuracy']
            inv_count = final_pelage_metrics['invisible']['count']
            print(f"  Final invisible accuracy: {inv_acc:.4f} (n={inv_count})")
    
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
        avg_acc = sum(r['performance']['final_val_accuracy'] for r in results_summary) / len(results_summary)
        best_acc = max(r['performance']['final_val_accuracy'] for r in results_summary)
        print(f"Average final validation accuracy: {avg_acc:.4f}")
        print(f"Best final validation accuracy: {best_acc:.4f}")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
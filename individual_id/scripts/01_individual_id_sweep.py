#!/usr/bin/env python3
"""
Script 01: Individual ID Sweep (Crops-Masks Config)
Multi-class classification experiment with varying samples per individual.
Uses crops-masks dataset configuration with detection filtering and temporal splits.
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import set_all_seeds, create_dataloaders
from utils.models import create_model
from utils.training import check_result_exists, MultiClassTrainer
from datasets import load_dataset
import torchvision.transforms as T
from PIL import Image


def get_job_combinations(job_idx: int, max_jobs: int = 24) -> list:
    """Map job index to list of (sample_size, approach, seed) tuples"""
    
    # Experimental parameters - detection-filtered experiments
    sample_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # 16 sizes
    approaches = ['pelage', 'pelage_abs']
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Generate all combinations: 16 sizes × 2 approaches × 8 seeds = 256 total
    all_combinations = []
    for sample_size in sample_sizes:
        for approach in approaches:
            for seed in seeds:
                all_combinations.append((sample_size, approach, seed))
    
    total_combinations = len(all_combinations)  # 256 total
    
    # Handle case where job_idx exceeds available jobs
    if job_idx >= max_jobs:
        return []
    
    # Distribute 256 combinations across 24 jobs
    # Jobs 0-15: 11 configs each (176 total)
    # Jobs 16-23: 10 configs each (80 total)
    if job_idx < 16:
        configs_per_job = 11
        start_idx = job_idx * 11
        end_idx = start_idx + 11
    else:
        configs_per_job = 10
        start_idx = 176 + (job_idx - 16) * 10
        end_idx = start_idx + 10
    
    if end_idx <= total_combinations:
        return all_combinations[start_idx:end_idx]
    else:
        return all_combinations[start_idx:total_combinations]


def load_crops_masks_dataset():
    """Load the wolverines dataset with crops-masks configuration"""
    print("Loading wolverines dataset (crops-masks config)...")
    dataset = load_dataset("kdoherty/wolverines", "crops-masks", split="train")
    print(f"Loaded {len(dataset)} samples from crops-masks config")
    return dataset


def create_temporal_split_dataset(dataset, individuals, sample_size, approach, seed):
    """Create date-based temporal train/val splits for individuals"""
    import random
    from collections import defaultdict
    
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Creating temporal split dataset: {sample_size} samples per class, approach={approach}")
    
    # Group samples by individual ID
    individual_samples = defaultdict(list)
    
    for i, sample in enumerate(dataset):
        individual_id = sample['id']
        if individual_id in individuals:
            # Apply filtering based on approach (ALL require megadetector_status == 1)
            if approach == 'pelage' and sample['pelage'] == 1:
                # For pelage==1: only include samples with detections
                if sample['megadetector_status'] == 1:
                    individual_samples[individual_id].append((i, sample))
            elif approach == 'pelage_abs' and sample['pelage'] == 0:
                # For pelage==0: also require detections for consistency
                if sample['megadetector_status'] == 1:
                    individual_samples[individual_id].append((i, sample))
    
    print(f"Samples found per individual (after filtering):")
    for ind_id in individuals:
        count = len(individual_samples[ind_id])
        print(f"  {ind_id}: {count} samples")
    
    # Check if we have enough samples
    train_dataset_indices = []
    val_dataset_indices = []
    individual_to_class = {}
    temporal_info = {}
    
    for class_idx, individual_id in enumerate(individuals):
        samples = individual_samples[individual_id]
        
        if len(samples) < sample_size * 2:
            raise ValueError(
                f"INSUFFICIENT SAMPLES: Individual {individual_id} has only {len(samples)} samples, "
                f"but need {sample_size * 2} (2x {sample_size} for train/val split)"
            )
        
        # Random sample 2x the required amount FIRST
        total_needed = sample_size * 2
        if len(samples) > total_needed:
            selected_samples = random.sample(samples, total_needed)
        else:
            selected_samples = samples
        
        # THEN sort by date (ymdh) 
        selected_samples.sort(key=lambda x: x[1]['ymdh'])
        
        # Split temporally: first half for train, second half for val
        train_samples = selected_samples[:sample_size]
        val_samples = selected_samples[sample_size:]
            
        train_dataset_indices.extend([idx for idx, _ in train_samples])
        val_dataset_indices.extend([idx for idx, _ in val_samples])
        
        individual_to_class[individual_id] = class_idx
        
        # Store temporal split info
        train_dates = [s[1]['ymdh'] for s in train_samples]
        val_dates = [s[1]['ymdh'] for s in val_samples]
        
        temporal_info[individual_id] = {
            'train_dates': train_dates,
            'val_dates': val_dates,
            'train_date_range': [min(train_dates), max(train_dates)] if train_dates else [],
            'val_date_range': [min(val_dates), max(val_dates)] if val_dates else [],
            'total_samples_available': len(samples),
            'samples_used': len(train_samples) + len(val_samples)
        }
        
        print(f"  {individual_id}: Train={len(train_samples)} (dates {min(train_dates)}-{max(train_dates)}), "
              f"Val={len(val_samples)} (dates {min(val_dates)}-{max(val_dates)})")
    
    # Create subset datasets
    train_subset = dataset.select(train_dataset_indices)
    val_subset = dataset.select(val_dataset_indices)
    
    return train_subset, val_subset, individual_to_class, temporal_info


def create_dinov3_transform():
    """Create DINOv3-specific transform pipeline for pre-cropped images"""
    print("DINOv3 transform: resize to 224x224 + normalize (using pre-cropped images)")
    
    return T.Compose([
        T.Resize(size=(224, 224), interpolation=Image.LANCZOS),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats for DINOv3
    ])


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
    
    # Load crops-masks dataset
    dataset = load_crops_masks_dataset()

    # Load sorted individuals and use top 3
    config_path = 'individual_id/results/feasible_individuals.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Index into sorted list for top 3 individuals
        n_individuals = 3
        individuals_sorted = config['individuals_sorted_by_pelage']
        feasible_individuals = individuals_sorted[:n_individuals]
        print(f"Using top {n_individuals} individuals by pelage count: {', '.join(feasible_individuals)}")
        
        # Show pelage counts for reference
        for ind_id in feasible_individuals:
            pelage_count = config['individual_pelage_counts'][ind_id]
            print(f"  {ind_id}: {pelage_count} pelage samples")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 00_count_individuals.py first to generate individual data.")
        return None
    
    # Create temporal dataset splits
    try:
        ind_train_dataset, ind_val_dataset, individual_to_class, temporal_info = create_temporal_split_dataset(
            dataset, feasible_individuals, sample_size, approach, seed
        )
    except ValueError as e:
        print(f"FATAL ERROR: {e}")
        raise e
    
    num_classes = len(feasible_individuals)
    print(f"Multi-class problem: {num_classes} individuals (classes)")
    
    print(f"Train samples: {len(ind_train_dataset)}")
    print(f"Val samples: {len(ind_val_dataset)}")
    
    # Create transforms for pre-cropped images
    batch_size = 16
    transform = create_dinov3_transform()
    crop_size = "pre-cropped → 224x224"
    
    # Create custom dataloaders for crops-masks dataset
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    
    class CropsMasksDataset(TorchDataset):
        def __init__(self, dataset, transform, individual_to_class):
            self.dataset = dataset
            self.transform = transform
            self.individual_to_class = individual_to_class
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            sample = self.dataset[idx]
            
            # Use cropped image
            if sample['megadetector_image'] is not None:
                image = sample['megadetector_image']
            else:
                # Should not happen since we filtered for detections
                raise ValueError(f"No cropped image for sample {idx}")
            
            # Apply transforms
            image = self.transform(image)
            
            # Get class label and pelage info
            individual_id = sample['id']
            class_label = self.individual_to_class[individual_id]
            pelage = sample['pelage']
            
            return image, class_label, pelage
    
    train_dataset_torch = CropsMasksDataset(ind_train_dataset, transform, individual_to_class)
    val_dataset_torch = CropsMasksDataset(ind_val_dataset, transform, individual_to_class)
    
    train_loader = DataLoader(train_dataset_torch, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model (modify for multi-class)
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device, num_classes=num_classes)
    
    print(f"Using device: {device}")
    print(f"Model classes: {num_classes}")
    
    # Create trainer using shared MultiClassTrainer
    trainer = MultiClassTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    # Train
    start_time = time.time()
    epochs = 50  # Reasonable for multi-class with limited samples
    
    # Enhanced training with pelage-specific metrics
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validation with detailed metrics including F1
        val_loss, val_acc, val_preds, val_labels = trainer.validate_epoch(val_loader)
        
        # Calculate F1 score (macro only)
        from sklearn.metrics import f1_score
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # Calculate pelage-specific metrics if available
        pelage_metrics = {}
        all_pelage = []
        
        # Extract pelage labels from validation set
        for batch_data in val_loader:
            if len(batch_data) == 3:  # Has pelage info
                _, _, batch_pelage = batch_data
                all_pelage.extend(batch_pelage.cpu().numpy())
        
        # Calculate metrics by pelage visibility
        if all_pelage and len(all_pelage) == len(val_preds):
            all_pelage = np.array(all_pelage)
            visible_mask = all_pelage == 1
            invisible_mask = all_pelage == 0
            
            if np.sum(visible_mask) > 0:
                val_preds_arr = np.array(val_preds)
                val_labels_arr = np.array(val_labels)
                visible_acc = sum(val_preds_arr[visible_mask] == val_labels_arr[visible_mask]) / np.sum(visible_mask)
                visible_f1 = f1_score(val_labels_arr[visible_mask], val_preds_arr[visible_mask], 
                                     average='weighted', zero_division=0)
                pelage_metrics['visible'] = {
                    'accuracy': visible_acc,
                    'f1_score': visible_f1,
                    'count': int(np.sum(visible_mask))
                }
            
            if np.sum(invisible_mask) > 0:
                val_preds_arr = np.array(val_preds)
                val_labels_arr = np.array(val_labels)
                invisible_acc = sum(val_preds_arr[invisible_mask] == val_labels_arr[invisible_mask]) / np.sum(invisible_mask)
                invisible_f1 = f1_score(val_labels_arr[invisible_mask], val_preds_arr[invisible_mask], 
                                       average='weighted', zero_division=0)
                pelage_metrics['invisible'] = {
                    'accuracy': invisible_acc,
                    'f1_score': invisible_f1,
                    'count': int(np.sum(invisible_mask))
                }
        
        # Save history
        trainer.train_history.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_acc
        })
        
        trainer.val_history.append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'accuracy': val_acc,
            'f1_macro': val_f1_macro,
            'pelage_metrics': pelage_metrics
        })
        
        # Print progress
        if True:  # verbose
            msg = f"Epoch {epoch+1:2d}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1_macro:.4f}"
            if pelage_metrics:
                if 'visible' in pelage_metrics:
                    msg += f", Vis F1={pelage_metrics['visible']['f1_score']:.4f}"
                if 'invisible' in pelage_metrics:
                    msg += f", Inv F1={pelage_metrics['invisible']['f1_score']:.4f}"
            print(msg)
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_val_loss, final_val_acc, final_preds, final_labels = trainer.validate_epoch(val_loader)
    
    # Calculate final F1 score (macro only)
    from sklearn.metrics import f1_score
    final_f1_macro = f1_score(final_labels, final_preds, average='macro', zero_division=0)
    
    # Get final pelage metrics
    final_pelage_metrics = trainer.val_history[-1]['pelage_metrics'] if trainer.val_history else {}
    
    # Calculate classification report
    from sklearn.metrics import classification_report
    final_report = classification_report(final_labels, final_preds, output_dict=True, zero_division=0)
    
    results = {
        'epochs_trained': epochs,
        'final_val_accuracy': final_val_acc,
        'final_val_f1_macro': final_f1_macro,
        'final_val_loss': final_val_loss,
        'final_classification_report': final_report,
        'final_pelage_metrics': final_pelage_metrics,
        'confusion_matrix': confusion_matrix(final_labels, final_preds).tolist()
    }
    
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
        'individual_counts': {ind_id: config['individual_pelage_counts'][ind_id] for ind_id in feasible_individuals},
        'temporal_split_info': temporal_info,
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
            'crop_size': crop_size,
            'approach': approach,
            'approach_details': f'Temporal split approach: {approach} (uses crops-masks config with detection filtering)',
            'dataset_config': 'crops-masks',
            'detection_filtering': True,
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
    print(f"  Final validation F1 (macro): {results['final_val_f1_macro']:.4f}")
    
    # Show pelage-specific performance if available
    if final_pelage_metrics:
        if 'visible' in final_pelage_metrics:
            vis_acc = final_pelage_metrics['visible']['accuracy']
            vis_f1 = final_pelage_metrics['visible']['f1_score']
            vis_count = final_pelage_metrics['visible']['count']
            print(f"  Final visible accuracy: {vis_acc:.4f}, F1: {vis_f1:.4f} (n={vis_count})")
        if 'invisible' in final_pelage_metrics:
            inv_acc = final_pelage_metrics['invisible']['accuracy']
            inv_f1 = final_pelage_metrics['invisible']['f1_score']
            inv_count = final_pelage_metrics['invisible']['count']
            print(f"  Final invisible accuracy: {inv_acc:.4f}, F1: {inv_f1:.4f} (n={inv_count})")
    
    print(f"  Training time: {training_time/60:.1f} minutes")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Individual ID classification sweep')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='individual_id/results',
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
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(sample_size, approach, seed, args)
            if result:
                results_summary.append(result)
        except Exception as e:
            print(f"Error training samples={sample_size}, approach={approach}, seed={seed}: {e}")
            continue
    
    print(f"\n" + "=" * 80)
    print(f"Job {args.idx} completed!")
    print(f"Successfully processed {len(results_summary)} configurations")
    
    if results_summary:
        avg_acc = sum(r['performance']['final_val_accuracy'] for r in results_summary) / len(results_summary)
        best_acc = max(r['performance']['final_val_accuracy'] for r in results_summary)
        avg_f1 = sum(r['performance']['final_val_f1_macro'] for r in results_summary) / len(results_summary)
        best_f1 = max(r['performance']['final_val_f1_macro'] for r in results_summary)
        print(f"Average final validation accuracy: {avg_acc:.4f}")
        print(f"Best final validation accuracy: {best_acc:.4f}")
        print(f"Average final validation F1 (macro): {avg_f1:.4f}")
        print(f"Best final validation F1 (macro): {best_f1:.4f}")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
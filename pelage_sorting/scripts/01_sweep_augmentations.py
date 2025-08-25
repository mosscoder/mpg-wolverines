#!/usr/bin/env python3
"""
Script 01: Augmentations Sweep
Sweeps over different data augmentation parameters.
Each job handles a subset of the parameter combinations across seeds.
"""

import sys
import os
import argparse
import time

# Add utils to path (assumes we cd to pelage_sorting in sbatch)
sys.path.append('.')

from utils.dataset import (
    load_wolverines_dataset, create_stratified_train_val_split,
    create_dataloaders, set_all_seeds
)
from utils.transforms import create_transform_from_params
from utils.models import create_model
from utils.training import (
    ModelTrainer, save_results, check_result_exists, 
    create_result_filename, get_best_crop_size_from_results
)


def generate_all_combinations():
    """Generate all parameter combinations for augmentation sweep"""
    
    # Get best crop size from script 00 results
    best_crop_size = get_best_crop_size_from_results()
    
    # Parameters from CLAUDE.md
    max_zoom_values = [1.0, 1.25, 1.5]
    h_flip_p_values = [0, 0.5]
    grayscale_p_values = [0, 0.25, 0.5]
    blur_types = ['none', 'moderate', 'high']
    blur_p_values = [0, 0.25, 0.5]
    cutmix_p_values = [0, 0.25, 0.5]
    seeds = [0, 1, 2]
    
    combinations = []
    
    # Generate all parameter combinations
    for max_zoom in max_zoom_values:
        for h_flip_p in h_flip_p_values:
            for grayscale_p in grayscale_p_values:
                for blur_p in blur_p_values:
                    for cutmix_p in cutmix_p_values:
                        for seed in seeds:
                            
                            # Skip blur type when blur_p is 0  
                            blur_types_to_use = ['none'] if blur_p == 0 else blur_types
                            
                            for blur_type in blur_types_to_use:
                                params = {
                                    'crop_size': best_crop_size,  # Automatically uses best from script 00
                                    'resize_size': 256,
                                    'max_zoom': max_zoom,
                                    'h_flip_p': h_flip_p,
                                    'grayscale_p': grayscale_p,
                                    'blur_type': blur_type,
                                    'blur_p': blur_p,
                                    'cutmix_p': cutmix_p,
                                    'learning_rate': 0.001,
                                    'batch_size': 32,
                                    'epochs': 10,
                                    'dataset_sample': 0.1
                                }
                                combinations.append((params, seed))
    
    return combinations


def get_job_combinations(job_idx: int, max_jobs: int = 40) -> list:
    """Map job index to list of (params, seed) tuples - supports up to 40 jobs"""
    
    # Generate all combinations
    all_combinations = generate_all_combinations()
    total_combinations = len(all_combinations)
    
    print(f"Total combinations generated: {total_combinations}")
    
    # Handle case where job_idx exceeds available work
    if job_idx >= total_combinations:
        return []
    
    # Distribute evenly across available jobs
    per_job = total_combinations // max_jobs
    remainder = total_combinations % max_jobs
    
    # Jobs with indices < remainder get one extra combination
    if job_idx < remainder:
        start_idx = job_idx * (per_job + 1)
        end_idx = start_idx + per_job + 1
    else:
        start_idx = remainder * (per_job + 1) + (job_idx - remainder) * per_job
        end_idx = start_idx + per_job
    
    end_idx = min(end_idx, total_combinations)
    job_combinations = all_combinations[start_idx:end_idx]
    
    return job_combinations


def train_single_config(params: dict, seed: int, args: argparse.Namespace) -> dict:
    """Train one configuration and return results"""
    
    # Set seed
    set_all_seeds(seed)
    
    # Create output filename
    filename = create_result_filename(
        {k: v for k, v in params.items() if k not in ['learning_rate', 'batch_size', 'epochs', 'dataset_sample']}, 
        seed
    )
    output_path = os.path.join(args.output_dir, filename)
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print(f"Training config: zoom={params['max_zoom']}, hflip={params['h_flip_p']}, "
          f"grayscale={params['grayscale_p']}, blur_type={params['blur_type']}, blur_p={params['blur_p']}, "
          f"cutmix={params['cutmix_p']}, seed={seed}")
    
    # Load dataset
    print("Loading dataset...")
    dataset, _ = load_wolverines_dataset()
    
    # Create non-overlapping stratified train/val split (10% each per class)
    train_images, train_labels, val_images, val_labels = create_stratified_train_val_split(
        dataset,
        train_percentage=params['dataset_sample'],  # 10%
        val_percentage=params['dataset_sample'],     # 10%
        seed=seed
    )
    
    print(f"Train samples: {len(train_images)}, Val samples: {len(val_images)}")
    
    # Create transforms
    train_transform = create_transform_from_params(params, is_train=True)
    val_transform = create_transform_from_params(params, is_train=False)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_images, train_labels, val_images, val_labels,
        train_transform, val_transform, params['batch_size']
    )
    
    # Create model
    device = "cuda" if args.device == "gpu" else "cpu"
    model = create_model(device=device)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=params['learning_rate']
    )
    
    # Train
    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=params['epochs'],
        verbose=False  # Less verbose for many configurations
    )
    training_time = time.time() - start_time
    
    # Prepare final results
    final_results = {
        'job_idx': args.idx,
        'params': params,
        'seed': seed,
        'training_time': training_time,
        'final_val_f1': results['final_val_metrics']['f1_score'],
        'final_val_precision': results['final_val_metrics']['precision'],
        'final_val_recall': results['final_val_metrics']['recall'],
        'final_val_accuracy': results['final_val_metrics']['accuracy'],
        'best_val_f1': results['best_val_f1'],
        'epochs_trained': results['epochs_trained']
    }
    
    # Save results
    save_results(final_results, output_path, args.idx, params, seed)
    print(f"✓ Saved: F1={final_results['final_val_f1']:.4f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Sweep over augmentation parameters')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (0-39)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='results/01_augmentations',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Augmentations Sweep - Job {args.idx}")
    print("=" * 60)
    
    # Get combinations for this job
    combinations = get_job_combinations(args.idx)
    
    if not combinations:
        print(f"No combinations for job {args.idx}")
        return
    
    print(f"Processing {len(combinations)} configurations...")
    
    # Train each combination
    results_summary = []
    for i, (params, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(params, seed, args)
            if result:
                results_summary.append(result)
        except Exception as e:
            print(f"Error training config {i+1}: {e}")
            continue
        
        # Progress update every 10 configs
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{len(combinations)} configurations")
    
    print(f"\n" + "=" * 60)
    print(f"Job {args.idx} completed!")
    print(f"Successfully processed {len(results_summary)} configurations")
    
    if results_summary:
        avg_f1 = sum(r['final_val_f1'] for r in results_summary) / len(results_summary)
        best_f1 = max(r['final_val_f1'] for r in results_summary)
        print(f"Average validation F1: {avg_f1:.4f}")
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
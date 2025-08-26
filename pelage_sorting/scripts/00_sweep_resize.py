#!/usr/bin/env python3
"""
Script 00: Resize Size Sweep
Sweeps over different resize sizes (no center cropping).
Each job handles one resize size across all 3 seeds.
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add utils to path (assumes we cd to pelage_sorting in sbatch)
sys.path.append('.')

from utils.dataset import (
    load_wolverines_dataset, create_stratified_train_val_split,
    create_dataloaders, set_all_seeds
)
from utils.preprocessing import preprocess_dataset, get_standard_transform
from utils.models import create_model
from utils.training import (
    ModelTrainer, save_results, check_result_exists, 
    create_result_filename
)


def get_job_combinations(job_idx: int, max_jobs: int = 24) -> list:
    """Map job index to list of (params, seed) tuples - supports up to 24 jobs"""
    
    # Parameters from CLAUDE.md
    resize_sizes = [256, 512, 768, 1024]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Generate all combinations
    all_combinations = []
    for resize_size in resize_sizes:
        for seed in seeds:
            params = {
                'resize_size': resize_size,
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 10,
                'dataset_sample': 0.1  # 10% per class
            }
            all_combinations.append((params, seed))
    
    total_combinations = len(all_combinations)  # 32 total
    
    # Handle case where job_idx exceeds available jobs
    if job_idx >= max_jobs:
        return []
    
    # Distribute 32 combinations across 24 jobs
    # Jobs 0-7: 2 configs each (16 configs)
    # Jobs 8-23: 1 config each (16 configs)
    if job_idx < 8:
        # Jobs 0-7 get 2 combinations each
        start_idx = job_idx * 2
        end_idx = start_idx + 2
        return all_combinations[start_idx:end_idx]
    else:
        # Jobs 8-23 get 1 combination each
        config_idx = 16 + (job_idx - 8)  # Start after the 16 configs from jobs 0-7
        if config_idx < total_combinations:
            return [all_combinations[config_idx]]
        else:
            return []


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
    
    print(f"Training config: resize_size={params['resize_size']}, seed={seed}")
    
    # Load dataset with error handling
    print("Loading dataset from HuggingFace...")
    try:
        dataset, _ = load_wolverines_dataset()
        print(f"✓ Dataset loaded successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ ERROR: Failed to load dataset: {e}")
        print(f"  This could be due to:")
        print(f"  - No internet connection")
        print(f"  - HuggingFace Hub access issues")
        print(f"  - Missing authentication token")
        return None
    
    # Preprocess images using preprocessing utility
    print(f"Preprocessing images to {params['resize_size']}x{params['resize_size']}...")
    dataset = preprocess_dataset(dataset, params['resize_size'])
    
    # Create non-overlapping stratified train/val split (10% each per class)
    try:
        train_images, train_labels, val_images, val_labels = create_stratified_train_val_split(
            dataset,
            train_percentage=params['dataset_sample'],  # 10%
            val_percentage=params['dataset_sample'],     # 10%
            seed=seed
        )
        print(f"✓ Train samples: {len(train_images)}, Val samples: {len(val_images)}")
    except Exception as e:
        print(f"✗ ERROR: Failed to create train/val split: {e}")
        return None
    
    # Create transforms (simple: already resized, just normalize)
    transform = get_standard_transform()
    train_transform = transform
    val_transform = transform
    
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
        verbose=True
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
    print(f"✓ Saved results to: {filename}")
    print(f"  Final validation F1: {final_results['final_val_f1']:.4f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Sweep over image crop sizes')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (0-39)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='results/00_image_size',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Image Size Sweep - Job {args.idx}")
    print(f"Arguments: {vars(args)}")
    print("=" * 60)
    
    
    # Get combinations for this job
    combinations = get_job_combinations(args.idx)
    
    if not combinations:
        print(f"No work assigned to job {args.idx}")
        return
    
    print(f"Processing {len(combinations)} configurations:")
    for i, (params, seed) in enumerate(combinations):
        print(f"  {i+1}. resize_size={params['resize_size']}, seed={seed}")
    
    
    # Train each combination
    results_summary = []
    for i, (params, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(params, seed, args)
            if result:
                results_summary.append(result)
            else:
                print(f"Skipped configuration {i+1} (already exists)")
        except Exception as e:
            print(f"Error training config {i+1}: {e}")
            continue
    
    print(f"\n" + "=" * 60)
    print(f"Job {args.idx} completed!")
    print(f"Processed {len(results_summary)} configurations successfully")
    
    if results_summary:
        avg_f1 = sum(r['final_val_f1'] for r in results_summary) / len(results_summary)
        print(f"Average validation F1: {avg_f1:.4f}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
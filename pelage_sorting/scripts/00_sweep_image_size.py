#!/usr/bin/env python3
"""
Script 00: Image Size Sweep
Sweeps over different square center crop sizes before resizing to 256x256.
Each job handles one crop size across all 3 seeds.
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
from utils.transforms import create_transform_from_params
from utils.models import create_model
from utils.training import (
    ModelTrainer, save_results, check_result_exists, 
    create_result_filename
)
from utils.preemption import CheckpointManager, ProgressTracker


def get_job_combinations(job_idx: int, max_jobs: int = 40) -> list:
    """Map job index to list of (params, seed) tuples - supports up to 40 jobs"""
    
    # Parameters from CLAUDE.md
    crop_sizes = [256, 384, 512, 640, 768, 1024, 1152, 1280]
    seeds = [0, 1, 2]
    
    # Generate all combinations
    all_combinations = []
    for crop_size in crop_sizes:
        for seed in seeds:
            params = {
                'crop_size': crop_size,
                'resize_size': 256,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'dataset_sample': 0.1  # 10% per class
            }
            all_combinations.append((params, seed))
    
    total_combinations = len(all_combinations)  # 24 total
    
    # Handle case where job_idx exceeds available work
    if job_idx >= total_combinations:
        return []
    
    # Simple 1:1 mapping for image size sweep (24 combinations, use jobs 0-23)
    # Jobs 24-39 will have no work
    return [all_combinations[job_idx]]


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
    
    print(f"Training config: crop_size={params['crop_size']}, seed={seed}")
    
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
    
    # Setup preemption handling
    checkpoint_manager = CheckpointManager()
    experiment_name = "00_image_size"
    
    # Get combinations for this job
    combinations = get_job_combinations(args.idx)
    
    if not combinations:
        print(f"No work assigned to job {args.idx}")
        return
    
    print(f"Processing {len(combinations)} configurations:")
    for i, (params, seed) in enumerate(combinations):
        print(f"  {i+1}. crop_size={params['crop_size']}, seed={seed}")
    
    # Setup progress tracking
    progress_tracker = ProgressTracker(len(combinations))
    completed_configs = []
    
    # Setup signal handler for preemption
    def save_checkpoint():
        remaining_combinations = combinations[len(completed_configs):]
        checkpoint_manager.save_checkpoint(
            args.idx, experiment_name, completed_configs, remaining_combinations
        )
    
    checkpoint_manager.register_signal_handler(save_checkpoint)
    
    # Train each combination
    results_summary = []
    for i, (params, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(params, seed, args)
            if result:
                completed_configs.append({'params': params, 'seed': seed})
                progress_tracker.add_completed({'params': params, 'seed': seed}, result)
                results_summary.append(result)
            else:
                print(f"Skipped configuration {i+1} (already exists)")
        except Exception as e:
            print(f"Error training config {i+1}: {e}")
            progress_tracker.add_failed({'params': params, 'seed': seed}, str(e))
            continue
    
    # Clear checkpoint on successful completion
    checkpoint_manager.clear_checkpoint(args.idx, experiment_name)
    
    print(f"\n" + "=" * 60)
    print(f"Job {args.idx} completed!")
    progress_tracker.print_progress()
    
    if results_summary:
        avg_f1 = sum(r['final_val_f1'] for r in results_summary) / len(results_summary)
        print(f"Average validation F1: {avg_f1:.4f}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
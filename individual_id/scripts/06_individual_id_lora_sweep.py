#!/usr/bin/env python3
"""
Script 06: Individual ID LoRA Sweep
Test LoRA fine-tuning of DINOv3 attention layers for individual ID classification.
Uses fixed 32 train + 32 val samples per individual (label==1 only).
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

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to wolverines root
sys.path.append(wolverines_root)

from utils.dataset import load_wolverines_dataset, set_all_seeds, create_dataloaders
from utils.preprocessing import get_height_crop_and_resize_transform
from utils.training import check_result_exists, MultiClassTrainer
from utils.individual_id import get_feasible_individuals, create_sweep_dataset

# PEFT imports for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not available. Install with: pip install peft")
    PEFT_AVAILABLE = False


def get_job_combinations(job_idx: int, max_jobs: int = 24) -> list:
    """Map job index to list of (lora_r, lora_alpha, seed) tuples"""
    
    # LoRA hyperparameters
    lora_ranks = [8, 16, 32]
    lora_alphas = [8, 16, 32, 64, 128, 256]
    seeds = [0, 1, 2]  # 3 seeds for faster experimentation
    
    # Generate all combinations: 3 ranks × 6 alphas × 3 seeds = 54 total
    all_combinations = []
    for lora_r in lora_ranks:
        for lora_alpha in lora_alphas:
            for seed in seeds:
                all_combinations.append((lora_r, lora_alpha, seed))
    
    total_combinations = len(all_combinations)  # 54 total
    
    # Handle case where job_idx exceeds available jobs
    if job_idx >= max_jobs:
        return []
    
    # Distribute 54 combinations across 24 jobs
    # Jobs 0-5: 3 configs each (18 total)
    # Jobs 6-23: 2 configs each (36 total)
    if job_idx < 6:
        configs_per_job = 3
        start_idx = job_idx * 3
        end_idx = start_idx + 3
    else:
        configs_per_job = 2
        start_idx = 18 + (job_idx - 6) * 2
        end_idx = start_idx + 2
    
    if end_idx <= total_combinations:
        return all_combinations[start_idx:end_idx]
    else:
        return all_combinations[start_idx:total_combinations]


def create_lora_model(base_model, lora_r, lora_alpha, device):
    """Create LoRA-enabled model from base DINOv3 model
    
    Args:
        base_model: Base WolverinesModel with frozen backbone
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        device: Device for model
        
    Returns:
        PEFT model with LoRA adapters
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library required for LoRA. Install with: pip install peft")
    
    # Define LoRA configuration
    modules_to_save = ["classifier"]
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value", "dense"],  # Attention projection layers
        modules_to_save=modules_to_save,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,  # For backbone fine-tuning
    )
    
    # Apply PEFT to the backbone only
    # First, unfreeze backbone parameters for LoRA
    for param in base_model.backbone.parameters():
        param.requires_grad = True
    
    # Apply LoRA to backbone
    lora_backbone = get_peft_model(base_model.backbone, config)
    
    # Create new model with LoRA backbone
    class WolverinesLoRAModel(torch.nn.Module):
        def __init__(self, lora_backbone, classifier):
            super().__init__()
            self.backbone = lora_backbone
            self.classifier = classifier
            
        def forward(self, x):
            # LoRA backbone is trainable now
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
            return self.classifier(features)
        
        def get_trainable_parameters(self):
            # Return both LoRA parameters and classifier parameters
            params = list(self.backbone.parameters()) + list(self.classifier.parameters())
            return (p for p in params if p.requires_grad)
    
    lora_model = WolverinesLoRAModel(lora_backbone, base_model.classifier)
    lora_model = lora_model.to(device)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print(f"LoRA Model: {trainable_params:,} trainable / {total_params:,} total parameters")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    return lora_model


def train_single_config(lora_r: int, lora_alpha: int, seed: int, args: argparse.Namespace) -> dict:
    """Train one LoRA configuration and return results"""
    
    # Set seed
    set_all_seeds(seed)
    
    # Create output paths
    filename = f"lora_r={lora_r}_alpha={lora_alpha}_seed={seed}.json"
    results_dir = os.path.join(args.output_dir, "lora")
    output_path = os.path.join(results_dir, filename)
    
    # LoRA adapter output path
    adapter_dir = os.path.join(args.output_dir, "lora_adapters", f"r={lora_r}_alpha={lora_alpha}_seed={seed}")
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print(f"Training LoRA: r={lora_r}, alpha={lora_alpha}, seed={seed}")
    
    # Load dataset - TRAINING SET ONLY for individual ID experiments
    print("Loading dataset...")
    train_dataset, _ = load_wolverines_dataset()
    print(f"Training dataset: {len(train_dataset)} samples (training set only)")

    # Load pre-computed feasible individuals
    try:
        feasible_config = get_feasible_individuals(args.output_dir + '/feasible_individuals.json')
        feasible_individuals = feasible_config['feasible_individuals_pelage_64']
        individual_counts = {
            ind_id: {'label_1': feasible_config['individual_pelage_counts'][ind_id],
                    'total': feasible_config['individual_total_counts'][ind_id]}
            for ind_id in feasible_individuals
        }
        print(f"Loaded feasible individuals: {', '.join(feasible_individuals)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 00_count_individuals.py first to generate feasible individuals.")
        return None
    
    # Filter dataset to only feasible individuals
    feasible_ids_set = set(feasible_individuals)
    print(f"Filtering dataset to {len(feasible_individuals)} individuals...")
    
    filtered_dataset = train_dataset.filter(
        lambda x: x['id'] in feasible_ids_set
    )
    
    print(f"Filtered dataset: {len(filtered_dataset)} samples (down from {len(train_dataset)})")
    
    if len(feasible_individuals) < 3:
        print(f"Error: Only {len(feasible_individuals)} feasible individuals, need at least 3")
        return None
    
    # Create individual dataset - fixed 32 samples per individual, pelage only
    sample_size = 32
    approach = 'pelage'  # Only label==1 samples
    
    ind_train_dataset, ind_val_dataset, individual_to_class = create_sweep_dataset(
        filtered_dataset, feasible_individuals, sample_size, approach, seed
    )
    
    num_classes = len(feasible_individuals)
    print(f"Multi-class problem: {num_classes} individuals (classes)")
    print(f"Train samples: {len(ind_train_dataset)}")
    print(f"Val samples: {len(ind_val_dataset)}")
    
    # Create transforms - same as training pipeline
    batch_size = 16
    transform = get_height_crop_and_resize_transform(height=1280, resize=728)
    resize_size = "728x728"
    
    # Use standard dataloaders
    train_loader, val_loader = create_dataloaders(
        ind_train_dataset, ind_val_dataset,
        transform, transform, batch_size
    )
    
    # Create base model first
    device = "cuda" if args.device == "gpu" else "cpu"
    
    # Import here to avoid circular imports
    from utils.models import create_model
    base_model = create_model(device=device, num_classes=num_classes)
    
    print(f"Using device: {device}")
    print(f"Model classes: {num_classes}")
    
    # Apply LoRA to create trainable model
    lora_model = create_lora_model(base_model, lora_r, lora_alpha, device)
    
    # Create trainer for LoRA model
    trainer = MultiClassTrainer(
        model=lora_model,
        device=device,
        learning_rate=0.001,  # Same as baseline
        weight_decay=0.01
    )
    
    # Train for same duration as baseline for fair comparison
    start_time = time.time()
    epochs = 50  # Same as baseline individual ID training
    
    # Training loop with pelage-specific metrics
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validation with detailed metrics
        val_loss, val_acc, val_preds, val_labels = trainer.validate_epoch(val_loader)
        
        # Calculate pelage-specific metrics (should all be label==1 for pelage approach)
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
            
            if np.sum(visible_mask) > 0:
                val_preds_arr = np.array(val_preds)
                val_labels_arr = np.array(val_labels)
                visible_acc = sum(val_preds_arr[visible_mask] == val_labels_arr[visible_mask]) / np.sum(visible_mask)
                pelage_metrics['visible'] = {
                    'accuracy': visible_acc,
                    'count': int(np.sum(visible_mask))
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
            'pelage_metrics': pelage_metrics
        })
        
        # Print progress
        msg = f"Epoch {epoch+1:2d}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
        if pelage_metrics and 'visible' in pelage_metrics:
            msg += f", Vis={pelage_metrics['visible']['accuracy']:.4f}"
        print(msg)
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_val_loss, final_val_acc, final_preds, final_labels = trainer.validate_epoch(val_loader)
    
    # Save LoRA adapter
    os.makedirs(adapter_dir, exist_ok=True)
    lora_model.backbone.save_pretrained(adapter_dir)
    
    # Calculate classification report
    from sklearn.metrics import classification_report
    final_report = classification_report(final_labels, final_preds, output_dict=True, zero_division=0)
    
    results = {
        'epochs_trained': epochs,
        'final_val_accuracy': final_val_acc,
        'final_val_loss': final_val_loss,
        'final_classification_report': final_report,
        'confusion_matrix': confusion_matrix(final_labels, final_preds).tolist()
    }
    
    # Prepare comprehensive results
    final_results = {
        'job_idx': args.idx,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
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
            'approach': approach
        },
        'training_time': training_time,
        'performance': results,
        'train_history': trainer.train_history,
        'val_history': trainer.val_history,
        'lora_config': {
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'target_modules': ["query", "key", "value", "dense"],
            'modules_to_save': ["classifier"],
            'lora_dropout': 0.05,
            'bias': "none"
        },
        'experimental_params': {
            'resize_size': resize_size,
            'approach': approach,
            'approach_details': f'LoRA fine-tuning: r={lora_r}, alpha={lora_alpha} (1280px height crop → 728x728 square resize)',
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'adapter_saved_to': adapter_dir
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Saved results to: {filename}")
    print(f"✓ Saved LoRA adapter to: {adapter_dir}")
    print(f"  Final validation accuracy: {results['final_val_accuracy']:.4f}")
    print(f"  Training time: {training_time/60:.1f} minutes")
    
    return final_results


def main():
    if not PEFT_AVAILABLE:
        print("Error: PEFT library not available. Install with: pip install peft")
        return
    
    parser = argparse.ArgumentParser(description='Individual ID LoRA classification sweep')
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
    print(f"Individual ID LoRA Classification - Job {args.idx}")
    print("=" * 80)
    
    # Get combinations for this job
    combinations = get_job_combinations(args.idx)
    
    if not combinations:
        print(f"No combinations for job {args.idx}")
        return
    
    print(f"Processing {len(combinations)} LoRA configurations:")
    for lora_r, lora_alpha, seed in combinations:
        print(f"  LoRA r: {lora_r}, alpha: {lora_alpha}, seed: {seed}")
    
    # Train each combination
    results_summary = []
    for i, (lora_r, lora_alpha, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(lora_r, lora_alpha, seed, args)
            if result:
                results_summary.append(result)
        except Exception as e:
            print(f"Error training LoRA r={lora_r}, alpha={lora_alpha}, seed={seed}: {e}")
            continue
    
    print(f"\n" + "=" * 80)
    print(f"LoRA Job {args.idx} completed!")
    print(f"Successfully processed {len(results_summary)} configurations")
    
    if results_summary:
        avg_acc = sum(r['performance']['final_val_accuracy'] for r in results_summary) / len(results_summary)
        best_acc = max(r['performance']['final_val_accuracy'] for r in results_summary)
        print(f"Average final validation accuracy: {avg_acc:.4f}")
        print(f"Best final validation accuracy: {best_acc:.4f}")
        print(f"Results saved to: {os.path.join(args.output_dir, 'lora')}")


if __name__ == "__main__":
    main()
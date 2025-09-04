#!/usr/bin/env python3
"""
Script 01: MegaDescriptor Sweep (Crops-Masks Config)
Multi-class classification experiment using MegaDescriptor feature extraction
with nearest neighbor classification on the crops-masks dataset configuration.
Uses date-based temporal train/val splits and detection filtering.
"""

import sys
import os
import argparse
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Assume script is run from wolverines root directory
sys.path.append('.')

from utils.dataset import set_all_seeds
from utils.training import check_result_exists
from datasets import load_dataset


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


def load_crops_masks_dataset():
    """Load the wolverines dataset with crops-masks configuration"""
    print("Loading wolverines dataset (crops-masks config)...")
    dataset = load_dataset("kdoherty/wolverines", "crops-masks", split="train")
    print(f"Loaded {len(dataset)} samples from crops-masks config")
    return dataset


def create_megadescriptor_transform():
    """Create MegaDescriptor-specific transform pipeline for pre-cropped images"""
    import torchvision.transforms as T
    from PIL import Image
    
    print("MegaDescriptor transform: resize to 384x384 + normalize (using pre-cropped images)")
    
    return T.Compose([
        T.Resize(size=(384, 384), interpolation=Image.LANCZOS),
        T.ToTensor(), 
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def extract_features(dataset, model, transform, device, batch_size=16, use_crops=True):
    """Extract MegaDescriptor features for all samples in dataset"""
    model.eval()
    features = []
    labels = []
    individual_ids = []
    pelage_labels = []
    
    print(f"Extracting features from {len(dataset)} samples...")
    print(f"Using {'cropped images' if use_crops else 'SAM masks'} for feature extraction")
    
    # Process in batches for memory efficiency
    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        batch_samples = [dataset[j] for j in range(i, batch_end)]
        
        # Prepare batch
        batch_images = []
        batch_labels = []
        batch_ids = []
        batch_pelage = []
        
        for sample in batch_samples:
            # Use cropped images or SAM masks based on use_crops flag
            if use_crops and sample['megadetector_image'] is not None:
                img = sample['megadetector_image']
            elif not use_crops and sample['sam_mask'] is not None:
                img = sample['sam_mask']
            else:
                # Skip samples without the required image type
                continue
                
            img_tensor = transform(img)
            batch_images.append(img_tensor)
            batch_labels.append(sample['id'])  # Use individual ID as label
            batch_ids.append(sample['id'])
            batch_pelage.append(sample['pelage'])
        
        if not batch_images:  # Skip empty batches
            continue
            
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Extract features
        with torch.no_grad():
            batch_features = model(batch_tensor).cpu()
        
        features.append(batch_features)
        labels.extend(batch_labels)
        individual_ids.extend(batch_ids)
        pelage_labels.extend(batch_pelage)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_samples)}/{len(dataset)} samples")
    
    # Concatenate all features
    if features:
        all_features = torch.cat(features, dim=0)
        print(f"Extracted features shape: {all_features.shape}")
        return all_features, labels, individual_ids, pelage_labels
    else:
        print("No valid samples found for feature extraction!")
        return torch.empty(0), [], [], []


def nearest_neighbor_classify(val_features, train_features, train_labels):
    """Classify using cosine similarity nearest neighbor"""
    
    # Normalize for cosine similarity
    val_features_norm = F.normalize(val_features, dim=1)
    train_features_norm = F.normalize(train_features, dim=1)
    
    # Compute cosine similarities
    similarities = torch.mm(val_features_norm, train_features_norm.T)
    
    # Find nearest neighbors (highest similarity)
    nearest_indices = similarities.argmax(dim=1)
    predictions = train_labels[nearest_indices]
    
    # Get similarity scores for analysis
    max_similarities = similarities.max(dim=1)[0]
    
    return predictions, max_similarities.numpy()


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
            # Apply filtering based on approach
            if approach == 'pelage' and sample['pelage'] == 1:
                # For pelage==1: only include samples with detections
                if sample['megadetector_status'] == 1:
                    individual_samples[individual_id].append((i, sample))
            elif approach == 'pelage_abs' and sample['pelage'] == 0:
                # For pelage==0: no filtering needed
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


def evaluate_single_config(sample_size: int, approach: str, seed: int, args: argparse.Namespace) -> dict:
    """Evaluate one configuration using MegaDescriptor + nearest neighbor"""
    
    # Set seed
    set_all_seeds(seed)
    
    # Create output filename
    filename = f"samples={sample_size}_approach={approach}_seed={seed}.json"
    output_path = os.path.join(args.output_dir, filename)
    
    # Check if should skip
    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None
    
    print(f"Evaluating: samples={sample_size}, approach={approach}, seed={seed}")
    
    # Load crops-masks dataset
    dataset = load_crops_masks_dataset()

    # Load top 3 individuals
    import json
    config_path = 'individual_id/results/feasible_individuals.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    individuals_sorted = config['individuals_sorted_by_pelage']
    feasible_individuals = individuals_sorted[:3]
    print(f"Using top 3 individuals: {', '.join(feasible_individuals)}")
    
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
    
    # Create MegaDescriptor model and transform
    import timm
    device = "cuda" if args.device == "gpu" else "cpu"
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    model = model.to(device).eval()
    
    transform = create_megadescriptor_transform()
    
    print(f"Using device: {device}")
    print(f"Model: MegaDescriptor-L-384")
    
    # Extract features
    start_time = time.time()
    
    print("Extracting training features...")
    train_features, train_labels, train_ids, train_pelage = extract_features(
        ind_train_dataset, model, transform, device, batch_size=16
    )
    
    print("Extracting validation features...")
    val_features, val_labels, val_ids, val_pelage = extract_features(
        ind_val_dataset, model, transform, device, batch_size=16
    )
    
    feature_time = time.time() - start_time
    
    # Convert string IDs to numeric labels for classification
    unique_ids = sorted(list(set(train_labels + val_labels)))
    id_to_label = {id_str: idx for idx, id_str in enumerate(unique_ids)}
    
    train_labels_numeric = torch.tensor([id_to_label[id_str] for id_str in train_labels])
    val_labels_numeric = torch.tensor([id_to_label[id_str] for id_str in val_labels])
    
    # Nearest neighbor classification
    print("Performing nearest neighbor classification...")
    start_time = time.time()
    
    val_preds, similarity_scores = nearest_neighbor_classify(
        val_features, train_features, train_labels_numeric
    )
    
    classification_time = time.time() - start_time
    
    # Convert to numpy for metrics
    val_preds = val_preds.numpy()
    val_labels_numeric = val_labels_numeric.numpy()
    
    # Calculate F1 score (macro only)
    val_f1_macro = f1_score(val_labels_numeric, val_preds, average='macro', zero_division=0)
    val_acc = sum(val_preds == val_labels_numeric) / len(val_labels_numeric)
    
    # Calculate pelage-specific metrics
    pelage_metrics = {}
    if len(val_pelage) == len(val_preds):
        val_pelage_np = np.array(val_pelage)
        visible_mask = val_pelage_np == 1
        invisible_mask = val_pelage_np == 0
        
        if np.sum(visible_mask) > 0:
            visible_acc = sum(val_preds[visible_mask] == val_labels_numeric[visible_mask]) / np.sum(visible_mask)
            visible_f1 = f1_score(val_labels_numeric[visible_mask], val_preds[visible_mask], 
                                 average='weighted', zero_division=0)
            pelage_metrics['visible'] = {
                'accuracy': visible_acc,
                'f1_score': visible_f1,
                'count': int(np.sum(visible_mask)),
                'avg_similarity': float(similarity_scores[visible_mask].mean())
            }
        
        if np.sum(invisible_mask) > 0:
            invisible_acc = sum(val_preds[invisible_mask] == val_labels_numeric[invisible_mask]) / np.sum(invisible_mask)
            invisible_f1 = f1_score(val_labels_numeric[invisible_mask], val_preds[invisible_mask], 
                                   average='weighted', zero_division=0)
            pelage_metrics['invisible'] = {
                'accuracy': invisible_acc,
                'f1_score': invisible_f1,
                'count': int(np.sum(invisible_mask)),
                'avg_similarity': float(similarity_scores[invisible_mask].mean())
            }
    
    # Calculate classification report
    final_report = classification_report(val_labels_numeric, val_preds, output_dict=True, zero_division=0)
    
    results = {
        'final_val_accuracy': val_acc,
        'final_val_f1_macro': val_f1_macro,
        'final_classification_report': final_report,
        'final_pelage_metrics': pelage_metrics,
        'confusion_matrix': confusion_matrix(val_labels_numeric, val_preds).tolist(),
        'avg_similarity_score': float(similarity_scores.mean()),
        'similarity_stats': {
            'min': float(similarity_scores.min()),
            'max': float(similarity_scores.max()),
            'std': float(similarity_scores.std())
        }
    }
    
    # Comprehensive results
    final_results = {
        'job_idx': args.idx,
        'sample_size': sample_size,
        'approach': approach,
        'seed': seed,
        'num_classes': num_classes,
        'individual_ids': feasible_individuals,
        'individual_to_class': individual_to_class,
        'temporal_split_info': temporal_info,
        'dataset_stats': {
            'train_samples': len(ind_train_dataset),
            'val_samples': len(ind_val_dataset),
            'samples_per_individual': sample_size
        },
        'feature_extraction_time': feature_time,
        'classification_time': classification_time,
        'performance': results,
        'experimental_params': {
            'model': 'MegaDescriptor-L-384',
            'classifier': 'nearest_neighbor_cosine',
            'approach': approach,
            'approach_details': f'Temporal split approach: {approach} (uses crops-masks config with detection filtering)',
            'dataset_config': 'crops-masks',
            'feature_dim': train_features.shape[1] if len(train_features) > 0 else 0,
            'batch_size': 16,
            'detection_filtering': approach == 'pelage'
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Saved results to: {filename}")
    print(f"  Final validation accuracy: {results['final_val_accuracy']:.4f}")
    print(f"  Final validation F1 (macro): {results['final_val_f1_macro']:.4f}")
    print(f"  Average similarity score: {results['avg_similarity_score']:.4f}")
    
    # Show pelage-specific performance if available
    if pelage_metrics:
        if 'visible' in pelage_metrics:
            vis_acc = pelage_metrics['visible']['accuracy']
            vis_f1 = pelage_metrics['visible']['f1_score']
            vis_sim = pelage_metrics['visible']['avg_similarity']
            vis_count = pelage_metrics['visible']['count']
            print(f"  Visible accuracy: {vis_acc:.4f}, F1: {vis_f1:.4f}, Sim: {vis_sim:.4f} (n={vis_count})")
        if 'invisible' in pelage_metrics:
            inv_acc = pelage_metrics['invisible']['accuracy']
            inv_f1 = pelage_metrics['invisible']['f1_score']
            inv_sim = pelage_metrics['invisible']['avg_similarity']
            inv_count = pelage_metrics['invisible']['count']
            print(f"  Invisible accuracy: {inv_acc:.4f}, F1: {inv_f1:.4f}, Sim: {inv_sim:.4f} (n={inv_count})")
    
    print(f"  Feature extraction time: {feature_time/60:.1f} minutes")
    print(f"  Classification time: {classification_time:.2f} seconds")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='MegaDescriptor individual ID classification sweep')
    parser.add_argument('--idx', type=int, required=True, 
                       help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                       default='megadescriptor/results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                       default='gpu', help='Device to use for feature extraction')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"MegaDescriptor Individual ID Classification - Job {args.idx}")
    print("=" * 80)
    
    # Get combinations for this job
    combinations = get_job_combinations(args.idx)
    
    if not combinations:
        print(f"No combinations for job {args.idx}")
        return
    
    print(f"Processing {len(combinations)} configurations:")
    for sample_size, approach, seed in combinations:
        print(f"  Samples: {sample_size}, Approach: {approach}, Seed: {seed}")
    
    # Evaluate each combination
    results_summary = []
    for i, (sample_size, approach, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = evaluate_single_config(sample_size, approach, seed, args)
            if result:
                results_summary.append(result)
        except Exception as e:
            print(f"Error evaluating samples={sample_size}, approach={approach}, seed={seed}: {e}")
            continue
    
    print(f"\n" + "=" * 80)
    print(f"Job {args.idx} completed!")
    print(f"Successfully processed {len(results_summary)} configurations")
    
    if results_summary:
        avg_acc = sum(r['performance']['final_val_accuracy'] for r in results_summary) / len(results_summary)
        best_acc = max(r['performance']['final_val_accuracy'] for r in results_summary)
        avg_f1 = sum(r['performance']['final_val_f1_macro'] for r in results_summary) / len(results_summary)
        best_f1 = max(r['performance']['final_val_f1_macro'] for r in results_summary)
        avg_sim = sum(r['performance']['avg_similarity_score'] for r in results_summary) / len(results_summary)
        print(f"Average final validation accuracy: {avg_acc:.4f}")
        print(f"Best final validation accuracy: {best_acc:.4f}")
        print(f"Average final validation F1 (macro): {avg_f1:.4f}")
        print(f"Best final validation F1 (macro): {best_f1:.4f}")
        print(f"Average similarity score: {avg_sim:.4f}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
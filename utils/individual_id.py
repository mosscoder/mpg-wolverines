"""
Individual ID utilities for wolverine identification experiments.
Shared functions for loading data, managing qualified individuals, and handling model checkpoints.
"""

import os
import json
import glob
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from .dataset import set_all_seeds


def get_qualified_individuals(min_label1_samples=32, analysis_path='individual_id/results/individual_analysis.json'):
    """Get individuals with minimum label==1 samples from individual_analysis.json
    
    Args:
        min_label1_samples: Minimum number of pelage visible samples required
        analysis_path: Path to individual_analysis.json file
        
    Returns:
        List of qualified individual IDs
    """
    if not os.path.exists(analysis_path):
        raise FileNotFoundError(f"Individual analysis not found at {analysis_path}. Run 00_count_individuals.py first.")
    
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    qualified = []
    individual_stats = analysis['individual_statistics']
    
    for ind_id, stats in individual_stats.items():
        label1_count = stats['label_1_samples']
        if label1_count >= min_label1_samples:
            qualified.append(ind_id)
            print(f"  {ind_id}: {label1_count} label==1 samples")
    
    print(f"Found {len(qualified)} qualified individuals with >= {min_label1_samples} label==1 samples")
    return qualified


def load_best_cv_epochs(cv_results_dir, verbose=True):
    """Load cross-validation results and determine optimal epoch count
    
    Args:
        cv_results_dir: Directory containing fold_*.json files
        verbose: Whether to print detailed results
        
    Returns:
        Tuple of (optimal_epochs, mean_cv_f1, fold_results)
    """
    cv_files = glob.glob(os.path.join(cv_results_dir, "fold_*.json"))
    
    if not cv_files:
        raise FileNotFoundError(f"No CV results found in {cv_results_dir}")
    
    if verbose:
        print(f"Found {len(cv_files)} CV result files")
    
    fold_results = []
    for cv_file in cv_files:
        with open(cv_file, 'r') as f:
            fold_data = json.load(f)
            fold_results.append(fold_data)
    
    # Extract best epochs and F1 scores from each fold
    fold_best_epochs = []
    fold_best_f1s = []
    
    for fold_data in fold_results:
        fold_idx = fold_data['fold_idx']
        best_epoch = fold_data['best_epoch']['epoch_num']
        best_f1 = fold_data['best_epoch']['val_f1']
        
        fold_best_epochs.append(best_epoch)
        fold_best_f1s.append(best_f1)
        
        if verbose:
            print(f"Fold {fold_idx}: Best epoch {best_epoch} (F1: {best_f1:.4f})")
    
    # Use mean best epoch across folds
    optimal_epochs = int(np.mean(fold_best_epochs))
    mean_cv_f1 = np.mean(fold_best_f1s)
    
    if verbose:
        print(f"Optimal epochs (mean across folds): {optimal_epochs}")
        print(f"Mean CV F1 score: {mean_cv_f1:.4f}")
    
    return optimal_epochs, mean_cv_f1, fold_results


def create_individual_datasets(train_dataset, test_dataset, individual_ids, label_only=True):
    """Create train/test datasets for individual ID classification
    
    Args:
        train_dataset: HuggingFace training dataset
        test_dataset: HuggingFace test dataset  
        individual_ids: List of individual IDs to include
        label_only: Whether to filter to label==1 (pelage visible) only
        
    Returns:
        Tuple of (train_encoded, test_encoded, train_counts, test_counts)
    """
    qualified_ids_set = set(individual_ids)
    
    # Filter datasets
    if label_only:
        train_filtered = train_dataset.filter(
            lambda x: x['id'] in qualified_ids_set and x['label'] == 1
        )
        test_filtered = test_dataset.filter(
            lambda x: x['id'] in qualified_ids_set and x['label'] == 1
        )
    else:
        train_filtered = train_dataset.filter(
            lambda x: x['id'] in qualified_ids_set
        )
        test_filtered = test_dataset.filter(
            lambda x: x['id'] in qualified_ids_set
        )
    
    print(f"Training samples: {len(train_filtered)}")
    print(f"Test samples: {len(test_filtered)}")
    
    # Count samples per individual
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for item in train_filtered:
        train_counts[item['id']] += 1
    
    for item in test_filtered:
        test_counts[item['id']] += 1
    
    print("Training samples per individual:")
    for ind_id in individual_ids:
        print(f"  {ind_id}: {train_counts[ind_id]}")
    
    print("Test samples per individual:")  
    for ind_id in individual_ids:
        print(f"  {ind_id}: {test_counts[ind_id]}")
    
    # Encode individual IDs as class labels
    train_encoded = train_filtered.class_encode_column('id')
    train_encoded = train_encoded.rename_column('id', 'class_label')
    train_encoded = train_encoded.rename_column('label', 'pelage')
    train_encoded = train_encoded.rename_column('class_label', 'label')
    
    test_encoded = test_filtered.class_encode_column('id')
    test_encoded = test_encoded.rename_column('id', 'class_label') 
    test_encoded = test_encoded.rename_column('label', 'pelage')
    test_encoded = test_encoded.rename_column('class_label', 'label')
    
    return train_encoded, test_encoded, dict(train_counts), dict(test_counts)


def save_model_checkpoint(model, individual_ids, output_path, config=None):
    """Save linear layer weights and metadata for individual ID model
    
    Args:
        model: Trained model with classifier attribute
        individual_ids: List of individual IDs (class names)
        output_path: Path to save checkpoint
        config: Optional config dict with training parameters
    """
    # Extract linear layer (last layer of classifier)
    linear_layer = model.classifier[-1]
    
    checkpoint = {
        'linear_weights': linear_layer.weight.data.cpu(),
        'linear_bias': linear_layer.bias.data.cpu() if linear_layer.bias is not None else None,
        'individual_ids': individual_ids,
        'num_classes': len(individual_ids),
        'model_config': config or {}
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(checkpoint, output_path)
    print(f"✓ Saved model checkpoint to: {output_path}")


def load_model_checkpoint(model_path, device='cpu'):
    """Load trained model weights and metadata
    
    Args:
        model_path: Path to saved checkpoint
        device: Device to load tensors on
        
    Returns:
        Tuple of (checkpoint_dict, individual_ids, linear_weights)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    individual_ids = checkpoint['individual_ids']
    num_classes = checkpoint['num_classes']
    linear_weights = checkpoint['linear_weights']
    
    print(f"Loaded model: {num_classes} classes")
    print(f"Individual IDs: {individual_ids}")
    print(f"Linear weights shape: {linear_weights.shape}")
    
    return checkpoint, individual_ids, linear_weights


def create_cv_folds(dataset, individual_ids, n_folds=5, seed=42, train_ratio=0.8):
    """Create stratified cross-validation folds for individual ID
    
    Args:
        dataset: HuggingFace dataset (should be pre-filtered to qualified individuals and label==1)
        individual_ids: List of individual IDs
        n_folds: Number of cross-validation folds
        seed: Random seed for reproducibility
        train_ratio: Ratio of data to use for training in each fold
        
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    set_all_seeds(seed)
    
    # Group indices by individual (only label==1 samples)
    individual_indices = defaultdict(list)
    for i, item in enumerate(dataset):
        if item['id'] in individual_ids and item['label'] == 1:
            individual_indices[item['id']].append(i)
    
    print(f"Label==1 samples per individual:")
    for ind_id in individual_ids:
        print(f"  {ind_id}: {len(individual_indices[ind_id])}")
    
    # Create folds for each individual
    folds = []
    for fold_idx in range(n_folds):
        train_indices = []
        val_indices = []
        
        for ind_id in individual_ids:
            indices = individual_indices[ind_id].copy()
            np.random.shuffle(indices)
            
            # Split based on train_ratio (e.g., 80% train, 20% val)
            n_samples = len(indices)
            val_size = max(1, int(n_samples * (1 - train_ratio)))
            train_size = n_samples - val_size
            
            # Rotate which samples are used for validation in each fold
            start_idx = (fold_idx * val_size) % n_samples
            end_idx = min(start_idx + val_size, n_samples)
            
            val_fold_indices = indices[start_idx:end_idx]
            train_fold_indices = indices[:start_idx] + indices[end_idx:]
            
            train_indices.extend(train_fold_indices)
            val_indices.extend(val_fold_indices)
            
            print(f"  Fold {fold_idx}, {ind_id}: {len(train_fold_indices)} train, {len(val_fold_indices)} val")
        
        folds.append((train_indices, val_indices))
        print(f"Fold {fold_idx} total: {len(train_indices)} train, {len(val_indices)} val")
    
    return folds


def calculate_metrics(predictions, labels, average='weighted'):
    """Calculate classification metrics
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dict with accuracy, f1_score, and classification_report
    """
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    f1 = f1_score(labels, predictions, average=average, zero_division=0)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }


def select_test_images_per_individual(test_dataset, individual_ids, seed=42):
    """Select one test image per individual for visualization
    
    Args:
        test_dataset: HuggingFace test dataset
        individual_ids: List of individual IDs
        seed: Random seed
        
    Returns:
        Dict mapping individual_id -> image info dict
    """
    set_all_seeds(seed)
    
    selected_images = {}
    
    for ind_id in individual_ids:
        # Find all test images for this individual with label==1
        individual_samples = [
            (i, item) for i, item in enumerate(test_dataset) 
            if item['id'] == ind_id and item['label'] == 1
        ]
        
        if individual_samples:
            # Randomly select one image
            idx, sample = np.random.choice(len(individual_samples), 1)[0]
            idx, sample = individual_samples[idx]
            selected_images[ind_id] = {
                'index': idx,
                'image': sample['image'],
                'sample': sample
            }
            print(f"{ind_id}: Selected test image {idx}")
        else:
            print(f"Warning: No label==1 test images found for {ind_id}")
    
    return selected_images


def get_feasible_individuals(config_path='results/feasible_individuals.json'):
    """Load pre-computed feasible individuals from script 00
    
    Args:
        config_path: Path to feasible individuals JSON file
        
    Returns:
        Config dict with feasible individuals and counts
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Feasible individuals config not found at {config_path}. Run script 00 first.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_sweep_dataset(dataset, individual_ids, sample_size, approach, seed):
    """Create train/val dataset views for individual ID sweep experiments
    
    Args:
        dataset: HuggingFace dataset to sample from
        individual_ids: List of individual IDs to include
        sample_size: Number of training samples per individual
        approach: 'pelage' (visible only) or 'pelage_abs' (invisible only)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        individual_to_class: Mapping from individual ID to class index
    """
    import random
    
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
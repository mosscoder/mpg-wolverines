import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import Dict, Any, List, Tuple, Optional
import time


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics"""
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_score': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
    }



class ModelTrainer:
    """Trainer class for wolverines classification"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01):
        
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
            model.get_trainable_parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = []
        self.val_history = []
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              epochs: int,
              verbose: bool = True) -> Dict[str, Any]:
        """Train the model"""
        
        start_time = time.time()
        best_val_f1 = 0
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_history.append(val_metrics)
            
            # Track best F1
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1_score']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
        
        training_time = time.time() - start_time
        
        return {
            'final_train_metrics': self.train_history[-1],
            'final_val_metrics': self.val_history[-1],
            'best_val_f1': best_val_f1,
            'training_time': training_time,
            'epochs_trained': len(self.train_history)
        }


def save_results(results: Dict[str, Any], 
                output_path: str,
                job_idx: int,
                params: Dict[str, Any],
                seed: int):
    """Save training results to JSON"""
    
    # Prepare results dictionary
    result_dict = {
        'job_idx': job_idx,
        'params': params,
        'seed': seed,
        'timestamp': time.time(),
        **results
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all results from a directory"""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return results
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except json.JSONDecodeError:
            print(f"Warning: Could not load {json_file}")
    
    return results


def check_result_exists(output_path: str) -> bool:
    """Check if result file already exists"""
    return Path(output_path).exists()


def create_result_filename(params: Dict[str, Any], seed: int) -> str:
    """Create standardized result filename"""
    param_strs = []
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            param_strs.append(f"{key}={value:.6f}")
        else:
            param_strs.append(f"{key}={value}")
    
    param_str = "_".join(param_strs)
    return f"{param_str}_seed={seed}.json"


def find_best_params_from_results(results_dir: str, 
                                  param_names: List[str] = None,
                                  metric: str = 'final_val_f1') -> Dict[str, Any]:
    """
    Find best parameters from experiment results.
    
    Args:
        results_dir: Directory containing result JSON files
        param_names: List of parameter names to group by (None = all params except seed)
        metric: Metric to optimize (default: final_val_f1)
        
    Returns:
        Dict with best parameters and their average performance
    """
    results = load_results(results_dir)
    if not results:
        return {}
    
    # Group by parameters (excluding seed)
    grouped = {}
    for result in results:
        params = result.get('params', {})
        if param_names:
            # Group by specific parameters
            key = tuple(params.get(name) for name in param_names)
        else:
            # Use all params except seed and metadata
            exclude_keys = {'seed', 'learning_rate', 'batch_size', 'epochs', 'dataset_sample'}
            key = tuple(sorted((k, v) for k, v in params.items() if k not in exclude_keys))
        
        if key not in grouped:
            grouped[key] = []
        
        # Extract metric value - handle nested structure
        metric_value = result.get(metric)
        if metric_value is None and 'final_val_metrics' in result:
            # Handle case where metric is in nested dict
            metric_key = metric.replace('final_val_', '')
            metric_value = result['final_val_metrics'].get(metric_key, 0)
        
        grouped[key].append(metric_value or 0)
    
    # Find best average performance
    best_key = None
    best_score = -1
    for key, scores in grouped.items():
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_key = key
    
    # Return params dict
    if param_names and best_key:
        result_dict = {name: value for name, value in zip(param_names, best_key)}
        result_dict['avg_score'] = best_score
        return result_dict
    elif best_key:
        result_dict = dict(best_key)
        result_dict['avg_score'] = best_score
        return result_dict
    
    return {}


def get_best_crop_size_from_results(results_dir: str = 'results/00_image_size') -> int:
    """
    Get best crop size from script 00 results.
    
    Args:
        results_dir: Directory containing script 00 results
        
    Returns:
        Best crop size or default 640 if no results
    """
    if not os.path.exists(results_dir):
        print(f"Warning: {results_dir} not found, using default crop_size=640")
        return 640
    
    best = find_best_params_from_results(results_dir, ['crop_size'])
    if best and 'crop_size' in best:
        print(f"Found best crop_size={best['crop_size']} (avg F1: {best.get('avg_score', 0):.4f})")
        return best['crop_size']
    else:
        print(f"No results in {results_dir}, using default crop_size=640")
        return 640


def get_best_augmentation_params(results_dir: str = 'results/01_augmentations') -> Dict[str, Any]:
    """
    Get best augmentation parameters from script 01 results.
    
    Args:
        results_dir: Directory containing script 01 results
        
    Returns:
        Dict with best augmentation parameters
    """
    if not os.path.exists(results_dir):
        print(f"Warning: {results_dir} not found, using default augmentation params")
        return {
            'max_zoom': 1.25,
            'h_flip_p': 0.5,
            'grayscale_p': 0.25,
            'blur_type': 'moderate',
            'blur_p': 0.25,
            'cutmix_p': 0.25
        }
    
    param_names = ['max_zoom', 'h_flip_p', 'grayscale_p', 'blur_type', 'blur_p', 'cutmix_p']
    best = find_best_params_from_results(results_dir, param_names)
    
    if best and len(best) > 1:  # More than just avg_score
        avg_score = best.pop('avg_score', 0)
        print(f"Found best augmentation params (avg F1: {avg_score:.4f}):")
        for key, value in best.items():
            print(f"  {key}: {value}")
        return best
    else:
        print(f"No results in {results_dir}, using default augmentation params")
        return {
            'max_zoom': 1.25,
            'h_flip_p': 0.5,
            'grayscale_p': 0.25,
            'blur_type': 'moderate',
            'blur_p': 0.25,
            'cutmix_p': 0.25
        }


def get_best_learning_rate_from_results(results_dir: str = 'results/02_learning_rate') -> Dict[str, Any]:
    """
    Get best learning rate and optimal epochs from script 02 results.
    
    Args:
        results_dir: Directory containing script 02 results
        
    Returns:
        Dict with best learning rate and optimal epochs info
    """
    if not os.path.exists(results_dir):
        print(f"Warning: {results_dir} not found, using default LR=0.001")
        return {'learning_rate': 0.001, 'optimal_epochs': 30}
    
    # Find best learning rate across all folds
    best = find_best_params_from_results(results_dir, ['learning_rate'])
    
    if best and 'learning_rate' in best:
        best_lr = best['learning_rate']
        avg_score = best.get('avg_score', 0)
        print(f"Found best learning_rate={best_lr} (avg F1: {avg_score:.4f})")
        
        # TODO: Analyze epoch-wise performance to find optimal number of epochs
        # For now, use a reasonable default based on typical training curves
        optimal_epochs = 30
        
        return {
            'learning_rate': best_lr,
            'optimal_epochs': optimal_epochs,
            'avg_score': avg_score
        }
    else:
        print(f"No results in {results_dir}, using default LR=0.001")
        return {'learning_rate': 0.001, 'optimal_epochs': 30}



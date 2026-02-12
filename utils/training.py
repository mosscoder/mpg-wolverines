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
    """Compute classification metrics using macro averaging for binary classification"""
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_score': f1_score(labels, predictions, average='macro'),
        'precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(labels, predictions, average='macro', zero_division=0)
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
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1_score']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1_score']:.4f}", 
                      flush=True)
        
        training_time = time.time() - start_time
        
        return {
            'final_train_metrics': self.train_history[-1],
            'final_val_metrics': self.val_history[-1],
            'best_val_f1': best_val_f1,
            'training_time': training_time,
            'epochs_trained': len(self.train_history)
        }
    
    def save_classifier_head(self, save_path: str):
        """
        Save only the classifier head (linear layer) weights.
        The frozen backbone is not saved since it's pretrained and unchanging.
        
        Args:
            save_path: Path to save the classifier weights (.pth file)
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save only the classifier state dict
        torch.save(self.model.classifier.state_dict(), save_path)
        print(f"Classifier head saved to: {save_path}")
    
    def load_classifier_head(self, load_path: str):
        """
        Load classifier head weights.
        
        Args:
            load_path: Path to load the classifier weights from (.pth file)
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Classifier weights not found at: {load_path}")
        
        # Load classifier state dict
        classifier_state = torch.load(load_path, map_location=self.device)
        self.model.classifier.load_state_dict(classifier_state)
        print(f"Classifier head loaded from: {load_path}")


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



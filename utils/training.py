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
        'f1_score': f1_score(labels, predictions, average='macro'),
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


class MultiClassTrainer:
    """Base trainer class for multi-class individual identification"""
    
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
                batch_images, batch_labels, _ = batch_data
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
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    batch_images, batch_labels, _ = batch_data
                else:
                    batch_images, batch_labels = batch_data
                
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        return avg_loss, accuracy, all_predictions, all_labels


class CVTrainer(MultiClassTrainer):
    """Cross-validation trainer for hyperparameter optimization"""
    
    def validate_epoch_with_f1(self, val_loader):
        """Validate epoch and return F1 score for optimization"""
        from sklearn.metrics import f1_score
        
        avg_loss, accuracy, predictions, labels = self.validate_epoch(val_loader)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        return avg_loss, accuracy, f1, predictions, labels
    
    def train_cv_fold(self, train_loader, val_loader, epochs=50, verbose=True):
        """Train one cross-validation fold"""
        epoch_results = []
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation with F1 score
            val_loss, val_acc, val_f1, _, _ = self.validate_epoch_with_f1(val_loader)
            
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            }
            epoch_results.append(epoch_result)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        return epoch_results


class FinalTrainer(MultiClassTrainer):
    """Final model trainer with evaluation capabilities"""
    
    def evaluate(self, test_loader, individual_ids):
        """Comprehensive evaluation on test set"""
        from sklearn.metrics import f1_score, classification_report, confusion_matrix
        
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    batch_images, batch_labels, _ = batch_data
                else:
                    batch_images, batch_labels = batch_data
                
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # Classification report with individual names
        report = classification_report(all_labels, all_predictions, 
                                     target_names=individual_ids,
                                     output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels
        }
    
    def train_final(self, train_loader, epochs, verbose=True):
        """Train final model for specified epochs"""
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'accuracy': train_acc
            })
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        print(f"Final training accuracy: {train_acc:.4f}")
        return train_acc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoModel
from datasets import load_dataset
from sklearn.metrics import f1_score
import numpy as np
from PIL import Image
import random
from typing import List, Tuple, Dict, Any


class WolverinesDataset(Dataset):
    def __init__(self, images: List[Any], labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        
    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x)
            # Get CLS token (first token)
            features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return features


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)


class WolverinesTrainer:
    def __init__(self, device: str = "mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize transforms for 512x512 images
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models
        print("Loading DINOv3 feature extractor...")
        self.feature_extractor = DINOv3FeatureExtractor().to(self.device)
        self.classifier = LinearClassifier().to(self.device)
        
        # Training configuration
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def create_stratified_split(self, dataset, train_ratio: float = 0.8) -> Tuple[List, List, List, List]:
        """Create stratified train/val split with 80/20 split per class"""
        
        # Group indices by label
        label_to_indices = {0: [], 1: []}
        for idx, item in enumerate(dataset):
            label = item['label']
            label_to_indices[label].append(idx)
        
        print(f"Label distribution in full dataset:")
        for label, indices in label_to_indices.items():
            print(f"  Label {label}: {len(indices)} samples")
        
        # Split each class by train_ratio
        train_images, train_labels = [], []
        val_images, val_labels = [], []
        
        for label in [0, 1]:
            indices = label_to_indices[label]
            
            # Calculate split sizes
            train_size = int(len(indices) * train_ratio)
            
            # Randomly shuffle and split
            random.shuffle(indices)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]
            
            # Add to splits
            for idx in train_idx:
                train_images.append(dataset[idx]['image'])
                train_labels.append(label)
            
            for idx in val_idx:
                val_images.append(dataset[idx]['image'])
                val_labels.append(label)
        
        print(f"\nStratified split created:")
        print(f"  Train: {len(train_images)} samples")
        print(f"  Val: {len(val_images)} samples")
        print(f"  Train label distribution: {np.bincount(train_labels)}")
        print(f"  Val label distribution: {np.bincount(val_labels)}")
        
        return train_images, train_labels, val_images, val_labels
    
    def create_dataloaders(self, batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders"""
        print("Loading wolverines dataset...")
        dataset = load_dataset("kdoherty/wolverines", split="train")
        
        # Create stratified split
        train_images, train_labels, val_images, val_labels = self.create_stratified_split(dataset)
        
        # Create datasets
        train_dataset = WolverinesDataset(train_images, train_labels, self.transform)
        val_dataset = WolverinesDataset(val_images, val_labels, self.transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model and return accuracy and F1 score"""
        self.feature_extractor.eval()
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                features = self.feature_extractor(images)
                
                # Classify
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_loss = total_loss / len(val_loader)
        
        return accuracy, f1, avg_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.feature_extractor.eval()  # Keep feature extractor frozen
        self.classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Extract features (no gradients)
            with torch.no_grad():
                features = self.feature_extractor(images)
            
            # Classify (with gradients)
            outputs = self.classifier(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        
        return accuracy, avg_loss
    
    def train(self, epochs: int = 10, batch_size: int = 16):
        """Main training loop"""
        print("Setting up training...")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(batch_size)
        
        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 60)
        
        best_f1 = 0
        
        for epoch in range(epochs):
            # Train
            train_acc, train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_acc, val_f1, val_loss = self.validate(val_loader)
            
            # Track best F1
            if val_f1 > best_f1:
                best_f1 = val_f1
            
            # Print epoch results
            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        print("-" * 60)
        print(f"Training completed!")
        print(f"Final validation F1 score: {val_f1:.4f}")
        print(f"Best validation F1 score: {best_f1:.4f}")
        
        return val_f1


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)
    
    print("Wolverines Classification Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = WolverinesTrainer(device="mps")
    
    # Train the model
    final_f1 = trainer.train(epochs=30, batch_size=16)
    
    print(f"\n🎯 Final validation F1 score: {final_f1:.4f}")


if __name__ == "__main__":
    main()
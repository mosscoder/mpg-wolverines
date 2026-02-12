import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random
import numpy as np
from typing import List, Tuple, Any
from sklearn.model_selection import KFold


class WolverinesDataset(Dataset):
    """Memory-efficient dataset class that works with HuggingFace datasets"""
    
    def __init__(self, hf_dataset, transform=None, label_key='label'):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.label_key = label_key
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']  # PIL Image from HuggingFace dataset
        label = item[self.label_key]
        
        if self.transform:
            image = self.transform(image)
        
        # Check if pelage information is available
        if 'pelage' in item:
            pelage = item['pelage']
            return image, label, pelage
        
        return image, label


def remap_pelage_labels(example):
    """
    Remap pelage labels for binary classification: 2 vs all
    - Original 2 (Full) → 1 (positive class)  
    - Original 0,1 (None/Partial) → 0 (negative class)
    """
    if example['label'] == 2:  # Full pelage
        example['label'] = 1
    else:  # None (0) or Partial (1) 
        example['label'] = 0
    return example


def load_wolverines_dataset():
    """Load the wolverines dataset from HuggingFace using pelage config with label remapping"""
    dataset = load_dataset("kdoherty/wolverines", name="pelage", split="train")
    test_dataset = load_dataset("kdoherty/wolverines", name="pelage", split="test")
    
    # Remap labels for binary classification (2 vs all)
    dataset = dataset.map(remap_pelage_labels)
    test_dataset = test_dataset.map(remap_pelage_labels)
    
    return dataset, test_dataset


def create_kfold_splits(dataset, n_folds: int = 5, seed: int = 42):
    """Create k-fold cross-validation splits using HF dataset select()"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Group indices by label for stratification
    label_to_indices = {0: [], 1: []}
    for idx, item in enumerate(dataset):
        label = item['label']
        label_to_indices[label].append(idx)
    
    folds = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for label in [0, 1]:
        indices = np.array(label_to_indices[label])
        label_folds = []
        
        for train_idx, val_idx in kf.split(indices):
            train_indices = indices[train_idx]
            val_indices = indices[val_idx]
            label_folds.append((train_indices, val_indices))
        
        if not folds:
            folds = label_folds
        else:
            # Combine folds from both labels
            for i in range(n_folds):
                train_0, val_0 = folds[i]
                train_1, val_1 = label_folds[i]
                folds[i] = (np.concatenate([train_0, train_1]), np.concatenate([val_0, val_1]))
    
    # Create HF dataset splits instead of extracting data into lists
    fold_datasets = []
    for train_indices, val_indices in folds:
        # Use HF dataset.select() to create efficient subsets
        train_dataset = dataset.select(train_indices.tolist())
        val_dataset = dataset.select(val_indices.tolist())
        fold_datasets.append((train_dataset, val_dataset))
    
    return fold_datasets


def create_dataloaders(train_dataset, val_dataset, 
                      transform_train, transform_val, batch_size: int = 32):
    """Create train and validation dataloaders from HuggingFace datasets"""
    
    train_pytorch_dataset = WolverinesDataset(train_dataset, transform_train)
    val_pytorch_dataset = WolverinesDataset(val_dataset, transform_val)
    
    train_loader = DataLoader(train_pytorch_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_pytorch_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# Individual ID Feature Importance Workflow

This workflow implements feature importance visualization for wolverine individual identification using DINOv3 features and linear classification.

## Overview

The workflow consists of three main steps:
1. **Cross-validation** to find optimal training epochs
2. **Final training** on full dataset with test evaluation
3. **Feature visualization** showing spatial activation patterns

## Qualified Individuals

Based on `results/individual_analysis.json`, we use individuals with 32+ pelage visible (label==1) samples:

- **BDF10-M6**: 111 pelage visible samples
- **HLC20-H3**: 112 pelage visible samples  
- **PA23-F1**: 34 pelage visible samples
- **Tex**: 41 pelage visible samples
- **Turk**: 66 pelage visible samples

## Scripts

### 1. Cross-Validation (`03_best_epochs.py`)
- 5-fold stratified CV using only label==1 samples
- Each fold uses ~20% training, ~80% validation per individual
- Trains for 50 epochs, tracks F1 score
- Finds optimal epoch count across folds

### 2. Final Training (`04_train_final.py`)
- Uses optimal epochs from CV
- Trains on full label==1 training set
- Evaluates on label==1 test set only
- Saves linear layer weights for visualization

### 3. Feature Visualization (`05_visualize_important_features.py`)
- Loads trained model weights
- Extracts top 3 globally important features using `max(|W|, axis=0)`
- Creates 5×4 visualization grid:
  - Rows: Each wolverine individual
  - Columns: RGB image + 3 feature activation heatmaps

## Usage

### Submit All Jobs
```bash
cd /home/kdoherty/wolverines

# Submit CV jobs (array 0-4)
sbatch individual_id/sbatch/03_best_epochs.sbatch

# Submit final training (waits for CV completion)
sbatch individual_id/sbatch/04_train_final.sbatch

# Submit visualization (waits for training completion)
sbatch individual_id/sbatch/05_visualize.sbatch
```

### Manual Execution
```bash
# Cross-validation (run each fold)
python individual_id/scripts/03_best_epochs.py --fold 0
python individual_id/scripts/03_best_epochs.py --fold 1
# ... etc for folds 2-4

# Final training
python individual_id/scripts/04_train_final.py

# Visualization
python individual_id/scripts/05_visualize_important_features.py
```

## Outputs

### Results
- `individual_id/results/best_epoch_cv/fold_{0-4}.json` - CV results per fold
- `individual_id/results/test_performance.json` - Final test performance

### Models
- `individual_id/models/individual_id_final.pth` - Trained linear layer weights

### Figures  
- `individual_id/figures/feature_importance_grid.png` - Feature visualization

## Technical Details

### Feature Importance
- Linear classifier weights: `W` shape `[5 classes, 768 features]`
- Global importance: `importance = max(|W|, axis=0)` 
- Top 3 features selected by highest importance values
- Same features visualized for all individuals

### DINOv3 Architecture
- Model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Patch size: 16×16 pixels
- Input: 1280px height crop → 728×728 resize
- Output: 45×45 = 2025 patch tokens + CLS/register tokens
- Classification uses CLS token (position 0)

### Visualization
- Extracts patch features (skip CLS/register tokens)
- Creates spatial maps: 45×45 feature activations
- Upsamples to 728×728 using bilinear interpolation
- Normalizes using 5th-95th percentile clipping
- Shows which image regions activate discriminative features

## Dependencies

- PyTorch 2.7.1+
- Transformers (development version)
- HuggingFace datasets
- scikit-learn
- matplotlib
- PIL/Pillow
- numpy

## Resource Requirements

- **CV**: A6000 GPU, 32GB RAM, ~4 hours per fold
- **Final Training**: A6000 GPU, 32GB RAM, ~2 hours  
- **Visualization**: A6000 GPU, 16GB RAM, ~1 hour
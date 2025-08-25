# Pelage Sorting Workflow

A systematic hyperparameter optimization workflow for identifying high-quality wolverine pelage images using DINOv3 features and linear classification. **Optimized for preemptible cluster with 40 concurrent jobs.**

## Installation

### Option 1: Using the Installation Script (Recommended)
```bash
# Create and activate environment
mamba create -n wolverines python=3.10
mamba activate wolverines

# Clone repository
cd /home/kdoherty/wolverines/
git clone <repo-url> pelage_sorting

# Run installation script (handles PyTorch CUDA versions automatically)
cd pelage_sorting
bash ../install_dependencies.sh
```

### Option 2: Using Conda Environment File
```bash
# Create environment from file
mamba env create -f environment.yml
mamba activate wolverines

# Note: You may need to edit environment.yml to match your CUDA version
```

### Option 3: Manual Installation
```bash
mamba create -n wolverines python=3.10
mamba activate wolverines

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install transformers from source (required for DINOv3)
pip install git+https://github.com/huggingface/transformers.git
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Overview

This workflow consists of 4 main experiments designed to optimize wolverine image classification:

1. **Image Size Sweep** (`00_sweep_image_size`): Find optimal center crop size
2. **Augmentation Sweep** (`01_sweep_augmentations`): Optimize data augmentation parameters  
3. **Learning Rate Sweep** (`02_sweep_lr`): Find optimal learning rate with cross-validation
4. **Final Test** (`03_test_performance`): Train final model with optimal parameters

## Directory Structure (on cluster)

```
/home/kdoherty/wolverines/pelage_sorting/
├── scripts/           # Python training scripts (use relative paths)
├── sbatch/           # SLURM batch scripts (handle cd and environment)
├── utils/            # Utility modules including preemption handling
├── results/          # JSON results from experiments
├── figures/          # Generated visualizations
├── logs/             # SLURM job logs
├── checkpoints/      # Preemption checkpoints
└── README.md         # This file
```

## Usage

### Running Experiments

Each experiment uses SLURM array jobs on the **preempt partition** with up to 40 concurrent jobs:

```bash
# 1. Image size sweep (24 configurations: jobs 0-23 active, 24-39 idle)
sbatch sbatch/00_sweep_image_size.sbatch

# 2. Augmentation sweep (486 configurations distributed across all 40 jobs)  
sbatch sbatch/01_sweep_augmentations.sbatch

# 3. Learning rate sweep (35 configurations: jobs 0-34 active, 35-39 idle)
sbatch sbatch/02_sweep_lr.sbatch

# 4. Final test performance (single job: job 0 only)
sbatch sbatch/03_test_performance.sbatch
```

### Generating Figures

After experiments complete, generate visualizations:

```bash
cd /home/kdoherty/wolverines/pelage_sorting
python scripts/04_make_figures.py \
    --results_base_dir results \
    --output_dir figures
```

### Monitoring Jobs

Check job status:
```bash
squeue -u kdoherty
```

View logs:
```bash
cd /home/kdoherty/wolverines/pelage_sorting

# Check specific job log
tail -f logs/00_image_size/00_sweep_image_size_JOBID_0.out

# Check error logs  
tail -f logs/00_image_size/00_sweep_image_size_JOBID_0.err

# Monitor all jobs in an experiment
ls -la logs/01_augmentations/
```

### Rerunning Failed Jobs

To rerun specific configurations or overwrite existing results:

```bash
cd /home/kdoherty/wolverines/pelage_sorting

# Rerun specific job index
python scripts/00_sweep_image_size.py --idx 3 --overwrite

# Rerun with different parameters
python scripts/01_sweep_augmentations.py --idx 25 --device cpu --overwrite

# Check job-specific checkpoints
ls -la checkpoints/
```

## Experiment Details

### 00_sweep_image_size.py

**Purpose**: Find optimal center crop size before resize to 256×256

**Parameters**:
- Crop sizes: [256, 384, 512, 640, 768, 1024, 1152, 1280]
- Seeds: [0, 1, 2]
- Dataset: 10% stratified sample per class
- Fixed: lr=0.001, batch_size=32, epochs=10

**Job Distribution**: Jobs 0-23 each handle one (crop_size, seed) combination. Jobs 24-39 are idle.

### 01_sweep_augmentations.py  

**Purpose**: Optimize data augmentation parameters

**Parameters**:
- max_zoom: [1.0, 1.25, 1.5]
- h_flip_p: [0, 0.5]
- night_sim_p: [0, 0.25, 0.5] × 3 modes
- blur_p: [0, 0.25, 0.5] × 3 types  
- cutmix_p: [0, 0.25, 0.5]
- seeds: [0, 1, 2]

**Total**: 486 combinations distributed across 40 jobs (~12 per job)

### 02_sweep_lr.py

**Purpose**: Find optimal learning rate with 5-fold cross-validation

**Parameters**:
- Learning rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- Folds: 5-fold stratified CV
- Epochs: 50 (with epoch-by-epoch tracking)

**Job Distribution**: Jobs 0-34 each handle one (lr, fold) combination. Jobs 35-39 are idle.

### 03_test_performance.py

**Purpose**: Final evaluation with optimal hyperparameters

**Configuration**: Uses best settings from previous experiments
- Full train/test split
- Optimal hyperparameters
- Comprehensive metric tracking

## Results Format

Each experiment saves JSON results with:

```json
{
  "job_idx": 0,
  "params": {...},
  "seed": 0,
  "final_val_f1": 0.85,
  "final_val_precision": 0.83,
  "final_val_recall": 0.87,
  "final_val_accuracy": 0.84,
  "best_val_f1": 0.86,
  "training_time": 1234.5,
  "epochs_trained": 10
}
```

Learning rate sweep additionally includes:
- `train_history`: Epoch-by-epoch training metrics
- `val_history`: Epoch-by-epoch validation metrics

## Generated Figures

The visualization script creates:

1. **Image Size Results** (`00_image_size_results.png`): Bar chart with 95% CI
2. **Augmentation Results** (`01_augmentation_results.png`): Multi-panel bar charts
3. **Learning Rate Results** (`02_learning_rate_results.png`): Line plots with CI ribbons  
4. **Final Performance** (`03_final_performance.png`): Training curves and metrics

## Dependencies

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Datasets (HuggingFace) 
- scikit-learn
- matplotlib, seaborn
- numpy, pandas

Environment setup:
```bash
mamba activate wolverines  # Assumes environment already exists
```

## Key Features

- **Preemption Ready**: Auto-requeue jobs with checkpointing and resume capability
- **High Throughput**: Up to 40 concurrent GPU jobs on preemptible partition
- **Fault Tolerance**: Automatically skips completed configurations and resumes interrupted work
- **Reproducibility**: Fixed seeds and parameter tracking  
- **Portable**: No hardcoded paths in Python scripts
- **Monitoring**: Comprehensive logging and progress tracking
- **Flexibility**: Easy parameter modification and rerunning

## Preemption Handling

This workflow is optimized for preemptible clusters:

### Automatic Features
- **Auto-requeue**: Jobs automatically restart if preempted (`--requeue` flag)
- **Checkpoint saving**: Progress saved periodically and on preemption signals
- **Resume capability**: Jobs resume from last checkpoint on restart
- **Signal handling**: Graceful shutdown on SIGTERM with state preservation

### Manual Recovery
```bash
cd /home/kdoherty/wolverines/pelage_sorting

# Check checkpoint status
ls -la checkpoints/
cat checkpoints/01_augmentations_job_025.json

# Clear specific checkpoint to restart from beginning
rm checkpoints/01_augmentations_job_025.json

# Force rerun with overwrite
python scripts/01_sweep_augmentations.py --idx 25 --overwrite
```

### Monitoring Preemptions
```bash
# Check job queue and reasons
squeue -u kdoherty -o "%.10i %.10P %.20j %.8u %.2t %.10M %.6D %R"

# Check for preempted jobs (PD with ReqNodeNotAvail)
squeue -u kdoherty -t PD

# View completed/failed jobs
sacct -u kdoherty --starttime=today --format=JobID,JobName,State,ExitCode,DerivedExitCode
```

## Notes

- **Base Directory**: All scripts assume you cd to `/home/kdoherty/wolverines/pelage_sorting/`  
- **Log Location**: Job logs saved in `logs/` (local to project)
- **Results Storage**: Individual JSON files in `results/` for easy analysis
- **Checkpoints**: Automatic checkpoint files in `checkpoints/` for preemption recovery
- **GPU Detection**: Jobs automatically detect and use available GPUs
- **Overwrite Logic**: Use `--overwrite` flag to force recomputation of existing results
- **Partition Benefits**: Preemptible partition offers ~5x more concurrent jobs than general
- **Resource Efficiency**: 40 concurrent jobs maximize GPU utilization during peak times

## Job Array Sizing Summary

| Experiment | Total Configs | Active Jobs | Idle Jobs | Distribution |
|------------|---------------|-------------|-----------|--------------|
| 00_image_size | 24 | 0-23 | 24-39 | 1 config per job |
| 01_augmentations | 486 | 0-39 | none | ~12 configs per job |
| 02_learning_rate | 35 | 0-34 | 35-39 | 1 config per job |
| 03_test_performance | 1 | 0 | 1-39 | Single job |
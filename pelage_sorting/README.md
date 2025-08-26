# Pelage Sorting Workflow

A systematic hyperparameter optimization workflow for identifying high-quality wolverine pelage images using DINOv3 features and linear classification. **Requires GPU/CUDA support - optimized for compute clusters with preemptible partition and 24 concurrent jobs.**

## Installation

**Prerequisites**: GPU cluster with CUDA support (CUDA 11.8 or 12.x) and mamba/conda

### Option 1: Using the Installation Script (Recommended)
```bash
# Clone repository
cd /home/kdoherty/wolverines/
git clone <repo-url> pelage_sorting

# Run installation script (creates environment and installs all dependencies)
cd pelage_sorting
bash ../install_dependencies.sh

# Script automatically:
# - Creates mamba environment 'wolverines' with Python 3.12
# - Detects CUDA version and installs appropriate PyTorch
# - Installs all required dependencies
```

### Option 2: Using Conda Environment File
```bash
# Create environment from file
mamba env create -f environment.yml
mamba activate wolverines

# Note: Requires CUDA 12.1 by default - edit environment.yml for different CUDA versions
```

### Option 3: Manual Installation
```bash
mamba create -n wolverines python=3.12
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
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Installation should show CUDA: True and GPUs > 0 for successful setup
```

## Overview

This workflow consists of 3 main experiments designed to optimize wolverine image classification:

1. **Resize Size Sweep** (`00_sweep_resize`): Find optimal resize size (no center cropping)
2. **Learning Rate Sweep** (`01_sweep_lr`): Find optimal learning rate with cross-validation
3. **Final Test** (`02_test_performance`): Train final model with optimal parameters

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

Each experiment uses SLURM array jobs on the **preempt partition** with up to 24 concurrent jobs:

```bash
# 1. Resize size sweep (32 configurations distributed across 24 jobs)
sbatch sbatch/00_sweep_resize.sbatch

# 2. Learning rate sweep (35 configurations distributed across 24 jobs)
sbatch sbatch/01_sweep_lr.sbatch

# 3. Final test performance (single job: job 0 only, others exit gracefully)
sbatch sbatch/02_test_performance.sbatch
```

### Generating Figures

After experiments complete, generate visualizations:

```bash
cd /home/kdoherty/wolverines/pelage_sorting
python scripts/03_make_figures.py \
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
tail -f logs/00_resize/00_sweep_resize_JOBID_0.out

# Check error logs  
tail -f logs/00_resize/00_sweep_resize_JOBID_0.err

# Monitor all jobs in an experiment
ls -la logs/01_learning_rate/
```

### Rerunning Failed Jobs

To rerun specific configurations or overwrite existing results:

```bash
cd /home/kdoherty/wolverines/pelage_sorting

# Rerun specific job index
python scripts/00_sweep_resize.py --idx 3 --overwrite

# Rerun with different parameters
python scripts/01_sweep_lr.py --idx 25 --device cpu --overwrite

# Check job-specific checkpoints
ls -la checkpoints/
```

## Experiment Details

### 00_sweep_resize.py

**Purpose**: Find optimal resize size (no center cropping)

**Parameters**:
- Resize sizes: [256, 512, 768, 1024]
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7]
- Dataset: 10% stratified sample per class
- Fixed: lr=0.001, batch_size=16, epochs=10

**Job Distribution**: 32 configs distributed across 24 jobs (jobs 0-7 handle 2 configs each, jobs 8-23 handle 1 config each).

### 01_sweep_lr.py

**Purpose**: Find optimal learning rate with 5-fold cross-validation

**Parameters**:
- Learning rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- Folds: 5-fold stratified CV
- Epochs: 50 (with epoch-by-epoch tracking)

**Job Distribution**: 35 configs distributed across 24 jobs (jobs 0-10 handle 2 configs each, jobs 11-23 handle 1 config each).

### 02_test_performance.py

**Purpose**: Final evaluation with optimal hyperparameters

**Configuration**: Uses best settings from previous experiments
- Full train/test split
- Best resize size from script 00
- Optimal learning rate from script 01
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

1. **Resize Size Results** (`00_resize_results.png`): Bar chart with 95% CI
2. **Learning Rate Results** (`01_learning_rate_results.png`): Line plots with CI ribbons  
3. **Final Performance** (`02_final_performance.png`): Training curves and metrics

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
- **High Throughput**: Up to 24 concurrent GPU jobs on preemptible partition
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
cat checkpoints/01_learning_rate_job_025.json

# Clear specific checkpoint to restart from beginning
rm checkpoints/01_learning_rate_job_025.json

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
- **Resource Efficiency**: 24 concurrent jobs maximize GPU utilization during peak times

## Job Array Sizing Summary

| Experiment | Total Configs | Array Jobs | Distribution |
|------------|---------------|------------|--------------|
| 00_resize | 32 | 0-23 | Jobs 0-7: 2 configs, Jobs 8-23: 1 config |
| 01_learning_rate | 35 | 0-23 | Jobs 0-10: 2 configs, Jobs 11-23: 1 config |
| 02_test_performance | 1 | 0-23 | Job 0: 1 config, Jobs 1-23: exit gracefully |
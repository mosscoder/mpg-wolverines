#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-39
#SBATCH --requeue
#SBATCH --output=logs/01_augmentations/%x_%A_%a.out
#SBATCH --error=logs/01_augmentations/%x_%A_%a.err
#SBATCH --job-name=01_sweep_augmentations

# Change to project base directory
cd /home/kdoherty/wolverines/pelage_sorting

# Create log directory if it doesn't exist
mkdir -p logs/01_augmentations

echo "Starting job $SLURM_ARRAY_TASK_ID on node $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/.bashrc
mamba activate wolverines

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Run script with array index (using relative paths)
python scripts/01_sweep_augmentations.py \
    --idx ${SLURM_ARRAY_TASK_ID} \
    --output_dir results/01_augmentations \
    --device gpu

echo "Job completed at $(date)"
#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1 
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/home/kdoherty/logs/wolverines/pelage/00_best_epoch/%x_%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/pelage/00_best_epoch/%x_%A_%a.err
#SBATCH --job-name=00_find_best_epoch

# Change to project base directory
cd /home/kdoherty/wolverines/pelage_sorting

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/pelage/00_best_epoch

echo "Starting job $SLURM_ARRAY_TASK_ID on node $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/.bashrc
mamba activate wolverines

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Run script with array index (using relative paths)
python scripts/00_find_best_epoch.py \
    --idx ${SLURM_ARRAY_TASK_ID} \
    --output_dir results/00_best_epoch \
    --device gpu

echo "Job completed at $(date)"
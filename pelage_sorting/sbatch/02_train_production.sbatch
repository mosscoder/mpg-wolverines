#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/home/kdoherty/logs/wolverines/pelage/02_production/%x_%A.out
#SBATCH --error=/home/kdoherty/logs/wolverines/pelage/02_production/%x_%A.err
#SBATCH --job-name=02_train_production

# Change to project base directory
cd /home/kdoherty/wolverines/pelage_sorting

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/pelage/02_production

echo "Starting production model training on node $(hostname) at $(date)"
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

# Run production training script (single job, no array)
python -u scripts/02_train_production.py \
    --idx 0 \
    --output_dir results/02_production \
    --device gpu

echo "Production model training completed at $(date)"
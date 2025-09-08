#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/home/kdoherty/logs/wolverines/pelage/01_test/%x_%A.out
#SBATCH --error=/home/kdoherty/logs/wolverines/pelage/01_test/%x_%A.err
#SBATCH --job-name=01_test_performance

# Change to project base directory
cd /home/kdoherty/wolverines/pelage_sorting

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/pelage/01_test

echo "Starting final test evaluation on node $(hostname) at $(date)"
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

# Run script (using relative paths)
python scripts/01_test_performance.py \
    --idx 0 \
    --output_dir results/01_test \
    --device gpu

echo "Final test evaluation completed at $(date)"
#!/bin/bash
#SBATCH --job-name=test_best
#SBATCH --partition=general
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/test_best/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/test_best/%A_%a.err

# Final test evaluation with best configs per backbone (dinov3, megadescriptor, bioclip2)
# 3 jobs, one per backbone

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/test_best

# Navigate to project directory
cd /home/kdoherty/wolverines

# Activate conda environment
source ~/.bashrc
mamba activate wolverines

# Set HuggingFace cache to cluster's shared cache (private dataset)
export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Run the final test evaluation
python -u test_best_configs/scripts/sweep_best.py \
    --idx $SLURM_ARRAY_TASK_ID \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

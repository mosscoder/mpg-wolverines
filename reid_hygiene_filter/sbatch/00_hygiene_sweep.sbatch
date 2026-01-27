#!/bin/bash
#SBATCH --job-name=hygiene_sweep
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/hygiene/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/hygiene/%A_%a.err

# Gallery Hygiene Sweep Experiment
# 6 thresholds × 6 gallery sizes × 8 seeds = 288 configs across 24 jobs (12/job)

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/hygiene

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

# Run the hygiene sweep script
python -u reid_hygiene_filter/scripts/00_hygiene_sweep.py \
    --idx $SLURM_ARRAY_TASK_ID \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

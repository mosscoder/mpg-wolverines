#!/bin/bash
#SBATCH --job-name=debug_test_eval
#SBATCH --partition=general
#SBATCH --array=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/debug/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/debug/%A_%a.err

# Debug test eval: unfiltered DINOv3, all training samples, open-set BA
# Single config — only job 0 does work, jobs 1-23 exit gracefully

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/DINOv3/debug

# Navigate to project directory
cd /home/kdoherty/wolverines

# Activate conda environment
source ~/.bashrc
mamba activate wolverines

# Set HuggingFace cache to cluster's shared cache (private dataset)
export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1

# Use node-local scratch for temp files (avoids /tmp full errors)
export TMPDIR=/scratch/kdoherty
mkdir -p $TMPDIR

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

python -u reid_openset/dinov3/debug/debug_test_eval.py \
    --device gpu --epochs 50 --seed 0

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

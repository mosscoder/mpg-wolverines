#!/bin/bash
#SBATCH --job-name=tune_epoch_count
#SBATCH --partition=general
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/debug/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/debug/%A_%a.err

# Tune epoch count across 3 gallery conditions (3 nodes):
#   idx=0: Unfiltered (all train minus val)
#   idx=1: Best R@1 filtered (threshold + gallery_size from hygiene)
#   idx=2: Best BA filtered (threshold + gallery_size from hygiene)

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

python -u reid_openset/dinov3/debug/tune_test_epoch_count.py \
    --idx $SLURM_ARRAY_TASK_ID \
    --device gpu --epochs 200 --eval-every 5 --seed 0

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

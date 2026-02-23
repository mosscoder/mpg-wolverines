#!/bin/bash
#SBATCH --job-name=opt_augmentation_sweep
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/augmentation/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/augmentation/%A_%a.err

# Augmentation Sweep for DINOv3 Re-ID with Temporal (ymdh) Split
# 4 conditions × 5 seeds = 20 configs, one per job
#
# 1 shared baseline + 3 individual augmentations:
#   idx  0-4:  none    (baseline)
#   idx  5-9:  blur
#   idx 10-14: jitter
#   idx 15-19: ir_sim
#
# Fixed: Best LR, image size, embedding dim from prior sweeps, 20 epochs

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/DINOv3/augmentation

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

# Run the augmentation sweep
python -u -m utils.reid opt_augmentation --model dinov3 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/dinov3/results/opt/augmentation \
    --device gpu \
    --epochs 20 --seeds 0 1 2 3 4

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

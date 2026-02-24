#!/bin/bash
#SBATCH --job-name=tune_step_count
#SBATCH --partition=general
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/step_count/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/step_count/%A_%a.err

# Sweep gallery threshold × sample size with fixed step budget (6 jobs):
#   idx=0: g>=0.00, n=64     idx=1: g>=0.00, n=128
#   idx=2: g>=0.25, n=64     idx=3: g>=0.25, n=128
#   idx=4: g>=0.50, n=64     idx=5: g>=0.50, n=128

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/DINOv3/step_count

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

python -u -m utils.reid tune_step_count --model dinov3 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/dinov3/results/step_count \
    --device gpu --seed 0

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

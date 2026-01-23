#!/bin/bash
#SBATCH --job-name=openset_arcface_hygiene
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/openset_arcface/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/openset_arcface/%A_%a.err

# Open-Set DINOv3 + ArcFace Gallery Hygiene Sweep Experiment
# 6 thresholds x 6 gallery sizes x 8 seeds = 288 configs across 24 jobs (12/job)
#
# Extends reid_dinov3_arcface with open-set evaluation:
# - Calibrates distance threshold on validation set (known individuals)
# - Evaluates Correct Flag Rate on rare/unknown individuals
# - Reports CFR at each quality level (q>=0.0, q>=0.1, etc.)

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/openset_arcface

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

# Run the open-set hygiene sweep script
python reid_openset_arcface/scripts/00_hygiene_sweep.py \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset_arcface/results \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

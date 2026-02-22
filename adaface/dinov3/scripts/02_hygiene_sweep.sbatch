#!/bin/bash
#SBATCH --job-name=adaface_hygiene
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/adaface/DINOv3/hygiene/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/adaface/DINOv3/hygiene/%A_%a.err

# Open-Set DINOv3 + QualityAdaFace + Raw Cosine Gallery Hygiene Sweep
# 2 loss modes x 6 gallery sizes x 8 seeds = 96 configs across 24 jobs (4/job)
#
# Loss modes: quality_aware (margin scaled by pelage_score), quality_ignorant (fixed margin)
# Gallery: always threshold=0.0 (no quality filtering), quality handled by loss
# Evaluates Balanced Accuracy = (Known Accept + Unknown Reject) / 2

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/adaface/DINOv3/hygiene

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

# Run the AdaFace hygiene sweep script
python -u -m utils.adaface_reid hygiene --model dinov3 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir adaface/dinov3/results/hygiene \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

#!/bin/bash
#SBATCH --job-name=adaface_opt_lr
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/adaface/DINOv3/lr/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/adaface/DINOv3/lr/%A_%a.err

# AdaFace Learning Rate Sweep for DINOv3 Re-ID with Temporal (ymdh) Split
# 4 LRs x 5 seeds = 20 configs, one per job
#
# LRs: 0.0001, 0.0005, 0.001, 0.005
# Fixed: 224px images, seeds=0,1,2,3,4, 20 epochs
# Loss: QualityAdaFace (quality-adaptive margin)

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/adaface/DINOv3/lr

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

# Run the AdaFace LR sweep script
python -u -m utils.adaface_reid opt_lr --model dinov3 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir adaface/dinov3/results/opt/lr \
    --device gpu \
    --values 0.0001 0.0005 0.001 0.005 \
    --image-size 224 \
    --embedding-dim 256 --epochs 20 --seeds 0 1 2 3 4

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

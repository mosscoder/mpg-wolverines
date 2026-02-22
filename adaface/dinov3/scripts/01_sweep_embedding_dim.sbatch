#!/bin/bash
#SBATCH --job-name=adaface_opt_emb_dim
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/adaface/DINOv3/embedding_dim/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/adaface/DINOv3/embedding_dim/%A_%a.err

# AdaFace Embedding Dimension Sweep for DINOv3 Re-ID with Temporal (ymdh) Split
# 2 dims x 5 seeds = 10 configs, one per job
#
# Dims: 256, 512
# Fixed: Best LR from prior sweep, native image size, seeds=0,1,2,3,4, 20 epochs
# Loss: QualityAdaFace (quality-adaptive margin)

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/adaface/DINOv3/embedding_dim

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

# Run the AdaFace embedding dim sweep script
# NOTE: Update --lr and --image-size from sweep 00 results
python -u -m utils.adaface_reid opt_embedding_dim --model dinov3 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir adaface/dinov3/results/opt/embedding_dim \
    --device gpu \
    --values 256 512 \
    --lr 0.0005 --image-size 224 \
    --epochs 20 --seeds 0 1 2 3 4

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

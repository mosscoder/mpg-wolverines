#!/bin/bash
#SBATCH --job-name=opt_lr_sweep
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/lr/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/lr/%A_%a.err

# Learning Rate Sweep for DINOv3 Re-ID with Temporal (ymdh) Split
# 10 learning rates = 10 configs, one per job
#
# LRs: 0.0001 to 0.001, step 0.0001
# Fixed: 224px images, seed=0, 50 epochs
#
# Temporal split by capture event (ymdh):
# - Earlier 50% of ymdh values -> train
# - Later 50% of ymdh values -> test
# - Hygiene validation indices excluded

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/DINOv3/lr

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

# Run the LR sweep script
python -u -m utils.reid opt_lr --model dinov3 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/dinov3/results/opt/lr \
    --device gpu \
    --values 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001 \
    --image-size 224 \
    --embedding-dim 128 --epochs 50 --seed 0

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

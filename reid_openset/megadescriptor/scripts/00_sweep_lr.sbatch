#!/bin/bash
#SBATCH --job-name=opt_lr_sweep_MD
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/lr/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/lr/%A_%a.err

# Learning Rate Sweep for MegaDescriptor Re-ID with Temporal (ymdh) Split
# 4 LRs x 5 seeds = 20 configs, one per job
#
# LRs: 0.0001, 0.0005, 0.001, 0.005
# Fixed: 384px images, seeds=0,1,2,3,4, 50 epochs
#
# Temporal split by capture event (ymdh):
# - Earlier 50% of ymdh values -> train
# - Later 50% of ymdh values -> test
# - Hygiene validation indices excluded

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/lr

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

# Run the LR sweep script
python -u -m utils.reid opt_lr --model megadescriptor \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/megadescriptor/results/opt/lr \
    --device gpu \
    --values 0.0001 0.0005 0.001 0.005 \
    --image-size 384 \
    --embedding-dim 256 --epochs 50 --seeds 0 1 2 3 4

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

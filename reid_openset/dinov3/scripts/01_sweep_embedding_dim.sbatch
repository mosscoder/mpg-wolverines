#!/bin/bash
#SBATCH --job-name=opt_emb_dim_sweep
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-34
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/embedding_dim/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/embedding_dim/%A_%a.err

# Embedding Dimension Sweep for DINOv3 Re-ID with Temporal (ymdh) Split
# 7 dims x 5 seeds = 35 configs, one per job
#
# Dims: 64, 128, 256, 512, 1024, 2048, 4096
# Fixed: Best LR from prior sweep, native image size, seeds=0,1,2,3,4, 50 epochs

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/DINOv3/embedding_dim

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

# Run the embedding dim sweep script
# NOTE: Update --lr and --image-size from sweep 00/01 results
python -u -m utils.reid opt_embedding_dim --model dinov3 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/dinov3/results/opt/embedding_dim \
    --device gpu \
    --values 64 128 256 512 1024 2048 4096 \
    --lr 0.0005 --image-size 224 \
    --epochs 50 --seeds 0 1 2 3 4

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

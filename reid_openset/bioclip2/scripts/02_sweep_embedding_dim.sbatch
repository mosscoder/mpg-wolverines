#!/bin/bash
#SBATCH --job-name=opt_emb_dim_sweep_bioclip2
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/opt_emb_dim_bioclip2/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/opt_emb_dim_bioclip2/%A_%a.err

# Embedding Dimension Sweep for BioCLIP-2 Re-ID with Temporal (ymdh) Split
# 4 embedding dims = 4 configs, one per job
#
# Dims: 64, 128, 256, 512
# Fixed: Best LR and size from prior sweeps, seed=0, 50 epochs

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/opt_emb_dim_bioclip2

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
python -u utils/reid.py opt_embedding_dim --model bioclip2 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/bioclip2/results/opt/embedding_dim \
    --device gpu \
    --values 64 128 256 512 \
    --lr 0.0005 --image-size 224 \
    --epochs 50 --seed 0

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

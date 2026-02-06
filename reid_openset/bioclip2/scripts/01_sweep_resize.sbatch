#!/bin/bash
#SBATCH --job-name=opt_resize_sweep_bioclip2
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-18
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/opt_resize_bioclip2/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/opt_resize_bioclip2/%A_%a.err

# Image Size Sweep for BioCLIP-2 Re-ID with Temporal (ymdh) Split
# 19 image sizes = 19 configs, one per job
#
# Sizes: 126 to 378, step 14 (multiples of ViT-L/14 patch size)
# Fixed: LR=5e-4, seed=0, 50 epochs
#
# Temporal split by capture event (ymdh):
# - Earlier 50% of ymdh values -> train
# - Later 50% of ymdh values -> test
# - Hygiene validation indices excluded

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/opt_resize_bioclip2

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

# Run the resize sweep script
python -u -m utils.reid opt_resize --model bioclip2 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/bioclip2/results/opt/resize \
    --device gpu \
    --values 126 140 154 168 182 196 210 224 238 252 266 280 294 308 322 336 350 364 378 \
    --lr 0.0005 \
    --embedding-dim 128 --epochs 50 --seed 0

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

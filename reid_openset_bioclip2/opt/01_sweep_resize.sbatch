#!/bin/bash
#SBATCH --job-name=opt_resize_sweep_bioclip2
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-16
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/opt_resize_bioclip2/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/opt_resize_bioclip2/%A_%a.err

# Image Size Sweep for BioCLIP-2 Re-ID with Temporal (ymdh) Split
# 17 image sizes = 17 configs, one per job
#
# Sizes: 128 to 384, step 16 (128, 144, 160, ..., 384)
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
python -u reid_openset_bioclip2/opt/01_sweep_resize.py \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset_bioclip2/opt/resize_results \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

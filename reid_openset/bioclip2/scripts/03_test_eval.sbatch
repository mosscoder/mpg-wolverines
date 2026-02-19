#!/bin/bash
#SBATCH --job-name=test_eval_bioclip2
#SBATCH --partition=general
#SBATCH --array=0-7
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/BioCLIP2/test_eval/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/BioCLIP2/test_eval/%A_%a.err

# Final Test Evaluation - BioCLIP-2 + ArcFace
# 8 seeds (one per job), best operating point from hygiene sweep
#
# - Gallery: train split, quality-filtered, sampled per seed
# - Query: full HF test split (fixed across seeds)
# - Unknown pool: test-split unknowns only
# - Evaluates R@1 and Balanced Accuracy on held-out test data

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/BioCLIP2/test_eval

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

# Run final test evaluation
python -u -m utils.reid test_eval --model bioclip2 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/bioclip2/results/test \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

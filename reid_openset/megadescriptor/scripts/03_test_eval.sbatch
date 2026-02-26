#!/bin/bash
#SBATCH --job-name=test_eval_MD
#SBATCH --partition=general
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16,babel-x9-24
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/test_eval/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/test_eval/%A_%a.err

# Final Test Evaluation - MegaDescriptor + ArcFace
# 4 tasks (one per job): closed_filtered, closed_matched, open_filtered, open_matched
# seed=0, full filtered gallery (no cap), event-matched subsampling for matched tasks
# Fresh ArcFace center p5 thresholds computed from the trained model

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/test_eval

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
python -u -m utils.reid test_eval --model megadescriptor \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/megadescriptor/results/test \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

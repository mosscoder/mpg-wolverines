#!/bin/bash
#SBATCH --job-name=openset_bioclip2
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/BioCLIP2/hygiene/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/BioCLIP2/hygiene/%A_%a.err

# Open-Set BioCLIP-2 + ArcFace + Raw Cosine Gallery Hygiene Sweep Experiment
# 6 thresholds x 6 gallery sizes x 8 seeds = 288 configs across 24 jobs (12/job)
#
# Uses raw cosine similarity instead of T-Norm:
# - L2-normalized embeddings for cosine similarity scores in [0, 1]
# - Calibrates threshold on raw cosine scores using LOO within gallery
# - Evaluates Balanced Accuracy = (Known Accept + Unknown Reject) / 2
# - Macro-averaged per-individual metrics for fair comparison across individuals

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/BioCLIP2/hygiene

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

# Run the open-set raw cosine hygiene sweep script
python -u -m utils.reid hygiene --model bioclip2 \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/bioclip2/results \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

#!/bin/bash
#SBATCH --job-name=openset_ndr_hygiene
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/openset_ndr/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/openset_ndr/%A_%a.err

# Open-Set DINOv3 + ArcFace + T-Norm + NDR Gallery Hygiene Sweep Experiment
# 6 thresholds x 6 gallery sizes x 8 seeds = 288 configs across 24 jobs (12/job)
#
# Extends reid_openset_tnorm with NDR (Novelty Detection Rate) metrics:
# - Computes dynamic thresholds from known score percentiles (1st, 5th, 10th)
# - Measures NDR at each FNR (False Novelty Rate) level
# - NDR = % of unknown wolverines correctly flagged as "Novel"
# - FNR = % of known wolverines incorrectly flagged as "Novel"

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/openset_ndr

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

# Run the open-set NDR hygiene sweep script
python -u reid_openset_NDR/scripts/00_hygiene_sweep.py \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset_NDR/results \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

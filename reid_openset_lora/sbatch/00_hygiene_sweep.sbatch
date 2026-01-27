#!/bin/bash
#SBATCH --job-name=openset_lora_hygiene
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/openset_lora/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/openset_lora/%A_%a.err

# Open-Set DINOv3 + ArcFace + LoRA Gallery Hygiene Sweep Experiment
# 6 thresholds x 6 gallery sizes x 8 seeds = 288 configs across 24 jobs (12/job)
#
# Key differences from reid_openset_arcface:
# - Backbone: LoRA adapters (r=16, alpha=32) instead of frozen
# - Target modules: q/k/v_proj (attention) + up/down_proj (MLP)
# - Trainable params: ~2.5M (LoRA + head) vs ~100K (head only)
# - Optimizer: AdamW with weight_decay=0.01

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/openset_lora

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
echo "Experiment: DINOv3 + ArcFace + LoRA (r=16, alpha=32)"
echo "========================================"

# Run the open-set hygiene sweep script with LoRA
python -u reid_openset_lora/scripts/00_hygiene_sweep.py \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset_lora/results \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

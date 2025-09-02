#!/bin/bash
#SBATCH --job-name=id_cv
#SBATCH --array=0-4
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/individual_id/cv_%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/individual_id/cv_%A_%a.err
#SBATCH --requeue

# Load modules
module load cuda/12.1
module load gcc/11.2.0

# Activate environment
source ~/.bashrc
mamba activate wolverines

# Change to project directory
cd /home/kdoherty/wolverines

# Create logs directory
mkdir -p /home/kdoherty/logs/wolverines/individual_id

# Create output directory
mkdir -p individual_id/results/best_epoch_cv

# Set Python to unbuffered mode for immediate output in logs
export PYTHONUNBUFFERED=1

# Run cross-validation fold
echo "Starting Individual ID Cross-Validation - Fold ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

python individual_id/scripts/03_best_epochs.py \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --output_dir individual_id/results/best_epoch_cv \
    --device gpu

echo "Cross-validation fold ${SLURM_ARRAY_TASK_ID} completed at $(date)"
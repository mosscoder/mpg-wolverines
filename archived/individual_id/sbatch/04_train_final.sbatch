#!/bin/bash
#SBATCH --job-name=id_final
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/individual_id/final_%j.out
#SBATCH --error=/home/kdoherty/logs/wolverines/individual_id/final_%j.err
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

# Create output directories
mkdir -p individual_id/models
mkdir -p individual_id/results

# Wait for CV jobs to complete (check for all fold results)
echo "Checking for CV results..."
while [ ! -f "individual_id/results/best_epoch_cv/fold_0.json" ] || \
      [ ! -f "individual_id/results/best_epoch_cv/fold_1.json" ] || \
      [ ! -f "individual_id/results/best_epoch_cv/fold_2.json" ] || \
      [ ! -f "individual_id/results/best_epoch_cv/fold_3.json" ] || \
      [ ! -f "individual_id/results/best_epoch_cv/fold_4.json" ]; do
    echo "Waiting for CV results to complete..."
    sleep 60
done

echo "All CV results found, starting final training..."

# Set Python to unbuffered mode for immediate output in logs
export PYTHONUNBUFFERED=1

# Run final training
echo "Starting Individual ID Final Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

python individual_id/scripts/04_train_final.py \
    --cv_results_dir individual_id/results/best_epoch_cv \
    --models_dir individual_id/models \
    --output_file individual_id/results/test_performance.json \
    --device gpu \
    --seed 42

echo "Final training completed at $(date)"
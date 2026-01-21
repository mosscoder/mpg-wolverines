#!/bin/bash
#SBATCH --job-name=id_viz
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/individual_id/viz_%j.out
#SBATCH --error=/home/kdoherty/logs/wolverines/individual_id/viz_%j.err
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
mkdir -p individual_id/figures

# Wait for final training to complete
echo "Checking for trained model..."
while [ ! -f "individual_id/models/individual_id_final.pth" ] || \
      [ ! -f "individual_id/results/test_performance.json" ]; do
    echo "Waiting for final training to complete..."
    sleep 60
done

echo "Trained model found, starting visualization..."

# Set Python to unbuffered mode for immediate output in logs
export PYTHONUNBUFFERED=1

# Run visualization
echo "Starting Individual ID Feature Visualization"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

python individual_id/scripts/05_visualize_important_features.py \
    --model_path individual_id/models/individual_id_final.pth \
    --output_path individual_id/figures/feature_importance_grid.png \
    --device gpu \
    --seed 42

echo "Feature visualization completed at $(date)"
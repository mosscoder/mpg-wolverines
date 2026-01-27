#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 
#SBATCH --time=24:00:00
#SBATCH --array=0-23%24
#SBATCH --requeue
#SBATCH --output=/home/kdoherty/logs/wolverines/megadescriptor/%x_%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/megadescriptor/%x_%A_%a.err
#SBATCH --job-name=megadescriptor_sweep

# Change to wolverines root directory
cd /home/kdoherty/wolverines

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/megadescriptor

echo "Starting job $SLURM_ARRAY_TASK_ID on node $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/.bashrc
mamba activate wolverines

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Set Python to unbuffered mode for immediate output in logs
export PYTHONUNBUFFERED=1

# Run script with array index
python -u megadescriptor/scripts/01_megadescriptor_sweep.py \
    --idx ${SLURM_ARRAY_TASK_ID} \
    --output_dir megadescriptor/results \
    --device gpu

echo "Job completed at $(date)"
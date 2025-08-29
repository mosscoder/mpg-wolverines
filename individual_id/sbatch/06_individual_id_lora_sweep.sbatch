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
#SBATCH --output=/home/kdoherty/logs/wolverines/individual_id/%x_%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/individual_id/%x_%A_%a.err
#SBATCH --job-name=individual_id_lora_sweep

# Change to individual_id directory
cd /home/kdoherty/wolverines/individual_id

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/individual_id

echo "Starting LoRA job $SLURM_ARRAY_TASK_ID on node $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/.bashrc
mamba activate wolverines

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check PEFT availability
python -c "try:
    from peft import LoraConfig
    print('PEFT available: True')
except ImportError:
    print('PEFT available: False')
    print('Install with: pip install peft')
"

# Run LoRA sweep script with array index
python scripts/06_individual_id_lora_sweep.py \
    --idx ${SLURM_ARRAY_TASK_ID} \
    --output_dir results \
    --device gpu

echo "LoRA job completed at $(date)"
#!/bin/bash
#SBATCH --job-name=hygiene_opt
#SBATCH --partition=preempt
#SBATCH --array=0-23
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/kdoherty/logs/wolverines/hygiene/opt_%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/hygiene/opt_%A_%a.err

echo "=========================================="
echo "Optimized Hygiene Sweep - Job ${SLURM_ARRAY_TASK_ID}"
echo "Started: $(date)"
echo "=========================================="

cd /home/kdoherty/wolverines

# Activate environment
source activate wolverines

# Set HuggingFace cache (for dataset, not metadata)
export HF_HOME=/data/hf_cache
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1

# Run optimized version
# No persistent cache needed - builds in-memory cache per job (0.032s)
python optimize_hf_loading/hygiene_sweep_optimized.py \
    --idx ${SLURM_ARRAY_TASK_ID} \
    --output_dir reid_openset_tnorm/results_optimized \
    --device gpu

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

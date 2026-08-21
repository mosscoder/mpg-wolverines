#!/bin/bash
#SBATCH --job-name=tune_step_count_MD
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --array=0-14
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/step_count/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/step_count/%A_%a.err

# Sweep gallery threshold × seed with CoverageSampler (15 jobs = 3 gallery_q × 5 seeds):
#   idx= 0- 4: g>=0.00, seeds 0-4
#   idx= 5- 9: g>=0.25, seeds 0-4
#   idx=10-14: g>=0.50, seeds 0-4

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/step_count

# Navigate to project directory
cd /home/kdoherty/wolverines

# Activate conda environment
source ~/.bashrc
mamba activate wolverines

# Set HuggingFace cache to cluster's shared cache (private dataset)
export HF_HOME=/data/user_data/kdoherty/hf_cache
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

python -u -m utils.reid tune_step_count --model megadescriptor \
    --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/megadescriptor/results/step_count \
    --device gpu

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

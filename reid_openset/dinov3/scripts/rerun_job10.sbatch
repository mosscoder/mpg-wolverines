#!/bin/bash
#SBATCH --job-name=rerun_hygiene_10
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/hygiene/rerun_10_%j.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/DINOv3/hygiene/rerun_10_%j.err

# Rerun of hygiene sweep job 10 (failed on RTX PRO 6000 / Blackwell - incompatible CUDA kernels)
# Excludes babel-l5 nodes (RTX PRO 6000)

mkdir -p /home/kdoherty/logs/wolverines/reid_openset/DINOv3/hygiene

cd /home/kdoherty/wolverines

source ~/.bashrc
mamba activate wolverines

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

python -u -m utils.reid hygiene --model dinov3 \
    --idx 10 \
    --output_dir reid_openset/dinov3/results \
    --device gpu

echo "========================================"
echo "End time: $(date)"
echo "Job completed"
echo "========================================"

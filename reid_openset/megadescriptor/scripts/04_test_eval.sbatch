#!/bin/bash
#SBATCH --job-name=test_step_eval_MD
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --array=0-14
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Every non-Blackwell GPU type on preempt: the RTX_PRO_6000 nodes are
# sm_120, unsupported by this env's PyTorch 2.7.1 build (max sm_90),
# and arrived post-maintenance with empty caches.
#SBATCH --constraint=6000Ada|A100_40GB|A100_80G|A100_80GB|A6000|H100|H200|L40|L40S
#SBATCH --time=48:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/test_eval/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/test_eval/%A_%a.err

# Final test evaluation from step-count sweep, five training replicates
# (15 jobs = 3 gallery_q x 5 RNG seeds):
#   task % 3 -> gallery_q (0: g>=0.00, 1: g>=0.25, 2: g>=0.50)
#   task / 3 -> training seed (0..4)
# Each reads 5 seed sweep results, finds best step per (metric, query_q).
# Outputs land in results/test_eval_seeds/ and results/embeddings/ with
# _seed={s} suffixes; results/test_eval/ (manuscript numbers) is untouched.

# Create log directory if it doesn't exist
mkdir -p /home/kdoherty/logs/wolverines/reid_openset/MegaDescriptor/test_eval

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

GAL_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=$((SLURM_ARRAY_TASK_ID / 3))

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID (gallery idx $GAL_IDX, seed $SEED)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "========================================"

python -u -m utils.reid test_step_eval --model megadescriptor \
    --idx $GAL_IDX \
    --output_dir reid_openset/megadescriptor/results/test_eval_seeds \
    --device gpu --seed $SEED --overwrite --export-embeddings
rc=$?

# Print completion info
echo "========================================"
echo "End time: $(date)"
echo "Python exit code: $rc"
echo "========================================"
exit $rc

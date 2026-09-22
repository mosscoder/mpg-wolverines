#!/bin/bash
#SBATCH --job-name=populate_hf_cache
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/kdoherty/logs/wolverines/populate_hf_cache_%j.out
#SBATCH --error=/home/kdoherty/logs/wolverines/populate_hf_cache_%j.err

# One-time populate of the shared HF cache at /data/user_data/kdoherty/hf_cache.
# The old cache lived at /data/hf_cache, which is node-local NVMe: it existed
# only on nodes earlier jobs had run on and is empty on the nodes the
# post-maintenance partitions schedule us to. The shared user_data automount is
# visible from every compute node, so downloading once here serves all
# subsequent offline jobs. Run downstream evals with
# --dependency=afterok:<this job id>.

cd /home/kdoherty/wolverines
source ~/.bashrc
mamba activate wolverines

export HF_HOME=/data/user_data/kdoherty/hf_cache
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
mkdir -p "$HF_HOME"

python -u - <<'EOF'
from datasets import load_dataset

for config in ["reidentification", "pelage"]:
    for split in ["train", "test"]:
        ds = load_dataset("mpg-ranch/wolverines", config, split=split)
        print(f"cached {config}/{split}: {len(ds)} rows", flush=True)
EOF

echo "populate complete"

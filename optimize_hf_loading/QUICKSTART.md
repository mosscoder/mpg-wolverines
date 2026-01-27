# Quick Start Guide

## Current Status (2026-01-27)

**Iterations Complete:** 0-2 of 10
**Current Speedup:** 2x faster (49.5% reduction)
**Best Version:** Iteration 2 (vectorized operations)

## Run Optimized Benchmark

### Local Testing (Single Config)
```bash
cd /Users/kdoherty/wolverines

# Run the optimized version
python optimize_hf_loading/02_benchmark_vectorized.py --local

# Expected output:
# Total: ~102s (was 203s baseline)
# Per-config operations: ~0.01s (was ~101s baseline)
```

### Compare Results
```bash
# Compare to baseline
python optimize_hf_loading/utils/compare_timings.py \
  optimize_hf_loading/results/baseline_timings.json \
  optimize_hf_loading/results/iteration_02_timings.json
```

### View Results
```bash
# Baseline
cat optimize_hf_loading/results/baseline_timings.json

# Iteration 2 (current best)
cat optimize_hf_loading/results/iteration_02_timings.json
```

---

## What Changed?

### Baseline (Slow)
```python
# Sequential dataset access (slow!)
for idx in indices:
    if dataset[idx]['pelage_score'] >= threshold:
        filtered.append(idx)
```
**Time:** 94-97s per config

### Iteration 2 (Fast)
```python
# Cached metadata + vectorized numpy operations
quality_arr = quality_cache[indices]  # Fast array lookup
mask = quality_arr >= threshold       # Vectorized comparison
return indices_arr[mask].tolist()     # Boolean indexing
```
**Time:** 0.01s per config (9,700x faster!)

---

## Key Files

### Use These (Optimized)
- `utils/metadata_cache.py` - Metadata caching utilities
- `utils/optimized_filters.py` - Vectorized filtering functions
- `02_benchmark_vectorized.py` - Current best benchmark

### Reference (Baseline)
- `00_benchmark_baseline.py` - Original slow version
- `reid_openset_tnorm/scripts/00_hygiene_sweep.py` - Target script to optimize

### Documentation
- `README.md` - Detailed progress and plans
- `PROGRESS_SUMMARY.md` - Executive summary
- `QUICKSTART.md` - This file

---

## Next: Deploy to Cluster

Once Iterations 3-5 are complete (target: 80-90% reduction):

### 1. Copy Files to Cluster
```bash
# From local machine
scp -r optimize_hf_loading/ cluster:/home/kdoherty/wolverines/

# Include optimized utils
scp utils/optimized_*.py cluster:/home/kdoherty/wolverines/utils/
```

### 2. Create Optimized SBATCH Script
```bash
#!/bin/bash
#SBATCH --job-name=hygiene_optimized
#SBATCH --partition=preempt
#SBATCH --array=0-23
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

cd /home/kdoherty/wolverines/pelage_sorting
source activate wolverines

# Use optimized version with disk cache
python reid_openset_tnorm/scripts/00_hygiene_sweep_optimized.py \
  --idx ${SLURM_ARRAY_TASK_ID} \
  --use_metadata_cache \
  --cache_dir /data/hf_cache
```

### 3. Submit Jobs
```bash
# Test single job first
sbatch --array=0 sbatch/00_hygiene_sweep_optimized.sbatch

# Verify results match baseline
diff -u results/baseline/threshold=0.00_gallery=2_seed=0.json \
        results/optimized/threshold=0.00_gallery=2_seed=0.json

# If identical, deploy all jobs
sbatch --array=0-23 sbatch/00_hygiene_sweep_optimized.sbatch
```

---

## Estimated Time Savings

| Version | Single Config | 288 Configs (24 jobs) | Total Experiment |
|---------|---------------|------------------------|------------------|
| **Baseline** | 101s | 12 configs/job × 101s | ~20 minutes/job |
| **Iteration 2** | 0.35s* | 12 configs/job × 0.35s | ~4 seconds/job + 102s cache |
| **Iteration 4 (disk cache)** | 0.01s | 12 configs/job × 0.01s | ~1 second/job + 2s cache |
| **Iteration 5 (parallel)** | 0.01s | 12 configs/job × 0.01s | ~1 second/job + 0.5s cache |

\* Plus one-time cache build (102s), amortized across all configs

**Current savings:** ~10 minutes per job × 24 jobs = **4 hours saved**
**Target savings:** ~19 minutes per job × 24 jobs = **7.6 hours saved**

---

## Troubleshooting

### ImportError: No module named 'optimize_hf_loading'
```bash
# Add project root to Python path
export PYTHONPATH=/Users/kdoherty/wolverines:$PYTHONPATH
python optimize_hf_loading/02_benchmark_vectorized.py --local
```

### Results don't match baseline
```bash
# Check for randomness issues - same seed should give same results
python -c "
import json
with open('optimize_hf_loading/results/baseline_timings.json') as f:
    baseline = json.load(f)
with open('optimize_hf_loading/results/iteration_02_timings.json') as f:
    optimized = json.load(f)

# Compare dataset sizes
print('Baseline dataset size:', baseline['metadata']['config_results'][0]['metadata']['dataset_size'])
print('Optimized dataset size:', optimized['metadata']['config_results'][0]['metadata']['dataset_size'])
"
```

### Cache directory full
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/datasets/kdoherty___wolverines

# Or specify different cache
export HF_HOME=/path/to/large/disk
python optimize_hf_loading/02_benchmark_vectorized.py --local
```

---

## Questions?

- **What's optimized?** Sequential dataset access → cached numpy arrays
- **Are results the same?** Yes! Only performance changes, not outputs
- **When to use?** After Iteration 4 (disk cache) for production runs
- **How much faster?** Currently 2x, target 10-100x with more iterations

See `PROGRESS_SUMMARY.md` for detailed analysis.

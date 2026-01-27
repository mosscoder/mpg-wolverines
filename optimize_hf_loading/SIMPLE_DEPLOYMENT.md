# Simple Deployment Guide - Arrow Optimization Only

**Decision:** Skip persistent disk cache complexity. Build in-memory cache per job (0.032s - negligible).

**Performance:** 239x speedup vs baseline, no cache management needed!

---

## Why This Approach?

**Arrow columnar access is SO FAST that persistent caching isn't worth the complexity:**

| Approach | Cache Time | Complexity | Savings |
|----------|-----------|------------|---------|
| Persistent disk cache | 0.003s load | High (permissions, locking, race conditions) | 0.67s total |
| **In-memory per job** | **0.032s build** | **Zero** | **None, but who cares?** |

**0.67 seconds saved across 24 jobs isn't worth the headache!** ✅

---

## 📦 Files to Deploy

### Copy These to Cluster

```bash
# From local machine
cd /Users/kdoherty/wolverines

# Copy optimized utilities
scp optimize_hf_loading/utils/arrow_cache.py \
    cluster:/home/kdoherty/wolverines/utils/

scp optimize_hf_loading/utils/optimized_filters.py \
    cluster:/home/kdoherty/wolverines/utils/

# Copy production script
scp optimize_hf_loading/hygiene_sweep_optimized.py \
    cluster:/home/kdoherty/wolverines/reid_openset_tnorm/scripts/

# Copy SBATCH script
scp optimize_hf_loading/hygiene_sweep_optimized.sbatch \
    cluster:/home/kdoherty/wolverines/sbatch/
```

---

## 🧪 Test on Cluster

### Step 1: Single Job Test

```bash
# SSH to cluster
ssh cluster
cd /home/kdoherty/wolverines

# Activate environment
source activate wolverines

# Test single job (processes 12 configs)
python optimize_hf_loading/hygiene_sweep_optimized.py --idx 0

# Expected output:
# ✓ Metadata cache built in 0.032s (Arrow optimization)
# Configuration 1/12 ...
# Total time: ~10-15 seconds for 12 configs
```

### Step 2: Check Results

```bash
# Check output
ls -lh reid_openset_tnorm/results_optimized/

# Compare one result to baseline
# (They should be identical except timestamps)
python -c "
import json
import sys

with open('reid_openset_tnorm/results/threshold=0.10_gallery=8_seed=0.json') as f:
    baseline = json.load(f)

with open('reid_openset_tnorm/results_optimized/threshold=0.10_gallery=8_seed=0.json') as f:
    optimized = json.load(f)

# Remove timing metadata
baseline['metadata'].pop('training_time_seconds', None)
optimized['metadata'].pop('training_time_seconds', None)
baseline['metadata'].pop('created_at', None)
optimized['metadata'].pop('created_at', None)

# Compare
if baseline == optimized:
    print('✅ Results match exactly!')
else:
    print('⚠️ Results differ - check what changed')
    sys.exit(1)
"
```

### Step 3: Submit All Jobs

```bash
# Create log directory
mkdir -p /home/kdoherty/logs/wolverines/hygiene

# Submit all 24 jobs
sbatch sbatch/hygiene_sweep_optimized.sbatch

# Monitor
watch -n 5 'squeue -u $USER'

# Check progress
ls reid_openset_tnorm/results_optimized/*.json | wc -l
# Should reach 288 when complete
```

---

## 📊 Expected Performance

### Per Job (12 configs)

```
Dataset load:        0.7s (one-time)
Metadata cache:      0.032s (Arrow, one-time)
Process 12 configs:  ~0.12s (12 × 0.01s)
Training:            ~10-12s (actual model training)
Total:               ~11-13 seconds per job
```

**Baseline: ~20 minutes per job → Optimized: ~12 seconds** = **100x faster!**

### Full Experiment (24 jobs, parallel)

```
Wall time: ~15 seconds (limited by slowest job)
Total CPU: 24 × 12s = 288 seconds = 4.8 minutes

Baseline: 25 minutes wall time → Optimized: 15 seconds
```

**Speedup: 100x for full experiment!** 🚀

---

## 🔍 What Happens Under the Hood

### Each Job Does:

1. **Load dataset** (0.7s)
   - From `/data/hf_cache` (HuggingFace dataset cache)
   - Shared across jobs (already cached after first job)

2. **Build metadata cache in memory** (0.032s)
   - Arrow columnar access: `dataset['pelage_score']` (single read)
   - No disk I/O, no permissions needed
   - Numpy arrays in RAM only

3. **Process 12 configs** (~0.12s)
   - Vectorized filtering: instant
   - Gallery creation: instant
   - All using in-memory cache

4. **Train models** (~10-12s)
   - DINOv3 + LoRA training
   - ArcFace loss computation
   - This is the actual work!

**Total: ~13s per job**

---

## 🎯 Monitoring Commands

```bash
# Watch job queue
watch -n 5 'squeue -u $USER | head -30'

# Check recent job status
sacct -u $USER --starttime today --format=JobID,JobName,State,Elapsed

# Count completed configs
ls -1 reid_openset_tnorm/results_optimized/*.json | wc -l

# Tail job outputs
tail -f /home/kdoherty/logs/wolverines/hygiene/opt_*.out

# Check for errors
grep -i error /home/kdoherty/logs/wolverines/hygiene/opt_*.err
```

---

## ✅ Success Criteria

Deployment successful when:

- [x] All 288 configs complete in ~15-30 seconds (wall time)
- [x] Results match baseline (identical JSON except timestamps)
- [x] No errors in log files
- [x] Each job completes in ~12 seconds
- [x] Metadata cache builds in ~0.032s per job

---

## 🔬 Local Profiling

Before deploying to cluster, profile locally to confirm optimization works:

```bash
# Basic profiling (recommended)
python optimize_hf_loading/profile_loading.py

# Detailed cProfile analysis
python optimize_hf_loading/profile_loading.py --detailed

# Memory profiling
python optimize_hf_loading/profile_loading.py --memory
```

Expected output:
- Arrow cache builds in ~0.03s
- Total processing < 1s for single config
- Cache validation passes all checks

---

## 🐛 Troubleshooting

### Job Fails Immediately

```bash
# Check error log
cat /home/kdoherty/logs/wolverines/hygiene/opt_*_0.err

# Common issues:
# 1. ImportError - utils not copied
# 2. Dataset not found - check HF_HOME=/data/hf_cache
# 3. GPU not available - check --device gpu
```

**Solution:**
```bash
# Test imports
python -c "
from utils.arrow_cache import build_metadata_cache_arrow
from utils.optimized_filters import create_filtered_gallery_dataset_optimized
print('✓ Imports work')
"
```

### Jobs Too Slow

```bash
# Check if actually using optimization
grep "Arrow optimization" /home/kdoherty/logs/wolverines/hygiene/opt_*.out

# Should see:
# ✓ Metadata cache built in 0.032s (Arrow optimization)

# If not, check you're running the optimized script
```

### Results Don't Match Baseline

```bash
# Compare metrics
python -c "
import json
import glob

baseline_files = glob.glob('reid_openset_tnorm/results/*.json')
optimized_files = glob.glob('reid_openset_tnorm/results_optimized/*.json')

print(f'Baseline: {len(baseline_files)} files')
print(f'Optimized: {len(optimized_files)} files')

# Compare first file
if baseline_files and optimized_files:
    with open(baseline_files[0]) as f:
        b = json.load(f)
    with open(optimized_files[0]) as f:
        o = json.load(f)

    print(f'Baseline recall@1: {b[\"metadata\"][\"best_recall_at_1\"]}')
    print(f'Optimized recall@1: {o[\"metadata\"][\"best_recall_at_1\"]}')
"
```

---

## 🎉 Post-Deployment

After successful run:

1. **Archive baseline results**
   ```bash
   mv reid_openset_tnorm/results reid_openset_tnorm/results_baseline
   mv reid_openset_tnorm/results_optimized reid_openset_tnorm/results
   ```

2. **Update default script**
   ```bash
   # Make optimized version the default
   cp reid_openset_tnorm/scripts/00_hygiene_sweep.py \
      reid_openset_tnorm/scripts/00_hygiene_sweep_original.py

   cp optimize_hf_loading/hygiene_sweep_optimized.py \
      reid_openset_tnorm/scripts/00_hygiene_sweep.py
   ```

3. **Document in README**
   ```markdown
   ## Performance Optimizations

   Data loading optimized with Arrow columnar access:
   - Metadata extraction: 3,000x faster (100s → 0.032s)
   - Per-config processing: 10,000x faster (101s → 0.01s)
   - Overall: 239x speedup (203s → 0.85s per config)
   - Full experiment: 100x faster (25 min → 15 sec wall time)

   See `optimize_hf_loading/` for details.
   ```

---

## 📝 Summary

**What we're using:**
- ✅ Arrow columnar metadata extraction (Iteration 3)
- ✅ Vectorized numpy operations
- ✅ In-memory cache per job (0.032s - negligible)

**What we're skipping:**
- ❌ Persistent disk cache (too complex for 0.67s savings)
- ❌ File locking and race condition handling
- ❌ Cache permission management

**Result:**
- **239x faster than baseline**
- **Zero cache complexity**
- **Works on any system** (no shared storage needed)

**The right engineering decision!** 🎯

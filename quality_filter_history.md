# Quality Filter Implementation History

## Executive Summary

Investigation of git history reveals that **recall@1 decreasing with higher quality thresholds** is likely due to **a difference in averaging methods** between arcface (macro-averaged) and tnorm (micro-averaged) scripts, combined with possible per-individual quality distribution imbalances.

## Timeline of Key Commits

| Commit | Date | Description | Impact |
|--------|------|-------------|--------|
| `db3d9d8` | Jan 23 | introducing open set tests | Initial implementation |
| `9cf1175` | Jan 23 | **fixed the query filters on open set** | Changed rare_indices from `quality_threshold=gallery_threshold` to `quality_threshold=0.0` (eval-time filtering). **ARCFACE ONLY** |
| `1ae4e85` | Jan 27 | optimize hf loading | Created tnorm script, added optimized_filters.py |
| `89aeff8` | Jan 27 | tnorm script missing optimizations, fixed | Updated tnorm to use vectorized filtering |
| `c4faeb3` | Jan 27 | print all q level performance per epoch | Added per-quality output |

## Critical Finding #1: Query Filter Fix (commit 9cf1175)

This commit changed how rare/unknown individuals are collected:

**BEFORE (broken):**
```python
rare_indices, rare_quality = get_rare_individual_indices(
    dataset, valid_individuals, id_to_indices,
    quality_threshold=gallery_threshold  # WRONG: filtered at collection time
)
```

**AFTER (fixed):**
```python
rare_indices, rare_quality = get_rare_individual_indices(
    dataset, valid_individuals, id_to_indices,
    quality_threshold=0.0  # CORRECT: get all, filter at eval time
)
```

**Files modified:** `reid_openset_arcface/scripts/00_hygiene_sweep.py`, `utils/triplet.py`
**Files NOT modified:** `reid_openset_tnorm`, `reid_openset_lora`

However, when tnorm was created (commit `1ae4e85`), it appears to have been created from the FIXED arcface version, as line 548 shows:
```python
rare_indices, rare_quality, rare_labels_str = get_rare_individual_indices(
    metadata_cache, valid_individuals, quality_threshold=0.0  # Correct
)
```

## Critical Finding #2: Different Recall@1 Computation Methods

### Arcface (uses macro-averaging)
```python
# From compute_recall_by_query_quality() -> compute_recall_at_k()
# Groups by individual, computes per-individual recall, then averages
individual_correct = defaultdict(int)
individual_total = defaultdict(int)
for i in range(query_embeddings.size(0)):
    query_label = query_labels[i].item()
    individual_total[query_label] += 1
    if query_label in top_k_labels:
        individual_correct[query_label] += 1
# Returns mean of per-individual recalls
```

### Tnorm (uses micro-averaging)
```python
# Inline in evaluate_recall_with_openset() lines 584-603
correct = (pred_labels == filtered_labels).float().sum()
recall = (correct / count).item()  # Simple correct/total
```

**Impact:** If quality scores are NOT uniformly distributed across individuals, macro vs micro averaging will give different results when filtering by quality.

## Critical Finding #3: Vectorized Filtering Order Change

The HF loading optimization (`1ae4e85`) changed filtering from row-by-row to vectorized:

**BEFORE (row-by-row):**
```python
filtered = []
for idx in indices:
    if dataset[idx]['pelage_score'] >= threshold:
        filtered.append(idx)
return filtered
```

**AFTER (vectorized):**
```python
indices_arr = np.array(indices)
quality_arr = quality_cache[indices_arr]
mask = quality_arr >= threshold
return indices_arr[mask].tolist()
```

The vectorized version notes: "Order may differ but doesn't affect downstream sampling"

This is true for gallery creation (which shuffles), but **may affect evaluation if order matters**.

## Hypothesis: Why Recall@1 Decreases with Higher Quality Thresholds

1. **Per-individual quality imbalance**: Some individuals may have most of their samples at low quality. When filtering to q>=0.5, these individuals contribute fewer samples, changing the effective population.

2. **Micro-averaging sensitivity**: The tnorm script uses micro-averaging. If low-quality samples are easier to match (perhaps because they're more typical/average poses), removing them decreases overall accuracy.

3. **Gallery-Query domain shift**: The gallery was quality-filtered at training time. High-quality validation queries may have different characteristics than the gallery samples they're being matched against.

## Files Involved

| File | Purpose | Recall Method |
|------|---------|---------------|
| `reid_openset_arcface/scripts/00_hygiene_sweep.py` | ArcFace training | Macro-averaged |
| `reid_openset_tnorm/scripts/00_hygiene_sweep.py` | T-Norm evaluation | Micro-averaged |
| `reid_openset_lora/scripts/00_hygiene_sweep.py` | LoRA training | (check) |
| `utils/triplet.py` | Shared evaluation functions | Macro-averaged |
| `utils/optimized_filters.py` | Vectorized filtering | N/A |
| `utils/arrow_cache.py` | Metadata caching | N/A |

## Recommendations

### Option A: Align Averaging Methods
Change tnorm to use macro-averaged recall@1 like arcface:
```python
# Replace inline computation with:
recall = compute_recall_at_k(filtered_query_emb, gallery_emb,
                             filtered_query_labels, gallery_labels, k=1)
```

### Option B: Add Diagnostic Output
Add per-individual quality distribution stats:
```python
def print_quality_by_individual(query_labels, query_quality, individuals):
    for ind in individuals:
        mask = (query_labels == ind)
        q = query_quality[mask]
        print(f"  {ind}: n={len(q)}, q_mean={q.mean():.2f}, n_q>=0.5={sum(q>=0.5)}")
```

### Option C: Verify with Synthetic Data
Create a test case where quality is uniformly distributed across individuals to confirm the filtering logic is correct in isolation.

## Verification Commands

```bash
# Check commit details
git show 9cf1175 --stat
git show 1ae4e85 --stat

# Compare recall computation
diff <(grep -A30 "def compute_recall" utils/triplet.py) \
     <(grep -A30 "Recall@1" reid_openset_tnorm/scripts/00_hygiene_sweep.py)

# Check quality distribution
python -c "
from datasets import load_dataset
ds = load_dataset('kdoherty/wolverines')['train']
import numpy as np
q = np.array(ds['pelage_score'])
print(f'Quality stats: mean={q.mean():.3f}, std={q.std():.3f}')
for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f'  q>={t}: {(q>=t).sum()} samples')
"
```

---

## Implemented Fixes (2026-01-27)

### Changes Made

1. **`utils/triplet.py`**: Added L2 normalization to `compute_recall_at_k` before computing distances
   - All scripts now use cosine-based ranking consistently

2. **`reid_openset_tnorm/scripts/00_hygiene_sweep.py`**:
   - Replaced inline closed-set recall computation with call to `compute_recall_at_k` (macro-averaged)
   - Added `compute_open_set_metrics_tnorm` function for macro-averaged open-set evaluation
   - Updated open-set metrics to use per-individual averaging for both known and unknown queries
   - Added `rare_labels_arr` initialization in else clause
   - Updated print statements to show individual counts

3. **`reid_openset_arcface/scripts/00_hygiene_sweep.py`** and **`reid_openset_lora/scripts/00_hygiene_sweep.py`**:
   - Updated print statements to show individual counts `[n_k]` and `[n_u]`

### Expected Output Format

```
Epoch   1/50: Loss=1.2345, ValLoss=1.1234, thresh=4.000
  q>=0.0: R@1=0.2500 (n=128), BA=0.6234 (K=0.75[5], U=0.50[3])
  q>=0.1: R@1=0.2600 (n=115), BA=0.6350 (K=0.77[5], U=0.52[3])
  ...
```

Where `[5]` indicates 5 known individuals and `[3]` indicates 3 unknown individuals used in macro-averaging.

# Open-Set Re-Identification with T-Norm Score Normalization

This experiment uses **frozen DINOv3 backbone** with **T-Norm (Test Normalization)** for improved open-set recognition.

## Overview

This experiment uses T-Norm score normalization to improve open-set re-identification:

1. **Frozen DINOv3 Backbone**: Efficient feature extraction (~100K trainable params in projection head only)
2. **T-Norm Score Normalization**: Normalize similarity scores to Z-scores using imposter distribution statistics

T-Norm normalizes similarity scores to Z-scores using statistics from imposter (negative) distributions in the gallery. This helps create more discriminative scores for distinguishing known individuals from unknown individuals.

### Key Differences from `reid_openset_arcface`

1. **Score Normalization**: T-Norm Z-scores instead of raw cosine similarity
2. **Threshold Scale**: Z-scores (e.g., 2.5, 3.8, 5.0) vs cosine (e.g., 0.3, 0.4, 0.5)
3. **Better Handling**: More aggressive filtering of low-quality gallery templates
4. **Statistical Properties**: Negatives are normalized to N(0,1) distribution
5. **Same Architecture**: Both use frozen DINOv3 with trainable projection head (~100K params)

## T-Norm Implementation

### Computation Steps

1. **Gallery Statistics** (offline): For each gallery sample g, compute μ[g] and σ[g] from its imposter cohort (samples NOT from the same individual)

2. **Score Normalization**: Apply normalization to similarity scores:
   ```
   S_norm(q, g) = (Sim(q, g) - μ[g]) / σ[g]
   ```

3. **Threshold Calibration**: Calibrate decision threshold on normalized scores using LOO within gallery

4. **Open-Set Evaluation**: Measure Balanced Accuracy = (Known Accept Rate + Unknown Reject Rate) / 2

### Expected Results

- **Threshold Scale**: Thresholds will be Z-scores (e.g., 2.5-5.0) instead of cosine distances
- **Volatility**: At small gallery sizes (e.g., 2), expect higher variance due to thin statistics
- **Unknown Rejection**: Should see improved Unknown Reject Rate compared to raw ArcFace scores

## Experiment Parameters

**Grid**: 6 thresholds × 6 gallery sizes × 8 seeds = 288 configurations
- **Thresholds**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] (image quality filtering)
- **Gallery Sizes**: [2, 4, 8, 16, 32, 64] (examples per individual)
- **Seeds**: [0, 1, 2, 3, 4, 5, 6, 7]

**Model**: Frozen DINOv3-ViT-B/16 + trainable projection head (768→128) + ArcFace loss
- **Backbone**: Frozen (no gradients)
- **Trainable Params**: ~100K (projection head only)

**Optimizer**: AdamW (lr=0.0005, default settings)

**Distributed**: 24 SLURM jobs (12 configs per job)

## Usage

### Submit Jobs

```bash
cd /home/kdoherty/wolverines
sbatch reid_openset_tnorm/sbatch/00_hygiene_sweep.sbatch
```

### Monitor Progress

```bash
# Check SLURM queue
squeue -u $USER

# View logs
tail -f /home/kdoherty/logs/wolverines/openset_tnorm/*.out

# Check results
ls -lh reid_openset_tnorm/results/
```

### Generate Figures

```bash
cd /home/kdoherty/wolverines
python reid_openset_tnorm/scripts/01_plot_hygiene_sweep.py
```

Figures saved to: `reid_openset_tnorm/figures/`

## Directory Structure

```
reid_openset_tnorm/
├── scripts/
│   ├── 00_hygiene_sweep.py      # Training script with T-Norm
│   └── 01_plot_hygiene_sweep.py # Plotting script
├── sbatch/
│   └── 00_hygiene_sweep.sbatch  # SLURM submission script
├── results/                      # JSON results from sweep
└── figures/                      # Generated plots
```

## Results Format

Each configuration saves a JSON file: `threshold={t:.2f}_gallery={g}_seed={s}.json`

### Result Structure

```json
{
  "config": {
    "threshold": 0.1,
    "gallery_size": 8,
    "seed": 0,
    "score_normalization": "T-Norm",
    ...
  },
  "epoch_history": [
    {
      "epoch": 1,
      "query_quality_metrics": {...},
      "open_set": {
        "threshold_calibration": {
          "method": "loo_tnorm",
          "threshold_mean": 3.85,  // Z-score threshold
          ...
        },
        "by_quality": {
          "q>=0.0": {
            "balanced_accuracy": 0.78,
            "known_accept_rate": 0.82,
            "unknown_reject_rate": 0.74
          },
          ...
        }
      }
    },
    ...
  ]
}
```

## Comparison with Baselines

Compare T-Norm normalization with:
- **`reid_openset_arcface`**: Frozen backbone + raw cosine similarity (baseline)
- **`reid_openset_tnorm`**: Frozen backbone + T-Norm score normalization (this experiment)

Expected improvements from T-Norm:
- Higher Balanced Accuracy from T-Norm (better open-set discrimination)
- Better separation between known and unknown individuals
- More robust to low-quality gallery images
- Statistical interpretability of thresholds (Z-scores)

## References

- T-Norm: Auckenthaler et al., "Score Normalization for Text-Independent Speaker Verification Systems", Digital Signal Processing, 2000
- ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
- DINOv3: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024

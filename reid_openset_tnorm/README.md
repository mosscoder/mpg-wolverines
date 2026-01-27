# Open-Set Re-Identification with LoRA + T-Norm Score Normalization

This experiment combines **LoRA-adapted backbone** from `reid_openset_lora` with **T-Norm (Test Normalization)** for improved open-set recognition.

## Overview

This experiment combines two key enhancements:

1. **LoRA Adapters**: Fine-tune DINOv3 backbone efficiently (~2.5M trainable params)
2. **T-Norm Score Normalization**: Normalize similarity scores to Z-scores using imposter distribution statistics

T-Norm normalizes similarity scores to Z-scores using statistics from imposter (negative) distributions in the gallery. This helps create more discriminative scores for distinguishing known individuals from unknown individuals.

### Key Differences from `reid_openset_lora`

1. **Score Normalization**: T-Norm Z-scores instead of raw cosine similarity
2. **Threshold Scale**: Z-scores (e.g., 2.5, 3.8, 5.0) vs cosine (e.g., 0.3, 0.4, 0.5)
3. **Better Handling**: More aggressive filtering of low-quality gallery templates
4. **Statistical Properties**: Negatives are normalized to N(0,1) distribution

### Key Differences from `reid_openset_arcface`

1. **Backbone**: LoRA adapters (~2.4M params) instead of frozen backbone
2. **Trainable Params**: ~2.5M (LoRA + head) vs ~100K (head only)
3. **Optimizer**: AdamW with weight_decay=0.01 vs SGD with momentum=0.9
4. **Learning Rate**: 0.0005 vs 0.001
5. **Score Normalization**: T-Norm Z-scores instead of raw cosine similarity

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

**Model**: DINOv3-ViT-B/16 + LoRA adapters + trainable projection head (768→128) + ArcFace loss
- **LoRA Config**: r=16, alpha=32, dropout=0.1
- **Target Modules**: q_proj, k_proj, v_proj, up_proj, down_proj
- **Trainable Params**: ~2.5M (LoRA ~2.4M + head ~100K)

**Optimizer**: AdamW (lr=0.0005, weight_decay=0.01)

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

Compare LoRA + T-Norm results with:
- **`reid_openset_arcface`**: Frozen backbone + raw cosine similarity (baseline)
- **`reid_openset_lora`**: LoRA-adapted backbone + raw cosine similarity (no T-Norm)
- **`reid_openset_tnorm`**: LoRA-adapted backbone + T-Norm score normalization (this experiment)

Expected improvements from combining LoRA + T-Norm:
- Higher Recall@1 from LoRA fine-tuning (better embeddings)
- Higher Balanced Accuracy from T-Norm (better open-set discrimination)
- Better separation between known and unknown individuals
- More robust to low-quality gallery images

## References

- T-Norm: Auckenthaler et al., "Score Normalization for Text-Independent Speaker Verification Systems", Digital Signal Processing, 2000
- ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
- DINOv3: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024

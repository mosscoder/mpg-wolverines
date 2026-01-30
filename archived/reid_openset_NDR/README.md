# Open-Set Re-Identification with NDR Metrics

This experiment extends `reid_openset_tnorm` by replacing balanced accuracy with **Novelty Detection Rate (NDR)** metrics, providing a more intuitive evaluation paradigm for open-set wolverine re-identification.

## Metric Definitions

### NDR Paradigm
- **Positive class**: Novel (unknown) wolverines
- **Negative class**: Known wolverines

### Key Metrics
- **False Novelty Rate (FNR)**: Percentage of known wolverines incorrectly flagged as "Novel" (the cost)
- **Novelty Detection Rate (NDR)**: Percentage of unknown wolverines correctly flagged as "Novel" (the goal)

### Evaluation Approach
We fix FNR at operational tolerance levels (1%, 5%, 10%) and measure NDR at each level:
- **NDR@FN=1%**: NDR when only 1% of known wolverines are misclassified as novel
- **NDR@FN=5%**: NDR when 5% false novelty rate is tolerated
- **NDR@FN=10%**: NDR when 10% false novelty rate is tolerated

## Dynamic Threshold Computation

Instead of using a fixed threshold calibrated from gallery statistics, thresholds are computed dynamically from the known query score distribution:

1. Compute max T-normed similarity score for each known query
2. For each FNR level (e.g., 5%), find the percentile threshold:
   - `Threshold = percentile(known_max_scores, FNR%)`
3. A query is flagged as "novel" if `max_score < threshold`

This ensures the FNR is exactly controlled at the specified level.

## Relationship to reid_openset_tnorm

| Aspect | reid_openset_tnorm | reid_openset_NDR |
|--------|-------------------|------------------|
| Score Normalization | T-Norm | T-Norm (same) |
| Threshold | Calibrated from gallery (LOO) | Dynamic from known percentiles |
| Primary Metric | Balanced Accuracy | NDR @ FNR levels |
| Architecture | DINOv3 + ArcFace | DINOv3 + ArcFace (same) |
| Grid Search | 6x6x8 = 288 configs | 6x6x8 = 288 configs (same) |

## JSON Output Structure

```json
{
  "epoch_history": [{
    "epoch": 1,
    "train_loss": 5.23,
    "val_loss": 4.87,
    "query_quality_metrics": {
      "q>=0.0": {"recall_at_1": 0.72, "count": 150}
    },
    "open_set": {
      "by_quality": {
        "q>=0.0": {
          "NoveltyDetectionRate@FalseNovelty=1%": 0.45,
          "NoveltyDetectionRate@FalseNovelty=5%": 0.72,
          "NoveltyDetectionRate@FalseNovelty=10%": 0.85,
          "Threshold_FN=1%": 5.23,
          "Threshold_FN=5%": 4.15,
          "Threshold_FN=10%": 3.82,
          "n_known_samples": 150,
          "n_unknown_samples": 89
        }
      }
    }
  }],
  "metadata": {
    "best_ndr_epoch": 35,
    "best_ndr_at_5pct": 0.78,
    "best_epoch": 40,
    "best_recall_at_1": 0.85
  }
}
```

## Directory Structure

```
reid_openset_NDR/
├── scripts/
│   ├── 00_hygiene_sweep.py      # Main training + NDR evaluation
│   └── 01_plot_hygiene_sweep.py # Visualization
├── sbatch/
│   └── 00_hygiene_sweep.sbatch  # SLURM job script
├── results/                      # JSON output files
├── figures/                      # Generated plots
└── README.md
```

## Running the Experiment

### Submit Jobs
```bash
cd /home/kdoherty/wolverines
sbatch reid_openset_NDR/sbatch/00_hygiene_sweep.sbatch
```

### Monitor Progress
```bash
squeue -u $USER
tail -f /home/kdoherty/logs/wolverines/openset_ndr/*.out
```

### Generate Figures
```bash
python reid_openset_NDR/scripts/01_plot_hygiene_sweep.py
```

## Output Figures

### open_set_ndr_faceted.png
3x2 faceted figure:
- **Columns**: FN rate (1%, 5%, 10%)
- **Row 1**: NDR vs gallery_size with strategy lines (baseline/minimal/optimal)
- **Row 2**: NDR vs quality_threshold with gallery_size lines

### closed_set_combined.png
Same as reid_openset_tnorm - Recall@1 evaluation for closed-set re-identification

### recall_curves.png
Epoch-wise recall@1 curves for each configuration

## Local Testing

```bash
# Test one configuration locally
cd /Users/kdoherty/wolverines
python reid_openset_NDR/scripts/00_hygiene_sweep.py --idx 0 --device cpu

# Verify JSON output
cat reid_openset_NDR/results/threshold=0.00_gallery=2_seed=0.json | python -m json.tool

# Test plotting (after results exist)
python reid_openset_NDR/scripts/01_plot_hygiene_sweep.py
```

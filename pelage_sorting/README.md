# Stage 1: pelage visibility classifier

A frozen DINOv3-ViT-B/16 encoder with one trained linear layer (768 to 2)
that scores each crop by the probability that the pelage pattern is clearly
visible. Trained on the `pelage` configuration of `kdoherty/wolverines`
(2,431 training and 270 test crops, labeled Full versus None or Partial).

Fixed settings: 224 by 224 resize with no center crop, no augmentation, AdamW
at learning rate 0.001 and weight decay 0.01, batch size 32, cross-entropy
loss. Metrics are macro-averaged over the two classes.

## Scripts, in run order

Run from the repository root in the `wolverines` environment. Scripts 00 to
02 have matching `sbatch/` files for a Slurm GPU partition.

| Script | What it does | Output |
|---|---|---|
| `scripts/00_find_best_epoch.py` | Five-fold stratified cross-validation on the training split, macro F1 recorded at each of 50 epochs; the epoch maximizing the cross-validated mean is selected (16 for the manuscript) | `results/00_best_epoch/` |
| `scripts/01_test_performance.py` | Trains on the full training split for the selected epochs and evaluates the test split each epoch | `results/01_test/final_test_performance.json` |
| `scripts/02_train_production.py` | Trains the production head on train plus test with the selected settings and saves only the linear layer | `results/02_production/` |
| `scripts/03_make_figures.py` | Training and validation curves for the three runs above | `figures/` (not tracked) |
| `scripts/04_head_ablation.py` | Reviewer-requested ablation: linear head versus a two-layer head (256 or 768 hidden units, ReLU) on identical cached features, epochs chosen by cross-validation, five seeds each. Runs locally on a single GPU or CPU | `results/04_head_ablation/` |
| `scripts/04_head_ablation_table.py` | Supplementary table fragment from the per-seed ablation JSONs | `results/04_head_ablation/table_s7_head_ablation.tex` |

```bash
sbatch pelage_sorting/sbatch/00_find_best_epoch.sbatch
sbatch pelage_sorting/sbatch/01_test_performance.sbatch
sbatch pelage_sorting/sbatch/02_train_production.sbatch
python pelage_sorting/scripts/04_head_ablation.py
```

## Reported result

At the selected 16 epochs the classifier reaches macro-averaged test F1 0.87
(precision 0.92, recall 0.84). The head ablation found no gain from added
capacity: F1 0.905 (linear, 28 epochs), 0.885 (256 hidden units), 0.856 (768
hidden units), means over five seeds at each head's own selected epoch count.

The production head is applied to every out-of-sample crop by
`hugging_face_dataset/v2/scripts/05_infer_pelage.py`, which produces the
`pelage_score` column of the re-identification dataset.

Experiment notes are in `RESEARCH_LOG.md`.

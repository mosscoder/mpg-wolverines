# Stage 1: pelage visibility classifier

A frozen DINOv3-ViT-B/16 encoder with one trained linear layer (768 to 2)
that scores each crop by the probability that the pelage pattern is clearly
visible. Trained on the `pelage` configuration of `mpg-ranch/wolverines`
(2,431 training and 270 test images, labeled Full versus None or Partial).

Fixed settings: 224 by 224 resize with no center crop, no augmentation, AdamW
at learning rate 0.001 and weight decay 0.01, batch size 32, cross-entropy
loss. Metrics are macro-averaged over the two classes.

Scripts live in `scripts/` and run in numbered order from the repository
root in the `wolverines` environment. Scripts 00 to 02 have matching `sbatch/`
files for a Slurm GPU partition.

## 00. Epoch selection

**Script:** `scripts/00_find_best_epoch.py`
**Slurm:** `sbatch pelage_sorting/sbatch/00_find_best_epoch.sbatch`

Runs five-fold cross-validation on the training split, stratified by label,
and records macro F1 at each of 50 epochs. The epoch that maximizes the
cross-validated mean is selected (16 for the manuscript).

**Output:** `results/00_best_epoch/`.

## 01. Test performance

**Script:** `scripts/01_test_performance.py`
**Slurm:** `sbatch pelage_sorting/sbatch/01_test_performance.sbatch`

Trains on the full training split for the selected number of epochs and
evaluates the test split at each epoch.

**Output:** `results/01_test/final_test_performance.json`.

## 02. Production classifier

**Script:** `scripts/02_train_production.py`
**Slurm:** `sbatch pelage_sorting/sbatch/02_train_production.sbatch`

Trains the production head on the training and test splits together with the
selected settings and saves only the linear layer.

**Output:** `results/02_production/`.

## 03. Training curves

**Script:** `scripts/03_make_figures.py`

Plots training and validation curves for the three runs above.

**Output:** `figures/`, which is not tracked.

## 04. Head ablation

**Scripts:** `scripts/04_head_ablation.py`, `scripts/04_head_ablation_table.py`

The reviewer-requested ablation compares the linear head with a two-layer head
(256 or 768 hidden units, ReLU) on identical cached encoder outputs, with
epochs chosen by cross-validation and five seeds each. It runs locally on a
single GPU or CPU (`python pelage_sorting/scripts/04_head_ablation.py`). The
table script then writes the fragment for Supplementary Table S6 from the
per-seed ablation JSONs (the file name keeps its earlier S7 number).

**Output:** `results/04_head_ablation/`, including
`table_s7_head_ablation.tex`.

## Reported result

At the selected 16 epochs the classifier reaches macro-averaged test F1 0.87
(precision 0.92, recall 0.84). The head ablation found no gain from added
capacity: F1 0.905 (linear, 28 epochs), 0.885 (256 hidden units), 0.856 (768
hidden units), means over five seeds at each head's own selected epoch count.

The production head is applied to every out-of-sample crop by
`hugging_face_dataset/v2/scripts/05_infer_pelage.py`, which produces the
`pelage_score` column of the re-identification dataset.

## See also

- [Repository overview](../README.md)
- [Dataset creation](../hugging_face_dataset/v2/README.md)
- [Role assignment and manuscript assets](../preprocessing/README.md)
- [Re-identification and novelty detection](../reid_openset/README.md)

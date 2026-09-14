# pelage_sorting research log

One entry per experiment, run order, kebab-case slugs. Five bullets each:
Objective, Methods, Positive, Negative, Follow-up. Keep entries under 250
words and shallow. Convention shared with the vole repo
(`local_drivers/RESEARCH_LOG.md` there is the canonical template).

## head-ablation

- **Objective**: Answer reviewer comment R1.4 for the pelage visibility
  classifier: does a head with one hidden layer beat the published linear
  head on frozen DINOv3-ViT-B/16 features?
- **Methods**: `scripts/04_head_ablation.py` (run 2026-08-25, results in
  `results/04_head_ablation/`). One frozen forward pass caches 768-d
  features for train, validation, and test (`features.npz`, ignored,
  regenerable). Heads train on the cached features with the production
  optimizer, learning rate, and batch size. Training length per head is
  chosen by five-fold cross-validation on the training split, then each
  head is trained on the full split with five seeds. Heads: Linear(768 to
  2), and Linear(768 to h), ReLU, Linear(h to 2) at h of 256 and 768.
  `scripts/04_head_ablation_table.py` writes the supplementary table
  fragment from the per-seed JSONs.
- **Positive**: The fidelity control passes. At the published 16 epochs
  the linear head gives test macro F1 0.884, standard deviation 0.023,
  bracketing the published single-seed 0.874. At its own selected 28
  epochs it gives 0.905, standard deviation 0.009.
- **Negative**: Capacity does not help. The 256-unit head gives 0.885
  (standard deviation 0.015, 34 epochs) and the 768-unit head 0.856
  (standard deviation 0.034, 12 epochs), a monotonic decline with head
  size. Frozen features, not the head, set performance.
- **Follow-up**: Reported as a supplementary table (fragment copied to
  Overleaf by `manuscript/update.py`) and one Methods sentence, recorded
  under R1.4 in the revision roadmap. Published numbers unchanged.

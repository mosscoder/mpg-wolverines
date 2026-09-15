# Stage 2: re-identification and novelty detection

Three frozen encoders, each with a projection head (linear layer, batch
normalization, L2 normalization) trained under ArcFace loss (margin 0.5,
scale 64) on the gallery, the training images of the six known individuals.
A query is assigned to the known individual of its nearest gallery image by
cosine similarity, or flagged as unknown if that similarity is below the
matched individual's novelty threshold, the 10th percentile of the similarity
between its gallery images and its learned ArcFace center.

| Encoder | Class | Directory |
|---|---|---|
| DINOv3-ViT-B/16 | general-purpose | `dinov3/` |
| BioCLIP-2 ViT-L/14 | biology-specific | `bioclip2/` |
| MegaDescriptor-L-384 | wildlife-specific | `megadescriptor/` |

Quality score thresholds of 0, 0.25, and 0.50 are applied to the gallery
before training (gallery quality threshold) and to the queries at evaluation
(query quality threshold). Every training entry point is `python -m
utils.reid <command> --model <encoder>`; the per-encoder `scripts/*.sbatch`
files wrap them as Slurm array jobs and are the record of the arguments used.

## Stages per encoder, in run order

| Sbatch | Command | Design | Output |
|---|---|---|---|
| `00_sweep_lr` | `opt_lr` | 4 learning rates by 5 seeds on validation queries | `results/opt/lr/` |
| `01_sweep_embedding_dim` | `opt_embedding_dim` | embedding dimension 256 or 512 by 5 seeds | `results/opt/embedding_dim/` |
| `02_hygiene_sweep` | `hygiene` | few-shot analysis: 3 gallery quality thresholds by 6 gallery sizes (2 to 64 images per individual) by 8 seeds, 144 runs; 300 steps, evaluated every 10, checkpoint chosen by cross-validated recall at rank one; test metrics recorded at every query quality threshold | `results/hygiene/threshold=0.00_gallery=16_seed=0.json` and so on |
| `03_tune_step_count` | `tune_step_count` | full-data step count: 3 gallery quality thresholds by 5 seeds, 3,000-step budget evaluated every 100 | `results/step_count/` |
| `04_test_eval` | `test_step_eval` | full-data comparison: one seed-0 model per gallery quality threshold trained to the step selected in 03, evaluated on the test split at every query quality threshold, 9 cells per encoder | `results/test_eval/` |
| `05_generate_embeddings` | `test_step_eval --export-embeddings` | the invocation that produced the tracked embedding exports read by the t-SNE figure | `results/embeddings/*.npz` |

```bash
sbatch reid_openset/dinov3/scripts/02_hygiene_sweep.sbatch   # and likewise per encoder and stage
```

Selected hyperparameters for the manuscript (learning rate 0.0005 and 256
dimensions for DINOv3 and MegaDescriptor, 0.001 and 512 for BioCLIP-2) are
listed in `summary/fewshot/hyperparameters.md`.

## Summaries and figures

| Script | What it does | Output |
|---|---|---|
| `aggregate_results.py` | Tidy tables and manuscript figures from every results JSON: few-shot recall and balanced accuracy, test evaluation, hyperparameters, LaTeX table fragments | `summary/fewshot/`, `summary/test_eval/`, `summary/latex/` |
| `make_tsne_raw_figure.py` | Manuscript Figure 7: t-SNE of each encoder's frozen output before training and its embedding after training at gallery quality threshold 0.50, from `results/embeddings/` | `summary/tsne/tsne_raw_vs_trained.png` |
| `fig34_step300.py` | Few-shot figures re-read at the fixed final checkpoint (step 300) instead of the selected checkpoint, for the review response | review response figures (not tracked here) |

## Evaluation

Implemented in `utils/reid_evaluation.py`.

- **Recall at rank one**: the fraction of known queries whose nearest gallery
  image belongs to the correct individual, computed per individual and
  averaged over individuals. Closed-set: computed before any novelty decision.
- **Balanced accuracy**: the mean of the known acceptance rate (known queries
  at or above their matched individual's threshold) and the unknown rejection
  rate (unknown queries below it), each averaged over individuals.
- **Novelty threshold**: per known individual, the 10th percentile of cosine
  similarity between its gallery embeddings and its L2-normalized ArcFace
  center (`compute_arcface_center_thresholds`).
- **Confidence intervals**: 95% t-intervals across seeds.

`bioclip2/PROJECTION_NOTES.md` records how BioCLIP-2's text-alignment layer is
handled.

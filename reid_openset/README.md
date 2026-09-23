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

Each stage below is one sbatch file in an encoder's `scripts/` directory,
wrapping the `utils.reid` command it names, and writes to that encoder's
`results/` directory. Stages run in numbered order, once per encoder, for
example `sbatch reid_openset/dinov3/scripts/02_hygiene_sweep.sbatch`.

## 00. Learning rate sweep

**Sbatch:** `scripts/00_sweep_lr.sbatch`
**Command:** `opt_lr`

Sweeps four learning rates (0.0001, 0.0005, 0.001, and 0.005) at an embedding
dimension of 256, with five seeds each, training for 100 steps and evaluating
every 10. Each run trains on the earlier half of every known individual's
training-period events and is scored by recall at rank one on the later half,
with up to 64 images per individual on each side. The learning rate with the
highest seed-averaged recall at its best step is selected.

**Output:** `results/opt/lr/`.

## 01. Embedding dimension sweep

**Sbatch:** `scripts/01_sweep_embedding_dim.sbatch`
**Command:** `opt_embedding_dim`

Repeats that design for embedding dimensions of 256 and 512 at the selected
learning rate.

**Output:** `results/opt/embedding_dim/`.

## 02. Few-shot experiment

**Sbatch:** `scripts/02_hygiene_sweep.sbatch`
**Command:** `hygiene`

Crosses 3 gallery quality thresholds with 6 gallery sizes (2 to 64 images per
individual) and 8 seeds, for 144 runs per encoder. Each run trains for 300
steps and is evaluated every 10 steps on the validation queries and on the
test queries at every query quality threshold. The reported checkpoint is the
step with the highest recall at rank one on the validation queries, averaged
over the eight seeds.

**Output:** `results/hygiene/threshold=0.00_gallery=16_seed=0.json` and so on.

## 03. Full-data step selection

**Sbatch:** `scripts/03_tune_step_count.sbatch`
**Command:** `tune_step_count`

Selects training length for the full-data experiment. It crosses 3 gallery
quality thresholds with 5 seeds on a 3,000-step budget evaluated every 100
steps, and for each pair of gallery and query quality thresholds it keeps the
step with the highest mean recall at rank one on the validation queries
(Supplementary Table S5).

**Output:** `results/step_count/`.

## 04. Full-data experiment

**Sbatch:** `scripts/04_test_eval.sbatch`
**Command:** `test_step_eval`

Trains one seed-0 model per gallery quality threshold and evaluates it once on
the test split at each query quality threshold, using the step selected for
that pair in 03, for 9 cells per encoder.

**Output:** `results/test_eval/`.

## 05. Embedding exports

**Sbatch:** `scripts/05_generate_embeddings.sbatch`
**Command:** `test_step_eval --export-embeddings`

Records the invocation that produced the tracked embedding exports read by the
t-SNE figure.

**Output:** `results/embeddings/*.npz`.

## Running on Slurm

A Slurm cluster with a GPU partition is a prerequisite: every stage is
launched as an array job. Every `.sbatch` file in this directory has the same
structure.

**The header** asks for one GPU, 16 CPUs, 32 GB, and 48 hours on a
preemptible partition with `--requeue`. Preemption means the scheduler may
kill a job to make room for higher-priority work and start it again later;
the training code checkpoints as it goes and resumes from the last
checkpoint, so a requeued job loses minutes, not hours. `--array=0-N`
launches N+1 independent tasks from one file, and each task receives its
number as `$SLURM_ARRAY_TASK_ID`.

**The body** creates the log directory, changes to the repository root,
activates the `wolverines` environment, points the Hugging Face cache at
shared storage in offline mode, sets a scratch directory, and runs one
command:

```bash
python -u -m utils.reid <command> --model <encoder> --idx $SLURM_ARRAY_TASK_ID \
    --output_dir reid_openset/<encoder>/results/<stage> --device gpu
```

The `--idx` value selects one configuration from the stage's grid (which
learning rate and seed, which threshold, gallery size, and seed, and so on),
so the file's header comment states how many tasks the stage has and what
each index means. Task counts: 20 for the learning rate sweep, 10 for the
embedding dimension, 24 for the few-shot experiment (six configurations per
task), 15 for the step count, and 3 for the test evaluation.

**On another cluster**, edit only the top of the file: the partition, QOS,
and any GPU `--constraint`, the four absolute paths (log directory,
repository root, Hugging Face cache, scratch), and the environment
activation line. Everything after `cd` is relative to the repository root.
The dataset must be in the cache before offline jobs run; `scripts/populate_hf_cache.sbatch`
downloads both configurations once on a node with network access.

Useful commands: `sbatch file.sbatch` submits, `squeue -u $USER` lists your
jobs, `sacct -j <jobid>` shows the state of each array task after it ends,
and `sbatch --array=3,7 file.sbatch` reruns only the tasks that failed. Logs
land in the directory named by `--output` and `--error`, one file per task.

Selected hyperparameters for the manuscript (learning rate 0.0005 and 256
dimensions for DINOv3 and MegaDescriptor, 0.001 and 512 for BioCLIP-2) are
listed in `summary/fewshot/hyperparameters.md`.

## Summaries and supplementary tables

**Script:** `aggregate_results.py`

Builds tidy tables and the manuscript figures from every results JSON,
covering few-shot recall at rank one and balanced accuracy, the full-data test
evaluation, and the selected hyperparameters. It also writes the LaTeX
fragments for Supplementary Tables S1, S3, S4, and S5.

**Output:** `summary/fewshot/`, `summary/test_eval/`, and `summary/latex/`.

## Embedding space figure

**Script:** `make_tsne_raw_figure.py`

Draws manuscript Figure 7, a t-SNE projection of each encoder's embeddings
after training with no quality filtering and after training at gallery and
query quality thresholds of 0.50, from `results/embeddings/`.

**Output:** `summary/tsne/tsne_raw_vs_trained.png`.

## Review response figures

**Script:** `fig34_step300.py`

Re-reads the few-shot figures at the fixed final checkpoint (step 300) instead
of the selected checkpoint, for the review response.

**Output:** review response figures, not tracked here.

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

## See also

- [Repository overview](../README.md)
- [Dataset creation](../hugging_face_dataset/v2/README.md)
- [Pelage visibility classifier](../pelage_sorting/README.md)
- [Role assignment and manuscript assets](../preprocessing/README.md)

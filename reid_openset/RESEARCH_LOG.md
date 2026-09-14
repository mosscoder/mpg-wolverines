# reid_openset research log

One entry per experiment, run order, kebab-case slugs. Five bullets each:
Objective, Methods, Positive, Negative, Follow-up. Keep entries under 250
words and shallow. Convention shared with the vole repo
(`local_drivers/RESEARCH_LOG.md` there is the canonical template).

The first two entries were written 2026-08-21 to 22, deleted with the
five-seed revert (db59393, 2026-08-24), and restored here 2026-09-14 from
git history, trimmed to what still holds.

## test-eval-reproducibility-probe

- **Objective**: Rerun the committed test_step_eval grid unchanged after
  the babel maintenance to export embeddings for the round-one response
  analyses and check run-to-run reproducibility of the reported metrics.
- **Methods**: Same code path, seed 0, no determinism flags, 9 GPU tasks
  (3 encoders by 3 gallery thresholds), 2026-08-21. The 27 rerun JSONs
  were compared against the committed values.
- **Positive**: Training reproduced almost exactly (losses match to the
  fourth decimal at every logged step, same tuned steps). The
  `--export-embeddings` path (7ce7426) works end to end and produced the
  22 tracked npz files in `{model}/results/embeddings/`.
- **Negative**: Recall@1 drifted despite identical config, standard
  deviation 0.027 across the 27 cells, worst cell moved 0.070. Cause is
  GPU float nondeterminism amplified by block-correlated test frames.
  Single-run cells carry roughly plus or minus 0.02 of invisible noise.
- **Follow-up**: [[five-seed-test-eval]]. The committed `results/test_eval/`
  numbers stayed canonical; the rerun JSONs lived only in session scratch
  and are gone.

## five-seed-test-eval

- **Objective**: Put an explicit error bar on every test-grid cell with
  across-seed replication, in place of a frame-level bootstrap for the
  editor's correlated-frames comment (E2).
- **Methods**: c2d4d47 (2026-08-21). test_step_eval gained deterministic
  kernels and seed-suffixed outputs, 45 tasks (3 encoders by 3 thresholds
  by seeds 0 to 4), smoke-gated on the dinov3 task 0.
- **Positive**: Absolute macro Recall@1 swung 0.716 to 0.790 across seeds,
  yet within-seed differences between filtered and unfiltered evaluations
  never exceeded 0.030 under any temporal thinning regime, so the E2
  no-inflation claim held independently of training noise.
- **Negative**: The Blackwell preempt nodes had no sm_120 kernels for
  PyTorch 2.7.1 and the sbatch reported COMPLETED on crashes (fixed in
  345dbcb and 641be52). Determinism holds per GPU model only.
- **Follow-up**: Reverted 2026-08-24 (db59393). The roadmap decided E2
  argues from the manuscript's reported results alone, so no event-matched
  control shipped and the seed-level code was removed. The only artifact,
  a test-crop EXIF timestamp CSV, was deleted 2026-09-14.

## tsne-best-conditions

- **Objective**: First Figure 7 candidate for R1.7, each encoder's
  embedding space at its best re-identification and best novelty
  detection filter combination.
- **Methods**: `make_tsne_figure.py` (74356c0, 2026-08-26), fit-on-all
  t-SNE per panel with Procrustes alignment, coordinates cached in
  `summary/tsne/tsne_coords.npz`.
- **Positive**: Established the palette, alignment, and marker
  conventions the final figure kept.
- **Negative**: Two rows of trained spaces could not show what training
  changed, which is the reviewer's actual question.
- **Follow-up**: Superseded by [[tsne-raw-vs-trained]]. Script, cache,
  and PNG removed 2026-09-14.

## tsne-raw-vs-trained

- **Objective**: Figure 7 for R1.7 and the mechanism cited in R1.2: the
  frozen feature space of each encoder before training against the
  ArcFace embedding after training at gallery filter 0.50.
- **Methods**: `make_tsne_raw_figure.py` (859c8e7, 2026-09-09). Row 1
  reads raw encoder features cached in `summary/tsne/raw_features/`
  (recomputed from the dataset if absent), row 2 reads the tracked
  embedding exports at the best balanced-accuracy step. Selection and
  coordinates tracked in `tsne_raw_meta.npz` and `tsne_raw_coords.npz`.
- **Positive**: Training separates the known individuals into compact
  clusters for DINOv3 with unknowns between them, partly for
  MegaDescriptor with unknowns at the cluster edges, and not for BioCLIP-2.
- **Negative**: Row 2 panels differ in how many unknowns they draw (the
  0.50 query filter leaves 78), which the caption must state.
- **Follow-up**: Asset pushed to Overleaf as `tsne_raw_vs_trained.png`;
  the tex block and caption are recorded under R1.7 in the roadmap.

## fig34-step300

- **Objective**: Show, for the R1.2 response, whether the non-monotonic
  Figure 4 curves survive when every configuration is read at the fixed
  final checkpoint (step 300) instead of the best cross-validated step.
- **Methods**: `fig34_step300.py`, first built in scratch 2026-09-09 and
  rebuilt in the repo 2026-09-14 after the scratch copy was wiped. Reads
  step 300 of `step_history` in the committed hygiene JSONs, manuscript
  layout and family-envelope convention, 95% t-intervals over 8 seeds.
  Writes `fig3_step300.png` and `fig4_step300.png` to
  `manuscript/review/external/figs/`.
- **Positive**: The dip-then-recover pattern (DINOv3 query filtering,
  MegaDescriptor none) flattens at step 300, supporting the letter's
  attribution of that pattern to checkpoint selection by recall at rank
  one.
- **Negative**: The rise-then-decline pattern (BioCLIP-2, and
  MegaDescriptor query filtering at 64) remains, so it is not a
  checkpoint artifact.
- **Follow-up**: Embed `fig4_step300.png` at the placeholder in the R1.2
  letter text.

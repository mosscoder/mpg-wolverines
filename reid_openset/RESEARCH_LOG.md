# reid_openset research log

One entry per experiment, run order, kebab-case slugs. Five bullets each:
Objective, Methods, Positive, Negative, Follow-up. Keep entries under 250
words and shallow. Convention shared with the vole repo
(`local_drivers/RESEARCH_LOG.md` there is the canonical template).

## test-eval-reproducibility-probe

- **Objective**: Rerun the committed test_step_eval grid unchanged (post
  babel maintenance) to export embeddings for the R1 response analyses and
  check run-to-run reproducibility of the reported metrics.
- **Methods**: Same code path, seed 0, no determinism flags. 9 GPU tasks
  (3 backbones x 3 gallery thresholds), 2026-08-21. Compared the 27
  rerun JSONs against the committed values (`git show HEAD:`), full
  comparison in scratchpad `fig5_delta.py`.
- **Positive**: Training itself reproduced almost exactly (losses match to
  the 4th decimal at every logged step, same tuned steps), confirming
  seeding and schedule are sound. Embedding npz exports work end to end.
- **Negative**: Recall@1 drifted despite identical config: sd 0.027
  across the 27 cells, 6 cells moved more than 0.04, worst -0.070
  (BioCLIP-2 g0.25 q0.25) and -0.053 (DINOv3 g0.00 q0.00). Cause is
  compounding GPU float nondeterminism amplified by block-correlated
  test frames (~48 events). Single-run cells carry invisible +/-0.02 sd
  noise, and any cross-cell contrast carries it twice.
- **Follow-up**: [[five-seed-test-eval]]. Rerun JSONs archived in session
  scratchpad (`rerun_20260821_jsons/`), working tree restored to the
  committed manuscript numbers.

## five-seed-test-eval

- **Objective**: Put an explicit error bar on every test-grid cell and
  make each replicate exactly re-runnable, replacing the E2 frame-level
  bootstrap (which assumes exchangeability of the very frames whose
  correlation E2 disputes) with across-seed replication.
- **Methods**: c2d4d47. test_step_eval gains deterministic kernels
  (cuBLAS workspace, cuDNN flags, `use_deterministic_algorithms`) and
  seed-suffixed outputs: JSONs in `results/test_eval_seeds/`, npz in
  `results/embeddings/` with `_seed={s}`. Arrays 0-14 per backbone
  (3 gallery thresholds x seeds 0-4), 45 tasks, smoke-gated on dinov3
  task 0. `results/test_eval/` and `summary/` deliberately untouched.
- **Positive**: The paired design works as intended. Absolute macro R@1
  swings 0.716 to 0.790 across seeds (matching the probe's noise
  estimate), yet within-seed differences between filtered and unfiltered
  evaluations never exceed 0.030 under any temporal regime (exact
  expectation for one-frame-per-event, 1,000 draws for spacing rungs; 20
  draws left ~0.02 Monte Carlo error that faked an event-vs-1000s gap),
  certifying the E2 no-inflation claim independently of training noise.
  Smoke cell matched the probe run to the 4th decimal.
- **Negative**: The new RTX_PRO_6000 (Blackwell, sm_120) preempt nodes
  fail twice over: PyTorch 2.7.1 has no sm_120 kernels, and the nodes
  arrived with empty or missing caches. Worse, test_step_eval's
  top-level except printed the traceback and exited 0, so every such
  crash reported COMPLETED (fixed: exit 1 in 345dbcb, sbatch rc
  propagation in 641be52, GPU-type constraint in 345dbcb). One
  fast-failing node black-holed most of a wave. Determinism holds per
  GPU model only; a preempted requeue restarts from step 0.
- **Follow-up**: E2 ladder and letter numbers via
  `manuscript/review/external/figs/make_e2_ablation.py` (seed-aware
  rework validated against the probe npz). Separate decision owed on
  adopting across-seed means for the manuscript test grid.

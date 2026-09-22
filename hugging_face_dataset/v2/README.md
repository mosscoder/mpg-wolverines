# Dataset creation

How the `mpg-ranch/wolverines` Hugging Face dataset was built from raw camera
trap images. Scripts live in `scripts/` and run in numbered order from the
repository root. Script 01 needs the MegaDetector environment
(`install_minimal_pytorchwildlife.sh` at the root); the rest use the
`wolverines` environment.

## 00. Field pre-screening

**Script:** `scripts/00_summarize_robust.py`
**Source:** the detections inventory spreadsheet (`data/Detections inventory_2023.08.29.xlsx`, not tracked)

Before automated processing, field biologists reviewed camera trap capture
events and recorded pelage visibility at the event level in the inventory's
`marks` field as Robust, Partial, or No:

- **Robust**: at least some images in the event show clear, identifiable pelage markings
- **Partial** and **No**: the markings were only partly visible or not visible, and the event does not proceed

| Metric | Count |
|--------|-------|
| Total camera trap events | 1,479 |
| Reviewed events | 1,385 |
| Unreviewed events | 94 |

Reviewed events by category:

| Category | Count | Percentage |
|----------|-------|------------|
| Robust | 588 | 42.5% |
| Partial | 489 | 35.3% |
| No | 308 | 22.2% |

The Robust designation applies to the camera trap event, not to individual images.
Robust events contain a mixture of high and low quality images, and that
mixture is what the pelage visibility classifier (stage 1) is trained on.
Only Robust events proceed to the pipeline.

**Output:** `data/robust_directories.json`, the event directories to crop.

## 01. MegaDetector cropping

**Script:** `scripts/01_run_megadetector.py`

MegaDetector v6 (PytorchWildlife) detects wolverines in each frame and crops
to the bounding box, filtering detections by confidence and saving crops with
trackable filenames. Bounding box geometry and confidence go to
`metadata/crop_metadata.csv` (one row per crop; the per-directory CSVs the
script writes along the way are merged into it and ignored by git).

Cropping to the bounding box reduces site-specific background (vegetation,
bait station structure, camera angle) so that downstream models learn from
the animal rather than from the scene.

## 02. Selecting images for annotation

**Script:** `scripts/02_prepare_labeling_data.py`

Images for human annotation are the **earliest camera trap events** for each
individual and color mode (day or night). The production classifier is
trained on these earliest events and applied to everything captured later, so
it never sees a "future" image during training.

Preparation: DINOv3 embeddings for every selected crop, K-means clusterings
(K from 2 to 64) to group visually similar images, and a SQLite database
(`data/labeling/pelage_labels.db`) holding embeddings, cluster assignments,
and labels.

## 03. Annotation interface

**Script:** `scripts/03_streamlit_labeling_app.py`

Images are displayed grouped by K-means cluster (the annotator picks K) so
that visually similar images are labeled together. Labels:

- **0 (None)**: pelage markings not visible
- **1 (Partial)**: some pelage markings visible
- **2 (Full)**: clear, complete pelage markings visible

A review mode allows relabeling. The first author produced every label; a
second-rater audit of the test set is reported in the manuscript.

## 04. Pelage dataset export

**Script:** `scripts/04_create_pelage_dataset.py`

Writes the `pelage` configuration: the annotated crops with a binary label
(Full versus None or Partial). Columns: `id`, `ymdh`, `color`, `label`,
`filename`, `confidence`, `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`,
`bbox_area`, `image`.

## 05. Pelage inference

**Script:** `scripts/05_infer_pelage.py`

Applies the production classifier (`pelage_sorting/`, stage 1) to every
out-of-sample crop, excluding all crops from the labeled events so that no
scored image was ever a training image. Output: `pelage_score` in [0, 1], the
probability that the pelage pattern is clearly visible, written to
`data/inference/pelage_inference_results.csv`.

## 07. Re-identification dataset export

**Script:** `scripts/07_create_reidentification_dataset.py`

Writes the `reidentification` configuration: every scored crop with its
individual identity and a temporal training and test split per individual (the most
recent 10% of each known individual's events are the test split). Columns:
`id`, `ymdh`, `color`, `pelage_score`, `data_source`, `filename`,
`megadetector_confidence`, `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`,
`bbox_area`, `image`. Per-individual split counts go to
`data/reidentification_dataset/individual_split_stats.csv`.

## 08 and 09. Provenance and capture rates

`scripts/08_reveal_provenance.py` writes `metadata/utilized_data_provenance.csv`,
one row per source frame with the task (pelage or re-identification) and split
it entered. `scripts/09_assess_quality_capture_rates.py` summarizes how many
high-quality images (score at or above 0.5) accumulate per individual per
event and per season, to `data/quality_capture_rates.json`.

`scripts/06_visualize_pelage_scores.py` drew the original submission's
score-by-decile grid and is superseded by
`preprocessing/make_quality_gradient_figure.py`.

## Event accounting

Not all 588 Robust events reach the re-identification dataset:

| Pipeline stage | Events | Images | Loss |
|----------------|--------|--------|------|
| Field pre-screening (Robust) | 588 | | |
| MegaDetector crops | 578 | 52,013 | 10 events with missing directories or no detections |
| Pelage classifier training | 20 | 2,701 | held out for training |
| Pelage inference and re-identification | 558 | 49,312 | |

**Non-overlap guarantee.** The 20 camera trap events used to train the pelage
visibility classifier share no image with the 558 events in the
re-identification dataset. Script 05 enforces this by excluding every crop
from a labeled event before inference. The 20 training events (two per
individual where available, from the earliest capture dates), keyed by the
minute of the first frame, are:

| Individual | Event 1 | Event 2 |
|------------|---------|---------|
| BDF10-M6 | 201602051910 | 201602091445 |
| HFW12-F7 | 201601301538 | 201602110139 |
| HLC20-H3 | 202003282359 | 202003291734 |
| HLC21-H1 | 202101141026 | |
| Turk | 202201200551 | 202201210934 |
| Tex | 202301041114 | 202301132101 |
| LH23-M1 | 202302152037 | 202302160805 |
| PA23-M1 | 202302182132 | |
| PA23-M2 | 202302251120 | 202302260158 |
| PA23-F1 | 202303072120 | 202303081456 |
| Powder Paws | 202303081238 | 202303172209 |

How the 558 inference events are assigned to gallery, validation, and unknown
roles is documented in [preprocessing/README.md](../../preprocessing/README.md).

## See also

- [Repository overview](../../README.md)
- [Pelage visibility classifier](../../pelage_sorting/README.md)
- [Role assignment and manuscript assets](../../preprocessing/README.md)
- [Re-identification and novelty detection](../../reid_openset/README.md)

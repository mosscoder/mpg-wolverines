# Wolverine Individual Re-identification Using Camera Trap Imagery

## Overview

Open-set re-identification of wolverines (*Gulo gulo*) from camera trap images using pelage (underbelly marking) patterns. This project implements a two-stage deep learning pipeline:

1. **Pelage Quality Classification**: Filter images where pelage markings are clearly visible
2. **Individual Re-identification**: Match individuals using metric learning in an open-set context

**Key Features:**
- MegaDetector v6 for automated wolverine detection and cropping
- Human-annotated pelage visibility labels with model-based scoring
- DINOv3-ViT-B/16 frozen backbone with task-specific heads
- ArcFace metric learning for individual embeddings
- Balanced Accuracy evaluation for fair open-set assessment

---

## Methods

### 1. Image Acquisition

Camera trap network with elevated bait stations designed to capture wolverine underbelly (pelage) markings as individuals climb to access attractants.

**Data source:** HuggingFace dataset `kdoherty/wolverines`

### 2. Field Pre-Screening

**Script:** `hugging_face_dataset/v2/scripts/00_summarize_robust.py`

**Source:** Detection inventory database (`Detections inventory_2023.08.29.xlsx`)

Before automated processing, field biologists reviewed camera trap capture events and categorized pelage visibility at the event level:
- **Robust**: At least some images in the capture event show clear, identifiable pelage markings
- **Not Robust**: No images in the event show sufficient pelage markings for identification

**Capture Event Statistics:**

| Metric | Count | Percentage |
|--------|-------|------------|
| Total capture events | 1,479 | — |
| Reviewed events | 1,385 | — |
| Unreviewed events | 94 | — |

*Breakdown by category (reviewed events only):*

| Category | Count | Percentage |
|----------|-------|------------|
| Robust | 588 | 42.5% |
| Partial | 489 | 35.3% |
| No | 308 | 22.2% |

**Important:** The "Robust" designation applies at the capture event level, not individual images. Events flagged as Robust contain a mixture of high and low quality pelage images. This heterogeneity is intentional — non-robust images from Robust events still enter the pipeline and are used to train the pelage quality classifier (Stage 1).

**Filtering:** Only capture events designated as "Robust" proceed to the automated pipeline. Events marked "Not Robust" are excluded entirely.

**Output:** `hugging_face_dataset/v2/data/robust_directories.json` — directories from Robust-flagged events ready for MegaDetector processing.

### 3. Automated Detection (MegaDetector)

**Script:** `hugging_face_dataset/v2/scripts/01_run_megadetector.py`

MegaDetector v6 (PytorchWildlife) detects wolverines in raw camera trap images and crops to bounding boxes:
- Filters detections by confidence threshold
- Saves crops with trackable filenames
- Records bounding box coordinates (x, y, width, height) and confidence scores

**Methodological note:** Cropping to bounding boxes removes site-specific background context (vegetation, bait station structure, camera angle). This forces downstream models to learn discriminative features from the wolverines themselves rather than exploiting spurious correlations with camera trap locations.

### 4. Pelage Quality Annotation

#### 4.1 Image Selection for Annotation

**Script:** `hugging_face_dataset/v2/scripts/02_prepare_labeling_data.py`

**Selection Strategy:**
Images selected for human annotation are filtered to the **earliest capture date** for each Individual x Color (day/night) combination:
- Ensures temporal independence: annotated images come from first encounters
- Production model is trained on earliest dates, then applied to all subsequent captures
- Prevents data leakage: model never sees "future" images during training

**Preparation:**
- DINOv3 embeddings generated for all selected crops
- K-means clustering (K=2 to 64) precomputed to group visually similar images
- SQLite database stores embeddings, cluster assignments, and labels

#### 4.2 Human Annotation Interface

**Script:** `hugging_face_dataset/v2/scripts/03_streamlit_labeling_app.py`

**Annotation Protocol:**
- Images displayed grouped by K-means clusters (annotator selects K value)
- Cluster-based workflow enables efficient batch labeling of similar images
- Annotators assign pelage visibility labels:
  - **0 (None)**: Pelage markings not visible
  - **1 (Partial)**: Some pelage markings visible
  - **2 (Full)**: Clear, complete pelage markings visible
- Labels assigned at ID x color x date group level for temporal consistency
- Review mode allows relabeling of previously annotated images

#### 4.3 Pelage Score Prediction

**Script:** `hugging_face_dataset/v2/scripts/05_infer_pelage.py`

**Training:**
- Binary classifier trained on human annotations (Full vs None/Partial)
- Model: DINOv3-ViT-B/16 (frozen) + linear classifier head
- Training set: Human-annotated crops from earliest capture dates

**Inference:**
- Production model applied to **all out-of-sample crops** (images NOT used in training)
- Excludes the earliest-date annotated images to prevent label leakage
- Output: `pelage_score` in [0,1] — probability that image shows clear pelage markings
- Final dataset includes both human-annotated samples (with original labels) and model-scored samples

### 5. Dataset Creation

**Scripts:**
- `hugging_face_dataset/v2/scripts/04_create_pelage_dataset.py` - Binary classification config
- `hugging_face_dataset/v2/scripts/07_create_reidentification_dataset.py` - Re-identification config

**Dataset Features:**

| Feature | Description |
|---------|-------------|
| `image` | Cropped wolverine image (PIL) |
| `id` | Individual identifier |
| `label` | Binary pelage visibility (0/1) |
| `pelage_score` | Model-predicted quality [0.0-1.0] |
| `megadetector_confidence` | Detection confidence |
| `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height` | Bounding box |
| `ymdh` | Temporal identifier (year-month-day-hour) |
| `station` | Camera trap location |

### 6. Pelage Quality Classification (Stage 1)

**Location:** `pelage_sorting/`

Binary classification to filter images suitable for re-identification.

**Architecture:**
- Backbone: DINOv3-ViT-B/16 (frozen, pretrained)
- Classifier: Linear layer (768 -> 2 classes)
- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)

**Training:**
- 5-fold stratified cross-validation
- Optimal epochs determined by validation F1

### 7. Individual Re-identification (Stage 2)

**Location:** `reid_openset_BA/`

Open-set metric learning for individual identification.

**Architecture:**
- Backbone: DINOv3-ViT-B/16 (frozen)
- Embedding Head: Linear (768 -> 128 dims, L2-normalized)
- Loss: ArcFace (margin=0.5, scale=64)
- Optimizer: AdamW (lr=0.0005, epochs=50)
- Batch Sampling: P-K (P=5 individuals, K=8 samples)

**Experimental Design:**
- Quality thresholds: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- Gallery sizes: [2, 4, 8, 16, 32, 64] examples per individual
- Seeds: 8 replicates per configuration
- Total: 288 configurations

### 8. Open-Set Evaluation Framework

**Task Separation:**
The system performs two distinct tasks measured separately:
1. **Detection**: Is query from a known or unknown individual?
2. **Identification**: If known, which individual is it?

**Metric Definitions:**

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Known Accept Rate (KAR)** | P(max_score >= threshold \| known) | System correctly accepts known individuals |
| **Unknown Reject Rate (URR)** | P(max_score < threshold \| unknown) | System correctly rejects unknown individuals |
| **Balanced Accuracy (BA)** | (KAR + URR) / 2 | Overall detection performance |
| **Recall@1** | P(top match correct \| query) | Identification accuracy |

**Why Balanced Accuracy over F1-Score:**

F1-Score has a fundamental problem for macro-averaging in open-set recognition:
- Recall = TP / (TP + FN) — comes from **unknown** samples only
- Precision = TP / (TP + FP) — mixes **unknown** (TP) and **known** (FP) samples

When macro-averaging F1 (compute per-individual, then average), precision uses a shared FP count from known individuals. This creates systematic bias against individuals with fewer samples:

| Individual | Samples | Detection Rate | TP | FP (shared) | Precision | F1 |
|------------|---------|----------------|-----|-------------|-----------|-----|
| A (many samples) | 100 | 80% | 80 | 10 | 80/90=0.89 | 0.84 |
| B (few samples) | 10 | 80% | 8 | 10 | 8/18=0.44 | 0.57 |

Same detection rate, but B is penalized because TP_B is small relative to shared FP.

Balanced Accuracy cleanly separates populations:
- Each individual contributes equally regardless of sample count
- No mixing of populations in either component
- Natural decomposition into two interpretable rates

**Threshold Calibration (Leave-One-Out on Gallery):**
1. Hold out individual H as simulated "unknown"
2. Compute H's rejection rate + remaining individuals' accept rate
3. BA = (rejection_rate + accept_rate) / 2
4. Find threshold maximizing BA
5. Final threshold = mean across all folds

### 9. Temporal Validation

Validation splits use most recent captures per individual to:
- Prevent temporal data leakage (same-day images in train/val)
- Ensure quality-stratified representation
- Test generalization to future encounters

---

## Reproducibility

- All random seeds tracked and reported in results JSON
- HuggingFace dataset versioned
- SLURM array jobs with checkpoint/resume for preemption handling

---

## Supplementary Materials

### S1. Computational Optimization

Dataset loading optimized using Apache Arrow columnar access.

**Performance Improvement:**

| Operation | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Metadata extraction | 100s | 0.03s | 3,125x |
| Quality filtering | 1s | 0.0001s | 10,000x |
| **Total per config** | **203s** | **0.75s** | **271x** |

**Full Experiment Impact (288 configurations):**
- Baseline: 8.1 hours
- Optimized: 3.4 seconds
- Speedup: 8,565x

**Technique:** Replace row-by-row dataset iteration with columnar extraction:

```python
# Slow: n random access operations
for idx in range(len(dataset)):
    quality = dataset[idx]['pelage_score']

# Fast: single columnar operation
quality_scores = np.array(dataset['pelage_score'], dtype=np.float32)
```

HuggingFace datasets use Apache Arrow, which stores data in columnar format:
- Row-based: Read row -> Extract field -> Repeat 44,201 times (cache misses, object allocations)
- Column-based: Read entire column in one sequential scan (zero-copy to numpy)

**Implementation:** `utils/arrow_cache.py`, `utils/optimized_filters.py`

### S2. Detailed Metric Justification

See `reid_openset_BA/METRICS.md` for comprehensive discussion of:
- Why F1 macro-averaging mixes populations unfairly
- Mathematical formulation of Balanced Accuracy components
- Separation of detection task from identification task
- Implementation details for LOO threshold calibration

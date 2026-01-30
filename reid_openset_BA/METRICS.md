# Open-Set Metrics: Why Balanced Accuracy over F1-Score

## Definitions

### System Actions

| Action | Meaning | Condition |
|--------|---------|-----------|
| **Accept** | System declares: "This query is from someone in the gallery" | `max_score >= threshold` |
| **Reject** | System declares: "This query is NOT from anyone in the gallery" | `max_score < threshold` |

### Ground Truth

| Term | Meaning |
|------|---------|
| **Known** | Query is from an individual who IS in the gallery |
| **Unknown** | Query is from an individual who is NOT in the gallery |

### Metrics

| Metric | Definition | Ideal |
|--------|------------|-------|
| **Known Accept Rate (KAR)** | Proportion of **known** queries that are **accepted** | 1.0 |
| **Unknown Reject Rate (URR)** | Proportion of **unknown** queries that are **rejected** | 1.0 |
| **Balanced Accuracy (BA)** | (KAR + URR) / 2 | 1.0 |
| **Recall@1** | Proportion of queries where top match is correct identity | 1.0 |

### Interpretation

- **High URR**: System correctly flags most unknowns as "not in gallery"
- **High KAR**: System correctly flags most knowns as "in gallery"
- **High Recall@1**: When system makes a match, it's usually the right person

---

## The Problem with F1-Score for Open-Set Recognition

In open-set recognition, we want to:
1. Correctly identify known individuals (closed-set)
2. Correctly flag unknown individuals as "not in gallery" (open-set)

F1-Score seems natural for the binary "unknown detection" task where Positive = Unknown:
- **TP**: Unknown correctly flagged
- **FP**: Known incorrectly flagged as unknown
- **FN**: Unknown missed (accepted as known)

However, F1 has a fundamental problem for macro-averaging.

### The Macro-Averaging Problem

**F1 = 2 * Precision * Recall / (Precision + Recall)**

Where:
- Recall = TP / (TP + FN) — comes from **unknown** samples only
- Precision = TP / (TP + FP) — mixes **unknown** (TP) and **known** (FP) samples

When we try to macro-average F1 (compute per-individual, then average):

**Recall per unknown individual U:**
- Recall_U = TP_U / (TP_U + FN_U) ✓ Clean, per-individual

**Precision per unknown individual U:**
- Precision_U = TP_U / (TP_U + FP)
- But FP is **shared** — it comes from known individuals, not U!
- FP is constant regardless of which unknown individual we're considering

This creates bias:

| Individual | Samples | Detection Rate | TP | FP (shared) | Precision | F1 |
|------------|---------|----------------|-----|-------------|-----------|-----|
| A (many samples) | 100 | 80% | 80 | 10 | 80/90=0.89 | 0.84 |
| B (few samples) | 10 | 80% | 8 | 10 | 8/18=0.44 | 0.57 |

Same detection rate, but B is penalized because TP_B is small relative to shared FP.

Similarly, if one known individual has many more samples than others, they dominate the FP count.

## The Solution: Balanced Accuracy

**BA = (unknown_reject_rate + known_accept_rate) / 2**

Both components can be cleanly macro-averaged:

### Unknown Reject Rate (macro-averaged)
For each unknown individual U:
- U_reject_rate = (U's samples below threshold) / (U's total samples)

Macro = mean across all U's

### Known Accept Rate (macro-averaged)
For each known individual K:
- K_accept_rate = (K's samples with max_score >= threshold) / (K's total samples)

Macro = mean across all K's

### Why This Works
- Each individual contributes equally regardless of sample count
- No mixing of populations in either component
- Natural decomposition into two interpretable rates

## Separation of Detection from Identification

**Design Principle: Balanced Accuracy measures detection, not identification.**

The system performs two distinct tasks:
1. **Detection**: Is this query from a known or unknown individual?
2. **Identification**: Given it's known, which individual is it?

These should be measured separately:

| Metric | Task | Condition |
|--------|------|-----------|
| **Known Accept Rate** | Detection | `max_score >= threshold` |
| **Unknown Reject Rate** | Detection | `max_score < threshold` |
| **Recall@1** | Identification | `argmax == true_label` |

### Why Separate?

If Known Accept Rate required both `score >= threshold` AND `correct match`, misidentification would be double-penalized:

1. Recall@1 penalizes misidentification (correct)
2. Known Accept Rate ALSO penalizes misidentification (redundant)

This conflation makes metrics harder to interpret. A low BA could mean:
- Poor threshold calibration (detection problem), OR
- Poor embedding quality (identification problem)

With separation, the metrics have clear interpretations:
- **Low BA, high Recall@1**: Threshold too aggressive (rejecting known individuals)
- **High BA, low Recall@1**: Good detection but poor discrimination between individuals
- **Low BA, low Recall@1**: Both systems need improvement

## Implementation

### LOO Threshold Calibration (on gallery)

For each held-out individual H:
1. H is "unknown" — compute H's rejection rate
2. Other individuals are "known" — compute macro known_accept_rate
3. BA = (H_rejection + macro_known_accept) / 2
4. Find threshold maximizing BA

Final threshold = mean across folds

### Open-Set Validation

Using calibrated threshold:
1. Unknown val samples → macro unknown_reject_rate
2. Known val samples → macro known_accept_rate
3. BA = average of the two rates

## Summary

| Metric | Macro-Averaging | Sample Count Bias |
|--------|-----------------|-------------------|
| F1-Score | Problematic (precision mixes populations) | Yes (unfair to small individuals) |
| Balanced Accuracy | Clean (each rate is independent) | No (equal weight per individual) |

For open-set recognition with imbalanced sample counts per individual, **Balanced Accuracy with macro-averaging** is the appropriate metric.

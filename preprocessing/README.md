# Role assignment and manuscript assets

Scripts that decide which individuals serve as known (gallery) individuals,
which serve as simulated unknowns, and which events are validation queries,
plus two generators for manuscript assets. Run from the repository root in the
`wolverines` environment; they read the local copy of the re-identification
dataset.

## Scripts

**`assign_reid_individuals.py`** decides which individuals are known. An
individual qualifies if it has at least 64 training images with a quality
score above 0.5 and at least one test image above 0.5. For each known
individual, the most recent training events are set aside as validation
queries until every quality score range ([0.75, 1], [0.5, 0.75), [0.25, 0.5),
and [0, 0.25)) is represented. Output goes to
`results/feasible_individuals.json`.

**`assign_reid_unknowns_all_test.py`** makes the individuals that do not
qualify into unknown individuals. It identifies their images in the upstream
training split so that all of an unknown individual's images serve as test
queries and none enter training. Output goes to
`results/unknown_assignment.json`.

**`make_table_individuals.py`** writes manuscript Table 2, giving images,
events, and day spans per individual and split, with the gap from the last
training or validation event to the first test event. Output goes to
`results/table_individuals.tex`.

**`make_quality_gradient_figure.py`** draws manuscript Figure 2, ten images
spanning the quality score range from one camera trap event for each of the
three best-sampled individuals. Output goes to `results/quality_gradient.png`.

## Roles

The 558 inference events (49,312 images) partition by identity and sample
sufficiency. Known individuals populate the gallery, validation queries, and
known test queries; simulated unknowns are never seen in training and appear
only as test queries for novelty detection.

| Role | Individual | Events | Images |
|------|-----------|--------|--------|
| Known | Turk | 164 | 12,454 |
| Known | HLC20-H3 | 148 | 13,515 |
| Known | BDF10-M6 | 82 | 5,157 |
| Known | LH23-M1 | 52 | 2,475 |
| Known | HFW12-F7 | 48 | 12,815 |
| Known | Tex | 11 | 343 |
| **Subtotal known** | **6 individuals** | **505** | **46,759** |
| Simulated unknown | PA23-F1 | 40 | 1,497 |
| Simulated unknown | PA23-M2 | 4 | 556 |
| Simulated unknown | Powder Paws | 6 | 418 |
| Simulated unknown | PA23-M1 | 2 | 71 |
| Excluded | HLC21-H1 | 1 | 11 |
| **Subtotal unknown and excluded** | **5 individuals** | **53** | **2,553** |

HLC21-H1's single remaining event yields 11 images, two above 0.25 and one
above 0.5, too few for per-individual novelty evaluation under the query
filters, so it is excluded. Per-split image, event, and span counts for every
individual are in `results/table_individuals.tex`.

## Temporal validation

Splits are ordered in time within each individual. The test split is the most
recent 10% of a known individual's events, held out whole, and the validation
queries are the most recent events before that. No image of an individual in
a later split predates an image of the same individual in an earlier one, so
the evaluation measures generalization to visits the models have not seen.

## See also

- [Repository overview](../README.md)
- [Dataset creation](../hugging_face_dataset/v2/README.md)
- [Pelage visibility classifier](../pelage_sorting/README.md)
- [Re-identification and novelty detection](../reid_openset/README.md)

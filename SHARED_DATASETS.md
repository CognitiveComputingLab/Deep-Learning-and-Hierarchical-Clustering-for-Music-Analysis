# Shared structural annotation format

This project uses a versioned, loss-aware JSON representation. It keeps
explicit hierarchies, flat partitions, overlapping spans, and point events as
different annotation kinds so converters do not invent tree structure.

Implementation:

- `src/shared_annotation_format.py`: validation and JSON I/O;
- `shared_format/schema_v1.json`: machine-readable schema;
- `src/dataset_converters.py`: source-specific conversion;
- `src/shared_evaluation.py`: safe evaluation adapters.

## Reproduce

```powershell
git submodule update --init --recursive
python scripts\convert_shared_datasets.py
```

Outputs under `data/shared_annotations_v1/` include one JSON per piece,
`manifest.csv`, `dataset_status.csv`, `conversion_errors.csv`, and
`summary.json`. Use `--quick` for one item per dataset.

Score-aligned Winterreise smoke test:

```powershell
python scripts\eval_greedy.py --data-root external\schubert_winterreise --piece n01 --quick --output-dir results\shared_format_smoke\winterreise_greedy
python scripts\evaluate_dp_stage.py --data-root external\schubert_winterreise --piece n01 --quick --output-dir results\shared_format_smoke\winterreise_dp
```

## Stage 2: Taking Form score alignment

Official DCML Beethoven and Mozart piano-sonata scores provide printed measure
numbers, time signatures, note events, and absolute quarterbeats. Align the
converted form hierarchies with:

```powershell
python scripts\align_taking_form_scores.py
```

The current audit aligns 89/120 movements: all 17 Mozart first movements and
72 Beethoven movements. Twelve Beethoven movements have no score in the
current DCML snapshot; 19 are rejected because repeated/ambiguous printed
measure numbers, different unfolding, or edition-length differences prevent a
unique mapping. These are not filled using nearest-measure guesses.

The aligner uses DCML printed measure number `mn`, not internal measure count
`mc`. Positive beats use `(beat-1)*beat_unit`; negative beats occur before
the printed downbeat. Simple metres use the denominator unit and compound
metres a dotted beat. Opening negative beats that overshoot an exactly encoded
anacrusis can snap only to score start; each is counted in the manifest.

Run the explicit-hierarchy evaluator with:

```powershell
python scripts\evaluate_shared_hierarchy_stage.py --dataset mozart --output-dir results\shared_hierarchy_stage_mozart_tree
```

The older evaluator also reports cumulative form-boundary levels and
validation-selected tree depths. loowcv_summary.csv is retained as a legacy
sensitivity analysis, not the primary tree metric. Flat-tree TED is omitted
because Taking Form has a genuine hierarchy.

The primary genuine-tree outputs are tree_measure_per_piece.csv and
tree_measure_summary.csv. They report reduced and full T-measures following
McFee, Nieto, and Bello (ISMIR 2015): LCA rank agreement is evaluated at the
ordered input-bin centres, so the two trees may have unequal depth.
equal_budget_summary.csv is a secondary per-level Boundary F1 result. The
evaluator also writes work-blocked bootstrap intervals and paired
DP-minus-Greedy permutation tests.

All 17 Mozart movements (17 works) and all 72 aligned Beethoven movements (27
works) completed. Greedy had higher mean full T-measure for every tested
distance on both corpora even though DP always had an equal or better additive
objective. This demonstrates objective/reference mismatch, not an error in
exact DP. Do not pool these results with ABC local-key scores because the
reference semantics differ.

## Current audit

The local snapshot converts 269/269 discovered files with zero failures:

| Dataset | Pieces | Semantics | Current use |
|---|---:|---|---|
| ABC/DCML | 70 | flat local-key partition and events | Direct Greedy/DP and Boundary F1 |
| Taking Form Beethoven | 103 | explicit form hierarchy | Tree reference; score alignment needed |
| Taking Form Mozart | 17 | explicit form hierarchy | Tree reference; score alignment needed |
| Algomus Bach fugues | 23 local files | overlapping motifs and events | Overlap reference; score needed |
| Algomus Mozart quartet movements | 32 | form hierarchy and events | Tree reference; score needed |
| Schubert Winterreise | 24 | flat local-key partition and events | Direct Greedy/DP and Boundary F1 |

There are 152 pieces with an explicit hierarchy, and 94 are immediately
compatible with the current pitch-class clustering pipeline. Annotation-only
conversion is never reported as algorithm-ready.

## Semantics and validity

ABC `localkey` is a flat partition, not an expert tree. Pedals are spans and
cadences/phrase ends are points. Taking Form parent links come only from its
documented coarse-to-fine columns; repeat shorthand is not expanded without a
score. Two non-monotonic measure typos receive deterministic recorded repairs,
and `X (Cadenza)` is preserved as a source point marker.

Dezrann time is interpreted as quarter notes from piece start. Fugue subjects
and countersubjects may overlap and are never forced into a tree. Only
explicitly numbered Mozart `top.*` layers become hierarchy nodes; tonality
remains a flat partition.

The Winterreise adapter accepts official DCML `notes/*.notes.tsv` and
`harmonies/*.harmonies.tsv` through `--winterreise-root`. Other multimodal
segmentations need a separate profile so alternatives are not collapsed.

Kirlin and Yust (2018) is retained as a literature lead, not counted as a
converted corpus: the tutor note did not specify a public data location or
exchange format. Schenkerian dependency graphs would also require a
graph-aware schema rather than being forced into ordered segmentation trees.

Before applying Greedy/DP to measure-beat sources, pair the exact score,
expand repeats consistently, and map positions to absolute quarterbeats.
Overlapping motives require overlap-aware metrics, not ordinary TED.

- Never infer hierarchy from local key, cadence, pedal, or motif presence.
- Never treat missing score alignment as zero accuracy.
- Preserve source labels, coordinates, and repair provenance.
- Split training/evaluation by complete work.
- Report boundary, hierarchy, span, and point-event tasks separately.

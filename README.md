# Deep Learning and Hierarchical Clustering for Music Analysis

This repository evaluates temporally contiguous greedy hierarchical clustering on music data. The current baseline stage compares hand-crafted pitch-class distances before dynamic programming or learned distances are introduced.

## Research questions

1. Do music-informed distance functions change greedy hierarchical clustering outcomes?
2. Which distance best recovers DCML local-key boundaries?
3. Are conclusions stable across input bin size, boundary tolerance, evaluation depth, and TED time-label bins?

## Evaluation design

DCML local-key annotations are segment boundaries, not hierarchical ground truth. Boundary precision, recall, and F1 are therefore the primary metrics. Order-preserving dynamic programming first maximises one-to-one matches and then minimises timing error. The primary pre-registered configuration is 8 quarterbeat bins, 8 quarterbeat tolerance, and depth 4; fixed prediction-budget results separate tree shape from boundary-count effects.

TED is auxiliary. It measures discrepancy between a predicted hierarchy and a **flat tree induced by DCML local-key segments**. It must not be interpreted as distance from an expert hierarchical tree. Node-count-scaled TED is not bounded by one; raw, pruned, node-count, leaf-count, pruning-depth, and label-bin sensitivity diagnostics are emitted together.

Distances are Euclidean, tonic-relative weighted chromagram, a two-dimensional circle-of-fifths embedding inspired by Tonnetz geometry, and 24-dimensional key-profile activation. The fixed C-major weighting is available only as an explicitly named ablation. Balanced and random adjacent-merge trees are structural baselines.

## Reproduce

```powershell
python scripts\eval_greedy.py
```

This discovers all paired files under `external/ABC/notes` and `external/ABC/harmonies`, uses 100 seeded random-baseline repetitions, and writes CSV, JSON, and plots to `results/greedy_eval`.

Useful development runs:

```powershell
python scripts\eval_greedy.py --piece n11op95_01 --quick
python -m pytest -q
```

Use `--help` for piece/method filters and every parameter grid. Add `--include-ablation` to include the fixed-C weighting.

## Outputs and interpretation

- `boundary_per_piece.csv`: primary per-piece parameter-grid results with TP, FP, FN, precision, recall, and F1.
- `boundary_macro_summary.csv`: work-level macro mean, standard deviation, and 16-quartet bootstrap 95% confidence intervals.
- `boundary_movement_descriptive.csv`: descriptive summaries across 70 movements.
- `boundary_fixed_budget_per_piece.csv`: equal prediction-budget and explicitly labelled oracle-budget diagnostics.
- `boundary_fixed_budget_summary.csv`: work-level fixed-budget confidence intervals.
- `boundary_micro_summary.csv`: aggregate-count micro metrics.
- `primary_paired_tests.csv`: work-blocked paired sign-flip permutation tests with Holm correction.
- `ted_auxiliary_per_piece.csv`: auxiliary flat-tree TED and all required size/pruning diagnostics.
- `metadata.json` and `run_status.csv`: definitions, seed, configuration, and success/failure audit trail.

Precision-recall plots contain tolerance/depth operating points; they are not confidence-threshold PR curves. Generated results are ignored by git and are reproducible from the command above.

## Validity limits

Local-key change is not equivalent to phrase, cadence, section, or complete musical-form hierarchy. Large tolerances can overstate boundary recovery; pruning aligns approximate scale rather than semantic level. Pitch-class centroids lose distributional information, and global-key estimation can be uncertain. Corpus-level uncertainty and parameter sensitivity must accompany method comparisons.

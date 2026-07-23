# Greedy baseline evaluation notes

## Current interpretation

DCML `localkey` changes provide boundary annotations only. The previous Op.95 analysis compared a binary predicted tree with a one-level tree induced from six local-key segments. Its TED values therefore reflected node-count and depth mismatch as well as temporal structure; they are not evidence of agreement with an expert hierarchy.

Boundary F1 is now the primary outcome. TED remains an auxiliary diagnostic and is always accompanied by normalisation, node counts, pruned leaf count, selected pruning depth, and center/width label-bin sensitivity.

## Distance terminology

| Category | Implementation |
|---|---|
| Baseline | Euclidean distance on L1-normalised chromagrams |
| Music-informed | Global-tonic-relative KS-weighted chromagram distance |
| Geometric | Two-dimensional circle-of-fifths embedding inspired by Tonnetz geometry |
| Key-aware | Distance between 24 KS key-profile activations |
| Ablation only | Fixed absolute C-major KS weighting |

The old `weighted` implementation used the KS C-major profile as absolute pitch-class weights. It is retained only under the explicit name `fixed_c_major_weighted_ablation`. The main weighted method estimates a single global tonic/mode from whole-piece notes and rotates the KS profile; it does not use DCML `globalkey`.

## Op.95 status

Earlier figures at tolerance 24 qb and depth 4 are exploratory and should not support general claims. The reproducible evaluation now includes tolerance 2, 4, 8, 12, and 24 qb; depths 1 through 6; bin sizes 2, 4, 8, and 16 qb; and nine TED label-bin settings. Op.95 remains a case study while corpus summaries determine the reported comparison.

## Reporting rules

- Report movement-level descriptive results, work-level macro statistics, 16-quartet bootstrap 95% CI, and aggregate-count micro metrics.
- Use the pre-specified 8 qb bin, 8 qb tolerance, depth 4 configuration for work-blocked paired method tests; apply Holm correction.
- Match boundaries with ordered dynamic programming: maximise TP before minimising total timing error.
- Report equal prediction-budget F1 alongside depth-based F1; label GT-count matching as oracle diagnostic.
- Include balanced and 100-repeat seeded random adjacent-merge structural baselines.
- Treat tolerance/depth precision-recall points as operating points, not confidence-threshold curves.
- Do not infer that a pruned tree represents the same semantic depth as the DCML segmentation.
- Do not generalise a single-piece observation to greedy clustering or musical structure overall.

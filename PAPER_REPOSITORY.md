# Paper repository scope

This repository contains only files needed to reproduce, audit, test or explain
the dissertation's ABC/DCML experiments.

## Retained method code

- Basic: adjacent bottom-up Greedy clustering and handcrafted distances.
- Intermediate: exact ordered dynamic programming, boundary-aware objectives,
  and calibrated mixture/diagonal Mahalanobis distance training.
- Advanced: Harmonic CNN + Temporal TCN Siamese distance learning, evaluated
  with both Greedy and exact DP.
- Pitch Scape generation for the three music-informed DP distances.

## Retained evidence

- `results/corpus_smoke`: complete 16-work Basic evaluation.
- `results/dp_stage_corpus_smoke`: complete 16-work legacy DP comparison.
- `results/optimized_stage_keyprofile_final`: complete 16-work optimized
  handcrafted Greedy/DP comparison.
- `results/dp_stage`: DP diagnostics and Pitch Scape figures.

These historical results remain diagnostics. The final dissertation table must
come from `results/dissertation_main/final_evidence` and is reportable only when
`final_audit.json` says `ready_for_dissertation: true`.

## Excluded material

Old hard-coded Op.95 scripts, corrupted-encoding visualizers, editor temporary
files, duplicated quick/debug outputs and Python caches were moved outside the
repository to the recoverable backup:

`C:/Users/admin/Desktop/Project_code/paper_cleanup_backup_20260830_170000`

The Taking Form and Algomus source corpora were restored because they are now
part of the tutor-requested shared-format study. Their production converters,
schema, validation tests, manifests, and source semantics are documented in
`SHARED_DATASETS.md`. They remain a separate cross-corpus evaluation scope
and are not mixed into the ABC-only headline table without score alignment.

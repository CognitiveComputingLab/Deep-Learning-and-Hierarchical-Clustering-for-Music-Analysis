# Review and integration of `0.zip`

## What the archive contains

The archive is a later, self-contained experiment snapshot rather than a small
patch. It contains alternative common modules, an aggregate-chroma neural
stack, an actor-critic/beam-search RL extension, external-corpus loaders,
experiment orchestration, frozen result tables, checkpoints and a final
evidence audit. Archive files were treated as review material, not as trusted
instructions or automatically valid current results.

Its strongest methodological contribution is experiment governance:
deterministic data manifests, frozen test access, grouped work splits, command
and environment capture, explicit result auditing, and honest negative-result
reporting. Its neural encoder is not a suitable replacement for the current
main model because it aggregates an interval to one 12-dimensional vector and
therefore loses temporal order before encoding.

## What was integrated

- `src/data_manifest.py` and `scripts/prepare_dissertation_data.py`: adapted to
  the project's actual paired ABC/DCML scope, recording incomplete pairs and
  SHA-256 hashes rather than silently dropping them.
- `scripts/run_dissertation_experiments.py`: adapted to the current Greedy,
  exact-DP and Harmonic-CNN+Temporal-TCN entry points. It records environment,
  commands, status, runtime and supports safe resume.
- `scripts/audit_dissertation_results.py`: a read-only frozen-evidence audit
  adapted to the current output schemas. It generates the direct
  Basic/Intermediate/Advanced comparison only after checking a common 16-work
  scope, grouped-fold leakage, run success and exact-DP objective dominance.
  The primary rows all come from the same nested evaluator and threshold
  protocol; incompatible fixed-depth outputs are exported only as diagnostics.
- Unit tests for pair discovery/checksums, comparable-scope auditing and
  rejection of a false DP optimality claim.

No existing clustering or learned-distance implementation was overwritten.

## What was deliberately not merged into the main method

- The archive's aggregate 12-D Siamese encoder: the current masked
  `(time, pitch-class)` TCN retains chronological information and is the more
  defensible Advanced model.
- Common core files with conflicting Greedy/DP APIs: wholesale replacement
  would invalidate existing tests and make old/new results incomparable.
- Binary checkpoints and result CSVs: these have archive provenance and must
  not be presented as outputs of the current code without exact environment,
  data-hash and commit reproduction.
- RL as a replacement for exact DP: the archive's own frozen results do not
  support that choice. RL remains an explicitly labelled exploratory ablation,
  not a globally optimal algorithm.
- Taking Form and fugue annotations in the DCML boundary experiment: their
  annotation semantics differ and require separate cross-corpus evaluation.

## Evidence learned from the archive

The archive's frozen 10/3/3 summary reported key-profile exact DP at roughly
F1 `0.524`, Siamese+DP at `0.508`, and joint RL at `0.477`. Its four-fold
work-blocked results reported Siamese+DP around `0.476` and RL beam search
around `0.458`, with RL considerably slower. These numbers are useful for
method selection and as a reproducibility target, but are not copied into the
current evidence pack.

The defensible thesis conclusion is therefore not “RL necessarily improves
hierarchical clustering.” It is:

1. Greedy is an efficient local-search baseline with irreversible merge
   errors.
2. Exact ordered DP removes that search defect and is globally optimal under a
   clearly stated frozen additive objective.
3. Deep metric learning addresses the separate unknown-distance problem, but
   its empirical benefit must be established on held-out works rather than
   assumed from model complexity.
4. If a learned objective improves while Boundary F1 does not, that is evidence
   of objective/annotation mismatch, not a failed optimality proof.

This framing aligns the implementation with the project brief while avoiding
claims that DCML supplies an expert hierarchical ground-truth tree.

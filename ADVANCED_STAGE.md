# Advanced neural-distance stage

This stage learns a small symmetric, temporally ordered pitch-class distance and compares adjacent
Greedy search with exact ordered dynamic programming under the same frozen
boundary-aware objective. DCML `localkey` annotations supervise local boundary
examples; they are not treated as hierarchical ground-truth trees.

## Run the experiments

Run the fast six-work integration check first:

```powershell
python scripts/evaluate_neural_distance_stage.py --quick `
  --output-dir results/neural_distance_stage_quick
```

Run the preregistered 16-work evaluation after the quick run and all tests
pass. The default evaluates the primary seed plus two outer-refit stability
seeds:

```powershell
python scripts/evaluate_neural_distance_stage.py `
  --output-dir results/neural_distance_stage --resume
```

For a faster primary-seed-only run, disable the stability repeats:

```powershell
python scripts/evaluate_neural_distance_stage.py `
  --stability-seeds `
  --output-dir results/neural_distance_stage_primary
```

Run Op. 95 as the only held-out work with `--outer-work n11op95`; all other
works remain available for leakage-free training. This is a diagnostic only
and must not replace the complete leave-one-work-out evaluation.

## What is trained

- `handcrafted_key_profile` is the strongest current handcrafted reference.
- `diagonal_mahalanobis` learns 12 non-negative pitch-class weights.
- `neural_circular_siamese` defaults to `harmonic_tcn_v3`: every example keeps
  its masked `(time, 12)` chroma sequence. A circular harmonic CNN with 16
  channels and pitch dilations `{1,3,4,5}` is followed by residual temporal
  TCN blocks with dilations `{1,2,4}`. Shared Siamese weights and Euclidean
  comparison preserve symmetry and exact joint-transposition invariance.
- Local boundary examples use contexts `{1,2,4,8}`. Leaf affinity uses fixed
  four-bin chronological neighbourhood embeddings. All embeddings are frozen
  before tree search, so the exact DP optimal-substructure assumption remains
  valid.
- The primary loss is work/class-balanced binary cross-entropy. Optional
  Hadsell-style metric-margin and hard-negative ranking losses are exposed as
  explicit ablations; they default to zero because the development Quick run
  showed lower held-out F1 when both were weighted by 0.25.
- AdamW uses learning rate `1e-3`, neural-only weight decay `1e-3`, batch size
  `64`, temporal dropout `0.15`, gradient clipping `1.0` and cosine annealing
  to `1e-5`. `--device auto` uses CUDA when available and otherwise CPU.
- Every learned distance supplies both the ordered affinity matrix and the
  local boundary-contrast features.
- Candidate checkpoints `{20,40,60,80,100}` and the tree/extraction operating
  point are selected only on grouped inner-validation works.
- The selected model is retrained on the outer-training works. The held-out
  work is evaluated once and never supplies training or selection labels.

The primary metric is work-macro Boundary F1 at 8 quarterbeats with a shared
validation-selected threshold. F0.5, precision, recall, tolerance sensitivity,
tree imbalance and objective gaps are secondary diagnostics.

## Main outputs

- `held_out_per_piece.csv` and `held_out_per_work.csv`: accuracy data.
- `work_macro_summary.csv` and `micro_summary.csv`: corpus summaries.
- `primary_paired_tests.csv`: work-blocked paired permutation tests.
- `objective_comparison.csv`: exact-DP minus Greedy objective gaps.
- `tree_diagnostics.csv`: depth, Colless, Sackin and root-split diagnostics.
- `learned_distance_diagnostics.csv`: held-out boundary/non-boundary distances.
- `outer_selections.csv`: checkpoint, lambda, balance and threshold choices.
- `fold_assignments.csv`, `neural_training_audit.csv` and
  `contrast_training_audit.csv`: leakage audit trail.
- `shards/<outer-work>/checkpoints/`: frozen learned models for one
  independently recoverable outer work.
- `shards/<outer-work>/complete.json`: atomic completion marker with hashes
  for every table and checkpoint.
- `run_context.json` and `run_progress.json`: scientific-configuration,
  dataset and implementation fingerprints plus recovery progress.
- `metadata.json`: definitions, seeds and the precise optimality scope.

The quick results are only an integration check. Do not draw dissertation
conclusions from two held-out works.

## Colab/A100 resumable formal run

Unzip the repository (including `external/ABC`) under `/content`, mount
Google Drive, and keep the output directory on Drive. Linux cells use a
backslash, not PowerShell's backtick:

```bash
python scripts/evaluate_neural_distance_stage.py \
  --data-root external/ABC \
  --device cuda \
  --seed 20260818 \
  --stability-seeds 20260819 20260820 \
  --output-dir /content/drive/MyDrive/dissertation_results/dissertation_main/neural \
  --resume
```

Each outer work is committed independently. Re-run the identical command after
an interruption. Completed shards are checksum-validated and loaded; the
interrupted outer work is recomputed. A code, data or scientific-configuration
change is rejected instead of silently mixing evidence.

After all 16 works complete, rebuilding the root CSV/JSON/plots without
training is supported by:

```bash
python scripts/evaluate_neural_distance_stage.py \
  --data-root external/ABC \
  --output-dir /content/drive/MyDrive/dissertation_results/dissertation_main/neural \
  --aggregate-only
```

The full default should contain 96 learned checkpoints: 16 outer works times
two learned families times three seeds. Old root-level checkpoints from the
pre-shard evaluator are ignored and cannot satisfy completion.

### Neural ablations

The following commands keep the data split and tree evaluation fixed:

```powershell
# Legacy two-layer encoder under the new training budget
python scripts/evaluate_neural_distance_stage.py --quick `
  --families neural_circular_siamese --neural-architecture legacy_v1 `
  --channels 8 --output-dir results/neural_distance_stage_quick_legacy10

# Aggregated Harmonic v2 (temporal-order ablation)
python scripts/evaluate_neural_distance_stage.py --quick `
  --families neural_circular_siamese --neural-architecture harmonic_v2 `
  --output-dir results/neural_distance_stage_quick_aggregate_v2

# Temporal TCN with optional metric/ranking terms enabled
python scripts/evaluate_neural_distance_stage.py --quick `
  --families neural_circular_siamese --metric-loss-weight 0.25 `
  --ranking-loss-weight 0.25 `
  --output-dir results/neural_distance_stage_quick_metric_ranking
```

## Defensible methodology claim

> A neural distance is learned from training-work local-key boundaries, after
> which exact dynamic programming finds the globally optimal ordered binary
> tree under the frozen learned boundary-aware objective.

Do not claim that DP recovers a globally optimal musical hierarchy or that
DCML supplies an expert hierarchical tree. Reinforcement learning is not part
of this stage and remains future or explicitly exploratory work.

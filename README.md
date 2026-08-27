# Deep Learning and Hierarchical Clustering for Music Analysis

This repository evaluates temporally contiguous hierarchical clustering on
music data, from interpretable Greedy and exact ordered DP baselines to a
Siamese circular/harmonic encoder and an adjacent-merge REINFORCE policy.

## Research questions

1. Do music-informed distances improve adjacent Greedy clustering?
2. Does exact ordered DP improve a shared objective and boundary recovery over Greedy?
3. Does Siamese metric learning generalise to unseen works better than handcrafted distance?
4. Does sequence-level reinforcement learning improve the complete merge policy?

## Evaluation design

DCML local-key annotations are segment boundaries, not hierarchical ground truth. Boundary precision, recall, and F1 are therefore the primary metrics. Order-preserving dynamic programming first maximises one-to-one matches and then minimises timing error. The primary pre-registered configuration is 8 quarterbeat bins, 8 quarterbeat tolerance, and depth 4; fixed prediction-budget results separate tree shape from boundary-count effects.

TED is auxiliary. It measures discrepancy between a predicted hierarchy and a **flat tree induced by DCML local-key segments**. It must not be interpreted as distance from an expert hierarchical tree. Node-count-scaled TED is not bounded by one; raw, pruned, node-count, leaf-count, pruning-depth, and label-bin sensitivity diagnostics are emitted together.

Distances are Euclidean, tonic-relative weighted chromagram, a two-dimensional circle-of-fifths embedding inspired by Tonnetz geometry, and 24-dimensional key-profile activation. The fixed C-major weighting is available only as an explicitly named ablation. Balanced and random adjacent-merge trees are structural baselines.

## Reproduce

The only supported formal orchestration entry point is:

```powershell
.\.venv\Scripts\python.exe scripts\run_dissertation_experiments.py --quick
.\.venv\Scripts\python.exe scripts\run_dissertation_experiments.py --stage all --device cuda --resume
```

It writes the immutable input/checksum audit, environment, per-stage status,
formal outputs, checkpoints, and zero-shot external results below
`results/dissertation_main/`. If a Taking Form or Fugue score cannot be paired
reliably, it is written to `exclusion_audit.csv`; ABC experiments continue and
the missing external stage is never reported as a successful run.

Useful component development runs:

```powershell
python scripts\eval_greedy.py --piece n11op95_01 --quick
python -m pytest -q
```

Intermediate-stage runs:

```powershell
python scripts\evaluate_dp_stage.py
python scripts\evaluate_dp_stage.py --quick
python scripts\train_parametric_distance.py --model both
python scripts\train_parametric_distance.py --quick --model both
```

Accuracy-oriented intermediate evaluation (recommended):

```powershell
python scripts\evaluate_optimized_stage.py
python scripts\evaluate_optimized_stage.py --quick
```

This keeps the required adjacent bottom-up greedy algorithm and compares it
with an exact ordered DP under one shared pairwise-affinity objective. The
objective is the similarity-revenue dual of Dasgupta hierarchical clustering
cost, restricted to contiguous binary trees. Affinities are RBF-calibrated
from the existing interpretable distance representations; no DCML annotation
is read until after each tree has been constructed.

The optimized evaluation ranks a tree boundary by the temporal span of its
lowest common ancestor. This supports equal prediction-budget comparisons and
avoids giving balanced and comb-shaped trees different boundary counts merely
because they are cut at the same depth. Fixed-depth results remain available
for comparison with the earlier experiment. Method, context radius, and fixed
boundary budget are selected with leave-one-work-out evaluation at the quartet
level, using the same selected configuration for greedy and DP.

The corresponding outputs are written to `results/optimized_stage`:

- `loowcv_summary.csv`: leakage-free work-level Greedy/DP Boundary F1;
- `loowcv_selections.csv`: training-only choice for every held-out work;
- `prominence_boundary_per_piece.csv`: equal-budget and oracle-budget results;
- `depth_boundary_per_piece.csv`: legacy-compatible depth results;
- `objective_comparison.csv`: exact DP minus greedy revenue gap;
- `tree_diagnostics.csv`: root split, depth, Sackin/Colless and singleton-child diagnostics.

`DP revenue >= greedy revenue` is an implementation guarantee. A higher
revenue is not by itself evidence of better music analysis; Boundary F1 and
tree-shape diagnostics must be reported separately.

### Boundary-aware intermediate evaluation

    python scripts\evaluate_boundary_aware_stage.py
    python scripts\evaluate_boundary_aware_stage.py --quick
    python scripts\evaluate_boundary_aware_stage.py --outer-work n11op95

This is the recommended experiment after the ordered-affinity baseline. It
replaces fixed-depth boundary extraction with a calibrated salience score:

    salience(k) = normalized LCA leaf span(k) * local boundary contrast(k)

Local contrast uses 1, 2, and 4-bin contexts under Euclidean,
circle-of-fifths, and key-profile representations. It subtracts within-side
dispersion from left-versus-right dissimilarity. Training-work median/MAD
statistics calibrate all nine features. The experiment reports an
equal-weight unsupervised scorer and a simple supervised non-negative logistic
scorer trained from DCML boundaries.

The shared tree objective is:

    J = (1-lambda) * normalized affinity revenue
        + lambda * span-weighted contrast
        - beta * mean squared child-size imbalance

The DP exactly maximises this additive objective. The Greedy baseline remains
the supervisor-required temporally adjacent bottom-up algorithm and uses the
same affinity, contrast, lambda, and beta. Lambda, beta, threshold, and budget
are selected using inner validation. A plain sum of boundary contrasts is not
used: every complete ordered binary tree contains every adjacent leaf boundary
exactly once, so that sum would be tree-invariant.

Evaluation is nested and blocked by complete quartet. The outer fold is used
only for final testing; three grouped inner folds select lambda, the salience
threshold, and the fixed boundary budget. The same selected configuration is
used for Greedy and DP. Threshold results are primary, fixed-budget results are
the fair equal-count comparison, and GT-count rows remain explicitly marked
oracle diagnostics. Use --outer-work to resume or distribute formal outer
folds without weakening their training split.

Outputs are written to results/boundary_aware_stage:

- held_out_per_piece.csv and held_out_per_work.csv: nested out-of-fold metrics;
- outer_selections.csv: training-only lambda/threshold/budget decisions;
- learned_contrast_parameters.csv: per-fold feature scales and weights;
- objective_comparison.csv: DP-minus-Greedy objective guarantees;
- boundary_salience.csv: every predicted boundary's contrast and LCA span;
- fold_assignments.csv and training_example_audit.csv: leakage audit trail;
- work_macro_summary.csv, micro_summary.csv, plots, and Holm-corrected tests.

DP Pitch Scapes for the three music-informed distances:

    python scripts\dptree_keyprofile.py
    python scripts\dptree_circle_of_fifths.py
    python scripts\dptree_weighted.py

The figures are saved under results/dp_stage/figures. Add --show to open the
Matplotlib window. For another piece, supply its matching --midi and --notes
paths; the output filename is derived from the notes filename. See DATASETS.md
for compatible formats and conversion guidance.

Pitch Scapes now default to an exact balance-regularized ordered-affinity DP
with beta=0.6. The regularizer is part of the objective, so the returned tree is
still a strict global optimum rather than a post-hoc visual rebalance. Run with
--balance-weight 0 for the unregularized ordered-affinity tree, or with
--objective additive --balance-weight 0 to reproduce the old comb-prone
additive figures. Every generated title and terminal report states the chosen
objective and beta.

The legacy clustering DP minimises the additive objective
`C(i,j) = min_k C(i,k) + C(k,j) + d(x[i:k], x[k:j])` over all ordered
binary trees with the fixed input leaves. This is a global optimum for that
objective only, not for Boundary F1 or musical structure in general. It is
different from the smaller alignment DP used only to match predicted and DCML
boundaries during evaluation.

Parameter training uses a deterministic 10/3/3 split of complete quartets, so
movements from one quartet cannot cross train, validation, and test. The
calibrated mixture combines Euclidean, circle-of-fifths, and key-profile
distances after division by training-only non-zero median scales. The diagonal
Mahalanobis model learns non-negative, sum-to-one pitch-class weights with
contrastive loss. Held-out test metrics are computed only after weights and
checkpoint are fixed.

The earlier 10/3/3 parameter script remains available for reproduction. The
boundary-aware entry point uses nested leave-one-work-out evaluation instead,
which is the preferred basis for corpus-level accuracy claims.

### Deep metric and reinforcement-learning extension

The advanced extension trains a compact circular harmonic Siamese encoder and
then two adjacent-merge REINFORCE policies. Four circular Conv1D branches use
dilations 1, 3, 4, and 5, followed by a 16-channel residual block and a
two-channel harmonic projection. Real/imaginary coefficients for harmonics 1,
3, 4, and 5 form the 16-D embedding. A common pitch-class transposition rotates
each harmonic pair orthogonally, so embedding Euclidean distance is exactly
joint-transposition invariant for any learned weights. The embedding itself is
equivariant rather than invariant and therefore retains tonic phase.

The policy receives harmonic amplitudes, pairwise difference amplitudes, and
relative complex phases. These candidate features—and hence deterministic
policy actions—are also invariant to jointly transposing an episode. One policy
freezes the pretrained encoder; the other fine-tunes it with sequence-level
reward. The terminal
reward is average precision of the tree's duration-normalised LCA temporal-span
boundary-prominence ranking against
training-work DCML local-key boundaries. Policy rollout itself is annotation
free. This learns an approximate strategy for a boundary-recovery proxy, not a
unique or expert-authored musical hierarchy.

The current cluster state still sums its constituent bins into one 12-D
pitch-class histogram. Consequently it cannot distinguish two intervals with
identical aggregate pitch content but different temporal order. A small
pitch-CNN plus temporal TCN/BiGRU is reserved as future work; it is not silently
added to this small-data experiment, and no Transformer is used.

The extension reports both the original aggregate-cluster Greedy algorithm and
a strict search-only comparison. In the latter, key-profile/Siamese affinity is
held fixed while adjacent average-linkage Greedy is compared with exact ordered
affinity DP. This prevents a Greedy-versus-DP row from silently changing both
the search algorithm and the tree objective. Each affinity Greedy/DP pair
shares one validation-selected fixed ABC boundary budget. A single
length-relative budget is selected across all methods on ABC validation and
frozen before zero-shot Taking Form/Fugue evaluation.

Run a development smoke test with:

    python scripts\train_deep_clustering.py --quick

Run the pre-specified three-seed experiment with:

    python scripts\train_deep_clustering.py --seeds 20260827 20260828 20260829

Before the RL comparison, run the formal metric-learning corpus ablation:

    python scripts\evaluate_neural_corpus.py --device cuda --seeds 20260827 20260828 20260829

This is a four-fold nested, work-blocked out-of-fold experiment over all 16
complete ABC works. Every work is outer-test data exactly once. Within each
fold, the remaining works are divided into training and validation works;
outer-test harmonies are not loaded until both MLP and circular-harmonic CNN
checkpoints and their shared Greedy/DP boundary budgets have been frozen. The
two encoders receive the same folds, examples, minibatch order, joint
transposition schedule, seeds, and optimizer settings. This is the formal
`MLP versus circular/harmonic CNN` ablation and the corpus-level Advanced
Neural+Greedy/DP baseline. It intentionally excludes RL to avoid multiplying
the exploratory policy experiment across 24 metric-training runs.

Outputs under `results/neural_corpus` use work-macro Precision, Recall, and F1
as primary metrics. Boundary-prominence AP, TED, tree shape, shared-objective
revenue, and runtime are auxiliary. `fold_assignments.csv` proves that each
work is tested once, `access_audit.csv` records the freeze-before-test order,
and `ablation_summary.csv` directly reports harmonic-CNN minus MLP and neural
minus key-profile DP differences. Architecture selection must be interpreted
from this pre-specified ablation, not from the earlier quick smoke result.

The extension uses the same deterministic 10/3/3 complete-work split as the
parametric experiment. Siamese checkpoints are selected by validation
work-macro boundary-classification AP. RL checkpoints are selected by
validation work-macro tree boundary AP (movement AP is averaged within each
work before works are macro-averaged). Each model's fixed boundary budget is
also selected on validation. All seed checkpoints and budgets are frozen
before test harmonies are loaded and the single held-out phase begins.

DCML boundary times are projected to distinct ordered internal bin edges with a
minimum-error dynamic program. This avoids collapsing two nearby annotations
onto one bin whenever enough internal edges exist. The projection error and any
unavoidable loss are recorded for every movement.

Outputs under `results/dissertation_main/deep` include metric/RL histories, model
checkpoints, held-out per-piece and per-work metrics, neural boundary
classification AP, tree diagnostics, deterministic action trajectories,
ablation summaries, learning curves, split assignments, and a configuration
record. `held_out_per_seed.csv` contains one work-macro row per model/seed;
`held_out_summary.csv` computes mean and standard deviation across those seed
rows. `access_audit.csv` verifies that test annotations were loaded after all
checkpoints/budgets were frozen, and `boundary_projection_audit.csv` records
the annotation-to-bin discretisation. `experiment_state.json` is updated after
each completed seed and phase. Per-seed checkpoints and histories support
`--resume`, so an interrupted overnight run does not look like a completed
experiment. The held-out set contains only three works, so this experiment is an
advanced exploratory extension and is not a basis for strong significance
claims.

### Greedy / DP / learned-distance Pitch Scapes

After the non-quick three-seed deep stage is complete, generate the five
pre-registered dissertation visualisations with:

    .\.venv\Scripts\python.exe scripts\run_dissertation_experiments.py --stage visualize --device cuda

The figures are written below `results/dissertation_main/pitch_scapes`, with
one music-named directory per movement. Each directory contains separate
key-profile affinity Greedy, exact key-profile ordered-DP, and harmonic-CNN
distance plus exact ordered-DP images, a three-panel comparison with ABC
local-key reference boundaries, complete node coordinates, and model/data
metadata. The Pitch Scape is constructed directly from the ABC notes TSV on
the same quarter-beat timeline as clustering, so MuseScore and MIDI export are
not required. The visualisation command rejects quick/smoke checkpoints and
selects its formal neural seed by validation AP only.

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

The intermediate scripts write to `results/dp_stage` and
`results/dp_parametric`. Their outputs include per-movement and per-work
Boundary F1, greedy/DP objectives and gaps, runtimes, split assignments,
training-only scales, learned weights, checkpoint history, and held-out test
comparisons.

Precision-recall plots contain tolerance/depth operating points; they are not confidence-threshold PR curves. Generated results are ignored by git and are reproducible from the command above.

## Validity limits

Local-key change is not equivalent to phrase, cadence, section, or complete musical-form hierarchy. Large tolerances can overstate boundary recovery; pruning aligns approximate scale rather than semantic level. Pitch-class centroids lose distributional information, and global-key estimation can be uncertain. Corpus-level uncertainty and parameter sensitivity must accompany method comparisons.

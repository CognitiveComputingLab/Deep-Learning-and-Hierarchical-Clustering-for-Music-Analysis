# Greedy baseline evaluation notes

## Intermediate DP scope

The new clustering DP enumerates every possible split of every contiguous
interval and therefore returns the exact minimum-cost ordered binary tree under
the stated additive child-distance objective. The main experiment uses no
length weighting and no balance penalty. Optional settings are objective
ablations and must be named as such.

An objective reduction does not imply better DCML boundary recovery. If DP
lowers its objective while Boundary F1 stays unchanged or falls, report this as
evidence that the objective and annotation criterion are not aligned.

The term dynamic programming now occurs in two separate places:

- clustering DP constructs the globally objective-optimal tree;
- boundary-alignment DP only computes one-to-one evaluation matches and never
  changes a predicted tree.

Parametric training is blocked by quartet: 10 works train, 3 validate, and 3
remain held out. Mixture scales are estimated from training examples only;
mixture weights are selected on training-work F1. Validation reports mixture
generalisation and chooses diagonal-Mahalanobis checkpoints. The test split
must not be inspected until model selection is complete.

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

## Accuracy-oriented ordered-affinity objective

The legacy additive child-distance objective remains available for exact
reproduction, but it can reward comb-shaped trees and is not the recommended
accuracy-oriented comparison. `scripts/evaluate_optimized_stage.py` uses the
same non-negative leaf affinity for both searches:

- greedy is temporally constrained bottom-up average linkage;
- DP exactly maximises ordered similarity revenue over all contiguous binary
  trees with the fixed leaves.

Tree boundaries are ranked by LCA temporal span. Equal-budget results are the
main fair tree comparison; fixed depth is retained as a compatibility result,
and GT-count (`oracle`) budgets must remain labelled diagnostic. Hyperparameter
selection is leave-one-quartet-out and one common configuration is used for
both searches, so the comparison isolates the search procedure.

On the 70-movement, 16-work key-profile scan at 8 qb and 8-qb tolerance,
all movements completed. The final out-of-fold work-level results were Greedy
F1 0.4402 and DP F1 0.4604; the paired work-level DP-minus-Greedy permutation
p-value was 0.0323. These are a historical run record: the corresponding CSVs
were lost with the previous machine and must be regenerated before the numbers
are quoted in the dissertation. They are not a promise for other datasets or
evidence of expert hierarchical agreement.

## Boundary-aware objective

The next experiment addresses the remaining target mismatch directly. For
every internal bin boundary it computes local left/right contrast at context
sizes 1, 2, and 4 with Euclidean, circle-of-fifths, and key-profile
representations. The feature is cross-context dissimilarity minus average
within-context dispersion. Feature median/MAD scales are fitted on training
works only.

Two scorers must be reported:

- equal-weight unsupervised contrast;
- a non-negative logistic scorer trained on work-balanced DCML boundary and
  deterministic hard-negative examples.

Annotations never enter held-out tree construction. The DP maximises
normalized affinity revenue plus contrast weighted by normalized LCA leaf
span, minus a normalized squared child-size imbalance penalty. The span
coupling is essential because an unweighted contrast sum is identical for
every complete ordered binary tree. The balance term prevents the positive
contrast reward or aggregate-distance geometry from being maximized by a comb
tree. Greedy remains adjacent and bottom-up, merging pairs with high affinity,
weak intervening contrast, and comparable sizes.

Primary boundaries use a threshold selected in inner grouped validation.
Fixed-budget results are reported in parallel using the same budget for both
searches; GT-count budgets remain oracle diagnostics. Lambda, threshold, and
budget, together with balance beta, are selected without the outer quartet,
and DP must never score below Greedy on the shared objective.

Run:

    python scripts\evaluate_boundary_aware_stage.py --quick
    python scripts\evaluate_boundary_aware_stage.py

If inner validation selects lambda zero, report that honestly: it means local
contrast helped boundary ranking but did not justify changing the tree under
that fold. If objective gains do not improve held-out Boundary F1, the correct
interpretation remains objective/annotation mismatch rather than DP failure.

## Deep-learning extension scope

The Siamese encoder learns a nonlinear embedding distance from balanced
adjacent-interval boundary examples. REINFORCE subsequently learns an ordered
adjacent-merge policy using the complete tree's boundary-prominence average
precision as a terminal training reward. A self-critical deterministic rollout
is the baseline. The frozen-encoder and jointly fine-tuned policies are both
reported so that any gain from sequence-level representation updates is an
explicit ablation.

This does not turn ABC into hierarchical ground truth. The reward is derived
from flat local-key boundaries, and the held-out split contains three complete
works. Describe the policy as an approximate learned strategy for a
boundary-recovery proxy. It has no global-optimality guarantee and must not be
called the true or best musical hierarchy. Test annotations are not loaded
until all seed checkpoints and model-specific budgets have been frozen on the
validation works; `access_audit.csv` records this order. RL validation first
averages movements within each work, then macro-averages works.

For search-only comparisons, the fixed key-profile or Siamese leaf affinity is
shared by adjacent average-linkage Greedy and exact ordered-affinity DP. The
legacy aggregate-cluster Greedy rows remain explicitly named as additional
baselines. Reference boundary times are projected to distinct ordered bin
edges by minimum-error dynamic programming, with collisions and projection
errors written to `boundary_projection_audit.csv`.

# Trained parametric distance with exact hierarchy DP

This experiment is the preferred Intermediate-stage Taking Form result. It
keeps the aligned shared-format hierarchy as the expert reference and never
uses held-out labels while constructing a tree.

## Method

The trained distance is a non-negative calibrated mixture:

    d(x,y)^2 = sum_m w_m ||f_m(x)-f_m(y)||^2 / s_m^2

where m is Euclidean, circle-of-fifths, or key-profile representation,
weights are non-negative and sum to one, and scales are estimated from
training works only. Its concatenated Euclidean embedding is implemented by
CalibratedRepresentationMixture, so the same learned distance supplies the
affinity matrix for both search algorithms.

The nine local training features combine the three distances with contexts
of 1, 2, and 4 bins. Explicit Taking Form boundaries provide graded targets:
coarse boundaries are stronger than fine boundaries. A deterministic
work-balanced soft-label objective learns non-negative feature weights. The
context weights are marginalised into the three distance weights.

Both searches then use the same learned affinity, boundary scores, contrast
weight, and balance weight:

    objective = (1-lambda) normalized_affinity
                + lambda hierarchical_boundary_reward
                - beta imbalance_penalty

Greedy remains adjacent bottom-up merging. Exact interval DP globally
maximises this additive objective over all contiguous ordered binary trees.
Every test piece verifies DP objective is at least the Greedy objective.

Lambda and beta are selected only on the training composer. The scorer uses
all training movements; objective selection uses one deterministic movement
per training work so works with more movements do not dominate. The other
composer is a completely held-out cross-corpus test.

## Reproduce

Beethoven to held-out Mozart:

    python scripts/evaluate_parametric_hierarchy_stage.py --train-dataset beethoven --test-dataset mozart --output-dir results/parametric_hierarchy_beethoven_to_mozart

Mozart to held-out Beethoven:

    python scripts/evaluate_parametric_hierarchy_stage.py --train-dataset mozart --test-dataset beethoven --output-dir results/parametric_hierarchy_mozart_to_beethoven

## Held-out results

| Train to test | Search | Full T | Reduced T | LCA Spearman |
|---|---|---:|---:|---:|
| Beethoven to Mozart | Greedy | 0.6596 | 0.4422 | 0.5714 |
| Beethoven to Mozart | Exact DP | 0.7326 | 0.6305 | 0.7108 |
| Mozart to Beethoven | Greedy | 0.6055 | 0.4499 | 0.5341 |
| Mozart to Beethoven | Exact DP | 0.6266 | 0.4769 | 0.5624 |

For Beethoven to Mozart, mean DP-minus-Greedy Full T is +0.0730 over 17
held-out works (paired sign-flip p=0.0041). For Mozart to Beethoven the
difference is +0.0211 over 27 works (p=0.0792), so the mean improvement is
positive but not conventionally significant.

These findings support a limited claim: exact DP improves hierarchy recovery
when paired with a training-aligned parametric objective. They do not imply a
global optimum of expert T-measure, and they do not justify claiming that every
distance, corpus, or individual piece improves.

## Outputs

- held_out_per_piece.csv and held_out_per_work.csv: full tree metrics;
- held_out_summary.csv: corpus means;
- paired_dp_vs_greedy.csv: work-blocked paired test;
- objective_comparison.csv: exact-DP optimality audit;
- training_objective_grid.csv: training-only lambda/beta selection;
- model_parameters.json: learned weights, scales, selection and leakage audit;
- training_example_audit.csv and parametric_training_history.csv.

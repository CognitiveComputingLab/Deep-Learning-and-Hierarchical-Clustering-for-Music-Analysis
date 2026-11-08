# Datasets, hierarchy evaluation, and result scope

## Why use both ABC and Taking Form?

| Property | ABC/DCML Beethoven string quartets | Taking Form piano sonatas |
|---|---|---|
| Musical input to this pipeline | Duration-weighted, 12-dimensional pitch-class bins extracted from score notes | The same type of bins, extracted from separately matched DCML piano-sonata scores |
| Expert reference | Harmonic analyses with flat local-key regions; changes define tonal boundary events | Measure/beat-positioned form labels organized from coarse to fine; parent/level relations define an explicit, partly specified hierarchy |
| Coordinate preparation | Notes and harmony TSVs already share score-relative quarterbeats | Form coordinates must be mapped from measure/beat to the exact score edition and unfolding before quarterbeat sampling |
| Current compatible scope | 70 movements from 16 complete quartet works | 89 of 120 annotated movements score-aligned: 17 Mozart movements/17 works and 72 Beethoven movements/27 works |
| Training signal used here | ABC local-key changes supervise the neural boundary representation; inner validation selects tree and reporting settings by work | Fixed-representation hierarchy study uses no fitted representation; separate parametric-mixture study trains on one composer's form labels and tests on the other |
| Principal reported metric | Work-macro local-key boundary F1 at 8-quarterbeat matching tolerance | Work-macro Full T-measure of expert form-hierarchy agreement; Reduced T-measure is complementary |

Both tasks use a representation-to-affinity-to-ordered-tree pipeline. They test **different claims**: ABC tests whether the tree yields useful tonal-change boundaries; Taking Form tests whether its relative hierarchy agrees with an expert form analysis. An ABC F1 score is not a form-hierarchy score, and a Taking Form T-measure is not a local-key detection score. The corpora also differ in genre, composer, and annotation practice, so any cross-dataset interpretation must be conditional rather than treated as a pooled estimate. Source descriptions: [ABC repository](https://github.com/DCMLab/ABC), [Taking Form paper](https://archives.ismir.net/ismir2019/paper/000084.pdf), and [Taking Form annotation format](https://github.com/MarkGotham/Taking-Form).

The saved alignment summary reports **89/120** Taking Form movements ready for this score-based pipeline. Twelve Beethoven movements lack a matching score in the local snapshot, and 19 have ambiguous repeat/measure or edition-length alignment. They are excluded from the tested sample rather than assigned zero accuracy. The source form markup does not prescribe a unique binary sequence of merges at every leaf-scale distinction. Thus an exact binary tree can disagree with an annotation in ways that reflect unannotated fine detail as well as actual structural error. See `data/shared_annotations_v1_aligned/summary.json`, `src/score_alignment.py`, and `src/dataset_converters.py`.

## How Full and Reduced T-measures are computed here

The implementation samples the centre time of each predicted input bin. It builds two matrices of least-common-ancestor (LCA) depth for these same times: one from the predicted binary tree and one from the aligned expert form hierarchy. For a query bin $q$, if the reference gives $\operatorname{LCAdepth}(q,i)>\operatorname{LCAdepth}(q,j)$, then $i$ is structurally closer to $q$ than $j$. The estimated tree receives credit when it preserves that strict ranking. Full T uses every pair of distinct reference ranks; Reduced T uses adjacent reference ranks. T-recall treats the expert ranking as reference, T-precision reverses the roles, and their harmonic mean is T-measure. Queries with no comparable rank pairs are omitted from the corresponding average; when none remain, the code returns zero. Ties in the estimated ranking do not satisfy a strict expert ordering.

This is the ranking approach of [McFee, Nieto, and Bello (ISMIR 2015)](https://archives.ismir.net/ismir2015/paper/000174.pdf), with explicit local choices: sampling at score-bin centres, comparison over the whole movement (`tree_window_qb = null`), and reference levels constructed from the source hierarchy. It does not require predicted and reference trees to have equal depth. The result measures agreement with the **annotated** hierarchy only; it cannot identify unannotated levels or decide which of several musically plausible analyses is uniquely correct. See `src/hierarchical_evaluation.py:tree_measure()`.

## When Precision, Recall, and F1 are informative

For ABC, each reported tree boundary is matched one-to-one and in order to at most one local-key reference change within 8 quarterbeats. The matching first maximizes the number of matches and then minimizes total timing error. Unmatched predictions are FP and unmatched references are FN: $P=TP/(TP+FP)$, $R=TP/(TP+FN)$, and $F1=2PR/(P+R)$, with zero-denominator cases set to zero. The primary threshold is chosen in inner work-blocked validation, not from the held-out work. Each work first averages its movement scores; the headline result is the mean of 16 work scores. It is therefore not generally the harmonic mean of the displayed macro P and macro R. See `src/greedy_evaluation.py:boundary_scores()` and `scripts/evaluate_neural_distance_stage.py`.

For Taking Form, a single boundary P/R/F1 collapses the source's depth distinctions. The saved `equal_budget_summary.csv` is useful only as a **per-reference-level diagnostic**: it chooses the number of predicted boundaries from that test movement's reference level, so the prediction budget uses test annotation information. It should not be presented as an operationally selected main score. At level 1 and level 3 with key-profile distance, the diagnostic F1 values are Mozart Greedy/DP **0.1275/0.0784** and **0.5737/0.5489**, and Beethoven Greedy/DP **0.2294/0.1720** and **0.5435/0.5181**, respectively. All use 8-quarterbeat matching tolerance. Because the annotation is partly specified, a musically meaningful predicted boundary without an annotated counterpart may count as FP. Full T is the more appropriate main endpoint for the current form-hierarchy claim.

## Saved Taking Form results and uncertainty

The following **fixed key-profile** results are from `results/shared_hierarchy_stage_{mozart,beethoven}_tree/`. Every value is a mean over works; for Beethoven, movements within a work are averaged first. The displayed 95% intervals are the existing 2,000-replicate **work bootstrap intervals for each search's mean**, not paired intervals for the Greedy–DP difference. The older additive merge-distance objective is minimized, and saved per-piece results confirm DP cost is no greater on all 17 Mozart and 72 Beethoven movements.

| Test corpus | Search | Full T, 95% work CI | Reduced T, 95% work CI |
|---|---|---:|---:|
| Mozart: 17 movements, 17 works | Greedy | 0.6430 [0.6016, 0.6832] | 0.5002 [0.4444, 0.5537] |
| Mozart: 17 movements, 17 works | DP | 0.5922 [0.5610, 0.6212] | 0.3995 [0.3698, 0.4282] |
| Beethoven: 72 movements, 27 works | Greedy | 0.5968 [0.5744, 0.6190] | 0.4353 [0.4067, 0.4657] |
| Beethoven: 72 movements, 27 works | DP | 0.5478 [0.5242, 0.5707] | 0.4050 [0.3801, 0.4306] |

On the per-movement comparison, DP has a **lower Full T** than Greedy for 13/17 Mozart and 47/72 Beethoven movements despite its better or equal cost. This is descriptive evidence of a mismatch between the optimized cost and the annotated form target; it does not identify a unique cause. The paired work-level tests already saved in the result folders provide further diagnostics; for key-profile Full T the Mozart DP-minus-Greedy difference is -0.0508 with Holm-adjusted $p=0.0524$. Avoid calling this a general proof that Greedy is more musically correct.

A **separate form-supervised parametric-mixture** experiment trains on one Taking Form composer and tests on the other. It uses a boundary-aware objective that DP maximizes; it is not the ABC neural model and must not be pooled with the fixed key-profile table.

| Train → held-out test | Works | Greedy Full T / Reduced T | DP Full T / Reduced T | Paired Full-T difference, unadjusted $p$ |
|---|---:|---:|---:|---:|
| Beethoven → Mozart | 17 | 0.6596 / 0.4422 | 0.7326 / 0.6305 | +0.0730, 0.0041 |
| Mozart → Beethoven | 27 | 0.6055 / 0.4499 | 0.6266 / 0.4769 | +0.0211, 0.0792 |

The opposite DP–Greedy patterns across the two Taking Form experiment designs are consistent with an interaction among representation, objective, and annotation target. Because both the representation and objective selection change, they do **not** isolate a causal effect of the representation alone. All numerical values and source-file hashes are in [evidence_snapshot.json](evidence_snapshot.json); selection and objective direction are in `results/parametric_hierarchy_{beethoven_to_mozart,mozart_to_beethoven}/model_parameters.json`.

## ABC context and robustness

The frozen primary ABC run has 70 successful movements in 16 works. At the primary seed, work-macro F1 is key-profile **0.4883 Greedy / 0.4873 DP**, diagonal Mahalanobis **0.4554 / 0.4458**, and neural circular Siamese **0.5148 / 0.5307**. The neural DP-minus-key-profile DP difference is +0.0434; its saved paired test gives unadjusted $p=0.0377$ but Holm-adjusted $p=0.2262$ across the recorded comparisons. Across the three outer-refit neural seeds, Greedy/DP F1 is **0.5148/0.5307**, **0.5369/0.5316**, and **0.5262/0.5146**. These additional seeds reuse the primary inner-validation selection and vary the outer refit; they are not three independent full nested searches. The ranking of Greedy versus DP therefore changes with seed. See `results/dissertation_main/final_evidence/` and `results/dissertation_main/neural/`.

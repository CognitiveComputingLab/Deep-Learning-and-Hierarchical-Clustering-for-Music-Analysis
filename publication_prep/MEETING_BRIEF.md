# Next meeting: one-page discussion brief

**Working question.** When does exact ordered-tree inference improve agreement with expert music analysis, and how does that depend on the similarity representation, tree objective, and annotation target?

## Three points to communicate

1. **What the model learns.** The ABC neural circular Siamese representation is trained on local-key boundary candidates from 70 quartet movements grouped into 16 works. Positives are candidate gaps nearest annotated changes; nearby ambiguous gaps are ignored. Its primary loss is work/class-weighted BCE. It has no complete expert-tree supervision. The cross-composer Taking Form parametric mixture is a separate model trained with form labels.
2. **What DP optimizes.** Exact DP searches all contiguous binary splits for a fixed additive whole-tree objective. The current Greedy baseline uses the same affinity and contrast evidence but a different local merge priority. DP's better objective value is guaranteed for that fixed objective and input, not for expert F1 or T-measure. The fixed key-profile Taking Form study minimizes an older merge-distance cost; the ABC main study and form-supervised Taking Form study maximize a boundary-aware objective.
3. **What the datasets show.** ABC provides flat tonal-boundary reference events: neural DP reaches work-macro F1 **0.5307** versus key-profile DP **0.4873** at the primary seed, while neural Greedy/DP ranking reverses across the three saved seeds. Taking Form provides explicit, partly specified form hierarchies: with fixed key profile, Greedy/DP Full T is **0.6430/0.5922** on Mozart and **0.5968/0.5478** on Beethoven. With a separately form-supervised representation, Greedy/DP Full T is **0.6596/0.7326** for Beethoven→Mozart and **0.6055/0.6266** for Mozart→Beethoven. These are distinct experiments with different training and objective choices.

## Questions to settle with Robert and Brian

- Should the paper's primary endpoint be ABC local-key boundary F1, Taking Form Full T, or an explicitly two-task study?
- For a search-method claim, should we add a Greedy baseline whose local priority is derived from the exact DP objective?
- Which targeted follow-up has priority: a matched-objective search comparison, a form-hierarchy representation ablation, or deeper Taking Form annotation-coverage sensitivity?
- Which venue and timetable determine the final experiment budget and division of writing work?

## Evidence and limits

The tables in [DATA_AND_EVALUATION.md](DATA_AND_EVALUATION.md) are generated from the completed local results listed, with SHA-256 source hashes, in [evidence_snapshot.json](evidence_snapshot.json). The code revision for this audit is `b053bf98ea08298f2438c2322a06c85e1e52d31e` on `research-sync-clean-20261107`. ABC's paired neural-DP versus key-profile-DP F1 test is unadjusted $p=0.0377$, Holm-adjusted $p=0.2262$; avoid a broad significance claim. Taking Form's current score alignment covers 89/120 annotated movements, and absent alignments are not scored as failures. The two Taking Form experiment designs cannot isolate the effect of representation by themselves because their objectives and training targets also differ.

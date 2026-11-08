# Publication preparation: meeting next steps

These English notes implement the six action items from the supervisors' meeting. They are a code-grounded working draft for a paper, not a claim that every historical experiment used the same objective or annotation target.

| Document | Purpose |
|---|---|
| [RELATED_WORK.md](RELATED_WORK.md) | Position non-sequential clustering, grammar-based parsing, dynamic programming, and music-structure analysis. |
| [METHODS.md](METHODS.md) | Define the representation-to-affinity interface, Greedy and exact inference, and neural supervision. |
| [DATA_AND_EVALUATION.md](DATA_AND_EVALUATION.md) | Compare ABC with Taking Form, define metrics, and report hierarchy results with uncertainty. |
| [DATA_RESOURCE_AUDIT.md](DATA_RESOURCE_AUDIT.md) | Assess other candidate hierarchical corpora without treating annotation availability as pipeline readiness. |
| [MEETING_BRIEF.md](MEETING_BRIEF.md) | One-page discussion summary and decisions for the next meeting. |
| [evidence_snapshot.json](evidence_snapshot.json) | Machine-readable result values, source-file SHA-256 hashes, and the code revision used for the audit. |

Regenerate the evidence snapshot from completed local outputs, without training:

```powershell
python scripts/build_publication_evidence.py --output publication_prep/evidence_snapshot.json
```

The snapshot is based on code revision `b053bf98ea08298f2438c2322a06c85e1e52d31e` on branch `research-sync-clean-20261107`. Result files under `results/` and aligned data under `data/` are local, ignored experiment artifacts; their checksums are recorded in the snapshot. The snapshot does **not** make those large artifacts available to another machine. Do not describe a checkout of this branch alone as a full reproduction of the reported numerical experiments.

The source analysis remains the saved 2026 dissertation experiment, not a newly trained model. Before a submission, confirm the paper's primary question and endpoint with the supervisors and freeze the final source, dataset revisions, configuration, and result archive together.

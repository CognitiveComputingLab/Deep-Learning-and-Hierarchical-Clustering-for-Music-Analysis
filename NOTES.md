# Op. 95 mvt.1 basic-stage results (2026-06-25)

## Ground truth
DCML localkey annotations from `n11op95_01.harmonies.tsv`:
| Segment | qb range   | localkey (in f minor) | Music-theoretic role |
|---------|-----------|-----------------------|----------------------|
| 1       | 0-82      | i (f)                 | Exp. primary theme   |
| 2       | 82-236    | VI (Ab)               | Exp. secondary theme |
| 3       | 236-342   | i (f)                 | Development          |
| 4       | 342-367   | VI (Ab)               | Dev. VI area         |
| 5       | 367-512   | I (F)                 | Recap (Picardy)      |
| 6       | 512-604   | i (f)                 | Coda                 |

## Settings
- Bin size: 8 qb (~2 measures)
- Total leaves: 76
- Splits collected: depth ¡Ü 4
- Tolerance: 24 qb (~6 measures)

## Results

| Distance   | raw TED | pruned TED | Prec | Rec  | F1   | Hits |
|-----------:|:-------:|:----------:|:----:|:----:|:----:|:----|
| euclidean  | 146     | 8          | 0.14 | 0.20 | 0.17 | 328?342 (14 qb off) |
| weighted   | 146     | 7          | 0.00 | 0.00 | 0.00 | none |
| tonnetz    | 145     | 10         | 0.22 | 0.40 | 0.29 | 72?82 (10), 232?236 (4) |
| keyprofile | 146     | 11         | 0.18 | 0.40 | 0.25 | 232?236 (4), 368?367 (1) |

## Findings

1. **Complementary distances**: tonnetz catches macro tonicization within
   exposition (72¡Ö82, entry of Ab-major secondary theme);
   keyprofile catches recap boundary (368¡Ö367). Neither dominates.

2. **All distances miss "soft" boundaries**: 342 (VI within development)
   and 512 (coda start) are never found. Both are gradual transitions
   without cadential arrival.

3. **The 236 boundary is universally caught** (232 in both tonnetz and
   keyprofile). This is the strongest DCML boundary in the piece:
   Ab-major ¡ú f-minor via V/iv, a functional-harmony extremum.

4. **Weighted stability metric is counter-productive**: F1 drops to 0.
   Adding stability weights to pointwise Euclidean amplifies short
   dissonances rather than long tonicization.

5. **Greedy commits early**: The [qb 200-604] region (development, dev-VI,
   recap, coda) is merged before any of the 3 GT boundaries inside it
   can be recovered. Consistent with the project outline's diagnosis.

## Implications for later stages

- DP stage: global optimality should recover at least 342 and 512, both
  currently blocked by greedy's early merge of [200, 604].
- DL stage: hand-crafted distances have a ceiling (max F1 = 0.29 on this
  piece). Learned distances should target the 60% of GT boundaries that
  no hand-crafted distance recovers.

## Caveats

- Single-piece result. Need Taking Form / Algomus data for cross-piece.
- DCML localkey and pitch-scape's PC-distribution structure are at
  different levels (functional harmony vs. statistical tonality).
  A form-level GT (Taking Form) might tell a different story.

  -----------------   ----above are evaluations for greedy approaches------------------------------------------

  
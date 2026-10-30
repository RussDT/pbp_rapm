# WNBA RTSS-to-PBP ShotQuality Student

`wnba_test/train_wnba_rtss_pbp_student.py` tests whether the richer RTSS
`qSQ` signal can teach a model that uses only fields available in the Gabriel
play-by-play. This is a research-only MID/THREE lane. It does not write VPM
possession targets, change the published WNBA board, or open RTSS season 2026.

## Research Question

The repo-owned PBP ShotQuality model covers more games and seasons than RTSS,
but PBP does not observe contest and tracking context. RTSS observes that
context but covers a smaller proprietary surface. The experiment therefore
uses RTSS as a teacher rather than mixing provider values into only some games:

```text
RTSS qSQ on matched shots
  -> outcome-safe PBP student
  -> one PBP prediction contract for every MID/THREE shot
```

Per the audited provider scale, `qSQ` is an expected-effective-field-goal
score. The teacher target is:

```text
teacher expected points = 2 * qSQ
```

`qSP` is retained as a shooter-inclusive benchmark. It is not the pure
shot-context teacher.

## Exact Join

The audited RTSS game inventory maps each provider game UUID to the official
WNBA game id. Within a game, `shots.pbpId` joins directly to the Gabriel event
number. The build never uses a keep-first provider join.

| Join field | Result |
| --- | ---: |
| Sanctioned 2023-2025 RTSS MID/THREE shots | 66,886 |
| Exact game/event matches to the PBP prediction surface | 61,522 |
| Duplicate RTSS game/event rows | 0 |
| Period mismatches | 0 |
| Outcome mismatches | 0 |
| Shot-value mismatches excluded | 1 |
| Validated teacher matches | 61,521 |

Validated matches are 14,449 in 2023, 21,633 in 2024, and 25,439 in 2025.
Outcome and shot value are join audits only; they are not student features.

## Student Contract

The fixed histogram-gradient-boosting student uses only pre-result PBP fields:

- shot value, distance, and legacy location;
- period and seconds remaining;
- MID/THREE zone, distance bucket, and shot family;
- clock bucket and court side;
- outcome-neutral shot-form tags such as pull-up, step-back, turnaround,
  fadeaway, driving, cutting, and putback.

Made/missed outcome, actual points, raw action labels, full descriptions,
made-only assist tags, `qSQ`, and `qSP` are forbidden inputs. A same-architecture
control is fit to actual outcomes on the exact same teacher-covered training
rows. That comparison isolates the benefit of the softer RTSS teacher label.

No hyperparameter search is performed. Validation is chronological:

- train 2023, score 2024;
- train 2023-2024, score 2025.

The student is scored both on all eligible Gabriel PBP shots and on the subset
with direct RTSS matches. Paired uncertainty resamples games, not shots.

## Results

### Teacher fidelity

| Validation | Matched shots | Student-vs-qSQ EV RMSE | Correlation |
| --- | ---: | ---: | ---: |
| 2024 | 21,633 | 0.16673 | 0.6885 |
| 2025 | 25,439 | 0.16459 | 0.6883 |

The student recovers a substantial but incomplete share of qSQ. The remaining
gap is expected because PBP does not observe RTSS contest and defender geometry.

### Standalone student

On RTSS-matched shots, the qSQ student beats the same-row PBP outcome learner
in both folds:

| Validation | Baseline-minus-student MSE | 95% game-clustered interval |
| --- | ---: | ---: |
| 2024 | +0.012058 | [+0.006180, +0.017797] |
| 2025 | +0.004722 | [+0.001013, +0.008346] |

Positive values favor the qSQ student. This establishes that RTSS supplies a
cleaner teaching signal than the same number of binary makes and misses.

The standalone student does **not** replace the mature PBP model. Against the
incumbent context-only prediction it is effectively tied on matched shots and
slightly worse on the all-PBP surface, with intervals crossing zero. Direct
provider qSQ and qSP remain better on matched shots because they retain rich
context that cannot be reconstructed from PBP.

### Frozen blend diagnostic

The blend weight is selected once on all 2024 PBP shots, then applied unchanged
to 2025:

```text
blended EV = incumbent EV
           + w * (qSQ-student EV - incumbent EV)
```

| Incumbent | 2024-selected student weight | 2025 MSE gain | 95% interval |
| --- | ---: | ---: | ---: |
| Context-only PBP | 0.41935 | +0.001247 | [+0.000151, +0.002282] |
| Context + prior shooter | 0.38400 | +0.001119 | [-0.000139, +0.002320] |

The context-only blend improves 2025 MSE by about 0.081% and clears the paired
game-clustered interval. The shooter-inclusive comparison is positive but
inconclusive. Because 2025 is already spent RTSS development evidence, neither
result is a promotion receipt.

## Decision

RTSS can improve a PBP-estimated ShotQuality surface, but the useful result is
complementary rather than a standalone replacement. Retain the 2024-selected
context blend as a frozen research candidate. Do not change the live WNBA VPM.

Before any VPM admission:

1. attach the frozen MID/THREE EV to first-chance possessions;
2. carry `actual - EV` residuals so row-level scoring closes exactly;
3. leave RIM unchanged;
4. compare the summed factor model on future actual possessions;
5. open RTSS 2026 only through the existing one-shot confirmation protocol.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python wnba_test/train_wnba_rtss_pbp_student.py \
  --workers 4 \
  --bootstrap-draws 5000
```

Generated and gitignored outputs live under
`wnba_test/Results/rtss_pbp_student/`:

- `forward_metrics.csv`;
- `paired_game_clustered_comparisons.csv`;
- `teacher_fidelity.csv`;
- `forward_predictions.parquet`;
- `manifest.json`.

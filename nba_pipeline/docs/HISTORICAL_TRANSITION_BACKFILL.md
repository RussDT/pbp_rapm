# Historical Transition Backfill

## Status

The frozen ShotQuality-descriptor classifiers have been scored across every
available NBA regular season and postseason from 1997 through 2026.

The production pass wrote:

- 60 season/type parquets;
- 6,328,268 field-goal-attempt rows;
- 37,979 game-file counts;
- 97 historical-safe pre-shot features per classifier;
- raw, calibrated, and binary transition probabilities;
- file, era, and feature-drift audits.

The generated artifacts live under:

```text
nba_pipeline/results/sq_transition_backfill/
```

They remain out of git under the repository artifact policy.

## Run

The model bundles and their OOF/forward prediction files must already exist
under `nba_pipeline/results/sq_transition_classifier/`.

```bash
.venv/bin/python nba_pipeline/scripts/score_historical_transition.py \
  --start-year 1997 \
  --end-year 2026 \
  --season-types all \
  --rebuild-calibration \
  --overwrite
```

Subsequent runs can omit `--rebuild-calibration` if
`transition_calibration.json` has version
`game_group_crossfit_logit_v2`.

Focused verification:

```bash
.venv/bin/python -m pytest \
  nba_pipeline/tests/test_historical_transition_backfill.py \
  nba_pipeline/tests/test_sq_challenger_models.py -q
```

## Clock normalization

Clock normalization is part of the model contract because the classifier uses
period clock and elapsed time from prior events.

- `time_quarter` is the preferred canonical source and is parsed as `MM:SS`.
- Modern raw files store `seconds_remaining_quarter` as the 0-59 seconds
  component, so the fallback is `minute * 60 + second`.
- Gabriel 1997-2013 raw files store `seconds_remaining_quarter` as total
  seconds remaining in the period, with values up to 720. For that schema the
  fallback uses the field directly.

The scorer still orders events by:

```text
game_id, period ascending, clock_sec descending, event_num ascending
```

Tests cover both second-field schemas and the canonical-time override. The
corrected 1997 and 2013 regular-season shot-clock means are 348.62 and 348.09
seconds remaining, respectively; the invalid pre-fix outputs that double
counted minutes were overwritten.

## Calibration without same-game leakage

The base ranking model and probability calibrator have distinct validation
rules.

### 2025

For each matched SQ row, the base score is the original game-group OOF model
probability. A five-fold logit calibrator is then applied cross-fit by game:
the row's entire game is excluded from the calibrator fit.

### 2026

For each matched SQ row, the base score is the untouched forward probability
from the classifier trained on 2025. Calibration is again cross-fit by game,
so no outcome from the row's game fits its calibrator.

Every calibration fold records both its fit-game and validation-game ids in
`transition_calibration.json`. Construction fails if the sets overlap, and a
unit test asserts the non-overlap contract.

Some raw 2025-26 FGA rows have no matching surviving SQ label/prediction. Those
rows retain the full-model score and are marked explicitly as:

```text
2025_full_model_unlabeled_row_fallback
2026_full_model_unlabeled_row_fallback
```

They receive the fold calibrator for their game when the game is known. An
unseen game uses a calibrator whose fitting data contain no rows from that
game. The 2025 fallback is not an OOF model evaluation and must not be reported
as such.

### 1997-2024

There are no historical SQ transition labels. These seasons use one pooled
logit calibrator fit on the labeled 2025 OOF and 2026 forward probabilities.
The output scope is deliberately named:

```text
retrospective_pooled_2025_2026_for_unlabeled_season
```

This corrects labeled-era base rate and calibration shape. It is not evidence
that the probabilities are calibrated within each historical era. No
historical make/miss result, text heuristic, or inferred transition label is
used to fit an era-specific calibrator.

The operating threshold is an explicitly retrospective F1 maximum on the
pooled game-cross-fitted labeled-era probabilities. The continuous
probability should be preferred for analysis; the `>= 0.5` label is also
retained.

## Labeled-era calibration results

| Target | Rows | Raw Brier | Cross-fit calibrated Brier | Raw log loss | Cross-fit calibrated log loss | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transition | 426,835 | 0.040145 | 0.039729 | 0.142816 | 0.141419 | 0.001471 |
| Transition + dunk/layup | 426,835 | 0.017164 | 0.016807 | 0.061639 | 0.060642 | 0.000482 |

The transition operating threshold is `0.373635` (F1 `0.8150`); the
transition-rim threshold is `0.430986` (F1 `0.8589`). These are
retrospective labeled-era operating points, not universal historical truth.

On matched 2026 regular-season rows, the game-cross-fitted calibrated
transition probability has log loss `0.144720`, Brier `0.041181`, ROC AUC
`0.968920`, and PR AUC `0.900716`. The transition-rim probability has log loss
`0.064006`, Brier `0.018025`, ROC AUC `0.989851`, and PR AUC `0.933230`.

## Output schema

Each `transition_backfill_{year}_{rs|ps}.parquet` contains identifiers and
clock context plus, for both transition targets:

- `*_probability_full_model`: frozen full-2025 model score;
- `*_probability_raw`: leakage-auditable base score used for calibration
  (OOF in matched 2025, forward in matched 2026, otherwise full model);
- `*_probability`: calibrated probability;
- `*_label_050`;
- `*_label_operating`;
- `*_operating_threshold`;
- `*_calibration_scope`;
- `*_base_probability_scope`;
- `sq_*`: surviving vendor label when a labeled prediction row matches.

The file also carries per-row feature missing counts/rates and the frozen
feature-contract version.

## Era audit

Shot-weighted regular-season calibrated probability means from the completed
run:

| Era | Shots | Transition | Transition operating-label rate | Transition-rim |
| --- | ---: | ---: | ---: | ---: |
| 1997-2003 | 1,262,794 | 0.1054 | 0.1081 | 0.0589 |
| 2004-2010 | 1,378,359 | 0.0970 | 0.1004 | 0.0581 |
| 2011-2016 | 1,180,249 | 0.1096 | 0.1137 | 0.0646 |
| 2017-2020 | 829,396 | 0.1354 | 0.1369 | 0.0708 |
| 2021-2024 | 843,626 | 0.1409 | 0.1393 | 0.0751 |
| 2025-2026 | 438,686 | 0.1505 | 0.1485 | 0.0819 |

This trend is plausible but not itself validation. Rule, pace, description,
and action-code regimes all change across the 30-year window.

## Drift audits

The output directory includes:

```text
transition_backfill_file_audit.csv
transition_backfill_era_audit.csv
transition_backfill_feature_drift.csv
transition_feature_reference_2025.csv
transition_backfill_summary.json
```

Feature drift is measured against ordinary 2025 PBP. A feature/file pair is
flagged when:

- missing-rate change is at least 0.25;
- finite standardized mean shift is at least one reference standard
  deviation; or
- a feature that is constant in the reference changes historically.

The constant-reference case is reported as
`reference_constant_change=true`, not as a fabricated million-sigma z-score.
Early eras show expected changes in action codes and prior-event mix.

## Interpretation guardrails

- The targets reproduce the surviving vendor `transition` and
  `transition + dunk/lay_up` descriptor semantics. They are not independent
  human ground truth.
- Discrimination is strongly validated in 2025 OOF and untouched 2026
  forward data. Calibration before 2025 is retrospective and unverified.
- Historical action text and event codes drift. Use the audit files whenever
  comparing distant eras.
- Do not treat an operating label as a possession-definition replacement
  without a separate row-level reconciliation against the intended possession
  surface.

The training feature contract and initial model evaluation are documented in
[ShotQuality-Era Challenger Models](./SHOTQUALITY_CHALLENGER_MODELS.md).

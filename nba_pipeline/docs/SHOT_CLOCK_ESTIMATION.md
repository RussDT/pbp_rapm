# Shot-Clock Estimation

`scripts/build_shot_clock_estimates.py` estimates the shot-clock time for each
field-goal attempt in a raw NBA PBP parquet. The source feed does not include a
native shot-clock column, so this is an inferred shot-level artifact, not a
replacement for any RAPM possession surface.

## Command

```bash
python nba_pipeline/scripts/build_shot_clock_estimates.py \
  --season 26 \
  --season-type PS \
  --output-dir nba_pipeline/results/shot_clock
```

Default inputs are `nba_pipeline/raw_data/NBA{YY}.parquet` for regular season
and `nba_pipeline/raw_data/NBA{YY}_PS.parquet` for playoffs. The 2026 playoff
run writes:

- `nba_pipeline/results/shot_clock/shot_clock_estimates_26_ps.parquet`
- `nba_pipeline/results/shot_clock/shot_clock_estimates_26_ps_diagnostics.csv`

These are generated artifacts and should stay out of git unless explicitly
curated.

## Output Schema

Each row is one FGA and can be joined back to raw PBP or shot-quality rows by
`game_id`, `period`, and `event_num`.

Important columns:

- `shot_clock_est`: integer estimate clamped to `[0, 24]`.
- `shot_clock_lower` / `shot_clock_upper`: the conservative ±1-second interval
  implied by whole-second PBP timestamps, also capped by the active reset.
- `shot_clock_raw`: unclamped integer estimate for diagnostics.
- `elapsed_since_reset`: game-clock seconds elapsed since the inferred reset.
- `rule_reset_len`: nominal reset under the season's NBA rule.
- `reset_len`: effective reset after capping the nominal reset by the remaining
  game clock. This prevents final-possession estimates such as `19` seconds
  when only `5` game-clock seconds remained.
- `reset_reason`: source of the active reset, such as `made_fg`,
  `def_rebound`, `off_rebound`, `turnover`, `made_final_ft`,
  `def_foul_reset14`, `jumpball_tip`, or `period_start`.
- `reset_event_num` / `reset_time_quarter`: event anchor for the active reset.
- `confidence`: `high`, `medium`, or `low`.
- `confidence_score`: stable numeric mapping (`0.90`, `0.60`, `0.20`) for
  filtering and downstream features. This is not an empirically calibrated
  probability of correctness until arena-clock labels are supplied.
- `confidence_reason`: `ok`, an advisory rule such as
  `def_foul_preserved_above_14`, or a diagnostic reason such as
  `raw_out_of_range` / `side_mismatch`.
- `needs_review`: true for medium/low-confidence rows.

## Inference Rules

Rows are sorted by `game_id`, `period`, `clock_sec` descending, then
`event_num` ascending. Event number alone is not safe because corrected NBA PBP
events can be appended with later action numbers at an earlier game clock.

Raw clock normalization is source-aware. Modern PBP stores
`minute_remaining_quarter` plus a `0-59` seconds component. Gabriel 1997-2013
stores total period seconds (up to `720`) in `seconds_remaining_quarter`.
When the latter column exceeds `59`, the estimator uses it directly instead of
adding the minute field a second time.

The responsible side is parsed by event type:

- Shots use the side whose description contains `MISS` or `PTS`.
- Turnovers use the side whose description contains `Turnover`.
- Rebounds use the side whose description contains `REBOUND`.
- Fouls use the side whose description contains foul text.
- Free throws use the side whose description contains `Free Throw`.

The live shot-clock state resets as follows:

- Period start and opening jump start at `24`.
- Made FG, turnover, defensive rebound, and final made live FT reset to `24`.
- Offensive rebound after a live FGA miss or final missed live FT resets to
  `24` through 2017-18 and `14` beginning in 2018-19.
- Defensive non-shooting fouls and kicked-ball violations reset to `14` only
  when the inferred clock is below `14`; otherwise the clock is preserved and
  the next shot keeps an advisory confidence reason.
- Technical/admin FT rows preserve the current shot-clock state.
- Timeouts, replays, and substitutions preserve the current shot-clock state.
- Jump balls parse `Tip to NAME` through a game-local player-name-to-side map.

## Validation

The 2026 playoff run over `raw_data/NBA26_PS.parquet` produced:

| Check | Result |
| --- | ---: |
| FGA estimate rows | `13,614` |
| High-confidence rows | `13,421` |
| High-confidence rate | `98.58%` |
| `shot_clock_est` range | `0` to `24` |
| Diagnostics rows | `193` |
| Side mismatch / raw out-of-range rows covered by diagnostics | `191 / 191` |

Manual spot-check windows used during implementation:

- `0042500101` Q1 events `7-20`
- `0042500102` Q1 events `23-30`
- `0042500101` Q3 events `468-489`
- `0042500104` Q2 events `310-327`

Known caveat: this is integer-second inference from event timestamps. Low
confidence rows usually involve corrected-event clusters, replay/jump-ball
oddities, or sequences where the real arena shot clock likely differed from a
simple PBP reset model.

## Independent-Label Validation

`scripts/validate_shot_clock_estimates.py` always checks state-machine
integrity (`0 <= estimate <= effective reset`, raw out-of-range counts, and
side mismatches). If a separate CSV/parquet with
`game_id, period, event_num, shot_clock_actual` is supplied, it also writes
MAE, RMSE, bias, within-one/two-second rates, and interval coverage by
confidence/reset reason.

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/validate_shot_clock_estimates.py \
  --estimates nba_pipeline/results/shot_clock/shot_clock_estimates_26_rs.parquet \
  --labels /path/to/independent_arena_labels.parquet \
  --output-dir nba_pipeline/results/shot_clock/validation
```

Before the archived 2014-15 labels were recovered, the full 2025-26 RS
integrity-only run produced `219,159` FGA estimates, `216,714` high
confidence (`98.884%`), zero effective-bound violations, `2,053` raw
out-of-range rows, and `310` side mismatches. Those figures establish coverage
and internal consistency only; they are not an absolute shot-clock MAE claim.

## Independent 2014-15 Labels

The current NBA liveData and PlayByPlayV3 event schemas do not expose a
per-shot arena shot clock. Their `clock` field is the period/game clock. The
official NBA shot-clock dashboard exposes aggregate shot-clock ranges, not
event rows joinable by game/event ID.

For independent validation, the repo can use the preserved 2014-15
`shot_logs.csv` export under:

```text
nba_pipeline/.cache/shot_clock_labels/shot_logs.csv
```

This file was scraped from the NBA's then-public player shot-log REST endpoint
and contains `GAME_ID`, `PERIOD`, `GAME_CLOCK`, `SHOT_NUMBER`, player ID, and
`SHOT_CLOCK`. The original NBA endpoint is deprecated; the local copy is a
generated cache and should not be committed.

`scripts/calibrate_shot_clock_estimates.py` joins labels by:

```text
normalized game_id
+ shooter_id
+ player-game FGA ordinal
```

It then accepts only rows whose period, make/miss result, point value, and game
clock (within 10 seconds) agree. On the surviving file:

| Join check | Result |
| --- | ---: |
| Label rows | 128,069 |
| Non-null shot-clock labels | 122,502 |
| Ordinal join candidates | 128,053 |
| Accepted independent labels | 121,783 |
| Coverage of non-null labels | 99.413% |
| Rejected/unmatched non-null labels | 719 |

## Residual Calibration

The raw state-machine estimate remains the interpretable first output. A
separate residual model corrects systematic PBP timing latency, especially the
dead time between a made basket and the next inbound.

Portable features are limited to:

- raw/clipped rule estimate and elapsed time since reset;
- effective and nominal reset length;
- current and reset game clocks;
- period/overtime/final-24-seconds flags;
- reset reason and confidence/review state.

It does not use the current shot result, player identity, shot-clock label, or
future event. The final test is the chronologically latest 20% of complete
games (`179` games); `717` earlier games fit the model. A three-fold
game-grouped training prediction supplies the 90% conformal interval width.

Held-out 2014-15 results:

| Model | Shots | MAE | RMSE | Bias | Within 1s | Within 2s | Within 3s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw reset rules | 23,044 | 2.864 | 4.187 | -2.162 | 23.78% | 50.91% | 69.29% |
| Residual-calibrated | 23,044 | 1.472 | 3.152 | +0.001 | 62.19% | 85.17% | 92.08% |

The calibrated 90% interval covered `89.90%` of held-out labels; its global
half-width was `2.567` seconds. Offensive-rebound and retained-jump-ball rows
remain the weakest reset families and should retain wider review flags.

Train/benchmark:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/calibrate_shot_clock_estimates.py \
  --raw nba_pipeline/raw_data/NBA15.parquet \
  --labels nba_pipeline/.cache/shot_clock_labels/shot_logs.csv \
  --output-dir nba_pipeline/validation/shot_clock_calibration \
  --output-name shot_clock_2015_absolute
```

Apply the serialized model to another estimate parquet while preserving
`shot_clock_est` and adding continuous calibrated value/lower/upper columns:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/calibrate_shot_clock_estimates.py \
  --model nba_pipeline/validation/shot_clock_calibration/shot_clock_2015_absolute_model.joblib \
  --apply-estimates nba_pipeline/results/shot_clock/shot_clock_estimates_26_rs.parquet \
  --output-dir nba_pipeline/results/shot_clock \
  --output-name shot_clock_estimates_26_rs
```

The calibrated model is validated on one historical season. Applying it to
modern seasons is a portable research backfill, not proof that timing latency
is stationary across eras or PBP providers.

## ShotQuality challenger

`scripts/train_sq_shot_clock_challenger.py` joins the calibrated 2026 RS clock
artifact to repo-owned SQ, tests incremental game-held-out make prediction, and
scores a neutral-14-second counterfactual for grenade burden. The 219,159-shot
join is complete, and clock improves both grouped-OOF and chronological losses.
It also combines conditioned shot EV with the current second-chance opportunity
prior to produce expected possession value at release. See
[Shot-Clock SQ and Grenade Burden](./SQ_SHOT_CLOCK_CHALLENGER.md).

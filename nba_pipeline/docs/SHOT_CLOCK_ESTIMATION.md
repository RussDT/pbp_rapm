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
- `shot_clock_raw`: unclamped integer estimate for diagnostics.
- `elapsed_since_reset`: game-clock seconds elapsed since the inferred reset.
- `reset_len`: `24` or `14`.
- `reset_reason`: source of the active reset, such as `made_fg`,
  `def_rebound`, `off_rebound`, `turnover`, `made_final_ft`,
  `def_foul_reset14`, `jumpball_tip`, or `period_start`.
- `reset_event_num` / `reset_time_quarter`: event anchor for the active reset.
- `confidence`: `high`, `medium`, or `low`.
- `confidence_reason`: `ok`, an advisory rule such as
  `def_foul_preserved_above_14`, or a diagnostic reason such as
  `raw_out_of_range` / `side_mismatch`.

## Inference Rules

Rows are sorted by `game_id`, `period`, `clock_sec` descending, then
`event_num` ascending. Event number alone is not safe because corrected NBA PBP
events can be appended with later action numbers at an earlier game clock.

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
  `14`.
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

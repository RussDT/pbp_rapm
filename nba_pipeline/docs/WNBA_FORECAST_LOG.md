# WNBA Pre-Tip Forecast Log (Phase 0 Operations)

Built August 10, 2026. The market-as-holdout lane from
`docs/WNBA_BETTING_PROGRAM.md`: every game day, emit frozen-model pre-tip
margin/win forecasts and capture market odds into append-only hash-chained
logs, then score against finals and the close. The ratings never see the
market; the logs make the evidence tamper-evident and as-of by construction.

## Files

| Script | Role |
| --- | --- |
| `wnba_test/wnba_gameday_common.py` | Shared contract: paths, tricode maps, boxscore cache, hash-chained log append/verify |
| `wnba_test/build_wnba_game_logs.py` | Final-game results table + player game logs from the CDN boxscore cache |
| `wnba_test/project_wnba_minutes.py` | Minutes/availability allocation (EWMA half-life 1.5, tuned on 2025 only) |
| `wnba_test/calibrate_wnba_forecast_sigma.py` | `calibrate` fits HCA/sigma on prior-season history boards; `replay` scores the 2026 pregame replay |
| `wnba_test/emit_wnba_pretip_forecasts.py` | Emits pre-tip forecast records for a game date |
| `wnba_test/capture_wnba_odds.py` | Captures ESPN odds snapshots (spread/ML/total) |
| `wnba_test/score_wnba_forecasts.py` | Joins forecasts + odds + finals; verdicts and metrics |

Logs and tables live under `wnba_test/Results/gameday/` (generated, not in
git): `forecast_log/forecasts.jsonl`, `odds_log/odds.jsonl`, `games.csv`,
`player_game_logs.csv`, `forecast_params.json`.

## Daily operation

```bash
.venv/bin/python wnba_test/build_wnba_game_logs.py
.venv/bin/python wnba_test/emit_wnba_pretip_forecasts.py --date "$(date +%F)"
.venv/bin/python wnba_test/capture_wnba_odds.py --date "$(date +%F)"
.venv/bin/python wnba_test/score_wnba_forecasts.py
```

Capture odds at least twice on a slate day when possible (early = opener
proxy, last pre-tip = close proxy); the scorer uses the latest pre-tip
snapshot per game as the close.

## Forecast model (v1, frozen)

- Team strength: sum of current-board `net_rapm` (positive-good both sides)
  weighted by projected minutes over 200 team minutes:
  `team_per100 = Σ net_rapm_i × minutes_i / 40`.
- Margin: `(home_per100 − away_per100) × pace/100 + HCA`;
  `p_home = Φ(margin/σ)`.
- Parameters from `forecast_params.json`, fit by
  `calibrate_wnba_forecast_sigma.py calibrate` on pregame-achievable
  prior-season history boards (2022-2025 development only): HCA ≈ 2.00,
  σ ≈ 12.66, `normal_std` selected (dev 90% coverage 0.905).
- **The scalar σ is explicitly a v1 placeholder.** The planned Phase 1.5
  distribution engine replaces it with a conditional σ (pace, totals level,
  rotation-shortening) and reprices derivatives from the full distribution;
  see the program doc.

## Record schemas

Both logs are hash-chained JSONL: each stored row carries `prev_hash` and
`record_hash = sha256(prev_hash + canonical_json(record))`, so any later
edit breaks every subsequent hash. `verify_log_chain` checks a whole file.
Records are sanitized before hashing (numpy scalars to native, NaN
rejected). Consumers must ignore unknown fields — the schema is
forward-extensible.

**`pretip_forecast`**: `emitted_at_utc, game_date, espn_event_id,
cdn_game_id, home_tricode, away_tricode, tipoff_utc, espn_state, board_path,
board_mtime_utc, minutes_source, home_per100, away_per100,
pace_possessions, hca_pts, margin_mean, margin_sigma, p_home_win,
home_rated_minutes_share, away_rated_minutes_share, low_coverage,
default_params, model_version`.

**`odds_snapshot`**: `captured_at_utc, game_date, espn_event_id,
home_tricode, away_tricode, tipoff_utc, state, provider, spread_home,
home_ml, away_ml, over_under` plus derived implied probabilities.
`spread_home` is the HOME line (negative = home favored); the
spread-vs-moneyline favorite sign agreement check runs at capture time.

Reserved optional fields for the future execution layer (documented now so
the schema does not churn later): `market_type, requested_stake,
accepted_stake, limit_event`.

## Integrity fences

- **Pre-tip only.** The emitter records the ESPN event state and by default
  emits only for state `pre` games. The scorer selects, per game, the last
  record with `emitted_at_utc` strictly before `tipoff_utc`; records failing
  that fence are excluded and tallied (`post_tip_excluded`), games with no
  known tipoff are tallied separately rather than silently trusted.
- **Fail-closed joins.** A forecast that cannot be matched to a final result
  gets `REFUSED` (never a guessed join); a forecast for a game that simply
  has not finished yet gets `PENDING`. All dates are US/Eastern on both
  sides of every join.
- **Frozen model.** `model_version` and the board mtime are recorded in
  every forecast; scoring is only comparable within a model version.
- **Coverage flag.** `low_coverage` marks games where either team's rated
  minutes share is below threshold — the program's interim no-bet filter.

## Scoring

`score_wnba_forecasts.py` reports, over scoreable (non-REFUSED,
non-PENDING) forecasts: margin RMSE/MAE, win log-loss and Brier, PIT
histogram and 90% interval coverage, spread performance against the captured
close, closing-line value, and **mean normal CRPS of the forecast
distribution against the realized margin, next to the close's CRPS under the
same σ** — the outcome-space ruler, so the model is judged against reality
and the market on the same scale, not only against the market.

## Provenance benchmarks (August 10, 2026)

The retrospective 2026 pregame replay (same construction, replayed as-of):
margin RMSE 13.28 with pregame-achievable boards, 13.18 conditional on
actual minutes, versus 13.12 for the DARKO-NPI benchmark on the same games —
statistically indistinguishable. The live log exists to measure the next
increment (minutes engine, conditional σ, props) against the close, not to
relitigate that comparison.

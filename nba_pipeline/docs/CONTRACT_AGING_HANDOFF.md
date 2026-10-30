# EPM contract-aging data handoff

Prepared 2026-09-09 for the contract projection task. This is data support and
a proposed validation design, not a fitted or promoted forecasting model.
No contract/analyzer repository is changed.

## Reproduce and consume

From the RAPM root, run `.venv/bin/python nba_pipeline/scripts/build_contract_aging_handoff.py`.
No network calls or credentials are needed. Outputs are private, gitignored,
under `nba_pipeline/results/contract_aging_handoff/`:

- `player_seasons.csv` / `.parquet`: 46,401 calendar-complete player-season rows,
  1997–2026, with RS GP, total minutes, derived MPG, reported season age, and
  explicitly marked no-appearance rows after each player's first observed year.
- `predictive_epm_snapshots.csv` / `.parquet`: 178,622 rows at 357 requested dates,
  2001-11-01 through 2026-08-21. Requested date, provider-effective date, last
  player game date, retrieval timestamp, and source path remain distinct.
- `snapshot_inventory.csv`, `provenance.json`: source inventory and SHA-256 hashes.
- `team_seasons.csv`: observed team games and seconds; these are not archived
  preseason schedules or a player-specific exposure denominator.
- `raw_census_audit.csv`, `season_census_audit.csv`, and per-year comparison CSVs:
  completeness gates and observed source discrepancies.

The RS sources are `player_sheets/year_totals/{1997..2026}.csv`, with unique
`PLAYER_ID` keys, positive GP, and nonmissing GP/MIN. MIN is total minutes,
not MPG. AGE is the NBA-reported season age; the scraper uses season-specific
`leaguedashplayerbiostats` (see `player_sheets/gen_totals/averages_scrape.py`).
There is no exact birthdate in this normalized handoff. Do not silently turn
this integer into an exact age at a historical snapshot. EPM has no age column.

The EPM sources are `nba_pipeline/cache/epm` (162 full snapshots with skills)
and `nba_pipeline/cache/epm_title` (297 monthly EPM-only snapshots). All 459 CSV
checksums match their manifests, and overlapping EPM/date values agree.
This is **predictive EPM**, not the provider's separate actual-season EPM.
The provider distinguishes these products in its [EPM explanation](https://dunksandthrees.com/about/epm)
and [actual-season table](https://dunksandthrees.com/epm/actual).

For authorized refreshes, `nba_pipeline/scripts/fetch_epm_data.py:load_api_key`
reads `EPM_API_KEY` or an explicitly supplied `--key-file`; it does not load a
dotenv file automatically. `fetch_monthly_epm_title.py` uses the same loader
and permits identical duplicate historical stint rows. Never log credentials.

## Completeness and dates

2026 has all 1,230 RS games and exactly the same 582 player IDs with positive
reconstructed PBP minutes as the source table. Source total minutes are
593,499.533 versus 593,500 from game lengths. Retain source GP/MIN: individual
PBP reconstruction differs on 16 GP values and by up to 27.3 minutes for one
player. The census test certifies any-minute participation, not perfect
individual lineup-clock measurement.

The loader sets `zero_exposure_certified` only for years with an exact matching
raw positive-minute/source player ID census and complete raw-vs-team game count.
Source GP/MIN nonmissingness, total-minute mass within two minutes, and individual
PBP differences are separate exposure-fidelity diagnostics. Thus 2003, 2005,
and 2016 can certify absence despite failing the minute-mass screen. This does
not certify their PBP minute allocation. Other no-appearance rows remain
**uncertified**; do not train zero workload labels from them. Several older
seasons fail, including incomplete 2011 raw game coverage. This gate is
conservative and can fail because of raw-lineup errors as well as source-table
omissions. Retain the original source GP/MIN for workload labels.

Team schedules vary: 2012 has 66 games/team; 2021 has 72; 2020 differs by team;
2013 includes a canceled game. Traded players can exceed 82 GP. Define the
workload denominator explicitly, and preserve observed team counts separately
from known-at-origin schedule assumptions.

`source_available_date` is unknown. These are retrospectively retrieved
historical tables, not authenticated archived publication vintages. A completed
RS table may be used only after that RS actually ended; for midseason origins,
use cutoff-reconstructed game data instead. Never infer availability from
filesystem modification timestamps.

The panel's attached post-season EPM date is July 1 except November 1 for
2020 and August 1 for 2021. This is a convenience state that can include
playoffs, **not** a safe automatic origin feature. At an origin, select only
a snapshot requested on or before the origin, retain its effective date, and
apply a declared freshness rule. Summer 2020 was still an active season.
Use the existing 2026-08-21 snapshot for the current forecast and label that date.

The 2021 August snapshot omits 43 RS players (9,068 minutes). Missing EPM must
remain missing; it does not mean no NBA participation or zero talent. A prior
eligible snapshot may supply an explicitly stale feature, but cannot be
silently carried forward as an observed future EPM target. Other missing
historical EPM/age counts are in `coverage.csv`.

## Existing aging work

`THIRTY_YEAR_AGE_ADJUSTED_RAPM.md` documents the accepted age-27-normalized
career RAPM curve. The actual file is
`nba_pipeline/results/longitudinal_orapm_drapm_age/joint_possession_1997_2026/age_curve.csv`.
`fit_joint_possession_age_curve.py` smooths joint possession age bins with a
slot-weighted cubic over ages 20–38. It is an all-era retrospective RAPM curve,
not an EPM forecast or an untouched aging validation. Do not reuse its endpoint
clipping as a forecast of zero deterioration after 38.

`fit_longitudinal_rapm_age_curve.py` fits consecutive within-player O/D changes,
with smoothed age effects and constrained calendar-pair effects;
`evaluate_age_curve_candidates.py` compares curve increments against no change.
The near-OLS annual curve is rejected (implausibly large Stockton translation
and worse than no age change). Linear age drift needs an explicit identifying
restriction; free age, period, and player/cohort terms do not identify it.

Private `nba_pipeline/results/epm_career_value/` contains an older predictive
EPM panel, ages, `active_career_projections.csv`, and `projection_methodology.json`.
The receipt describes boosting plus Monte Carlo, using RS+PS possessions and
continuation outcomes. Original fitting code was not located in the searched
script/tmp surfaces. Its documented holdout summary is insufficient to certify
chronological training isolation. Treat it as a lead, not a production model.

## Regular-season average predictive-state targets

Run `.venv/bin/python nba_pipeline/scripts/build_contract_rs_epm_targets.py`
after the main handoff loader. It exports `rs_pregame_epm_targets_2002_2026.csv`
(12,380 player-season rows) from 621,799 positive-minute game rows, plus
`rs_player_game_minutes_2002_2026.csv`, target coverage, provenance, and game
audits. This keeps the origin EPM state separate from a target-season average.

Each target is the minutes-weighted average of the latest available predictive
EPM snapshot **strictly before** the game date, with a 35-day request-age cap.
This is a step approximation to pregame predictive EPM, not realized RS EPM.
The archive has different snapshot frequencies; a monthly-only sensitivity
is appropriate. On same-day requests, `game_dt` often equals the game date;
that metadata is ambiguous despite the stated provider exclusion contract.
The strict-prior-date rule avoids asserting that these rows are proven safe.
All used rows pass requested/effective/player-game date gates.

For 2019–2026, dated minutes come from the existing
`results/estimated_skills/player_game_lines.parquet`, filtered to positive RS
minutes (zero-minute admin rows include unrelated historical IDs). For
2002–2018, minutes are reconstructed from raw lineups with dates from
`player_sheets/gen_totals/game_dates.csv`. The clock is parsed from `time_quarter`:
before 2014 the numeric `seconds_remaining_quarter` already contains total
clock seconds, unlike the modern remainder field. Never add minutes*60 to
that older total. Per-game allocated minutes cannot exceed game-clock mass.

2026 covers 99.1173% of RS minutes; 511 of 582 players clear the transparent
95% per-player coverage screen. 2006 has only 89.04% pooled coverage; 2011
has one missing raw game. Keep `season_game_schedule_complete`, individual
coverage, and the independent workload-census flags as separate gates.
Missing target EPM is not a zero target. Unobserved portions of an otherwise
covered season can be selection-biased, especially near debut or exit.

`target_available_not_before` is the day after the season's last observed RS
game, not proof of original provider publication time. Historical weights
retain the known PBP lineup-minute approximation; use source season GP/MIN
for workload labels. Do not silently repair official player totals from noisy
historical lineup allocations.

## Recommended contract model and validation

The contract task's proposed direct horizon 1–5 additive forecasts of
future EPM minus current EPM are a defensible simplicity control. Fit smooth
age and baseline-quality effects with regularization, plus measured history,
experience and workload. Estimate regression empirically: predictive EPM is
already stabilized. Never apply signed percentage aging, which can make a
negative player improve merely because a multiplier shrinks toward zero.

Use separate participation probability, games conditional on participation,
and minutes per game conditional on playing. An absent season can be injury,
exit, or a temporary absence; do not declare retirement from one missed year.
Keep missing future seasons right-censored. A last-EPM-within-one-year origin
cohort is valid if preregistered, but needs an explicit unsupported/comeback
group. Do not fit aging only to surviving veterans and then extrapolate it to
the full origin cohort.

Preserve participation/quality/workload dependence and cross-year dependence
when drawing simulations. Joint **out-of-fold**, horizon-specific residual
vectors are useful; a single unconditional residual pool is not sufficient
for all ages, horizons, and workloads. Export expected minutes and expected
EPM-times-minutes separately. If value is possession-based, preserve pace
and possession uncertainty rather than treating minutes as possessions.

At each chronological origin, train only on target outcomes fully observed
by that date, with earlier nested tuning/calibration. Overlapping 5-year
targets require special care: a later-origin training row cannot use a label
that matures after the validation origin. Grade each horizon and cumulative
contract value, compare no-change EPM and simple additive-age controls, and
report paired uncertainty clustered by player and calendar origin/season.
Check Brier/log loss and calibration for participation; MAE/RMSE and bias for
conditional EPM and workload; and expected-value error for the full cohort.
Keep one future confirmation period untouched. No fitted alternative was
built or promoted in this handoff.

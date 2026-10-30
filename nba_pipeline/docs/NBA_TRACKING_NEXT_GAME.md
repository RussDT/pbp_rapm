# Tracking stats: next-game per-100 filters

Six Gabriel tracking game-log stats had measured panels but no fitted model:
potential assists, drives, and minutes of possession per 100 offensive
possessions; contested shots, defended field-goal attempts (DFGA), and
contested defensive rebounds per 100 defensive possessions. This lane fits the
same next-game OU-versus-random-walk protocol that `NBA_DEFLECTIONS_NEXT_GAME.md`
established for deflections, one stat at a time, on the owner's next-game
objective. Research/forecast-lane only: nothing enters RAPM, VPM, a prior, or
a public table.

## Protocol (registered before fitting)

**Panel.** `tracking_backfill.parquet` rows 2019-2026 joined exactly on
player, season, season type, and game date to the resolved PBP
`player_game_lines.parquet` for offensive and defensive possessions. Duplicate
join keys are excluded as ambiguous (476 rows). Exposure is valid only when
the tracking log's minutes agree with the PBP lines within 0.5 and AST/FGA
match; otherwise the row is unscored. Potential assists and DFGA use only the
`*_observed` values where `*_is_measured` is true. Repaired `*_filled`
values, inferred omitted zeros, and estimates are never observations, and a
missing tracking game is unscored rather than zero. Every row carries a
per-stat `_value`, `_valid`, and `_reason` (observed, unmeasured,
negative_value, missing_or_ambiguous_exposure, box_or_minutes_mismatch,
zero_exposure). The full panel is 217,163 player-games; measured coverage
per stat and season is written to `source_audit.json` (roughly 99% of
minutes for PA/drives/possession minutes and DFGA, 98% for contested shots,
94-95% for contested DREB, whose hustle field is missing more often).

**Objective.** For each valid game, predict `100 * stat / exposure` before
ingesting that game's count or exposure. Squared errors are weighted by the
game's possessions within each season, then seasons are weighted equally.
Rows need at least one earlier measured observation. Calendar gap and season
crossing are known before tip; no projected minutes are involved.

**Split.** 2019 warm-up (mean rate, warm-up variance, and mean exposure from
all 2019 rows); tune and select on 2020-2022; 2023-2025 is frozen historical
stress; 2026 receives filtering and export only; 2027 is untouched.
`next_game_cells` refuses any requested season at or beyond 2026, and
`evaluate` refuses a panel that contains seasons beyond 2026.

**Models.** Next-game random walk (`run_filter`, five parameters: game, day,
and offseason process noise, observation noise, initial variance) and
next-game two-component OU with a common population attractor
(`run_filter_v2`, six parameters). Both tuned directly on the loss with
log-space differential evolution, seed 20260912, 60 iterations, population
multiplier 15, tolerance 1e-7, no polish, bounds from `stat_bounds` and
`bounds_v2`. Predictions are floored at zero. Baselines are strictly prior:
trailing ten observations, season-to-date with career fallback, and career.

**Selection.** `next_ou` only if its development gain over
`next_random_walk` exceeds one paired game-cluster bootstrap SE; otherwise
the random walk. There is no forward incumbent for these stats, so this is
the only gate. The frozen choice and parameters are written to `frozen.json`
before stress, and stress never switches the selection. Uncertainty: 2,000
paired game-cluster bootstrap draws within each season, seed 20260912, plus
player-clustered and >=200-prior-possession sensitivities and per-season
wins. No remaining-season secondary test exists here because no
longer-horizon comparator was ever fitted for these stats.

## Completed run (September 14, 2026)

All six stats selected `next_ou` on development. Development windows hold
67,909-75,946 rows across 3,608-3,623 games per stat.

| Stat (per 100) | Dev OU vs RW | Stress rows / games | Stress MSE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Potential assists (off) | +0.593% [+0.430, +0.756], 3/3 | 82,534 / 3,918 | 21.281 | 4.613 |
| Drives (off) | +0.998% [+0.775, +1.212], 3/3 | 82,592 / 3,918 | 22.494 | 4.743 |
| Possession minutes (off) | +2.502% [+2.020, +2.964], 3/3 | 82,592 / 3,918 | 1.349 | 1.162 |
| Contested shots (def) | +0.211% [+0.122, +0.293], 3/3 | 78,286 / 3,931 | 21.138 | 4.598 |
| DFGA (def) | +0.339% [+0.247, +0.426], 3/3 | 80,008 / 3,937 | 42.129 | 6.491 |
| Contested DREB (def) | +0.084% [+0.026, +0.145], 3/3 | 73,012 / 3,916 | 3.625 | 1.904 |

Frozen 2023-2025 stress, selected OU versus each comparator on identical
rows and exposure weights (percent next-game MSE improvement, game-clustered
95% interval, per-season wins):

| Stat | vs next-game RW | vs trailing ten | vs season-to-date | vs career |
| --- | ---: | ---: | ---: | ---: |
| Potential assists | +0.858% [+0.706, +1.015], 3/3 | +6.30% [+5.64, +7.27], 3/3 | +7.22% [+6.23, +8.40], 3/3 | +12.03% [+11.27, +12.90], 3/3 |
| Drives | +1.144% [+0.952, +1.341], 3/3 | +5.87% [+5.39, +6.37], 3/3 | +8.04% [+7.49, +8.61], 3/3 | +22.96% [+22.23, +23.66], 3/3 |
| Possession minutes | +2.480% [+2.088, +2.871], 3/3 | +10.70% [+9.92, +11.49], 3/3 | +16.38% [+15.52, +17.30], 3/3 | +38.80% [+37.79, +39.75], 3/3 |
| Contested shots | **-0.098% [-0.189, -0.010], 0/3** | +5.96% [+5.33, +6.63], 3/3 | +7.52% [+6.26, +8.94], 3/3 | +18.86% [+18.18, +19.59], 3/3 |
| DFGA | +0.269% [+0.184, +0.359], 3/3 | +7.34% [+6.88, +7.82], 3/3 | +6.39% [+5.81, +7.00], 3/3 | +3.57% [+3.14, +4.04], 3/3 |
| Contested DREB | +0.127% [+0.062, +0.189], 3/3 | +8.89% [+7.95, +10.08], 3/3 | +9.16% [+7.85, +10.49], 3/3 | +2.82% [+1.86, +4.03], 3/3 |

Reading the table:

- **Five of six OU selections hold on stress** with game-clustered intervals
  above zero, three season wins out of three, and matching player-clustered
  intervals (PA +0.858% [+0.724, +0.997]; drives +1.144% [+0.933, +1.367];
  possession minutes +2.480% [+2.127, +2.827]; DFGA +0.269% [+0.189,
  +0.348]; contested DREB +0.127% [+0.064, +0.194]). The >=200 prior
  possession sensitivity moves each by less than 0.03 points.
- **Contested shots is a documented reversal.** OU cleared the development
  gate (+0.211%) but loses to the re-tuned random walk on every stress
  season (-0.122%, -0.093%, -0.076%; pooled interval just below zero). The
  selection is frozen from development and is not switched on stress; the
  receipt records the loss. In practice the two filters are within 0.1% of
  each other for this stat, both are 6-19% better than the naive baselines,
  and the reversal is the reason to prefer the lab's paired-CI protocol over
  a bare one-SE gate before any further tracking work. Its 2026 export uses
  the frozen OU.
- **Contested DREB is the weakest real gain** (+0.084% dev, +0.127% stress;
  the 2023 season interval alone crosses zero at [-0.027, +0.201]). Its
  per-100 rate is small (mean 1.64) and its hustle field is missing on about
  5% of minutes.
- **Possession minutes is the strongest** (+2.5% over the random walk and
  +39% over career), consistent with a role stat that drifts within a
  season and mean-reverts slowly, exactly the shape the OU attractor fits.
- Gains versus the random walk are small in percent for the same reason as
  every next-game lane: single-game rows are mostly sampling noise. Paired
  season wins and intervals are the evidence, not the percent alone.

All twelve optimizers hit the 60-iteration budget without formal convergence
(random walk about 20 s per stat, OU about 57 s, single-threaded BLAS). As
with the PBP lab, `(s_mu, s_d, q_off, r)` are jointly scale-invariant for the
predictions, so the flag is cosmetic; treat each receipt as a fixed-budget
experiment.

Selected OU parameters (`tau_mu, s_mu, tau_d, s_d, q_off, r`) are in each
stat's `frozen.json`; mean 2019 rates per 100 are PA 8.97, drives 8.81,
possession minutes 4.19, contested shots 12.66, DFGA 19.62, contested DREB
1.64.

## Current states

`current_2026.csv` per stat holds one row per player with a 2026
observation: last measured date (archive ends June 13, 2026), the selected
model's postgame per-100 state, observed games, and career exposure. Counts:
582 for PA, drives, and possession minutes; 581 for contested shots and
DFGA; 578 for contested DREB. Leaders with at least 1,000 career
possessions of the relevant kind: PA Trae Young 27.27 and T.J. McConnell
27.13; drives McConnell 29.99 and Deni Avdija 28.41; possession minutes Jalen
Brunson 13.10; contested shots Donovan Clingan 22.17; DFGA Clingan 34.80;
contested DREB Nikola Jokić 5.41. These are last-game filtered estimates, not
projections advanced to a later date, and the latent variances are not
calibrated intervals.

## Deliverables and reproduction

`build_tracking_next_game.py` owns this lane. It reuses `run_filter`,
`run_filter_v2`, `build_baselines`, and the bounds from the shared engines,
and `compare` from `build_deflections_next_game.py`; each `frozen.json`
records SHA-256 hashes of the harness, both engines, the deflections harness,
and the panel, and `evaluate` refuses to run if any changed.

The same harness owns the fouls-drawn family (`NBA_FOULS_DRAWN_NEXT_GAME.md`;
`--family fouls_drawn`, output root `fouls_drawn_next_game/`, warm-up 2020
and development 2021-2022 because the CDN foul ledger starts in 2019-20).
Adding that family changed the harness hash after the six-stat run was
frozen, so `evaluate` on this root now refuses by design.
`harness_equivalence_2026-09-14.json` in this root records a scratch rerun
of all six stats under the extended harness: stress reports, stress cells,
predictions, and 2026 states were byte-identical.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_tracking_next_game.py prepare
for s in potential_assists drives possession_minutes contested_shots dfga contested_dreb; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
    nba_pipeline/scripts/build_tracking_next_game.py tune --stat "$s" &
done; wait
for s in potential_assists drives possession_minutes contested_shots dfga contested_dreb; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
    nba_pipeline/scripts/build_tracking_next_game.py evaluate --stat "$s" &
done; wait
.venv/bin/python nba_pipeline/scripts/build_tracking_next_game.py summary
.venv/bin/python -m pytest nba_pipeline/tests/test_tracking_next_game.py -q
```

`--stat all` runs the stats serially in one process. Run the six as separate
processes rather than a fork pool (fork pools crash on macOS after BLAS is
touched), and keep BLAS single-threaded.

Generated files under `results/estimated_skills/tracking_next_game/`
(gitignored):

- `panel.parquet`, `source_audit.json`: the joined measured-only panel with
  per-stat validity, input hashes, and coverage by stat and season.
- `<stat>/frozen.json`: selected model, parameters, development metrics,
  engine selection, bounds, DE progress, dependency hashes, selection rule.
- `<stat>/stress_report.json`: primary, player-clustered, established-player,
  and yearly comparisons on 2023-2025. No 2026 scores.
- `<stat>/development_cells.parquet`, `<stat>/stress_cells.parquet`:
  matched pregame predictions, observed per-100 targets, and exposure.
- `<stat>/predictions.parquet`: dated pregame and postgame per-100 states for
  both filters through 2026.
- `<stat>/current_states.csv`, `<stat>/current_2026.csv`: last measured
  states for all players and for players observed in 2026.
- `summary.md`, `summary.json`: the cross-stat table above and the top-five
  2026 states per stat.

Nine tests (`tests/test_tracking_next_game.py`) verify that the registry
uses only measured columns and exposure bases (no `*_filled` or `*_estimate`
column is referenced), the measured-only mask never touches repaired values,
the exposure join requires box agreement and unique keys with the right
reasons, the current game's count and exposure cannot change its pregame
prediction, baselines exclude the current game, season fallback and the
2026/2027 guards raise, possession minutes scale as expected, the one-SE
gate does not force OU, and (for the fouls-drawn family) reconciled zeros
are observations while unmeasured games are never scored.

## Caveats and next steps

- These are next-game per-100 estimates of tracking counts, not skills
  with a value scale, and they carry no opponent, teammate, home/rest,
  playoff, or team-change context. The lab winners on the PBP panel
  (multiplicative context, the playoff term, team-change variance inflation,
  cross-skill form) are the natural next step here, but this lane has now
  spent 2023-2025 as stress exactly as deflections did. A tracking lab has
  to declare 2020-2025 as development and lean on 2027.
- The playoff-quartile declines in `TRACKING_PLAYOFF_QUARTILES.md` remain
  descriptive. With a filter in place, the lab-style playoff term (flag plus
  playoff-times-excess) is the right test, not the additive offset that
  failed for deflections.
- Contested shots should be re-examined before it is built on: the OU gain
  did not survive stress, and the best explanation is that its dev gate was
  narrow rather than that the random walk is right.
- 2026 has been filtered and exported but never scored; 2027 is untouched.

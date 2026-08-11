# WNBA Predictive Box Panel (Kalman + Differential Evolution)

Built August 10, 2026. DARKO-style per-player, per-stat predictive box-score
panel for the WNBA. Implementation: `wnba_test/build_wnba_kalman_box.py`.
Outputs live under `wnba_test/Results/gameday/kalman_box/` (generated
artifacts, not in git).

## What it is

One independent Kalman filter per player per stat, tracking a latent
**per-minute rate** for each of 14 box stats: `pts, fgm, fga, fg3m, fg3a,
ftm, fta, oreb, dreb, ast, stl, blk, tov, pf`. Each observed game updates the
state with observation variance `r / minutes` (more minutes = more
informative game). Between observations the state variance grows by a
three-part drift:

```
q_game                    per game played
q_day * days_elapsed      calendar drift (injury/form between games)
q_offseason               once per season boundary
```

Per-stat hyperparameters `(v0, r, q_game, q_day, q_offseason)` are tuned by
differential evolution (seed `20260810`, `maxiter=30`, `popsize=12`, bounds
scaled from each stat's pre-2026 rate variance). The DE budget is
deliberately capped; runs terminate on the iteration limit rather than a
convergence claim, and the tuned parameters are accepted on their evaluation
results, not on optimizer status.

## Evaluation discipline

- Objective: **one-step-ahead minutes-weighted squared error in per-minute
  rate space**, i.e. every prediction for game N uses only games < N.
- Development: seasons 2023-2025, eligible observations require at least one
  prior observation (15,521 obs). 2022 warms the filters only.
- **2026 is a frozen-parameter holdout** (4,717 obs): hyperparameters were
  tuned on ≤2025 exclusively, then the filter was run forward through 2026
  without refitting.
- Baselines, all as-of: career-to-date rate, season-to-date rate (falling
  back to career at a player's first game of a season), and trailing-10-game
  rate.

## Results (August 10, 2026 build)

All 14 stats beat all three baselines on development **and** on the 2026
holdout. Holdout improvement vs the trailing-10 baseline ranges from about
6.2% (TOV, FGA) to 11.5% (PF) minutes-weighted MSE; PTS improves 7.8%,
AST 6.3%, DREB 7.3%. Improvements vs the season-to-date baseline are larger
(8.7%-14.3% on holdout) because early-season season-to-date estimates are
noisy exactly where the filter's carried state helps most.

Panel: 24,932 player-game rows, 358 players with current states.

## Outputs

| File | Contents |
| --- | --- |
| `hyperparams.json` | Per-stat DE-tuned `(v0, r, q_game, q_day, q_offseason)`, bounds, DE diagnostics, dev objective |
| `predictions.parquet` | One-step-ahead predicted rate + variance for every eligible player-game-stat |
| `current_states.csv` | Latest filtered rate and variance per player per stat (the "board") |
| `eval_report.json` | Full per-stat dev/holdout comparison vs all three baselines |

## Fences

- The panel is a **props/forecast-lane input**, not a rating: it never enters
  the box-prior ridge, the component solves, or any public RAPM/VPM surface.
  Per the standing rule, only stabilized player-season summaries could ever
  flow toward rating priors, and that would need its own forward gate.
- Predicted rates are per-minute; converting to per-game projections requires
  the minutes engine (`wnba_test/project_wnba_minutes.py`) — do not multiply
  by trailing averages.
- 2026 stays out of any refit until the season completes; in-season use is
  frozen-parameter filtering only (state updates are not "tuning").
- Rebuild command:
  `.venv/bin/python wnba_test/build_wnba_kalman_box.py` (add `--quick` for a
  reduced DE budget smoke run).

## Relationship to the props prereg

`docs/WNBA_PROPS_PREREG.md` names a per-game-average heuristic as the
baseline its distributional model must beat. This panel is the natural mean
engine for that model (rates × projected minutes), with the count
distribution layered on top. The prereg's binding split (develop 2023-2025,
shadow-confirm 2026) matches this panel's discipline, so its predictions can
be consumed there without spending anything new.

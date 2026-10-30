# Deflections: next-game per-100 protocol

Owner correction, September 12, 2026: the primary objective is **next-game
deflections per 100 defensive possessions**, not remaining-season skill.
The longer-horizon experiment stays intact as a named comparator.

Registered before fitting: reuse the exact observed-hustle/defensive-possession
panel and masks from `NBA_ESTIMATED_DEFLECTIONS.md`. Warm-up is 2019;
tune/select on 2020–2022; report 2023–2025 as already-spent historical stress;
2026 receives filtering/export only; 2027 remains the confirmation window.

For each observed game, make the prediction before ingesting that game's
count or exposure. Score `100 * deflections / defensive_possessions`, weighting
squared errors by defensive possessions within each season, then weighting
seasons equally. No per-game totals or projected minutes are involved.
Require at least one earlier measured observation for the shared evaluation
sample. Calendar gap and season crossing are known before the game. Missing
hustle games are unscored and do not become zero observations; the next
available measured game still uses only earlier measured information.

Tune random-walk and two-component OU filters directly on this loss, using
the earlier experiment's warm-up constants, bounds, log-space differential
evolution, seed 20260912, 60 iterations, population multiplier 15, tolerance
1e-7, and no polish. Compare both with the frozen longer-horizon selected
filter and strictly prior trailing-ten, season-to-date (career fallback),
and career baselines on the same rows.

Choose between the newly tuned filters on development using one SE: OU must
beat the random walk by more than one paired bootstrap SE. Then retain the
longer-horizon incumbent unless that next-game candidate beats it by more
than one SE on **next-game** loss. Write the frozen choice and parameters
before stress. No stress-based switching.

Primary uncertainty: 2,000 paired game-cluster bootstrap resamples within
each season, preserving all players in a game together. Report per-season
results and a >=200 prior-defensive-possession sensitivity for naive
baseline fragility. Repeated-player dependence across games is not fully
captured by game clustering; include player-clustered sensitivity as well.

Remaining-season cutoff error is secondary only. Keep raw frozen receipts
and output files separate under `results/estimated_skills/deflections_next_game/`.
The future-rate target cannot select this run's parameters or winner.

Export last-observed per-100 player states with dates. These are not forecasts
advanced to an arbitrary future date, and latent variances are not calibrated
uncertainty intervals. Nothing enters production RAPM/VPM.

## Completed next-game run

Development: 73,073 player-game rows across 3,623 games. The next-game OU
candidate beat the next-game random walk by 0.191% MSE and the frozen
longer-horizon OU by 0.176%; both clear the registered one-SE gates. The
selected label is `next_ou`. Both optimizers hit their 60-iteration budgets
without formal convergence; preserve the receipt as a fixed-budget experiment.

Historical 2023–2025 stress: 78,291 player-game rows across 3,931 games.
Primary MSE is **7.08946** and RMSE **2.66260 deflections per 100 defensive
possessions**. Comparators use the exact same rows and exposure weights.

| Comparator | Next-game MSE improvement | Game-clustered 95% interval |
| --- | ---: | ---: |
| Frozen longer-horizon OU | **0.328%** | **0.258%–0.400%** |
| Next-game-tuned random walk | 0.129% | 0.057%–0.202% |
| Trailing ten observations | 9.056% | 7.187%–12.270% |
| Season-to-date / career fallback | 10.599% | 7.722%–14.490% |
| Career average | 4.723% | 2.779%–8.065% |

The main incumbent comparison also survives player clustering:
**0.328% [0.237%, 0.428%]**. Its year-specific improvements are +0.123% in
2023, +0.192% in 2024, and +0.599% in 2025. These are small but consistent
historical next-game gains, not a claim of large practical lift or untouched
confirmation.

With at least 200 prior defensive possessions, gains are 0.335% versus the
longer-horizon incumbent, 0.134% versus next-game random walk, 6.317% versus
trailing ten, 6.506% versus season-to-date, and 1.578% versus career. This
removes extremely small-history naive-baseline failures from the comparison.

The secondary remaining-season test is **1.357% worse** than the original
longer-horizon model, with a gain interval [-3.261%, +0.454%]. The point
estimate illustrates the horizon tradeoff, but this secondary difference is
not resolved. It did not affect selection.

## Deliverables and reproduction

`build_deflections_next_game.py` owns this run. The older
`build_estimated_deflections.py` and its frozen artifacts are unchanged.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_deflections_next_game.py tune
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_deflections_next_game.py evaluate
.venv/bin/python -m pytest nba_pipeline/tests/test_deflections_next_game.py -q
```

Generated files under `results/estimated_skills/deflections_next_game/`:

- `frozen.json`: selected model, parameters, development comparisons, and
  hashes of the harness, shared engines, source panel, and incumbent receipt.
- `stress_report.json`: primary, yearly, player-clustered, established-player,
  and secondary-horizon comparisons. No 2026 target scores.
- `development_cells.parquet`, `stress_cells.parquet`: matched pregame
  predictions, observed per-100 targets, and exposure weights.
- `remaining_season_cells.parquet`: secondary diagnostic only.
- `predictions.parquet`: dated pregame and postgame per-100 states for all
  three filters, including frozen filtering through 2026.
- `current_states.csv`: all 1,261 players' last measured states.
- `current_2026.csv`: **581 players**, each with its actual last-observation
  date (archive ending June 13, 2026), model label, and supporting exposure.

The current table is led by Thybulle 9.08, Cason Wallace 7.46, Dru Smith
7.42, Ausar Thompson 7.01, and Jordan Goodwin 6.89 per 100 (at least 1,000
historical observed defensive possessions). These are last-game estimates,
not September projections.

Five regression tests verify that current-game counts/exposure cannot affect
pregame estimates, baselines exclude the current game, season fallback and
score windows are correct, season weighting and paired-cluster controls work,
and the selection rule does not force a challenger. Final artifact checks
also verify immutable receipts, disjoint development/stress windows, no 2026
scores, and unique finite player exports.

This changes the selected **deflections** research estimate. The other 17
historical skills have not yet been retuned for next-game accuracy, and no
production prior or public metric has been changed.

Playoff follow-up: [NBA_DEFLECTIONS_PLAYOFF_CONTEXT.md](NBA_DEFLECTIONS_PLAYOFF_CONTEXT.md) tests an additive prior-season playoff offset. It worsens historical stress playoff predictions; the current OU remains unchanged.

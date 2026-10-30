# Rolling predictive rim points saved / 100

September20 repair: recovered cumulative2020–2022 RS counts are documented in
[Rim measurement repair](RIM_DEFENSE_MATCHUP_SKILLS.md). Versioned refreshed
states are `results/estimated_skills/predictive_rim_saved_recovered_v3/`.
Use `build_predictive_rim_saved.py --source <repaired player_games.parquet>
--out <new directory> --frozen <original frozen.json>` to replay the same730-day
volume/suppression and K100 parameters without retuning or opening new scores.
Original fit diagnostics inside the copied frozen receipt describe the original
fit, not re-evaluated performance on repaired data. The fixed parameters remain
historically selected; source repair alone does not make replay leakage-free.
The original artifacts and saved v18 ratings remain unchanged.


Research estimate built September 14, 2026. Run:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python nba_pipeline/scripts/build_predictive_rim_saved.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python nba_pipeline/scripts/compare_rim_saved_blocks.py
```

Artifacts: `results/estimated_skills/predictive_rim_saved/`, containing
`frozen.json`, `stress_report.json`, `blocks_control.json`,
`dated_predictions.parquet`, `current_2026.csv`, and searchable `index.html`.
No replacement of the main estimated-skill table or VPM inputs is made.

## Source correction

The rolling experiment exposed repeated season totals in the original unprefixed
2020-2022 fields: about 194-213 rim contests per 100 defensive possessions.
The measurement builder now rejects that entire field family, preserving
missingness, and checks season contest volume for implausible totals. The earlier
measurement-layer coverage claim is superseded in RIM_DEFENSE_MATCHUP_SKILLS.md.
Only explicit prefixed game-level counts enter. Supported history is 2019 and
2023-2026; absent seasons do not create zero observations.

## Meaning and algorithm

This is **league-relative six-foot contest savings**, not the provider's
opponent-normalized retrospective savings. Let q be the league defended-six-foot
make probability computed from all available earlier dates, with a 65/100 initial
pseudo-count. Every row on a date receives the same baseline; current-day games
cannot enter it. Provider normal percentages and annual BAM outcomes never enter.
The changing pool of covered defenders can still affect this league baseline.

Maintain two exponentially discounted sufficient-statistic pairs for each player:
contest attempts / defensive possessions, and `(q*attempts - makes)` / attempts.
Predict before consuming the current game. Missing games do not update either
state; elapsed calendar time still discounts previous observations. Measured
zero-attempt games update volume but not finishing suppression.

Contest volume is `(discounted attempts + 1000*2019 league volume) /
(discounted defensive possessions + 1000)`. Finishing suppression is discounted
excess misses divided by `(discounted attempts + K)`, shrunk toward zero and
bounded so predicted FG% remains in [0,1]. Then:

`predicted saved points/100 = 2 * predicted contests/100 * predicted suppression`.

Positive is better. Export pregame and postgame states, underlying opportunity
rates and suppression, allowed FG%, and exact dates. Display is the latest
covered postgame state, **not advanced to today's date**. The current inventory
contains 577 players with a 2026 observation; it is not a verified active roster.
No opponent-specific forecast, rest/playoff adjustment, age/physical prior,
block prior, deterrence credit, free throws, or rebound continuation is included.

The model's whole-possession interpretation is conditional on covered tracking
observations. Small/missing-game selection can bias volume; 2026 coverage is
exported. Historical source revisions are not guaranteed to be contemporaneous.
No posterior uncertainty interval is exported; bootstrap intervals below measure
predictive comparison uncertainty, not uncertainty for individual players.

## Selection and results

2019 initializes the league volume prior. With 2020-2022 rejected, development
is 2023; 2024-2025 are historical stress and 2026 is export-only. The initial
invalid run was discarded after a source-integrity failure. These are historical
development results, not a fresh untouched holdout. 2027 remains protected.

Volume half-life grid: 90/365/730 days, selected on exposure-weighted next-game
contest-rate MSE. Suppression half-life grid: 365/730/1460 days, K=100/400/1600
attempts, selected on next-game saved-points/100 MSE with a paired game-clustered
one-standard-error preference for larger K, then longer half-life. Volume prior
exposure is fixed at 1000 possessions. Selected: **730 days for both states,
K=100 defended attempts**. Hyperparameters remain fixed during stress/export.

On 48,721 player-games in 2,623 games, equal-season weighted:

| Comparison | MSE reduction | Game-clustered 95% interval |
|---|---:|---:|
| Saved points vs zero personalized suppression | 1.220% | [0.986%, 1.450%] |
| Allowed rim points vs same volume + league FG% | 0.323% | [0.139%, 0.486%] |
| Saved points vs blocks-only mapper | 0.091% | [-0.092%, 0.252%] |
| Blocks + rim estimate vs blocks-only mapper | 0.249% | [0.163%, 0.330%] |

Saved-points gains versus zero are 1.300% in 2024 and 1.135% in 2025; the
player-cluster sensitivity interval is [0.820%, 1.622%]. Allowed-points errors
score actual `2*makes/defensive possessions*100`, not just percentage errors.
Realized defensive exposure is supplied, so this is not a minutes/game forecast.

The blocks control uses a fixed 365-day discounted block rate, 1000-possession
prior from 2019, and Ridge(alpha=1000) with an intercept trained only on earlier
seasons. Its joint version adds the pregame rim estimate. Both are diagnostic;
no post-stress model swap was made. Rim finishing supplies incremental information
alongside blocks, but the standalone estimate has not clearly beaten blocks.
There is no direct EPM comparison or future-possession VPM admission here.

Three timing/algebra tests cover no current/future outcomes in the prediction,
exact product closure, measured zero-attempt behavior, and exclusion of the
entire current date from the league baseline. Together with measurement tests,
six tests pass.

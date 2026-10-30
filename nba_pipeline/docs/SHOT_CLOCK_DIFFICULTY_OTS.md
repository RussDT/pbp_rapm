# Shot-Clock Difficulty and Clock-Adjusted TS

`scripts/analyze_shot_clock_difficulty_ots.py` builds a player context stat
from the 2024-2026 shot-clock endpoint surface and tests whether it helps
explain or predict offensive TS RAPM.

## Definitions

Each FGA or ordinary live-ball FT trip is priced by a game-held-out league
clock curve. The curve is fit separately by season and by 24- versus 14-second
reset family with an isotonic TSx2 regression. The endpoint's own game is never
used to set its expectation.

For a player or player-season:

```text
shot-clock difficulty
  = league-reference TS% - clock-expected TS%

clock-adjusted TS%
  = raw TS% + shot-clock difficulty
```

Positive difficulty means the player's clock mix was harder than the league's
average attempt mix. The identity closes exactly on every aggregate row. The
stat includes ordinary FT attempts by mapping them to the shooter and the foul
or and-one clock. The 2024-2026 build has 41 anchors with conflicting listed FT
shooters; those shooter assignments are excluded instead of choosing one.

The primary clock is the raw rule clock. It has only whole-second observational
resolution, but it is available and stable through zero. A sensitivity surface
uses the FGA-calibrated clock with every value below four seconds floored at
four; this avoids presenting the known sparse calibrated tail as exact timing.

## Run

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/analyze_shot_clock_difficulty_ots.py
```

Generated research artifacts are under
`results/shot_clock_difficulty/2024_2026/`:

- `shot_clock_difficulty_player_season.csv`
- `shot_clock_difficulty_24_26.csv`
- `shot_clock_difficulty_crossfit_curves.csv`
- `shot_clock_difficulty_ots_regression_metrics.csv`
- `shot_clock_difficulty_future_ots_regression_metrics.csv`
- `shot_clock_difficulty_zts_incremental_metrics.csv`
- `shot_clock_difficulty_zts_overlap.csv`
- `zts_role_numbers_player_season.csv`
- matched analysis rows, standardized coefficients, and held-out predictions

## oTS regression result

The descriptive 2024-2026 regression uses 508 players and the point-valued
public oTS target. Replacing raw TS-added with raw clock-adjusted TS-added lowers
repeated player-fold RMSE from `1.1427` to `1.1279` (`-0.0148`), but its paired
bootstrap interval `[-0.0355, +0.0039]` crosses zero. Adding difficulty as a
separate feature lands at `1.1238`, also directionally favorable with an
interval that crosses zero.

The stricter season-forward test uses 2024 features to predict 2025 annual oTS
and 2025 features to predict 2026 annual oTS. Each transition is trained only
on the other transition. Across 746 player-seasons / 444 players:

| Model | Future oTS RMSE | Change vs box scoring |
| --- | ---: | ---: |
| Box scoring baseline | 1.5648 | — |
| Raw clock-adjusted TS-added | 1.5349 | -0.0299 |
| Calibrated-floor-4 clock-adjusted TS-added | 1.5354 | -0.0294 |
| Box scoring + raw difficulty | 1.5404 | -0.0244 |

The raw adjusted replacement improves both target years: `1.5944 -> 1.5432`
for 2025 and `1.5346 -> 1.5265` for 2026. Its player-clustered bootstrap interval
for pooled RMSE change is `[-0.0574, -0.0016]`. Raw difficulty alone has interval
`[-0.0426, -0.0056]`.

## Relationship to zTS / playtype difficulty

The existing playtype-adjusted scoring stat is:

```text
zTS = pt_adj_rTS = rTSPct + playtype_diff
```

`playtype_diff` prices the expected TS of the player's playtype mix relative
to the league mix. It is an upstream role-burden adjustment. Its four
explanatory buckets are creation (handler, isolation, post-up), shooting
(spot-up, off-screen, handoff), early offense (`ee`: putback, cut,
miscellaneous, roll man), and transition.

`zts_role_numbers_player_season.csv` contains role TSA/100, attempt share,
actual TS%, playtype-expected TS%, and within-role rTS for 2024-2026. The
source has current 2026 lower-level rows even though the thin
`player_stats_export.csv` had not refreshed its 2026 zTS columns.

Playtype difficulty and raw clock difficulty have moderate player-season
correlation (`Pearson 0.317`, `Spearman 0.314`) on the 746-row forward sample,
but their predictive gains overlap heavily:

| Exact-common-sample comparison | Baseline RMSE | Challenger RMSE | Change | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| zTS vs box, same window | 1.1427 | 1.0918 | -0.0509 | [-0.0803, -0.0234] |
| Clock on top of zTS, same window | 1.0918 | 1.0849 | -0.0069 | [-0.0259, +0.0119] |
| zTS vs box, season forward | 1.5648 | 1.5352 | -0.0296 | [-0.0552, -0.0043] |
| Clock on top of zTS, season forward | 1.5352 | 1.5295 | -0.0057 | [-0.0143, +0.0025] |

The combined replacement improves both forward target years, but the
increment beyond zTS is small and its player-clustered interval crosses zero.
The current reading is that zTS absorbs most of the useful shot-clock role
signal. Clock remains a useful diagnostic and a possible within-playtype
burden feature, not a justified standalone production upgrade.

### Decompressed `Points + TSA` test

The original zTS note reported `R^2 = 0.536` for the unconstrained
`Points + TSA` model on its then-current 2022-2026 artifact. Because the box
and playtype source tables have since refreshed, that exact level no longer
reproduces; the meaningful comparison is the increment on one current common
sample.

The raw-clock endpoint surface was therefore backfilled through 2022, limited
to regular seasons, and tested against the same pure
`ts_22_26_rs_a3000_5000` oTS target. Clock difficulty is converted to value as:

```text
clock difficulty points/100
  = 2 * TSA100 * clock difficulty percentage points / 100
```

| Current 2022-2026 model | Full-sample weighted R² | Repeated-fold OOF R² |
| --- | ---: | ---: |
| Points + TSA | 0.4977 | 0.4867 |
| + clock-difficulty points | 0.5052 | 0.4922 |
| + playtype-difficulty points | 0.5889 | 0.5766 |
| + playtype and clock | 0.5889 | 0.5733 |

Thus clock adds `0.0075 R^2` in the unconstrained descriptive fit and `0.0055`
OOF. At `5000+` offensive possessions the increments are `0.0103` and
`0.0064`, respectively. Once playtype difficulty is present, clock adds
essentially zero in-sample and slightly worsens repeated-fold OOF. On the
scale of the older `0.536` snapshot, the standalone clock increment is roughly
an increase to `0.544`, not remotely the playtype jump to `0.618`.

This is promising development evidence, not a production admission. There are
only two season transitions, both were examined during development, and the
clock effect still bundles transition opportunity, shot location, offensive
role, and late-clock burden. The next test should freeze the definition and
score a fresh season or a richer predeclared historical clock-label surface.

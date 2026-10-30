# WNBA Playtype / Shot-Role Difficulty

## Question

The NBA scoring chain uses `zTS = rTS + playtype_diff`, where `playtype_diff`
originally came from Synergy and is now better served by the frozen six-family
PBP proxy in `PBP_PLAYTYPE_DIFFICULTY_OTS.md`. The WNBA has no Synergy feed and
no equivalent role-difficulty surface at all. Can one be recovered from what we
already own?

Yes. WNBA play-by-play descriptions carry the same action vocabulary as the
NBA's, so the NBA taxonomy ports without modification, and independent Second
Spectrum tracking confirms the resulting labels mean what they claim.

## Build

`wnba_test/build_wnba_playtype_difficulty.py` reads regular-season play-by-play
straight from `nba_pipeline/external/gabriel_wnba/pbp_data/` (game ids
`102YYNNNNN`). No lineup processing is involved: the feature needs only the
shooter, the description, and the shot value, so it is built from the raw
per-game files rather than the normalized lineup surface.

Each field-goal attempt is assigned one of the six frozen NBA families using the
identical regular expressions:

- `CUT_2`, `SELF_CREATED_3`, `OTHER_3`, `SELF_CREATED_2`, `OTHER_RIM_2`, `OTHER_2`

Families are priced from the immediately preceding completed regular season,
shrunk toward the prior-season league mean with 100 FGA, and the feature
season's player mix is valued at those fixed prices. Make/miss, realized points,
score state, and assist tags never enter the family assignment.

Representative 2026 prices (points per FGA, prior season 2025, league `1.000`):

| Family | Price | Reading |
|---|---:|---|
| `CUT_2` | 1.417 | easiest |
| `OTHER_RIM_2` | 1.196 | |
| `OTHER_3` | 1.037 | |
| `SELF_CREATED_3` | 0.935 | |
| `SELF_CREATED_2` | 0.863 | |
| `OTHER_2` | 0.822 | hardest |

## The one required WNBA change: season centering

The NBA definition subtracts the *prior* season's league points per FGA. That
is safe in the NBA because its description vocabulary is stable. The WNBA's is
not, and the drift is large enough to swamp the signal.

| Season | `CUT_2` share | `OTHER_2` share | `driving` token |
|---|---:|---:|---:|
| 2013 | 0.001 | 0.412 | 6.9% |
| 2018 | 0.043 | 0.192 | 13.2% |
| 2022 | 0.054 | 0.077 | 16.5% |
| 2026 | 0.066 | 0.036 | 22.7% |

Scorers kept adding descriptors. Under the NBA convention that shows up as a
pure labeling artifact in the level: the league-mean difficulty swings from
`+0.64` (2013) to `-3.07` (2018) TS points without any real change in how hard
WNBA shots were.

The production field `playtype_diff` therefore subtracts the *feature season's
own* FGA-weighted mean instead. This is a per-season constant, so it changes no
player's rank and no spread — it only removes the artifact. `playtype_diff_prior_league`
retains the NBA-exact convention for cross-league comparison.

Seasons before 2018 are built but carry `descriptor_regime = sparse_descriptions`
and must never be used as a feature or a price. Before 2016 the string `cutting`
does not appear in WNBA play-by-play at all, and `SELF_CREATED_3` is so rare
(0.1% of attempts) that its price is noise.

## Face validity

2026 hardest diets: Carla Leite, Courtney Williams, Teja Oblak, Kelsey Plum,
Olivia Miles, Paige Bueckers, Caitlin Clark, Chelsea Gray. Easiest: Natasha
Howard, Kamilla Cardoso, Naz Hillmon, Dearica Hamby, Angel Reese.

Self-creating guards on one end, cut-and-putback bigs on the other. Same-player
year-over-year correlation is `0.75`-`0.89` across modern seasons, so role is a
far more stable trait than shooting efficiency itself (`0.45`).

## External validation against tracking

`shots.pbpId` in the RTSS export aligns to the play-by-play event number, which
gives a shot-level join with no player crosswalk required: `100,201` shots over
739 games in 2023-2025, `91.9%` of RTSS shots with a `qSQ`, and `100%` agreement
on the two/three flag.

Tracking descriptors confirm that the description-derived families mean what
they claim, having never seen a description:

| Family | Catch-and-shoot | Mean dribbles before | Putback share |
|---|---:|---:|---:|
| `SELF_CREATED_3` | 0.058 | 3.79 | 0.000 |
| `SELF_CREATED_2` | 0.063 | 3.76 | 0.010 |
| `OTHER_3` | 0.949 | 0.14 | 0.000 |
| `CUT_2` | 0.809 | 0.22 | 0.001 |
| `OTHER_2` | 0.750 | 0.78 | 0.026 |
| `OTHER_RIM_2` | 0.510 | 1.44 | 0.238 |

`OTHER_RIM_2` is the one genuinely mixed bucket: it is where putbacks live.

### qSQ is not on the WNBA points scale for threes

Provider `qSQ` cannot be used raw. Pooled over the joined sample it is nearly
calibrated on twos and badly inflated on threes, stably across all three
seasons:

| Season | Zone | Mean `qSQ` | Actual FG% | Gap |
|---|---|---:|---:|---:|
| 2023 | THREE | 49.74 | 34.70 | +15.04 |
| 2024 | THREE | 49.28 | 33.90 | +15.38 |
| 2025 | THREE | 49.66 | 33.72 | +15.94 |
| 2023 | TWO | 46.20 | 48.63 | -2.43 |
| 2024 | TWO | 47.35 | 48.84 | -1.48 |
| 2025 | TWO | 47.84 | 49.53 | -1.70 |

The ranking is also weaker on threes: the 2025 decile calibration slope is
`0.956` on twos but `0.704` on threes, and the three-point decile spread is
compressed (`qSQ` spans `29.9`-`62.0` where actual make rates span `19.4`-`43.6`).
This sharpens the existing note that raw three-point `qSP-qSQ` has a level
shift — the level problem is in `qSQ` itself, not only in the difference.

### Agreement after recalibration

Rescaling `qSQ` to the WNBA points scale within season and zone, tracking and
play-by-play agree on family values almost exactly:

| Family | Tracking-implied points | Actual points |
|---|---:|---:|
| `CUT_2` | 1.396 | 1.461 |
| `OTHER_RIM_2` | 1.211 | 1.178 |
| `OTHER_3` | 1.048 | 1.045 |
| `SELF_CREATED_3` | 0.918 | 0.930 |
| `SELF_CREATED_2` | 0.851 | 0.865 |
| `OTHER_2` | 0.821 | 0.764 |

Family-level correlation `0.99`. At player-season level (100+ joined attempts)
the proxy and the tracking measure correlate Pearson `0.760` / Spearman `0.751`
pooled over 318 player-seasons, with the tracking measure carrying about `1.7x`
the spread and a regression slope of `1.298`.

So the PBP proxy recovers the shot-family ordering essentially completely and
about three-quarters of the player-level ordering. The residual is what
descriptions physically cannot see: defender distance and contest level. Angel
Reese is the clearest example — the proxy reads her rim-and-putback diet as easy
(`-2.55`) while tracking, which sees the contests, reads it as hard (`+2.55`).

## Is the number the right size?

Ranking correctly is not enough: `playtype_diff` is denominated in TS percentage
points so that it can be *added* to `rTS`. The scale test is whether one point
of extra diet difficulty costs one point of observed TS.

Weighted regression of next-season TS on next-season role, over 426
player-season pairs from feature seasons 2019-2024, with player-clustered
bootstrap:

| Specification | Slope | 95% CI |
|---|---:|---|
| next-season role | **-0.814** | [-0.995, -0.638] |
| current-season role | -0.386 | [-0.606, -0.184] |

The metric is close to correctly scaled but mildly over-dispersed. **Use
`0.81 x playtype_diff`, not the raw value, when adding it to `rTS`.**

### The shortfall belongs to the descriptor proxy, not to the WNBA

Running the identical specification on NBA data separates the two explanations.
Production NBA `zTS` imposes a coefficient of exactly `1.0`, and on the measure
it is actually applied to — Synergy `playtype_diff` — that is correct. But the
PBP descriptor proxy fits short in the NBA too, by about the same margin as
here:

| League | Role measure | Slope | 95% CI | Rows |
|---|---|---:|---|---:|
| NBA | Synergy `playtype_diff` | -1.044 | [-1.211, -0.875] | 592 |
| NBA | PBP proxy, same rows | -0.861 | [-0.959, -0.755] | 592 |
| NBA | PBP proxy, 12 transitions 2014-2026 | -0.767 | [-0.818, -0.712] | 3,858 |
| WNBA | PBP proxy | -0.814 | [-0.995, -0.638] | 426 |

The two NBA measures correlate `0.915` and still disagree on scale, and the NBA
per-transition point estimate is weaker than `1.0` in 12/12 seasons (CI excludes
`-1.0` in 9/12). So `0.81` is a property of pricing a *description-derived* shot
mix, not a WNBA-specific correction, and it is not evidence against the NBA
production constant.

This also flags a hazard for `PBP_PLAYTYPE_DIFFICULTY_OTS.md`, which proposes
the PBP proxy as the leading candidate to replace Synergy in a scoring prior. If
that swap is ever made by dropping the proxy into the `zTS` formula at the
inherited `1.0`, it would be over-scaled by roughly 15-25%, and because `zTS`
feeds the volume conversion and then `SV/100` at slope `0.800618`, the error
would reach the published leaderboard field rather than staying in a rate
column. The proxy needs its own fitted coefficient.

## Forward test, and why the naive adjustment loses

Leave-one-season-out ridge, target next-season TS%, weighted by next-season TSA,
identical rows and alpha selection for every model:

| Model | RMSE | Change vs box | 95% CI | P(improves) |
|---|---:|---:|---|---:|
| Box baseline | 4.0534 | — | — | — |
| Imposed 1:1 adjustment | 4.3220 | **+0.2686** | [+0.143, +0.402] | 0.000 |
| Box + difficulty | 3.9940 | -0.0595 | [-0.143, +0.023] | 0.923 |
| Box + difficulty + next-season role | 3.9179 | **-0.1355** | [-0.246, -0.038] | 0.996 |

The NBA's exact spec — replace raw efficiency with difficulty-adjusted
efficiency — is clearly **worse** than doing nothing, losing in 6/6 seasons.
That is not a defect in the measure; it is the cancellation mechanism, and the
numbers pin it down:

- role persistence slope `rho = 0.757`
- TS carry-over slope `lambda = 0.397`
- role price `beta = 0.814`

Predicted current-role slope `beta x (lambda - rho) = -0.293`, measured `-0.386`
(CI `[-0.606, -0.184]`). Role persists roughly twice as strongly as shooting
efficiency does. So a hard diet this season mostly predicts a hard diet next
season, which drags next season's raw TS *down* by more than the skill
adjustment lifts it. Adjusting the input without re-applying the output role
double-counts in the wrong direction.

Once next season's role is supplied, the pairing is worth `3.34%` RMSE with a CI
excluding zero and a win in 6/6 seasons. That variant is a mechanism diagnostic,
not a forecast — it uses information unavailable at forecast time.

The contemporaneous check is unambiguous in every season: difficulty correlates
`-0.27` to `-0.45` with same-season observed TS.

## How to use it

1. **Reader-facing role-adjusted efficiency.** `WNBA rTS_role = rTS + 0.81 x playtype_diff`.
   This is the skill-estimate lane and it is what the measure is for. Label it
   role-adjusted efficiency, never shot quality, playmaking, or impact.

   For display, use the pre-scaled column so no arithmetic happens at read
   time: `playtype_diff_scaled = ROLE_DISPLAY_COEFFICIENT x playtype_diff`,
   giving the exact identity `rTS + playtype_diff_scaled = rTS_role`.
   **`playtype_diff` itself stays raw and must not be pre-scaled in the
   build.** Only one of this section's three consumers wants the coefficient:
   forecasting (2) explicitly must carry the raw value as its own regressor,
   priors (3) fit their own weight against a component target, and the NBA
   `playtype_diff` columns — Synergy and PBP proxy alike — are unscaled, so a
   pre-scaled WNBA feature would silently become a different object sharing a
   name and would break the NBA-to-WNBA hierarchical prior lane. The
   coefficient is recorded in `manifest.json` as `role_display_coefficient`
   so any stored artifact can be traced to the value it was built under.

   **Do not call this `zTS`.** `zTS` is the NBA production object defined at
   `docs/README.md` and in the root instruction files as `zTS = rTS +
   playtype_diff` with a coefficient of exactly `1.0`, and it is only the rate
   step of a chain that continues `zTS add-4 = 2 * TSA100 * (zTS + 4) / 100` and
   `SV/100 = -1.211295 + 0.800618 * zTS add-4`, where `SV/100` is the published
   leaderboard field. The WNBA quantity differs on both counts: its coefficient
   is the fitted `0.81`, not `1.0`, and no WNBA volume conversion or `SV/100`
   calibration exists. The `+4` shift and both regression constants are
   NBA-fitted and must not be transplanted. If a WNBA scoring-value field is
   ever wanted, the volume step and its constants have to be refit on WNBA data
   as their own exercise.
2. **Forecasting.** Never substitute adjusted for raw efficiency. Carry
   `playtype_diff` as its own regressor and pair it with a projection of next
   season's role.
3. **Priors.** A candidate feature for the MID/THREE shooting components in the
   `build_wnba_box_priors.py` lane. It is not promoted; see the gate below.

## What this is not

Contemporaneous correlation with the annual no-prior oTS RAPM components is
near zero (`0.15`, `0.03`, `0.01` for 2023-2025). Do not read that as a
refutation and do not read it as support: annual WNBA component RAPM at a
500-possession floor is far too noisy to discriminate anything at this effect
size. The box-TS forward test is the informative one and the RAPM surface is
not a usable target until it is pooled or prior-centered.

This measures the *value* of the shot mix a player takes, not conversion
difficulty and not contest. It cannot see pick-and-roll handling, handoffs,
off-screen actions, screening, or turnover-bearing playtype possessions.

## Status and gate

Development evidence. Feature seasons 2019-2024 and target seasons 2020-2025 are
spent. **2026 is untouched** and is the confirmation season: rerun with
`--last-season 2025` once 2026 is complete, and require the scale slope to hold
near `-0.81` and `plus_difficulty` to beat the box baseline before anything is
promoted to a published surface or to a prior. Do not re-tune the `0.81`
coefficient on 2026.

## Publication: the databallr `Role` and `zTS` columns

The WNBA stats table on databallr shows two columns immediately after `rTS%`:

| Column | Value | Scale |
|---|---|---|
| `Role` | `0.81 x playtype_diff` | TS percentage points |
| `zTS` | `rTS% + Role` | TS percentage points |

`zTS` is summed on the client against the same `rTS%` the row renders, so the
identity closes on the numbers a reader can see. The raw `playtype_diff` is
never displayed and never leaves the research artifact.

**What shipping this column does and does not claim.** This is a *descriptive*
shot-mix difficulty reading, not a promoted predictive feature. The forward gate
in `Status and gate` below is unchanged and still unmet: `Role` is not a prior,
not a VPM input, and not a forecasting feature, and nothing in the production
WNBA rating lane consumes it. What does ship ahead of that gate is the `0.81`
coefficient itself, which is fitted on development seasons 2019-2024. If the
2026 confirmation moves that slope, the displayed values move proportionally --
the ordering and the sign do not, since the coefficient is a single positive
constant.

Two publication rules are enforced in `publish_wnba_playtype_role.py` rather
than trusted downstream, because a consumer cannot re-derive them:

- **Modern seasons only (2018+).** Rows flagged `sparse_descriptions` never
  publish. Pre-2018 WNBA descriptor vocabulary drifts hard enough to read as
  league-wide difficulty rather than player role.
- **Regular season only.** The artifact is built from regular-season
  descriptions, so playoff rows are withheld rather than inheriting a
  regular-season role. The backend merge re-checks this.

A `50` PBP-FGA floor also applies, matching the attempt floor databallr already
uses to qualify a player for overall-shooting percentiles. Below it the
six-family mix is noise: a 1-FGA player can post a `-12` role.

### Flow

```
gabriel_wnba PBP  ->  build_wnba_playtype_difficulty.py
                  ->  publish_wnba_playtype_role.py  (filter, scale, validate)
                  ->  Supabase wnba_playtype_role     (live path)
                  +   databallr public/*.csv          (committed fallback)
                  ->  _apply_playtype_role in backend/app/routes/supabase/wnba.py
                  ->  WNBA_ADVANCED_FIELDS in src/services/api.ts  (projection)
                  ->  computeRow in src/utils/wnbaAdvanced.ts  (zTS = rTS + Role)
                  ->  Role / zTS columns in WNBAStatsTable.tsx
```

The projection step is load-bearing and easy to miss. `/wnba_advanced_stats`
supports an opt-in `fields=` list and the client always sends one, so a merged
key that is not in `WNBA_ADVANCED_FIELDS` is dropped by `_project_rows` and the
column renders blank against a `200` response. `role` and `playtype_role` are
both listed, and a frontend test asserts it.

The daily refresh is `wnba_test/run_daily_playtype_role_refresh.sh`, scheduled by
`com.russellthomas.wnba-playtype-role-refresh` at 06:00 local. It syncs the
Gabriel checkout, rebuilds, republishes, upserts Supabase, and stages the
fallback CSV. It never commits or deploys databallr. Out of season it is a
byte-identical no-op. A refreshed board reaches readers within roughly 25
minutes: a 900s role-map cache plus the route's 600s response cache.

Tests: `nba_pipeline/tests/test_wnba_playtype_role_publish.py`,
databallr `backend/tests/test_wnba_playtype_role.py`, and databallr
`scripts/wnba-playtype-role.test.ts`.

## Reproduction

```bash
.venv/bin/python wnba_test/build_wnba_playtype_difficulty.py
.venv/bin/python wnba_test/evaluate_wnba_playtype_difficulty.py

# Reader-facing Role / zTS board (add --upload and --stage-databallr to ship).
.venv/bin/python wnba_test/publish_wnba_playtype_role.py --rebuild
```

Artifacts land under `wnba_test/Results/playtype_difficulty/`:
`wnba_playtype_player_season.csv` (carrying raw `playtype_diff`, the NBA-exact
`playtype_diff_prior_league`, and display-only `playtype_diff_scaled`),
`wnba_playtype_category_prices.csv`,
`wnba_playtype_league_category_mix.csv`, `manifest.json`, and `evaluation/`
with `forward_metrics.csv`, `scale_calibration.csv`, `forward_comparisons.csv`,
`forward_per_season.csv`, `contemporaneous.csv`, and `panel.csv`.

Tests: `nba_pipeline/tests/test_wnba_playtype_difficulty.py`.

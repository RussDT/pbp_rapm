# Offensive Impact Portability

`scripts/analyze_offensive_impact_portability.py` tests whether a player's
regular-season offensive impact changes differentially as the defensive
personnel on the floor improve. The research question is the O-EPM × opposing
lineup D-EPM interaction, not the trivial main effect that better defenses
allow fewer points.

## Result

The 2018-19 through 2025-26 evidence rejects the motivating direction. Better
offensive players do **not** retain more relative impact against stronger
defensive personnel. The primary interaction is `-0.060` points per 100 per
`+1` frozen O-EPM and `+1` opposing five-man D-EPM, with a game-clustered 95%
interval of `[-0.076, -0.044]`. Positive would mean that stronger offensive
players translate better; the observed negative coefficient means that they
decline slightly more.

The magnitude is real but modest. Moving from the season-centered average
defensive environment to the regular-season top decile changes the estimated
relative impact of the approximately `0 O-EPM` tier by `-0.17` points per 100
and the `+4 and above` tier by `-0.64`. The gap compresses by about `0.47`
points per 100, roughly 9% of the tiers' baseline O-EPM separation. Offensive
impact is therefore mostly portable in practical terms, with a detectable
nonlinearity opposite the original mediocre-player-collapse hypothesis.

The direction survives:

- possession, equal-player-season, and equal-player weighting;
- removal of all player-games containing flagged garbage-time possessions;
- prior-season June 30, frozen December 1, and rolling pregame O-EPM;
- teammate-quality and other control ablations;
- a same-sample end-season EPM diagnostic;
- all eight seasons as negative point estimates, with five season intervals
  below zero.

The shuffled-defense interaction is `-0.013 [-0.028, +0.001]`; the next-game
defense placebo is `+0.004 [-0.011, +0.020]`.

## Primary design

- Seasons: 2018-19 through 2025-26.
- Regular-season window: December 1 through the end of each regular season.
- Frozen player O-EPM and role: latest game-agnostic provider snapshot
  requested on or before December 1.
- Opponent defense: sum of the actual five defenders' D-EPM from the latest
  snapshot requested on or before the game. The provider contract says a
  date-D prediction excludes games on D.
- Coverage: all five opposing defenders are required in the primary sample.
- Unit: player-game aggregated from standard possessions.
- Outcome: team points per 100 offensive possessions while the player is on
  court; no personal-shooter credit assignment.
- Adjustment: player-season fixed effects, rolling O-EPM of the other four
  offensive players, home court, rest, offensive score margin, and
  garbage-time share.
- Inference: WLS with possession weights and game-clustered uncertainty.

The primary table contains 149,201 player-games, 3,909 player-seasons, 1,101
players, and 6,978,486 player-possessions. The defensive-personnel sample has
1,436,428 regular-season possessions and 128,083 playoff possessions.

## Basketball interpretation

Among approximately average O-EPM players, the role result cuts against a
simple creator-collapse story. Creators and secondary creators have flatter
adjusted response slopes than low-usage shooters, big finishers, and
connectors. These are transparent E-Skills rules, not proprietary playtype
labels, so treat the exact archetype boundaries as descriptive.

The high-O-EPM interaction is concentrated in first-chance shot value:

- three-point shotmaking interaction: `-0.034 [-0.046, -0.022]`;
- rim-finishing interaction: `-0.020 [-0.027, -0.013]`;
- turnover-rate interaction: approximately zero;
- transition frequency, rim/three frequency, FT value, and second chance:
  intervals cross zero.

This says stronger defenses compress stars' on-court shot conversion value a
little more. It does not show that they lose access to shots or turn the ball
over disproportionately.

## Playoffs and repeated exposure

The regular-season model predicts playoff player-games using the actual
defensive lineup quality faced. The possession-weighted playoff residual is
`-2.46` points per 100 with a game-clustered interval of `[-3.12, -1.81]`.
All displayed O-EPM and role slices are negative. This is evidence of an
additional postseason environment not captured by personnel D-EPM alone, but
it is not causal: playoff teams, players, minutes, health, and matchups are
selected.

Within player-series, the residual changes `-0.35` points per 100 per
additional game, interval `[-0.74, +0.04]`. The O-EPM × series-game term is
`+0.063 [-0.000, +0.126]`. Both are exploratory and narrowly fail the stated
uncertainty standard.

## Identification and normalization

The differential interaction is identified; an absolute division of the
common decline between “the field got better” and each of five offensive
players is not. Tier portability slopes are therefore centered on the
possession-weighted league-player average. Do not interpret `-0.17` as the
entire decline in team offense for an average player. It is the player's
relative portability adjustment beyond the common defensive-environment
effect.

EPM is an estimated external input and carries measurement error. Player-level
slopes are noisy even after empirical-Bayes shrinkage and remain diagnostics,
not a released metric. The analysis establishes adjusted association, not a
causal effect of athleticism, scouting, or scheme.

## Reproduction and artifacts

```bash
.venv/bin/python nba_pipeline/scripts/analyze_offensive_impact_portability.py
.venv/bin/python -m pytest nba_pipeline/tests/test_offensive_impact_portability.py -q
```

Generated, gitignored artifacts live under
`nba_pipeline/results/offensive_impact_portability/`. Important files:

- `REPORT.md`: complete empirical readout;
- `player_game_regular_season.parquet` and `player_game_playoffs.parquet`;
- `player_season_inspection.csv` and `player_season_defense_buckets.csv`;
- `model_a_tier_slopes.csv`, `model_b_continuous_interactions.csv`, and
  `model_c_player_specific_slopes.csv`;
- `mechanism_continuous_interactions.csv` and
  `mechanism_response_slopes.csv`;
- `playoff_player_predictions.csv`, `playoff_residual_summaries.csv`, and
  `playoff_series_exposure_test.csv`;
- `robustness_weighting_and_garbage.csv`,
  `season_by_season_interactions.csv`, and `falsification_tests.csv`;
- five PNG figures and a provenance/audit `manifest.json`.

The FIRST_CHANCE/SECOND_CHANCE mechanism targets use expected FT value, while
the primary RAPM outcome uses actual points. Rare expected-versus-actual FT
conversion gaps are explicit in the manifest; do not claim that the mechanism
target closes exactly to actual RAPM points on those rows.

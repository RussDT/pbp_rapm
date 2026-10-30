# Team Game-Log Net DECOMP

`scripts/build_team_game_net_decomp.py` rolls the possession-level individual
DECOMP inputs up to a descriptive team-vs-team game log. It answers a game
accounting question—where did the scoring margin come from?—rather than fitting
a one-game RAPM or assigning isolated causal credit to players.

## Inputs

For a season and season type, the builder reads:

- raw `NBA*.parquet` for home/away team identity, lineup sides, and final score;
- isolated `processed_netcont_lane/FIRST_CHANCE*.parquet` for the first-chance
  net-continuation tree and terminal possession keys;
- isolated `processed_netcont_lane/SECOND_CHANCE_CLEAN*.parquet` for gross and
  net-continuation second-chance value;
- canonical `results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet` for
  per-FGA `RUSSELL_SHOTQUALITY v2` values.

The first- and second-chance rows must match one-to-one on game, period, clock,
and terminal event number. Offense is assigned from the raw home/away lineup
slots, not guessed from the game ID or score.

## Two additive views

### Net continuation

The first view uses:

1. rim frequency;
2. rim finishing;
3. midrange scoring;
4. three-point frequency;
5. three-point finishing;
6. free-throw value;
7. turnover value;
8. net second-chance value.

It uses the isolated `kappa(zone)` continuation pricing. A missed shot receives
its expected recoverable continuation in the shot branch, and that same amount
is removed from the second-chance branch. This changes attribution without
changing the summed team total.

Turnover rows follow the same alias contract as `rapm.py`: each non-turnover
component receives its exact loaded-sample mean, while turnover value removes
the full mean non-turnover counterfactual. This exposes the cost of failing to
reach a shot while preserving row closure.

### ShotQuality

The second view uses:

1. `RUSSELL_SHOTQUALITY v2` expected shot value;
2. shotmaking versus Russell v2 expectation;
3. free-throw value;
4. turnover value;
5. gross second-chance value.

The builder assigns the chronological first FGA in every standard possession,
joins `shot_quality_season_ev`, and preserves the public first-chance FT
replacement baseline. The exact row identity is:

```text
Russell SQ v2 + actual-minus-v2 shotmaking + FT value = first-chance scoring
```

The v2 estimate is shooter-specific: shot context plus empirical-Bayes
player-season talent. It deliberately excludes the provider `initial_ev`
column and the repo's defender-lineup adjustment. This is retrospective game
accounting, not a shooter-neutral look-quality claim or an as-of pregame
forecast.

## Scoreboard bridge

The processed scoring surfaces use expected FT makes, consistent with the
individual RAPM pipeline. Therefore their game totals can differ slightly from
the literal final score. The builder adds one transparent game-team component:

```text
FT conversion vs expectation = actual final score - modeled team points
```

With that bridge, both views satisfy:

```text
sum(team components - opponent components) = actual final score margin
```

The builder raises on row, game, or series closure gaps above `1e-9` before
rounding exports.

## Usage

```bash
python nba_pipeline/scripts/build_team_game_net_decomp.py \
  --year 2026 \
  --season-type ps \
  --team NYK \
  --opponent CLE \
  --games 1-4
```

Generated, gitignored outputs go to
`nba_pipeline/results/team_game_net_decomp/`:

- `*_games.csv`: one row per selected game, final scores, modeled totals,
  closure, and wide net-component margins;
- `*_components.csv`: long game × view × component table with both teams'
  point totals, target-team net points, and per-100 rates;
- `*_series.csv`: component totals over the selected games.

Positive `net_points` always means the perspective team benefited relative to
the opponent. Negative values mean the opponent won that component.

## Interpretation boundary

This is retrospective team offense minus opponent offense on the realized
possessions and lineups. It is ideal for explaining a game or series. It is not
lineup-adjusted player impact, a pregame forecast, proof of repeatable team
skill, or causal credit for a specific player or coaching decision.

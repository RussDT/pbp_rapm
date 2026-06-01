# Career Teammate Summary

`scripts/build_career_teammate_summary.py` builds each player's most-played-with
career teammates across regular season plus playoffs.

## Command

```bash
python nba_pipeline/scripts/build_career_teammate_summary.py \
  --start-year 1997 \
  --end-year 2026 \
  --season-types all \
  --top-n 15
```

Default outputs:

- `nba_pipeline/results/career_teammates/career_teammate_pairs_97_26_all.csv`
- `nba_pipeline/results/career_teammates/career_teammate_top15_97_26_all.csv`

## Source Surfaces

The script uses two repo-native surfaces because no single current artifact has
both exact elapsed time and RAPM possession scoring:

- `raw_data/NBA*.parquet` and `raw_data/NBA*_PS.parquet` for same-team elapsed
  seconds from home/away lineups and event-clock deltas.
- `processed/RAPM*.parquet` and `processed/RAPM*_PS.parquet` for same-team
  offensive/defensive possessions and points while the pair was on court.

The processed RAPM scoring value is `Net_Diff`, matching the pipeline's RAPM
surface. That means free throws use the RAPM expected-FT treatment rather than
pure scoreboard points.

## Output Definitions

The top-N file is oriented by anchor player: a pair appears once for each player
when it lands inside that player's top-N teammate list. Top-N ranking is by
`shared_poss`, with `shared_minutes` retained as a clock-time reference.

- `shared_minutes`: minutes the two players were on the floor as teammates.
- `player_total_minutes`: total raw-PBP minutes for the anchor player.
- `pct_player_minutes_with_teammate`: `shared_minutes / player_total_minutes`.
- `teammate_total_minutes`: total raw-PBP minutes for the teammate.
- `pct_teammate_minutes_with_player`: `shared_minutes / teammate_total_minutes`.
- `off_poss`: possessions where the pair shared the offensive lineup.
- `def_poss`: possessions where the pair shared the defensive lineup.
- `shared_poss`: `off_poss + def_poss`.
- `player_total_poss`: anchor player's total offensive plus defensive
  possessions over the selected window.
- `pct_player_poss_with_teammate`: `shared_poss / player_total_poss`.
- `teammate_total_poss`: teammate's total offensive plus defensive possessions.
- `pct_teammate_poss_with_player`: `shared_poss / teammate_total_poss`.
- `points_for`: `Net_Diff` summed on shared offensive possessions.
- `points_against`: `Net_Diff` summed on shared defensive possessions.
- `ortg`: `100 * points_for / off_poss`.
- `drtg`: `100 * points_against / def_poss`.
- `net_rating`: `ortg - drtg`.
- `player_ortg`, `player_drtg`, `player_net_rating`: the anchor player's
  total on-court rating over the selected window.
- `player_without_teammate_ortg`, `player_without_teammate_drtg`,
  `player_without_teammate_net_rating`: the anchor player's on-court rating
  after subtracting this pair's shared offensive/defensive possessions and
  points.
- `teammate_ortg`, `teammate_drtg`, `teammate_net_rating`: the teammate's total
  on-court rating over the selected window.
- `teammate_without_player_ortg`, `teammate_without_player_drtg`,
  `teammate_without_player_net_rating`: the teammate's on-court rating after
  subtracting this pair's shared offensive/defensive possessions and points.

Generated CSVs live under `results/` and should be treated as artifacts unless
there is an explicit reason to commit a curated copy.

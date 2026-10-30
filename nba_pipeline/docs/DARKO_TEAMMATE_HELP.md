# DARKO Teammate Help

`scripts/build_darko_teammate_help.py` builds the data behind **The Help
Index**, a 1997-2026 regular-season and playoff teammate-context explorer.

## Question

For a player on a particular possession, how much DARKO value did the other
four players on his team provide?

The possession-level quantity is:

```text
teammate_help(player, possession, metric)
  = sum(same_season_DARKO(teammate, season, metric) for the other four teammates)
```

The supported metrics are `dpm`, `o_dpm`, and `d_dpm`. A player-season value is
the possession-weighted mean of the possession-level four-teammate sum.

Career rankings use the same exposure contract. For each metric and schedule
universe, the app computes:

```text
career_help(player)
  = sum(player_season_help * player_season_rated_exposures)
    / sum(player_season_rated_exposures)
```

The available career universes are regular season, playoffs, and combined.
Combined pools the underlying RS and playoff rated exposures; it is not an
unweighted average of the two schedule-level results. Career coverage is total
rated exposure divided by total exposure across the selected universe.

This is lineup context, not player credit. A high value means the player shared
the floor with strong teammates; it does not say that he caused their ratings
or that his own impact was lower.

## Source surfaces

- Lineups and exposure weights: `processed/RAPM{YY}.parquet` and
  `processed/RAPM{YY}_PS.parquet`.
- Required lineup fields: `O1..O5`, `D1..D5`.
- Required date field: `game_date`.
- DARKO history: the public `dpm_history.csv` Google Sheet export, currently
  `15,054` rows spanning 1997-2026.

The builder refuses a processed file whose `game_date` coverage is below
`99.9%`. It does not silently substitute a season-end date.

## Same-season retrospective rule

The public DARKO history is not a daily archive. It has one dated row per
player-season. The canonical Help Index applies that same-season row to every
regular-season and playoff exposure in the matching season, including games
before the row's source date.

- A player's 1997 DARKO value applies to his full 1997 season, so 1997 is no
  longer an artificial low-coverage boundary.
- A prior-season value is not carried into the next season.
- A player without a DARKO row for the matching season remains missing; missing
  ratings are never zero-filled.
- The same-season value is used for both the regular season and playoffs.
- This is deliberate hindsight. It describes season-level teammate talent and
  must not be presented as a pregame or date-correct forecast.

The legacy strict game-date contract remains available through
`--rating-timing asof` for audit reproduction. In that mode only, the builder
carries the latest prior row forward, never uses a future row, and populates
`avg_rating_lag_days` and `prior_snapshot_pct`. Those diagnostics are blank or
zero in the canonical same-season build.

## Exposure and pair contracts

Each processed possession contributes one team exposure for the offensive
lineup and one for the defensive lineup. Within a valid five-player team
exposure:

- `total_exposures` counts the focal player's appearances.
- `rated_exposures` counts appearances where all four teammates have complete
  `dpm`, `o_dpm`, and `d_dpm` values. The focal player's own DARKO may be
  missing because it is not part of his teammate sum.
- `coverage_pct = rated_exposures / total_exposures`.
- `shared_exposures` counts all focal-player/teammate co-appearances.
- `avg_teammate_dpm` is the teammate's possession-weighted DARKO on the focal
  player's fully rated exposures.
- `contribution_dpm` uses the focal player's total rated-exposure denominator:

```text
contribution_dpm(player, teammate)
  = sum(teammate_dpm * shared_rated_exposure)
    / player_rated_exposures
```

The oriented pair contributions close exactly:

```text
sum(contribution_dpm over teammates) = help_dpm
sum(contribution_o_dpm over teammates) = help_o_dpm
sum(contribution_d_dpm over teammates) = help_d_dpm
```

Career shared-floor rows preserve the same identity over the selected regular
season, playoff, or combined schedule. They sum season-level
`shared_exposures`, `rated_exposures`, and weighted DARKO numerators before
division:

```text
career_avg_teammate_dpm
  = sum(weighted_teammate_dpm) / sum(pair_rated_exposures)

career_contribution_dpm
  = sum(weighted_teammate_dpm) / sum(player_rated_exposures)

career_share_of_player_exposures_pct
  = 100 * sum(shared_exposures) / sum(player_total_exposures)
```

Therefore career teammate contributions also sum to the matching career help
value. Combined always pools the RS and playoff numerators and denominators.

The app labels these two quantities separately and puts the player's own
impact first in Lineup Company:

- **DPM impact** is the teammate's own same-season rating during
  fully rated shared exposures. This is the large value on the teammate row.
- **Help contribution** is that teammate's weighted contribution to the focal player's
  four-teammate sum. It reflects both teammate quality and how often the pair
  shared the floor, so it is normally smaller in magnitude than DPM impact and
  appears in the supporting row detail.

Each teammate row also exposes its own DPM coverage. Coverage can still be
incomplete when a lineup player has no DARKO row for that season. Those rows
show `DPM unavailable`; the builder never substitutes a different season or
zero.

The rebuilt 1997-2026 RS+PS surface has `20,660` player-season rows and
`300,944` oriented player/teammate-season rows. Overall fully rated anchor
exposure coverage is `99.98%`; 1997 coverage is `99.37%` in the regular season
and `99.98%` in the playoffs. Maximum pair-to-player reconciliation gaps are
`3.55e-15` for DPM and O-DPM and `1.78e-15` for D-DPM.

## Build

```bash
.venv/bin/python nba_pipeline/scripts/build_darko_teammate_help.py \
  --start-year 1997 \
  --end-year 2026 \
  --season-types all \
  --rating-timing same-season \
  --darko nba_pipeline/.cache/teammate_help/dpm_history.csv \
  --app-data-dir teammate-help-app/public/data
```

Use `--rating-timing asof` only to reproduce the older strict dated audit.

The canonical generated analysis artifacts are:

- `results/darko_teammate_help/season_player_help.csv`
- `results/darko_teammate_help/season_teammate_help.csv`
- `results/darko_teammate_help/metadata.json`

The app directory receives a compressed player-season index, one compressed
teammate file per season and schedule type, and 64 small player-ID shards for
each career schedule. The sharding avoids downloading an all-player career
teammate table when one player is selected. Deployment data is committed in the
standalone app source; the larger analysis CSVs remain generated artifacts.

## Estimated Championship Help

> Superseded: the public championship surface now uses monthly pre-postseason
> EPM and actual normalized playoff rotations. See
> [EPM Championship Help](./EPM_CHAMPIONSHIP_HELP.md). The material below
> documents the retired DARKO prototype only.

`scripts/build_championship_help.py` adds a separate retrospective model for
the question: how much did a player's season-specific teammate context change
his modeled probability of winning the championship?

The estimation universe is 1998-2026. Inputs use every regular-season team
lineup, including minutes when a selected player is off the floor. Playoff PBP
supplies only the champion label. The retrospective surface uses the exact
same-season DARKO snapshot; it does not carry the prior year's snapshot through
the current season. This is intentionally hindsight-aware description, not an
as-of forecast.

For each team-season and metric, every player's contribution is:

```text
player_team_contribution
  = same_season_DARKO * player_team_exposures / team_total_exposures

team_strength = sum(player_team_contribution over the full roster)
```

Therefore starters, reserves, and bench-only stretches all enter the team's
title environment. The player contributions close to team DPM, O-DPM, and
D-DPM at machine precision. Same-season DARKO covers `99.9996%` of player
exposure slots; the tiny uncovered remainder is neutral `0.0` and remains
reported in team coverage rather than removing a team from the championship
choice set.

The probability unit is one team-season, not one player-season. Each season has
exactly one champion and includes all `29` or `30` teams. A nonnegative
regularized conditional-logit score is converted through a within-season
softmax:

```text
P(team wins title | season)
  = exp(team_score) / sum(exp(score) over every team that season)
```

Consequently each season's actual team probabilities sum exactly to `1.0`.
Overall-DPM and separate offense/defense specifications are compared across six
regularization strengths by leaving out one full season at a time. Selection
uses the one-standard-error rule on champion log loss: among candidates within
one standard error of the best score, prefer stronger regularization, then the
simpler feature set at the same `C`. This prevents a handful of historical title
winners from producing implausibly sharp probability tails. The 2026-08 build
selected overall DPM at `C=0.2`:

- `863` team-seasons, `29` seasons, and exactly `29` champion labels;
- leave-one-season-out champion log loss `1.814260`;
- season-level multiclass Brier score `0.752243`;
- top-one accuracy `37.93%`;
- maximum annual probability-mass gap `6.66e-16`.

The displayed actual title probability is the held-out model's team
probability and is identical for every qualifying player assigned to that team.
For traded players, the focal player is assigned to the team for which he
appeared latest in the regular season. A focal player needs at least `500`
exposures on that final team and an exact same-season DARKO row.

For player `p` in season `s`, regular-season exposure splits the actual team
strength into three exhaustive contexts:

```text
self_strength = focal_DPM * focal_team_exposures / team_total_exposures

on_court_help
  = sum(teammate_DPM * shared_with_focal_exposures / team_total_exposures)

off_floor_help = team_strength - self_strength - on_court_help
```

`off_floor_help` is every teammate contribution during team minutes when the
focal player sat; it is not a nominal starter/reserve label. The probability
split holds self fixed and applies the exact two-order Shapley decomposition to
the on-court and off-floor groups:

```text
self_context = P(self_strength)

on_court_added = 1/2 * [P(self + on) - P(self)
                         + P(full) - P(self + off_floor)]

off_floor_added = 1/2 * [P(self + off_floor) - P(self)
                          + P(full) - P(self + on)]

self_context + on_court_added + off_floor_added = P(actual team)
```

Opponent scores stay fixed and every counterfactual is renormalized against the
full season choice set. Therefore every qualifying player on the same team gets
identical actual team title odds, but a different self/on-court/off-floor split.
Zero DPM is league average, not replacement level. Career values sum the three
season buckets; they are not a redistribution of observed rings and not causal
credit.

The combined teammate total is also allocated over individual teammates using
`128` deterministic Shapley permutations. The order-averaged marginal
probabilities close exactly to the actual-versus-neutral season total; the
maximum build gap is `7.22e-16`, and the three-bucket probability identity
closes within `4.44e-16`. A teammate's individual probability attribution uses
all of his team exposures, while `shared_exposures` separately reports actual
minutes alongside the focal player. Career teammate rows are summed and ranked.
The app publishes the top 20 in career view and switches to that season's
teammates when a season is selected.

All 29 or 30 regular-season teams remain in the annual probability distribution,
including teams that miss the playoffs. Playoff PBP is used only to identify the
single champion. The model never divides credit among playoff qualifiers and
never uses playoff games, series length, or minutes, because those quantities
are downstream of winning and would reward teams for surviving longer.

Canonical analysis artifacts live under `results/championship_help/`:

- `team_seasons.csv`, `team_player_contributions.csv`, and
  `team_player_pair_exposures.csv`;
- `player_team_seasons.csv` and `player_teammate_team_seasons.csv`;
- `champions.csv`;
- `model_cv.csv`, `model_oof_predictions.csv`, and
  `team_season_estimates.csv`;
- `player_season_estimates.csv` and
  `player_teammate_season_estimates.csv`;
- `career_player_estimates.csv` and `career_teammate_estimates.csv`;
- `metadata.json`.

Build or refresh the full model and app shards with:

```bash
.venv/bin/python nba_pipeline/scripts/build_championship_help.py
```

Use `--reuse-data` to refit and republish from the existing team/roster/pair CSVs
without rereading every raw and processed season. The app receives a compressed
career index plus 64 player-ID shards under
`teammate-help-app/public/data/championship-help/`.

Limitations: same-season DARKO contains hindsight; the model does not identify
injuries, bracket strength, opponent matchups, coaching, or causal teammate
effects; and selection/tuning occurred on this historical window.
Leave-one-season-out scoring protects whole-season folds but does not turn this
retrospective product into a future-title forecasting claim.

## App

`teammate-help-app/` is a standalone vinext/Sites project. The 2026-07 redesign
replaced the original single-file dashboard with a light, editorial component
architecture:

- `app/help-index-app.tsx` owns one normalized view-state model
  (`player, metric, career-vs-season, career schedule, season, season type`);
  every surface (hero, career arc, Lineup Company, rankings) derives from it,
  and the address bar mirrors it continuously so any URL is shareable.
- `app/career-rankings.ts` keeps the exposure-weighted career aggregation and
  standing math (unchanged contract, exercised directly by the tests).
- `app/lib/` holds the typed data layer (`data.ts` caches the gzip player index
  and season/career pair shards; the fetch helper detects the gzip magic bytes
  so transparently-decoded dev responses still parse) and formatting/metric
  configuration.
- `app/components/` holds `player-search`, `career-arc`, `lineup-company`,
  `rankings`, and the season-by-season `championship-help` panel.

Product behavior:

- opens on the selected player's combined RS+playoff career; choosing Season, a
  year, an RS/PS toggle, or a career-arc bar switches every season-dependent
  surface together, and a prominent "Back to career view" control restores the
  active career schedule everywhere;
- the career arc draws zero-baseline bars per season (combined mode pools RS+PS
  numerators over rated exposures, matching the career contract) with the
  selected season highlighted in gold and unavailable seasons drawn as empty
  stubs;
- Lineup Company is a shareable card headed by the focal player and an explicit
  period label; rows order by frequency rank and show share together, the
  teammate's same-season DPM impact as the primary number, shared possessions,
  help contribution, and pair-level rated coverage; rows open that teammate's
  profile; the card opens with 10 teammates and expands by 10; Share preserves
  player, metric, view, schedule, season, and type in the URL;
- rankings follow the active context (single season or career schedule), flip
  between most and least help, expose coverage and possession filters, open
  with 25 players, and expand by 25 through the full qualified field;
- Estimated Championship Help shows actual-era titles, expected titles under
  actual and neutral teammate context, the summed probability difference, a
  selectable season spike history, and the top 20 Shapley-attributed teammates.
  It uses same-season team-level regular-season inputs, includes bench/off-floor
  team minutes, excludes the focal player's own team-strength contribution from
  help, and displays held-out team odds that sum to 100% across every season;
- career standing (help, rank, percentile with a marker track, coverage,
  seasons) uses the fixed qualification thresholds: 50% rated coverage plus
  `10,000` exposures for RS/combined or `1,000` for playoffs; percentile is
  `(field_size - rank + 1) / field_size`;
- missing same-season DARKO stays visibly missing (em-dash impact, "no same-season DARKO on
  shared floor", 0% rated) — never zero-filled;
- interaction states never change geometry (hover/selection recolor only), data
  loads into fixed-height skeleton rows, and the mobile layout condenses lineup
  rows to two-line entries that keep the impact number primary.

Run `npm test` inside the app to build it, verify the server-rendered shell and
design-system invariants (no orange, no thick top borders, geometry-stable
selection), confirm the full time/schedule span, recheck browser-artifact pair
reconciliation, prove combined career exposure and help reconcile to the RS and
playoff components, and confirm career pair contributions close to career help
for every schedule on a second reference career. The tests also confirm that
championship-help career totals equal the sum of season estimates and that the
published model metadata and Shapley reconciliation remain valid.

# DARKO Teammate Help

`scripts/build_darko_teammate_help.py` builds the data behind **The Help
Index**, a 1997-2026 regular-season and playoff teammate-context explorer.

## Question

For a player on a particular possession, how much DARKO value did the other
four players on his team provide?

The possession-level quantity is:

```text
teammate_help(player, possession, metric)
  = sum(as_of_DARKO(teammate, game_date, metric) for the other four teammates)
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

## Historical as-of rule

The public DARKO history is not a daily archive. It has one dated snapshot per
player-season. For every game, the builder selects the player's latest DARKO
row whose snapshot date is less than or equal to `game_date`.

- A prior-season snapshot may carry forward when it is the latest information
  available.
- A future snapshot is never pulled backward.
- A rookie or previously uncovered player has no rating until his first
  snapshot date.
- `prior_snapshot_pct` compares the chosen DARKO season label to the processed
  file's season-ending year, not the game date's calendar year. This distinction
  matters for October-December games.
- `avg_rating_lag_days` exposes the possession-weighted age of the four
  teammate snapshots used.

The strict rule protects against look-ahead bias but limits coverage. The
1997 regular season has almost no earlier DARKO snapshot history and should be
treated as a boundary season rather than a normal fully covered result.

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

- **DPM impact** is the teammate's own possession-weighted as-of rating during
  fully rated shared exposures. This is the large value on the teammate row.
- **Help contribution** is that teammate's weighted contribution to the focal player's
  four-teammate sum. It reflects both teammate quality and how often the pair
  shared the floor, so it is normally smaller in magnitude than DPM impact and
  appears in the supporting row detail.

Each teammate row also exposes its own DPM coverage. Historical as-of coverage
is genuinely incomplete: `18.62%` of RS, `10.35%` of playoff, and `17.89%` of
combined career pair rows have no fully rated shared exposure. Those rows show
`DPM unavailable` instead of substituting a future or season-end rating.

The full 1997-2026 RS+PS build has `20,660` player-season rows and `300,944`
oriented player/teammate-season rows. Overall fully rated anchor exposure
coverage is `69.40%`. Maximum reconciliation gaps are:

- DPM: `3.55e-15`
- O-DPM: `3.55e-15`
- D-DPM: `1.78e-15`

## Build

```bash
.venv/bin/python nba_pipeline/scripts/build_darko_teammate_help.py \
  --start-year 1997 \
  --end-year 2026 \
  --season-types all \
  --darko nba_pipeline/.cache/teammate_help/dpm_history.csv \
  --app-data-dir teammate-help-app/public/data
```

The canonical generated analysis artifacts are:

- `results/darko_teammate_help/season_player_help.csv`
- `results/darko_teammate_help/season_teammate_help.csv`
- `results/darko_teammate_help/metadata.json`

The app directory receives a compressed player-season index, one compressed
teammate file per season and schedule type, and 64 small player-ID shards for
each career schedule. The sharding avoids downloading an all-player career
teammate table when one player is selected. Deployment data is committed in the
standalone app source; the larger analysis CSVs remain generated artifacts.

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
- `app/components/` holds `player-search`, `career-arc`, `lineup-company`, and
  `rankings`.

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
  teammate's actual as-of DPM impact as the primary number, shared possessions,
  help contribution, and pair-level rated coverage; rows open that teammate's
  profile; the card opens with 10 teammates and expands by 10; Share preserves
  player, metric, view, schedule, season, and type in the URL;
- rankings follow the active context (single season or career schedule), flip
  between most and least help, expose coverage and possession filters, open
  with 25 players, and expand by 25 through the full qualified field;
- career standing (help, rank, percentile with a marker track, coverage,
  seasons) uses the fixed qualification thresholds: 50% rated coverage plus
  `10,000` exposures for RS/combined or `1,000` for playoffs; percentile is
  `(field_size - rank + 1) / field_size`;
- missing as-of DARKO stays visibly missing (em-dash impact, "no as-of DARKO on
  shared floor", 0% rated) — never zero-filled;
- interaction states never change geometry (hover/selection recolor only), data
  loads into fixed-height skeleton rows, and the mobile layout condenses lineup
  rows to two-line entries that keep the impact number primary.

Run `npm test` inside the app to build it, verify the server-rendered shell and
design-system invariants (no orange, no thick top borders, geometry-stable
selection), confirm the full time/schedule span, recheck browser-artifact pair
reconciliation, prove combined career exposure and help reconcile to the RS and
playoff components, and confirm career pair contributions close to career help
for every schedule on a second reference career.

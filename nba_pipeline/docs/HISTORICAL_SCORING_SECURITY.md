# Historical Own-Scoring and Ball-Security Value

**Status:** research-only annual NBA player board, 1997-2026. This model joins
the existing production-equivalent zTS scoring-value bridge to point-valued
scoring-turnover and bad-pass-turnover lanes. It is **own scoring plus ball
security**, not total offense, passing value, shot quality, ORAPM, VPM, or a
replacement for production rZTOV.

## What the model answers

The board asks a deliberately narrow question:

> How much scoring value did a player generate through their own shot volume,
> role-adjusted efficiency, and mix-adjusted ball security?

It preserves three exact identities:

```text
zTS = rTS + role difficulty
turnover value = scoring-turnover value + bad-pass-turnover value
scoring impact without bad-pass = scoring value + scoring-turnover value
scoring-security value = scoring value + turnover value
```

The completed 1997-2026 build has 11,605 player-seasons, 11,583 complete
scoring-security values, and 10,062 rows above the 1,000-offensive-possession
display floor. The maximum in-memory identity gap is below `2e-15`; the written
CSV (6 dp) closes every identity within `1e-6`.

## Scoring-value contract

The reader-facing production zTS chain is unchanged:

```text
zTS = rTS + playtype difficulty
zTS add -4 = 2 * TSA100 * (zTS + 4) / 100
scoring value / 100 = -1.211295 + 0.800618 * zTS add -4
```

The new work extends the role input backward; it does not retune the public
scoring-value coefficients or relabel zTS as overall offense.

Scoring context comes from two explicitly separated sources:

| Seasons | Source | Status |
| --- | --- | --- |
| 1997-2000 | official raw PBP FGA, FTA, points, and resolved offensive possessions | reconstructed box approximation |
| 2001-2026 | core `merged_rs_full.csv.gz` `TSA100`, `TS_pct`, and `rTSPct` | existing exact pipeline context |

When a modern core row is missing `playtype_diff`, the refreshed Databallr
`player_zts` row supplies `TSA100`, `rTS`, and observed role together. This
prevents a current observed role from being mixed with a stale local scoring
snapshot, as happened for 2026 before this guard was added.

The pre-2001 reconstruction uses `TSA = FGA + 0.44 * FTA`. On the first exact
overlap season, 2001, it has `0.53 pp` possession-weighted rTS RMSE, `0.42 pp`
MAE, and `0.38` TSA/100 RMSE among 312 players with at least 1,000 offensive
possessions. That supports a useful backcast but requires the visible
`raw_pbp_box_approximation` source label.

Raw 1997-2000 PBP stores shooter surnames rather than reliable display names.
The board therefore resolves names by exact `nba_id + year` against
`external/bref_ft_pct_1997_2000.csv`; no fuzzy surname matching is allowed.
This supplies full names for 1,392 of 1,393 early rows. The lone unresolved row
is unnamed `nba_id=1` in 1997 with zero TSA.

## Historical role seam

The role source is selected row by row:

| Seasons | Preferred role source | Final-board treatment |
| --- | --- | --- |
| 1997-2005 | calibrated six-family PBP shot mix | retrospective estimate |
| 2006-2026 | refreshed Databallr Synergy-backed `player_zts` snapshot | observed |
| Any observed-era row without Synergy coverage | no substitute | explicit missing value, excluded from the reader |

The refreshed Databallr snapshot contains 10,109 regular-season
player-seasons and 9,918 non-null observed role values. The 2006-2013
calibration slice remains 3,653 rows with 3,631 non-null roles. The build never
replaces a 2006-2026 value with a model prediction; an entirely missing season
raises an error so a stale source cannot silently activate the backcast. The
exported `estimated_role` field is blank from 2006 onward; modern PBP estimates
remain only in the separate calibration cache and overlap audit.

The PBP estimator reuses the outcome-invariant six-family taxonomy from
`analyze_pbp_playtype_difficulty_ots.py`:

- cut two;
- self-created three;
- other three;
- self-created two;
- other rim two; and
- other two.

For 1998 onward, the current shot mix is priced with the immediately previous
regular season. The repo has no `NBA96.parquet`, so 1997 alone uses an explicit
same-season retrospective category price.

The calibrated mapper is a sign-constrained compositional ridge on five shot
shares, with `OTHER_2` as the omitted reference. It deliberately excludes raw
PBP difficulty because that quantity is already a weighted combination of the
same shares. More self-created threes, other threes, or self-created twos can
never lower estimated role difficulty; more cuts or other-rim twos can never
raise it. The selected alpha is `0.01`.

### Role calibration receipt

| Test | Rows | RMSE | R2 | Correlation |
| --- | ---: | ---: | ---: | ---: |
| 2006-2013 leave-one-season-out selection | 3,127 | 0.94 pp | 0.698 | 0.865 |
| 2006-2013 all-anchor refit audit | 3,127 | 0.93 pp | 0.700 | 0.866 |
| 2014-2026 later observed-era audit | 5,668 | 0.89 pp | 0.798 | 0.882 |

The previous redundant share-plus-raw model had `0.931` pp LOSO RMSE, only
`0.004` better, but degraded to `1.486` pp in the later-era stress test and
could assign a negative conditional coefficient to self-created threes. The
constrained mapper removes that pathology and transports much better without
using the later era for selection. It remains restricted to 1997-2005;
observed-era gaps stay missing rather than falling back. The uncalibrated raw
PBP price has only `0.334` correlation with historical Synergy role, so the
share calibration is doing real work rather than merely rescaling one number.

## Turning rZTOV lanes into points

Each row takes its turnover lanes from one of two ledgers, recorded in
`turnover_source`:

| `turnover_source` | Seasons | Ledger |
| --- | --- | --- |
| `observed` | 2014-2026 (5,370 rows) | the observed rZTOV ledger, `results/rztov_seasons_2014_2026.csv`, row-for-row the numbers the [rZTOV explorer](./RZTOV.md) shows |
| `box_estimated` | 1997-2013, plus 49 modern rows without observed creation labels | the annual [box-estimated rZTOV backcast](./NBA_RZTOV_BOX_BACKCAST.md) |

Both ledgers report season-centered scoring-turnover excess and bad-pass
turnover excess. Prices are fitted separately on each ledger, so an observed
row is never priced with a backcast slope. For every complete rolling
five-season window of a ledger, the builder:

1. possession-weights each player's annual lane excess into the window;
2. changes sign so positive means turnovers saved;
3. joins the matching near-unpenalized `alt_*_tov_net_value` offense RAPM
   (`results/tov_price_targets/*_all_rb_se_a20_40`, built by
   `scripts/build_tov_price_targets.py`); and
4. fits a possession-weighted through-origin point price separately by lane.

A season takes the price of the window that ends in it. The 1997-2000
backcast rows inherit the earliest 1997-2001 historical fit, and the observed
2014-2017 rows inherit the first complete observed window, 2014-2018, never a
price from the other ledger.

| Ledger | Window | Scoring-TOV point price | Bad-pass-TOV point price |
| --- | --- | ---: | ---: |
| box-estimated | 1997-2001 | 1.158 | 1.380 |
| box-estimated | 2009-2013 | 1.119 | 1.345 |
| observed | 2014-2018 | 1.041 | 1.015 |
| observed | 2022-2026 | 0.902 | 1.020 |

Across the 26 box-estimated windows, scoring-TOV prices range from `0.869` to
`1.175` and bad-pass prices from `0.968` to `1.380`. Across the nine observed
windows they range `0.838`-`1.041` and `0.944`-`1.035`. Median target R2 is
`0.62` / `0.51` for the scoring lane (box / observed) and `0.28` / `0.20` for
the noisier bad-pass lane. These are empirical RAPM bridges, not claims that
every real turnover has a fixed event value.

**Why the target is a near-unpenalized solve (correction, 2026-10-19).** Before
this date the slope was fitted on the production alpha-`2000/4000` components and
read about half these values (observed 2022-2026: `0.535` / `0.767`). A ridge
coefficient used as the dependent variable carries its shrinkage straight into
the slope. The alpha `20/40` target sits on the plateau of the alpha path, and a
direct possession-level lineup IV agrees with it (`0.88`-`1.01` scoring,
`0.96`-`0.97` bad pass). See [rZTOV points conversion](./RZTOV.md#points-conversion).

**Open issue: pre-2014 bad-pass prices above one kept possession.** The
1997-2001 window prices a bad pass at `1.380` (90% interval `1.25`-`1.51`), above
its `1.22` kept-possession baseline. The backcast regressor is not the cause. On
the same 2014-2018 rows with the same target, box-estimated and observed excess
give `1.151` and `1.015`, which is within the overlapping bootstrap intervals.
The window-level slope of observed on backcast excess is `1.01`-`1.04` in all
three modern windows, so the backcast is not a compressed regressor that inflates
the slope. The remaining candidates are the pre-2014 source itself, which has
partial lineups with unknown slots excluded and no team turnovers, and the
bad-pass lane's low fit (median R2 `0.28`). These rows are research-only; treat
their bad-pass prices as unresolved, not as evidence that a bad pass costs more
than a possession.

## Playtype mix RAPM (research column)

A scoring value built on zTS credits a player for taking hard shots. A
reasonable objection is the "DeRozan effect": a high-usage mid-range creator
may also push the lineup around them toward harder shots, so part of what zTS
credits is paid for by teammates. The board answers that with one research
column, `playtype_mix_rapm`. It is joined one-to-one from the
[two-season TD550 playtype-mix board](./TD_OVERALL_PLAYTYPE_MIX_OTS.md#annual-reader-board-1997-2026)
and is present on all 11,605 rows:

```text
mix_rapm          = share * own_diet + (1 - share) * teammate_diet     (TS-point scale, + harder)
playtype_mix_rapm = -mix_rapm / 50 * league FGA per 100                (pts/100, + easier)
                  = own_mix_value100 + teammate_mix_value100
```

Owner decision (2026-09-26): the column is the full lineup value, own diet
plus teammate diet, not a split. The own/teammate halves stay on the mix board
for audit only. The solve covers every season from 1997 through 2026 using a
two-season window (the feature season plus the prior season) with a 550-day
half-life, so the prior season carries about 0.63 weight.

The column is never added to zTS, SV/100, scoring impact, or scoring
security, and it is never the default rank. It is ungated at this spec. Its
TD700 predecessor failed the creation-controlled future-oTS increment
(`−0.0040 [−0.0233,+0.0136]`), so it is descriptive, not a forward gain, and it
never feeds VPM or a prior.

Reading 2026 (385 qualified players):

| Player | Own half | Teammate half | Playtype mix RAPM |
| --- | ---: | ---: | ---: |
| DeMar DeRozan | −0.41 | −0.49 | −0.90 (hardest) |
| Jonas Valančiūnas | −0.18 | −0.57 | −0.75 |
| Anthony Edwards | −0.39 | −0.34 | −0.73 |
| Deandre Ayton | −0.07 | −0.60 | −0.67 |
| Stephen Curry | +0.15 | +0.74 | +0.89 |
| Nikola Jokić | +0.15 | +0.89 | +1.04 |
| Draymond Green | +0.18 | +1.01 | +1.20 (easiest) |

The qualified 2026 standard deviation is `0.46` pts/100. DeRozan's diet read
easier than average from 2011 to 2016, turned hard in 2017 (`−0.50`), and
peaked at `−1.49` in 2022.

Two cautions:

- **Complement effect.** The teammate half measures what is left for
  teammates, and it drives most of the spread. A center who finishes the
  lineup's easy rim chances can read hard (Valančiūnas, Ayton) almost entirely
  through teammates. Playmakers read easy because the shots they create go to
  teammates. DeRozan and Edwards are distinctive because both halves are hard.
  Show the mix board's split in any row detail that interprets a value.
- **Early eras are coarser.** 1997 has no prior season, so it is priced on
  itself and fit on one season. Before 2009, play-by-play rarely tags
  self-created threes, so older diets mostly separate rim, other twos, threes,
  and transition.

The diet is stable across disjoint two-season windows three years apart
(`scripts/audit_playtype_mix_stability.py`; players with more than 1,500
weighted FGA of exposure in both windows):

| Windows | Players | `mix_rapm` r | Movers r (n) | Teammate diet r |
| --- | ---: | ---: | ---: | ---: |
| 2003-04 vs 2006-07 | 187 | 0.62 | 0.62 (142) | 0.58 |
| 2010-11 vs 2013-14 | 207 | 0.47 | 0.45 (162) | 0.43 |
| 2017-18 vs 2020-21 | 208 | 0.48 | 0.48 (178) | 0.48 |
| 2022-23 vs 2025-26 | 236 | 0.59 | 0.53 (159) | 0.59 |

"Movers" are players whose frequent-teammate sets (at
least 300 shared lineup FGA) overlap by a Jaccard index below `0.15`. The
correlation holds for them, so the diet travels with the player rather than
with one roster.

## Lineup turnover view (research columns)

The lane prices above are regressions of near-unpenalized lane RAPM on the box
excess. They say how many points an excess turnover costs, not whose turnover
the lineup avoided.

For transparency the board joins the
[Lineup Turnover Board](./LINEUP_TOV_BOARD.md) one-to-one on
`(nba_id, year)` (`--lineup-tov`). It is a two-season, 550-day half-life
lineup ridge of on-court team turnovers with the player on offense, split
exactly per lane:

```
lineup_{lane}_tov_total_value100 = own + teammate + team      (pts/100, + good)
lineup_tov_value100              = scoring total + bad-pass total
```

Each piece is saved/100 × `lineup_tov_point_price`, the window's net
first-chance value of a kept possession (`1.21`-`1.32`).
`lineup_{lane}_tov_raw_value100` is the unadjusted on-court rate at the same
price. `lineup_team_tov_coverage` is `absent` through 2013, because Gabriel
source rows omit team turnovers there.

**Never added.** These 13 columns are never added to turnover value,
scoring impact, or security, and they carry no board identity. The build
checks only their split closure (`lineup_tov_gaps`).

**Pass-through.** The build also writes
`lineup_tov_pass_through_1997_2026.csv`: per-season, OffPoss-weighted slopes
and correlations of each lineup value on the box value. For the total, the
slope is `0.58` with correlation `0.67` in 2026; the 2015-2026 means are
`0.50` and `0.64`. The slope fell from about `0.8` on 2026-10-19, when the box
prices were corrected upward. Most of the remaining shortfall is the lineup
board's own ridge shrinkage (`3000/5000`).

**Own is not box excess.** Lineup own is measured against a flat one-fifth
share, not the box workload allowance, so creators read negative on own and
positive on teammates. Compare lane totals.

## Outputs and build

Canonical builder:

```bash
.venv/bin/python nba_pipeline/scripts/build_historical_scoring_security.py \
  --refresh-anchor-from-databallr
```

Subsequent local rebuild:

```bash
.venv/bin/python nba_pipeline/scripts/build_historical_scoring_security.py
```

Standalone scoring-first leaderboard:

```bash
.venv/bin/python nba_pipeline/scripts/build_historical_scoring_impact_html.py
```

The HTML defaults to `scoring_impact_no_badpass100` and exposes search,
season, possession-floor, and role-source filters. Its 13 sortable columns
sit under three group headers: own scoring (blue), ball security (gold), and
two violet research columns, Playtype mix RAPM and Lineup TOV. Missing values sort last in
both directions. Filters, sort, and the open row persist in the URL hash, so a
view can be linked, and an omitted parameter resets to its default. Rows open
with a click, Enter, or Space. The ledger shows the full arithmetic: the zTS
chain, both turnover lanes at their prices, the three sums, and a separate
mix card with the direction, the spec, and the 1997 or pre-2009 caveat.

A full-width **Turnovers in lineup data** card sits below them. It holds a
small table with rows for the scoring lane, the bad-pass lane, and both lanes,
and columns for:

- the box ledger;
- lineup own, teammates, and team;
- the lineup total;
- the raw on-court rate.

Notes under the table cover the flat-share caveat, the season's pass-through
slope, and the team-turnover coverage.
Below 720px, the formula strip stacks into label/value rows, the controls
stop being sticky, optional columns collapse, and the page has no horizontal
scroll. The payload is packed column-wise to keep the page near 3.4 MB. The
pass-through CSV is embedded as `PASS`. The
builder refuses to write the page if any payload row breaks the security,
SV-formula, scoring-TOV, or bad-pass-TOV identity by more than `5e-5`.

After the PBP role cache exists, use `--reuse-role-cache` for a fast rebuild.
The full 2006-2026 Databallr observed snapshot is passed with
`--synergy-anchor`; only its 2006-2013 slice trains the backcast. The refresh
flag performs a paginated `player_zts` select and never writes to Databallr or
Supabase.

Generated artifacts live under
`nba_pipeline/results/historical_scoring_security/`:

- `historical_scoring_security_1997_2026.csv` — complete research board;
- `historical_scoring_security_1997_2013.csv` — historical slice;
- `databallr_synergy_zts_2006_2026.csv` — generated read-only observed snapshot;
- `role_calibration_loso.csv` — all calibration candidates;
- `role_overlap_summary.csv` — anchor and later-era receipts;
- `premerged_scoring_bridge_validation_2001.csv` — raw-to-exact scoring audit;
- `turnover_point_price_path_1997_2026.csv` — 26 empirical lane prices;
- `lineup_tov_pass_through_1997_2026.csv` — per-season box-to-lineup slopes;
- `role_source_coverage.csv` — annual provenance counts; and
- `manifest.json` — status, identities, coverage, and limitations.

The generated reader is
[`historical-scoring-impact.html`](./historical-scoring-impact.html).

Regression coverage is in
`nba_pipeline/tests/test_historical_scoring_security.py`.

Reconciliation before any integration or publish:

```bash
cd nba_pipeline/scripts
../../.venv/bin/python audit_scoring_rztov_reconciliation.py            # uses the saved Databallr snapshot
../../.venv/bin/python audit_scoring_rztov_reconciliation.py --fetch-live  # read-only refresh first
```

It writes `results/scoring_rztov_integration/reconciliation_receipt.json` and
exits nonzero if any check misses its tolerance. The handoff contract is
[SCORING_RZTOV_DATABALLR_HANDOFF.md](./SCORING_RZTOV_DATABALLR_HANDOFF.md).

## Interpretation and stop rule

Use `scoring_value100` when the question is role-adjusted own-shot value. Use
`turnover_value100` when the question is mix-adjusted ball security. Their sum,
`scoring_security_value100`, is useful as a compact historical scoring-and-care
stat but omits passing creation, teammate shot value, offensive rebounding,
spacing, screening, and every defensive contribution.

For a scoring-first view that deliberately excludes passing mistakes, use
`scoring_impact_no_badpass100 = scoring_value100 + scoring_tov_value100`.
`badpass_tov_value100` remains a separate column, and adding it back closes
exactly to `scoring_security_value100`.

This build is a completed research extension of the existing zTS and rZTOV
surfaces, not a new production leaderboard or VPM input. Promotion would need
a separately declared future-impact test and a public naming/maintenance
decision; the current historical overlap evidence alone is not that gate.

# Historical VPM predictive feature inputs

The historical feature adapter is a separately named **fixed empirical-Bayes
input candidate**, not a replay of the current annual-OU skill estimator.
The sibling VPM harness must fit its dated prior mappings against these same
inputs before producing ratings. A feature file alone is not a VPM rating.

## Scope and contract

`nba_pipeline/scripts/historical_features.py` writes
`nba_pipeline/results/estimated_skills/vpm_historical_features_v1/`:

- `features.parquet`: one player/date row, canonical `own_*`, `def_*`,
  `ctx_mpg`, `ctx_games`, side exposure, explicit missingness and recipe.
- `provenance.parquet`: feature/date/player, last observed date, actual
  discounted measurement exposure, estimator, population mean and fixed prior
  exposure. `parameter_fit_end` is null because no fitted skill parameters are
  used; the manifest explicitly declares a fixed retrospective research design.
- `manifest.json`: source/artifact hashes, coverage by feature/date and limits.
- `lines/`: isolated pre-2014 player-game extracts and source audit receipts.
- `player_game_lines.parquet`: combined real player-game rows 2000–2026,
  with exposure, minutes, dates, phase and same-season exact-ID names. Annual
  box aggregates never enter this file or double career exposure.
- `independent_audit.json` and `source_field_audit.json`: full date/missingness
  gates, literal source reconstructions, subtype availability/nondegeneracy,
  and excluded raw measurements by game/reason/date.

The requested report calendar is Jan 1, Apr 1, Jul 1 and Oct 1, 2014–2026:
52 cuts through October 1, 2026. Additional annual October 2008–2025 anchors,
December 1, 2020 and September 1, 2026 support chronological calibration and
comparison. October 2020 is still a quarterly report date; the sibling mapper
may use December 2020 as its delayed preseason training anchor.

Each input measurement and each population mean uses observations strictly
before the estimate date. Same-day data are excluded. No 2027 observations
are read. Historical source publication timestamps remain uncertified.

## Fixed backbone estimator

For numerator `N`, opportunities `E`, and cut `t`, each known player receives:

```
w_i = 2 ** (-(t - game_date_i) / 550 days)
league = sum_i(w_i * N_i) / sum_i(w_i * E_i)
rate100 = 100 * (sum_player(w_i*N_i) + 1000*league)
                / (sum_player(w_i*E_i) + 1000)
```

The population is the available prefix only, not a future season mean.
Positive opportunity rows with observed zero counts are measurements.
Missing observations contribute neither counts nor exposure. No observed
opportunities means missing even when a population mean exists.

Shooting percentages use the same formula with 100 prior attempts and scale
1. Zone makes are the product of estimated attempts and estimated percentage,
so makes cannot exceed attempts. A player with measured zero zone attempts
has an unknown personal percentage; the product uses the dated population
percentage and retains the percentage missing flag. These are estimated rates,
so shrinkage may produce positive expected attempts after zero observed counts.

Offensive counts use offensive possessions; rebounds, steals and blocks use
defensive possessions. Total personal fouls use offensive plus defensive
possessions. Usage events are `FGA + .44*FTA + TOV`, not official USG%.
`ctx_mpg` is the unshrunk trailing 365-day regular-season minutes average over
strictly earlier positive-minute appearances. `ctx_games` reports its sample
size. It is not a future availability or minutes projection.

TD550, K1000 and percentage K100 are declared conservative fixed design
choices, not estimates selected using later outcomes or claimed historical
optima. They require forward comparison and should not be confused with the
learned annual-OU filter parameters used by the selected current model.

## Tracking

Only validated native measurements enter tracking summaries. The adapter
reuses the independently repaired shooting-foul source, observed defensive
tracking rows and `monthly_defense_v1/rim_counts.parquet`. The latter already
rejects unsafe cumulative snapshots, nonconsecutive gaps and unverified
post-trade increments. It never differences provider percentages.

Additional defensive counts retain their TD550/K1000 prefix-population
definition. DFGA uses observed validated DFGA measurements. Rim contests use
the same volume estimator; rim suppression is a TD550/K100 attempt residual
against the strictly earlier-day league rim percentage. Rim saved equals
`2 * estimated contests100 * estimated suppression` exactly.

Entire unavailable tracking families remain null with a missing flag. The
adapter does not make up old tracking measurements or substitute zero ability.
The historical mapper is responsible for its explicitly trained available
feature recipe and missingness behavior.

## Legacy source gates

The existing modern per-game extractor is reused inside an isolated pandas
reader facade. Historical `seconds_remaining_quarter` already contains the
whole clock, so the adapter first parses `time_quarter` as `MM:SS`; it does
not add historical minutes to that already-complete seconds field. Canonical
raw and processed files remain unchanged.

Pre-2014 raw sources omit steal/block text and block-player IDs. The modern
extractor's apparent zero totals are explicitly changed to missing before
feature estimation. A separate annual source recovers actual pre-2014 steals
and blocks from `player_sheets/year_totals/{year}.csv`: unique exact NBA IDs,
integer nonnegative STL/BLK, positive GP/minutes and full NBA Advanced `POSS`
are checked for every year 2000–2013. Aggregate implied pace must lie between
70 and 120 possessions per48minutes and totals must be nondegenerate.

Only the completed regular season is admitted, with conservative availability
July1 afterward. June30 is an explicit aggregate end marker, not a claimed
game date. Annual full counts are divided by the full official `POSS` total,
never by an incomplete canonical PBP denominator. `POSS` is the NBA Advanced
overall playing-exposure measure, not a claim of exact canonical defensive
possession identity. The source scraper declares `MeasureType=Advanced` and
`PerMode=Totals`. All source vintages remain uncertified.

These annual measurements are combined with daily canonical defensive
possession measurements from2014 onward without overlap. Provenance names
this source and cadence explicitly. The changed denominator/cadence is a
historical limitation requiring predictive overlap validation, not a silently
recovered per-game STL/BLK archive. Current-season early steal/block skill
updates before2014 do not occur until the next July1 aggregate is available.

The raw archive is incomplete in some warmup years. Initial audits count
1188 RS games in2000 and1189 in2001–2004, when the league had29teams; do not
compare those years to the later30-team1230-game schedule. 2011 has1229
and a documented missing raw game. 2013 has1229 and requires its actual
season schedule when judging completeness. No player-game is invented for a
missing game and no complete-season claim is made. 2005–2010
have1230 RS games, and lockout2012 has990.

The final feature source gate excludes 14 games: the four existing2014/2015
score-accounting quarantines, nine confirmed duplicate-event2002playoff games,
and `0029901052` (raw189 points versus numeric final188). A zero last-score
string alone is not corruption. Numeric scoreboard maxima recover final146
for `0029900739`,214 for `0020200077`, and208 for `0021500359`, all matching
their raw points, so those correct raw measurements are preserved despite
canonical possession mismatches. The target audit files are hashed inputs;
feature and lineup-target source gates serve different roles.

Shooting/nonoffensive PF and offensive fouls drawn remain missing before the
first validated measurements. They are not extrapolated from the historical
action subtype code. Aggregate personal PF and assists are separately
nondegenerate in the early PBP archive. Unresolved assist-name tags are counted
in per-year receipts; these sources are not claimed to exactly reconcile every
player's official assists.

## Legal annual-OU comparison package

`historical_features_ou.py` writes a separate
`results/estimated_skills/vpm_historical_features_ou_v1/` package with the same
wide feature interface and long provenance. It uses the original annual OU
export code and original defensive OU/rim engines, leaving their source and
model files unchanged. Legal original vintages start October2015. October2020
uses the2019 model until the delayed December2020 fit becomes available. The
latest existing2025 model is carried through October2026; no2026 fit is
invented. All learned model fit dates precede each emitted cut.

Tracking extra summaries and MPG are copied from the final gated fixed-EB
package because those fields already use that same fixedTD550/K1000 or
observedMPG definition in the selected method. The OU own-skill states are
never replaced by EB estimates. Pre-2015-October OU is unavailable; the sibling
harness must choose an explicit historically trained backbone policy.

Original OU own-skill source quarantine conventions are preserved verbatim,
rather than pretending the source was newly refitted after the historical
raw audit. This is a disclosed difference from the newly gated EB source.

The completed fixed-EB package contains159,526 rows across60dates and
2,981,339 provenance rows. All52requested quarterly cuts are present. The OU
package contains136,786 rows across47legal dates. On September1,2026, all25
own-skill values and their missingness reproduce the original monthly annual-OU
package exactly for its1,707 original players, with maximum gap0.
The separate OU audit checks1,454,172 provenance rows (1,140,440 trained and
313,732 fixed extra-field rows), verifies every learned model checksum and
fit-cut/fit-end fence, and reproduces42,675 current-reference cells exactly.

## Operation

```
.venv/bin/python nba_pipeline/scripts/historical_features.py --stage lines
.venv/bin/python nba_pipeline/scripts/historical_features.py --stage features
.venv/bin/python nba_pipeline/scripts/historical_features_audit.py
.venv/bin/python nba_pipeline/scripts/historical_features_ou.py --workers 2
.venv/bin/python nba_pipeline/scripts/historical_features_ou_audit.py
.venv/bin/python -m pytest nba_pipeline/tests/test_historical_features.py nba_pipeline/tests/test_historical_features_ou.py -q
```

Generated parquet and logs stay outside Git. Code, tests and this receipt are
the reproducible interface. A changed extraction source lock requires a fresh
versioned output directory. The15targeted tests cover prefix invariance,
missing versus observed zero, annual source availability, coherent shooting,
clock reconstruction, quarter coverage, and legal OU model vintages. The
independent artifact audit reconstructs38,192 literal source cells, with
maximum numerical gap1.78e-15, and checks all emitted date/missingness and
shooting/rim identities.

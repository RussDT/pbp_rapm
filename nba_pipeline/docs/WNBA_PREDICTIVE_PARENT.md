# WNBA predictive parent

Owner-requested September 29, 2026 build. Status: completed research build; not a public-model promotion.
The sibling VPM harness owns modeling, configuration, evaluation, and boards.
RAPM owns the isolated actual-point and per-game count adapter.

## Why a new lane

The NBA simplicity result motivates mapping own estimated skills directly to
future offense/defense, then inserting the prior inside the lineup ridge.
The earlier WNBA predictive lane used component priors, settings tuned across
2023–2025 for its historical filters, and a target named actual points that
actually included full-season expected FT percentages. Its reported MSEs are
not interchangeable with the actual-point test here.

## Data contract

`wnba_test/export_wnba_predictive_targets.py` calls the canonical clean-chance
parser with opt-in `actual_fts=True`. The existing default remains expected FT.
The new branch does not load player-season FT percentages. First-FGA identity
is captured before terminal/lineup filtering, preventing an excluded possession
from donating its shot to the next terminal.

Every admitted game has ten distinct positive player IDs on each possession,
an exact date, and both teams' possession point sums matching the boxscore.
The initial frozen source contains 212,152 possessions in 1,287 games,
2022–2026 through August 30, with 95 source games excluded. The exclusion
ledger records scoring, lineup, and date failures. These are source-integrity
exclusions, not performance-based exclusions; results apply to this subset.
Player-game zone counts reconcile exactly to official FGA/FGM. Exposure is
the mean of observed offensive and defensive possessions, not a fixed
possessions-per-minute estimate.

The parser still computes historical continuation children internally, but the
parent uses only actual FG+FT points. No season-level kappa enters its target.
The SQ challenger uses **gross** first-chance MID/THREE actual-minus-context EV
residuals. It does not claim net-continuation denoising. Rim stays actual.

## Chronology and modeling

- Skill vintages are fitted before January 1 of their emission year. Per-stat
  discounted sufficient-count filters select half-life and pseudo-exposure
  using only earlier next-game observations. Volume uses on-court possessions;
  percentages use attempts. League pooling excludes the current game date.
- Snapshots contain only games strictly before each estimate date. Predictive
  makes per 100 equal predictive attempts per 100 times predictive make
  percentage. No total impact rating or cumulative wins enters a feature.
- Preseason own skills map to fully completed next-season parent O/D labels.
  The selected historical mapper can use 2010–2023 labels for 2024; it cannot use 2024 or later labels.
  Mappers are frozen within each season; own-skill snapshots refresh at cuts.
- Entrant population effects are profiled jointly with player coefficients in
  historical label solves, rather than averaging tiny-minute shrunk labels.
  Future lineup players without a prior WNBA appearance receive that explicit
  fallback. College/international background priors are not yet available.
- One generalized ridge fits player offense, points-allowed defense, season
  intercepts, and a centered home term. Prior means enter the ridge objective.
  Defense changes sign once for positive-good display.
- Pure RAPM and parent candidates receive separate development selection on the
  same decay/offense/defense grid. Parent maps also vary mapper strength and
  prior scale. Conservative paired one-SE selection precedes SQ testing.
- The SQ challenger fits context EV from earlier seasons only, using fixed
  retrospective smoothing settings. It solves EV parent plus two residuals
  and verifies exact equal-penalty coefficient closure. Residual penalties are
  selected on development; admission requires the paired 95% interval to
  exclude zero against the selected parent.

## Evaluation contract

Selection is 2024 and 2025. Monthly May–September cuts predict disjoint future
windows, with the final window through October. Training is strictly before
the cut. The primary target is future actual possession points; games are
weighted equally within cuts and cuts equally within season. Uncertainty is
paired and game-clustered. The 2026 replay is spent, report-only stress. 2027
is protected and is not loaded by the experiment.

Game-margin error is also reported by summing signed possession forecasts.
Both models use the same **realized future lineups and possession counts**:
this isolates player quality but is not a full pregame game-forecast test.
Historical architecture/grid choices and retrospectively acquired sources mean
these are chronological development replays, not recreated public vintages.
Intervals are conditional on selection; untouched confirmation is still needed.

## Selected model and completed evidence

Immutable `research_harness_wnba_v6_parent.json` / `wnba_predictive_parent_v6`:

- Direct actual-point O/D parent; **SQ disabled** in the recommended model.
- Five-season likelihood window, 550-day half-life, offense alpha **2000**,
  defense alpha **3000**, prior scale **1.0**, standardized mapper ridge **10**.
- Offensive mapping: predicted rim/mid/three/FT makes and attempts, assists,
  turnovers, offensive rebounds. Defensive mapping: defensive rebounds,
  steals, blocks, personal fouls. No post-solve impact blending.
- Low observed exposure blends toward the profiled entrant population with
  `own_weight = min(observed_WNBA_possessions / 2000, 1)`. Players beyond
  2,000 observed possessions retain their own-skill prior exactly. Exposure
  means observed source history, not a certified complete career ledger.
- Prior snapshots refresh fully at each monthly cut through the same frozen
  annual mapper. No 2026 outcome selects these settings.

The initial modern-only all-feature parent was inconclusive. Compact recipes,
entrant blending, and an entrant-only control were evaluated separately.
Version 6 extends priors with historical evidence and wins the final paired
one-SE family comparison against versions 1–5. That family choice is adaptive
development research, not an untouched model-selection exercise.

Primary differences below are candidate minus comparator in possession MSE
×1,000,000; negative is better:

| Comparison | 2024–2025 development | 2026 spent stress |
|---|---:|---:|
| Selected parent − independently tuned pure RAPM | **−286 [−554, −17]** | −276 [−621, +80] |
| SQ parent − selected parent | −39 [−219, +150] | −174 [−443, +102] |
| Selected parent − DPM, all rows | +170 [−231, +562] | −220 [−874, +423] |
| Selected parent − PI, all rows | +477 [+25, +926] | −109 [−795, +536] |

Development covers 545 games in ten disjoint monthly windows; stress covers
281 games in four windows. Both development seasons favor the selected parent.
The independently tuned pure control is `1000/3000`, TD550. The direct
parent clears its development gate; its 2026 point estimate stays positive
but the interval crosses zero. **Do not claim confirmed superiority over
DPM/PI**: PI wins development, and stress comparisons are inconclusive.
The SQ increment fails admission even though its stress point estimate is
favorable. Its separately retained `10000/20000` residual penalties do not
change the recommended parent.

Conditional game-margin RMSE (shared realized future rotations and possession
counts) changes from **12.536 to 12.423** in development and **12.617 to
12.476** in stress. This is a player-quality isolation comparison, not a
pregame availability/minutes system.

External snapshots use provider dates **strictly before** each cut. Their
calendar publication semantics are uncertified. All-row comparisons use zero
for missing benchmark players; a separate common panel requires all ten
ratings and remains inconclusive versus DPM. Full DPM coverage is 78,735 of
89,096 development rows and 38,391 of 46,880 stress rows. The archived DPM
source ends July 30, 2026; PI/NPI end July 23. These coverage/timing limits are
preserved in the benchmark receipts. No external rating trained a prior.

## Historical compatibility and identity audit

`export_wnba_historical_parent.py` provides **1,980 annual RS count rows**
from 2009–2021 and **1,755 parent-label rows** from 2010–2021. Annual counts
are conservatively timestamped December 31, so they cannot predict games in
the same season. Historical snapshots use fixed 550-day smoothing and
500-possession/100-attempt pseudo-exposure until modern game data permit
next-game parameter tuning; annual aggregates never enter that next-game
scoring objective. From 2023 onward the skill parameters are fitted afresh
using only earlier modern games, with historical states carried in.

LUMA provides validated RS aggregate stints priced as actual points divided by
Oliver possessions. Its prior-free O/D labels use 2000/5000, centered by side
exposure. These are **compatible historical labels**, not the modern exact
possession substrate and not a reconstruction of first chances, SQ, or net
continuation. The modern likelihood and every evaluation row remain on the
actual-FT, complete-game event source.

Crosswalk admission requires a normalized exact name unique in both source
providers; numeric IDs cannot override conflicting names. There are 1,755
mapped of 1,819 historical label rows and zero admitted numeric/name conflicts.
Unmatched rows remain excluded with an audit. No fuzzy name joins are used.
Historical provider publication vintages and older career coverage are not
certified. College/international entrant priors remain unavailable.

## Board and validation artifacts

The selected board is dated **August 31, 2026**, using games through
**August 30**, not a September 29 live refresh. It contains 361 fitted
players and 229 who appeared in the available 2026 source. It is not a
certified next-season roster.

Sibling output `vpm/results/harness/wnba_predictive_parent_v6/` contains:

- `wnba-predictive-vpm.html`: searchable/sortable reader with model switching
  and visible-row CSV export;
- `wnba_predictive_board.csv` / `.parquet`: O/D/VPM, controls, raw ratings,
  observed exposure, own/background prior weights, dates, and source labels;
- `board_manifest.json`: full-universe side-specific exposure centering offsets
  and prediction-preserving intercept shifts; no reader-filter recentering;
- config/source locks, per-stage code manifests, annual skill parameters,
  historical and modern labels, development grid and selection;
- paired game-level development/stress scores and external benchmark coverage.

The curated sibling receipt is `vpm/evidence/wnba_predictive_parent_v6_summary.json`.
No raw provider ratings are included in that receipt or the reader.

The selected development build reproduced exactly under its frozen selection
lock. Browser verification covered search, ascending/descending sorting,
model switching, CSV download (229 rows), and the visual layout.

Seventeen focused tests cover actual FT versus unchanged legacy behavior,
future exclusion, coherent makes/attempts, rookie mean handling, joint-ridge
closure, benchmark pairing, and historical aggregate tuning exclusion.

## Frozen 2027 trigger

Retain the v6 direct parent and pure control unchanged. Refresh audited source
coverage through the complete 2026 season before building 2027 matured labels.
Fit 2027 skill parameters using only pre-2027 games, map only matured labels,
and emit monthly pre-cut ratings under the same window/weighting rules.
Evaluate the protected 2027 windows once without reselection. A public
promotion additionally needs certified source/roster freshness and a shared
pregame minutes/availability test. Do not turn a favorable spent stress result
into an unannounced SQ promotion or retune entrant exposure on 2027.

## October 1 rating-spread audit

The owner flagged compressed ratings. Among the 229 players with a 2026
appearance, selected-board unweighted standard deviations are 2.01 offense,
1.04 defense, and 2.62 total points/100. Defensive own-prior SD is only 0.55.
The prior map learns shrunk annual labels, and the final ridge also regularizes
player estimates. These facts explain narrow estimates but do not establish
that wider estimates would predict better; posterior mean spread is not an
estimate of the population's latent talent variance.

The original selected defense penalty was at the search's lower boundary.
Immutable `research_harness_wnba_v7_spread_audit.json` expands the search to
offense 250/500/1000/2000/4000, defense 250/500/1000/2000/3000/5000, prior scale
0.5/1/1.5/2/3, half-life 365/550, mapper 10. Across 300 parent and 60 pure
specifications, the same paired one-SE rule selects **exactly v6 again**.
Even the minimum-MSE candidate still uses defense 3000. This closes the old
lower-boundary gap; no expanded-grid board replaces v6.

`wnba-predictive --stage calibration` audits selected pre-cut parent forecasts
on 2024–2025 only. It separates the training-exposure-centered O/D contribution
from the baseline, preserving every prediction including unknown-player
fallbacks. Slopes use equal season/cut/game weighting and remove a diagnostic
intercept per cut so environment drift does not masquerade as spread error.
Those fitted intercepts never enter a forecast. Game-bootstrap intervals are
conditional on the already-selected model, not independent confirmation.

| Forward forecast scale diagnostic | Slope | Game-bootstrap 95% interval |
|---|---:|---:|
| Common O/D multiplier, pooled development | 0.959 | [0.838, 1.077] |
| Offense, jointly fitted with defense | 0.969 | [0.840, 1.101] |
| Defense, jointly fitted with offense | 0.934 | [0.701, 1.150] |

A multiplier above one would support expansion. The common slope is compatible
with one. Defense varies from 1.176 in 2024 to 0.700 in 2025; neither supports
a reliable blanket expansion. Carrying the 2024 O/D slopes (0.927/1.176) into
2025 with the original pre-cut baseline changes MSE by +118e-6
[−38,+280], and conditional margin RMSE from 13.403 to 13.452.

Fixed exploratory 25% and 50% common expansions on the same 2025 forecasts
worsen MSE by +436e-6 [+95,+781] and +1340e-6 [+667,+2036], respectively.
Conditional margin RMSE rises to 13.717 and 14.258. These are realized-lineup
diagnostics on spent development, not new pregame or untouched tests.

**Decision:** retain v6 values and scale. Narrow defense remains a limitation
of available signal; merely weakening shrinkage or enlarging the displayed
numbers did not improve this development test. No 2026 outcomes selected this
audit, and no 2027 source was evaluated. Do not interpret the intervals above
as individual-player rating uncertainty.

Artifacts: `wnba_predictive_spread_v7/development_grid.csv` and selection;
v6 `calibration_report.json` / `calibration_development_predictions.parquet`;
curated `vpm/evidence/wnba_predictive_spread_audit_v7_summary.json`.
Reproduce calibration with the v6 command below and `--stage calibration`;
reproduce the expanded search with the v7 config and `--stage development`.

## Reproduction

From the RAPM checkout, using its environment:

```bash
.venv/bin/python ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_wnba_v6_parent.json \
  wnba-predictive --stage prepare
```

Then run stages `development`, `stress`, and `board` with the same command.
Use `audit` to fit/audit annual feature and label packages without evaluation.
The `benchmark` stage compares against strictly pre-cut external DPM, PI, and
NPI snapshots, never using them as inputs or selecting settings on them.
Config and source hashes lock the experiment; changed observations require a
new experiment identity. Generated artifacts remain ignored in
`wnba_test/Results/predictive_parent_targets_v1/` and sibling
`vpm/results/harness/wnba_predictive_parent_v6/`.

Existing WNBA daily/publication paths and NBA v18/v34 remain unchanged.

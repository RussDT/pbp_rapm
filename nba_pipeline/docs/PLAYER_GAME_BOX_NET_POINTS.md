# Player game net points: box coefficients and PBP difficulty

Owner clarification, 2026-09-13: the intended prior is **game box features scored
with learned factor coefficients**, not a historical player-rating anchor.
Traditional box stats are the primary inputs. PBP-derived playtype difficulty
is an explicitly requested extension. Every reader must expose the actual box,
feature transformation, coefficient, intercept, game contribution, and team
reconciliation before asking the owner to assess player values.

## Calibration correction (2026-09-13)

The original game totals are **not approved as completed-game credit**. Their
single-season STL coefficient is 0.3933; removing mapper ridge gives 0.4031,
so mapper regularization alone does not account for the small coefficient.

Run `scripts/audit_player_game_five_year_coefficients.py` for the corrected
five-year calibration audit. It fits each 2014–2018 through 2018–2022 window
separately against its matching actual Net DECOMP targets, using matching
RS+PS box totals, side-possession weights and unpenalized least squares.
Overlapping windows are sensitivities, not independent validation folds.
All six factors are refitted. dTOV includes STL and OFFD; OFFD is the sum of
the provider's disjoint Charge Fouls Drawn and Offensive Fouls Drawn categories.

Some season sources have only 500 player rows. A missing player row is not
zero: exclude the whole player-window if any played season/type is missing
or its source GamesPlayed differs from the official game-box count. Sparse
OFFD event cells within an otherwise covered source row represent zero.

The audited 2018–2022 fit (706 eligible dTOV rows) is:
`dTOV/100 = -1.598391 + 0.880471*STL100 + 0.692775*OFFD100`.
STL ranges 0.8763–0.9126 across the five windows. No coefficient floor,
post-hoc inflation, or exposure threshold was selected to force STL above 1.
These targets still carry 2000/4000 RAPM regularization. Five-year fitting
improves target/exposure matching but does not establish literal event prices.

The reader displays all original-versus-five-year coefficients and preserves
old totals only for audit. Five-year coefficients have not silently replaced
game outputs. Local raw PBP lacks offensive-foul drawer attribution; the NBA
CDN returned 403 and PBP Stats returned 500 during recovery attempts. Do not
substitute total fouls drawn, charges only, season-rate imputation, or zeros.
Game-level OFFD ingestion and event-credit calibration remain unfinished.

Artifacts: `results/player_game_box_net_points/five_year_audit/` contains
coefficients, models, coverage, the exact training panel, and source paths.

Builder: `scripts/build_player_game_box_net_points.py`.
Configuration: `config/player_game_box_net_points.json`.
PBP adapter: `scripts/player_game_playtype.py`.

This supersedes the *player-allocation* choice in
[the v1 prototype](PLAYER_GAME_NET_POINTS.md); it reuses v1's audited team
component targets and player possession exposure. V1's historical player
anchors, arbitrary event opportunity prices, and shooting residual add-ons are
not inputs to the new box calculation.

## Existing work recovered

The sibling VPM scripts contain the earlier models:

- `scripts/estimate_2026_boxscore_factors.py`: native six-factor box priors,
  compact oTS formula/ridge, tracking and role features;
- `scripts/decomp_prior_harness.py`: single-season and multi-year gross DECOMP
  box priors, fitted at EFG/FT/TOV/SC atoms;
- `results/decomp_priors_stacked1y_14_26_curated_v1/manifest.json` and
  `decomp_selected_features.csv`: original feature recipes and model receipts.

The old compact formula was `-1.882 + .604*pts100 + .703*ptdiff_pts100
-.619*TSA100 + .666*rim_ast100 -.059*onball_pct`. It is not silently installed
here: its target, features and native-factor pricing differ from current Net
DECOMP. The recovered DECOMP feature lists also require richer tracking/role
fields than the owner currently wants.

The new coefficients are therefore an explicit **compact adaptation/refit** of
the existing box-to-factor ridge method to Net DECOMP, not a reproduction of
the old frozen coefficients. No earlier predictive validation is inherited.

## Headline feature contract

| Factor | Exact game inputs |
|---|---|
| oTS | PTS, FGA, FTA, FG3A, AST |
| oTOV | TOV, FGA, FTA, AST |
| oSC | OREB, MISSED_FG = FGA − FGM |
| dTS | BLK, PF |
| dTOV | STL |
| dSC | DREB, BLK |

All are actual game counts from the local traditional-box archive. Minutes and
FGM/FG3M/FTM/DREB etc. are shown even when not used in a particular factor.
Source `POSS` is shown next to canonical offensive and defensive exposure so
denominator differences can be inspected. No actual tracking stat, player
identity coefficient, position indicator, or estimated potential assists is a
headline input.

The existing potential-assist backcast was recovered in
`build_nba_rztov_box_backcast.py`; it uses assist points, bad-pass turnovers and
other PBP-derived inputs as well as ordinary assists/volume. The tracking
backfill estimator also uses preceding observed tracking history. Neither is
silently substituted for a traditional box-only estimate here. AST remains an
explicit input; an estimated-opportunity extension needs its own displayed
recipe and source contract.

## Coefficient fitting and game calculation

Build a player-season panel from regular-season plus playoff game boxes and
matching single-season actual Net DECOMP targets. Current feature/alpha choices
are registered in the JSON config before fitting. Training uses 2014–2022;
2023–2025 are historical report-only season-target checks. No 2026 target is
used in coefficient fitting or alpha selection, and no protected future gate is
changed. This is retrospective season-to-game transport, not demonstrated
single-game attribution accuracy.

Features are counts per 100 canonical side possessions. The target is factor
points per 100. Fit weighted standardized ridge, with sqrt(possessions) weights
normalized to mean one within each training fold. Minimum season exposure is
250 side possessions. Development uses expanding season-forward folds after
four training seasons and chooses the largest alpha within one paired-season
standard error of the minimum. Refit on all 2014–2022 only.

Export coefficients back to physical units, including the recovered intercept:

```
rate = intercept + Σ coefficient × (100 × count / side possessions)
box game points = intercept × side possessions / 100 + Σ coefficient × count
```

Then reconcile each factor to its measured team result:

```
team residual = measured team factor − sum(player box-factor estimates)
player adjustment = player side possessions / team player-side possessions
                    × team residual
final player factor = box-factor estimate + player adjustment
```

That adjustment is explicitly visible, not a box coefficient or a hidden
player skill. Large residuals expose how much of the result the compact box
model fails to explain. Opponent-three conversion and the unequal-possession
baseline retain v1's transparent scoreboard identity. Positive-good defensive
signs are retained once; they are not negated again during box fitting.

## PBP difficulty and stabilized zTS proxy

Reuse `classify_pbp_shots` and `estimate_category_prices` from
`analyze_pbp_playtype_difficulty_ots.py`. The six buckets are CUT_2,
SELF_CREATED_3, OTHER_3, SELF_CREATED_2, OTHER_RIM_2 and OTHER_2.
Classification ignores the shot result and made-only assist tags.

For season Y, prices use only completed regular season Y−1, with the existing
100-FGA category-price shrinkage. For each game:

```
raw difficulty = 50 × (prior-season league points/FGA − priced game mix)
prior difficulty = same prices applied to player's Y−1 regular-season mix
w = game FGA / (game FGA + K)
stabilized difficulty = w × raw difficulty + (1−w) × prior difficulty
```

K = 20 is an **explicit provisional sensitivity**, not a tuned/validated player
prior. A player without previous-season FGAs gets a neutral difficulty prior
and a missing-prior flag. No current-game outcome enters the prior or bucket
prices. No-FGA games have no observed playtype adjustment; they remain missing
in the diagnostic and contribute zero candidate role points.

The conventional display uses `TSA = FGA + 0.44*FTA`:

```
TS% = 100 × PTS / (2 × TSA)
rTS = TS% − prior-season league TS%
PBP zTS proxy = rTS + stabilized difficulty
PBP_ROLE_POINTS = 2 × TSA × stabilized difficulty / 100
```

Actual TS/makes/misses are not shrunk. The rate is explicitly a **PBP zTS proxy**,
not production Synergy zTS or optical shot quality. Prior-season league TS is
shown in the reader; this is not a silent substitution into the production
reader's same-season baseline. Every game's six category counts, prices,
shares and difficulty contributions are exported and visible per player.

The PBP role term is a separate oTS candidate, not a headline input. Its
compact season-target coefficient is approximately −0.0203, alongside about
+0.3011 for raw points; do not force the sign positive or import the previous
future-oTS study's ~0.78 relative multiplier. That study used a different target
and additional creation controls. The candidate training feature uses the
full-season mix while its game application would use prior-stabilized mix;
this transport and K remain unvalidated. The PBP diagnostic is useful without
pretending this particular small regression has admitted it into game credit.

Worked sample: Jalen Brunson, 2026-05-19, 38 points on 29 FGA and 10 FTA:
TS 56.886%, raw difficulty +5.359, prior +3.890, weight 29/49 = 0.59184,
stabilized difficulty +4.760, PBP zTS proxy +4.027 relative to 2025 league TS.

## Outputs and checks

```bash
.venv/bin/python nba_pipeline/scripts/build_player_game_box_net_points.py
```

By default reads the three source-audited v1 games already built under
`results/player_game_net_points/`. It does not rerun or change v1's attribution.
The current sample is NYK–CLE games 1, 3 and 4, 65 player-game rows. Game 2's
source mismatch remains a blocker to expanding that series, not a reason to
ignore the source gate.

New outputs under gitignored `results/player_game_box_net_points/`:

- `review.html`: actual boxes, factor estimates, all coefficients, player
  multiplication ledgers, team reconciliation, PBP categories and zTS proxy;
- `players.csv/.parquet`: box values, diagnostic PBP columns, each factor's box
  contribution and explicit team adjustment;
- `feature_terms.csv/.parquet`: every intercept/event/reconciliation term;
- `coefficients.csv/.parquet`, `models.json`: exact primary model parameters;
- `playtype_candidate_coefficients.csv`, `playtype_candidate_models.json`:
  separate role candidate, not used in headline;
- `playtype_buckets.csv`, `playtype_prices.csv`: PBP calculation inputs;
- `reconciliation.csv/.parquet`: box sums → team targets → final sums;
- `training_coverage`, `development_alpha_grid`, `historical_report`: fit
  provenance and season-target diagnostics;
- `manifest.json`: complete config, source hashes, underlying team contract,
  selected games and implementation provenance.

Require traditional shooting identities, unique player/date keys, exact box
team agreement, player-box points summing to scoreboard, PBP FGA counts matching
the game boxes, physical-coefficient prediction equivalence, all term sums, all
six factor sums, and scoreboard reconciliation. The current focused suite has
21 passing tests across the new box adapter, v1, team builder, and existing PBP
playtype method. These prove calculation integrity, not fair causal allocation.

## Possession attribution audit and feature/target mismatch

`build_player_game_attribution_audit.py` builds a separate one-game review at
`results/player_game_attribution_audit/review.html` for 0042500301. It exports
every possession's six-factor centered actual/three-adjusted observation, each
player's canonical lineup exposure, exact team FT-bridge reconciliation, and
dTOV known-actor/unidentified-event/baseline accounting. These on-court results
are shared observations, not summed player credit. Baseline and unidentified
credit remain unallocated. Existing box estimates are controls, not BPM.

The first complete event-to-box check found **16 first-chance steals and 5
second-chance steals**, exactly reconciling to 21 official steals. In this Net
DECOMP substrate only first-chance possession deletion belongs to dTOV; steals
after an initial shot end second-chance sequences and belong to dSC. The old
all-STL dTOV feature therefore mismatches its target. The two-stage shrinkage
explanation was incomplete. Both old coefficient fits remain unapproved.

The dTOV turnover price is 1.2923055 points in this season/type ledger. Known
first-chance steals can be shown at that accounting price without claiming
sole causal responsibility. No corresponding arbitrary dSC steal price is
invented. The same opportunity-stage audit must precede future fits for TOV,
OFFD, and other actor inputs. Do not force all steals into dTOV without also
changing and validating the six-factor target contract.

Next required work: recover exact offensive-foul drawers; build matching
first/second-chance actor features across training windows; fit the candidate
on that contract; add a faithful documented game-BPM benchmark; validate
allocation and implement separately labeled pregame opponent adjustment.
No claim of improvement over BPM or finalized fantasy scoring is made.

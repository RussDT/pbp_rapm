# Game and Season VPM Impact Architecture

Current replacement: [realized game-credit allocation](VPM_ACTUAL_CREDIT.md).

> **Superseded — owner correction, October 25, 2026.** The estimator proposed
> below and implemented in `vpm_actual_impact_v1` / compensated v2 answered the
> wrong question. The owner wants to divide credit for realized games, then
> aggregate that credit into seasons. Learning a new same-game box-stat-to-impact
> mapping is not the requested allocation architecture. In particular, the
> fitted defensive-rebound slope assigned shooting-outcome credit to rebounders
> through same-game association. Numerical closure does not validate that
> assignment. Both implementations remain withheld; do not publish their boards.
>
> The replacement must start with the observed Net possession/lane accounting
> and specify how its credit is divided among players. Keep the value being
> divided distinct from evidence about responsibility for it. A defensive
> rebound can inform possession-completion responsibility without establishing
> responsibility for the preceding miss. Use the existing VPM framework where
> justified; do not silently invent a new box-score metric or replace the
> rejected mapping with another outcome-correlation fit. The precise allocation
> rules and treatment of unidentified/shared credit still need to be specified
> and audited before another rating board is produced. The original design is
> retained below as a record of the rejected approach, not an active recipe.

October 24, 2026. Architecture recommendation for the owner's clarified product:
**single-game impact and single-season impact, each with a complete Net DECOMP
breakdown, impact rate and total value**. Implementation and model selection
remain subject to the gates below.

## Decision

Build one descriptive impact engine at **player-game grain**. Publish Game VPM
from those records and derive Season VPM from the same records. Every game,
component and exposure must reconcile to the season product within one release.

| Output | Question | Units |
|---|---|---|
| Game VPM | What was the player's adjusted impact rate in this game? | O/D/Total points per 100 possessions |
| Game Value | How much did that game performance contribute over the possessions played? | Adjusted net points above average |
| Season VPM | What was the player's average impact rate this season? | Exposure-weighted O/D/Total points per 100 |
| Season Value | How much did the player contribute over the season? | Sum of Game Value |
| Net DECOMP | Where did that impact come from? | Identical additive tree in every output |

Keep Predictive VPM as the separate ability-at-a-date product. The existing
[annual](VPM_1997_HISTORY.md) and [daily](VPM_DAILY_1997_HISTORY.md) histories
use earlier evidence with TD550; neither is a season-restricted actuals ledger.
Summing daily predictive ratings over minutes would measure production through
a predictive-rating lens, rather than the requested game-performance estimate.

A change in a season rating's cumulative value is not a single-game impact
estimate: new observations can revise all earlier season estimates. Game
impact must exist explicitly in the estimator and export.

## Shared estimator

Use **observed game statistics plus season-level adjusted lineup evidence**,
with partial pooling across a player's games. Historical seasons teach the
impact mapping and regularization; the target player's performance evidence
belongs to the requested season.

```text
Admitted actual-scoring possessions and observed player-game summaries
    -> pooled game-stat / season-lineup impact estimator
    -> player-game O/D coefficients and exact Net lanes
    -> matching actual game O/D exposure -> Game Value by lane
    -> sum values and side exposures -> Season VPM and Season Value
```

Do not solve a standalone unpooled RAPM for each game. A game's rotations
provide little information for separating teammates and opponents. The result
needs a shared mapping and regularization; numerical solvability alone is not
useful individual identification.

### Model specification

For player i, game g in season s, write native O/D impact schematically as:

```text
beta_ig = Z_ig B + a_is + u_ig

minimize over B, a, u, gamma:
    sum_g || y_g_actual - X_g beta_g - H_g gamma ||^2
    + sum_is a_is' Lambda_season a_is
    + sum_ig u_ig' Lambda_game u_ig
    + mapper_regularization(B)
```

- `Z_ig B` is the impact signal from that game's observed box-score and
  admitted advanced statistics, including opportunities and coverage.
- `a_is` is the shared season lineup adjustment beyond those measured stats.
  It carries information about impact that the game statistics miss.
- `u_ig` is an optional strongly pooled game-specific lineup adjustment. Its
  incremental value must pass the registered comparison; the simpler `u=0`
  model is the first implementation and required control.
- `H_g gamma` contains registered context outside player credit: initially
  a season intercept and home court. No free game or team intercept silently
  removes the performance being estimated. Rubberband is a separate sensitivity.

Use points per possession in the likelihood, with native beta equal to
`(O, -D) / 100`; report positive-good defense. Give admitted possessions uniform
time weights within the season. The season adjustment has a new player-season
key each year. Career skill filters, TD550 and age projections belong to the
predictive product, rather than being reused as performance observations.

Fit B on completed earlier seasons through the full historical lineup
likelihood, jointly profiling player-season residuals and, for the challenger,
local game residuals. Freeze B and regularization for the target season. Estimate
its season/game adjustments from admitted actual scoring through the requested
cutoff. Do not treat shrunken standalone RAPM labels as exact training responses.

The season adjustment is not itself Season VPM. Season VPM is the exposure-
weighted aggregate of the final player-game coefficients. With `u=0`, Game VPM
is a game-stat-informed impact estimate with a shared season lineup adjustment;
it must not be described as an independent measurement of that game's lineup
effect. Show the game-stat signal and shared adjustment in the detailed download.

Same-game statistics are intentionally allowed in this descriptive estimate.
Statistics and lineup outcomes share games; this is an empirical-Bayes/
penalized estimator, not two independent likelihoods. Historical validation must
rebuild every same-game-derived input inside its folds.

### Game evidence and season boundaries

Build observed **player-game** summaries from validated counts, rather than
reusing the predictive-stat state exports. Their career memory and neutralizing
filters answer a different question. Preserve observed makes with attempts,
actual FT, exact opportunities, missingness and optional tracking coverage.
No-opportunity and unavailable are different states. Never manufacture a rate
by dividing by a zero denominator.

Begin with a compact core available across the intended eras. Use a historically
calibrated basis and reliability rule for extreme tiny-exposure game rates;
do not clip final impact ratings or erase observed counts. A modern replacement
requires both mapper admission and player-level observation. Tracking gaps
remain missing, rather than zero or a reason to discard a valid core game.

Admit only completed games for the exact NBA ending season, phase and exclusive
cutoff, using source season keys even in the October 2020 bubble. A season is
not the preceding 365 calendar days. Target statistics and lineup exposure
share the same admitted census. Quarantined games leave targets, counts and
pricing together. Keep official GP/minutes and coverage separate from native
admitted exposure.

Regular season is the first release. Playoffs are separately labeled and fitted;
their tiny samples need explicit reliability reporting. A combined release
should fit the union under an explicit phase/context model and aggregate its
game ledger. It is not an arithmetic average of two independently fitted rates.

## Exact Net DECOMP at both grains

Use the existing price-free actual-point atoms and
`vpm_net_lane_targets.targets`. Reprice continuation and reference values from
exactly the admitted season/phase games through the cutoff. Use actual FG and
FT, raw unfiltered continuation counts, matched nonturnover references and the
standard midrange-reference display convention.

Fit additive lane mappings by retargeting the same historical model geometry.
Every lane uses identical features, nuisance design, exposure, preprocessing,
regularization and solver. Their mapper coefficients, shared season effects,
game effects and final player-game coefficients must sum to a separately
solved actual-point Parent. There is no independently tuned penalty per lane
and no balancing component. Factor the common system once for all lane RHSs.

The native representation has nine additive lanes plus two auxiliary
mid-reference contrasts. The reference contrasts redistribute the shot lanes;
they are not two extra additive buckets. The player-facing tree is the same
for a game and a season:

```text
Total = Offense + Defense
Offense = oTS + oTOV + oSC
Defense = dTS + dTOV + dSC
TS = Rim + Mid + Three + FT

Standard shot detail:
    rim frequency, rim conversion, midrange value,
    three frequency, three conversion, FT
Direct lanes:
    net turnover value, net second-chance value
```

These are lineup-adjusted team contributions associated with a player's
performance. Rim value is not simply the player's own rim points; second chance
is not simply the player's rebound count. Retain the existing Net continuation
payback and FT-reference identities.

### Realized shooting

The [predictive Parent](VPM_NET_LANE_DECOMPOSITION.md) replaces the weak-ridge
response to mid/three actual-minus-SQ residuals with a more strongly shrunk
response. Do not apply that extra predictive correction to the actuals total.
Use the actual-point likelihood and its common regularization throughout.

An optional observed-SQ breakdown may split conversion into EV and
actual-minus-EV. Both children must use the same linear operator and additive
prior allocation so their sum is the actual conversion lane. Retain the full
realized residual in the target; do not suppress it with the predictive
12000/48000 residual penalties. Earlier/unobserved SQ remains unavailable,
not an invented zero. Actual Net detail can ship before optional SQ detail.

Ordinary regularization still stabilizes who gets credit for an outcome.
An actual-point target is not a uniquely identified causal assignment of all
shooting luck to individuals.

## Rates and total value

Use one league-season reference frame per release: center offensive and
defensive player-game coefficients on their respective full-season-to-date
exposures. Save raw coefficients, offsets and prediction-preserving context
intercepts for every lane. Do not recenter each game separately or recenter
after search, qualification or team filters.

For each player-game, use matching admitted O/D possessions:

```text
OV_ig = O_ig * OPoss_ig / 100
DV_ig = D_ig * DPoss_ig / 100
GameValue_ig = OV_ig + DV_ig

SeasonOV_i = sum_g OV_ig
SeasonDV_i = sum_g DV_ig
SeasonValue_i = sum_g GameValue_ig

SeasonO_i = 100 * SeasonOV_i / sum_g OPoss_ig
SeasonD_i = 100 * SeasonDV_i / sum_g DPoss_ig
SeasonVPM_i = SeasonO_i + SeasonD_i
```

Apply the same equations to every lane. Zero side exposure has no side rate;
its observed contribution is zero. Do not divide total value by five or halve
the O/D contributions. Use side-specific exposure rather than assuming the two
possession totals are equal.

| Example | O rate | D rate | O possessions | D possessions | Game Value |
|---|---:|---:|---:|---:|---:|
| Game 1 | +4 | +1 | 60 | 60 | +3.0 points |
| Game 2 | +2 | -1 | 40 | 40 | +0.4 points |
| Season aggregate | +3.2 | +0.2 | 100 | 100 | +3.4 points |

The season impact rate is +3.4 in this example, and Season Value equals the
sum of the two Game Values. This example is illustrative, not a fitted result.

For trades, sum the corresponding game rows into each team stint; those values
sum exactly to the player's season value. No extra independent team-stint fit
is needed. Default game leaderboards sort by Game Value, since rate alone can
promote a one-minute appearance. Show rate, exposure and low-sample flags beside
it. Season views support both rate and total-value rankings with explicit
qualification rules.

Total value is **adjusted net points above league average**, rather than an
exact allocation of scoreboard margin. Opponent adjustment, nuisance effects,
shrinkage and model residuals prevent an exact scoreboard identity. Preserve
that residual outside player credit. Wins above average or replacement need a
separately declared baseline and conversion; the existing predictive team-win
logistic is not automatically an additive individual-wins formula.

## Historical revisions and publication

Use a **coherent revised ledger** as the default actuals product. When more
season evidence arrives, the shared adjustment, reference prices and game
estimates may change. Publish a new release containing mutually consistent
game and season tables. Every season total sums games from that same release.
Once a season is closed, changes require an explicit source/model revision.

Preserve as-published historical releases. A game viewed as estimated the next
morning and a game viewed using the final season reconstruction can differ;
show the estimate cutoff and model vintage. Never aggregate rows silently
selected from different releases and call them the coherent season total.

This revision policy is part of the product, not a leakage claim: a final
retrospective game estimate can use other games in that season for its shared
adjustment. An as-of release cannot use games on/after its cutoff. Neither
is a pregame forecast.

Use versioned keys:

```text
game:   (release_id, model_id, season_end_year, phase, game_id, player_id)
season: (release_id, model_id, season_end_year, phase, player_id)
```

Include game date, exclusive estimate cutoff, data-through date, model release
date, source/recipe hashes, mapper training cutoff, source-vintage/backcast
flags, raw/centered O/D and lanes, model adjustments, O/D possessions and values,
coverage, and audit status. Store centers/context once per product. Use an
explicit uniform-weight policy and null half-life, rather than zero as a
no-decay sentinel.

Consumer navigation: **Predictive / Season / Game**, with matching Net detail
and **Rate / Total Value**. A selected season and game must retain the same
release and reference frame. Downloadable game and season files expose the
aggregation identity directly.

## Validation and implementation gates

The architecture fixes player-game grain and exact aggregation. It does not
establish the best feature basis, pooling strength or usefulness of a separate
game residual. Register these comparisons before producing their scorecards:

| Model | Game-specific statistical signal | Lineup adjustment | Purpose |
|---|---|---|---|
| Game statistical model | Observed game stats | None | Simple performance baseline |
| Shared season model | Same observed game stats | Shared player-season adjustment; u=0 | Recommended first implementation |
| Pooled game model | Same observed game stats | Shared season plus strongly pooled game residual | Challenger requiring incremental evidence |

Also retain season-only actual-point zero-prior and frozen-preseason-prior
controls to diagnose the season aggregates. A preseason control must be an
audited estimate released before that season, with a fixed series definition;
the selected September 2026 v61 board cannot serve as the 2025 preseason input.

Learn mapping, preprocessing and pooling only from earlier completed seasons.
Use the established 2020-2022 development window for selection, with a small
preregistered grid and fixed rule. Keep 2023-2025 report-only historical stress,
2026 report-only and 2027 outcomes protected. These comparisons are research
evidence, not untouched confirmation. Legacy predictive penalties are control
settings, not calibrated game-impact precision.

Validate with full-game held-out folds and known-effect lineup simulations.
Rebuild all season adjustments, observed summaries, references and preprocessing
from each training fold. A held-out game's observed box summary is allowed only
in a **box-conditioned descriptive reconstruction** score, explicitly so labeled.
It cannot be called a game forecast or compared with pregame EPM forecast error.
For an auxiliary genuine prediction test, exclude that game's statistics too.

Implementation refinement, October 25: select by equal-season held-game
possession error after subtracting each game's offensive-side observed and
predicted scoring averages. Same-game box statistics already contain scoring;
game-margin reconstruction can reward an exact scoreboard identity. It is
reported as a diagnostic, never the selection objective. Centering is applied
only to this scoring contrast, not the estimator or published player values.

Held-out outcomes assess conditional reconstruction and attribution stability;
there is no observed ground truth for each individual's adjusted game effect.
Use simulations to examine recovery under known effects, shared rotations,
extreme exposure and correlated game noise. Report paired game-cluster
uncertainty and low-exposure, trade and era/coverage slices. Do not select on
in-sample fit or player rankings that look familiar.

Required algebra/source checks are:

- Actual FG+FT and Net action/processed-row closure, matched game identity and
  exposure, and exclusion of quarantined source/pricing games.
- Direct Parent versus all lane priors, effects, final rates and context;
  normal equations and prediction-preserving centering.
- Game O+D and lane closure; exact game-to-season O/D value/exposure/rate
  aggregation; team-stint partitions; zero exposure and missing coverage.
- An as-of prefix remains identical after later games are appended or changed.
  A uniform-weight endpoint is unchanged by reordering its games. Different
  coherent revised releases may intentionally differ.
- Full-precision downloads, reader filtering without recentering, phase/season
  keys, release IDs and game-to-season drilldown all agree with the package.

For uncertainty, resample whole games and rebuild their observed inputs and
season adjustments. Preserve O/D and lane covariance for totals. Historical
game IDs omitted in a resample require an explicitly defined conditional
prediction/draw for that game, not a fabricated resampled appearance. Label
fixed-mapper/recipe intervals as conditional; they do not include model selection
or cure shared-lineup nonidentification. Validate the uncertainty procedure
before publishing numeric game-level intervals.

### Ownership and computation

RAPM owns canonical actual-point atoms, validated observed player-game
summaries, exposure/identity/scoring gates and Net row algebra. The sibling VPM
harness owns historical mapping, hierarchical estimation, comparison, exact
game/season aggregation and immutable product packages. VPM consumes admitted
summaries rather than raw event-credit rows or proprietary defender assignments.

Reuse the existing Net target adapter and profiled likelihood/sparse solver
primitives. Preserve frozen experiments. Add a new `actual-impact` command under
`vpm/run_harness.py` with `prepare`, `calibrate`, `compare`, `ledger`, `audit`
and `package` stages behind a fresh config/output root.

With `u=0`, game-stat prior means can be offset from each possession response,
and the shared season adjustment reuses one season O/D normal system. If the
game-residual challenger is admitted, eliminate game-local O/D blocks through
a Schur complement rather than building a dense matrix across all player-games.
Cache each common factorization for all Net RHSs. Profile nuisance effects
exactly; subtracting only y's mean does not generally profile an intercept from
an uncentered lineup design.

Deliver in this order:

1. Prepare a completed modern regular season and a few actual games from the
   existing clean census. Freeze a game-stat basis and historical training plan.
2. Implement the game statistical baseline and shared season adjustment;
   produce one player-game ledger and mechanically derive its season table.
3. Complete historical calibration/comparisons and decide whether game residuals
   add enough identified signal to retain. Freeze the full recipe.
4. Independently verify direct-parent/lane equations, actual game/season value
   sums, chronology, exposure and the coherent reader/download package.
5. Publish the paired Game and Season product behind its own release pointer.
   Extend eras, playoff coverage and nightly wiring through separate source/
   operational gates.

No separate season-first publication is needed while game values remain undefined.
Keep selected predictive `current_model.json`, v61, completed histories and their
recipes separate. The existing `nba_no_prior_vpm_td550_snapshots` publisher
cannot receive these rows: its table denotes a different estimator.

This architecture review adds no NBA model fits, calculated game ratings,
deployment or schedule. The next concrete milestone is the paired player-game
ledger and derived season-control package, not another predictive replay.

## Source basis

Repository contracts: [Parent Net decomposition](VPM_NET_LANE_DECOMPOSITION.md),
[annual history](VPM_1997_HISTORY.md), [daily history](VPM_DAILY_1997_HISTORY.md),
[actual-point source history](VPM_REPRODUCTION.md) and [model map](VPM_MODEL_MAP.md).

[Dunks and Threes' official methodology](https://dunksandthrees.com/about/epm),
checked October 24, 2026, distinguishes predictive EPM, season Actual EPM and
single-game Estimated Game Production. It describes the latter as observed
box-score production. That illustrates distinct product questions; it does not
validate our proposed game-level adjusted estimator or identify its formula.

[Joseph Sill's original RAPM paper](https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing)
supports regularized lineup estimation and out-of-sample testing. The shared
game/season actuals model, exact ledger contract and proposed pooling comparison
above are our architecture, with no claimed single-game causal identification
or validated wins conversion.

# VPM pregame prior training plan

Created October 21, 2026. **Owner authorized implementation after the full-history library completed; evidence-panel work is underway.** See [implementation status](VPM_PREGAME_BUILD_STATUS.md) for the neutral-core bridge, exposed legacy identity issue, tests and remaining gates. No improved VPM has been established.
This work belongs to the existing predictive VPM RESEARCH project.

## The proposed change

Learn how a player's **predictive box-score/advanced stats immediately before a game** translate into impact in that game's actual lineups. Repeat across historical games, retaining all eligible older observations with decay. The objective is a prior for current, near-term impact. A separate future-year forecast could still use the existing annual-snapshot approach.

The selected model currently holds a dated stats snapshot fixed while training against actual lineup scoring over the following 365 days. Those training windows pool across years; VPM does not use only one year of total history. The proposed approach updates the input snapshot before each game, so a player's improvement, decline or role change during the year is represented when it becomes observable.

| | Selected annual-snapshot calibration | Proposed game-by-game calibration |
|---|---|---|
| Player inputs | Predictive stats at cohort start | Predictive stats available before each game |
| Outcome | Actual possession scoring during the following 365 days | Actual possession scoring in that game |
| Lineup adjustment | Joint O/D player residuals within cohort | Joint O/D player residuals within matched training blocks |
| Training history | Pool fully matured historical cohorts | Pool completed historical games; test recency decay |
| What a new game supplies | Eventually contributes to a matured annual cohort | Supplies a completed target; later stats states can incorporate it |

The proposal remains forward prediction: the outcome happens after the input snapshot. It is **regularized, lineup-adjusted prior calibration against actual points**, rather than a regression onto precomputed APM or RAPM player labels. It does not require fitting a separate impact model for every game.

## What the code audit established

Most core filters already retain older player observations. This is primarily a change to the timing of calibration inputs, plus a separate test of training-example decay—not a universal replacement of one-year stat windows.

- The annual OU replay reads its available history, beginning in 2014. Legal annual OU parameter vintages begin in October 2015. Existing monthly exports save neutral postgame states and hold the latest earlier state at a cut; those exports are not neutral states explicitly evolved to every game date. See [monthly replay](../scripts/build_monthly_own_skills.py), [snapshot selection](../scripts/export_vpm_own_skills.py), and [legal vintages](../scripts/historical_features_ou.py).
- The filter kernel already calculates predictions before an observation. Its exported prediction includes game context, so a neutral adapter must expose the evolved latent state and restore the dated league center using the existing correction conventions. Simply shifting postgame exports or copying contextual forecasts would change the quantity being estimated. See [OU wrapper](../scripts/monthly_ou_filter.py) and [kernel](../scripts/monthly_ou_kernel.cpp).
- Earlier historical coverage uses a distinct decayed empirical-Bayes estimator with all admitted earlier observations. It is not the selected OU estimator extrapolated backward. Early annual steal/block sources also cannot be presented as daily observations. See [historical features](../scripts/historical_features.py) and [quarterly history](VPM_QUARTERLY_HISTORY.md).
- MPG is a real hard-window exception: trailing-365-day regular-season minutes per positive-minute appearance. Preserve it in the matched timing test; evaluate a decayed all-history alternative separately. The [matched MPG decision](VPM_MPG_DECISION.md) retained the existing MPG input.
- Some measurement panels contain only positive-denominator rows. The new exporter must query states for **every relevant player-game**, including games with no attempts in that family. A query advances the state through elapsed time without inventing a measurement. See [measurement panels](../scripts/build_nba_estimated_skills.py).

## Keep three kinds of recency separate

| Clock | What gets older | Initial treatment |
|---|---|---|
| Predictive-stat memory | A player's past measurements | Retain each family's legal, frozen estimator and its own decay/dynamics |
| Prior-calibration memory | Historical games used to learn the stats-to-impact mapping | First test no additional decay; then a small declared half-life grid |
| Final lineup-rating memory | Past lineup evidence updating a player's prior | Keep selected Parent TD550 unchanged |

“All history” means all available, validated, chronologically usable observations for that source. A state-space filter can carry their information recursively; it need not reread every game daily. There is no reason to force every stat family or all three clocks to share a half-life. The strength of decay should be selected on prediction, rather than assumed to be heavy.

## Phase 1: build an immutable pregame evidence panel

Start over the legal annual OU period. Use existing annual parameter vintages, with each vintage fitted only on outcomes available before its deployment period. Extend the earlier empirical-Bayes history only after the overlap-period timing test, as a separately labeled estimator/history change.

**New source discovery, October 21:** [Databallr game-log inspection](DATABALLR_GAME_LOG_SOURCE_AUDIT.md) verified Denver core logs in all 30 seasons 1997–2026 and daily player core examples back to Jordan in 1997. `report=true` adds merged tracking/advanced reports where available. Audit this source for an earlier daily observation panel before relying on annual-only early counts; it does not establish legal earlier OU parameters. Keep the source/history extension separate from the matched timing test and respect tracking-vintage/FT reliability flags.

For each game, freeze the neutral stats snapshot before tipoff, then attach it to that game's ten-player possession lineups. With date-only sources, use an earlier-date cutoff; do not infer within-day availability. Preserve the distinction between an event-date-safe reconstruction and a historically certified publication vintage.

Store one row per player/game, with long-form per-family provenance:

| Field group | Required contents |
|---|---|
| Identity and cutoff | Exact player ID, game ID, season ID, game date, prediction cutoff and time precision |
| Predictive state | Neutral pregame value, uncertainty where available, units, missing flag/reason, pregame accumulated exposure |
| Chronology | Last admitted observation date, source watermark, parameter training cutoff/vintage |
| Reproduction | Source, parameter and code hashes; estimator recipe; feature/preprocessing version |
| Admission | Mapper-admitted family, observed status, aggregate/replacement choice |

Use the audited actual-FG-plus-FT possession targets, exact O1–O5/D1–D5 identities and existing scoring/entrant exclusions. Each target possession appears once in the new panel. Actual game minutes or possessions determine target exposure; **they do not enter pregame predictors or pregame eligibility**. Do not replace future-exposure thresholds with another future-minute gate. Any reliability rule must use information available before the game.

Apply model-aware historical fallbacks **before preprocessing and design construction**: retain aggregate DREB/PF until the replacement is both admitted by that dated mapper and observed for the player. An observed zero is valid. Calibration and deployment must use the same transformation; see the v64 rejection and v65 repair in [quarterly history](VPM_QUARTERLY_HISTORY.md).

First deliverable: a small pilot covering ordinary games, season transitions, trades, long absences, zero opportunities, newcomers and delayed tracking. Expand only after these checks pass:

1. Changing the target game's box score or any later observation cannot change its pregame features.
2. Changing later data cannot change an earlier fitted preprocessor, parameter vintage or mapper.
3. Independent scalar replay agrees with prediction-before-update, including idle-time evolution and context removal.
4. Pregame exposure excludes the target game; missing measurements never become observed zeros.
5. Every retained possession joins exactly ten valid player states, including explicit population/missing-state fallbacks where needed; every exclusion has a reason.
6. Target scoring, dates and lineup IDs match the clean source manifests; source and lineup coverage are reported by season and family.

**Exit:** reproducible pregame data and chronology receipt. No prior fit is needed to establish this milestone.

## Phase 2: fit the prior directly against lineup outcomes

For possession p in game g, define:

- y_p: actual offensive points on the audited target row;
- z_i,g: player i's standardized neutral predictive stats frozen before game g;
- A_p: summed offensive feature vectors and negative summed defensive vectors, divided by 100;
- X_p: corresponding signed O/D player indicators, without /100 scaling;
- u: O/D player residuals in points per possession within a training block (beta's displayed impact coefficients are in points per 100);
- C_p: intercept/context nuisance design.

Fit the shared offensive and defensive mapping coefficients beta through:

```text
prediction_p = A_p beta + X_p u + C_p gamma

minimize sum_p w_p * (y_p - prediction_p)^2
       + u' Lambda u
       + beta' R beta
```

The learned prior for a player is the O/D mapping applied to their current pregame stats. The player residuals allow lineup evidence not explained by those stats; they are regularized nuisance terms during calibration. They are not future knowledge available at deployment. Do not fit free player effects separately for every game.

Profile the residuals and nuisance terms out of the weighted joint regression, preserving full covariance. Use the current cohort-block residual definition and O2000/D4000 calibration penalties for the matched timing comparison. Initially retain mapper ridge O10/D100, with feature units/scaling explicitly matched. These calibration penalties differ from the final Parent O4000/D8000 penalties.

The current [forward mapper](../../../vpm/vpm/harness/repaired_forward.py) uses a static player-feature prior design within each cohort. Its [profile solver](../../../vpm/vpm/harness/long_spm.py) supplies useful algebra, but game-varying A requires a new dynamic-design implementation; it cannot be achieved by substituting a minutes-weighted annual average into the existing call.

Build game-level sufficient-statistic caches and sparse residual solves. Verify the profiled solution against a small direct joint solve, including weights, missing flags, centering, units and O/D signs. At each fit cutoff, normalize the new arms' total training weight to M0, the unweighted common-control possession count. Retain the stated penalties in the resulting native-point summed objective; an equivalent mean objective divides every term, including penalties, by M0. Do not normalize within each historical season, which would erase between-season decay. Record raw/effective possession weight and normalization so a shorter half-life does not silently change total data-versus-penalty scale. This is a declared weighting convention, not extra independent evidence.

### Season and home context

The current forward calibration has cohort intercepts but no explicit actual-season or home indicator. Test context separately from input timing:

- Use categorical season effects, with an identified intercept/reference convention. Never use season ID as a numeric trend by accident.
- Use a centered offense-side home indicator, with neutral-site games at zero, from the repaired venue source. Record unresolved venue coverage.
- Start the timing comparison with matched current nuisance terms. Then test adding season/home jointly to both timing arms. Keep context contributions separate from player impact.

The existing [context ridge helpers](../../../vpm/vpm/harness/context_ridge.py) can supply mechanics, but earlier context experiments do not establish a game-forecast benefit. Any new scoring baseline or home coefficient used in forecasts must be estimated from past data, not from the evaluation season's eventual scoring.

**Exit:** a verified dynamic mapper and a reconstructed static-input control on identical data.

## Phase 3: isolate timing, then test decay

Register the experiment before fitting; reserve the next unused immutable experiment ID during implementation. Keep the exact feature recipe, entrant handling, actual-FT targets, SQ construction, output scale 1, final Parent O4000/D8000/TD550 and mid/three residual O12000/D48000 fixed. No manual player adjustments or added feature search.

Run the following comparisons in order:

| Comparison | Question answered |
|---|---|
| Archived selected v61 | What does the existing selected system predict? |
| Matched static control A | What does annual-snapshot calibration do on the new common, deduplicated panel and solver? |
| Pregame arm B | Does refreshing stats before each game help, with identical games, weights, residual blocks, context and penalties? |
| A and B with season/home | Does explicit context help independently of input timing? |
| Chosen timing/context recipe with calibration decay | Does giving recent training games more influence improve forecasts? |

The current annual cohorts contain overlapping COVID games. For A/B, assign each game once to the latest eligible annual starting snapshot whose following-365-day window contains it, using the same assignment and training blocks for B. Use their common eligible game set and disclose gaps. Reproduce the original archive separately; differences caused by deduplication, eligibility or solver normalization cannot be attributed to pregame timing.

For calibration decay, use all eligible completed games before each mapper fit cutoff T, with:

```text
w_game = 2 ** (-(T - game_date).days / H)
```

Test H = 365, 730, 1460 days, plus no additional decay. All possessions within a game share its age weight. Do not multiply by game minutes again: possession rows already supply playing exposure. Normalize the objective as specified above, report effective sample size and retain older source observations without an arbitrary calendar truncation.

Only after the matched timing/decay test, evaluate these separate extensions if justified:

1. Earlier eligible empirical-Bayes history, with estimator differences disclosed and overlap controls.
2. All completed training games beyond annual-cohort eligibility, using declared nonoverlapping annual residual blocks.
3. Decayed all-history MPG versus the retained trailing-365-day MPG control, with explicit shrinkage/exposure semantics.

**Exit:** one declared historical candidate, or a documented decision to retain the current method. No superiority claim is implied by the new training horizon.

## Phase 4: evaluate the thing we want to predict

Use chronological replay. At every annual mapper fit, admit only games completed before that cutoff; fit upstream parameters, preprocessing and mappings from legal earlier data. Within the following season, freeze mapper coefficients while advancing each player's pregame state after completed games. Refresh states and re-estimate mapping coefficients are different operations.

For the matched A/B comparison, give both arms the same rating refresh schedule and the same pregame current-state inputs at deployment; only their mapper-training input timing differs. Report two views:

1. **Rolling pregame forecasts:** each game's ratings use only earlier games, with the common refresh schedule. Primary loss is equal-season game MSE; also report RMSE, paired game-cluster differences and season-level uncertainty.
2. **Existing seven-test benchmark:** retain the established annual-cut game/team-season, continuity and veterans-only scorecard for continuity with selected v61 and dated EPM. Label its frozen-cut horizon separately from the rolling test.

Hold actual-minutes/pace conventions, missing-player fills, entrant classifications, home handling and forecast probability conversion constant across systems. Actual minutes/pace make these conditional rating-quality tests, not full pregame availability forecasts. If EPM lacks comparable dated daily snapshots, restrict direct comparison to common dates or the established annual-cut benchmark rather than interpolate future values.

Use 2020–22 for the declared historical selection rule: literal minimum equal-season game MSE, deterministic lexical ties. Freeze the resulting recipe before 2023–26 diagnostics. Both periods have already been used in prior research; later scores are **spent historical evidence**, not an untouched holdout. Report every registered candidate, including null results. No 2027 outcomes are read or used under this plan.

Supplement the game tests with prior-only and final-rating native actual-point errors on common possessions; offense-only/defense-only diagnostics; coverage, prior spread and reliability by pregame exposure; entrants, team changers, long absences and low-continuity teams. Native possession fit alone is insufficient for adoption. Do not select a second winner from diagnostic subgroups.

Paired intervals must state whether upstream fits and model selection are frozen. Conditional intervals are not full model uncertainty. Preserve clustered dependence; do not treat possessions from the same game or overlapping tests as independent replications.

**Exit:** an audited scorecard distinguishing timing, context, decay and extra-history effects, with the existing selected model still reproducible.

## Phase 5: preserve decomposition and define operations

Apply the existing exact Net lane accounting to the new learned prior and final Parent solve. Reconstruct own, entrant and combined O/D priors; retain TS/TOV/second-chance and shot-frequency/conversion/FT/SQ children, including the midrange-reference convention. Use common designs, weights, penalties and nuisance treatment for additive closure. Season/home terms stay outside player lanes; no unattributed intercept is hidden in a lane. See [exact Net decomposition](VPM_NET_LANE_DECOMPOSITION.md).

The proposed operating cadence is:

- After each completed game and validated source arrival: update player states using frozen legal filter parameters.
- Before a game: produce a dated neutral snapshot and refresh the Parent rating from earlier lineup evidence using the common declared schedule.
- Annually at the declared cutoff: refit the mapper using eligible past games and the frozen selected decay recipe. Changing the refresh cadence or retuning filters is a separate experiment.

Build immutable checkpoints, source freshness checks and rollback before claiming live readiness. The existing [daily operations inventory](ESTIMATED_SKILLS_DAILY.md) documents unwired adapters; this plan does not make them live.

Deliver an implementation receipt, reproducible configuration/manifests, chronology and solver tests, exact decomposition audit, and one readable comparison against the selected model. Preserve selected **v61 clean dmap_a100**, quarantined v64 and separate v65 history. Changing the selected pointer, publishing results or starting protected-outcome evaluation is outside this plan.

## Implementation order and completion criteria

| Milestone | Concrete output | Required evidence |
|---|---|---|
| 1. Pregame data | New neutral game-date exporter and immutable pilot/panel | Prefix invariance, independent replay, coverage and fallback checks |
| 2. Dynamic calibration | New sibling VPM harness module and static control | Direct/profile solve parity, identified nuisance terms, objective/penalty scaling |
| 3. Registered experiment | Immutable candidate registry and chronological annual mappings | Matched A/B, separate context and decay comparisons, all candidates retained |
| 4. Forecast evidence | Rolling and seven-test scorecards with paired results | Common sources/conventions, no future inputs, spent-data labels |
| 5. Handoff | Exact Net lanes, operations specification and recommendation | Prior/final closure, reproduction, unchanged selected-model artifacts |

The first implementation task should be **Milestone 1**, not a new model fit. It establishes whether the required pregame evidence actually exists and behaves correctly before we interpret predictive results.

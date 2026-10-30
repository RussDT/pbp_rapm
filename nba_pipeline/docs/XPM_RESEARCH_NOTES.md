# George Berry xPM: methods and considerations for VPM

Recorded 2026-09-20. Research notes and owner requirements, not an implemented
model change or a verified reproduction of xPM.

## Sources and evidence status

- Owner-supplied post: George Berry, **“xPM: Bayesian advanced stats for the
  WNBA and NBA — my take on prior informed rapm”**, dated September 6, 2026;
  explicitly a living post. Attachment:
  `/Users/russ/.codex/attachments/131f0d71-4eb1-44c1-aa15-28bece80d287/pasted-text.txt`.
- Owner-pasted [VORP.app](https://vorp.app/) model 1.1 scorecard, NBA and WNBA
  publication dates September 20, 2026. Preserve this version separately from
  narrative-era results; the attachment's figures are absent from its text.
- These notes distinguish author-reported behavior from proposed VPM work.
  We have not audited his code, priors, snapshots, training data or bootstrap.
  The public package names are leads, not evidence that the full xPM pipeline
  and historical predictions are publicly reproducible.

## Owner's exact objective and architecture

Build **2020–2026 VPM with no future-data leakage**, evaluated against EPM,
DPM/DARKO, LEBRON, BPM and xPM where dated historical ratings are obtainable.
Use ending-year season labels explicitly: 2020 means 2019–20, etc.

The current VPM is the **simpler own-skill O/D parent plus the midrange and
three-point ShotQuality EV/residual split** (v18 baseline). It is NOT the full
component-prior model. Do not turn this request into a component rebuild.
Preserve the current v18 artifacts as reproduction controls; their historical
skill parameters and selected penalties are not automatically vintage-safe.

The owner's principal concern is missing college/G-League/rookie priors,
illustrated by Alex Antetokounmpo. Entrant handling is part of the requested
complete model, not a cosmetic leaderboard repair. This remains within the
existing VPM RESEARCH project; no model promotion or new project is authorized
by these notes.

## What the author says he built

| Area | Reported approach | Consideration for VPM |
| --- | --- | --- |
| Core model | Prior-informed RAPM fit as one Bayesian model | Compare statistical assumptions and predictive results, not the Bayesian label alone |
| Joint terms | Box priors, player random effects, position adjustments, home court and aging in the same model | Potentially account for joint uncertainty; require separate ablations before adoption |
| Units | xPM is points above replacement per 100 possessions; replacement is zero | Establish centering before comparing player values with average-centered VPM; equal common offsets cancel in balanced five-versus-five contrasts but not replacement/WAR accounting |
| Two time views | Age-forward preseason xPM versus year-end xPM without age forwarding | Separate current ability from next-season forecast and test the aging increment |
| Coverage | NBA and WNBA treated as first-class products | Evaluate leagues separately; NBA success does not establish WNBA transfer |
| Cutoffs | Fits and pipelines only access data available by the date, including normalization | Audit every learned upstream stage and every source's availability date |
| Uncertainty | Bayesian fitting produces uncertainty intervals; scorecard lists intervals as upcoming work | Distinguish model capability from published/calibrated intervals |
| Implementation | `blayers` for Bayesian inference, `iv` for pipelines | Inspect only if useful; no reason to replace the existing harness solely for tooling |
| Separate submodels | Minutes forecasting and rookie pseudo-box generation sit outside the joint model | Joint player estimation still leaves important uncertainty and timing outside the fit |

The post does not specify the full likelihood, prior distributions, time-decay
recipe, random-effect covariance, exact position treatment, inference accuracy,
or hyperparameter selection history. It does not demonstrate that Bayesian
joint fitting caused the reported benchmark advantage. Prior-centered ridge
itself has a Gaussian Bayesian interpretation.

## Rookies, lower leagues and the Alex issue

### Reported xPM recipe

1. Translate college or foreign-league statistics into a rookie NBA box-score
   line; use that synthetic line as the player's cold-start box prior.
2. Blend cold-start box evidence with observed NBA performance as it arrives.
3. Author says roughly 20 games give the model a useful handle on impact;
   treat this as an empirical claim, not a universal reliability constant.
4. Evaluate draft prediction against actual pick order and a model combining
   statistics with pick information. He reports NBA pre-draft statistics alone
   are somewhat worse than pick order, while a combined model can improve
   ordering, with information concentrated around picks 5–10.
5. WNBA pre-draft statistics perform much worse than pick order. The post cites
   a large Olivia Miles miss as a failure example. An anecdote is not a
   calibration study or a prescription to tune to that player.

**The post does not explicitly document a G-League translation.** College and
foreign-league pseudo-box priors are the reported method; extending the idea to
our G-League evidence is our proposal.

### VPM requirements

- Recover actual dated college/G-League evidence and stable IDs. Audit Alex
  (`1630828`) end to end, alongside other entrants, veterans and returnees.
- Existing dated public Alex G-League ratings show evidence exists; they are
  not an NBA-scale prior, and their quoted standard errors do not automatically
  capture cross-league translation uncertainty.
- Fit translations only on historical movers whose relevant NBA outcomes have
  matured before each cutoff. Embedded `unified-rapm` all-time mappings fail
  this historical contract without refitting.
- Compare translated pseudo-box-to-O/D with a compact direct O/D transfer
  mapping if feasible. Keep the chosen path inside the simpler parent model.
- Account for source coverage, age, staleness, role/league differences and
  translation error. Broad uncertainty is not a reason to assume average NBA
  ability, nor is low NBA playing time itself proof of poor ability.
- Fit an explicit fallback for entrants without usable lower-league evidence.
  Do not average zero-shrunk low-minute RAPM to learn that fallback or tune it
  to make Alex look reasonable. Preserve prior mean and precision separately.
- Include fringe players and report conditioning on NBA participation. A
  player who never appears has no observed NBA impact label; keep selection
  and availability outcomes separate from impact.
- Draft position is a potential dated input, not mandatory and not useful for
  every entrant. Compare stats-only, pick-only, combined and fallback rules on
  chronological cohorts before choosing.
- Test learning curves by NBA exposure and background instead of hard-coding
  a 20-game transition. Own box and lineup evidence share games and are not
  automatically independent evidence.

## Evaluation design worth adopting

### Frozen preseason player-rating diagnostic

The author freezes ratings before the target season and evaluates future game
margins using actual player minutes and possessions. This holds exposure
constant across methods. It tests ratings conditional on realized exposure;
it is not a deployable pregame forecast because exposure is observed afterward.

He gives rookies a shared replacement value because not every comparator
projects rookies. He reports sensitivity to replacement choice and a second
comparison with prospectively fitted method-specific calibration and rookie
baselines. The pasted scorecard does not fully expose those sensitivity runs.

For VPM, publish BOTH:

- a common entrant-fallback comparison isolating returning-player ratings;
- a complete-model comparison including dated entrant priors, with explicit
  coverage and fallback rules for every method.

Do not silently drop games with unknown players or compare each model on its
own favorable sample. Report all-game coverage and the common-method sample.
Check exact EPM variant (predictive snapshot versus retrospective actual),
DPM vintage, LEBRON version, and BPM season. Current downloadable historical
ratings are not proof that those values existed before the evaluated games.

### Team-season aggregation

Aggregate frozen ratings using actual game minutes and possession-weighted
season ratings; report RMSE, correlation and slope. This reveals systematic
team-level misses that noisy game scores can obscure. It still conditions on
realized deployment and is not an independent preseason win-total forecast.

### Real pregame forecasts

Use projected minutes based on prior games, periodic player ratings and
pre-tip information. The narrative describes preseason player fits corrected
with team residuals: if one team has exceeded its baseline by one point and
its opponent has fallen short by one, shift their matchup by two points.
The later scorecard says periodic ratings; the precise model 1.1 update recipe
needs verification rather than assuming the narrative remained unchanged.

For VPM, keep a team-residual layer separate from player ratings, shrink it,
select its memory/update rate chronologically, and apply the same procedure to
all player-model comparisons. Never carry a team correction onto a traded
player as individual ability. Test ratings, minutes, injury handling and team
corrections separately. Also forecast pace for a genuinely pregame point-margin
forecast; actual possessions are allowed only in the labeled exposure diagnostic.

The author notes incomplete historical injury data, injuries included in
lineups but not fully integrated into rollforward, and possible weekly refits.
He contrasts refitting current player effects with a harder player-by-year
architecture. Consider refresh cost and reproducible as-of snapshots, without
assuming either architecture is superior.

## Metrics, uncertainty and interpretation

- Margin: RMSE, correlation and calibration slope; also retain mean error and
  intercept to distinguish bias from spread errors.
- Win probabilities: log loss, Brier, probability calibration slope, pick
  accuracy, and sample counts. Verify the exact definition of `p slope`.
- Brier CORP decomposition: `Brier = UNC - DSC + MCB`; report uncertainty,
  discrimination and miscalibration on the same sample. In-sample isotonic
  recalibration is a diagnostic, not a prospective performance result.
- Publish season rows and pooled results with explicit weighting, dates,
  postseason rules and coverage. Correlation or pick accuracy alone is not
  enough to choose a model.
- Author reports bootstrap probability of being best among player metrics,
  excluding betting lines, and robustness to calibration/rookie treatment.
  We need paired score differences and intervals, exact resampling unit,
  season sensitivity and selection history. Aggregate scores cannot recover
  statistical significance. Dependence across games deserves sensitivity
  analysis; resample repeated game rows together.
- Betting lines provide a strong information-rich benchmark, not an oracle
  ground truth. Match games and line timestamps. Against-the-spread accuracy
  alone does not establish profitability; the author explicitly treats it as
  unsettled.
- Distinguish prior mapping error, current impact uncertainty, future ability
  drift, minutes/availability uncertainty and game noise. Propagate covariance
  for O+D and lineup sums; marginal error bars cannot simply be added.

## September 20 model 1.1 scorecard transcription

These are author-reported numbers from the owner's paste, not our results.

| Evaluation | Seasons | N | RMSE | r | Margin slope | Brier | p slope | Pick % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NBA preseason, actual exposure | 2004–2025 | 28,342 games | 12.79 | .402 | .870 | .210 | .982 | 67.1 |
| NBA preseason team | 2004–2025 | 660 team-seasons | 2.77 | .817 | .912 | — | — | — |
| NBA rollforward, projected minutes | 2025 | 1,315 games | 14.54 | .472 | 1.092 | .206 | 1.002 | 69.0 |
| WNBA preseason, actual exposure | 2009–2025 | 3,881 games | 12.37 | .394 | .889 | .214 | .994 | 66.6 |
| WNBA preseason team | 2009–2025 | 206 team-seasons | 4.44 | .728 | .989 | — | — | — |
| WNBA rollforward, projected minutes | 2026 | 311 games | 13.28 | .413 | .856 | .219 | .851 | 66.2 |

Rollforward rows are completed games with closing lines, including
preseason-based picks; they are different universes from the preseason rows.
The full supplied NBA year rows show material era variation: game RMSE rises
from 11.69 in 2004 to 14.70 in 2025; margin slopes span .624–1.124.
Team-season RMSE spans 2.10–3.45 and slope .640–1.297. Do not infer a causal
reason for that variation from this table alone.

Common NBA preseason comparison: **2020–2025, 7,737 games**, actual exposure:

| Model | RMSE | Log loss | Brier | MCB | DSC | r | Margin slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xPM | 14.06 | .616 | .2135 | .0017 | .0353 | .409 | .983 |
| EPM | 14.17 | .623 | .2163 | .0019 | .0327 | .396 | .902 |
| LEBRON | 14.21 | .624 | .2170 | .0016 | .0318 | .387 | .993 |
| DPM | 14.28 | .627 | .2182 | .0018 | .0307 | .379 | .914 |
| BPM | 14.50 | .643 | .2225 | .0057 | .0303 | .381 | .680 |
| Home Mean | 15.42 | .688 | .2473 | .0002 | .0000 | — | — |

Shared UNC=.2471; rounding explains small identity discrepancies. xPM's
0.11 RMSE advantage over EPM is about 0.8% RMSE or 1.5% MSE. Most of its
reported Brier advantage appears in discrimination, not miscalibration.
No paired interval can be deduced from these aggregates. There is no supplied
2026 NBA cross-metric row and no supplied historical xPM prediction file.

The older narrative gives different rollforward Brier/line comparisons
(NBA .218 versus line .207; WNBA .206 versus line .199) and says WNBA game
correlation is higher. Do not splice those into the September 20 scorecard,
whose pooled preseason NBA/WNBA correlations are .402/.394. Record version
and sample differences before interpreting apparent contradictions.

## Product and communication ideas

The author builds player cards/top-ten lists, rate impact, projected wins above
replacement, game win probabilities, team rotation/minute forecasts, playoff
simulation/modal brackets and draft analysis from the model. Availability can
change total-value rankings even when per-possession rankings differ.

Consider for our reader, after validation:

- Separate current estimate, age-forward projection, expected playing time,
  accumulated value and projected value; expose each assumption.
- Show a player's prior evidence and NBA update, missing-source status and
  calibrated uncertainty, especially for entrants and injured returnees.
- Publish methodology, timestamped model versions, benchmark samples,
  limitations, sensitivity results and forthcoming changes.
- Show team rotations so users can identify whether a surprising team forecast
  comes from player quality, expected minutes or availability.
- Treat player ranking anecdotes and draft successes as illustrations, never
  acceptance criteria. Do not copy xPM's replacement zero or player-WAR-to-team
  win claims without a separate explicit counterfactual and win conversion.
- The author questions whether our WNBA VPM is prospective. That is his
  interpretation of public documentation, not a verified audit; make current
  fit, historical as-of reconstruction and scored future prediction distinct.
- His announced priorities are uncertainty intervals, all-season rollforward
  and as-of API access, improved rookies and improved injury data. Those are
  useful completeness checks for our own product.

## Leakage-free 2020–2026 work contract and next decisions

1. Preserve/reproduce v18, then inventory each season's source coverage and
   all external metric vintages. No silent substitute for missing xPM history.
2. Define preseason snapshots (including the delayed 2020–21 start), outcome
   dates, regular-season/postseason scope and exact final-score game targets.
3. Fit every upstream skill filter, normalizer, imputer, aging/transfer model,
   prior mapper, SQ model and calibration only on eligible earlier evidence.
   Mature forward labels before training. Audit season-level net-continuation
   prices and expected-FT construction too; score literal actual final margins.
4. Select tuning parameters using earlier inner folds, or freeze them from
   pre-2020 development. Today's v18 penalties selected on later seasons cannot
   simply be called leakage-free for 2020. Preserve that distinction even if
   the architecture is held fixed. Log method-selection and data-cutoff dates.
5. Build the entrant/G-League path and explicit missing-data fallback; give the
   Alex audit priority without making his desired rating an optimization target.
6. Save per-player preseason ratings, dependency manifests and per-game
   predictions. Prove future-append invariance by changing later data and
   checking earlier emissions do not change.
7. Score the common-exposure game and team-season panels against all available
   metrics, with both shared entrant fallback and full entrant models. Use
   identical earlier-data calibration opportunities and report raw sensitivity.
8. Separately score truly pregame predictions with shared forecast minutes,
   availability, pace and context. Add the team-residual layer only as an
   explicitly evaluated increment.
9. Publish coverage and paired uncertainty, not just winner rankings. A nested
   historical reconstruction can exclude future data while still remaining
   retrospective research; it cannot erase our prior exposure to these seasons.
   Preserve untouched 2026–27 for the final frozen confirmation.

Open evidence requests: xPM historical player snapshots or per-game forecasts;
exact benchmark vintage/cutoff rules; rookie sensitivity results; bootstrap
method; model 1.1 update schedule; full generative specification; interval
coverage; historical injury availability; lower-league translation details.
Contacting the author requires an explicit user instruction; no outreach has
been sent.

Related: [VPM consolidation plan](VPM_CONSOLIDATION_PLAN.md),
[predictive build path](PREDICTIVE_VPM_BUILD_PATH.md),
[own-skill integration](../../../vpm/vpm/docs/OWN_SKILL_PARENT_INTEGRATION.md),
[SQ parent](../../../vpm/vpm/docs/OWN_SKILL_SQ_PARENT.md).

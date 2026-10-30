# One VPM product, two model versions

September 20, 2026. Owner-requested implementation plan. No model, rating,
artifact, or release default changes as part of writing this plan.

## The decision

Make the existing sibling **`research/vpm` the product repository**. Extend its
working harness with selected modules from the embedded `unified-rapm`, and
keep RAPM as the authoritative producer of possession targets and own skills.
One command surface, one input catalog, one evaluation system, one reader.

Keep exactly two active model architectures:

| Model | What it estimates | Why it exists |
| --- | --- | --- |
| **VPM Parent** | Own-skill offensive/defensive priors → SQ EV parent plus separately regularized midrange/three residuals | Our current baseline; improve its information and uncertainty without rebuilding component priors |
| **VPM Components** | Own-skill component priors → individual additive component solves → summed offense/defense | The requested rebuilt version; must earn its extra complexity against Parent |

Both use the same identity, as-of features, lower-league information, target
versions, evaluation rows, and forecast assumptions. They differ in how priors
and lineup evidence become impact. Component detail is useful only if the
summed prediction works.

Our current model is the owner-preferred starting point. Existing development
results justify preserving it, not claiming it is already the best future-game
predictor. Andrew's repository supplies valuable missing machinery; its whole
application need not replace our model.

## Freeze what we have first

The exact Parent baseline is sibling config
`vpm/configs/research_harness_v18_own_skills_sq_parent_joint.json`:

- Own O/D prior scale **1.5**, EV-parent O/D penalties **4000/8000**, TD **550**.
- Midrange and three residuals each use **12000/48000**, with zero prior means.
- Exact joint ridge machinery; archived targets matching the original manifest.
- Original feature/imputation behavior retained in this immutable reproduction.

The coherent shooting correction becomes an explicitly named successor change:
impute attempt rate and probability, then derive makes. Preserve v18 as its
comparison. Do not silently replace the original features during a refactor.

The K=500 zero-fallback and learned −6.59-fallback experiments remain preserved
research receipts. **Exclude them from the proposed baseline and build queue.**
Their fitted centers do not establish replacement ability. The report should
eventually put Parent first and label those experiments as historical trials;
the current HTML has not been changed by this plan.

Keep the old 750/3000-family VPM, EPM-skill parent, plain RAPM, and SQ no-prior
as frozen benchmark configurations. They are controls, not additional active
development projects.

## Repository and data ownership

Use the existing VPM repo rather than starting a fourth modeling tree. The
following is a target layout, not a description of files already implemented:

```text
research/vpm/
  README.md                    # what to run and the current decision
  vpm/run_harness.py            # retain the existing collaborator entry point
  vpm/configs/                 # immutable parent, components, and controls
  vpm/data_contracts/           # IDs, dates, units, targets, source versions
  vpm/adapters/                # RAPM exports, NCAA/G-League, benchmark inputs
  vpm/features/                # dated neutral skill snapshots
  vpm/priors/                  # parent, component, and entrant mappings
  vpm/uncertainty/             # prior precision, impact covariance, forecast error
  vpm/models/                  # parent and component orchestration
  vpm/evaluation/              # common possession and game forecasts/scores
  vpm/product/                 # shared board, player audit, exports
  vpm/docs/                    # methodology, runbook, decisions
  tests/                       # contract, parity, and leakage checks
```

Move existing harness internals into this structure only when a concrete
change requires it; retain compatibility commands. The first implementation
wraps working code and produces a score, rather than reorganizing every file.

RAPM continues to own raw NBA PBP, lineup construction, target algebra,
ShotQuality generation, and own estimated-skill producers. VPM consumes pinned
exports and a versioned solver interface. Preserve one authoritative producer
per dataset; do not silently run Andrew's older vendored copies of our PBP or
estimated-skills code.

Import selected Andrew modules with source revision, attribution, tests, and
explicit changes. Treat the nested repo as a donor/reference snapshot. Never
make VPM's execution depend on his public HTML, his laptop paths, or copying
the whole nested repository into another nested repository.

Use a configurable external **`VPM_DATA_ROOT`** for immutable source snapshots,
derived features, targets, caches, run outputs, and forecast logs. Initially
register existing paths: no bulk data move is needed. Small configs and schemas
belong in Git; generated parquets and model outputs need a separate backup and
restore manifest. Record source/code hashes, environment, seed, clocks, units,
sample definition, target version, and output hashes for every completed run.

## What to reuse

| Capability | Starting implementation | Integration rule |
| --- | --- | --- |
| Current Parent and evaluation | Sibling `vpm/harness/own_parent.py`, `own_sq_parent.py`, existing harness | Reproduce before generalizing; keep original configs and results |
| Own predictive features | RAPM `export_vpm_own_skills.py` and `config/estimated_skills_registry.json` | Export neutral state, observation exposure, source timing and uncertainty; next-game context remains separate |
| College/G-League box priors | Andrew `daily_v5/daily_prior/{panels,features,mapping,box_translate,box_prior}.py` | Reuse adapters and fitted-model structure; rebuild translations using only pre-cut information |
| Lower-league RAPM and identities | Andrew `xleague.py`, `aggregate.py`, `data_identity/` | Exact IDs and an audited NBA-scale translation; recover missing data/crosswalk artifacts |
| Rookie-year adjustment | Andrew `build_rookie_model.py` | Candidate for future-year projection, distinct from current-impact prior; validate on all eligible entrants |
| Player-specific prior precision | Andrew `daily_v5/research/prior_precision.py` | Adapt clock/unit checks; calibrate penalties in our actual solver objective |
| Impact uncertainty | Andrew `daily_v5/research/ridge_uncertainty.py` | Reuse covariance calculations; separate sampling uncertainty from future drift |
| Replay and forecast evaluation | Our possession harness plus selected Andrew `replay/` and `harness/metrics.py` primitives | One scorer that actually builds and scores both models on the same rows |
| Minutes, availability, game projection | Andrew `minutes_model/`, `depth_model/`, game/projection modules | Integrate after player-model comparison; both VPMs and EPM receive identical forecasts |
| Reader | Existing own-skill HTML report and its data exporters | One model selector and player audit; retain original run links |

Andrew's saved all-time mover translations are useful reproduction controls,
not historically date-safe priors. His cached-score harness does not itself
re-solve candidates. His latent career smoother is a descriptive surface, and
backward smoothing must not enter a historical predictive snapshot. His
display interval corrections are not automatically forecast intervals.

## Shared priors: use the information we actually have

Build one dated player evidence packet. It contains exact IDs, NBA/college/
G-League observations, age and elapsed time, neutral skill estimates, exposure,
coverage, and source uncertainty. An international/no-data entrant must have an
explicit coverage status rather than silently looking like an average NBA player.

Translate college/G-League evidence onto the NBA target scale using historical
movers whose relevant NBA outcomes were available before the cutoff. Fit to
the same target basis used at inference: a career-anchor translation cannot
be assumed valid for a short daily estimate or rookie-year forecast. Include
translation error and staleness in uncertainty. Record the last actual game,
not just the date of a carried-forward lower-league board row.

Start with one small, testable transfer model. For Parent, map dated lower-league
box/factor evidence to future NBA O/D alongside own NBA skills. For Components,
map the same evidence to the exact component target definitions. Andrew's six
factors are inputs, not an automatic one-to-one mapping to our ten components.

For players without usable prior-league data, learn an entrant population model
from the eligible historical entrant cohort. Account for the noisy NBA targets;
do not estimate its mean by averaging zero-shrunk low-minute RAPM. Include
players who barely play, and report the distinction between an all-entrant
prediction and an exposure-weighted prediction conditional on playing. The
existing mapper's minimum future-possession gate cannot define this cohort.
Players who never play have no observed NBA impact label; preserve their
selection/availability outcome rather than inventing an impact target.

NBA evidence should increasingly inform the estimate as it accumulates. That
transition comes from calibrated evidence reliability and elapsed time, not
a permanent negative adjustment based on lifetime minutes. Lack of NBA data
is uncertainty; the prior mean should come from player/background evidence.

**Alex is an audit case, not a tuning target.** His published lower-league
rating and box estimate demonstrate available evidence, but those public JSON
values are rounded, dated diagnostics. Recover the underlying dated inputs,
verify ID `1630828`, then show exactly how prior and NBA evidence combine.
Do not tune the model until his number “looks right.” Audit other low-minute
entrants, NBA veterans, returnees and players without lower-league coverage too.

## Uncertainty that changes the model appropriately

Each prior emits a mean and uncertainty **in impact units**. The initial
version can use calibrated side-specific variances; label covariance
approximations and validate them before publishing summed intervals.

There are three different quantities to keep separate:

1. **Prior prediction error:** uncertainty in the skills-to-impact mapping,
   including translation and measurement uncertainty.
2. **Current impact estimation error:** uncertainty after the lineup solve,
   including lineup connectivity and dependence between coefficients.
3. **Future forecast error:** current-impact uncertainty plus future changes,
   minutes/availability uncertainty, and game noise for the forecast horizon.

Propagate skill uncertainty through the mapper, including covariance where
available, and estimate the unexplained mapping error on chronological folds.
Do not add measurement uncertainty again if an empirical prediction-error
estimate already includes it. Own box skills and lineup evidence share games;
their errors are not independent evidence by default. Treat the initial model
as empirical Bayes, validate the full procedure, and do not claim exact
Bayesian posterior probabilities from tuned ridge penalties.

Convert prior uncertainty to penalties using the scale of our weighted
objective; `1 / variance` alone is insufficient. Preserve the fixed-alpha
baseline. Coherent-only, transferred-prior means, and calibrated precision
are separate comparisons so we can identify what helps. If a prior mean is
scaled by 1.5, its corresponding variance scales by 1.5².

Estimate uncertainty for O+D, component sums and lineup contrasts jointly or
through joint game resampling. Summing independent marginal standard errors
is not valid. Use prediction errors and coverage against the correct target
and horizon to calibrate forecast intervals; a next-season noisy RAPM label
does not directly validate a current latent-ability interval.

## How the two model versions develop

**Parent** retains the O/D mapper and EV-parent + two residual architecture.
First test coherent shooting; then lower-league/entrant information; then
calibrated heterogeneous precision. Keep residual prior means zero initially.
The baseline does not need component-prior reconstruction to absorb these
improvements. Historical artifacts remain immutable; a daily update necessarily
refreshes dated features and the lineup solve, but should reuse cached inputs.

**Components** rebuilds the prior mappings against matured, date-correct
component targets. It uses the canonical net-continuation shot components,
FT, turnover and second chance, with mid/three EV and residual children.
Start residual priors at zero. Any shotmaking residual prior is a later,
separately tested increment that must account for shooter talent already in SQ.

Retain the existing structured 750/3000-family configuration as the first
penalty control. Select a small registered family of component penalties,
mapper regularization and prior strength on the **summed future actual-point
error**. Do not pick a separate “best” model by each component's R², imitate
EPM ratings, or force component priors to an O/D total with an untested split.
Preserve target-row closure and the additive coefficient sum; neither implies
that the component model must equal Parent's fitted total.

The first architecture comparison holds inputs, target version, date cuts,
sample, decay, and forecast assumptions fixed. New SQ targets, FT repairs or
feature families get explicit versions and matched controls; they cannot be
hidden inside the component rebuild. Proposed improvements remain finite and
named, rather than a new untracked model for every idea.

## One evaluation system

Use the existing VPM harness as the operational spine. Borrow useful Andrew
functions and checks, not the entire research-management framework.

**Level 1 — player quality.** Build both models at each historical cutoff and
score actual future possession points, on identical rows. Retain the common-EPM
sample and an all-player sample that includes rookies and uncovered players.
Report cohort coverage, entrant/fringe/veteran performance, paired game-clustered
uncertainty, and season-level consistency. Repeated appearances of the same
game in overlapping evaluation cuts must remain in the same resampling cluster.

**Level 2 — games with shared realized exposure.** Aggregate each model under
identical realized lineups/minutes. This isolates player-rating quality and is
explicitly labeled a realized-usage diagnostic, not a pregame forecast.

**Level 3 — genuine pregame prediction.** Freeze the roster, availability,
minutes, player ratings, pace and game-context information before tipoff.
Swap only Parent, Components, or complete EPM in the initial comparison.
Score final margin, win-probability log loss/Brier and interval coverage;
freeze the primary endpoint and comparison policy before scoring. Only this
level supports the claim that the complete system predicts games better.

Historical seasons already inspected remain development evidence. Fixed own
skill parameters selected through 2025 do not become vintage-safe simply by
refitting a later mapping. Either rebuild upstream filters within each fold or
label that historical comparison as retrospective development. Record source
availability and methodology-fit dates separately from game dates.

Protect 2026–27. Reconcile the existing holdout registry, including its
November 1, 2026 registered cut, before launching any confirmation run. Lock
the primary candidate, frozen Parent control, finite secondary comparisons,
refresh schedule, sample and success criterion before the first target game.
Do not select between Parent and Components after opening confirmation scores.

## Build order and completion receipts

These are consecutive milestones inside the existing VPM RESEARCH project.
Do not activate every workstream at once.

| Order | Deliverable | Completion receipt |
| --- | --- | --- |
| **1. Preserve and reproduce** | Register source/data locations, recover minimal inputs, and wrap frozen v18 in the shared run interface | Reproduce saved features, priors, coefficients and predictions at current and replay cuts within registered numerical tolerances; hashes for frozen sources |
| **2. Player evidence audit** | Exact identity and source-coverage table; recover NCAA/G-League data/crosswalk and raw prior inputs | Alex plus an entrant/veteran audit set traced end to end; missing inputs named; no website JSON used as training data |
| **3. Parent improvements** | Coherent features, dated transfer/entrant means, then calibrated precision | Paired ablations versus frozen Parent on identical actual-point rows, with coverage and uncertainty diagnostics |
| **4. Component rebuild** | New component prior mappings and a small structured joint selection | Parent-vs-Components receipt plus plain-RAPM, SQ-no-prior and EPM controls; additive closure and matching-source checks |
| **5. Shared game forecasts** | Reuse audited minutes/availability and game modules behind the same forecast interface | Saved pre-tip predictions for every model, dependency clocks, actual outcomes, reproducible paired scores |
| **6. One reader and freeze** | Parent/Components selector, per-player evidence/prior/solve breakdown, comparisons and run links | One local report from run artifacts; prospective recipe and evaluation registry frozen; clean-clone reproduction from the minimal data bundle |

Build the score logging and cutoff contract during milestone 1; do not defer
the ability to evaluate until after both models are written. Audit imported
uncertainty code with its focused tests, then test parity, algebra, identity,
source timing, zero-history behavior and full-model forecasts. Infrastructure
work should directly enable the next receipt.

**First work session:** inventory the exact v18 config and its source manifests,
write the existing-input dependency map, reproduce one saved cut in a new output
directory, and list the minimum missing Andrew data needed for milestone 2.
This is a concrete stopping point before any statistical change.

## The product we should be able to operate

The proposed command surface should expose source checks, one as-of build per
model, a paired evaluation, a player audit, a pregame forecast and a report.
Extend `vpm/run_harness.py`; these capabilities are planned, not newly working
commands. Every command should identify its model/config and output manifest.

The reader answers: Which model is this? What information did this player have?
What did the prior predict? How much did NBA evidence change it? How uncertain
is the current estimate and the future forecast? Does Components improve on
Parent, and does either improve on EPM on the same sample?

Keep one short status page with the current baseline, current challenger,
completed receipt, next physical action, missing inputs and frozen future test.
No separate public app, new target family, international-data acquisition
campaign or full-repo rewrite is needed to finish the first milestones.

## Source receipts

- [Current Parent specification and actual-points sensitivity](../../../vpm/vpm/docs/OWN_SKILL_SQ_PARENT.md)
- [Own-skill integration](../../../vpm/vpm/docs/OWN_SKILL_PARENT_INTEGRATION.md)
- [Existing predictive build path](PREDICTIVE_VPM_BUILD_PATH.md)
- [Estimated-skills registry](NBA_PREDICTIVE_BOX_STATS_REGISTRY.md)
- [Andrew's application/data boundary](../../unified-rapm/README.md)
- [Andrew's cross-league prior implementation](../../unified-rapm/daily_v5/daily_prior/box_prior.py)
- [Andrew's prior precision contract](../../unified-rapm/daily_v5/research/prior_precision.py)
- [Andrew's ridge uncertainty primitives](../../unified-rapm/daily_v5/research/ridge_uncertainty.py)
- [Andrew's latent-smoothing caveats](../../unified-rapm/daily_v5/research/latent/README.md)

Before moving or reclassifying existing files, run the repo inventory dry run
described in [Repo cleanup plan](REPO_CLEANUP_PLAN.md). This planning pass does
not move code or data, refit models, change the report, or publish anything.

## October 15 prior-training design recommendation

[Forward-trained VPM architecture](VPM_FORWARD_TRAINED_ARCHITECTURE.md) specifies
the recommended next Parent training design: learn prior means and then
precision through the existing lineup solve against completed future actual
games. Matured 365-day RAPM remains a baseline/diagnostic. Temporal residuals
and component extensions follow within that architecture. This is a design
recommendation, not an implemented or promoted replacement; v18/v39/v40/v42
remain immutable controls.

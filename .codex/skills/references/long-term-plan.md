# Predictive Six-Factor RAPM and VPM Roadmap

## Table of Contents

- [North Star](#north-star)
- [What Beating EPM Means](#what-beating-epm-means)
- [System Architecture](#system-architecture)
- [Why Decomposition Matters](#why-decomposition-matters)
- [Validation Ladder](#validation-ladder)
- [Fair Game-Prediction Benchmark](#fair-game-prediction-benchmark)
- [Prior Architecture](#prior-architecture)
- [Lineups, Availability, and Synergy](#lineups-availability-and-synergy)
- [Research Priorities](#research-priorities)
- [Non-Goals](#non-goals)
- [Promotion Standard](#promotion-standard)

## North Star

Build a date-aware, interpretable NBA player metric whose player estimates
produce better **future game predictions** than EPM.

The intended product is not merely a cleaner retrospective RAPM or a rating
that resembles public consensus. It must improve forecasts when converted into
team and lineup expectations for games that had not happened at estimation
time.

Preserve two simultaneous requirements:

1. Predictive accuracy: win on frozen future games.
2. Structural legibility: explain the estimate through additive basketball
   mechanisms without sacrificing the forecast.

## What Beating EPM Means

Do not define success as:

- higher correlation with current or future EPM;
- higher correlation with a noisy future RAPM estimate;
- a better-looking leaderboard;
- one favorable point estimate with an interval crossing zero;
- an ensemble that contains EPM unless the claim is explicitly about the
  ensemble rather than standalone VPM.

Define success before opening the holdout. A valid standalone claim requires:

- frozen as-of VPM and EPM values;
- the same future games;
- the same player availability, lineup, and minute assumptions when isolating
  player-metric quality;
- a predeclared game forecast construction;
- a primary scoring rule, such as game-margin squared error, with any win
  probability/log-loss metric declared in advance;
- paired game-level uncertainty;
- wins across multiple dates or one untouched season, not one cherry-picked
  segment.

Report effect sizes and intervals. “Beat” means the predeclared primary error
is lower and the paired uncertainty clears the promotion threshold.

## System Architecture

Build the system in layers:

```text
canonical PBP and box/provider data
    -> date-safe event/possession targets
    -> additive DECOMP/VPM component solves
    -> stabilized factor-specific skill priors
    -> as-of player offense/defense estimates
    -> lineup/team forecast using expected availability and minutes
    -> future game score, margin, and win probability
```

Keep each layer replaceable and auditable. A better shot model should not
require rewriting the lineup solver; a better availability model should not
change historical player estimates.

Use the current ten-component VPM as the player-estimation core:

- eight shot/FT/EV-residual components;
- direct turnover value;
- direct second-chance value;
- offense and defense estimated separately;
- all components additive to actual possession value.

## Why Decomposition Matters

Use decomposition as predictive infrastructure:

- Different mechanisms stabilize at different speeds.
- Each component can use its own offense/defense alpha.
- Priors can be narrow and mechanism-specific instead of broad star halos.
- Failure analysis can identify whether a miss came from shot value, turnover
  value, second chance, availability, or lineup interaction.
- Denoised training targets can reduce luck while exact residuals preserve the
  actual-points total.
- Product explanations can expose the same structure used by the model.

Do not promote a component merely because its own target improves. The summed
future possession and game forecasts decide whether the decomposition helped.

## Validation Ladder

### Stage 0: Algebra and Provenance

Require exact row identities, stable keys, source hashes, as-of dates, config
locks, and reproducible manifests. Do not evaluate a model whose target layer
does not close.

### Stage 1: Component Reliability

Measure multi-season stability, shrinkage curves, calibration, and forward
component prediction. Use this to choose candidate priors and alphas, not to
claim the final metric wins.

### Stage 2: Summed Future Possession Prediction

Grade all components together on future actual possession points. Compare on
identical rows against:

- plain RAPM;
- EV no-prior;
- the accepted prior lane;
- complete EPM converted to the same prediction surface.

Use equal-cut or another frozen weighting rule and paired game-clustered
intervals. This is the current VPM harness's main scientific gate.

### Stage 3: Lineup-Level Prediction

Aggregate as-of player estimates into actual or expected five-man lineups.
Score future lineup offensive, defensive, and net outcomes. Keep an additive
player-only forecast separate from any synergy model.

### Stage 4: Game-Level Prediction

Construct pregame forecasts using only information known before tipoff. Score
future game margins and, when modeled, win probabilities.

Run two comparisons:

1. Player-model isolation: VPM and EPM share the same minutes/availability and
   game context.
2. Full-system comparison: each complete forecasting system uses its frozen
   supporting models.

The first tells whether the player metric is better. The second tells whether
the product is better.

### Stage 5: Untouched Confirmation

Freeze the whole pipeline before opening a new season or predeclared window:

- component targets and alphas;
- prior mappings and scale;
- returning-player and absence rules;
- lineup/minute and availability assumptions;
- game-context adjustments;
- metrics and pass/fail thresholds.

Run once. Report success or failure without retuning on the opened window.

## Fair Game-Prediction Benchmark

Use these rules when comparing with EPM:

- Fetch or freeze EPM values using their real as-of availability contract.
- Never use end-of-season EPM to predict earlier games.
- Map player IDs and active rosters before seeing outcomes.
- Use the same possessions/games and minimum coverage rules.
- Center and scale ratings only with training/development data.
- Do not give one model actual future minutes and the other projected minutes.
- Cluster comparisons by game; do not treat possessions as independent evidence
  for a game-level claim.
- Inspect early-season, trades, rookies, returning players, low-exposure
  players, and availability shocks as predeclared slices.
- Report standalone VPM, standalone EPM, and any VPM/EPM ensemble separately.

Complete EPM remains benchmark material. EPM predictive `p_*` rates may be
candidate skill priors because they are estimated box-score skills rather than
complete impact outputs. Never use `eWINS` as a prior.

## Prior Architecture

Build narrow, factor-specific priors from stabilized estimated skills:

```text
raw date-safe stats
    -> stat-specific talent estimates
    -> component/side prior mappings
    -> prior-centered lineup solve
```

Prefer:

- possession-weighted multi-season skill estimates;
- stat-specific half-lives and empirical-Bayes shrinkage;
- age curves estimated only from training dates;
- compact one-to-five-feature recipes;
- separate offense and defense mappings;
- forward labels matured before the emission date;
- expanding-window season-forward mapper diagnostics.

Candidate mechanisms include:

- shot location, finishing, and attempt-load skills;
- turnover type, creation load, and passing burden;
- second-chance creation and completion;
- rim deterrence, contest, foul discipline, disruption, and block recovery;
- physical measurements when they improve future forecasts;
- stabilized clock/grenade, defender-role, or continuation summaries after
  multi-season admission tests.

Reject broad “offense” or “defense” halos that spray one signal across every
component without incremental evidence.

## Lineups, Availability, and Synergy

Do not confuse a player rating with a complete game forecast.

Separate:

- additive player values;
- expected minutes and availability;
- opponent and home/rest context;
- lineup fit and non-additive synergy;
- coaching and deployment effects.

First test additive VPM against additive EPM under common supporting inputs.
Then test whether a separately validated lineup/synergy layer adds value.

Do not stack retrospective mechanism drilldowns onto player VPM. A synergy
term needs shared-sample shrinkage and its own forward validation.

## Research Priorities

Prioritize work by expected leverage on future games:

1. Build the direct VPM-versus-EPM game-prediction harness.
2. Freeze a common availability/minutes layer for player-model isolation.
3. Extend reliable as-of histories for rookies, returning players, and traded
   players.
4. Improve defensive shot-value priors, where box scores miss the most.
5. Admit richer ShotQuality only through a new frozen forward target gate.
6. Test mechanism-specific personnel and role features rather than generic
   impact leakage.
7. Add lineup/synergy residuals only after the additive player base is stable.
8. Calibrate game margins and win probabilities on development seasons.
9. Preserve an untouched season for the final standalone EPM comparison.

## Non-Goals

- Do not optimize the metric to reproduce EPM.
- Do not select on public leaderboard aesthetics.
- Do not call component-level accuracy a game-prediction win.
- Do not use raw provider rows or fold predictions as player priors.
- Do not let an availability model conceal a weaker player metric.
- Do not reopen spent holdouts.
- Do not silently promote a richer SQ target, prior family, or absence rule.
- Do not collapse the ten factors into one opaque learned score unless that
  candidate is evaluated as a separate, interpretable tradeoff.

## Promotion Standard

Promote only when all of the following hold:

1. Target algebra and provenance pass.
2. The summed future possession forecast improves.
3. The direct game-level comparison improves under common supporting inputs.
4. Paired uncertainty clears the predeclared threshold.
5. Key date/role/exposure slices do not reveal a catastrophic regression.
6. An untouched future window confirms the frozen specification.
7. The public explanation matches the actual model and data lineage.

Short version:

```text
date-safe estimated skills
+ additive DECOMP/VPM lineup truth
+ fair lineup/game aggregation
+ untouched future tests
= a credible attempt to beat EPM at predicting games
```

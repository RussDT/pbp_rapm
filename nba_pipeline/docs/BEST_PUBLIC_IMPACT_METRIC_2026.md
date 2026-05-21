# Best Public Impact Metric (2026 Direction)

## Bottom Line

The best public metric for this repo is not pure RAPM, pure box score, or pure DRL.

It should be a **hybrid public impact stack**:

1. **Factor-specific box/track priors**
2. **Time-decayed six-factor RAPM as the stable core**
3. **A DRL/Shapley value layer for context, leverage, and off-ball defense**
4. **A separate interaction layer for pair/lineup fit**

The public-facing headline number should still be one scalar impact estimate, but it should sit on top of the six-factor decomposition rather than replace it.

## What Recent Public Work Changes

### 1. EPM confirms that priors plus decayed RAPM are still the public baseline to beat

[EPM](https://dunksandthrees.com/about/epm) is explicit about the architecture: machine-learned skill estimates feed a statistical plus-minus prior, which then feeds exponentially decayed RAPM. That is still the strongest public template for a stable predictive metric.

Takeaway for us:
- The public leaderboard number needs a strong prior and recency handling.
- Pure single-season no-prior RAPM is not enough if the goal is "best public metric."

### 2. LEBRON confirms the same lesson in a simpler form

[LEBRON](https://www.bball-index.com/lebron-introduction/) is also explicit: box prior + regularized on/off + luck adjustment + stabilization. That is not as structurally rich as our framework, but it validates the same basic public-metric recipe.

Takeaway for us:
- Do not hand-wave prior weighting.
- Build priors directly into the factor estimation path.

### 3. DRL/Shapley adds something RAPM-family metrics still miss

The [DRL/Shapley Sloan paper](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/699f0fc784ddace11b3f824f_Final%20NBA%20DRL%20Sloan%20Paper.docx.pdf) matters because it directly targets three real blind spots in additive RAPM:
- context-sensitive action value
- off-ball / distributed defensive credit
- interaction effects and synergy

Its value is not that it replaces the six-factor framework. Its value is that it prices **when** and **with whom** impact happened.

### 4. New lineup / interaction papers say we should model fit separately

[Hypergraph APM](https://arxiv.org/abs/2403.20214) argues that player value and combination value should be estimated jointly, not as an afterthought.

[L-RAPM](https://arxiv.org/abs/2601.15000) argues that lineup ratings need informed priors because lineup samples are extremely sparse, and that the advantage grows as samples shrink.

Takeaway for us:
- Pair and lineup value should not be forced into the main additive player number.
- Fit should be a separate modeled output with its own shrinkage.

## What This Repo Already Has That Public Metrics Usually Do Not

This project already has a stronger mechanical foundation than most public all-in-one metrics:
- six-factor RAPM reconstruction with very high fit
- [TS decomposition](./RAPM_METHODOLOGY.md) through ShotQuality / Contest / FT
- [turnover decomposition](./tov-decomposition.md)
- second-chance and first-chance structure
- rubberband state correction in the RAPM path

That means the core advantage of this repo is not "we can copy EPM." The advantage is:

**we already know where lineup value comes from mechanically.**

The right next move is to add the missing public-metric pieces without throwing away that structure.

## Recommended Metric Architecture

### Layer 1: Factor priors

For each player-season, estimate a prior mean for:
- `oTS`
- `oTOV`
- `oREB`
- `dTS`
- `dTOV`
- `dREB`

These should come from box score and, where available, tracking-derived models.

Critical rule:

**the prior must be factor-specific, and the prior strength must also be factor-specific.**

That matters because our own work already shows box predictability differs dramatically by factor:
- `oREB` is relatively predictable from public stats (`R^2 ~= 0.67`)
- `oTS`, `oTOV`, `dTOV` are moderate (`R^2 ~= 0.55-0.59`)
- `dTS` is much weaker (`R^2 ~= 0.41`)
- `dREB` is weakest (`R^2 ~= 0.18`)

So the right prior policy is:
- stronger prior on `oREB`
- medium prior on `oTS`, `oTOV`, `dTOV`
- weaker prior on `dTS`
- weakest prior on `dREB`

Do not use one scalar prior weight for all six factors.

### Layer 2: Time-decayed six-factor RAPM posterior

Run the existing six-factor RAPM system as the main additive player model, but replace zero-centered shrinkage with factor priors.

This should remain the core public estimate because it has the best mix of:
- interpretability
- stability
- transportability across teams and roles
- mechanical explainability

This layer should continue to carry:
- rubberband adjustment
- multi-year data
- time decay

If forced to choose one public number today, this is still the correct core.

### Layer 3: DRL/Shapley context overlay

Use the event-level DRL/Shapley model as a **context and residual layer**, not the primary leaderboard engine.

It should provide:
- leverage-adjusted value
- off-ball / defensive-presence credit
- action values that vary by state
- value that lives in `RESID`

This layer should answer questions the additive factor model cannot:
- Which players create extra value in high-leverage states?
- Which players gain value through off-ball defense not visible in box priors?
- Which actions are systematically mispriced by fixed linear weights?

Important constraint:

The DRL layer should be shrunk hard before it touches the main public rating. Until it proves itself over large forward tests, it should mostly live as:
- `context_value`
- `defensive_presence`
- `action_value_profile`

### Layer 4: Fit / synergy module

Build pair and lineup value as a separate output family:
- pair synergy
- lineup fit
- team fit

Do not fully fold this into the main player rating early.

Reason:
- synergy is real
- synergy is unstable
- public users confuse fit with talent if both are mixed into one number

The main public scalar should stay close to "portable impact." Fit should be published beside it.

## Recommended Public Outputs

The best public package is not just one leaderboard column. It is one main number plus a small set of interpretable companions.

### Headline columns

- `Impact`: main public all-in-one estimate
- `Off`
- `Def`

### Mechanical decomposition

- `oTS`
- `oTOV`
- `oREB`
- `dTS`
- `dTOV`
- `dREB`

### Overlay columns

- `Context`
- `Def Presence`
- `Synergy`

### Optional detail views

- ShotQuality / Contest / FT splits
- bad-pass vs scoring turnovers
- second-chance / first-chance
- action value profile by class

## How the Main Number Should Be Built

The main number should be built in this order:

1. Estimate factor priors from box/tracking models.
2. Fit time-decayed six-factor RAPM with those priors.
3. Convert factor posteriors into point-value `Off`, `Def`, and `Impact`.
4. Add only a **shrunk residual overlay** from the DRL/Shapley model.
5. Keep synergy mostly separate unless validation shows clear incremental value after shrinkage.

A practical formulation is:

`Impact = Base_Factor_Impact + lambda_ctx * Context_Residual + lambda_fit * Fit_Residual`

where:
- `Base_Factor_Impact` comes from the factor posterior
- `Context_Residual` comes from the DRL/Shapley layer
- `Fit_Residual` comes from pair/lineup modeling
- `lambda_ctx` and `lambda_fit` are learned from forward validation, not chosen by feel

My default expectation is:
- `lambda_ctx` should be modest at first
- `lambda_fit` should be smaller still

## What We Should Not Do

### 1. Do not replace the six-factor core with pure DRL

That would improve elegance and probably hurt public usefulness. The public metric needs stability, interpretation, and trust.

### 2. Do not use one overall box prior

That would flatten archetypes and break the core insight of this repo.

### 3. Do not publish synergy as if it were portable player talent

Fit matters, but fit is not the same thing as independent player quality.

### 4. Do not optimize only one metric

The target is not just lower RMSE. The public metric should sit on a Pareto frontier of:
- forward game prediction
- future player prediction
- year-to-year stability
- decomposition quality
- interpretability

## Immediate Research Program

### Phase 1: Build six-factor priors

Use the existing box-score-on-RAPM work to generate prior means for all six factors.

Deliverables:
- prior models per factor
- prior uncertainty per factor
- prior-only evaluation

### Phase 2: Refit six-factor RAPM with priors

This is the most important next model.

Deliverables:
- factor posteriors
- new public `Impact`, `Off`, `Def`
- forward validation against current no-prior td700 and td700+rubberband

### Phase 3: Keep DRL as an overlay path

Use the new DRL/Shapley implementation to model:
- `Context`
- `Def Presence`
- action values

Deliverables:
- forward tests showing whether DRL residuals add signal beyond factor RAPM
- heavy shrinkage rules for low-sample players and volatile states

### Phase 4: Add fit explicitly

Test:
- pair synergy
- lineup priors
- hypergraph or L-RAPM style interaction outputs

Deliverables:
- separate fit tables
- evidence for whether fit helps prediction enough to partially enter the main public number

## Final Recommendation

If the goal is the **best public impact metric**, the target should be:

**factor-prior, time-decayed, rubberband-aware six-factor RAPM as the base, with DRL/Shapley and synergy used as shrunk overlays rather than replacements.**

That is the best synthesis of:
- what the public metric leaders already do well
- what the newest research adds
- what this repo already does better than almost anyone publicly

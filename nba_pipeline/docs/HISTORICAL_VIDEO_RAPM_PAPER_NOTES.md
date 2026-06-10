# Historical Video RAPM Paper Notes

Source: Justin Jacobs, "Possession-Level Player Impact in the Pre-Play-by-Play NBA Era: A Video-Reconstructed RAPM Database, 1984-1996", arXiv:2605.24056v1, 2026-05-22.

Source URL: https://arxiv.org/html/2605.24056v1

This note extracts the useful statistical and data-construction ideas from the paper for this repo. The paper is not a DRL/Shapley paper. It is a classical RAPM paper for sparse, hand-reconstructed historical stint data. Its value for us is methodological discipline: how to reconstruct historical possession surfaces, choose shrinkage under partial coverage, quantify uncertainty, and validate biased samples.

## Main Takeaway

The paper's core contribution is a full workflow for turning incomplete historical video into a possession-weighted, ridge-regularized player impact estimate:

1. Reconstruct stint-level lineup, points, and possession data directly from video.
2. Avoid box-score possession estimates when the source box score is biased or internally inconsistent.
3. Estimate separated offense and defense RAPM with possession-weighted ridge regression.
4. Scale the ridge penalty by logged-game coverage and validate it against cross-validation.
5. Publish posterior uncertainty intervals and sample-size guidance, not only point estimates.
6. Validate against team-level outcomes, win-loss samples, face validity, and coverage-bias diagnostics.

For our goals, the strongest immediate ideas are the posterior covariance/credible interval layer, coverage-aware alpha selection, formal split-possession sensitivity checks, and validation tables that separate data coverage problems from modeling problems.

## Data Reconstruction Techniques

### Direct Possession Counting

The paper argues that historical RAPM should not depend on box-score possession estimators when the box-score inputs are suspect. Instead, it logs:

- offensive team
- defensive team
- five offensive players
- five defensive players
- offensive possessions in the stint
- defensive possessions in the stint
- offensive points
- defensive points

This maps naturally onto our processed RAPM rows, especially for Gabriel historical source repairs and any future video-assisted validation work.

### Possession Definition

The possession definition follows the standard Oliver-style logic:

- starts when a team gains control
- ends on a score, live/dead turnover, final free throw that ends the sequence, defensive rebound after a miss, or period expiration
- and-1 sequences count as one possession
- technical free throws do not create possessions
- jump balls create a new possession only for the gaining team

Repo relevance: this is consistent with our existing standard possession guardrails around and-1 scoring, admin/technical free throws, missed final free throws, and period-end inheritance. It reinforces that possession logic should be audited from event mechanics first, not from box-score totals.

### Split-Possession Protocol

The paper handles substitutions that occur inside a live possession sequence, most often dead-ball substitutions during free throws. Its baseline convention:

- count the split possession for both adjacent stints during logging
- correct game totals by subtracting duplicate split counts
- apportion the split possession fractionally in processed stint data, usually 0.5 to the closing stint and 0.5 to the opening stint
- test sensitivity against 1.0/0.0 and 0.0/1.0 alternatives

The paper reports that player RAPM estimates were essentially unchanged across the three split-possession conventions. We should copy the pattern, not necessarily the exact conclusion: any historical or video-ingest surface with split possessions should include a sensitivity harness that proves the convention is harmless.

### Quality Control Pipeline

The paper's QC sequence is useful and operational:

- score reconciliation: stint points must sum to the official final score
- lineup consistency: substitution application must produce plausible active lineups
- minutes reconciliation: reconstructed player minutes should match official minutes within a tolerance
- possession plausibility: direct counts are compared to Oliver estimates, but only as a review trigger
- exclusion criteria: unresolved score, lineup, or footage-quality failures are excluded

Repo application: this is a good template for hardening Gabriel historical validation and any friend-PBP/video-assisted ingest. In our repo, the closest analogs are O/D overlap checks, incomplete-lineup counts, same-side duplicate-slot checks, score/points reconciliation, and possession-total drift reports.

### Sampling Bias Accounting

The paper is explicit that footage availability is not a random sample. It names three selection problems:

- high-profile game bias
- temporal clustering within a season
- opponent-distribution imbalance

This is highly relevant to historical RAPM. Any partial-coverage historical estimate should publish coverage metadata by team, opponent, date segment, and player possession exposure. Without that, posterior intervals understate the true uncertainty because they assume the observed stints are representative.

## Estimation Techniques

### Zero-Possession Filtering

Rows with offensive possessions below one are excluded from the regression. They can exist around lineup changes before a complete possession occurs. This is a simple but important preprocessing step for stint-level reconstruction.

Repo relevance: for event-level parquets we usually preserve zero-possession admin scoring rows when they are real scoring events. For stint-level reconstructed RAPM, zero-possession stints are not regression observations unless they carry an explicit target we want to model.

### Separated Offense/Defense Design Matrix

The paper uses a separated design:

- column 0: intercept
- columns 1..P: offensive player indicators, encoded as +1
- columns P+1..2P: defensive player indicators, encoded as -1

The response is offensive points per 100 possessions:

```text
y_i = 100 * Oscore_i / Oposs_i
```

The prediction is:

```text
yhat_i = beta_0 + sum(offensive beta_j) - sum(defensive beta_j)
```

Under the reporting convention, positive offense is good and positive defense is also good because the defensive block enters with a negative sign in the design.

Repo relevance: this aligns with our sign-flipped defensive display convention for public factors. It also reinforces a cleaner route for metrics that are naturally "offense generated vs defense prevented" rather than net-margin-only targets.

### Possession-Weighted Ridge Regression

The estimator minimizes:

```text
sum_i w_i * (y_i - x_i beta)^2 + lambda * ||beta||^2
```

with:

```text
beta_hat = (X' W X + lambda I)^(-1) X' W y
```

The weights are stint offensive possessions. This makes long stints more informative than one- or two-possession stints.

Repo relevance: our possession-level rows are usually one possession each, so this is less important for modern PBP RAPM. It matters for aggregated stints, reconstructed historical data, and any future compression of event rows into stint summaries.

### Heteroskedastic Error Model

The paper motivates the possession weights with:

```text
epsilon_i | w_i ~ Normal(0, sigma^2 / w_i)
```

Longer stints have lower variance in points per 100 possessions. This is the statistical reason for possession-weighted regression, not just a heuristic.

### Bayesian Ridge Interpretation

The ridge solution is interpreted as a Gaussian posterior mean:

```text
beta_k ~ Normal(0, tau^2)
lambda = sigma^2 / tau^2
```

This framing is useful for two reasons:

- it justifies shrinkage toward league average under limited exposure
- it provides a posterior covariance matrix for uncertainty intervals

Repo relevance: this is a clean way to explain alpha choices and uncertainty in public RAPM DECOMP work. It also points toward informative priors for sparse historical or WNBA surfaces.

### Coverage-Scaled Lambda

The paper's production lambda rule is:

```text
lambda = (games_logged / games_season) * 5000
```

The stated intuition is that the full-season penalty approaches the standard full-season RAPM scale, while smaller partial samples get a coverage-adjusted penalty. The paper compares this rule to 5-fold RidgeCV over a log-spaced grid and reports close agreement for well-observed seasons when the correct season schedule denominator is used.

Important caveat: their appendix notes the CV comparison used an unweighted objective while the production model is possession-weighted. The proper refinement is weighted CV with possession weights passed into fitting.

Repo application:

- build a weighted grouped-CV alpha harness for historical windows and sparse WNBA windows
- report alpha alongside coverage and schedule denominator
- avoid treating one alpha as universal across materially different row granularities
- use grouped splits by game to avoid possession/stint leakage

### Mean-Centering

After fitting, offensive and defensive coefficients are centered separately:

```text
ORAPM_j = beta_off_j - mean(beta_off)
DRAPM_j = beta_def_j - mean(beta_def)
RAPM_j = ORAPM_j + DRAPM_j
```

The intercept is not centered. This makes player values interpretable relative to league average rather than relative to the arbitrary ridge prior zero.

Repo relevance: this matches our need to distinguish solver coefficients, display columns, intercepts, and public centered values. It is especially important when adding season fixed effects, age offsets, or fixed-season baselines.

### Multi-Season Pooling

The paper treats each player-season as a distinct entity in pooled multi-season runs, then computes player aggregate RAPM as a possession-weighted average over seasons.

Repo relevance: this is conservative and avoids assuming player stability. Our rolling windows often use player IDs across seasons; for historical exploration, a player-season parameterization could be a useful audit variant when aging, team role, or incomplete coverage creates drift.

## Uncertainty Techniques

### Residual Variance Estimate

The paper estimates residual variance from the weighted residual sum of squares with a degrees-of-freedom correction for the effective observation count and number of parameters.

This is a reminder that alpha tuning is not the only uncertainty control. We should persist residual variance summaries for major public RAPM outputs, especially sparse surfaces.

### Posterior Covariance Matrix

The posterior covariance is:

```text
Sigma_beta = sigma_hat^2 * (X' W X + lambda I)^(-1)
```

This gives uncertainty for every player coefficient and every linear combination of coefficients. For total RAPM, the exact variance includes:

```text
Var(off_j + def_j) =
  Var(off_j) + Var(def_j) + 2 * Cov(off_j, def_j)
```

The paper's implementation currently omits the covariance cross-term in the reported interval calculation and flags that as a refinement. We should include the cross-term if we implement this.

### Credible Intervals

The paper reports 95% intervals from posterior standard errors. That is valuable because it prevents over-reading sparse estimates.

Repo application:

- add optional uncertainty outputs for `rapm.py`, at least for classical RAPM and public factor families
- include intervals for component factors, not only total RAPM
- include a flag for prior-dominated players or components
- include exposure and effective exposure beside every interval

### Interval Width Decomposition

The paper decomposes interval width into:

- residual variance
- possession exposure
- ridge penalty
- collinearity/design covariance

This is useful because it tells the operator why an estimate is uncertain. A player can have many possessions but still be poorly identified if those possessions occur in stable, non-diverse lineup contexts.

### Sample Size Guidance

The paper derives required possession counts for target credible interval half-widths. The exact numbers depend on residual variance and lambda, but the technique is reusable:

1. choose target interval width
2. use realized residual variance and lambda
3. solve for required possession exposure
4. publish which players/components are above or below the threshold

Repo application: for Decomp RAPM, this could be more useful than a single `possessions` cutoff. Each component has different variance and sparsity, so each component should have its own reliability threshold.

## Validation Techniques

### Team-Level Point Differential Prediction

The paper validates player RAPM by aggregating player estimates into a team predicted rating and regressing actual team net rating on that predicted rating. A calibrated model should have slope near 1, intercept near 0, and high R^2.

Repo application: use this as a standard validation table for new weighted-factor families. For each public bundle, aggregate player estimates to team-season predictions and compare to observed team net rating or component-specific team outcomes.

### Win-Loss Prediction Under Partial Coverage

The paper compares:

- observed record in logged games
- maximum likelihood full-season projection from logged win percentage
- Bayesian shrinkage toward .500 using a Beta prior

This is not directly a RAPM validation, but it diagnoses whether the logged sample behaves like the full season.

Repo application: for historical partial coverage, publish sample-vs-full team outcome diagnostics. This tells us whether model errors are caused by the ridge model or by biased source coverage.

### Face Validity With Room For Surprise

The paper uses historical consensus top-player lists as a sanity check, while explicitly allowing RAPM to disagree with box-score reputation.

Repo application: this is the right posture for public RAPM DECOMP and WNBA outputs. Face validity is a bug detector, not a requirement that the model reproduce conventional rankings.

### Coverage-Bias Diagnostics

The paper shows aggregate MLE win projection errors are related to coverage rate and game-importance bias. The lesson is to validate coverage before validating the model.

Repo application:

- team coverage by season
- opponent coverage by team
- home/road coverage
- month/season-segment coverage
- player exposure distribution
- leverage/collinearity diagnostics

## What Could Be Useful For Our Goals

### 1. Add Posterior Uncertainty To Classical RAPM Outputs

This is the biggest practical idea. `rapm.py` could optionally write:

- residual variance
- posterior covariance diagonal
- off/def/total standard errors
- 95% intervals
- interval half-width
- effective exposure
- prior-dominated flag

For public outputs, this would make Decomp RAPM more defensible. It would also help decide which component estimates should be shown, hidden, or marked exploratory.

### 2. Build Coverage-Aware Alpha Selection

Our current `--cv-alpha` is already useful. The paper suggests a second lens: alpha should be interpretable relative to coverage, schedule size, and row granularity.

Recommended exploration:

- add grouped weighted CV by `gameid`
- log row count, possession count, game coverage, and schedule denominator beside alpha
- compare CV alpha to simple coverage-scaled formulas per metric family
- avoid using the formula blindly for event-level modern PBP rows, because the paper's rule is calibrated to aggregated historical stints

### 3. Improve Historical Ingest Validation

The paper's QC sequence is a strong checklist for Gabriel and friend-PBP work. We should add or keep validation outputs for:

- final score reconciliation
- player minutes reconciliation when reliable source minutes exist
- lineup transition validity
- possession total plausibility
- O/D overlap and duplicate-slot checks
- partial-lineup rates
- exclusion/repair reason counts

### 4. Add Split-Possession Sensitivity Tests

For historical or manually reconstructed stints, build a harness that reruns the same RAPM surface under:

- 0.5/0.5 split
- closing-stint allocation
- opening-stint allocation

Then report max player delta, rank correlation, and affected players. If deltas are negligible, the convention becomes defensible.

### 5. Publish Coverage-Bias Diagnostics With Historical RAPM

For pre-2014 and especially pre-PBP surfaces, public outputs should not only have player ranks. They should also expose:

- game coverage by team-season
- opponent mix
- date clustering
- home/road split
- player exposure distribution
- players below inferential thresholds

This matters for our historical windows because source availability is not a probability sample.

### 6. Use Informative Priors Carefully

The paper suggests a Statistical Plus-Minus style prior from box-score metrics. For us this is useful but risky:

- useful for sparse historical, WNBA, and low-possession component estimates
- risky because historical box-score inputs can be biased
- better if priors are transparent, optional, and audited against no-prior results

Potential prior sources in this repo's ecosystem include DPM history, box-score SPM, previous-season RAPM, or component-specific priors. Any public prior should be documented as a prior, not silently blended.

### 7. Add Reliability Gates Per Component

Decomp RAPM components do not all have the same signal density. Rim FG%, midrange FG%, turnover value, second chance, and FT severity have different variance and opportunity counts.

The paper's sample-size framework suggests a better public gate:

- compute component-specific residual variance
- compute component-specific interval widths
- publish or rank only estimates below a max interval threshold
- keep sparse estimates available as exploratory artifacts

### 8. Separate Reconstruction Error From Model Error

The paper's strongest operational discipline is that reconstruction, sampling, and model estimation are separate uncertainty layers.

For this repo:

- reconstruction errors: lineup side, duplicate events, missing IDs, score mismatch
- sampling errors: partial historical footage, non-random team/opponent coverage
- model errors: alpha, collinearity, target design, priors

Future docs and validation should keep those separated rather than treating every surprising player result as a solver problem.

## What Not To Copy Blindly

- The paper's `lambda = coverage * 5000` rule is calibrated to historical stint-level RAPM, not necessarily our modern possession-level, multi-metric, timedecay, fixed-effect solves.
- The paper's CV comparison is unweighted; our alpha validation should be possession-weighted or row-weighted according to the solved target.
- The paper uses player-season coefficients in pooled runs; our player-ID rolling windows are a different modeling choice and should be compared, not replaced by default.
- The paper's credible interval implementation omits the off/def covariance cross-term. If we implement intervals, include the cross-term.
- The paper validates against team net rating and win-loss outcomes; component RAPM needs component-specific validation targets too.

## Candidate Exploration Tickets

1. Add `--uncertainty` to `rapm.py` for posterior SE/CI output on single-metric solves.
2. Add weighted grouped alpha CV with `sample_weight` and `gameid` folds.
3. Add `scripts/audit_historical_coverage_bias.py` for team/opponent/month/home-road coverage tables.
4. Add split-possession sensitivity harness for reconstructed stint data.
5. Add component-specific reliability thresholds to weighted-factor display bundle builds.
6. Add team-level validation reports for Decomp RAPM bundles.
7. Add optional informative-prior mode that accepts prior means/variances and writes a prior audit file.

## Bottom Line

The paper is useful less because of its player results and more because it treats historical RAPM as an estimation system with auditable data construction, shrinkage, uncertainty, and validation. The best next repo-level exploration is to add uncertainty and coverage diagnostics around our existing RAPM/Decomp RAPM outputs, then use those diagnostics to decide which component estimates are publication-grade.

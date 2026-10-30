# 21st Century RAPM

**Age-adjusted career O-RAPM and D-RAPM, 1997-2026**

## Status

This is the research parent for one career offensive and defensive RAPM
coefficient per player over 1997-2026, normalized to age 27. It is the direct
Engelmann-comparable surface. It is not yet the historical net-DECOMP model.

`21st Century RAPM` is the reader-facing name. The date subtitle must remain
visible because the underlying possession sample begins in 1997, not 2000.

The accepted curve substrate is the joint possession-level age model described
below. The earlier near-OLS annual-RAPM change curve is rejected: it implied a
`+7.63` net age translation for John Stockton and was worse than a zero-change
age forecast on the annual diagnostic.

## Estimand

For each player, estimate:

```text
O-RAPM at age 27
D-RAPM at age 27
Net RAPM at age 27 = O-RAPM + D-RAPM
```

Reader-facing defense is always positive-good. `rapm.py` stores its raw defense
coefficient as points allowed, so the export flips defense exactly once.

This is an age-normalized career rate, not career value, a peak-season rating,
or a claim about the player's observed impact at every age.

## Possession model

- Seasons: 1997 through 2026.
- Sample: regular season and playoffs, `7,157,906` possessions.
- Target: pure actual point differential.
- Player penalties: offense `2000`, defense `3000`.
- Game state: validated smooth clock-aware rubberband surface, `alpha_rb=100`.
- Season environment: each season's raw regular-season scoring mean is a fixed
  offset and is used for both RS and PS rows from that season.
- Reference age: 27.

Fixed season baselines are essential here. With player, age, and freely fitted
season effects in one model, the linear age trend is an age-period-cohort
identity: a linear age tilt can be exchanged for a calendar trend plus player
cohort constants. Omitting age 27 resolves the category reference but does not
resolve that linear tilt. Subtracting observed season scoring environments and
not fitting season dummies makes the identifying assumption explicit.

## Curve construction

1. Fit one joint possession regression with player O/D coefficients, separate
   offense and defense integer-age effects, fixed season baselines, and the
   rubberband surface. Age 27 is omitted.
2. Use ages 20-38 for curve fitting. Ages 18-19 have too little support and are
   excluded.
3. Fit separate slot-weighted cubic curves to the raw offense and defense age
   bins. Convert defense from raw points-allowed sign to positive-good.
4. Anchor both curves exactly at age 27.
5. Freeze the table, apply `curve(age) - curve(27)` as a known lineup offset,
   and refit only the player and rubberband coefficients. Ages below 20 and
   above 38 use endpoint clipping.

Selected positive-good curve values, in points per 100 relative to age 27:

| Age | O effect | D effect | Net effect |
|---:|---:|---:|---:|
| 20 | -2.095 | -1.077 | -3.172 |
| 21 | -1.486 | -0.828 | -2.315 |
| 23 | -0.601 | -0.430 | -1.031 |
| 25 | -0.121 | -0.157 | -0.278 |
| 27 | 0.000 | 0.000 | 0.000 |
| 30 | -0.394 | +0.036 | -0.358 |
| 32 | -0.973 | -0.061 | -1.034 |
| 34 | -1.753 | -0.245 | -1.997 |
| 36 | -2.687 | -0.507 | -3.194 |
| 38+ | -3.730 | -0.839 | -4.569 |

The cubic's slot-weighted fit is `R²=0.9782` on offense and `0.8393` on defense.

## Validation and rejected alternative

The annual diagnostic uses `11,267` consecutive player-season transitions for
`2,135` players. Its target panel uses the prediction-oriented annual penalties
`2000/3000`; pair weights are harmonic consecutive-season possessions.

| Curve candidate | O MSE gain vs zero | D MSE gain vs zero | Net MSE gain vs zero |
|---|---:|---:|---:|
| Predictive annual curve | +0.0485 | +0.0071 | +0.0912 |
| Joint possession cubic | +0.0379 | +0.0042 | +0.0700 |
| Rejected near-OLS annual curve | -0.0095 | -0.0028 | -0.0140 |

Positive is better. The selected joint curve improves all three targets. The
rejected steep curve loses to predicting no age change. This diagnostic is not
an independent possession holdout; it is a mechanism and scale gate.

Additional receipts:

- Frozen cubic versus the unsmoothed joint age-bin board: player net correlation
  `0.9952`, possession-weighted MAE `0.190`.
- Frozen cubic versus the exact no-age fixed-season control: net correlation
  `0.9728`, possession-weighted MAE `0.511`.
- Frozen cubic versus the prior published historical age-polynomial board: net
  correlation `0.9698`, possession-weighted MAE `0.449`.

## Negative-LAKER fringe prior

The prior-informed research board changes only the prior center for players
with very little NBA evidence. It does not change the age curve, target,
rubberband surface, possession sample, or `2000/3000` player penalties.

The selected rule is:

1. Use regular-season LAKER rows from 1997-2026 and exact NBA player IDs.
   Exclude provider `TOT` rows and minutes-weight the team stints.
2. Translate LAKER offense and positive-good defense to age 27 by subtracting
   the same frozen age effects used by the possession model.
3. A player is eligible only when age-27 career LAKER net is negative and
   career regular-season minutes are at most `1000`. Players outside this rule
   retain the ordinary zero prior.
4. Stabilize each eligible O/D center by
   `career_minutes / (career_minutes + 500)`. The scale is `1.0`; this
   reliability term is already the shrinkage of the external estimate.
5. Flip the stabilized defense prior exactly once when it enters `rapm.py`,
   whose raw defense coefficient is points allowed.

The `500`-minute stabilization constant was selected on `1,253`
consecutive-season negative-LAKER transitions totaling `728,339` next-season
minutes. The selection minimized minutes-weighted next-season offense MSE plus
defense MSE. Its combined MSE was `4.6266`, a `1.2718` gain over predicting
zero. Raw LAKER (`K=0`) lost to zero because tiny samples include extreme
values, so an unstabilized LAKER prior is rejected.

The `1000`-minute cutoff is a deliberately simple definition of a fringe
career, not a leaderboard-tuned hyperparameter. Sensitivities at `500` and
`2000` minutes preserve the top of the board, but `2000` begins moving players
with at least 10,000 model possessions more than necessary. The selected prior
applies to `673` of `2,384` LAKER-covered players.

Relative to the zero-prior parent, the selected solve has:

| Group | Net mean change | Net MAE change | Maximum absolute change |
|---|---:|---:|---:|
| Eligible LAKER players | -0.785 | 0.793 | 3.016 |
| Players with 10,000+ model possessions | +0.018 | 0.029 | 0.247 |
| Baseline top 100 | — | — | correlation `0.99989` |

This is a true prior-informed RAPM, not pure zero-centered RAPM. LAKER contains
plus-minus information from the same historical period, so the chronological
LAKER stability check is not an independent possession-level promotion test.
Keep the zero-prior parent as the clean methodological control and label the
negative-LAKER board explicitly until it clears that gate.

## Adaptive young-player precision challenger

A single offense alpha and defense alpha do not imply equal posterior
certainty. Ridge already shrinks players with fewer possessions more, but raw
possession count overstates the independent information in one season and the
fixed age curve otherwise applies a full age-27 translation with no additional
uncertainty. This is the Dylan Harper failure mode: the negative-LAKER rule did
nothing because his LAKER was positive and his `1558` RS minutes exceeded its
cutoff, while the fixed age curve added `+2.915` net points to age 27.

The adaptive challenger separates prior mean from prior precision:

- Prior mean is unchanged: stabilized negative LAKER when eligible, zero
  otherwise.
- A player receives extra precision only with exactly one observed season,
  age `<=22.5`, and no observation within four years of age 27.
- Development first-to-second-season transitions cover feature seasons
  1998-2021 (`639` players). Report-only stress uses feature seasons 2022-2025
  (`167` players).
- The annual calibration converts both seasons to age 27 and approximates the
  incremental ridge attenuation by
  `(possessions + alpha) / (possessions + multiplier * alpha)`.
- Development selects offense multiplier `4` and defense multiplier `3` from
  `1, 1.5, 2, 3, 4, 6, 8, 12`. The exact career solve therefore uses effective
  penalties `8000/9000` for the `98` eligible players and `2000/3000` for
  everyone else.

Relative to multiplier 1, the selected rule reduces next-season age-27 annual
RAPM MSE by `21.9%` on offense and `21.0%` on defense in development. The
later stress cohort also improves: `40.7%` offense and `12.9%` defense. The
multiplier was not selected from the final historical leaderboard.

In the exact 7.16M-possession solve, Dylan Harper moves from `+6.58` to `+2.56`
at age 27: offense `+2.93 -> +1.02`, positive-good defense `+3.65 -> +1.55`.
The rule is symmetric: implausibly negative one-season estimates also move
toward their prior centers. Noneligible players move `0.013` net points on
average because the joint solve reallocates some lineup credit; this transfer
is inherent to true prior-centered RAPM rather than a post-solve haircut.

This remains a research challenger. The annual attenuation validation is a
forward coefficient-stability proxy, not an exact future-possession test of
the full 30-year solve. Preserve the fixed-alpha and negative-LAKER-only boards
as controls until a date-correct possession gate is completed.

## Scale and extrapolation audit

The final board must show both the normalized rating and the age translation.
For players not observed near 27, the normalized rating is an extrapolation.

Examples:

| Player | O age 27 | D age 27 | Net age 27 | Mean observed age | Curve translation to 27 | Net at observed-age mix |
|---|---:|---:|---:|---:|---:|---:|
| John Stockton | 6.74 | 3.25 | 9.98 | 38.19 | +4.04 | 5.95 |
| Michael Jordan | 7.12 | 2.20 | 9.33 | 37.02 | +3.41 | 5.92 |
| Kevin Garnett | 2.88 | 6.67 | 9.55 | 28.86 | +1.28 | 8.27 |
| LeBron James | 6.99 | 2.39 | 9.38 | 29.67 | +1.92 | 7.46 |
| Nikola Jokic | 6.95 | 2.09 | 9.04 | 26.49 | +0.57 | 8.48 |

The curve translation is the exposure-weighted frozen age offset, not the
difference between two independently fit leaderboards. In the exact no-age
control, Stockton is `+6.63`; the final age-27 coefficient is `+9.98`, a
`+3.36` coefficient change after the whole model refits.

Audit labels:

- `near_reference`: the player is observed within one year of 27.
- `moderate_extrapolation`: nearest observed age is more than one and at most
  four years from 27.
- `long_extrapolation`: nearest observed age is more than four years from 27.
- `missing_age_history`: no usable age audit row.

## Main artifacts

Generated artifacts stay out of git:

```text
nba_pipeline/results/longitudinal_orapm_drapm_age/joint_possession_1997_2026/
  fixed_rs_season_baselines.csv
  age_curve.csv
  age_curve_receipt.json
  annual_change_candidate_validation.csv
  annual_change_candidate_validation_receipt.json

nba_pipeline/results/rapm_97_26_all_pure_rb_fse_agevalues_a2000_3000_results.csv
nba_pipeline/results/rapm_97_26_all_pure_rb_fse_agevalues_a2000_3000_age27_audit.csv

nba_pipeline/results/longitudinal_orapm_drapm_age/negative_laker_priors/
  negative_laker_m1000_k500_s1_off_prior.csv
  negative_laker_m1000_k500_s1_def_prior.csv
  negative_laker_m1000_k500_s1_audit.csv
  negative_laker_stabilization_validation.csv

nba_pipeline/results/rapm_97_26_all_pure_rb_fse_agevalues_a2000_3000_offdefprior_neglaker_m1000_k500_s1_results.csv
nba_pipeline/results/rapm_97_26_all_pure_rb_fse_agevalues_a2000_3000_offdefprior_neglaker_m1000_k500_s1_age27_audit.csv

nba_pipeline/results/longitudinal_orapm_drapm_age/adaptive_young_penalties/
  adaptive_penalties_young1s_o4_d3.csv
  adaptive_penalties_young1s_o4_d3_audit.csv
  adaptive_young_penalty_validation.csv

nba_pipeline/results/rapm_97_26_all_pure_rb_fse_agevalues_a2000_3000_offdefprior_neglaker_m1000_k500_s1_young1s_o4_d3_results.csv
nba_pipeline/results/rapm_97_26_all_pure_rb_fse_agevalues_a2000_3000_offdefprior_neglaker_m1000_k500_s1_young1s_o4_d3_age27_audit.csv
```

The final results CSV uses raw solver defense. Use the audit CSV for the
positive-good reader surface.

## Reproduction

Fit and freeze the curve:

```bash
python nba_pipeline/scripts/rapm.py RAPM 97 26 ALL \
  --pure --rubberband --age-dummies \
  --fixed-season-baselines path/to/fixed_rs_season_baselines.csv \
  --darko-history nba_pipeline/.cache/teammate_help/dpm_history.csv \
  --off-alpha 2000 --def-alpha 3000 --cores 14

python nba_pipeline/scripts/fit_joint_possession_age_curve.py \
  --age-effects nba_pipeline/results/rapm_97_26_all_pure_rb_fse_agefe_a2000_3000_age_effects.csv \
  --output nba_pipeline/results/longitudinal_orapm_drapm_age/joint_possession_1997_2026/age_curve.csv
```

Fit the frozen career parent:

```bash
python nba_pipeline/scripts/rapm.py RAPM 97 26 ALL \
  --pure --rubberband \
  --fixed-season-baselines path/to/fixed_rs_season_baselines.csv \
  --age-curve-values path/to/age_curve.csv \
  --age-curve-reference-age 27 \
  --darko-history nba_pipeline/.cache/teammate_help/dpm_history.csv \
  --off-alpha 2000 --def-alpha 3000 --cores 14
```

Build and apply adaptive young-player precision:

```bash
python nba_pipeline/scripts/build_adaptive_young_player_penalties.py \
  --annual-panel path/to/annual_orapm_drapm_panel.csv \
  --age-curve path/to/age_curve.csv \
  --output-dir path/to/adaptive_young_penalties \
  --development-end-year 2021 --off-alpha 2000 --def-alpha 3000

python nba_pipeline/scripts/rapm.py RAPM 97 26 ALL \
  --pure --rubberband \
  --fixed-season-baselines path/to/fixed_rs_season_baselines.csv \
  --age-curve-values path/to/age_curve.csv --age-curve-reference-age 27 \
  --darko-history nba_pipeline/.cache/teammate_help/dpm_history.csv \
  --off-prior path/to/negative_laker_m1000_k500_s1_off_prior.csv \
  --def-prior path/to/negative_laker_m1000_k500_s1_def_prior.csv \
  --prior-label neglaker_m1000_k500_s1 \
  --player-penalty-multipliers path/to/adaptive_penalties_young1s_o4_d3.csv \
  --penalty-label young1s_o4_d3 \
  --off-alpha 2000 --def-alpha 3000 --cores 14
```

Build and fit the negative-LAKER challenger:

```bash
python nba_pipeline/scripts/build_negative_laker_priors.py \
  --source-csv nba_pipeline/cache/laker_history/laker_history_rs_1997_2026.csv \
  --age-curve path/to/age_curve.csv \
  --output-dir nba_pipeline/results/longitudinal_orapm_drapm_age/negative_laker_priors \
  --minute-threshold 1000 --stabilization-minutes 500 --prior-scale 1

python nba_pipeline/scripts/rapm.py RAPM 97 26 ALL \
  --pure --rubberband \
  --fixed-season-baselines path/to/fixed_rs_season_baselines.csv \
  --age-curve-values path/to/age_curve.csv --age-curve-reference-age 27 \
  --darko-history nba_pipeline/.cache/teammate_help/dpm_history.csv \
  --off-prior path/to/negative_laker_m1000_k500_s1_off_prior.csv \
  --def-prior path/to/negative_laker_m1000_k500_s1_def_prior.csv \
  --prior-label neglaker_m1000_k500_s1 \
  --off-alpha 2000 --def-alpha 3000 --cores 14
```

Before a public promotion, add a chronological possession-held-out comparison
with player availability held fixed. Do not retune the curve to make historical
leaderboard names look right.

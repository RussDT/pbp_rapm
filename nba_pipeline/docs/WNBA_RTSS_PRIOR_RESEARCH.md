# WNBA RTSS Component-Prior Research

## Status

Research-only. The frozen live WNBA rating remains the box-prior
`5000/40000/1.5` specification. Nothing in this experiment is wired into the
2026 daily update.

## Question

Can 2023-2025 RTSS tracking mechanisms improve the factor-specific WNBA priors
beyond the current public box-only models?

This is a prior experiment, not a lineup-repair or lineup-alpha experiment.
The annual component RAPMs remain the training targets and the current-season
lineup likelihood remains unchanged.

## Feature build

Run:

```bash
.venv/bin/python wnba_test/build_wnba_rtss_prior_features.py --workers 4
```

The builder:

- reads canonical 2D `basketball-markings.json` facts for all 761 games;
- maps provider UUIDs only through audited `exact` or
  `high_confidence_consensus` event-ID crosswalk rows;
- aggregates opportunity-denominated player-season mechanisms;
- shrinks rates and averages toward the season league mean; and
- writes an ignored research surface under
  `wnba_test/Results/rtss_prior_research/`.

The current output has 497 mapped player-seasons and 75 stabilized features:
152 in 2023, 161 in 2024, and 184 in 2025.

The 75th feature is exact on-ball time percentage: RTSS touch time divided by
the offensive chance floor time reconstructed from `chance_players`. It was
added for the separate shooter/passer/presence role experiment and does not
change the previously executed 24-component screening result.

Feature families are deliberately compact:

- MID/3 shot context and shooter adjustment;
- touch creation and pressure;
- passing and advantage creation;
- drives, screens, and handoffs;
- matchups, rotations, closeouts, and contests; and
- offensive/defensive rebound opportunities.

Per the project semantic contract, `qSQ` is league-average-shooter shot
quality, while `qSP` includes the shooter. The shooter candidate is therefore
`qSP - qSQ`, converted to expected points. These fields are used only for MID
and THREE; RIM is excluded.

## Validation design

Run:

```bash
.venv/bin/python wnba_test/evaluate_wnba_rtss_priors.py
```

For each of 12 components on offense and defense:

1. fit on 2023;
2. select Ridge alpha and the tracking lane on 2024;
3. require the selected challenger to beat paired box-only 2024 RMSE;
4. refit on 2023-2024; and
5. open the selected challenger once on 2025.

Every box/tracking comparison uses the same mapped players, at least 100 RTSS
chances, and at least 250 target possessions. Each 2025 component-side test has
151 players. The test compares box-only, box plus `q` fields, box plus
mechanisms, and box plus both, but only the best 2024 challenger is evaluated
as the primary 2025 candidate.

This is a component-target screening test. It is not yet the final
future-possession component-sum test required for live promotion.

### Sample-size limitation

The 2024 development fold necessarily fits the RTSS addition on 2023 alone
because only three tracking seasons are available and 2025 was reserved for
confirmation. That is about 130 target-qualified player-seasons, not 130
independent seasons, and is insufficient for production feature selection.
The final 2025 challenger is refit on 2023-2024, but the feature-family and
alpha decision is still based on the underpowered one-season development fit.
Interpret every result below as screening evidence only.

The box backbone is less constrained: box/component targets exist for
2022-2025 and should use every earlier season in each rolling fold. Future
tracking work should estimate RTSS skills with multi-year, stat-specific
stabilization and add only a compact residual correction to that better-trained
box prior.

The shot prior remains component-specific on both offense and defense:

- RIM frequency;
- RIM within-zone finishing;
- MID frequency;
- MID within-zone finishing;
- THREE frequency; and
- THREE within-zone finishing.

RTSS `qSQ` and `qSP-qSQ` are eligible only for MID/THREE. RIM priors should use
rim attempts/finishing plus drive, touch, passing, and screen mechanisms rather
than provider shot quality.

## Results

Fourteen of 24 component-side challengers beat box-only in 2024 and were
eligible for confirmation. On 2025:

- 9 of 14 improved both RMSE and correlation;
- 9 of 14 improved RMSE;
- 10 of 14 improved correlation;
- eligible challengers reduced mean paired RMSE by `0.00492`; and
- applying the development gate across all 24 slots reduced mean RMSE by
  `0.00287`.

The cleanest confirmed gains were:

| Component | Side | Box RMSE | RTSS RMSE | Relative gain | Correlation change |
|---|---:|---:|---:|---:|---:|
| THREE frequency | defense | 0.02837 | 0.02635 | 7.09% | +0.197 |
| FT frequency | defense | 0.36664 | 0.34405 | 6.16% | +0.126 |
| FT total | defense | 0.36748 | 0.34532 | 6.03% | +0.141 |
| Second chance | offense | 0.80608 | 0.77090 | 4.36% | +0.100 |
| RIM frequency | offense | 0.15597 | 0.15272 | 2.08% | +0.024 |

Smaller two-metric confirmations occurred for defensive RIM FG, defensive MID
FG, offensive first-chance completion, and defensive turnover value.

Five development-eligible candidates failed 2025 confirmation: defensive MID
frequency, offensive MID FG, offensive THREE FG, offensive turnover value, and
defensive bad-pass turnover value.

The defensive evidence is materially better than the offensive evidence:
six of eight development-eligible defensive challengers improved RMSE in 2025,
versus three of six on offense.

## What to trust

The best current feature-family candidates are:

1. defensive closeout/contest/rotation and drive-help activity;
2. offensive rebound opportunity and crash context;
3. offensive drive/touch creation for RIM frequency; and
4. passing/touch structure as a cautious first-chance-completion supplement.

Do not interpret individual Ridge coefficients causally. Several mechanisms
are correlated, especially touch/pass features and crash/rebound-opportunity
features. The confirmation supports the family as an incremental prior input,
not a clean causal credit split.

## qSQ and qSP finding

`qSQ` is stable across seasons, but the three-point `qSP - qSQ` league mean
changes from about `-0.030` expected points in 2023 to `+0.027` in 2024 and
`+0.021` in 2025. That is consistent with a provider calibration or model
vintage shift and makes raw cross-season levels unsafe.

The two `q`-only challengers that passed 2024 development did not clear 2025
on both metrics. `qSQ`/`qSP` therefore do not earn a standalone prior
promotion. Before another test:

- verify the provider timing/model-vintage contract;
- transform shooter adjustment to a season-relative skill estimate;
- freeze the transformation without consulting the next holdout; and
- retest on new 2026 RTSS data or another genuinely unspent future season.

## Three-season development cross-validation

After the 2025 holdout was opened, all three RTSS seasons were reclassified as
development data and evaluated with symmetric leave-one-season-out
cross-validation:

```bash
.venv/bin/python wnba_test/cross_validate_wnba_rtss_priors.py
```

This corrected the earlier weak comparator:

- the box backbone uses every non-held-out 2022-2025 box/component season;
- box predictions for RTSS training rows are themselves cross-fitted;
- RTSS predicts only the residual left by box;
- the two non-held-out RTSS seasons tune and fit the residual Ridge model; and
- all tracking fields are exposure-weighted and standardized within season.

The pooled out-of-fold surface has 412 player-seasons: 129 in 2023, 132 in
2024, and 151 in 2025. The 2023 and 2024 folds use future seasons, so this is
model-development CV rather than an historical as-of simulation.

### Creation-role semantics

The rich feed confirms a useful distinction unavailable in the aggregate PBP
Stats box surface:

- `shots.assistOpp` is populated on all 111,956 shots and identifies a
  potential-assisted attempt on both makes and misses;
- `shots.assisted` is mostly a made-shot outcome and is not used to label
  creation role;
- self-created means `not shots.assistOpp`;
- `passes.assistOpp` identifies a potential assist created; and
- the player-season surface includes potential assists created and received,
  zone-specific potential-assisted shares, self-created attempt rates, and
  role-split shooting/context estimates.

The audit finds 56,899 potential-assisted shots, including 32,672 misses, and
56,750 pass-level potential assists.

### Cross-validation result

The best compact tracking lane lowers pooled RMSE and raises correlation for
15 of 24 component-sides. The largest gains are:

| Component | Side | RMSE change | R2 change | Correlation change | Winning folds |
|---|---:|---:|---:|---:|---:|
| THREE frequency | defense | -5.81% | +0.108 | +0.181 | 3/3 |
| Second chance | offense | -5.09% | +0.086 | +0.092 | 3/3 |
| FT frequency | defense | -2.54% | +0.044 | +0.061 | 3/3 |
| FT total | defense | -1.99% | +0.035 | +0.053 | 2/3 |
| MID frequency | defense | -1.31% | +0.024 | +0.040 | 3/3 |

For the six offensive shot priors:

| Prior | Selected development lane | RMSE change | Box R2 | RTSS R2 | Winning folds |
|---|---|---:|---:|---:|---:|
| RIM frequency | drives + creation roles | -0.42% | 0.310 | 0.316 | 2/3 |
| RIM finishing | box only | — | 0.178 | — | — |
| MID frequency | creation roles only | -0.52% | 0.498 | 0.504 | 2/3 |
| MID finishing | creation/touch structure | -0.79% | 0.126 | 0.140 | 3/3 |
| THREE frequency | creation + screens + q | -0.59% | 0.243 | 0.252 | 2/3 |
| THREE finishing | box only | — | 0.148 | — | — |

The strongest direct self-created/potential-assisted result is MID:

- role-only MID frequency improves pooled RMSE by `0.52%`;
- role-only MID finishing improves pooled RMSE by `0.73%` and wins all three
  folds; and
- adding touch/dribble and catch-and-shoot structure raises the MID finishing
  gain to `0.79%`, also in all three folds.

Role-split actual FG percentages add little after the box model already knows
overall zone FG%. The incremental information is the creation context itself:
potential-assisted share, after-pass self-creation, dribble burden, and
catch-and-shoot mix. Raw `q` fields do not improve MID finishing beyond that
compact creation model. They help THREE frequency modestly but do not rescue
THREE finishing.

### Development decision

Retain box-only for offensive RIM finishing and THREE finishing. Freeze the
four offensive shot-frequency/MID-finishing tracking candidates and the
stronger defensive/second-chance candidates for a later as-of cutoff harness.
Because every 2023-2025 season now participated in model development, none is
a fresh confirmation set.

## Direct low-alpha ORAPM/DRAPM prior experiment

The component targets above remain the preferred additive architecture. A
separate experiment asks a narrower question: if RTSS is trained directly on
noisy annual total offense and defense RAPM, which tracking mechanisms survive
held-out-season testing?

Run:

```bash
.venv/bin/python wnba_test/cross_validate_wnba_rtss_total_priors.py
```

The script builds pure annual total targets from matching
`FIRST_CHANCE + SECOND_CHANCE_CLEAN` rows and compares RTSS-only, box-only, and
box-plus-RTSS models. The target solve uses orthogonal alpha sweeps: offense
alpha varies from `25` to `750` while defense stays at `1000`, then defense
alpha varies from `50` to `1000` while offense stays at `750`. This is required
because offense and defense are coupled in the same lineup solve; moving both
penalties at once cannot identify which change caused a result.

The prediction harness uses the same 412 qualified player-seasons as the
component CV (`129/132/151` by year), symmetric leave-one-season-out outer
folds, within-season exposure-weighted RTSS standardization, and inner-season
Ridge-alpha selection. A nested compact selector admits an RTSS field only if
it improves both inner held-out seasons. Feature interpretation comes from
30-repeat permutation importance on outer held-out seasons, mechanism-family
permutation, and standardized coefficient sign stability. These are
development estimates; all three seasons are spent.

### Target and model result

Offense contains real tracking-predictable signal, but progressively looser
targets trade repeatability for variance:

| Off alpha (def=1000) | Target SD | Mean YoY corr | RTSS-only R2 | RTSS-only corr | Box R2 |
|---:|---:|---:|---:|---:|---:|
| 25 | 5.122 | 0.275 | 0.223 | 0.472 | 0.256 |
| 50 | 4.783 | 0.280 | 0.238 | 0.488 | 0.276 |
| 100 | 4.325 | 0.291 | 0.258 | 0.508 | 0.302 |
| 250 | 3.540 | 0.318 | 0.292 | 0.541 | 0.347 |
| 500 | 2.846 | 0.349 | 0.320 | 0.567 | 0.387 |
| 750 | 2.430 | 0.370 | 0.335 | 0.580 | 0.402 |

`off_alpha=250` is the lowest defensible exploratory target: it still has a
wide `3.54` point standard deviation, while RTSS explains `29.2%` of held-out
variance at `0.541` correlation. The `25-100` targets are more dramatic but
less repeatable and less predictable. This is a research target choice, not a
new live-solve alpha.

Direct defense does not clear the same standard:

| Def alpha (off=750) | Target SD | Mean YoY corr | RTSS-only R2 | RTSS-only corr | Box R2 |
|---:|---:|---:|---:|---:|---:|
| 50 | 3.945 | 0.108 | 0.002 | 0.052 | 0.043 |
| 100 | 3.516 | 0.108 | 0.008 | 0.089 | 0.052 |
| 250 | 2.819 | 0.113 | 0.004 | 0.072 | 0.069 |
| 500 | 2.236 | 0.129 | 0.003 | 0.074 | 0.080 |
| 1000 | 1.673 | 0.159 | 0.007 | 0.098 | 0.091 |

Even `def_alpha=1000` has only `0.159` mean year-over-year correlation and
RTSS-only `R2=0.007`. Low-alpha annual DRAPM is therefore not a suitable
training target for a direct defensive prior. The better defensive path is
still factor-specific priors and substantially stronger lineup shrinkage.

### Features that matter

At the exploratory `off250/def1000` target, the RTSS-only model's stable
offensive families rank:

1. creation role;
2. playmaking;
3. realized role-split finishing;
4. offensive rebound positioning;
5. screening/off-ball role;
6. shot profile; and
7. MID/THREE shot quality and shooter adjustment.

The strongest individual held-out-season fields have the same coefficient
sign in all three outer fits: pass-led-shot share, pass-turnover share
(negative), potential-assisted THREE finishing, pick-pop share, self-created
RIM rate, MID attempt share (negative), potential assists received, offensive
rebound opportunities, pick-roll share (negative), and self-created MID
finishing. These are predictive role/talent proxies, not causal credit
assignments. In particular, pick-pop and pick-roll signs describe player role
within this correlated feature set.

For defense, only defensive rebound-opportunity positioning has a meaningful
three-fold permutation contribution. Rotation and contest fields are small;
MID/THREE `qSQ/qSP` allowed is near zero. That feature ranking must not be used
to manufacture a direct DRAPM prior from an essentially unpredictable target.

The full RTSS offense model is useful on its own, but box remains stronger.
Nested compact RTSS additions worsen pooled box RMSE at every tested alpha and
win only one of three outer folds at `off250/def1000` (`+0.43%` pooled RMSE,
where positive is worse). At the strongest-shrunk `off750/def1000` target the
gap is only `+0.09%`, still with one fold win. RTSS therefore does not earn a
monolithic increment on top of box. Use the already validated
component-specific residual lanes instead.

## NBA-to-WNBA bootstrap

Men's data can strengthen the WNBA prior architecture, but the local inputs
support two different levels of transfer:

1. **Box/component transfer is feasible now.** The repo has long NBA component
   histories and aligned RIM/MID/THREE, FT, turnover, and second-chance target
   families. These can estimate stable feature sets, coefficient signs, and a
   prior distribution over box-to-component slopes.
2. **Exact RTSS-role transfer is not currently feasible.** The local rich
   markings export is WNBA-only. No local NBA surface has the identical
   `shots.assistOpp`, `passes.assistOpp`, `qSQ`, `qSP`, contest, and screen
   schema. Made-only assisted-shot PBP is not a substitute for potential
   assists on misses.

The preferred model is hierarchical partial pooling, not naive NBA/WNBA row
pooling. For each component:

```text
beta_WNBA = beta_NBA + delta_WNBA
```

with a penalty on the WNBA deviation:

```text
loss_WNBA + lambda_transfer * ||beta_WNBA - beta_NBA||^2
```

Implementation requirements:

- use the same atomic target definition in both leagues;
- standardize inputs within league-season;
- fit NBA slopes on historical NBA seasons only;
- select `lambda_transfer` using WNBA outer-season cross-validation;
- compare WNBA-only, NBA-only, naive pooled, and hierarchical models on the
  exact same WNBA folds;
- allow component-specific transfer strength; and
- retain the WNBA-only RTSS residual layer until identical men's tracking
  fields are available.

The NBA should have the most influence where WNBA sample is weakest and the
mechanism is structurally portable: box-to-zone frequency, box-to-finishing,
turnover, FT, and second-chance relationships. It should have little automatic
authority over league-specific intercepts, raw rates, `qSP-qSQ` calibration,
spacing environment, or the exact value of self-created versus assisted roles.

This hierarchy can enlarge the effective box-prior training sample
substantially without pretending NBA possessions are WNBA possessions. It
does not create new independent validation; a frozen transfer specification
still needs an as-of WNBA cutoff test or fresh 2026+ confirmation.

## Decision

Keep the live WNBA metric unchanged.

Freeze the strongest tracking families as candidates for the next full
component-sum future-possession gate. The 2025 confirmation has now been
opened and must not be reused for further feature selection. A live promotion
requires fresh 2026-or-later RTSS confirmation, additive prior construction,
and a final comparison against the existing box-only prior inside the frozen
single-season solve.

## Artifacts

- `wnba_test/build_wnba_rtss_prior_features.py`
- `wnba_test/evaluate_wnba_rtss_priors.py`
- `wnba_test/cross_validate_wnba_rtss_priors.py`
- `wnba_test/cross_validate_wnba_rtss_total_priors.py`
- `nba_pipeline/tests/test_wnba_rtss_total_priors.py`
- `nba_pipeline/tests/test_wnba_rtss_prior_features.py`
- `wnba_test/Results/rtss_prior_research/player_season_features.csv`
- `wnba_test/Results/rtss_prior_research/development_2024_grid.csv`
- `wnba_test/Results/rtss_prior_research/confirmation_2025.csv`
- `wnba_test/Results/rtss_prior_research/confirmation_2025_predictions.csv`
- `wnba_test/Results/rtss_prior_research/selected_model_coefficients.csv`
- `wnba_test/Results/rtss_prior_research/evaluation_manifest.json`
- `wnba_test/Results/rtss_prior_cross_validation/season_folds.csv`
- `wnba_test/Results/rtss_prior_cross_validation/pooled_results.csv`
- `wnba_test/Results/rtss_prior_cross_validation/best_by_component.csv`
- `wnba_test/Results/rtss_prior_cross_validation/out_of_fold_predictions.csv`
- `wnba_test/Results/rtss_prior_cross_validation/fold_coefficients.csv`
- `wnba_test/Results/rtss_prior_cross_validation/cross_validation_manifest.json`
- `wnba_test/Results/rtss_total_prior_cv/model_summary.csv`
- `wnba_test/Results/rtss_total_prior_cv/target_reliability.csv`
- `wnba_test/Results/rtss_total_prior_cv/feature_importance.csv`
- `wnba_test/Results/rtss_total_prior_cv/family_importance.csv`
- `wnba_test/Results/rtss_total_prior_cv/full_fit_coefficients.csv`
- `wnba_test/Results/rtss_total_prior_cv/candidate_oof_priors.csv`
- `wnba_test/Results/rtss_total_prior_cv/report.json`

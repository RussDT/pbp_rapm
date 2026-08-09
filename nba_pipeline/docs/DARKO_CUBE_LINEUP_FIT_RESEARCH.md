# DARKO Cube Lineup-Fit Research

**Date:** 2026-07-28
**Status:** read-only research note; no production model or data changes
**Decision scope:** a separate, explainable fit layer for five-player DARKO Cube lineups

## Executive decision

Ship no more than one score from the current evidence: **Creation Capacity and
Load Balance** (usage), and even that should remain experimental until it is
re-estimated against the actual Cube baseline and clears a new untouched
forward test.

The repository does not currently support publishing spacing, playmaking,
defensive versatility, turnover complementarity, or second-chance
complementarity as additional Cube points:

- In the 2014-2026 historical Five-Man Fit refit, the usage feature family
  improved possession-weighted forward MSE by `1.50%` in 2018-2023 development
  and `2.24%` in the already-opened 2024-2026 later-season readout. Usage was
  the winning offensive feature group. Adding on-ball or playmaking weakened
  development performance; adding the broader pressure group, which includes
  three-point volume, rim/FT pressure, turnover burden, and usage-weighted
  shooting, made it worse than the EPM-only baseline
  ([historical report](../results/five_man_fit/historical_epm_fit_report.json);
  [handoff](./FIVE_MAN_FIT_LAB.md#2014-2026-historical-epm-refit)).
- The strongest defensive construction challenger—steal/block activity plus
  defensive rebounding and rim-protection balance—improved development MSE by
  `1.95%` but lost `-1.25%` in 2024-2026. It is correctly rejected in the
  product ([historical report](../results/five_man_fit/historical_epm_fit_report.json);
  [handoff](./FIVE_MAN_FIT_LAB.md#2014-2026-historical-epm-refit)).
- The older recovered Five-Man Fit artifact has useful spacing/gravity fields,
  but its original training pipeline and exact validation contract were lost.
  Its 459 output rows cannot establish new coefficients or causal evidence
  ([handoff](./FIVE_MAN_FIT_LAB.md#original-training-pipeline-recovery-boundary);
  [recovered rows](../../five-man-fit-lab/data/lineups.json)).
- The active decomposition supplies excellent candidate inputs and exact
  additive identities, but a decomposition is not evidence of nonlinear fit.
  `ALT_TOV_VALUE` and `SECOND_CHANCE_CLEAN`, for example, are already parts of
  player impact. A Cube fit bonus made directly from their sums would usually
  count the same impact twice
  ([public DECOMP contract](./PUBLIC_DECOMP.md#public-tree);
  [source-row algebra](./PUBLIC_DECOMP.md#source-row-algebra)).

The default answer to the central complementarity question is therefore:
**add the good-oriented offensive and defensive player components in the
existing impact baseline; apply no additional interaction unless a
cross-fitted residual model shows transported incremental value.** Positive
second-chance offense paired with negative second-chance defense offsets
arithmetically. Extreme turnover components may be strengths or weaknesses,
but the repository has not shown that their combination creates either
synergy or redundancy beyond the additive values.

## Product context and handoff inputs

The product sketch targets approximately 100% summed usage and illustrates
the following symmetric piecewise curve:

| Summed usage | Prototype score |
| ---: | ---: |
| 80% | -2.5 |
| 90% | -1.0 |
| 95% | -0.5 |
| 100% | 0.0 |
| 105% | -0.5 |
| 110% | -1.0 |
| 120% | -2.5 |

The DataBallr handoff also establishes the current Cube baseline:

```text
Cube score
= sum of five DARKO DPM values at coefficient 1 per player
+ 1 point for a complete PG/SG/SF/PF/C assignment
+ 0.2 point for each prior-teammate pair
```

There are no minutes weights and no extra era normalization. The reference is
the average player, while the selectable player pool is restricted to
`dpm > 0`. Usage is `player_shift.Usage`, interpreted as each player's
percentage-point season/team role demand. If any selected player lacks usage,
the prototype withholds the usage score; production missingness/coverage has
not yet been audited. These are cross-repository product facts supplied in the
DataBallr usage-prototype handoff; the referenced DataBallr implementation
files were not available for local inspection in this RAPM checkout.
The coordinator identified the following handoff sources:

- `src/pages/devCubeUsageFitPrototypeModel.ts`
- `src/pages/devCubeUsageFitPrototype.tsx`
- `src/pages/DevCubeBoard.tsx`
- `src/components/laker82/duel/duelModes.ts`
- `backend/app/routes/supabase/sixrings_factor_game.py`
- `migrations/20260723190119_darko_cube_force_position_columns.sql`
- `migrations/20260724035016_mp_darko_cube_pair_weighted_teammate_bonus.sql`
- `migrations/20260725211246_remove_darko_cube_second_pick_bonus.sql`
- `fresh_docs/CubeUsageFitPrototype.md`
- `fresh_docs/SixRingsDuels.md`

The dated migrations are authoritative over stale documentation that still
describes the teammate bonus as `+0.25`; the current handoff value is `+0.2`.

The symmetric curve remains a UI hypothesis. This task's read-only audit of
`historical_epm_fit.parquet` found:

- a one-feature high-side hinge above 100%, constrained to a non-positive
  coefficient and selected in 2018-2023,
  improved MSE by `2.245%` in development and `2.115%` in the already-opened
  2024-2026 readout, with a slope near `-1.406` points per 10 excess usage
  points;
- the exact high-side prototype curve improved MSE by `2.499%` and `2.166%`;
- the symmetric mirrored curve improved only `1.263%` and `1.162%`.

For the curve comparison, each candidate was centered using only its training
sample before scoring the next season; the product-facing penalty below is
instead anchored at zero for role demand at or below 100%. The one-feature
audit selected its hinge and non-negative magnitude in the 2018-2023
development folds, then applied that specification forward. It is a task-local
sensitivity analysis over the checked-in historical fit artifact, not a
checked-in production model.

Those results make the overload side pilot-ready, but do not support mirroring
it below 100%. The later window is not an untouched holdout.

## 1. Estimand

Let \(l\) be a five-player lineup, \(g\) a game or stint, \(s\) a season, and
\(S=1\) the product-relevant sample (valid five-player identities, adequate
coverage, and the Cube-eligible `dpm > 0` pool). Let \(Y_{lgs}\) be offense,
defense, or net points per 100 possessions. Let \(B_{lgs}\) be the existing
Cube expectation: five unit-weight DARKO DPM values, the complete-position
bonus, and prior-teammate bonuses. In the historical outcome model, the
equivalent baseline must also control the opponent actually faced. Let
\(X_{ls}\) contain only pre-outcome composition features.

The target fit function is:

\[
\tau(x)=
E\left[Y_{lgs}-B_{lgs}\mid X_{ls}=x,\,
\text{roles/minutes},\text{era},S=1\right].
\]

Equivalently, estimate a residual model:

\[
Y_{lgs}
=B_{lgs}
+f_{\theta}(X_{ls})
+u_{\text{team-season}}
+u_{\text{lineup}}
+\epsilon_{lgs}.
\]

The desired score is \(f_{\theta}\), not the exact-lineup random effect. The
random effects absorb coaching, continuity, team scheme, and repeated-lineup
identity; they must shrink toward zero and must be zero for a hypothetical
Cube five with no shared possessions. This matches the existing builder's
decision to set hypothetical exact-five identity to zero
([builder contract](./FIVE_MAN_FIT_LAB.md#hypothetical-five-builder)).

The estimand is:

- incremental to the **complete Cube baseline**, not merely incremental to raw
  player sums;
- opponent- and season/era-adjusted during estimation, with no additional era
  normalization applied to the published Cube formula;
- evaluated under Cube's equal one-player weights, with no minutes weighting;
- predictive and transportable, not causal player credit;
- defined separately for offense and defense before any net rollup;
- evaluated on future games, teams, and player combinations.

The existing Five-Man Fit offensive target is a useful prototype: season-
relative actual ORtg minus five-player O-EPM expectation, adjusted for
possession-weighted opponent D-EPM. The defense target mirrors it with
positive-good signs
([target construction](../scripts/build_historical_five_man_epm_fit.py#L556-L629)).

## 2. Repository data audit

| Surface | Grain | Local coverage | Relevant fields | Use and limitation |
| --- | --- | --- | --- | --- |
| `LINEUP_NET_RATING_POSSESSIONS*.parquet` | One counted scoreboard possession with `O1-O5`, `D1-D5`, points, teams, game | 2014-2026 RS, 13 files | Actual action-score points and possession flags | Best current outcome surface for exact-five offense/defense; explicitly separate from solver-oriented RAPM parquets ([contract](./LINEUP_NET_RATING_PARQUET.md#current-duplicate-rule)). |
| Historical Five-Man Fit model/report | One qualifying exact-five lineup-season after aggregation | 1,469 feature-complete lineup-seasons from about 3.07 million source possessions; at least 200 possessions on each side; 2014-2026 | Opponent-adjusted EPM residuals, usage/on-ball/passing/E-Skills features, early/late splits | Closest direct evidence. Development is 2018-2023; 2024-2026 has already been opened and is not pristine confirmation ([scope](../results/five_man_fit/historical_epm_fit_report.json)). |
| Season-end Dunks & Threes snapshots | Player-season | 2014-2026 | `p_usg`, rim/3PA/FTA/TOV/TS%, DRB/STL/BLK skills, O-EPM, D-EPM | First-party player-skill/impact inputs used by the historical model. They are season-end rather than as-of-game, so a Cube production experiment needs date-safe snapshots ([loader](../scripts/build_historical_five_man_epm_fit.py#L268-L297)). |
| Historical role cache / stats table | Player-season | Feature-complete model window 2014-2026 | on-ball-time%, potential assists/100 | Supports creation/playmaking hypotheses. Current on-ball/playmaking feature groups did not win selection ([loader](../scripts/build_historical_five_man_epm_fit.py#L190-L265)). |
| `FIRST_CHANCE` + `SECOND_CHANCE_CLEAN` | One processed standard possession with exact offensive/defensive fives | RS and PS files for 1997-2026 | first-chance points, turnover and shot atoms, physical second-chance value | Exact mechanism ledger and candidate residual features. Historical Cube-fit modeling should use the validated 2014+ exact-lineup window unless older lineup quality is separately certified ([recovered data](./VPM_REPRODUCTION.md#recovered-local-data)). |
| `ASSIST_POINTS`, `RIM_ASSIST`, `TOV` and shot-role parquets | Possession, attempt, or five focal-player opportunities depending on metric | Core families span 1997-2026; specialized shooter/passer role surfaces are recent subsets | assisted points, rim assists, bad-pass/scoring TOV, self-/teammate-shot roles | Useful for role-balance candidates, but some labels are descriptive PBP proxies rather than tracking-based gravity, screening, or movement ([role caveats](../scripts/build_historical_five_man_epm_fit.py#L47-L76); [TOV contract](./tov-decomposition.md#overview)). |
| Active player DECOMP / VPM inputs | Player-window or player-season coefficients | current public rolling windows; underlying possession families 1997-2026 | rim/three frequency and making, FT, TOV value, second chance | Stable, interpretable candidate inputs. They are already components of player impact, so only centered nonlinear interactions may enter a fit test ([public tree](./PUBLIC_DECOMP.md#public-tree)). |
| NBA matchup assignments | Shot-defender link | season-end 2017-2025 | one or multiple assigned defenders per shot | Provider-specific assignment, not closest-defender geometry. Useful for future defensive-role summaries after strong shrinkage; not a generic versatility label. |
| Rich ShotQuality defender geometry | Shot | 2024-25 only | closest/second-closest distances and angles, contest/obstruction/paint geometry | Stronger than assignment identity for 2024-25 threes, but too narrow and licensed/provider-bound for a sustainable all-player Cube score. There is no native secondary-defender identity. |
| Recovered original Five-Man Fit table | Exact-five published lineup | 459 lineups with at least 200 possessions | usage, on-ball, passing, pressure, spacing/gravity diagnostics, baseline, residual, reliability | Product-recovery evidence only; original training code and sample construction are unavailable ([status](./FIVE_MAN_FIT_LAB.md#status)). |

The clean net-rating surface uses actual action-score points, hard period
boundaries, and explicit zero-possession administrative free throws. This
matters: a 2026 scan found 3,507 period-boundary anomaly groups under the
standard RAPM split, so the solver parquet should not silently become the
lineup outcome surface
([initial evidence](./LINEUP_NET_RATING_PARQUET.md#initial-evidence)).

The actual checked-in current DECOMP artifact is
[`master_results/weighted_factors_decomp_22_26_all_rb_se_a2000_4000.csv`](../master_results/weighted_factors_decomp_22_26_all_rb_se_a2000_4000.csv).
It contains 1,020 player rows for the 2022-2026 `ALL` window. Its filename
records score-margin rubberband adjustment, season effects, and offense/defense
ridge alphas of 2,000/4,000. It is produced by
[`run_decomp_rolling.py`](../scripts/run_decomp_rolling.py) from the exact
first-chance and second-chance target tree. In the active contract, the
product's broad `oTS`/`dTS` concepts map to EFG plus FT value,
`oTOV`/`dTOV` to point-valued first-chance turnover value, and `oSC`/`dSC` to
physical `SECOND_CHANCE_CLEAN`, with defense oriented positive-good. The older
six-factor `oREB`/`dREB` columns are rebound-opportunity factors, not aliases
for this active second-chance component
([TOV/SC reference](./tov-decomposition.md#weighted-factors-column-reference)).
The DataBallr mock decomposition values were not used here. Because the
checked-in artifact is outcome-derived over its full window, an interaction
experiment must rebuild its inputs as-of or cross-fitted; it cannot join the
same full-window coefficients back to the same outcomes.

The NBA's own glossary defines usage as the percentage of team plays used by a
player while on the floor, using field-goal attempts, possession-ending free
throws, and turnovers in the numerator
([NBA Stats glossary](https://www.nba.com/stats/help/glossary?hidenav=true)).
That is why five historical player usage values should be described as **role
demand**, not as a literal projection of how one new lineup's possessions will
be allocated.

### Measured evidence already available

1. **Usage:** The historical refit normalizes five `p_usg` values to displayed
   shares summing to 100%, while retaining their raw sum as role demand. Its
   shortfall hinge begins below 92% and is fixed at `-1.5` points per 10
   percentage points; overload begins above 103% and is learned with other
   usage-distribution terms
   ([feature construction](../scripts/build_historical_five_man_epm_fit.py#L373-L406)).
   Usage alone won group selection (`+1.50%` development MSE, `+2.24%`
   later-season MSE).
2. **On-ball/playmaking:** Neither family alone improved development MSE; adding
   either to usage reduced the gain. They remain zero-contribution diagnostics
   in the builder
   ([ablation report](../results/five_man_fit/historical_epm_fit_report.json);
   [builder description](./FIVE_MAN_FIT_LAB.md#hypothetical-five-builder)).
3. **Spacing/pressure:** The pressure family includes rim pressure, FT pressure,
   three volume, turnover burden, and usage-weighted TS%. Pressure alone lost
   `-1.17%` development MSE; usage plus pressure lost `-0.20%` versus the
   EPM-only baseline. This is not a clean spacing-only experiment, but it is
   direct evidence against publishing the current aggregate as an extra score
   ([feature family](../scripts/build_historical_five_man_epm_fit.py#L70-L76);
   [ablation report](../results/five_man_fit/historical_epm_fit_report.json)).
4. **Defensive construction:** Activity plus possession-finish features failed
   later-season transport (`-1.25%` MSE gain). Workload/on-ball defensive
   groups did not beat EPM in development. A descriptive top-two usage share
   versus defense-above-EPM correlation is only `0.0181`
   ([historical product](../../five-man-fit-lab/data/historical_lineups.json)).
5. **Generic all-in defensive strength:** Adding opposing-lineup D-EPM totals
   or richer aggregates did not improve forward 2024-25 three-point loss;
   physical shot geometry did. This is a shot-level result rather than a full
   lineup-fit test, but it reinforces the need for mechanism-specific defensive
   features instead of another impact sum
   ([comparison](./SQ_DEFENDER_PROVIDER_COMPARISON.md#opposing-lineup-d-epm-result)).
6. **Exact-lineup evidence:** A 5,000-possession prior was selected for the
   historical exact-lineup residual. It improved confirmation early-to-late
   offense MSE by `1.23%`, but a hypothetical Cube lineup receives zero identity
   effect. This is evidence for shrinkage, not a general composition score
   ([reliability](./FIVE_MAN_FIT_LAB.md#2014-2026-historical-epm-refit)).
7. **Pair synergy:** The DRL/Shapley work has a separate pair-synergy
   diagnostic, but it remains unstabilized and is not a public/production
   output. It is a useful experimental template for minimum samples,
   permutation tests, and multiplicity control, not evidence for a Cube bonus
   ([build status](./DRL_SHAPLEY_BUILD_STATUS.md);
   [implementation](../scripts/train_drl_shapley.py)).

## 3. Candidate features and nonlinear forms

Every candidate begins with good-oriented, centered player-season inputs. Raw
component sums remain in \(B\). The fit model receives only terms that cannot be
recreated by adding player impact: weak-link values, concentration, imbalance,
or explicitly centered pair interactions.

### A. Creation capacity and load balance — retain

Inputs:

- raw five-player predicted-usage demand;
- usage shares normalized across the five equal-weight selected players;
- top-one and top-two share;
- usage HHI or equivalent concentration;
- overload and shortfall hinges.

Provisional form:

\[
f_{\text{usage}} =
\beta_L(k_L-D)_+
+\beta_H(D-k_H)_+
+g(\text{top-two share},\text{HHI}),
\]

with monotonic non-positive tail slopes and knots selected inside the outer
training sample. The current `92%` and `103%` knots are useful challengers, not
Cube constants. Do not include signed `D-100%` alongside both hinges; the
historical code removed that redundant regressor.

For the first Cube pilot, use the simpler high-side challenger
`beta * max(sum(player_shift.Usage) - 100, 0)` and set the low-side penalty to
zero. The handoff audit supports approximately `-1.406` points per 10 excess
usage points as a provisional slope, but it must be re-estimated inside the
complete Cube-baseline experiment. A bounded dev-board contract is:

\[
f_{\text{usage,pilot}}
=\max\left(-3.0,\;-0.1406(D-100)_+\right).
\]

The `-3.0` floor is a product-risk cap, not an empirical tail estimate; only
eight historical lineup-seasons had role demand of at least 120%. Do not
mirror this slope below 100%, and do not award a positive reward below 100%.

### B. Creator-to-finisher role balance — test, do not publish

Inputs available now:

- usage/on-ball creator shares;
- potential assists/100;
- rim and FT pressure;
- teammate-shot role summaries or assisted/rim-assisted possession rates.

Simple challengers:

- top creator share × mean of the other four finish-pressure scores;
- creator potential assists × teammate rim/three finishing opportunity;
- shortage hinge for no high-on-ball creator;
- overlap hinge for two extreme on-ball creators.

The historical `passer_finisher_fit` uses each passer's potential assists times
the other four players' finish pressure
([construction](../scripts/build_historical_five_man_epm_fit.py#L413-L425)).
The current playmaking/on-ball groups did not validate, so this is a sharper
pre-registered retry, not evidence-backed Cube points.

### C. Defensive coverage and rim-help balance — test only with better roles

Inputs required:

- date-safe, stabilized assigned-versus-help defender role;
- position/size/length or defensible role taxonomy;
- zone-specific rim/three suppression and block/rim-help priors;
- projected matchup-role minutes.

Simple challengers:

- weak-link or bottom-two coverage;
- at-least-one-rim-protector hinge;
- role-coverage count/entropy;
- diminishing returns for multiple players with the same defensive role.

Do not substitute five-player D-EPM, steal/block sums, or a generic
"versatility" label. Those formulations have already failed a relevant
forward test. The repository can begin shot-specific role research now, but a
general sustainable versatility score likely needs new matchup/tracking data.

### D. Possession reliability balance (turnovers and second chance) — test,
not publish

Inputs available now:

- offensive TOV prevention, including bad-pass and scoring-TOV children;
- defensive turnover creation;
- offensive and defensive `SECOND_CHANCE_CLEAN`;
- usage/creator share to place ball-security skills in role context.

Simple challengers:

- primary-creator share × creator TOV prevention;
- creator burden × best teammate release-valve playmaking;
- weak-link defensive rebounding or at-least-one-possession-finisher hinge;
- centered pair products and HHI terms within offense or defense;
- a pre-specified balance term such as
  `min(z_off_SC_positive, z_def_SC_positive)` only after both additive sums are
  in the baseline.

The raw cross-side rule is still additive:

```text
net second-chance baseline
= summed offensive second-chance value
+ summed positive-good defensive second-chance value
```

Thus a positive offensive value and negative defensive value simply offset.
The same rule applies to offensive TOV prevention and defensive TOV creation.
A product, minimum, or imbalance term is an empirical hypothesis. It must
predict future residuals; basketball storytelling is not sufficient.

### Rejected current candidate: standalone spacing/gravity

A sustainable spacing test should use pre-outcome shooter talent and a
weak-link/creator-teammate form—such as bottom-two shooting, nonspacer count, or
creator × teammate shooting—not summed offensive impact. Even then:

- EPM/DARKO impact already reflects much of the spacing environment;
- the historical pressure family failed incremental development selection;
- the recovered original deployment says spacing was tested and excluded after
  worsening 2024-25 development folds
  ([recovered deployment](../../five-man-fit-lab/recovery/original-deployment.html));
- recovered spacing/gravity fields lack the original validation pipeline;
- tracking geometry is narrow in time and not a stable all-player input.

Accordingly, no standalone spacing score should ship. A new spacing experiment
may proceed only against the complete Cube baseline and must report its
increment over that baseline, not its raw relationship with offense.

## 4. Empirical test design

### Analysis rows and targets

1. Build one row per lineup-game side, or stint-game side when reliable stint
   boundaries exist, from the clean scoreboard possession surface. Retain
   lineup-season aggregates only for product display.
2. Estimate offense and defense separately. Use actual scoreboard points for
   the outcome; keep expected-FT and other mechanism ledgers as diagnostics or
   pre-outcome features, never as a substitute scoreboard.
3. Construct the baseline from the same version of DARKO/Impact Monster and
   the same teammate/full-lineup bonuses used by Cube. Adjust offense for faced
   opponent defense and vice versa.
4. Center the target within season and include era/season effects. Include home
   court, garbage-time policy, playoffs policy, and team/coach-season effects
   if they match the product estimand.
5. Weight by possessions, cap any lineup-season's leverage, and report
   unweighted sensitivity.

### Leakage and double-counting controls

- Use impact and skill snapshots known before the game. Season-end EPM is
  suitable for retrospective evidence but leaks future games if used as an
  in-season forecast.
- Cross-fit any player component estimated from the same possession outcomes.
  The outer validation game, season, team, or player must not contribute to its
  feature value.
- Residualize only once against the complete Cube baseline. A feature is not
  incremental merely because it predicts raw ORtg.
- Compare four nested models for every candidate:
  `player sums`; `complete Cube baseline`; `baseline + linear component sums`;
  `baseline + candidate nonlinear terms`. The last contrast is the fit claim.
- Run permutation/placebo tests within season and role strata. Shuffling player
  identities should erase the claimed gain.
- Audit algebraic aliases. Do not enter both a parent and children as if they
  were independent signals; the public DECOMP children close exactly to their
  parents.

### Validation splits

- Primary: expanding season-forward folds; freeze all transforms, knots,
  penalties, and calibration inside each training window.
- Team transport: leave one team-season out, then whole-franchise holdouts.
- Player transport: hold out all lineups containing selected players; include
  a rookies/new-player cold-start readout.
- Combination transport: score unseen exact fives separately from repeat
  lineups. This is the Cube-relevant result.
- Role/minute transport: Cube uses equal one-player weights. Retain alternate
  starter/closing-lineup minute scenarios only as an external-validity
  sensitivity, not as production scoring inputs.
- Era sensitivity: pre/post pace-and-spacing eras and shortened seasons.
- Baseline sensitivity: repeat with the product's league-average-player
  baseline and an alternative replacement-level offset.

The 2024-2026 Five-Man Fit window is already opened and cannot be the final
confirmation set. Freeze the next specification before using a new season.

### Regularization, hierarchy, and uncertainty

- Start with monotonic hinges and a small ridge model. Permit pair terms only
  after their parent main effects are present.
- Use hierarchical shrinkage for team-season, lineup identity, pair identity,
  and player-role effects. Exact-lineup effects must shrink much harder than
  composition features.
- Cluster uncertainty by game; use a game-cluster bootstrap for loss
  differences and coefficient intervals.
- Report fold-by-fold sign, loss difference, calibration slope/intercept,
  distribution shift, and uncertainty—not only pooled MSE.
- Preserve the existing minimum of 200 possessions per side for historical
  display and 75 per early/late half as an audit benchmark. Do not use that
  threshold to grant a hypothetical Cube lineup false certainty. New exact
  fives receive zero identity effect and a composition-score uncertainty based
  on feature support.
- Mark a lineup out of support when any core input is missing or its feature
  vector lies beyond the validated training range.

### Calibration into Cube points

Fit in points per 100 first. Cube uses one raw score point per DPM point
numerically, so the natural starting contract is a one-to-one mapping:

\[
\text{Cube fit points}=f_{\theta}(X).
\]

Estimate a held-out calibration slope as a diagnostic; introduce a non-unit
multiplier only in a new version if that evidence requires it. Do not hand-tune
a multiplier in the UI and apply no minutes multiplier. For a penalty-only
usage product, anchor `D <= 100` at zero rather than centering the published
score and accidentally creating a low-demand bonus. Derive any later display
range from held-out predictions; the pilot's `[-3.0, 0.0]` range is a
temporary risk bound, not a learned tail.

### Evidence threshold

Pre-register a practical minimum gain before opening the next test season.
Promotion requires all of:

1. positive incremental loss improvement versus the **complete Cube baseline**
   in expanding development folds;
2. positive improvement in an untouched season with a game-cluster interval
   excluding zero, or two consecutive untouched seasons if power is low;
3. stable direction under team/player/combination holdouts and replacement/
   minute sensitivities;
4. acceptable calibration and no meaningful subgroup harm or missingness;
5. a player-facing explanation that does not rename an additive impact
   component as fit.

The current usage evidence satisfies the first item against an EPM baseline,
not the complete Cube baseline, and its later-season window is no longer
untouched.

## 5. Production recommendation

| Score | Plain-language definition | Exact inputs | Provisional form | Display | Evidence and decision |
| --- | --- | --- | --- | --- | --- |
| **Creation Capacity and Load Balance** | Does the five ask for more offensive role demand than one lineup can support? | `player_shift.Usage` for all five exact card-year/team records, input-vintage/coverage flags, and the complete Cube baseline | `max(-3.0, -0.1406 * max(sum(Usage)-100, 0))`; no underuse reward or penalty | Experimental range `[-3.0, 0.0]` Cube points; withhold when any input is missing | **Moderate evidence; sole pilot-ready candidate.** High-side challengers improved development MSE about `2.25-2.50%` and later MSE about `2.12-2.17%`; mirrored underuse was materially weaker. Refit against Cube and confirm untouched before promotion. |
| **Creator–Finisher Balance** | Do the primary creators have teammates whose finishing roles complement their passing and pressure? | date-safe on-ball/usage role, potential assists, rim/FT pressure, teammate-shot/assist summaries, and input support flags | One or two centered creator × teammate-finisher terms plus creator-shortage/overlap hinges | `0` Cube points; diagnostic badge only | **Hypothesis, currently negative/weak.** Existing on-ball/playmaking groups lost selection. Test a narrower role-matched form now with existing data. |
| **Defensive Coverage Balance** | Does the five cover the necessary point-of-attack, help, and rim-protection jobs without a weak link? | stabilized assigned/help roles, size/position, zone suppression, rim-help/block priors, and matchup-role exposure | Weak-link, at-least-one-rim-protector, and role-coverage/diminishing-return terms | `0` Cube points; no display range until validated | **Not production-ready.** Generic defensive construction failed confirmation. Requires better matchup/role data and a new forward gate. |
| **Possession Reliability Balance** | Does the lineup put ball security, turnover pressure, rebounding, and second-chance conversion in complementary roles? | date-safe o/d TOV children, o/d `SECOND_CHANCE_CLEAN`, creator share, defensive-rebound/rim roles, and additive baseline components | Creator × ball-security, weak-link finish, and centered pair/balance terms after additive sums | `0` Cube points; show additive diagnostics instead | **Hypothesis only.** Exact inputs exist, but no residual interaction test has shown incremental fit. Positive offense plus negative defense should simply sum until it does. |

Although four concepts are listed, the production score count today is **one
experimental candidate and zero fully confirmed scores**. The other three are
prioritized experiments, not permission to award Cube points.

## 6. Sustainable architecture

### Versioned fit-score contract

Store one immutable manifest per release, for example:

```json
{
  "contract_id": "darko_cube_fit_v1_usage_dev",
  "status": "experimental",
  "baseline_id": "cube_baseline_<version>",
  "input_vintages": {},
  "eligible_population": {},
  "units": "points_per_100",
  "sign": "positive_is_good",
  "features": [],
  "knots": {},
  "coefficients": {},
  "calibration": {},
  "display_policy": {},
  "validation_artifacts": {},
  "trained_through": "YYYY-MM-DD"
}
```

The feature pipeline should emit a reproducible lineup feature row, input
coverage flags, projected minute/role assumptions, support diagnostics, and a
content hash. Coefficients, knots, scaling, caps, and calibration live in the
manifest/config—not TypeScript, UI copy, or a dev-board slider.

### Model lifecycle

- Build features from versioned as-of player snapshots.
- Audit `player_shift.Usage` coverage over the full eligible `dpm > 0` Cube
  pool before exposing the score; keep the explicit withhold behavior for
  missing inputs.
- Freeze production coefficients for a season; refit on a scheduled annual
  cadence unless a documented data break requires an emergency version.
- Keep `experimental`, `validated`, `rejected`, and `retired` statuses.
- Monitor input coverage/drift, score distribution, tail frequency, feature
  support, calibration, team/player subgroup error, and realized incremental
  MSE versus baseline.
- Maintain a changelog with old/new score correlations and player-lineup
  deltas.
- Withhold the fit adjustment and fit-adjusted final score when a required
  input is missing. For merely out-of-support but complete inputs, surface the
  support warning and follow the frozen contract's conservative cap; never
  silently impute a favorable fit.

### DataBallr dev-board consumption

The board should ingest only an experiment result bundle containing:

- contract and baseline IDs;
- feature definitions and input coverage;
- development/confirmation fold metrics and clustered intervals;
- coefficients/knots only for the selected frozen experiment;
- lineup predictions with uncertainty/support flags;
- an explicit decision: `explore`, `reject`, `candidate`, or `promote`.

The board may visualize alternative knots or score mappings, but it must label
them as simulations and must not write those numbers back into the model
contract. In particular, the current 110%/120% usage penalties remain an
editable UI sketch until a frozen experiment bundle supersedes them.

## 7. Prioritized experiments

### Can run immediately with existing repository data

1. **Rebuild the usage test against the actual Cube baseline.** Use
   2014-2026 clean lineup-game outcomes, date-safe DARKO/Impact Monster inputs,
   the `+1` complete-position and `+0.2` teammate-pair bonuses, equal player
   weights, and nested selection of hinge knots. Pre-register the high-side
   hinge as the primary challenger and the symmetric prototype as a falsified/
   secondary arm. This is the only near-production task.
2. **Component-interaction residual screen.** Use cross-fitted player DECOMP
   inputs to test a pre-registered small set: creator × TOV prevention,
   bottom-two defensive-rebound/SC value, at-least-one positive rim/SC
   finisher, and centered o/d TOV or SC balance. Compare against additive
   component sums already in the baseline.
3. **Sharper creator–finisher test.** Replace the broad playmaking group with
   two or three role-matched features. Reject it if it again fails season-
   forward selection.
4. **Spacing falsification test.** Use date-safe shooter talent, bottom-two
   shooting, nonspacer count, and creator × teammate shooting. Treat a null as
   the expected outcome because impact already absorbs spacing. Do not use raw
   lineup offense as the target.
5. **Reliability and support audit.** Quantify repeat-lineup rates, unseen-five
   transport, player cold starts, and how often Cube drafts fall outside the
   historical feature support.

### Requires new or better data

6. **Defensive versatility/coverage.** Acquire or build multi-season,
   role-specific matchup and help assignments with position/size context.
   Current assignment coverage is provider-specific and rich geometry is only
   2024-25.
7. **Screening, movement, and true gravity.** PBP and on-court residuals cannot
   identify optical screening, off-ball movement, or help displacement.
   Tracking-derived, pre-outcome measures are required.
8. **Rotation-aware Cube fit.** If drafted fives are not assumed to play all
   possessions together, the product must supply substitution/role/minute
   scenarios. Exact-five scores alone cannot estimate a whole rotation.

## Decisions needed

1. Which date/vintage of DARKO DPM and `player_shift.Usage` should be frozen for
   each game or Cube release?
2. What historical inputs reproduce the `+1` full-position and `+0.2` prior-
   teammate bonuses so the complete Cube baseline can be tested out of sample?
3. Should the natural one-to-one residual-point/Cube-point mapping remain, or
   does a held-out calibration slope justify a separately versioned multiplier?
4. What practical minimum held-out improvement justifies visible Cube points?
5. What is actual `player_shift.Usage` coverage in the eligible `dpm > 0`
   player pool, and how often would five-player scores be withheld?

Until those decisions are made, the sustainable product posture is: show the
additive Impact Monster components as explanations, keep arbitrary fit sliders
clearly experimental, and award no new points beyond the one usage candidate
that will be re-tested against the complete Cube baseline.

---
name: six-factor-rapm
description: "Russell Thomas's Six-Factor RAPM framework, public possession-structure framing, and ShotQuality TS decomposition for NBA player impact analysis. Use when: (1) discussing, explaining, or writing about Six-Factor RAPM, the six-factor decomposition, possession-stage impact, or structural decomposition of player impact, (2) building or modifying databallr pages that display RAPM data (oTS, oTOV, oREB/oSC, dTS, dTOV, dREB/dSC columns), (3) writing methodology sections, tooltips, or help text for RAPM-related UI, (4) analyzing player profiles using the six factors or ShotQuality TS components (SQ, Contest/Make, FT), (5) creating content about how basketball impact works mechanically, (6) answering questions about what RAPM numbers mean or how to interpret the decomposition, (7) building comparison, randomizer, or visualization features around the six-factor data."
---

# Six-Factor RAPM Knowledge Base

Use this skill in layers. Start with the routing rules and core doctrine below. Only read the deeper reference files that match the task.

## Quick Routing

1. Identify which six-factor series the user means.
2. Use the framework summary and caveats for general explanation.
3. If the task is public-facing copy, explainer text, a methodology intro, SixRings 5Y factor-mode language, `The Shape of Impact`, or any `oSC` / `dSC` possession-stage explanation, load [references/possession-structure-framing.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/possession-structure-framing.md).
4. If the task mentions active Alt3 EFG-value weighted-factor artifacts, `weighted_factors_alt3_efg_value_*`, `oSC`, `dSC`, `SECOND_CHANCE_CLEAN`, or public point-valued factors, read the "Alternate Clean 3-Factor Decomposition" and "Active Alt3 EFG-Value Weighted-Factors Bundle" sections below before answering.
5. If the task mentions current `2026` single-season true-prior outputs, `ss_*` beta fields, `2026` priors, or `rapms/master_results/weighted_factors_26_rs_rb_trueprior.csv`, load [references/2026-single-season-true-prior.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/2026-single-season-true-prior.md) before answering.
6. If the task is about long-term roadmap, future modeling direction, EPM-inspired prior architecture, estimated skills, or how the single-season metric should evolve, load [references/long-term-plan.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/long-term-plan.md) before answering.
7. Load factor-specific or methodology references only if the task needs more detail.
8. Treat current offensive-value notes as active `2026` modeling guidance, not timeless doctrine.

## Current 2026 Series Taxonomy

Before writing copy, labeling data, or comparing exports, determine which series is in scope.

### Pure Six-Factor RAPM

- Pure lineup-derived factors are the `actual_*` fields in `rapm/russell/results/2026_hybrid_factor_rapm.csv`.
- `pure_off`, `pure_def`, and `pure_net` are the point-value summaries from the pure lineup solve.
- This is the `td700` lineup-based, time-decayed, no-box-prior solve.
- Use the "no box priors / pure lineup signal" caveat only for this series.

### 2026 Hybrid Six-Factor RAPM

- The publishable stabilized series is the `hybrid_*` output built by `rapm/russell/build_hybrid_2026_factors.py`.
- It is not a re-solved RAPM with nonzero priors inside the stint regression.
- It is a post-solve empirical Bayes blend:

  `hybrid = (pure * factor_specific_possessions + box_prior * k) / (factor_specific_possessions + k)`

- Offense factors use `off_poss`. Defense factors use `def_poss`.
- `box_*` fields are the prior estimates. `prior_weight_*` fields are the row-level prior shares.
- When discussing the current site board or `factor-lab-next/public/rapm/2026_hybrid_factor_rapm.csv`, call it `Hybrid RAPM`, `Stabilized RAPM`, or `Hybrid Six-Factor RAPM`.

### True Prior Six-Factor RAPM

- The true prior series is built by `rapm/russell/build_true_prior_factor_rapm.py`.
- It is not the hybrid blend. It is a real prior-centered solve inside the RAPM regression:

  `min ||W^(1/2)(y - Xβ)||² + λ_off ||β_off - μ_off||² + λ_def ||β_def - μ_def||²`

- Current-player comparison exports from this build use `prior_*` columns for the true prior solve and `pure_*` for the pure solve.
- Do not describe `prior_*` as the hybrid blend.

### 2026 Single-Season True-Prior Beta

- The current local single-season beta is a distinct `2026` series and not the same thing as the long-window `21_26` true-prior export.
- Its active local spec is `26 26 RS`, no within-season time decay, `off_alpha=3000`, `def_alpha=5000`.
- When the task is about current `2026` priors, `ss_*` fields, or `rapms/master_results` single-season artifacts, load [references/2026-single-season-true-prior.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/2026-single-season-true-prior.md).
- Do not collapse this series into the hybrid board or the td700 single-season test run.

### Naming Rules

- `actual_*` = pure lineup-derived six-factor RAPM.
- `box_*` = box prior estimate.
- `hybrid_*` = stabilized hybrid output.
- `prior_*` = true-prior RAPM output from the prior-centered solve.
- Do not describe `hybrid_*` as pure RAPM.
- Do not apply the pure-method "no priors" caveat to the hybrid export.

### REB vs SC Label Discipline

There are two related but distinct six-factor surfaces:

- Legacy pure/hybrid/true-prior six-factor files use `oREB` / `dREB` as native rebounding-rate RAPM factors.
- Active public Alt3 EFG-value weighted-factor files use point-valued `oSC` / `dSC` for direct second-chance impact from `SECOND_CHANCE_CLEAN`.
- Do not rename `oREB` / `dREB` columns to `oSC` / `dSC` unless the artifact is actually an active Alt3 EFG-value weighted-factor bundle.
- Do not describe public `oSC` / `dSC` as residual buckets. In the active EFG-value bundle they are direct second-chance RAPM coefficients.

## Core Doctrine

### Level 1: Six-Factor RAPM

Three independent ridge regressions on the same lineup stint data each predict a different dependent variable. Each regression uses a +1/-1 design matrix and produces both an offensive and defensive coefficient per player.

|            | Shooting (TS) | Turnovers (TOV) | Rebounding (REB) |
|------------|--------------|-----------------|------------------|
| **Offense** | oTS          | oTOV            | oREB             |
| **Defense** | dTS          | dTOV            | dREB             |

- Units are native units: TS%, TOV%, OREB%, not points.
- The second-stage reconstruction regression converts factors to point value and fits total impact at about `R^2 = 0.985`.
- TS, TOV, and REB models are independent from one another. Within each model, offense and defense are estimated together.
- Rough implication: nearly all player impact on scoring margin can be organized through these six channels.

### Level 1A: Public Possession-Structure Framing

For public explainers, SixRings 5Y factor modes, and active Alt3 EFG-value artifacts, frame impact through the possession chain:

`Turnover Value -> Shot Value -> Second-Chance Value`

Basketball impact can be explained as three possession questions applied to both sides of the ball:

- Before the shot: does the possession survive into a scoring attempt, or die before a shot exists?
- The shot: once the possession becomes an attempt, how valuable is that attempt?
- After the miss: does the missed shot end the possession, or turn into second-chance value?

In the active point-valued public framing:

- `Offense = oTOV + oTS + oSC`
- `Defense = dTOV + dTS + dSC`
- `Total RAPM = oTOV + oTS + oSC + dTOV + dTS + dSC`

Use point-per-100 language only when the artifact is already point-valued, such as `weighted_factors_alt3_efg_value_*` or the SixRings factor-game pool. For native pure/hybrid TS/TOV/REB factors, keep the native-unit language and reconstruction-regression caveat.

### Level 2: ShotQuality TS Decomposition

The TS factors decompose into Shot Quality, Shot Making, and Free Throw value using ShotQuality pre-shot expected values and foul-type-aware FT baselines.

- `SQ + Contest + FT = TS` on every possession by construction.
- Any new decomposition should first prove this kind of row-level identity in the processed parquet/CSV surface before solving RAPM. Coefficient sums can show tiny separate-ridge / CSV-rounding gaps, but the row algebra should be exact when the component definitions share the same possession universe.
- The same stint design matrix is reused for four scorings.
- The decomposition reconstructs TS with `R^2 > 0.9999`.
- Units are points per `100` shots, not possessions.

Read [references/ts-decomposition.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/ts-decomposition.md) when the task needs the exact possession logic, foul baselines, or player examples.

### Conceptual Interpretation Rules

- RAPM measures how a player changes team outcomes while on the floor, not merely what they personally record.
- Six-Factor RAPM is an additive map of where scoreboard impact comes from, not just a descriptive category set.
- `oTS` is not personal scoring efficiency. It is team point generation per shot while the player is on offense.
- `oTS` can be read as `oEFG + oFT`, where `oFT` is offensive free-throw point generation, not only self foul drawing.
- In the ShotQuality layer, `oTS` opens into `oSQ + oMake + oFT`.
- `oTOV` contains both scoring-turnover pressure and bad-pass turnover burden.
- `oREB` contains both extra-chance creation and second-chance conversion.
- `oSC` / `dSC`, when present in active Alt3 EFG-value artifacts, are direct second-chance value factors: possession extension on offense and possession completion on defense.
- Shot diet categories like Creation, Spacing, Transition, and Finishing are supporting archetype evidence, not RAPM factors.

## Current 2026 Modeling Rules

Use these as the active `2026` offensive-value guidance. For the detailed note set, read [references/offensive-value-notes.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/offensive-value-notes.md). For the current implementation state and next-model brief, read [offensive-value-model-handoff.md](/Users/russellthomas/Docs/2026_NBA_PIPELINE/docs/offensive-value-model-handoff.md).

- Treat current playtype `TOV%` as an all-in turnover drag term, not a pure scoring-turnover metric.
- Treat current playtype `rPPP` as already playtype-relative. Do not add another playtype-difficulty adjustment onto `rPPP`.
- Preferred site explanation stack:
  - `Shot Efficiency`
  - `Turnover Drag`
  - `Own Scoring Value`
  - `Teammate PPP RAPM`
  - `Total Offensive Value`
- Preferred identities:
  - `Own Scoring Value = Shot Efficiency - Turnover Drag`
  - `Total Offensive Value = Own Scoring Value + Teammate PPP RAPM`
- Do not define public `Playmaking` as a residual from own playtype `rPPP`.
- Do not force a strict `rTS + sTOV + pTOV = playtype rPPP` identity.
- Do not use assist-box proxies as the final public definition of playmaking once a teammate-impact RAPM term exists.
- The current exact own-possession accounting layer is `playtype_TS_rPPP + playtype_TOV_rPPP = playtype_rPPP`.
- The current presentation bridge under discussion is `pt_adj_rTS + TOV_value = PPP_eq`, where `PPP_eq = playtype_adj_rPPP * 50`.
- Do not confuse the exact `playtype_TOV_rPPP` accounting term with the `pt_adj_rTS`-anchored `TOV_value` bridge.
- Current strongest simple scoring-prior family is the decayed `pt_adj_ts_add` family, with `pt_adj_ts_add_m3` the best simple variant tested so far against `oTS`.
- Current modeling has partially split into two parallel tracks:
  - playtype turnover contamination extraction
  - `oTS` scoring-prior / shrinkage experiments
- Use the playtype and box layer as future prior material, not just display material.
- Current public naming preference for the scoring-efficiency bridge is:
  - `rTS` = raw scoring efficiency relative to league
  - `zTS` = playtype-adjusted scoring efficiency
- Current clean target preference for `zTS` work is pure `2022-2026` regular-season `oTS`:
  - no playoffs
  - no time decay
  - no rubberband
  - weighted by `5-year offposs`
- Current key result:
  - `zTS add @ 0` already beats plain `TS add` at every tested baseline against clean `oTS`
- Current preferred public fixed-volume ladder:
  - `zTS@20 = (TSA100 / 20) * (zTS + 5) - 5`
  - use this when translating burden into one comparable scoring-value axis
- Current clean turnover-prior target preference is pure `2022-2026` regular-season `oTOV`:
  - no playoffs
  - no time decay
  - weighted by `5-year offposs`
- Current clean turnover result:
  - separate `oTOV_bp` and `oTOV_sc` priors outperform a single direct total-turnover prior on the matched sample
- Current compact turnover-load framing:
  - best simple public structure is `oTOV ~ TOV_100 + Load`
  - current best single turnover-free load term is `creation_TSA100`
  - the current best compact burden basket is `creation_TSA100 + PASSING_Potential Assists/100 + shooting_TSA100 + transition_TSA100`, with `PASSING_on-ball-time%` acting more like a style tax than a pure burden term
- Current offensive-load caveat:
  - the compact burden term is still too guard-biased
  - add a hub-creation proxy such as `box_creation` before treating it as final "true offensive load"

## Key Caveats

Always include the relevant caveat for the series or metric in use.

1. Pure RAPM is pure lineup signal with no box priors. It is noisy for low-minute players.
2. Hybrid RAPM is a post-solve stabilization blend, not a pure solve and not a true-prior regression solve.
3. True-prior RAPM is a prior-centered solve inside the regression, not the hybrid blend.
4. All RAPM outputs are lineup-level signals, not direct causal proof about isolated individual skill.
5. The SQ / Contest boundary depends on ShotQuality's pre-shot model. TS totals and FT totals do not.
6. The framework surfaces patterns and mechanism hypotheses, not proof of mechanism.

## Alternate Clean 3-Factor Decomposition

The repo also has a parallel clean 3-factor offensive split built on a shared first-chance possession universe:

- `FIRST_CHANCE`
- `FIRST_CHANCE_CLEAN`
- `ALT_TS`
- `ALT_EFG_VALUE`
- `ALT_EFG_RIM_FREQ`
- `ALT_EFG_RIM_FG`
- `ALT_EFG_MID_FREQ`
- `ALT_EFG_MID_FG`
- `ALT_EFG_THREE_FREQ`
- `ALT_EFG_THREE_FG`
- `ALT_FT`
- `ALT_FT_FREQ`
- `ALT_FT_SEVERITY`
- `ALT_TOV_VALUE`
- `ALT_BADPASS_TOV_VALUE`
- `ALT_SCORING_TOV_VALUE`
- `SECOND_CHANCE_CLEAN`

Interpretation rules:

- this is a parallel build, not a replacement for the standard six-factor framework
- `ALT_TS` is not literal TS%; it is a first-chance scoring-surrogate metric defined on all first-chance possessions
- `ALT_TS` / `ALT_TOV_VALUE` are solved off `FIRST_CHANCE` using a same-sample non-turnover first-chance TS baseline
- `ALT_TOV_VALUE` is `-avg_non_turnover_first_chance_TS` on true first-chance turnover rows and `0` elsewhere
- `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE = ALT_TOV_VALUE` at the processed-row level
- `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT` at the processed-row level; frequency prices each first-chance FT trip at average trip value, while severity captures trip value above or below that average
- a true first-chance turnover row must have no prior first-chance FGA/FT completion event inside the same possession; if a made shot, FGA, or FT completion already happened, the scoring stays in first chance but the later terminal turnover is not the first-chance TOV bucket
- `FIRST_CHANCE_CLEAN` is the direct parent surface: ExpFT-adjusted first-chance scoring on non-turnover rows, `0` on true first-chance turnover rows
- the clean possession split itself is exact at the processed-row level:
  - `FIRST_CHANCE + SECOND_CHANCE_CLEAN = RAPM scoring layer`
- the clean first-chance scoring identity is exact for the loaded sample:
  - `ALT_TS + ALT_TOV_VALUE = FIRST_CHANCE`
- current public Alt3 means `weighted_factors_alt3_efg_value_*`, not the older `weighted_factors_alt3_*`
- the active EFG-value weighted-factors build displays direct `SECOND_CHANCE_CLEAN` as `oSC` / `dSC`
- in the active EFG-value weighted-factors build, `ALT_EFG_BASELINE` is the balancing bucket that closes the public display to total RAPM; second chance is not residualized
- the older `weighted_factors_alt3_*` path is legacy/audit only; it rebuilt residual `oSC` / `dSC` as `off - oFC` and `def - dFC`

### Active Alt3 EFG-Value Weighted-Factors Bundle

Use this section when explaining columns from `weighted_factors_alt3_efg_value_*`.

Source scripts:

- `nba_pipeline/scripts/build_alt3_efg_value_display_bundle.py` merges solved component RAPMs into a display bundle
- `nba_pipeline/scripts/build_alt3_efg_value_weighted_factors.py` converts that display bundle to the public weighted-factors schema

Public sign and prefix conventions:

- `o*` columns are offensive point-value RAPM components
- `d*` columns are defensive point-value RAPM components, sign-flipped so positive means good defense
- `n*` columns are net components and should be read as offense plus positive-good defense
- `off`, `def`, and `net_rapm` are the total offensive, defensive, and net RAPM values in the same public sign convention
- `Latest_Year`, `player_name`, and the player universe come from the matching legacy alt3 weighted-factors file; that legacy file is not the interpretation source

Main public identities:

- `oFC = oALT_EFG_VALUE + oALT_FT + oALT_TOV_VALUE`
- `dFC = dALT_EFG_VALUE + dALT_FT + dALT_TOV_VALUE`
- `oDISPLAY_SUM = oFC + oSC`
- `dDISPLAY_SUM = dFC + dSC`
- `nDISPLAY_SUM = oDISPLAY_SUM + dDISPLAY_SUM`
- `oRESID = off - oDISPLAY_SUM`
- `dRESID = def - dDISPLAY_SUM`
- `RESID = net_rapm - nDISPLAY_SUM`
- residual columns should be near `0`; visible gaps are usually CSV rounding unless a bundle was assembled incorrectly

Shot-value columns:

- `ALT_EFG_VALUE` in the public weighted-factors file is the displayed total first-chance shot value. It equals the balancing baseline plus the six atomic shot-value pieces.
- `ALT_EFG_BASELINE` is the balancing / replacement-shot bucket that makes the EFG-value display close to total RAPM. It is not second chance and should not be described as an observed shot-location skill bucket.
- `ALT_EFG_ATOMS_VALUE` is the sum of the six atomic first-chance shot-value pieces.
- `ALT_EFG_RIM_FREQ`, `ALT_EFG_MID_FREQ`, and `ALT_EFG_THREE_FREQ` are shot-location frequency values: value from shifting first-chance attempts toward or away from each zone relative to the average first-chance EFG shot value.
- `ALT_EFG_RIM_FG`, `ALT_EFG_MID_FG`, and `ALT_EFG_THREE_FG` are within-zone make values: value from making more or fewer shots than the loaded-sample zone average.
- `ALT_EFG_RIM`, `ALT_EFG_MID`, and `ALT_EFG_THREE` in this public bundle are derived totals from the two atomic children for each zone. Do not confuse them with the older broad zone solver outputs, which are audit-only.
- `ALT_EFG_ATOMS_VALUE = ALT_EFG_RIM_FREQ + ALT_EFG_RIM_FG + ALT_EFG_MID_FREQ + ALT_EFG_MID_FG + ALT_EFG_THREE_FREQ + ALT_EFG_THREE_FG`
- `ALT_EFG_VALUE = ALT_EFG_BASELINE + ALT_EFG_ATOMS_VALUE` in the public bundle

Free-throw columns:

- `ALT_FT` is first-chance free-throw value on the same possession universe as the shot and turnover components.
- `ALT_FT_FREQ` is the value of generating first-chance FT trips, priced at the loaded-sample average first-chance FT-trip value.
- `ALT_FT_SEVERITY` is the value above or below that average trip, combining trip type mix and FT shooting.
- `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT` before display rounding.

Turnover columns:

- `ALT_TOV_VALUE` is the point-valued first-chance turnover complement, not the older rate-style `ALT_TOV`.
- True first-chance turnover rows get `-avg_non_turnover_first_chance_TS`; non-turnover rows get `0`.
- `ALT_BADPASS_TOV_VALUE` and `ALT_SCORING_TOV_VALUE` split `ALT_TOV_VALUE` into bad-pass and non-bad-pass first-chance turnover value.
- `ALT_TOV_VALUE_bp` duplicates `ALT_BADPASS_TOV_VALUE` for compatibility.
- `ALT_TOV_VALUE_sc` duplicates `ALT_SCORING_TOV_VALUE` for compatibility.
- `ALT_TOV_LOSS_VALUE` and `ALT_TOV_VALUE_split` duplicate the summed turnover child value for compatibility.
- The intended child identity is `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE = ALT_TOV_VALUE` before display rounding.

Second-chance columns:

- `SC` and `SC_CLEAN` in this file are direct `SECOND_CHANCE_CLEAN` RAPM coefficients.
- `oSC` / `dSC` are not residual buckets in the active EFG-value bundle.
- The active clean scoring surface is `FIRST_CHANCE + SECOND_CHANCE_CLEAN`, with `ALT_EFG_BASELINE` absorbing the public display residual needed to close to total RAPM.

Assist drilldown columns:

- `ASSIST_POINTS` and `RIM_ASSIST` are included as supporting RAPM drilldowns.
- They are not additive children of `ALT_EFG_VALUE`, `ALT_FT`, `ALT_TOV_VALUE`, or `SC`, and they should not be included in `DISPLAY_SUM`.

Column-family checklist for the public file:

- identifiers: `player_id`, `player_name`, `Latest_Year`
- offense components: `oALT_*`, `oSC`, `oSC_CLEAN`, `oFC`, `oDISPLAY_SUM`, `oRESID`, `oASSIST_POINTS`, `oRIM_ASSIST`
- defense components: `dALT_*`, `dSC`, `dSC_CLEAN`, `dFC`, `dDISPLAY_SUM`, `dRESID`, `dASSIST_POINTS`, `dRIM_ASSIST`
- net components: `nALT_*`, `nSC`, `nSC_CLEAN`, `nDISPLAY_SUM`, `RESID`, `nASSIST_POINTS`, `nRIM_ASSIST`
- totals and weights: `off`, `def`, `net_rapm`, `possessions`, `off_poss`, `def_poss`

Implementation guardrail:

- For every new factor decomposition, verify the parquet or CSV row algebra before interpreting player RAPMs. Good checks include max row gap, sum gap, and child-parent identity counts. Do not use coefficient-level residuals alone to debug the math, because independent ridge solves, centering, and rounded exports can create small coefficient gaps even when row algebra is exact.

## Reference Loading Map

Load only the file you need.

### Factor Deep Dives

- [references/factor-oTS.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/factor-oTS.md): offensive shooting efficiency, spacing, gravity, playmaking, ShotQuality sub-decomposition.
- [references/factor-oTOV.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/factor-oTOV.md): offensive turnover avoidance, ball security, scoring vs bad-pass turnover roles.
- [references/factor-oREB.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/factor-oREB.md): offensive rebounding, rebound burden, self-OREB structure.
- [references/factor-dTS.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/factor-dTS.md): defensive shooting suppression, rim protection, invisible defense.
- [references/factor-dTOV.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/factor-dTOV.md): defensive turnover creation, disruption, gambling tradeoffs.
- [references/factor-dREB.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/factor-dREB.md): opponent OREB denial, scheme effects, low box predictability.

### Framework References

- [references/interpretation-guide.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/interpretation-guide.md): audience-facing interpretation and archetype guidance.
- [references/possession-structure-framing.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/possession-structure-framing.md): public possession-chain framing, SixRings 5Y factor-mode copy, `oSC` / `dSC` explanations, and example peak lists.
- [references/technical-methodology.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/technical-methodology.md): regression specs, ridge setup, time-decay, reconstruction regression.
- [references/ts-decomposition.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/ts-decomposition.md): full ShotQuality TS decomposition narrative.
- [references/offensive-value-notes.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/offensive-value-notes.md): current `2026` notes on playtype TOV, own scoring, teammate PPP RAPM, and priors.
- [references/2026-single-season-true-prior.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/2026-single-season-true-prior.md): current March `2026` single-season true-prior beta spec, artifact paths, schemas, naming rules, and `rapms` mirror locations.
- [references/long-term-plan.md](/Users/russellthomas/.agents/skills/six-factor-rapm/references/long-term-plan.md): long-term roadmap for the single-season true-prior metric, estimated skill priors, validation ladder, and EPM-inspired architecture.

### Current 2026 Source Of Truth

- Box prior builder: `rapm/russell/estimate_2026_boxscore_factors.py`
- Hybrid builder: `rapm/russell/build_hybrid_2026_factors.py`
- True-prior builder: `rapm/russell/build_true_prior_factor_rapm.py`
- Rolling tuner / benchmark harness: `rapm/russell/tune_true_prior_rolling.py`
- Current published app data: `factor-lab-next/public/rapm/2026_hybrid_factor_rapm.csv`
- Current box-prior base rows: `/Users/russellthomas/Docs/csvs/merged_rs_full.csv.gz`
- Current V2-only tracking / hustle supplement: `rapm/bdl_v2_stats.csv`

### Box-Score Predictability Summary

| Factor | Ridge R^2 | XGB R^2 | Spearman | What Box Scores Miss |
|--------|----------|---------|----------|----------------------|
| oTS | 0.594 | 0.610 | 0.692 | ~41% -- spacing, gravity, off-ball movement |
| oTOV | 0.591 | 0.606 | 0.704 | ~41% -- team system effects, tempo management |
| oREB | 0.673 | 0.662 | 0.752 | ~33% -- scheme effects, shot-type rebounding |
| dTS | 0.405 | 0.489 | 0.596 | ~60% -- communication, help defense, funneling |
| dTOV | 0.548 | 0.576 | 0.666 | ~45% -- team pressure, positioning, deterrence |
| dREB | 0.175 | 0.258 | 0.382 | ~82% -- boxing out, scheme, team coordination |

## Data And UI Conventions

### Current Published Hybrid Board

- File: `factor-lab-next/public/rapm/2026_hybrid_factor_rapm.csv`
- Core interpretation:
  - `actual_*` = pure lineup-derived factors
  - `box_*` = box prior estimates
  - `hybrid_*` = stabilized published factors
  - `prior_weight_*` = row-level prior shares in the hybrid blend
  - `pure_off`, `pure_def`, `pure_net` = point summaries from the pure lineup solve
  - `hybrid_off`, `hybrid_def`, `hybrid_net` = point summaries from the hybrid series

### Current True-Prior Outputs

- Canonical weighted master results:
  `/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/master_results/weighted_factors_21_26_all_rb_td700_trueprior.csv`
- Mirrored publish target:
  `/Users/russellthomas/Docs/rapms/master_results/weighted_factors_21_26_all_rb_td700_trueprior.csv`
- Current-player comparison export:
  `rapm/russell/results/2026_true_prior_factor_rapm.csv`

### Current 2026 Single-Season True-Prior Outputs

- Current no-decay weighted factors:
  `/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/master_results/weighted_factors_26_rs_rb_trueprior.csv`
- Mirrored `rapms` target:
  `/Users/russellthomas/Docs/rapms/master_results/weighted_factors_26_rs_rb_trueprior.csv`
- Readable prior pool:
  `/Users/russellthomas/Docs/rapms/master_results/2026_single_season_true_prior_box_priors_fullpool.csv`
- Joined current-player export:
  `/Users/russellthomas/Docs/rapms/master_results/2026_single_season_true_prior_factor_rapm.csv`

### Legacy Pure Long-Window Reference

- File: `~/Docs/pbp_rapm/nba_pipeline/master_results/weighted_factors_21_26_all_td700.csv`
- Columns: `player_id, player_name, Latest_Year, oTS, oTOV, oREB, dTS, dTOV, dREB, off, def, net_rapm, RESID, possessions, off_poss, def_poss`
- `off` = ORAPM, `def` = DRAPM, `net_rapm` = off + def, `RESID` = residual not explained by the six factors

### databallr UI Conventions

- Positive values show a `+` prefix for RAPM fields.
- Green is good for the player's team. Red is bad for the player's team.
- Positive `dTS`, `dTOV`, and `dREB` are good for the player's team in native REB-factor files.
- Positive `dTS`, `dTOV`, and `dSC` are good for the player's team in active Alt3 EFG-value / public second-chance files.
- Default display precision is two decimals for factor values.
- Tooltip copy should include a brief factor explanation plus the lineup-adjusted caveat.

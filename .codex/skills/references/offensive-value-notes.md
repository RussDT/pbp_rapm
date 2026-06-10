# Offensive Value Notes

These notes capture the active `2026` offensive-value discussion that sits next to the six-factor framework. They are current modeling guidance, not permanent doctrine.

Current handoff:

- [offensive-value-model-handoff.md](/Users/russellthomas/Docs/2026_NBA_PIPELINE/docs/offensive-value-model-handoff.md)

## Current Findings

- Current playtype turnover signal comes from Hoopology / playtype possession data and is very close to full offensive turnover volume.
- On the local `2026` regular-season dataset, playtype-implied turnovers were about `25,535` versus `27,282` total player turnovers, or roughly `93.6%` coverage.
- Because of that, current playtype `TOV%` should be treated as an all-in turnover drag signal.
- Do not describe current playtype `TOV%` as a pure scoring-turnover metric.
- The final four-bucket TS layer now reconciles exactly to player `TS_pct`.
- The playtype source layer now has an exact own-possession `rPPP` decomposition:
  - `playtype_TS_rPPP + playtype_TOV_rPPP = playtype_rPPP`
  - same exact identity exists by bucket
- `playtype_diff` should be treated as an upstream scoring-role adjustment.
- Current clean scoring bridge:
  - `pt_adj_rTS = rTSPct + playtype_diff`
- Playtype difficulty and ShotQuality-style TS adjustments overlap, but they are different layers:
  - playtype difficulty = role burden
  - ShotQuality = shot-level burden inside the role

## March 14, 2026: Clean Turnover Split And Offensive Load

### Clean turnover-split priors

Current clean target for turnover prior work:

- pure `2022-2026` regular-season `oTOV` targets
- no playoffs
- no time decay
- weighted by `5-year offposs`

Current best large-feature weighted 5-fold CV results:

- `oTOV_bp`: `0.5216`
- `oTOV_sc`: `0.4848`
- `oTOV`: `0.5778`

Current key result:

- the split-sum estimate beats the direct total-turnover fit on the common modeled sample
- `pred_oTOV_bp + pred_oTOV_sc`: weighted `R^2 = 0.6162`
- direct `pred_oTOV`: weighted `R^2 = 0.6109`

Current interpretation:

- if the goal is predictive turnover priors, scoring-turnover and bad-pass-turnover lanes should be estimated separately when possible
- if the goal is a simpler public framework, total-turnover burden can still be modeled compactly as `TOV_100 + load`

### Bad-pass prior takeaways

- bad-pass turnover volume alone is not enough to predict clean `oTOV_bp`
- a lean bad-pass-centered model can work, but it still needs burden and context
- the playtype-pressure block is the biggest additive layer beyond turnover counts

Current bad-pass ablation:

- full `oTOV_bp` model: `R^2 = 0.5216`
- remove the playtype block: `0.4570`

Practical implication:

- bad-pass RAPM is not just "passer sloppiness"
- it reflects turnover lane volume plus the offensive role and pressure environment where those passes happen

### Compact offensive-load framing

Current preferred framing for a simple turnover-value model:

`oTOV ~ turnover rate term + offensive load term`

Best single turnover-free load term paired with `TOV_100`:

- `creation_TSA100`

Current fixed-`TOV_100` benchmarks:

- `TOV_100 + creation_TSA100`: weighted CV `R^2 = 0.4876`
- `TOV_100 + offensive_load`: `0.4813`

Current compact burden build:

- add `PASSING_Potential Assists/100` -> `0.5216`
- add `transition_TSA100` -> `0.5440`
- add `shooting_TSA100` -> `0.5558`
- add `PASSING_on-ball-time%` in the compact version -> `0.5580`
- replacing `TOV_100` with `true_badpass_tovs_100 + scoring_tovs_100` nudges the compact model to `0.5598`

Current load-only term implied by the compact model:

`Load_v1_raw = 0.463*z(creation_TSA100) + 0.310*z(PASSING_Potential Assists/100) + 0.089*z(shooting_TSA100) + 0.081*z(transition_TSA100) - 0.138*z(PASSING_on-ball-time%)`

Interpretation:

- `creation_TSA100` is the main burden axis
- `Potential Assists/100` is the second burden axis
- shooting and transition matter, but less
- once creation and passing burden are already in the model, extra on-ball time reads more like style / efficiency tax than pure burden

### Current hub-creator caveat

The compact burden term is useful, but it is still too guard-biased to be treated as final "true offensive load."

Observed issue:

- heliocreator guards rank where expected
- hub big creators like Nikola Jokic get pushed too low because self-created scoring burden is overweighted relative to hub passing burden

Current best corrective candidate:

- `box_creation`

Add-one screen on top of the compact base:

- add `box_creation` -> `R^2 = 0.5670`
- add `PASSING_AVG_DRIB_PER_TOUCH` -> `0.5619`
- add `PASSING_AVG_SEC_PER_TOUCH` -> `0.5606`
- add `PASSING_on-ball-time%` -> `0.5580`
- add `PASSING_AST PTSCreated/100` -> `0.5580`

Current takeaways:

- raw touch volume is too blunt
- touch style helps some
- a hub-creation proxy helps more
- `PASSING_PotAss/Passes` and `touch_burden` are not useful enough in the current compact build

## Preferred Explanation Stack

Preferred site-level framing:

- `Shot Efficiency`
- `Turnover Drag`
- `Own Scoring Value`
- `Teammate PPP RAPM`
- `Total Offensive Value`

Preferred identities:

- `Own Scoring Value = Shot Efficiency - Turnover Drag`
- `Total Offensive Value = Own Scoring Value + Teammate PPP RAPM`

Current exact accounting layer:

- `playtype_TS_rPPP + playtype_TOV_rPPP = playtype_rPPP`

Current display bridge under discussion:

- `PPP_eq = playtype_adj_rPPP * 50`
- `TOV_value = PPP_eq - pt_adj_rTS`
- so `pt_adj_rTS + TOV_value = PPP_eq`

Interpretation rules:

- `Turnover Drag` is the top-line playtype turnover term.
- The final teammate-offense / creation term should be impact-based.
- Do not treat assist-box stats as the final definition of playmaking once a RAPM term exists.
- `playtype_diff` is the bridge from `rTSPct` to a playtype-adjusted TS framing.
- Do not add `playtype_diff` back onto `rPPP`; `rPPP` is already playtype-relative.
- Do not confuse the exact `playtype_TOV_rPPP` accounting term with the `pt_adj_rTS`-anchored `TOV_value` bridge.

Current strong scoring-prior result:

- the `pt_adj_ts_add` family outperformed the plain `ts_add` family against both `actual_oTS` and `hybrid_oTS`
- lower baselines performed better than `0`
- current best simple variant is `pt_adj_ts_add_m3`
- with ultra-simple `2021-2026` regular-season decay, `pt_adj_ts_add_m3` reached about:
  - `R^2 = 0.496` vs `actual_oTS` at `500+` current minutes
  - `R^2 = 0.570` vs `hybrid_oTS` at `500+` current minutes
  - `R^2 = 0.561` vs `actual_oTS` at `1000+` current minutes
  - `R^2 = 0.607` vs `hybrid_oTS` at `1000+` current minutes

Updated scoring-prior direction:

- deeper negative starts kept helping after `m3`
- the playtype-adjusted family flattened around `m7` to `m9`
- adding decayed `PASSING_AtRimAssists/100` pulled the best baseline back toward `m5` / `m6`
- adding decayed `3PR` provided a small additional bump

Current compact favorite from this branch:

- `pt_adj_ts_add_m5 + decayed PASSING_AtRimAssists/100 + decayed 3PR`

Current caution:

- this is now partially a separate `oTS`-prior thread, not the same task as the playtype turnover contamination build
- document and treat it as a parallel track

## March 14, 2026: `zTS` and the clean `oTS` scoring target

### Naming and interpretation

Current preferred public naming:

- `rTS` = raw scoring efficiency relative to league average
- `zTS` = playtype-adjusted scoring efficiency

Current bridge:

`zTS = rTSPct + playtype_diff`

Public logic:

- `FG%` sees makes and misses
- `eFG%` adds shot value
- `TS%` adds free throws
- `rTS` adds league context
- `zTS` adds scoring-role difficulty

### Current clean target

For the current `zTS` work, use:

- pure `2022-2026` regular-season `oTS`
- no playoffs
- no time decay
- no rubberband
- weighted by `5-year offposs`

Important framing:

- `oTS` is the offensive true-shooting factor from the six-factor RAPM decomposition
- it is a lineup-based, ridge-regularized estimate of how much a player changes team points per shot
- it is a cleaner target for a scoring stat than total `ORAPM`, which also mixes in turnover and offensive-rebounding impact

### The add concept

- an `add` stat is a rate stat turned into a value stat
- its job is to combine volume and efficiency on one scale
- the ideal scoring stat would let `usage x relative efficiency` get you close to scoring value
- raw `rTS` fails at that when role changes get misread as efficiency changes
- `zTS` is the current preferred fix to the efficiency term before multiplying by volume

### Current one-feature ladder result

At a `0` baseline against the clean `oTS` target, weighted `R^2` on the `5000+ offposs` sample:

| Stat | `R^2` |
| --- | ---: |
| `FG add` | `0.050` |
| `eFG add` | `0.098` |
| `TS add` | `0.289` |
| `zTS add` | `0.554` |

Interpretation:

- `zTS` is not a small refinement on `TS`
- on this task, the jump from `TS` to `zTS` is of the same order as the historic `FG -> eFG -> TS` upgrades

### Baseline sweep result

Weighted full-sample correlations on `5000+ offposs`:

| Start | TS add `r` | Positional TS add `r` | zTS add `r` | Positional zTS add `r` |
| --- | ---: | ---: | ---: | ---: |
| `+2` | `0.459` | `0.523` | `0.694` | `0.674` |
| `+1` | `0.499` | `0.567` | `0.722` | `0.704` |
| `+0` | `0.537` | `0.607` | `0.744` | `0.729` |
| `-1` | `0.572` | `0.642` | `0.761` | `0.749` |
| `-2` | `0.603` | `0.672` | `0.774` | `0.764` |
| `-3` | `0.630` | `0.696` | `0.784` | `0.775` |
| `-4` | `0.654` | `0.716` | `0.790` | `0.782` |
| `-5` | `0.673` | `0.732` | `0.793` | `0.787` |
| `-6` | `0.689` | `0.743` | `0.794` | `0.789` |
| `-7` | `0.702` | `0.751` | `0.794` | `0.789` |
| `-8` | `0.712` | `0.757` | `0.792` | `0.788` |
| `-9` | `0.720` | `0.760` | `0.790` | `0.786` |
| `-10` | `0.725` | `0.761` | `0.786` | `0.783` |

Main conclusions:

- every family improves once the baseline moves below `0`
- plain `TS add` needs a much deeper negative baseline
- `zTS add` peaks around `-6` / `-7`
- positional `zTS add` never beats plain `zTS add`
- `zTS add @ 0` already beats `TS add` at every tested baseline

### Decompressed model comparison

On the matched pure `22-26 RS` sample:

| Sample | `Pts + TSA` `R^2` | `Pts + ptdiff pts + TSA` `R^2` | Best `TS add` `R^2` | Best `zTS add` `R^2` |
| --- | ---: | ---: | ---: | ---: |
| All | `0.536` | `0.618` | `0.525` | `0.614` |
| `5000+` | `0.547` | `0.632` | `0.536` | `0.631` |
| `10000+` | `0.524` | `0.615` | `0.512` | `0.615` |
| `15000+` | `0.560` | `0.638` | `0.549` | `0.642` |

Interpretation:

- `Pts + TSA` and plain `TS add` are basically the same family
- `Pts + ptdiff pts + TSA` and `zTS add` are basically the same family
- the decompressed predictive model is only marginally better than the one-number `zTS add` version
- because the predictive gap is tiny, the current preferred public stat remains the simpler `zTS add`

### Current fixed-volume ladder

With a `-5` baseline:

`zTS_add_m5 = 2 * TSA100 * (zTS + 5) / 100`

Current league-average weighted means on the matched player sample:

- average `TS add -5`: `2.10`
- average `zTS add -5`: `2.14`

Current preferred fixed-volume conversion:

`zTS@20 = (TSA100 / 20) * (zTS + 5) - 5`

Meaning:

- this converts burden into `zTS`-equivalent points on a fixed `20 TSA100` ladder
- once converted, players can be compared on one scale without separately reading volume
- `zTS@20` is the current preferred public ladder if a fixed-volume display is needed

## What Not To Force

- Do not force a strict `rTS + sTOV + pTOV = playtype rPPP` identity.
- Playtype `PPP` and `TOV%` contain some passing-turnover contamination and are not clean enough to treat that identity as observed truth.
- Do not define public `Playmaking` as a residual from own playtype `rPPP`.
- Do not use `DiscountedAssistPPP` minus league average as the final public playmaking metric. It is acceptable only as a provisional descriptive proxy.
- Do not treat raw rolled-up `playtype_rTOV%` as the final creator-fair turnover stat. It is still useful, but not yet a clean scoring-vs-passing split.

## Relationship To Priors

The playtype / box layer is useful not only for display but also for future prior construction.

- descriptive own-scoring and turnover signals can inform offensive priors
- teammate-offense RAPM can anchor the playmaking / teammate-lift term
- the longer-term goal is a structural offensive prior system rather than a single ad hoc site stat

## Next Model Brief

There are now two immediate next modeling paths:

1. estimate how much passing-turnover contamination sits inside playtype `PPP` and playtype `TOV%`
2. test predictive expected-TS priors while keeping playtype difficulty

Related `oTS` prior question:

- whether current `box_oTS` should be trusted more aggressively
- and whether stronger shrinkage should happen post-solve (`hybrid`) or inside the solve (`true-prior`)

Current answer:

- hybrid only wants modest shrinkage
- current true-prior `oTS` does not beat pure `oTS`
- future-window prediction is the only real standard for deciding otherwise

Expected-TS direction:

- use EPM / DARKO expected TS%
- convert to relative TS
- keep `playtype_diff`
- test `expected_pt_adj_rTS` and its TS-add family as the next scoring prior

Contamination direction:

Current recommended path:

1. build a player-level contamination model
   - candidate features:
     - playtype mix
     - playtype turnover mass
     - `PASSING_Potential Assists/100`
     - `PASSING_on-ball-time%`
     - bad-pass turnovers
     - scoring turnovers
2. build a within-player playtype allocation model
   - allocate estimated passing-turnover mass across playtypes

Goal:

- separate scoring-turnover drag from passing-turnover drag more honestly
- improve creator fairness without a hand-wavy “grace” term
- create better priors for future turnover RAPM splits

## Repo Source

See [offensive-value-decomposition-notes.md](/Users/russellthomas/Docs/2026_NBA_PIPELINE/docs/offensive-value-decomposition-notes.md) for the fuller repo note and the empirical turnover coverage findings.

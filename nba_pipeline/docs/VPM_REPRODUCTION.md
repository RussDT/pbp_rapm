# VPM Reproduction and Recovery Status

October15 follow-up: the [new SQ temporal repair and paired rerun](VPM_SQ_TEMPORAL_REPAIR.md)
is complete in sibling v39/v40. The original artifacts described here remain
preserved; their identified leakage warning is not retroactively erased.

The October 15 field benchmark adds the seven LOKI-style preseason game/season
comparisons for the monthly own-SQ history (v37 harness config), against EPM
and lagged annual DARKO/LEBRON/MAMBA. See [protocol, results and limitations](VPM_FIELD_BENCHMARK.md).
This does not change the original model/target recipes below.

This document records the reproducible VPM foundation recovered on
2026-07-23. It is deliberately separate from both the forward Net DECOMP build
and the gross compatibility DECOMP build.

## Predictive north star

The long-term objective is to beat standalone EPM on frozen, date-correct
future NBA game prediction. The current future actual-possession harness is an
intermediate scientific gate, not the final product claim. A fair player-metric
comparison must use identical games and shared availability/minute assumptions,
predeclare its game-level scoring rule, and report paired game uncertainty.
Evaluate lineup synergy and each model's complete availability/forecasting
system separately so those layers cannot conceal a weaker player estimate.

## Model-family boundary

- **Selected forward NBA factor family:** `weighted_factors_decomp_net_*`;
  gross `weighted_factors_decomp_*` remains compatibility/rollback during the
  1Y-5Y migration.
  It is the exact eight-component actual-points decomposition documented in
  [Public Eight-Component DECOMP](./PUBLIC_DECOMP.md).
- **Legacy/audit families:** `weighted_factors_alt3_efg_value_*` and
  `weighted_factors_alt3_*`. They are useful recovery evidence, not the active
  public interpretation.
- **VPM research lane:** an EV-target evolution of stabilized DECOMP. Its
  canonical research package is the private sibling checkout
  `/Users/russ/Documents/Codex/basketball/nba/research/vpm`. VPM is not a
  renamed Alt3 export and is not currently a public production artifact.

## Target formulation recovered

Public DECOMP uses five first-chance shot children:

```text
DECOMP_RIM_FREQ
+ DECOMP_RIM_FG
+ DECOMP_MID_FG
+ DECOMP_THREE_FREQ
+ DECOMP_THREE_FG
= actual first-chance EFG points
```

VPM keeps rim targets unchanged. For the chronological first FGA of a
possession, it replaces midrange and three-point make/miss value with a
walk-forward per-shot EV:

```text
residual = actual_points - expected_points
EV_make_target = original_make_target - residual
```

The two residuals use already registered research aliases:

- three-point residual:
  `DECOMP_EFG` / `FC_DECOMP_EFG_Diff`;
- midrange residual:
  `DECOMP_MID_VALUE` / `FC_DECOMP_MID_VALUE_Diff`.

Thus the VPM first-chance solve has nine components:

```text
five EV-adjusted shot children
+ ALT_FT
+ ALT_TOV_VALUE
+ three-point actual-minus-EV residual
+ midrange actual-minus-EV residual
= FIRST_CHANCE actual points
```

Adding `SECOND_CHANCE_CLEAN` gives ten components that still target actual
possession scoring. EV de-noises the training target; forward evaluation must
still be against actual points.

The private season-out VPM configuration used 2021-2026 inputs, TD550, and
component-specific offense/defense ridge alphas. Its held-out validation and
prior recipes remain research evidence; they are not re-selected by the
target builder added here.

## Event-enrichment ownership boundary

The cross-repo contract is now explicit:

- `nba_pipeline` owns canonical versioned event/enrichment views;
- RAPM research owns shot-clock SQ, second-chance opportunity, defender-role,
  state-value, and event-credit models;
- VPM does not ingest raw event rows, provider defender joins, fold-level
  predictions, or state/event attribution rows;
- VPM may consume only stabilized player-season summaries after reliability
  and incremental-target tests.

For the current shot-clock/possession-quality work, possible future VPM priors
are stabilized player-season grenade rate, possession grenade burden per
attempt, and clock-adjusted shooter-talent residual. The shot-level OOF table,
reset provenance, player-zone OREB predictions, and all raw defender/event
assignments remain RAPM overlays.

Canonical integration plan:
`/Users/russ/Documents/Codex/basketball/shared/NBA_EVENT_ENRICHMENT_PLAN.md`.
Current clock experiment:
[Shot-Clock SQ and Grenade Burden](./SQ_SHOT_CLOCK_CHALLENGER.md).

## Recovered local data

### Present

- `raw_data/`: 60 RS/PS parquets covering 1997-2026 (`433M`).
- `processed/`: `2.3G`; the `RAPM`, `FIRST_CHANCE`,
  `FIRST_CHANCE_CLEAN`, and `SECOND_CHANCE_CLEAN` families each have all
  60 RS/PS files.
- repo-owned shot quality:
  `results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet` plus its
  calibration/talent summaries (`844M`, 6,327,218 unique shots per the
  canonical PBP shot-quality doc).
- private VPM EV intermediates:
  - `russell_sq_v2_ev_16_22.parquet`: 1,008,038 mid/three shots;
  - `russell_sq_v2_ev_21_26.parquet`: 892,628 mid/three shots;
  - `oof_ev_24_26.parquet`: 445,698 shots;
  - career talent seed parquets.
- private licensed audit table:
  `sq_shots_keyed.parquet`, 697,925 rows. It must remain in the private
  repository and must not be copied into this public production tree.
- active DECOMP recovery artifact:
  `master_results/weighted_factors_decomp_22_26_all_rb_se_a2000_4000`
  in CSV and parquet form (97-column schema).
- legacy Alt3 recovery artifacts:
  11 `weighted_factors_alt3_efg_value_*` parquet files under the extracted
  `rapms_master_results/` handoff, plus the local 2022-26 legacy Alt3 CSV.
- private published VPM/Alt3 snapshots, configs, evidence, and validation
  outputs in the sibling VPM checkout.

### Missing or incomplete

- No external ShotQuality source CSV directory survived.
- Raw 2024 and 2025 parquets do not contain `initial_ev` or
  `sq_descriptor_bundle`. Only 2026 RS/PS is locally enriched.
- No `all_shots_with_xfg*.parquet` or xFG summary survived. Only
  `scripts/fix_xfg_parquet.py` remains.
- No local `results/transition_classifier/` artifact survived.
- The original historical/monthly `cache/epm/` grid did not survive, but the
  required research grid has now been repulled: 80 checksummed game-agnostic
  snapshots from 2014-06-30 through 2026-07-23. Each manifest preserves the
  requested date and provider-effective date separately. Annual historical
  snapshots cover every forward-label cut and denser snapshots cover current
  evaluation/emission dates.
- The exact `results/three_priors/` as-of-date 12-prior files referenced by
  the private VPM replication guide did not survive in either checkout.
  Other prior research outputs do exist, but they are not drop-in substitutes.
- Only the 2022-26 gross compatibility DECOMP rolling artifact is present locally;
  the current 1Y-4Y public DECOMP rolling files need rebuilding.

These gaps block a from-scratch rebuild of the historical licensed
ShotQuality comparisons, the original xFG lane, and the exact old per-cut
EPM-prior files. They no longer block a clean EPM12 rebuild: the recovered
recipes can now be remapped to mature forward labels from the checksummed
as-of snapshot grid. They do **not** block reproduction of the shipped
no-prior EV target layer because its walk-forward EV parquets survived.

## EPM prediction inputs

Dunks & Threes exposes game-agnostic, date-aware predictive rates including:

- `p_fg3pct` and `p_fg3a_100`;
- `p_fgpct_mid` and `p_fga_mid_100`;
- `p_fgpct_rim`, `p_ftpct`, and other estimated skills;
- `oepm`, `depm`, and `epm`.

Only the `p_*` fields are eligible prior features. `eWINS`/`ewins` is
explicitly excluded because it is a cumulative EPM impact output rather than
a predicted box-score skill; `oepm`, `depm`, and total `epm` remain
benchmark/blend fields only.

The provider contract says values requested for date D exclude games played
on D and update overnight. Cache game-agnostic mode (`game_optimized=0`) for
portable player-skill priors:

```bash
set -a
source .env
set +a
.venv/bin/python nba_pipeline/scripts/fetch_epm_data.py \
  --dates 2025-11-01 2025-12-01 2026-01-01
```

Generated CSVs and manifests live under `nba_pipeline/cache/epm/` and remain
gitignored. The fetcher reads `EPM_API_KEY`, never writes credentials, refuses
silent overwrite, validates player identity/schema/mode, and records the
provider effective date separately from the requested prediction date.

There are two distinct prediction uses:

1. **Shot expectation:** use `p_fg3pct` or `p_fgpct_mid` as an as-of shooter
   talent anchor, then let repo-owned shot context and clock describe the
   attempt. Do not put shooter skill into the context-only SQ field.
2. **Possession/lineup prediction:** map EPM skills to stabilized component
   priors or compare/blend complete EPM with VPM in prediction space.

The surviving private evidence makes the second use especially promising.
Across nine pre-registered 2025-26 tiles, gain versus vanilla was `+270` for
VPM, `+401` for EPM, and `+542` for the fixed 50/50 VPM-EPM blend; the
game-clustered interval for blend minus VPM was `[+121, +438]`. The old
shot-level EPM-feature A/B was positive but much smaller: attempt-weighted
log loss improved by only `0.000037`, with wins in four of six zone-season
cells. Therefore EPM is not automatically the SQ answer; every feature/prior
still needs a fresh-date incremental gate.

Raw daily snapshots remain research inputs. Under the cross-repo ownership
contract, VPM may consume only stabilized player-season prior outputs after
reliability and incremental-target tests, not provider rows attached directly
to shots or possessions. None of this changes the active public
`weighted_factors_decomp_*` family; Alt3 remains legacy/audit.

## Reproducible target builder

`scripts/build_vpm_target_swap.py` is an argument-driven, production-side
replacement for the private hardcoded attachment scripts. It:

- writes only to an isolated output directory;
- never edits the canonical `processed/FIRST_CHANCE*.parquet` files;
- uses clock-first raw ordering before assigning the first FGA to a terminal
  possession, so corrected late-number events are handled consistently with
  current PBP rules;
- rejects conflicting EV keys and zone mismatches;
- preserves the original parent/reserved columns in `VPM_ORIGINAL_*` audit
  fields;
- verifies row-level EV-target plus residual closure before writing;
- can add relative symlinks to unchanged `RAPM` and
  `SECOND_CHANCE_CLEAN` files for downstream research loaders.

The preferred collaborator command now runs from the sibling VPM research
repo and orchestrates both frozen EV sources through this adapter:

```bash
python vpm/run_harness.py \
  --config vpm/configs/research_harness_v3_epm12.json \
  prepare-targets
```

It discovers this checkout through the sibling layout, `PBP_RAPM_ROOT`, or
`--pipeline-root`, combines the 2016-2020 and 2021-2026 reports, and writes a
checksummed `results/vpm_target_swap/target_manifest.json`. Source/year routing
lives in the VPM-side `vpm/configs/vpm_target_sources.json`, separate from
immutable experiment configs. Direct adapter use remains available for
pipeline development and debugging. The completed local
2021-2026 adapter command was:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/build_vpm_target_swap.py \
  --ev-parquet \
  /Users/russ/Documents/Codex/basketball/nba/research/vpm/vpm/data/russell_sq_v2_ev_21_26.parquet \
  --years 2021-2026 \
  --season-types all \
  --output-dir nba_pipeline/results/vpm_target_swap/processed \
  --report nba_pipeline/results/vpm_target_swap/attach_report.csv \
  --link-unchanged-families
```

Generated outputs remain gitignored research artifacts.

## 1997-2015 Russell v2 historical bridge

`scripts/build_russell_v2_historical_bridge.py` now supplies a separately
labeled, outcome-safe historical EV source for a 1997-2026 no-prior VPM. It
does not overwrite the frozen native-v2 files used from 2016 onward.

The chronological native-overlap gate selects the calibrated v2-distilled
stack for midrange and rejects it for threes, where the portable-v3 fallback
remains better on held-out 2023-2026 outcomes. The resulting 1997-2015 parquet
contains 2,583,850 unique shots over 23,851 games and closes
`ev_pts = p_make * shot_value` exactly. Historical safe-crossfit probabilities
remain audit-only because fold predictions are not a production VPM target.

The 2015/2016 handoff is not level-seamless: mean probability changes by
`+0.024857` for midrange and `-0.007360` for threes, versus actual-rate changes
of `+0.005920` and `+0.004126`. A combined 1997-2026 VPM must therefore use
fixed season effects or an explicitly audited equivalent season-centering
treatment. Do not erase the seam with outcome-informed post-hoc calibration.

Canonical methods and receipts:
[Russell ShotQuality v2 Historical Bridge](./RUSSELL_V2_HISTORICAL_BRIDGE.md).
The generated EV parquet is:

```text
nba_pipeline/results/historical_shot_quality/russell_v2_historical_bridge/
russell_v2_historical_bridge_ev_1997_2015.parquet
```

The bridge makes target preparation possible; it does not itself run or
validate the full historical VPM. In a no-prior run, player prior means are
zero while the registered ridge penalties and ten-component closure remain.

## Current validation

The 2021-2026 RS+PS build produced:

- 1,530,670 `FIRST_CHANCE` rows;
- 810,740 rows with a midrange or three-point EV attached;
- midrange:
  - 341,421 chronological first-chance FGAs;
  - 341,016 identity-gated rows (99.881%);
  - 313,660 EV attachments (91.978% of identity-gated rows);
- three-point:
  - 497,087 chronological first-chance FGAs;
  - 497,080 identity-gated rows (99.999%);
  - 497,080 EV attachments (100.000% of identity-gated rows);
- maximum per-file shot-target closure gap:
  `8.88e-16`;
- combined nine-component first-chance solver-target closure:
  1,530,670 rows, maximum absolute gap `5.33e-14`,
  zero failures above `1e-10`.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  nba_pipeline/tests/test_vpm_target_swap.py \
  nba_pipeline/tests/test_stabilized_target_helpers.py \
  nba_pipeline/tests/test_decomp_shot_tree.py
```

Expected result: `11 passed`.

## Hardened research harness

The private sibling checkout now has an argument-driven control plane at:

```text
/Users/russ/Documents/Codex/basketball/nba/research/vpm/vpm/run_harness.py
```

Its checked-in contract is
`vpm/configs/research_harness_v1.json`; operator documentation is
`vpm/docs/RESEARCH_HARNESS.md`. The harness discovers this production checkout
through `PBP_RAPM_ROOT`, `--pipeline-root`, or the sibling workspace layout, so
it does not restore the old hardcoded `/Users/russellthomas/Docs/...` paths.

The isolated EV target lane was extended through 2016-2020 from the surviving
walk-forward historical EV parquet. Combined 2016-2026 RS+PS validation now
covers `2,781,191` `FIRST_CHANCE` rows with maximum EV-adjusted shot-tree gap
`5.35e-14` and summed gap `-4.77e-12`. Generated parquets and harness outputs
remain gitignored.

The harness provides:

- a fail-fast data doctor with per-family coverage, EPM manifest/checksum, data
  grain, and full row-closure checks;
- uniform 365-day forward component labels;
- completed-season-only, possession-weighted, time-decayed empirical-Bayes
  skill estimates;
- compact component/side prior recipes that exclude complete EPM totals;
- label-maturity checks before every prior emission;
- expanding-window season-forward diagnostics for every prior mapper;
- summed future actual-possession evaluation against vanilla RAPM and as-of
  EPM on identical rows;
- fixed VPM/EPM prediction blends;
- protected holdout states and an explicit override for spent/virgin cuts;
- paired game-clustered bootstrap artifacts versus both vanilla RAPM and the
  EV no-prior model;
- automatic development-only promotion gates and immutable resolved-config
  locks;
- a frozen v2 returning-player challenger combining TD550, TD1500, and the
  skill-prior anchor.

Raw shot/provider rows, shot-level OOF predictions, grenade outputs without a
multi-season forward backfill, and retrospective role-swap outputs are blocked
from direct prior use. The nine 2025-26 tiles are recorded as spent
confirmation evidence and cannot select the next specification.

The completed development result uses four 90-day cuts in 2024 and 2025
(2,115 games; 414,776 possessions), equally weighted by cut:

- EV no-prior: `+297.88e-6` versus plain RAPM, clustered 95% interval
  `[+185.62, +402.52]`;
- compact x2 skill priors: `+8.24e-6` beyond EV no-prior, four wins in four
  cuts, interval `[+2.79, +13.62]`;
- complete EPM benchmark: `+239.88e-6` versus plain RAPM;
- fixed VPM/EPM 50:50 ensemble: `+505.69e-6` versus plain RAPM;
- frozen TD550/TD1500 absence rule: rejected for standalone VPM because it
  won only two of four cuts and its no-prior interval crossed zero;
- absence-adjusted VPM/EPM ensemble: retained only as an external ensemble
  challenger; it improved the ordinary 50:50 blend by `+30.84e-6`, interval
  `[+5.76, +57.56]`.

Standalone VPM's point estimate is `+66.32e-6` better than EPM on these cuts,
but the paired interval is `[-142.71, +274.85]`. That is competitive/parity
evidence, not a demonstrated EPM win. No 2025-26 spent confirmation tile was
used for selection.

### EPM12 predictive-rate development base

`vpm/configs/research_harness_v3_epm12.json` rebuilds the historical 12-prior
idea inside the mature-label harness. Every mapper uses only the latest
checksummed game-agnostic snapshot requested on or before its label/emission
date and only D&T predictive `p_*` rates. Complete EPM and eWINS are blocked by
config validation.

On the same four development cuts:

- EPM12 x1: `+109.03e-6` beyond EV no-prior, 4/4 cuts, paired
  game-clustered 95% interval `[+69.15,+149.43]`;
- EPM12 x2: `+153.02e-6` beyond EV no-prior, 4/4 cuts, interval
  `[+73.38,+233.87]`;
- EPM12 x2 versus accepted box-prior x2: `+144.70e-6`, interval
  `[+67.83,+223.85]`;
- EPM12 x2 versus complete-EPM point predictions: `+211.02e-6`, interval
  `[+9.17,+405.32]`.

This makes EPM12 x2 the leading research base for residual stacking. The public
reader now displays this research board so its player estimates and additive
components can be inspected directly; that display does not reopen the nine
spent 2025-26 tiles or count as untouched confirmation. The next promotion
receipt remains the untouched 2026-27 cut.
The equal-cut quadratic optimum along the exact affine prior-scale path is
`2.1764`; x2 is retained as the simpler frozen value because it is within
`1.02e-6` of that fitted optimum and has the stronger worst-cut result.

The 2026-07-24 research board uses the 2026-07-23 requested snapshot
(provider-effective through 2026-06-13), covers all 582 current PBP players,
and closes all ten-component, three-factor, and net identities within
`8.88e-16`:

- `../vpm/vpm/results/harness/vpm_hardened_dev_v3_epm12/player_estimates_epm12_x2_2026-07-24.csv`;
- matching `.manifest.json`.

### EPM12 direct-parent simplicity baseline

The sibling immutable config
`vpm/configs/research_harness_v12_epm12_parent_baseline.json` tests whether
the component architecture adds predictive value beyond a strong parent. It
maps the exact side-wise union of the EPM12 date-safe `p_*` fields directly to
matured 365-day future pure O/D RAPM labels and fits one prior-centred parent
solve. Development is restricted to 2020-2022; 2023-2026 is frozen historical
stress and 2027 remains untouched.

The one-SE development choice is mapper alpha `0.1`, offense/defense solve
alphas `4000/8000`, and prior scale `1.5`. On 406,110 common 2023-2026 stress
possessions across 4,181 games:

- parent versus plain RAPM: `+254.11e-6 [171.85,336.95]`;
- parent versus complete EPM: `+386.16e-6 [241.80,531.66]`;
- selected decomposed VPM versus parent: `+41.55e-6 [-82.46,165.29]`;
- parent versus the selected 25% post-solve pull: `+53.80e-6 [1.43,105.91]`.

The parent preserves about 86% of selected VPM's gain over plain RAPM. Thus
the strong prior and shrinkage do most of the predictive work; the remaining
component gain is unresolved rather than disproven. Keep the decomposed
incumbent pending untouched confirmation, reject the extra post-solve pull,
and require future component challengers to beat this parent as well as RAPM.
Canonical details and commands live in the sibling
`vpm/docs/EPM12_PARENT_BASELINE.md`.

### EPM12 counting-rate correction

The original shooting recipes combined raw predictive percentage and attempt
rate only as separate additive mapper columns. The corrected immutable sibling
configs are `research_harness_v13_epm12_makes_netcont.json` and
`research_harness_v14_epm12_makes_parent_baseline.json`. They derive
`3FGM/100`, rim `FGM/100`, midrange `FGM/100`, and `FTM/100` as exact
attempt-rate times predictive-percentage products, then pair makes with
attempts.

Coefficients remain learned as-of each cut from fully matured 365-day labels:
minimum 500 future possessions, `min(possessions,5000)` player weights,
equalized label-cut weight, weighted feature standardization, and standardized
ridge/OLS. The generated `prior_models.json` is the coefficient receipt.

On the identical 406,110-possession 2023-2026 stress universe, corrected
decomposition is only `+13.95e-6 [-116.41,+140.69]` better than its parent.
The corrected parent beats RAPM by `+259.66e-6 [+177.61,+342.11]` and the
post-solve blend by `+58.64e-6 [+5.58,+111.32]`. Corrected decomposition loses
to the old misspecified lane by `22.05e-6 [4.32,39.69]` on spent stress, so
neither representation may be newly selected from that comparison. Preserve
the public board and use 2027 as the protected test of the corrected lane.
Canonical sibling receipt: `vpm/docs/EPM12_COUNTING_RATE_CORRECTION.md`.

### Preseason position-allocation audit

The sibling VPM command `run_harness.py offseason-position-audit` audits the
preseason alpha lane without creating another selection surface. For each
target season it assigns returning players their latest listed `Pos2` strictly
before the preseason snapshot, rebuilds the exact realized lineups behind the
saved evaluation rows, and groups held-out errors by offensive and defensive
lineup size. The primary buckets count listed centers (`0`, `1`, `2+`); a
frontcourt sensitivity counts listed PF/C players (`0-1`, `2`, `3+`). Scores
use equal season weight and paired game-clustered bootstrap intervals.

The report-only 2023-2026 stress sample contains 404,986 position-complete
possessions across 4,198 games. Selected x2 versus selected no-prior produces:

- defensive zero-center: `+330.92e-6`, 95% interval
  `[+130.59,+529.94]`;
- defensive one-center: `+51.09e-6`, interval `[-26.98,+129.75]`;
- defensive two-plus-center: `-146.07e-6`, interval
  `[-419.04,+112.87]`, with a negative point estimate in all four seasons;
- defensive three-plus PF/C sensitivity: `-127.30e-6`, interval
  `[-296.91,+43.96]`;
- offensive two-plus-center falsification: `+282.36e-6`, interval
  `[+13.62,+537.12]`.

The current selected-alpha inspection board is also less center-skewed than
the same-date v4 board: listed-center mean defensive VPM falls from `+1.360`
to `+1.029`, while mean center net VPM falls by `0.430`. The selected prior
still carries a strong mechanism-level positional tilt (listed-center mean
defensive prior `+0.932` versus point-guard `-0.374`), led by rim protection
and second chance; dTOV partially offsets it.

The correct interpretation is narrow: the prior is useful overall and helps
perimeter-heavy defenses, but appears to over-credit jumbo defense. This is
not evidence that every center is overrated, and the selected alphas already
mitigate the problem relative to v4. Never recenter positions or tune a
correction on the spent 2023-2026 rows. Any correction candidate belongs in
2019-2022 development and must be frozen before the untouched 2027 receipt.
The canonical implementation and full report live in the sibling VPM repo at
`vpm/harness/position_audit.py` and
`vpm/docs/OFFSEASON_POSITION_BIAS_AUDIT.md`.

### Defensive prior scale and allocation follow-up

The sibling `offseason-prior-scale` lane holds offense at x2 and varies only
the defensive prior-center multiplier. On 2020-2022 development, x1.5 is the
empirical best and the one-standard-error lowest-scale rule selects x1; x1
through x2 are all statistically tied. In the frozen 2023-2026 historical
replay, however, x1 loses to x2 by `-17.75e-6 [-32.86,-2.37]` overall. The
position slices show why: x1 improves the 2+ center and 3+ PF/C point estimates
but clearly hurts zero-center defenses.

The separate `offseason-defense-allocation` falsification keeps x2 and offense
unchanged, fits one defensive parent from `p_stl_100`, `p_blk_100`, and
`p_drb_100`, and projects the four existing defensive child priors to that
exact budget. Parent/child closure is exact, but the candidate loses to the
current EPM12 x2 row on 2020-2022 development by
`-7.72e-6 [-15.09,-0.33]`; it is rejected before stress.

Neither a global defense haircut nor a single player-level parent is the
proper fix. The evidence identifies lineup-dependent rim-protection
redundancy, which belongs in a separate interaction estimated jointly with
the additive player coefficients. Do not change the accepted board or open
2027. Full report:
`vpm/docs/DEFENSIVE_PRIOR_SCALE_AND_ALLOCATION.md` in the sibling repo.

The first joint interaction candidate uses pairwise overlap among positive
preseason-standardized `p_blk_100` signals as a separate block in all ten
component solves. It wins all three 2020-2022 seasons but fails the clustered
admission interval (`+6.31e-6 [-1.48,+14.12]`). Its sign is also wrong for the
diagnosed problem: it rewards overlap and shifts two-plus-center defenses
another `-0.18` points per 100 stronger, losing
`-91.68e-6 [-217.70,+38.29]` in that development bucket. The candidate is
rejected without opening stress.

Minutes and possessions remain reliability/availability candidates rather
than direct per-possession quality features. On the August 21 board, absolute
defensive-prior magnitude has `r=0.003` with accumulated defensive
possessions, so there is no support for a blanket exposure multiplier. The
canonical box panel has MPG and possessions but no height or wingspan.
Databallr's separate Supabase `measurements_master` table is a valid
`nba_id`-keyed source for no-shoes height and wingspan. The sibling legacy
`decomp_prior_harness.py --physical-features` lane already tested those fields
and wingspan-minus-height as direct player-prior features and rejected the
candidate at the possession gate (commit `b0ac33e`); its gitignored cached CSV
is currently absent. A lineup-length redundancy interaction remains a
different, untested specification and cannot be inferred from that direct
feature result.

### Matchup/BAM parent-prior audit

The sibling `offseason-constrained-priors` command recovers and tests the
regular-season BAM matchup archive without admitting raw matchup rows to VPM.
The legacy pair formula is held fixed; a lag-safe 2019-2022 grid selects only
exposure shrinkage, season decay, and history length. The one-SE
most-regularized row is `K=8000` partial possessions, a two-season maximum
window, and a 0.5-season half-life.

Against the recovered annual dRAPM sensitivity target, this specification has
a highly stable coefficient: standardized mean/SD `0.152/0.013` on 2019-2022
LOYO and `0.142/0.008` on the 2023-2025 expanding replay, with positive gains
in every year. That screen is mechanism evidence, not admission.

The decisive test maps positive-good BAM into the native negative-good defense
coefficient convention, centers it on common effective defensive exposure,
and introduces it only through an independent parent actual-points solve at
alphas 2000/4000. Adding that parent delta to selected EPM12 x2 yields:

- 2020-2022 development: `+6.84e-6 [-6.93,+20.05]`, only 2/3 positive;
- 2023-2026 all-lineup stress: `+1.44e-6 [-8.98,+11.97]`;
- returning-lineup stress: `-2.29e-6 [-18.75,+14.26]`;
- common-EPM stress: `-2.05e-6 [-18.19,+14.52]`.

The all-lineup effect reverses to `-15.72e-6` in 2026. The candidate therefore
fails the parent-information gate and the closed family stops before child
allocation. Keep TD-BAM as a benchmark/mechanism feature, not a VPM prior.
Future matchup work should separate assignment burden, opponent strength,
suppression, and turnover creation inside a jointly trained parent-defense
bank. The 2026 matchup source is incomplete, and 2027 remains both untouched
and source-blocked. The canonical report is the sibling
`vpm/docs/MATCHUP_BAM_PARENT_PRIOR.md`.

### Completed-season parent ORAPM/DRAPM priors

The sibling VPM command `build-parent-rapm-priors` implements that jointly
trained parent bank without changing public RAPM or the accepted VPM board.
Its primary labels are exact annual regular-season direct parent solves from
this repository:

```bash
.venv/bin/python nba_pipeline/scripts/rapm.py RAPM YY YY RS \
  --pure --rubberband --season-effects \
  --off-alpha 2000 --def-alpha 4000 --cores 2
```

The canonical target files are
`nba_pipeline/results/rapm_YY_rs_pure_rb_se_a2000_4000_results.csv`. Raw `off`
is positive-good offense; raw `def` is points allowed, so the sibling feature
model publishes positive-good DRAPM as `-def` and flips it back exactly once
for solver-native defense prior files. The repo's `pure` convention still uses
shooter-expected FT value where available.

The registered season-forward panel uses completed 2018 features for 2019
targets through completed 2022 features for 2023 targets, then emits a 2024
prior from completed 2023. Group selection requires 4/5 conditional MSE wins
plus pairwise coefficient-vector and emitted-contribution stability. Complete
EPM is benchmark-only. Matchup burden uses cross-fitted `p_*`-skill opponent
strength weighted by NBA partial possessions; legacy BAM and matchup outcome
suppression are forbidden candidate fields.

The selected development bank beats a lag-parent-plus-role ridge 5/5 on both
sides. Frozen replay on 2024-2026 also wins 3/3: mean parent-target MSE gains
are `+0.1625 [+0.0540,+0.2794]` offense and
`+0.0764 [+0.0509,+0.1028]` defense under a player-cluster bootstrap. The
isolated matchup-assignment increment is much smaller and inconclusive:
`+0.00301 [-0.00341,+0.00955]`, positive in 2/3 stress seasons. These intervals
do not include RAPM target-estimation covariance, and the bank has not cleared
a summed future-possession VPM gate. Status remains development-only.

Canonical sibling config/report:
`vpm/configs/research_harness_v8_parent_rapm_priors_2018_2023.json` and
`vpm/docs/PARENT_RAPM_PRIORS_2018_2023.md`. The 2026 matchup source remains
incomplete, so the command does not emit a 2027 prior.

### Accepted x2 player board

The harness now converts the frozen standalone v2 x2 candidate into an as-of
player board with `estimate-board`. The 2026-07-24 run used all aligned
2016-2026 data through 2026-06-13, the registered TD550/component alphas, and
completed-season skill priors at scale `2.0`. It did not use an EPM blend or
the rejected absence adjustment.

Its midrange and three-point EV targets are the frozen walk-forward
`RUSSELL_SHOTQUALITY v2` predictions in
`russell_sq_v2_ev_16_22.parquet` and `russell_sq_v2_ev_21_26.parquet`.
Those models use PBP shot context, strictly prior EPM shooter-skill snapshots,
and expanding repo-owned shooter talent. The current board has not been
rebuilt with the later rich-geometry ShotQuality challenger, historical
safe-crossfit, shot-clock/grenade, defender-role, or continuation models.

The generated CSV contains 1,556 players and 582 current players. Current
eligibility comes from latest-season PBP participation rather than box-panel
coverage; 122 current players lack a 2026 box-feature row and 56 lack any box
history, but no current player lacks a name. All ten-component, three-factor,
and net identities close within `8.88e-16`.

Artifacts:

- `../vpm/vpm/results/harness/vpm_hardened_dev_v2_board_2026-07-24/player_estimates_2026-07-24.csv`;
- matching `player_estimates_2026-07-24.manifest.json`.

Values are league-centered points per 100 possessions with positive defense
meaning good defense. The board is a production estimate from the accepted
candidate, not another validation cut and not new evidence that VPM has
conclusively beaten EPM.

### VPM Lab

The public interactive reader now shows the EPM12 x2 predictive research board:
[VPM Lab](https://vpm-lab-2026.russdt.chatgpt.site). Its source lives in the
nested `vpm-explorer/` site project. The product includes:

- the 582-player current board with a 1,556-player full-history toggle,
  constrained to a viewport-relative scroll region with a sticky header;
- search and sorting across VPM offense, defense, net, and all six point-valued
  possession-stage factors;
- a first-class ten-component architecture view and a `3 x 2` player factor
  hierarchy (`oTS/oTOV/oSC` over `dTS/dTOV/dSC`) integrated with the player
  header: identity connects directly to offense, defense, and total VPM, and
  each TS tile summarizes its eight exact children through four additive
  display rollups: Rim (rim frequency + rim FG), Mid (mid EV + mid residual),
  3P (three frequency + three EV + three residual), and FT value. TOV and SC
  remain direct one-component factors, and the full ten-component architecture
  remains available below the player view. Desktop compresses the connected
  player header, intro, side bars, factor cards, and inter-row spacing so both
  sides read as one screen without removing the explanatory copy. Mobile
  retains the same six-tile map in one compact frame and hides secondary prose.
  The default editorial card uses a warm-white field, black rules, bold Geist,
  and chartreuse only for aggregate VPM totals. An in-card White/Dark switch
  retains the original restrained blue/green dark treatment for direct
  comparison;
- a five-player additive study, including the Brown/James/Embiid/Edgecombe/
  Maxey preset, with both six-factor and ten-component sums;
- a separate catalog of non-additive RAPM/research drilldowns, including FT
  and TOV children, assist/scoring roles, transition, clock/grenade, defender,
  and rebound-opportunity layers; and
- a casual-first `/about` guide that defines **Impact Monster** as the ten-factor
  breakdown, RAPM as the parent impact total, and the `TOV/TS/SC` possession
  stages only as a friendly organizer. It shows the ten factor types on both
  offense and defense closing back to the corresponding RAPM side, with model
  mechanics and uncertainty kept in optional deeper reading; and
- an explicit model card covering frozen v2 shot EV, EPM12 predictive-prior
  coverage, alpha provenance, complete-EPM/eWINS boundaries, and second-chance
  turnover interpretation.

The separation is deliberate: the ten accepted VPM inputs close to the board
total, while the broader research drilldowns are mechanism evidence and
candidate prior material. They must not be stacked onto the current VPM total.
The TOV tiles are labeled explicitly as offensive and defensive turnover value
rather than relying on the less direct possession-survival/deletion language.

The site ships a frozen copy of the accepted CSV and manifest under
`public/data/`. Regenerate the harness board and deliberately refresh both
files before publishing a new site version; do not silently point the reader
at a mutable research artifact.

## Next reproducible steps

1. Keep the accepted compact x2 prior lane and its player-board recipe frozen;
   do not select against the spent 2025-26 tiles.
2. Keep the pure absence rule rejected. Treat the absence-adjusted EPM blend
   as an external ensemble challenger, not standalone VPM.
3. Use EPM12 x2 as the frozen research baseline for separately motivated
   residual prior layers. New layers must beat EPM12—not merely EV no-prior—
   on summed future actual points. Complete EPM stays benchmark/blend material;
   eWINS stays forbidden.
4. Build a multi-season forward grenade/clock-burden panel before admitting
   that source; the current 2026-only player summary remains blocked.
5. Freeze the winning development config, absence rule, and prior scale before
   the next untouched 2026-27 confirmation.
6. Build the missing 1Y-4Y forward Net DECOMP rolling artifacts
   independently of VPM:

```bash
PYTHONPATH=. .venv/bin/python nba_pipeline/scripts/run_decomp_rolling.py \
  --windows 2026,2025-2026,2024-2026,2023-2026
```

Do not use legacy Alt3 files as replacements for those active DECOMP outputs.

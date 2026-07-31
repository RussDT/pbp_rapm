# VPM Reproduction and Recovery Status

This document records the reproducible VPM foundation recovered on
2026-07-23. It is deliberately separate from the active public DECOMP build.

## Model-family boundary

- **Active public NBA factor family:** `weighted_factors_decomp_*`.
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
- Only the 2022-26 active public DECOMP rolling artifact is present locally;
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
6. Rebuild the missing 1Y-4Y active public DECOMP rolling artifacts
   independently of VPM:

```bash
PYTHONPATH=. .venv/bin/python nba_pipeline/scripts/run_decomp_rolling.py \
  --windows 2026,2025-2026,2024-2026,2023-2026
```

Do not use legacy Alt3 files as replacements for those active DECOMP outputs.

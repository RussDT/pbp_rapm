# WNBA Predictive Net-DECOMP VPM (Time-Decay Prior-Centered Research Lane)

Built August 26, 2026. Owner-sanctioned research lane: the WNBA Kalman
predictive panels (14 box stats + 8 estimated skills) become factor-specific
prior centers inside a multi-year time-decay prior-centered ridge on the exact
ten-target net-continuation DECOMP family, gated on forward actual-possession
prediction. Implementation: `wnba_test/build_wnba_predictive_net_vpm.py`.
Artifacts live under `wnba_test/Results/predictive_net_vpm/` (generated, not
in git).

**Status: development-only.** The live published WNBA VPM (single-season
box-prior gross lane, `5000/40000/1.5`) is untouched and remains canonical.
This lane is the explicit research exception to the standing "Kalman panels
are never rating priors" rule; that exception covers this lane only.
Promotion requires the untouched 2027 forward gate plus explicit owner
sign-off. No proprietary ratings (DPM/PI/NPI/EPM) appear anywhere as
features, targets, or tuning inputs.

## September 29 target audit

The older loader calls `FIRST_CHANCE.Net_Diff + SECOND_CHANCE_CLEAN.Net_Diff`
`_actual_points`, but these processed files carry expected, full-season FT
percentages. Consequently the historical MSE receipts below are **expected-FT
scoring evaluations**, not literal actual-point validation. The older Kalman
parameters also used 2023–2025 tuning for early-season replays. Preserve these
receipts as historical development evidence; do not compare their absolute MSE
directly with the repaired target lane.

The new sibling [WNBA predictive parent](./WNBA_PREDICTIVE_PARENT.md) uses an
isolated actual-FT rebuild, whole-game score reconciliation, and annually
refitted past-only skills. The published gross board is unchanged.

## Protocol

- **Targets, folds, scoring**: reused verbatim from
  `wnba_test/tune_wnba_daily_net_hyperparams.py` — the ten-target structural
  net family (`RIM_FREQ/RIM_FG/MID_FREQ/MID_FG/THREE_FREQ/THREE_FG/FT/TOV/
  NET_BASE/SECOND_CHANCE`) closing exactly to actual possession points;
  rolling-origin completed-season folds (train = all completed seasons before
  the validation season, RS+PS, 2022 onward); game-weighted future possession
  MSE with the tuner's paired one-SE machinery.
- **Labels** (for prior-model training): per-season single-season PURE
  zero-centered ten-target solves at the tuner-selected `2000/5000`,
  exposure-centered per side. 2026 labels are partial-season (RS to date) and
  are used only in mechanism reporting, never for selection.
- **Kalman as-of features**: each player's latest `post_rate` per box
  stat/skill strictly before the target season's first possession date
  (boundaries 2023-05-19 / 2024-05-14 / 2025-05-16 / 2026-05-08). 2022 is
  panel warm-up and never receives a Kalman prior. Missing skill states
  (about 1% of cells) are snapshot-mean imputed. The comparison box-snapshot
  arm uses the previous season's stabilized `load_box_features(S-1)` row, so
  both arms carry only information complete before the fold season.
- **Prior models**: compact factor-specific `StandardScaler + Ridge` models
  per (target, side) — Kalman maps mirror the box-prior lane's
  component-relevant philosophy on panel states; the box arm reuses
  `build_wnba_box_priors.feature_columns` through the gross-prefix map (with
  `NET_BASE -> ALT_FC_COMPLETION`). Training pairs are strictly year-forward
  (states as-of season T -> season-T labels, T < fold season). Ridge alpha in
  `{1, 10, 100, 1000}` chosen inside year-forward training folds only
  (fallback `100` when the fold has no interior year); predictions clip to
  the training labels' [q01, q99]. Selected alphas concentrate at 100
  (97/180 models) — the priors are deliberately blunt.
- **Solve**: prior insertion exactly as `clean_decomp_wnba` does it —
  `prior_scale` multiplies the prior, exposure-centering per side (weighted
  by decayed exposures), then `rhs += penalties * prior` — generalized to the
  multi-target time-decay gram system. One coefficient per player per
  component side over the window; every mapped player has a Kalman prior by
  construction (the panel covers 2022+), while the box arm covers 156-182 of
  206-288 mapped players and the rest center on zero.
- **Grid/selection**: base off/def alphas `{2000/5000, 5000/40000}` x
  half-life `{none, 1100d, 550d}` x prior scale `{1.0, 1.5}`, selected ONLY
  on the 2024 and 2025 forward folds with the tuner's paired one-SE rule
  (most-shrunk, then longest history, then smallest prior scale).

## Evidence accounting

Folds 2024 and 2025 are development/selection. **Fold 2026 is SPENT
report-only evidence** — the pure tuner's alpha selection already consumed
the 2026 fold, so nothing here may be selected or re-tuned on it. The
untouched confirmation season is **2027**, and the frozen candidates for that
gate are exactly the dev selections below (the 2026-favoring recency
observations later in this doc do not modify them).

## Selected configurations (dev folds 2024+2025 only)

| Arm | off/def alpha | half-life | prior scale | dev MSE |
| --- | --- | --- | --- | --- |
| Kalman prior | 2000/5000 | none | 1.0 | 1.2211469 |
| Box-snapshot prior | 2000/5000 | none | 1.5 | 1.2210994 |
| Pure (same config) | 2000/5000 | none | — | 1.2212540 |

Both arms independently reproduce the pure tuner's `2000/5000`, no-decay
base; `5000/40000` rows are all outside the one-SE set on dev. For the
Kalman arm the unconstrained best was scale 1.5 (1.2211345) with scale 1.0
inside one SE; the predeclared tie-break takes the smaller scale.

## Forward results (game-clustered bootstrap, x1e-6 possession MSE, negative = candidate better)

| Comparison | fold 2024 | fold 2025 | dev pooled | fold 2026 (SPENT) |
| --- | --- | --- | --- | --- |
| Kalman vs pure | -101.6 [-254.0,+44.8] | -112.6 [-243.7,+23.1] | **-107.1 [-210.3,-6.3]** | -23.1 [-137.0,+89.9] |
| Box (scale 1.0) vs pure | -73.2 [-191.1,+40.9] | -163.7 [-265.3,-62.9] | **-118.4 [-197.2,-43.7]** | +53.0 [-47.7,+150.2] |
| Box selected (1.5) vs pure | -84.4 [-263.7,+88.1] | -224.8 [-378.8,-61.8] | **-154.6 [-265.2,-42.2]** | +100.1 [-46.0,+253.7] |
| Kalman vs box (same config) | -28.4 [-92.0,+37.7] | +51.0 [-23.9,+127.5] | +11.3 [-39.5,+63.8] | **-76.1 [-137.7,-18.4]** |
| Single-season box lane vs pure | +347.4 | +227.9 | +287.7 [-83.5,+672.4] | **-1065.5 [-1655.2,-453.5]** |
| Kalman vs single-season lane | -449.1 | -340.6 | -394.8 [-787.7,+15.1] | +1042.4 [+416.6,+1649.9] |

Headline: **both predictive-prior arms beat the pure multi-year solve on the
development folds with pooled CIs excluding zero** (2/2 folds each). This is
a solve-level forward gain, in contrast to the NBA v5 netcont-feat null. The
Kalman and box arms are statistically tied on dev; on spent 2026 the Kalman
arm keeps a small insignificant edge over pure while the box arm flips
slightly negative, and Kalman clearly beats box. The gains are ~0.009% of
the 1.221 absolute MSE — normal scale for these gates.

## Per-component attribution (single-target coefficient swaps into pure)

Dev-mean swap deltas (x1e-6): every one of the ten targets helps for both
arms. **TOV dominates** (-67.8 Kalman / -64.1 box, ~45% of the swap-sum),
then RIM_FG (-24.4 / -20.6), FT (-15.9 / -17.9), MID_FG (-14.9 / -6.4, the
Kalman `sk_mid_pct` edge), SECOND_CHANCE (-3.5 / -15.4, the box
oreb/self-oreb/dreb edge). On spent 2026, TOV (+29/+44) and MID_FG flip
positive while SECOND_CHANCE, FT, RIM_FREQ, and NET_BASE keep helping.

## Mechanisms (no bare nulls)

- **Label space** (priors vs zero-centering against realized next-season
  single-season labels): the Kalman prior beats zero in 87.5% of the 20
  (target, side) cells on dev (mean weighted RMSE ratio 0.963, mean weighted
  corr 0.255); box 90% (0.968, 0.232). Offense sides fit much better than
  defense (e.g. MID_FREQ off ratio 0.900/corr 0.534 vs def 0.986/0.179).
  THREE_FG is the weakest family (ratios ~1.0, def corr -0.007) — long-range
  make luck stays unpredictable, matching NBA experience.
- **Forward-error slopes** (rZTOV-style): regressing (label_S - pure_coef)
  on (prior - pure_coef) gives mean slopes 0.86 (Kalman) / 0.85 (box) with
  correlations ~0.7 — where the prior disagrees with the pure multi-year
  solve, the future single-season label moves most of the way toward the
  prior. (Caveat: the shared pure_coef term makes this slope generous; the
  label-space ratios above are the cleaner statement.)
- **The single-season-lane 2026 flip is the likelihood window, not the
  prior** (`ss_flip_probe.csv`): the bare previous-season-only PURE solve at
  2000/5000 is +842 [+478,+1211] worse than the multi-year pure on dev but
  -972 [-1546,-421] better on spent 2026. Box priors help the single-season
  lane everywhere (dev +842 -> +293 pure->prior); the production
  `5000/40000` alphas hurt it on dev (+1171 pure). Inside the multi-year
  lane the same recency story appears: on spent 2026, 550d decay improves
  pure by -497 [-800,-190] and Kalman+550d reaches -592 [-877,-317], while
  dev folds mildly punished decay. The 2026 fold rewards recency hard —
  plausibly expansion-era roster churn (Golden State 2025; Toronto/Portland
  2026) — but 2024/2025 did not. This is recorded as a spent-fold
  observation; the frozen 2027 candidates stay the dev selections, and any
  decay/recency re-selection would need its own predeclared gate.

## Research board

`research_board_fold2026.csv`: the Kalman-arm coefficients trained on
2022-2025 at the selected config (288 players, all with priors), with
per-100 off/def/net, the raw prior centers, and the pure-solve counterpart
columns. Solver convention: `def100_raw` is points allowed (negative =
good), `net100 = off100 - def100_raw`. Top of board: Stewart, Wilson,
Skylar Diggins, Plum, Gray, Alyssa Thomas — priors are modest (about
+0.5 to +1.8 per 100) relative to solved values at these exposures, so
lineup evidence dominates for high-minute players.

## Current board (live-2026 companion)

`build_wnba_predictive_net_vpm_current.py` emits `research_board_current.csv`
(+ `research_board_current_meta.json`): the same frozen dev-selected Kalman
config read from `report.json`, but the likelihood window includes the live
2026 rows on disk (222,483 rows through 2026-08-23 at first build; 46,977
of them 2026). By default the prior stays the phase-1 season-start 2026
Kalman file — the within-season fresh-blend refresh (Phase 2 below) is not
applied — and 2026 rookies with no 2023-2025 panel history
(e.g. Olivia Miles, Azzi Fudd) carry `has_prior = 0` as pure lineup
evidence. `--prior-mode fresh` opts into the frozen-w* fresh-blend prior
and writes separate `research_board_current_fresh.*` files (see Phase 2
below); the default outputs are untouched. `net100_decay550_sensitivity` is a labeled sensitivity column
motivated only by the spent-2026 recency observation; the frozen 2027
candidate remains the no-decay solve. Versus the fold board, active-player
net correlates 0.94 with mean absolute movement 0.71 per 100 — the live
rows move mid-exposure players most (Timpson +3.4, Paopao -2.3). The
runner refuses to run if `report.json`'s selection ever gains a decay term
without a deliberate update, and 2027 rows raise everywhere.
`research_board_current_components.csv` carries the full per-player net
decomposition — off/def/net per 100 for each of the ten targets — with
per-player closure to the board totals asserted at 1e-9. Largest
active-player spreads: TOV (0.97 std), SECOND_CHANCE (0.92), THREE_FG
(0.82); THREE_FREQ is nearly flat (0.06).

## Phase 2 — in-season fresh-blend prior (development)

`research_wnba_fresh_blend_phase2.py` tests the pre-registered in-season
refresh: at cut date t inside season Y, `prior_Y(t) = mu_start_Y + w *
(mu_fresh_Y(t) - mu_start_Y)`, where mu_fresh applies the identical frozen
year-forward models (trained on seasons < Y) to Kalman snapshots at t.
Selection cells (`--phase select`, write-once `selection.json`): seasons
2024/2025 x cut fractions 0.30/0.50/0.70 of each season's distinct RS game
dates (2024: 06-11/07-01/08-17 of 87 dates; 2025: 06-20/07-15/08-10 of
101); training = 2022..Y-1 plus season-Y rows before t, validation =
season-Y RS+PS rows from t. Coverage rule: both-priors players blend
normally, in-season debutants (26-43 per cell) get `w * mu_fresh`,
start-only players carry mu_start (zero observed). Gates passed:
reproduced season-start priors matched `prior_predictions_kalman_{Y}.csv`
at max abs diff 1.0e-16, and a direct w=0.5 solve matched the w=0/w=1
interpolation at 6.7e-16, so each cell's MSE(w) is exactly quadratic and
only the endpoints are solved (frozen 2000/5000, no decay, scale 1.0).

Pre-registered rule: pooled MSE(w) = mean over the six cells of the
cell-level game-mean MSE(w); w* = pooled vertex clamped to [0,1], rounded
to the nearest 0.05. Result: **w* = 1.00**. The pooled unclamped vertex is
4.19 (game-clustered 95% CI [2.85, 5.57]; 0, 0.5, and 1.0 all sit below
the interval), with per-cell vertices 2.56-5.61 and positive
residual-on-movement slopes in 6/6 cells. Mechanism, not a bare
direction: the frozen prior models shrink in-season movement hard — mean
|Delta off prior| 0.08-0.14 per 100 (def 0.01-0.02, max ~0.9) — the
direction is consistently right (weighted residual-movement correlation
+0.011 to +0.019), so even the full snapshot swap under-moves. This is
the opposite regime from the NBA rZTOV fresh-snapshot lane (MSE-optimal
swap weight ~0.5 from double counting); here nothing indicates double
counting at these movement scales. Paired deltas vs w=0 (e-6, negative =
candidate better, game-clustered): w=0.25 `-31.1 [-41.6, -20.5]`, w=0.5
`-60.4 [-80.4, -40.3]`, w=0.75 `-87.7 [-118.6, -56.4]`, w=1.0
`-113.1 [-153.1, -72.0]` — 6/6 cells at every weight; per-cell w=1
intervals exclude zero in 4/6.

Alpha-retune arm (report-only, the two f=0.50 cells): the best re-tuned
w=1 config (off 4000 / def 5000) beats the incumbent w=0 by
`-178.5e-6 [-350.2, -5.0]` and beats the frozen-alpha w=1 blend by
`-89.5e-6` — but that interval crosses zero (`[-260.7, +87.5]`), and both
arms carry the same w=1 prior, so no re-tuned no-blend configuration
beats the blend (they are the same prior at w*=1; the comparison is an
alpha observation inside it). `off_alpha` 1000 clearly hurts (+184 to
+326e-6). The lane config stays 2000/5000.

2026 (spent/descriptive — not confirmation; run only after selection.json
was frozen): cuts 2026-06-10/07-03/07-29 (31/51/71 of the 101 RS dates
played so far), frozen <2026 models, season-start reproduction 1.0e-16,
linearity 4.4e-16. Pooled vertex 5.40 [3.68, 7.19]; w=0.5 vs w=0
`-134.6e-6 [-182.2, -87.7]`, w=1.0 (frozen w*) vs w=0
`-256.1e-6 [-353.1, -166.5]`, 3/3 cells each; in-season debutants 62-70
per cell.

The headline fold board and the default current board keep the
season-start prior: adopting w* into any surface is an explicit owner
decision, and 2027 is the untouched confirmation for this rule. Artifacts
live under `Results/predictive_net_vpm/fresh_blend/` (`cells.csv`,
`paired_deltas.csv`, `alpha_grid.csv`, `mechanisms.csv`, write-once
`selection.json`, `report2026.csv`) and stay out of git.

An owner-requested companion reading exists:
`build_wnba_predictive_net_vpm_current.py --prior-mode fresh` writes
`research_board_current_fresh.csv` (+ components, meta) with the frozen-w*
fresh prior — today's Kalman states through the identical frozen models,
phase-2 coverage rule, with the season-start reproduction and same-model
checks re-run at build time. First build (through 2026-08-28, identical
window as the default board): active-player net correlates 0.9935 with the
season-start board, mean |move| 0.19 per 100. Veterans with 2,000+
possessions barely move; the action is the 74 in-season debutants who now
carry `w* * mu_fresh` priors (Olivia Miles +4.35 -> +5.31, entering the
top six) and low-sample rookies whose weak box seasons now pull them down
(Prosper -1.01 -> -2.09). The default invocation still emits the
season-start board unchanged.

## Artifacts

All under `wnba_test/Results/predictive_net_vpm/`: `report.json` (full
spec, selections, comparisons, audits, source hashes), `labels_by_season.csv`,
`kalman_asof_features_{2023..2026}.csv`, `prior_predictions_{arm}_{fold}.csv`,
`selection_grid_{kalman,box}.csv`, `fold_scores.csv`,
`paired_comparisons.csv`, `component_attribution.csv`, `mechanisms.csv`,
`prior_model_metadata.json`, `ss_flip_probe.csv`,
`research_board_fold2026.csv`, `research_board_current.csv`
(+ components, meta), `research_board_current_fresh.csv` (+ components,
meta; only after a `--prior-mode fresh` run), plus a `_cache/` of the
home/away transform. The phase-2 fresh-blend artifacts live in the
`fresh_blend/` subdirectory with their own `_cache/`.

## Fences

- Research-only. Nothing here feeds the live published WNBA VPM board, its
  publish path (`prepare_wnba_vpm_publish.py`), the daily update, or any
  Supabase/databallr surface.
- The sanctioned Kalman-prior exception is scoped to this lane. Everywhere
  else the standing rule holds: Kalman/estimated-skills panels are
  props/forecast inputs, never rating priors.
- 2026 is spent in this lane (the pure tuner's selection consumed it);
  every 2026 number above is report-only. 2027 is the untouched gate, and
  its frozen candidates are the dev selections in this doc.
- No proprietary ratings (DPM/PI/NPI/EPM) as features, targets, or tuning
  inputs anywhere in this lane.

## Rebuild

```
.venv/bin/python wnba_test/build_wnba_predictive_net_vpm.py            # full lane
.venv/bin/python wnba_test/build_wnba_predictive_net_vpm.py --no-cache # force re-transform
.venv/bin/python wnba_test/build_wnba_predictive_net_vpm_current.py    # as-of-current board
.venv/bin/python wnba_test/build_wnba_predictive_net_vpm_current.py --prior-mode fresh # frozen-w* companion (*_fresh files)
.venv/bin/python wnba_test/research_wnba_fresh_blend_phase2.py --phase select     # phase 2 (refuses if selection.json exists)
.venv/bin/python wnba_test/research_wnba_fresh_blend_phase2.py --phase report2026 # phase 2 2026 descriptive
```

Inputs: `wnba_test/Processed/FIRST_CHANCE{22..26}[_PS].csv` +
`SECOND_CHANCE_CLEAN` pairs, the Kalman panels under
`wnba_test/Results/gameday/{kalman_box,estimated_skills}/predictions.parquet`,
and the PBP Stats box cache via `load_box_features`. The run is
deterministic (fixed bootstrap seeds) and takes about two minutes with the
transform cache warm.

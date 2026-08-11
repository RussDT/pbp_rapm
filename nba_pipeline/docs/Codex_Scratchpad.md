# Codex Scratchpad

This file is short-term session memory only.
Durable operating rules belong in root `AGENTS.md` and `CLAUDE.md`.

## 2026-08-11 — v5 netcont-feat VPM challenger: documented null

- vpm harness gained a third prior feature source `epm_box_tracking`
  (`build_epm_box_tracking_panel`): EPM `p_*` natives + 4 derived interactions,
  latest fully-completed-season `boxscore.csv` + name-crosswalked tracking CSVs,
  a two-season possession-weighted blend with a >50%-coverage guard, and
  `dumb_pf_dec` — HL550-decayed strictly pre-cut non-shooting foul rate from
  repo-owned `v5_events/foul_events.parquet`, shrunk with K=20 pseudo-games
  (replaces vendor `DumbPenaltyFouls_100`, empty 2024-25+; registered
  deviation, ground-truth panel r=0.96). Recipes can declare
  `fallback_features`: rows missing base features get a separately fit
  fallback model, rows missing both are dropped and counted in the manifest —
  no silent NaN/fallback emission. Incumbent EPM-only recipes reproduce v4
  netcont priors to max |diff| = 0.0.
- `research_harness_v5_netcont_feat.json`: 14 recipes (9 upgraded, 3
  incumbents, new d_three_freq/d_ft), `forward_target_source_subdir` reuses the
  9 v4 net forward-target parquets, ridge_alpha 0 everywhere (harness-lane
  protocol; lab alpha conventions are non-transferable). All recipes beat
  their incumbents in label space on all validation cuts.
- Solve level is a NULL: netcontfeat x2 vs netcont x2 `-21.28e-6`
  `[-60.34,+18.78]`, 1/4 cuts, `P(<=0)=0.861`; vs gross epm12_x2 `-2.85e-6`
  (neutral). Both scales still clear no-prior (+155.38/+132.01e-6, 4/4) — the
  enrichment consumed the netcont repricing edge without adding to it.
  Second straight lesson (after own-lag) that label-space prior wins do not
  transfer to summed solve-level forward gains. netcont x2 stays the leading
  research base; spent tiles and 2026-11-01 untouched.
- Provenance fix shipped: `vpm/scripts/cross_experiment_bootstrap.py` is the
  committed cross-lane bootstrap (asserts identical possession universes via
  per-cluster row counts + actual-value fingerprints); it exactly reproduces
  the recorded v4 netcont-vs-EPM12 `+18.43e-6`. 37/37 harness tests pass
  (also fixed a stale monkeypatch in the target-source-manifest test).
- Docs: vpm `RESEARCH_HARNESS.md` v5 section, `NET_CONTINUATION_DECOMP.md`
  follow-up note, root `CLAUDE.md` netcont bullet updated with the null.

## 2026-08-10 — RTSS-augmented WNBA board (development, 2025)

- New `wnba_test/build_wnba_rtss_augmented_board.py`: solves the 2025 board
  twice through the frozen live objective (`5000/40000`, prior scale `1.5`),
  box-only vs box+RTSS corrections, and reports per-player deltas. Hard fence
  `MAX_DEVELOPMENT_YEAR = 2025`; the 2026 confirmation season is untouched.
- Candidate lanes read at runtime from v2 `best_by_component.csv`, filtered to
  additively-mappable prefixes with one correction per (parent, side). Drops
  `ALT_TOV_VALUE` / `ALT_BADPASS_TOV_VALUE` (no defensible parent) and the
  `ALT_FT_FREQ`/def child in favor of the `ALT_FT`/def parent. 13 applied.
- Gotcha worth remembering: `build_clean_weighted_factors` also solves
  `RAPM` and the derived display children, and takes their priors from the
  same dict. Correcting only the nine additive components leaves those stale —
  `off`/`def`/`net_rapm` came back byte-identical between arms and the display
  residual blew out to `5.28`. Fix is to correct the fitted twelve and re-run
  `_derive_additive_priors`; residual back to `0.0000`.
- Result: net correlation `0.971`, mean absolute net move `0.53`, mean rank
  shift `7.6`. Concentrated on defense as hoped — def correlation `0.874`,
  def spread `0.863 -> 1.094` (+27% separation).
- `--forward-check` (rate 2024, score 2025) says RTSS does NOT help:
  `+9.9e-6` MSE, CI `[-120.4, +153.6] e-6`. Corrections fit on 2023 alone
  there, so it is the weakest form of the test, but it is the only
  development-safe forward fold available. No promotion.
- Carry-forward mode added (`--year 2026 --rtss-feature-year 2025`): the fence
  is on RTSS consumption, not board year, so a 2026 board is buildable
  without touching the holdout mirror; `--forward-check` refused there.
  Script also refuses outright if the feature file was rebuilt with
  `--include-confirmation-season`. Box-only arm reproduces the live
  `wnba_current_metrics.csv` to `max |diff| = 0.0` — good isolation check.
  2026 result: net corr `0.969`, def corr `0.845`, def spread `0.779 ->
  `0.941` (+21%), mean rank shift `6.1`. Same defensive signature as 2025.
- Confirmation still BLOCKED, and the earlier "101 RTSS games unsynced" note
  was wrong: those 101 are unplayed. Mirror is 229 ENDED / 229 covered / 0
  missing markings; RTSS trails the CDN by 13 played games. Real blocker is
  just that 2026 RS is 242/330.
- Docs: `WNBA_RTSS_PRIOR_RESEARCH.md` gained "RTSS-augmented board, 2025"
  and "Carry-forward 2026 board" sections.

## 2026-08-10 — WNBA predictive program executed end to end (Phase 0/1/2-prep + Kalman)

- Plan of record `docs/WNBA_BETTING_PROGRAM.md` adopted; game-day lane shares
  `wnba_test/wnba_gameday_common.py` (paths, tricodes, hash-chained JSONL logs;
  `canonical_json` sanitizes numpy and rejects NaN). Built + tested (32/32,
  offline): `build_wnba_game_logs.py` (1311 finals / 31,765 player rows,
  Eastern game dates, cached non-finals refetched each run — 1022200029 is
  permanently non-final at the CDN, skipped by design), `project_wnba_minutes.py`
  (EWMA HL 1.5 tuned on 2025 only; beats both baselines 2025 and 2026),
  `calibrate_wnba_forecast_sigma.py` (HCA 2.0003, sigma 12.6648 normal_std;
  2026 replay RMSE 13.2768 ≈ DARKO-NPI 13.118), emit/capture/score with a
  strict `emitted_at < tipoff` + `espn_state=='pre'` fence, reachable PENDING,
  CRPS-vs-realized beside close-CLV. First live emission 2026-08-11 slate:
  3 forecasts + 3 DraftKings snapshots, chains verified; model disagrees with
  the market on WAS@LVA (model WAS by 4, market LVA -7.5) — watch the log.
- `build_wnba_kalman_box.py`: DARKO-style per-stat Kalman panel, DE-tuned on
  ≤2025 (seed 20260810), 2026 frozen holdout; all 14 stats beat trailing-10 on
  dev AND holdout. Props-lane input only; doc `docs/WNBA_KALMAN_BOX.md`.
- One-shot RTSS confirmation: prereg amended (design review, pre-outcome) to a
  strictly forward primary — solve 2025 around each prior arm (corrections fit
  2023-2024 only), score 2026 possessions; in-sample 2026 nowcast demoted to
  secondary descriptive. Harness `run_wnba_rtss_confirmation.py --check`
  correctly BLOCKED (242/330 finals, 101 RTSS games unsynced — needs
  `aws login`). Defense-alpha grid extension on dev folds retained 40000
  (monotonic degradation through 160000). Phase 3 preregs frozen (props,
  cold-start). Next: conditional-sigma CRPS kill test → Phase 1.5 simulator.

## 2026-08-10 — Net-continuation VPM forward test (v4 netcont) run end to end

- `build_vpm_target_swap.py --net-lane` (+ `--rapm-source-dir`) is the net EV swap:
  `net_EV = ev_pts + kappa_z*(1-p_make)`, common gross+net identity gate so both
  trees swap identical rows, SC parquets rewritten to `SC_NET_CONT`, kappa recovered
  per file (agreement 1.0), machine-precision closures on all 22 files 2016-2026.
  Source lane is the isolated `processed_netcont_lane/` rebuild (row-aligned with
  canonical FC/SC/RAPM); swap output `results/vpm_target_swap_netcont/`.
- vpm harness now supports `data.component_tree: "net"` plus `target_swap_subdir` /
  `processed_source_subdir` / `target_source_manifest` overrides, all defaulting to
  gross (v3 lane regression-verified unchanged). Experiment
  `research_harness_v4_netcont.json`: gross-mapped frozen alphas on the ten net
  components, 12 EPM `p_*` recipes retargeted, forward labels + priors refit on net
  targets (9 cuts), evaluated at the 4 dev cuts.
- Results (4,000-rep paired game-clustered bootstraps): netcont_x2 vs net ev_noprior
  `+176.65e-6 [95.31,260.61]`, 4/4 cuts, all promotion gates pass. Cross-experiment
  on the proven elementwise-identical possession universe: netcont_x2 vs gross
  epm12_x2 `+18.43e-6 [2.19,36.01]`, 3/4 cuts, `P(<=0)=0.014`; net ev_noprior vs
  gross ev_noprior neutral (`-5.21e-6`, CI straddles 0); netcont_x1 loses (`-35.55`).
  Read: repricing alone doesn't change summed no-prior prediction; the gain is priors
  mapping onto correctly priced components. Development evidence only — accepted
  boundary unchanged until untouched 2026-27. Docs: `NET_CONTINUATION_DECOMP.md`,
  vpm `RESEARCH_HARNESS.md`; receipt `bootstrap_cross_vs_gross_epm12.json`.

## 2026-08-10 — rZTOV promoted to a per-season stat; explorer rewired

- `scripts/build_rztov_explorer.py` is now the whole lane (panel, putbacks, per-window
  fits, per-season scoring, explorer JSON, both result CSVs). The scratchpad prototype
  `build_rztov_seasons.py` is superseded. Methodology doc: `docs/RZTOV.md`.
- Switched the fit target from the point-valued `alt_*_tov_value` solves to the
  count-basis `tov_{lo}_{hi}_rs_se_a1500_1500` solve, one per trailing window. A count
  target gives kappa a known physical `-1.00`; fitted `0.75`-`0.91` is ridge solve
  recovery, and break-evens are `x/y` ratios so uniform shrinkage cancels out of them
  entirely. The `0.73 pts/turnover` figure stays labeled indicative — the two solves
  are not shrunk identically.
- Break-evens drift for real across the era (2014-18 → 2022-26 window: self
  `13.49→15.80`, ast `6.48→5.05`, pass `11.02→9.26`), and the drift **survives**
  consistent putback treatment. The earlier version only had putbacks for 2022-26 and
  silently `fillna(0)`'d 2014-21, which confounded the finding; putback counts are now
  built in-script for all seasons and cached to `results/rztov_putbacks.csv`.
- Two season adjustments: per-window refit, then re-center each season's **excess**
  (not the ratio — centering the ratio reintroduces a Jensen artifact) on its own
  possession-weighted mean. Centres are small, max `|0.11|`/100.
- Multi-season boards are possession-weighted roll-ups of separately priced season
  rows, verified in-browser to 2e-15. Ledger closure also 2e-15 once the season
  calibration row was signed `+centre` (it raises allowance, so it raises saved).
- Validation 2022-26 `n=568`: excess `-0.784` vs TOV-value RAPM (raw TOV/100 `-0.071`,
  per-attempt `-0.419`), yoy `+0.811` (n=2,934), R2w ladder `0.001 / 0.330 / 0.723`
  against unrestricted `0.743`. Per-season qualified sd `2.88`-`3.38`.
- The per-self-created-attempt rate is roughly flat across creation load
  (`6.4%` at 3-6 → `6.0%` at 20+), so the explorer's curve panel argues *one rate per
  area, not one rate overall* rather than the old (wrong, mix-intercept-dropped)
  "heavy creators turn it over less" framing.
- Ledger restructured to two **measured** lanes: shot attempts (`v_self + v_ast` vs
  `sc_100`) and potential assists (`v_potast` vs `bp_100`). The old cross-sectional
  `alloc_*` split of `sc_100` across the two attempt rows is gone — it priced assisted
  above self-created in every window and forced the two displayed `TOV%` cells to a
  league constant `1.4215`. Merging removes the fabrication instead of re-basing it; the
  attempt mix now shows up only where it is real, inside the blended break-even. Self /
  assisted survive as driver rows on the same columns, muted: exact volume, allowance and
  break-even, with `TOV` / `TOV%` starred because they re-split the lane by **within-player
  marginal risk** (`7.7%` vs `2.8%`, adjacent-season deltas) instead of the discarded
  cross-sectional weights. Drivers close to the lane at `8.9e-16` and their play shares sum
  to `100%` on every row.
- Denominator moved from involvement to **plays** = `involve + act`, i.e. the boxscore
  TOV% shape `TOV / (FGA + 0.44 FTA + TOV)` extended to passing. Allowance and excess are
  counts, so nothing was refit; only the divisor moved. Verified `delta * plays = saved`
  at `4.4e-16` and full ledger closure at `1.2e-15` on all 6,088 rows. yoy `+0.811` ->
  `+0.812`; qualified sd `2.63` on the 22-26 board.
- Publishing constraints for this lane: never name the tracking vendor in output;
  screen/pick data cannot be published, so offensive fouls stay inside the scoring
  turnover bucket and there is no screener area. Built page is checked ASCII-only and
  greps clean for both.

## How To Use

1. Read the latest entries at session start.
2. Append one short entry at session end.
3. When a lesson becomes durable, move it into `AGENTS.md` / `CLAUDE.md`.
4. Keep only the latest 5 entries in this file.

Never store secrets, tokens, or credentials here.

## Session Log Template

```md
### 2026-08-09 — Merged remixed PR #31
- Task: Merge `RussDT/remixed` PR #31 and deploy the backend only if required.
- Outcome: Merged verification-only scripts, tests, and docs as `3277c154`; no backend runtime deployment was required.

### 2026-08-09 — Turnover risk and TSx2 by shot clock
- Task: Describe where turnovers occur, how turnover risk changes as a possession survives, how actual points per true-shooting attempt changes, and how much total possession value remains after including rebound continuation.
- Fix applied: Added an exploratory endpoint/hazard/shot-value builder over 2024-2026 RS+PS, split 24-second starts from 14-second resets, mapped ordinary FTs to the same-clock foul or and-one FGA, included unmatched offensive fouls, and kept raw versus shot-calibrated clock estimates as a sensitivity range. Mapped initial 24-second cycles to canonical processed possession terminals and emitted a quarter-second evaluation grid for first-chance, second-chance, and total remaining value.
- Prevention rule added: Do not treat the FGA-trained clock calibrator as turnover-event ground truth; separate turnover volume from conditional hazard, show the final four seconds with and without shot-clock violations, and treat the TSx2 and possession-value curves as descriptive state selection rather than a pure causal waiting effect. A quarter-second evaluation grid is a smooth model surface, not quarter-second observed timing; use the raw whole-second curve when the calibrated tail becomes sparse below four seconds.
- Outcome: 886,202 endpoints and 111,224 turnovers; on 24-second starts conditional turnover incidence rises from roughly 1.7-2.2% early to 10.7-12.8% late, while calibrated TSx2 falls monotonically from 1.433 to 0.876 (raw 1.468 to 0.992). The FT join covers 99.61% of ordinary attempts. Across 755,570 uniquely mapped initial cycles, calibrated total remaining value falls from 1.160 points at 24 seconds to 1.007 at 4 seconds; second chance contributes roughly 0.15-0.17 points and exact row closure holds.

### 2026-08-10 — WNBA RTSS defensive features and alpha-edge resolution
- Task: Refresh the live WNBA board, add the missing RTSS defensive features (offensive fouls drawn, zone-split allowed shot quality, closest-defender dFGA), and resolve the alpha grid edges.
- Fix applied: Read the previously unused `fouls` table (offensive fouls drawn and charges drawn are defensive credits via `fouledId`), split `def_mid3_qsq_epv` into MID and THREE, and added closest-defender dFGA volume/points-allowed/allowed-FG%/contest-mix by zone — 29 new features, 75 -> 105. Wired them into defensive CV arms only. Added `extend_wnba_offense_alpha_grid.py` mirroring the defense extension.
- Prevention rule added: The RTSS feature builder fences the confirmation season by default; `--include-confirmation-season` requires an explicit `--output` so the fenced development surface is never overwritten. A one-sided alpha grid cannot close an edge — sweep both directions before claiming an interior optimum.
- Outcome: Defensive CV improved on 9 of 12 sides (THREE finishing +0.06% -> -4.92%, RIM frequency -0.73% -> -4.58%, defensive TOV value +0.03% -> -1.51% on 3/3 folds); pooled 15/24 -> 17/24. Both alphas confirmed interior: offense `5000` beats `4000` by 2.85e-6 and `6000` by 3.56e-6; defense `40000` already held. Live board refreshed to 2026-08-10 (242 games, 224 players, exact additive closure); A'ja Wilson leads at 7.57 net.

### 2026-08-10 — Shot-clock difficulty and clock-adjusted TS
- Task: Convert inferred shot-clock context into a player TS adjustment and test whether it helps oTS regressions.
- Fix applied: Added game-held-out isotonic TSx2 pricing by season and 24/14-second reset family, exact raw TS + clock difficulty = adjusted TS aggregation, a raw-clock primary surface, a calibrated-floor-4 sensitivity, and same-window plus season-forward ridge comparisons. Ordinary FT attempts inherit the foul/and-one clock and shooter; 41 conflicting FT-shooter anchors are excluded from player attribution.
- Prevention rule added: Clock adjustment is context, not pure shooting talent or causal grenade burden. Keep raw timing primary, never present quarter-second or sub-four calibrated precision as observed, and require fresh-season evidence before admitting the feature to oTS/VPM priors.
- Outcome: On 508 same-window players, adjusted TS-added improves RMSE by 0.0148 but its interval crosses zero. On the two season-forward transitions (746 player-seasons), raw adjusted TS-added improves future annual oTS RMSE from 1.5648 to 1.5349. The zTS ablation confirms playtype burden is the stronger layer: zTS alone reaches 1.5352, and clock on top reaches 1.5295, only -0.0057 incremental with CI [-0.0143, +0.0025]. A literal 2022-2026 pure-RS `Points + TSA` backfill gives only `+0.0075 R^2` from volume-scaled clock difficulty (`+0.0055` repeated-fold OOF), versus about `+0.091` from current playtype difficulty; clock adds zero after playtype. Exported current creation/shooting/early-offense/transition role numbers; keep clock diagnostic until a fresh forward test.

### 2026-08-11 — PBP-only playtype difficulty forward test
- Task: Measure how much broad own-shooter playtype difficulty can be recovered from ordinary PBP and compare it directly with Synergy on future oTS.
- Fix applied: Added a six-bucket outcome-invariant classifier, prior-season category pricing, strict expanding chronological evaluation, exact-common candidate rows, a shared baseline-selected ridge alpha, and paired player-cluster uncertainty.
- Prevention rule added: Keep this own-shooter prior separate from lineup `PLAYTYPE_PROXY_PTS`; do not use makes, misses, points, or made-only assist tags in category assignment, and do not promote from already examined 2025-2026 targets.
- Outcome: PBP and Synergy difficulty correlate `0.908`, but PBP wins: future-oTS RMSE falls from `1.5303` to `1.4966` versus `1.5135` for Synergy on 735 player-seasons. Adding Synergy after PBP worsens RMSE. Freeze for untouched 2026-27 confirmation and a later summed-possession gate.

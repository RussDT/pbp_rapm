# Codex Scratchpad

This file is short-term session memory only.
Durable operating rules belong in root `AGENTS.md` and `CLAUDE.md`.

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

## 2026-08-09 — zTOV% mix-adjusted turnover expectation prototype

- `geniussports/` at repo root holds 2013-14→2025-26 SS/Genius player-season CSVs
  (self-created & assisted shot attempts incl. misses, passes/potential assists with
  `% Turned Over`, drives, picks, touches; RS+PS). Join by normalized name+season to
  the vpm panel `nba_id`; required aliases: Enes Freedom (retroactive), KJ Martin
  (2020+ son vs 2014 father), Juancho Hernangomez.
- Repo raw-PBP player TOV split (event_type 5, desc contains 'Bad Pass'; player1
  attribution, 'No Turnover' excluded) matches the legacy panel split r=0.999/1.000.
- Original `shot_*` Genius files count ≈ FGA only (league self+ast 18.1 vs FGA 18.0;
  TSA gap ≈ 0.44×FTA). User-supplied `shot_fixed_*` files are TSA-basis (fouled
  attempts included; league 19.7 vs live TSA 20.1) — use those. Local vpm
  `boxscore.csv` 2025-26 stat columns can be stale even when OffPoss matches live
  (SGA TSA 35.3 stale vs 33.3 live); verify against Supabase
  `player_stats_with_metrics` (`nonPassTOVPct` is a per-play scoring-TOV rate ≈
  sc/(TSA100+tov100), not a TOV share; raw-PBP split cross-checks at r 0.995).
- Final v3 fits (22-26 vs `alt_{scoring,badpass}_tov_value` off-RAPM): box expected
  rates 6.0% self / 5.5% assisted per attempt + 11.7% per potential assist (+0.46
  base); per-target impact breakevens 10.4/7.4/13.8; joint simultaneous solve
  (one κ=−0.81, intercept −0.27) gives breakevens 13.7/6.3/11.0 with exact identity
  pred = a + κ·rZTOV·involve/100, R²w 0.668 (unrestricted 0.675, r +0.755).
- zTOV% = (actual − mix-expected)/involvement: corr with TOV value RAPM −0.63 (raw
  TOV/100 −0.12, per-attempt −0.41), YoY r +0.74. Badpass TOV ≈ costlier than scoring
  (−0.68 vs −0.58); passes/100 conditional on potential assists negative — only
  creation passes earn allowance. Board: `results/ztov_research_22_26.csv`; explorer
  artifact `https://claude.ai/code/artifact/528791bf-bbb1-4637-9fbb-fa78fa372392`.
- zTS-compression test (same data lane, target `ts_22_26_rs_a3000_5000` off, 393
  players 5000+ poss): Genius self/assisted mix is NOT a playtype substitute. Add-stat
  ladder at 0 baseline: rTS .157 → self/ast-mix zTS .236 → +per-bucket qSQ .336 →
  playtype zTS (`pt_adj_rTS`, live Supabase) .460; playtype at 0 beats every genius
  variant at any baseline. Cause: assisted bucket pools rim finishing (69 TS) with
  movement threes (52) — role2 range [−2.7,+2.0] vs playtype_diff [−6.2,+3.2], r .57.
  League TS by bucket: self ~56.5, assisted ~61.5; conditional on qSQ the gap flips
  (self > ast) — creation deficit is all shot quality. `results/zts_two_bucket_22_26.csv`.
  Supabase scale trap: `mod_ts` is a fraction; `mod_rTS`/`playtype_diff`/`pt_adj_rTS`
  are already percentage points.

## 2026-08-08 — Net-continuation (kappa) DECOMP tree built and validated

- New FC_DECOMP_NET_* tree prices first-chance FGA misses at ActualNet + kappa(zone)
  (measured: rim ~0.47, mid ~0.33, three ~0.32 pts/miss, stable 2024-2026) with the
  identical kappa paid back out of the possession's SC ledger (SC_NET_CONT_Diff);
  closure exact per possession, enforced at build. Gross columns byte-identical.
- Validation rebuild 2024-2026 RS+PS in `processed_netcont_validation/`; matched-alpha
  solves: six-component sums unchanged (corr 0.998) but attribution moves — rim contest
  compresses ~24%, the SC exposure artifact disappears (Kessler +0.68 -> -0.07), Jokic
  d_SC unchanged, and miss-forcing wings without rebounding lose 0.3-0.4 real pts/100.
- Doc: `docs/NET_CONTINUATION_DECOMP.md`. Research lane only; production/frozen lanes
  untouched. EV-lane composition algebra (expected-continuation in EV comps, deflated
  residuals) is written up there for the follow-on.

## How To Use

1. Read the latest entries at session start.
2. Append one short entry at session end.
3. When a lesson becomes durable, move it into `AGENTS.md` / `CLAUDE.md`.
4. Keep only the latest 5 entries in this file.

Never store secrets, tokens, or credentials here.

## Session Log Template

```md
### YYYY-MM-DD HH:MM (local)
- Task:
- What went wrong:
- User correction:
- Fix applied:
- Prevention rule added:
- Outcome:
```

## Session Log

### 2026-08-08 — Championship Help rebuilt with monthly EPM and playoff rotations
- Task: Replace the retrospective DARKO title model with monthly EPM and rethink the neutral teammate baseline.
- User correction: Use the playoff rotation that actually played; neutral should reflect shortened postseason rotations rather than blindly setting four teammates and the bench to 0.
- Fix applied: Pulled 297 private monthly EPM-only snapshots from November 2001 through July 2026, froze the last snapshot before each postseason, weighted by actual normalized playoff lineups, and set the common neutral origin to the season-average playoff rotation. Actual-context lift still replaces the focal player with 0 EPM; normalized self/on-floor/off-floor allocations close by continuous Shapley.
- Prevention rule added: Keep raw provider snapshots private; never let later playoff EPM updates feed the rating being allocated, never let series length change team strength, and define neutral against the playoff field rather than the all-minute regular-season pool.
- Outcome: 2002-2026 has 400 team-seasons and 4,894 player-seasons; overall EPM C=0.2, LOYO log loss 1.711152, annual mass gap 4.44e-16, and attribution gaps at floating-point noise.

### 2026-08-08 — The Help Index one-screen reader redesign
- Task: Make the title-odds player view understandable without a long page scroll.
- Fix applied: Replaced five competing headline cards with the exact neutral + self + teammate = actual equation, placed replacement lift beside it, bounded season/teammate lists in one panel, and collapsed rankings, DARKO, and methodology.
- Prevention rule added: The primary title answer must remain visible as one coherent workspace; diagnostic and legacy surfaces default closed and may not compete with the player story.
- Outcome: The compact reader passes the full build, lint, and nine-test deployment suite and preserves the same EPM data/model outputs.

### 2026-08-09 — Merged remixed PR #31
- Task: Merge `RussDT/remixed` PR #31 and deploy the backend only if required.
- Outcome: Merged verification-only scripts, tests, and docs as `3277c154`; no backend runtime deployment was required.

### 2026-08-09 — Turnover risk and TSx2 by shot clock
- Task: Describe where turnovers occur, how turnover risk changes as a possession survives, and how actual points per true-shooting attempt changes at the same time.
- Fix applied: Added an exploratory endpoint/hazard/shot-value builder over 2024-2026 RS+PS, split 24-second starts from 14-second resets, mapped ordinary FTs to the same-clock foul or and-one FGA, included unmatched offensive fouls, and kept raw versus shot-calibrated clock estimates as a sensitivity range.
- Prevention rule added: Do not treat the FGA-trained clock calibrator as turnover-event ground truth; separate turnover volume from conditional hazard, show the final four seconds with and without shot-clock violations, and treat the TSx2 curve as descriptive state selection rather than a pure causal waiting effect.
- Outcome: 886,202 endpoints and 111,224 turnovers; on 24-second starts conditional turnover incidence rises from roughly 1.7-2.2% early to 10.7-12.8% late, while calibrated TSx2 falls monotonically from 1.433 to 0.876 (raw 1.468 to 0.992). The FT join covers 99.61% of ordinary attempts.

# Codex Scratchpad

This file is short-term session memory only.
Durable operating rules belong in root `AGENTS.md` and `CLAUDE.md`.

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

## 2026-08-08 — v5 own-lag prior bundle fails the solve gate (informative null)

- Joint alphas x scale harness on v5 (`vpm/evidence/alpha_harvest_v5feat/`): paired
  v5-vs-v4 at matched cells = -1.06e-6, CI [-35.8,+33.4], 3/8 cuts. Registered rule
  required >= 0 -> v5 rejected; EPM12 x2 remains the leading research base.
- Mechanism: prior increment at harvested alphas v5 +390.7 vs v4 +391.9 (x1e-6) —
  own-lag added ~nothing in-solve despite +3-23% label-space wins, because the solve
  window already contains the seasons the lag summarizes. Derived rule: in-solve prior
  features must be informationally EXTERNAL to the solve window; own-solve lags are
  for no-possession consumers (season-start, cross-league) only.
- Falsifiable prediction half-confirmed: one-SE base alphas leave the 750 floor as
  prior scale rises (any informative prior absorbs the persistence role), but v5's
  ranges are identical to v4's. Verdict + full numbers appended to
  `vpm/evidence/prior_feature_lab/PREREGISTRATION_V5.md`. 2026-27 untouched.
- Rim-DFG screen (exploratory, v6 ledger): `rim_points_saved_100` beats blocks-only
  d_rim_fg prior +1.24% (5/6) in label space; freq leg marginal, contested-shots dead.
  External info (passes post-v5 rule). Jokic audit: tracking rim_dif% +0.6 AGREES with
  the solve's ~neutral rim contest; his dVPM is rebounding/fouls/positioning. Also:
  local boxscore.csv/tracking 2025-26 rows are partial early-season snapshots — panel
  rule must use 2025 until refreshed (calendar-complete != file-complete).

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

### 2026-08-08
- Task: Diagnose and fix playoff handling in multi-year/timedecay RAPM after playoff inclusion was observed to lower participants' ratings.
- Fix applied: Added `rapm.py --ps-prior` (+ `--ps-prior-stage2 full|ps-only`): stage-1 RS-only solve, stage-2 refit shrunk toward the RS solution via the existing prior-centered ridge; companion `*_psprior_stage.csv` audit. Fixed the silent max-weight bug for unparseable dates in `compute_time_decay_weights`. Built `diagnose_ps_pooling.py` (RS-vs-candidate playoff delta report) and `evaluate_ps_handling.py` (forward gate, paired game-clustered MSE vs rs_only). New doc `docs/PLAYOFF_HANDLING.md`.
- Prevention rule added: measured mechanisms — playoff lineups are ~+7.3 pts/100 stronger both sides (good-vs-good ridge compression), PS environment is 1.4-3.6 pts/100 lower-scoring, td700 upweights same-season PS rows ~1.11x. Full-sample prior-centered stage 2 expands the rating scale ~1.28x (reduced effective shrinkage) — always run a shrinkage-dose control candidate when gating prior-centered variants.
- Outcome: Forward gate (22-24->25, 23-25->26): `all_se` tied-best with rs_only (-9.2e-6, n.s.); both psprior modes significantly worse (+244/+357e-6), a1500 control worse (+157e-6). Verdict: `-se` on the daily ALL runs, keep `psprior ps-only` as the descriptive playoff-attribution surface (22-26: non-participants move by one constant; early exits -0.16..-0.28, deep runs +0.32), never as the canonical rating. 2026-27 untouched.
- Production flip (operator-approved, same day): added `--season-effects` to the daily 1Y-6Y and both td700 runs with `_se` filenames flowing through; updated sync lists, `upload_timedecay_rapm.py` (now `weighted_factors_21_26_all_rb_se_td700.csv`), `upload_six_factor.py`, `update_2026_{josh_rapm,purerapm,scaledoutput}.py`, and the CLAUDE.md publish list. 14-26 rb run and skipped scposs mapper left unchanged. Filename cross-check vs `build_result_suffix` passed for all eight runs; external rapms-repo consumers must switch to `_se` names.

### 2026-08-08 14:10 (local)
- Task: Explain Kalman filters visually.
- Fix applied: Built and verified an interactive predict-measure-update explainer with adjustable sensor noise and responsive desktop/mobile layouts.
- Outcome: The visual shows how uncertainty controls the weighted blend and how repeated updates narrow the estimate.

### 2026-08-08
- Task: Prior feature lab (16 engineers + 4 adversarial reviewers) and the v4 solve-level gate vs EPM12.
- Fix applied: All 16 recipes reproduced and confirmed: 8 replacements (makes-above interactions swept the FG/FT recipes), 2 new defensive priors (d_ft foul volume, d_three_freq defender profile), 5 honest nulls; emitted the 14-recipe v4 bundle date-safely and ran the joint alphas x scale harness.
- Prevention rule added: Label-space mapper gains do not translate 1:1 to solve level — paired v4-vs-EPM12 at matched cells is only +7 to +22e-6 with CIs crossing zero, because priors and the likelihood substitute. Judge prior upgrades by the paired solve-level delta, never the prior-increment or label-space numbers alone.
- Outcome: v4 is non-worse with positive point estimates and two extra covered sides; screens after the lab found the o_ft role-split (+4.2%, craft-over-role beta 2.4x) and the own-lag primitive (+13.7% d_rim_freq label-space) — own-lag ties to the historical alpha-floor on rim deterrence (low alpha was a poor man's persistence prior) and is the leading v5 candidate.

### 2026-08-08 — The Help Index championship-help launch
- Task: Build and publish season-level estimated championship help from DARKO teammate context.
- Fix applied: Added final-team player/team segmentation, regular-season-only common-sample own-plus-help features, leave-one-season-out model selection, actual-versus-neutral title probabilities, exact deterministic Shapley teammate attribution, app shards, tests, docs, and the new Help Index panel/social card.
- Prevention rule added: Use playoff data only for champion labels; never let traded-player early segments inherit titles, use realized playoff exposure as a predictor, or describe probability differences as causal ring credit.
- Outcome: The 1998-2026 spline model selected at `C=1.0` posts LOYO log loss `0.153037` and AUC `0.832146`; 7,495 player-seasons and 115,936 pair-seasons reconcile at `5.55e-16`; the app displays season spikes and top-20 career teammate estimates.

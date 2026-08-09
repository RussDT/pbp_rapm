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

### 2026-08-08 — Championship-help team-probability rebuild
- Task: Correct the championship surface after auditing whether annual title odds close across teams and whether 2009 used the intended DARKO timing.
- What went wrong: The first build repeated the champion label on every qualifying player, so its independent player probabilities summed near a champion roster size rather than one; strict as-of lookup also made the 2008-09 display use 2007-08 DARKO.
- Fix applied: Rebuilt one row per team-season from every regular-season lineup and exact same-season DARKO, fit a leave-one-season-out conditional softmax over all teams, held each focal player's own exposure-weighted contribution fixed, included teammates' bench/off-floor minutes, and reallocated the normalized probability delta with deterministic Shapley permutations.
- Prevention rule added: Any quantity labeled team title odds must have one champion label and exactly unit probability mass inside every season; keep retrospective same-season estimates distinct from as-of forecasts.
- Outcome: 863 team-seasons across 29 years close annual probability mass within `6.66e-16`; the selected offense/defense model is `C=5`, LOYO champion log loss `1.668791`, and 212,045 focal-player/teammate season rows reconcile within `9.99e-16`.

### 2026-08-08 — Championship self/on-court/off-floor closure
- Task: Make identical team odds explicit for every teammate while separating focal impact, teammates shared on court, and team minutes while the focal player sat.
- What went wrong: The prior counterfactual closed only self versus all teammates, used an over-sharp minimum-log-loss model, and the season selector left career headline/table data visible.
- Fix applied: Added shared-lineup pair exposures, exact self + shared-floor + off-floor two-group Shapley closure, one-standard-error model selection, season-specific teammate shards, and season-responsive UI cards/tables. All regular-season teams remain in the choice set; playoff PBP supplies only the champion label.
- Prevention rule added: Same-team actual odds must be byte-identical across qualifying players; never allocate credit by playoff qualification or realized playoff exposure; validate annual mass and all probability buckets independently.
- Outcome: Overall-DPM `C=0.2`; annual mass gap `6.66e-16`, three-bucket gap `4.44e-16`. 2009 CLE is `29.625%` for all 14 qualifying players; LeBron splits to `11.013%` self, `+21.676 pp` shared-floor, `-3.064 pp` off-floor. Parker 2004 falls from `83.8%` to `40.6%` team odds.

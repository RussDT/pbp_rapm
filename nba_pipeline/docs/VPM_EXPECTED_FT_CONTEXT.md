# Extended-defense VPM with dated FT and context

Owner authorized October 19, 2026 after reviewing the v44 one-standard-error
preference for core defense. This is a new immutable experiment, v50; v44/v45,
the old selection receipt and all published boards remain unchanged.

## Specification written before the new scores

- Use all seven defensive covariates (blocks, steals, DREB, rim saved points,
  rim contests, DFGA, personal fouls), with the same twelve offensive inputs,
  imputation, missing flags, mapper ridge .1, scale1 and entrant blending.
- Reuse six-season matched actual-FG/FT SPM calibration. Compare context-free,
  centered home, smooth rubberband and both context specifications. Context
  is estimated jointly, including in SPM calibration. Keep season intercepts
  in calibration, intercept in deployment, and rubberband alpha100.
- For lineup deployment, add a separately generated expected-FT target. Use
  neutral pregame FT predictions from existing annual past-only own-skill
  models. Model training and shooter observations must precede the game day.
  Preserve actual points separately; no current-game FT outcome is an input
  to its own expectation. Keep observed possession boundaries and lineups.
- Primary fixed-setting ablations: extended defense with actual FT; expected
  FT; expected FT+home; expected FT+rubberband; expected FT+both. Use existing
  parent4000/8000, TD550 and mid/three residual12000/48000 throughout.
- Additionally tune the full extended/expected-FT/home/rubberband candidate
  on 2020–2022: O2000/4000/8000, D4000/8000/16000, TD275/550/1100. Select the
  literal minimum equal-season game MSE, deterministic lexical ties. No
  one-SE preference. Report the fixed ablations regardless of selection.
- Freeze selection before emitting/scoring 2023–2026 stress. Score the same
  seven game/team tests and paired errors versus saved long-SPM, previous
  VPM and EPM. Actual future minutes/pace and the existing +3/bubble0 home
  forecast convention remain shared across systems to isolate rating changes.
- Home context is zero for the certified 2020 bubble and unresolved venue
  rows. Other nominal home designations retain the existing venue limitation;
  this experiment does not certify a complete historical neutral-site list.
  Rubberband uses repaired possession-start score and clock. These
  observed game-state variables enter training only, never pregame forecasts.
- Expected FT changes only deployment denoising; long-SPM calibration retains
  actual FG+FT paired with observed counts, so its existing statistical
  calibration contract remains coherent. Mid/three SQ targets stay frozen.
- FT actual-minus-expected is retained as the negative of `ft_delta` for
  accounting. It is not added back to the denoised training target. The field
  named `expected_ft_points` is the whole actual-FG plus expected-FT parent,
  not FT points alone. Observed FT attempts and possession boundaries remain.
- No2027 outcomes, public publication, automatic promotion or new untouched
  historical claim. All historical stress was already spent.

## Verification contract

Audit source hashes, exact possession/lineup alignment, FT expectation dates,
actual-point invariance, expected-FT delta closure, repaired context alignment,
joint normal equations and SQ parent/residual coefficient closure. Compare
the new context solver with an independent dense solve and the context-free
incumbent. Perturb future/current-day FT counts to prove prediction prefix
invariance. Independently reconstruct game/team scores and paired error gains.

Status: completed and independently audited. The expanded defense / context
candidates do not improve the headline game forecast; retain v44/v45.
Generated targets, models, boards and score artifacts remain private and gitignored.

## Source and numerical verification

All22 season/phase FT files pass an independent raw-event to canonical-row
reconstruction, maximum delta gap5.69e-14 points.97 FT events after a game's
last retained canonical possession remain unassigned; no lineup credit is
invented. The event auditor preserves the frozen builder's original minute/
seconds ordering, including rare one-second disagreements with cached MM:SS
strings. Context uses the repaired processed-possession score/clock transform.
Modern non-FT team-heave/block companion rows can share an event number;
their equal-clock terminal lookup is distinct from FT event identity.

Thirty focused tests pass, including independent dense/profile equivalence,
context-free incumbent equivalence, prior/target linearity, neutral bubble
handling, exact identity joins and real FT same-day/future-prefix mutation.
All21 new context calibration fits have normal-equation errors below6e-14.

## Reproduction

From the RAPM checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python \
  nba_pipeline/scripts/build_vpm_dated_ft_context.py --stage all
.venv/bin/python nba_pipeline/scripts/audit_vpm_dated_ft_context.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python \
  ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v50_expected_ft_context.json \
  expected-ft-context --stage all
```

The harness supports `prepare`, `calibrate`, `tune`, `boards`, `evaluate`,
`audit`, and `all`. Completed target/board artifacts are reused only under
their source/hash contracts. Main result directory:
`../vpm/vpm/results/harness/vpm_extended_dated_ft_context_v1/`.

## Results and decision

The2023–2026 stress block contains4,920 games and120 team-seasons. These are already-spent historical outcomes, with shared actual minutes/pace. Lower RMSE is better. Fixed ablations use4000/8000/TD550; full-package minimum-MSE tuning chooses4000/8000/TD275.

| Model | Development game RMSE | Stress game RMSE | Stress team-net RMSE |
|---|---:|---:|---:|
|Existing core defense|13.77877|13.96041|3.23521|
|Seven defense fields, actual FT|13.76204|13.98111|3.23205|
|Seven fields + expected FT|13.76273|13.97640|3.22111|
|Seven fields + expected FT + home|13.76027|13.97836|3.22435|
|Seven fields + expected FT + rubberband|13.79022|14.05591|3.36797|
|Seven fields + expected FT + both|13.79024|14.06301|3.37973|
|Full package, minimum-MSE tuned|13.76862|14.03401|3.30321|
|EPM benchmark|13.78461|13.94152|3.12000|

The seven-field extension had lower development error, but its stress game error is0.0207 points RMSE worse than the core. Its paired game MSE gain versus core is−0.5783[−1.2172,+0.0598], unresolved. Expected FT recovers0.0047 game RMSE inside the seven-field model: paired gain+0.1317[−0.0700,+0.3415], unresolved. Do not call either a confirmed game improvement.

Adding home after FT changes game MSE by−0.0549[−0.1069,−0.0020]; adding rubberband changes it by−2.2290[−2.9724,−1.5143]. The fixed full package is worse than core by2.8751 MSE[1.7272,4.0083]. Even the development-tuned full package loses by2.0603[0.9625,3.1835]. These intervals are unadjusted and concern overlapping historical comparisons.

The FT-only increment is positive in the neutral actual-point lineup diagnostic: MSE1.415044884→1.415021126, gain23.758e-6[1.843e-6,45.256e-6],423,644 possessions /4,273 games. It excludes unseen-player lineups and uses no realized future score/clock. The combined seven-field+FT difference versus core is unresolved on this diagnostic. Rubberband loses here too. This small conditional-possession FT gain does not establish an all-game gain or justify promotion.

**Decision:** retain the existing core VPM as control; keep dated FT as an implemented research option. Do not promote the expanded/full-context model, retune on stress, or claim the earlier known-state rubberband result transports to neutral pregame game sums. The one-SE preference has not been reapplied: the new rule selected the literal full-package development minimum, which still loses.

All seven tests are saved in `scorecard.csv`, with squared correlation and direct forecast R² separately. Paired game/team intervals are in `paired_comparisons.csv`; exact FT increments are in `actual_point_incremental_comparisons.json`. All thirty focused tests,22 source/event audits,21 real context calibration normal-equation checks and the independent game/team/seven-score/paired-point audit pass. The field reconstruction maximum gap is7.11e-15, scorecard gap1.78e-15, and no2027 outcomes were used.

## Full seven-test stress receipt

R² cells below are squared correlations, matching the existing field headline. Only team-net cells are RMSE. Direct forecast R² and every error metric remain separately saved in `scorecard.csv`.

|Test|Core|Seven actual FT|Seven expected FT|Expected FT + home|Expected FT + both|Tuned full|EPM|
|---|---:|---:|---:|---:|---:|---:|---:|
|game_margin|0.1866|0.1849|0.1853|0.1851|0.1818|0.1819|0.1889|
|game_margin_low_continuity|0.2957|0.2926|0.2950|0.2953|0.2890|0.2879|0.2981|
|team_win_pct|0.6691|0.6680|0.6706|0.6701|0.6539|0.6590|0.6890|
|team_win_pct_low_continuity|0.6201|0.6551|0.6630|0.6620|0.6382|0.6178|0.4860|
|team_net_rating|3.2352|3.2321|3.2211|3.2244|3.3797|3.3032|3.1200|
|game_margin_veterans|0.1763|0.1743|0.1747|0.1744|0.1710|0.1710|0.1795|
|team_win_pct_veterans|0.6457|0.6437|0.6459|0.6454|0.6297|0.6342|0.6665|

The full fixed package is not the development-best ablation: expected FT+home has the lowest fixed-setting development MSE. Tuning was explicitly restricted to the full package as specified before scoring; its TD275 choice is not a claim that it beats every ablation. No selection or parameter change followed stress inspection.

# Blending previous and long-sample VPM

Owner requested October15,2026. Immutable sibling v46 / `long-spm-blend`,
experiment `vpm_long_spm_blends_v1`. This tests **previous repaired VPM v39**
against **long-sample VPM v44** using new-model weights0%,25%,50%,75%,100%.
Every player receives the same weight on offense and defense. No EPM enters
these blends; it remains an external benchmark. All existing boards remain unchanged.

## Result

**50/50 is the best of these five weights for2023–2026 game errors.** Its
RMSE13.9474 is below previous13.9659 and new13.9604, and close to EPM13.9415.
Its paired game MSE gain over previous is+0.5164[+0.0094,+1.0399]; against
the pure new model,+0.3636[−0.1580,+0.8773], unresolved. Against EPM the
gain is−0.1637[−1.4688,+1.1625], unresolved. Intervals are unadjusted for
the five weights and seven overlapping tests. Historical stress was already
spent before this owner-requested analysis; this is not fresh confirmation.

The declared2020–2022 development game-loss rule chooses **0% new /100% previous**.
Previous MSE188.3529 versus25%188.4353,50%188.7129,75%189.1860,100%189.8544.
Only0% and25% are within their paired one SE of the minimum, and the frozen
incumbent preference retains0%. The later50/50 game advantage is a research
finding, not the weight chosen by the development rule. No promotion or
additional stress tuning;2026–27 remains untouched.

|New-model share|Stress game RMSE|Stress team net RMSE|
|---:|---:|---:|
|000%|13.9659|3.2892|
|025%|13.9527|3.2626|
|050%|13.9474|3.2446|
|075%|13.9500|3.2355|
|100%|13.9604|3.2352|
|EPM|13.9415|3.1200|
|DARKO_lagged|14.0554|3.4102|

Team net error favors the pure new model;75% is almost identical. EPM has
lower team net error than all five blends. The50/50 paired team-net gain
over previous is+0.2910[−0.2699,+0.7218], unresolved; versus EPM it is
−0.7933[−1.1749,−0.3466], favoring EPM. Low-continuity team results use only
13 team-seasons across four season blocks.

## All seven stress tests

R² is squared correlation. RMSE is lower-is-better. Same4,920 games and120
team-seasons; equal season weighting. Low-continuity games492, teams13.

|Test|0% new|25%|50%|75%|100%|EPM|
|---|---:|---:|---:|---:|---:|---:|
|Game margin R²|0.1856|0.1869|0.1874|0.1873|0.1866|0.1889|
|Low-continuity game R²|0.2949|0.2971|0.2978|0.2973|0.2957|0.2981|
|Team win% R²|0.6609|0.6654|0.6682|0.6693|0.6691|0.6890|
|Low-continuity team win% R²|0.5606|0.5815|0.5982|0.6110|0.6201|0.4860|
|Team net RMSE (lower)|3.2892|3.2626|3.2446|3.2355|3.2352|3.1200|
|Veteran game R²|0.1753|0.1765|0.1770|0.1769|0.1763|0.1795|
|Veteran team win% R²|0.6365|0.6413|0.6442|0.6456|0.6457|0.6665|

## Definition and checks

Blend player ratings linearly: `(1-w)*previous + w*long-SPM`. The endpoints
share the same fitted player universe, likelihood, penalties4000/8000,
TD550, repaired SQ and fixed entrant blend. In this setting, blending prior
means inside the ridge solve gives the same blended fitted rating, since
the solve and exposure centering are linear with shared geometry. No model
is refitted or rescaled in this diagnostic. Missing-player fills and the
under250-minute veteran override remain identical.

Game forecasts blend with the same home term. Team quality is the linear
blend of minute-weighted season-centered quality. Team win percentages are
**recomputed from each blended game margin through the fixed logistic
curve, then averaged by team**; endpoint win probabilities are not blended.

The independent validator reconstructs all1,001 saved score rows,105 paired
stress MSE point estimates, the full50/50 game bootstrap, nonlinear team
win aggregation, endpoint/fill/veteran identities, development selection
and source hashes. Player-row reconstruction differs by at most floating
point roundoff. The full seven-test cards include development2020–22,
stress2023–26, mixed2021–26, extended2020–26 and each individual year.

All systems share actual future minutes and pace, so these are conditional
player-quality tests. Historical source/model publication vintages remain
uncertified. These blends do not create a new untouched confirmation set.

## Reproduce

```bash
.venv/bin/python ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v46_long_spm_blends.json \
  long-spm-blend

.venv/bin/python ../vpm/vpm/scripts/validate_long_spm_blends.py \
  ../vpm/vpm/results/harness/vpm_long_spm_blends_v1
```

Private artifacts: `scorecard.csv`, `paired_comparisons.csv`,
`game_predictions.parquet`, `team_predictions.csv`, `selection.json`,
`blend_manifest.json`, `independent_validation.json`, and `blend_report.md`
under sibling `vpm/results/harness/vpm_long_spm_blends_v1/`. Small curated
summary: sibling `vpm/evidence/long_spm_blends_v46_summary.json`.

Parent methodology: [long-sample VPM](VPM_LONG_SPM_TRAINING.md).

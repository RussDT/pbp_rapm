# Zero-prior VPM: seven-test field benchmark

October 15, 2026. Immutable sibling v42 / `vpm_zero_prior_clean_sq_field_v2`,
run through `zero-prior-field` in the canonical VPM harness. This is a matched
ablation of the repaired v39 own-SQ parent, not the older independently
penalized ten-component no-prior experiment. V41 built boards but stopped
before evaluation artifacts on a string/numeric CSV franchise-ID comparison;
v42 normalizes exact identity types. Both are preserved.

## What no prior means

Every player-impact ridge mean is zero. Own-skill and rookie/G League prior
maps are absent from the zero-arm fit. Hold parent O/D penalties 4000/8000,
TD550 and both mid/three residual penalties 12000/48000 fixed. No retuning.
The identity is `zero_parent(actual) − zero_parent(residuals) + strong(residuals)`,
equivalently `zero_parent(EV) + strong(residuals)`; both constructions agree.

**Shooter talent remains in SQ.** The repaired shot targets retain strictly
earlier game-day shooting history and requested-before-game EPM predictive
shooting skills. No prior refers to the player-impact regression mean; it
does not mean the targets contain no estimated shooting information. The
existing `VPM_no_entrants` removes only entrant priors and is not zero-prior.

## Results

Same v40 census: 7,230 games / 180 team-seasons, 2020–21 through 2025–26.
Low continuity: 853 games / 20 team-seasons. Equal season weights. All
incumbent game and team predictions reproduce v40 within 1e-12. Native
zero-prior coverage equals prior VPM because fitted player universes match.

| Test | No-prior VPM | Prior VPM | EPM | Lagged DARKO | Lagged LEBRON | Lagged MAMBA |
|---|---:|---:|---:|---:|---:|---:|
| Game margins R² ↑ | .1600 | .1789 | .1785 | .1661 | .1524 | .1812 |
| Low-continuity games R² ↑ | .2562 | .2860 | .2850 | .2812 | .2716 | .2940 |
| Team win percentage R² ↑ | .5907 | .6379 | .6544 | .6205 | .5928 | .6705 |
| Low-continuity teams R² ↑ | .3905 | .3865 | .4092 | .4794 | .4527 | .5277 |
| Team net rating RMSE ↓ | 3.4807 | 3.1861 | 3.0839 | 3.3445 | 3.4646 | 3.2036 |
| Veterans: game margins R² ↑ | .1575 | .1676 | .1684 | .1570 | .1462 | .1755 |
| Veterans: team win percentage R² ↑ | .5768 | .6121 | .6320 | .5956 | .5698 | .6546 |

No-prior game RMSE is **14.1726**, versus prior VPM **13.9928** and EPM
**14.0085**. Prior VPM gains **5.0653 MSE [3.8885, 6.2872]** over no-prior
on games and **1.9639 [1.0090, 2.8623]** on team net rating. Priors improve
six headline point estimates; low-continuity-team R² is slightly higher
without priors, with an unresolved difference. Direct error favors priors
in all seven tests; the low-continuity-team interval crosses zero. The
veteran season-win R² difference also has an interval crossing zero.

R² is squared correlation, not direct forecast R². RMSE is lower-is-better.
DARKO/MAMBA/LEBRON are archived previous-season endpoints, not certified
preseason publication snapshots. Actual future minutes and pace are shared
oracle inputs. Fixed common missing-player fills and veteran overrides remain.
These are retrospective development tests; historical architecture selection
and provider vintages remain caveats, and 2026–27 is untouched.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v42_zero_prior_clean_sq_field.json \
  zero-prior-field

.venv/bin/python ../vpm/vpm/scripts/validate_zero_prior_field.py \
  ../vpm/vpm/results/harness/vpm_zero_prior_clean_sq_field_v2
```

Artifacts are private/ignored: seven boards and manifests; source/code locks;
scorecard; full game/team predictions; player inputs; coverage; paired prior
and zero-prior comparisons; independent seven-test validation. The baseline
v34/v37 and repaired v39/v40 code, boards and score artifacts remain unchanged.

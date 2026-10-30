# Predictive VPM ShotQuality temporal repair

Owner authorized October 15, 2026 after the field-benchmark leakage audit.
Preserve the original monthly v34 and field v37 artifacts. No impact alpha,
prior, continuity rule, missing-player fill, home term or calibration is tuned.
Historical scores remain development evidence after repairing the computation.

## Repair specification

Sibling immutable v39 / `vpm_clean_sq_preseason_v2` builds new shot EVs and
seven dated preseason own-SQ boards. V38's initial preparation failed on
mixed pandas timestamp precision before writing features or fitting models;
v39 normalizes timestamps to nanoseconds and uses a new output directory.
Field v40 / `vpm_field_benchmark_clean_sq_v1` runs the unchanged seven-test
contract on the new boards.

The new EV fitter reads only raw, unmodeled PBP shot-context/outcome fields
from the canonical parsed shot cache. None of its full-sample EV, context,
shooter-talent, defender or assisted-make outputs enter the feature set.

Shooter talent is still part of SQ: the fitter rebuilds shooter-history
features using completed earlier game-days and retains requested-before-game
EPM predictive shooting skills. The excluded fields are the old cache's
full-sample fitted outputs. Within-season talent evidence can update before
each historical shot's game day; current-game outcomes never enter its
features. A zero player-impact-prior solve retains these same SQ targets.

- Replace full-2015–2026 shooting means with the immediately preceding complete
  season's zone/jumper means. The 2015 warm-up uses 2014 league evidence.
- Rebuild half-weight pre-2015 career count seeds from observed mid/three
  shots through 2014. Hold the legacy zone-specific seed convention fixed.
- Shooter features use only completed game-days strictly before the shot's
  game day; current-game results never enter those features.
- Omit the transition probability for every year because the original dated
  training lineage is missing. A modern-trained historical backcast would
  not establish historical availability. No new transition proxy is tuned.
- Fit action/family category mappings only on the training prefix; unseen
  categories map to −1. Parse period clock from authoritative MM:SS rather
  than the legacy seconds-remainder field.
- Keep only requested-before-game EPM predictive shooting skills, with
  checksummed requested/effective-date metadata; complete EPM is not a feature.
- Hold the original gradient-boosting hyperparameters fixed and fit each
  scored season from earlier seasons only, including early stopping wholly
  inside that training prefix. No selection uses the repaired scorecard.

This changes more than the global league means. Do not attribute all numerical
changes to a single leakage patch: the repair also removes uncertified
transition and fixes same-game history, category vocabulary and clock handling.

## Possession and prior boundary

RAPM's `rebuild_monthly_sq_atoms.py` owns target writing. Rebuild the canonical
first-FGA/child eligibility audit from raw actual-FT PBP with the shooter
season FT lookup disabled. Attach new EV using the existing target adapter's
chronological first-FGA and identity gates. Replace only the two gross SQ
residual atoms. Require every actual point, FC/SC value, lineup, date, turnover,
miss and continuation count to remain identical. Continuation pricing stays
strictly before each rating cut.

Actual-point parent labels and entrant-prior packages do not depend on SQ.
Reuse their immutable hashed artifacts with a separate reuse receipt. Own
skill states, matured-label eligibility, own-prior maps, parent4000/8000,
scale1.5, TD550 and residual12000/48000 are retained. Verify prior preservation
against the old boards. Seven cuts cover October2019–October2025, with
December2020 replacing October2020. No 2027 data is requested or admitted.

## Validation and reproduction

Use the existing sibling harness; run stages sequentially:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python \
  ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v39_clean_sq.json \
  chronological-sq --stage prepare
```

Then use stages `fit --workers 4`, `targets --workers 4`, `boards`, and `audit`.
The audit perturbs a future shot suffix, independently refits both 2020 EV
models using a truncated panel, independently refits a 2024 VPM board with
future targets/labels/skill rows removed, checks all target fields and own-skill
date provenance, and writes `temporal_audit.json`.

Run field v40 with `preseason-benchmark --stage prepare`, then `evaluate`.
Validate the saved results with `vpm/scripts/validate_field_benchmark.py`.
`vpm/scripts/compare_clean_sq_field.py` pairs the old/new game/team results,
asserts identical samples/targets/coverage/competitor predictions and player
priors, and bootstraps the before/after error difference without retuning.
Generated features, models, logs and predictions stay private and gitignored.

These checks establish computational chronology for the audited reconstruction.
They do not certify retrospective provider publication/model-training vintages,
turn spent seasons into fresh holdouts, or replace actual future minutes/pace
with pregame forecasts. Untouched 2026–27 promotion needs a separate frozen
protocol.

## Completion receipt

Completed: **22 SQ models, 1,614,212 scored shots, 22 RS/PS target files with
2,781,191 possessions, and seven rebuilt preseason VPM boards**. The old/new
EV files have exactly the same shot keys and made outcomes. All actual
scoring, FC/SC, lineup/date and count fields are identical. Rebuilding the
actual-FT eligibility checks changes 54 midrange and 35 three-point attachment
decisions; these 89 decisions are part of the repair, not a sample change.

All seven boards retain **exactly identical own, entrant and combined prior
means/weights and prior NBA exposure** relative to the old boards. RMS player
rating movements range from 0.0490 to 0.0617 points/100 across cuts. The change
is in the SQ likelihood correction, not a newly trained prior family.

### Temporal and artifact checks

- Perturbing 306,006 post-October2024 shot outcomes, identities, categories and
  skill inputs leaves the audited 1,454,030 earlier shots' talent features
  unchanged: maximum difference **0**.
- Independently refitting 2020 mid/three models from a truncated panel exactly
  reproduces all 129,866 saved shot probabilities: maximum difference **0**.
- Independently refitting the 1,349-player October2024 VPM board after removing
  future target/count/label/skill rows gives maximum rating and prior changes
  **0**. This exercises the EV/target/prior/possession-solve chain with the
  all-target field invariants; it is a representative full-data refit, not a
  claim about every possible provider revision.
- All 2,627,535 own-skill provenance date fences pass. The lineage validator
  verifies **800 saved input/artifact hashes**, all 22 model training windows,
  and all seven matured-label/board fences.
- Independent benchmark validation passes: game/player uniqueness, exact
  five-player minute mass, scoreboard wins/points, shared veteran overrides,
  and weighted-OLS versus direct-error definitions.
- **72 focused tests pass**. Both repos pass `git diff --check`.

Receipts: `temporal_audit.json` and `lineage_validation.json` in the new monthly
experiment; `independent_validation.json`, `prior_preservation.csv`, and
`repair_comparison.json` in the new field experiment.

### All seven tests

R² below is squared correlation; net-rating RMSE is lower-is-better. Same
7,230 games / 180 team-seasons, 2021–2026. All comparator predictions,
minutes, pace, coverage and subgroup assignments are unchanged.

| Test | Original VPM | Repaired VPM | EPM |
|---|---:|---:|---:|
| Game margin R² | 0.1791 | 0.1789 | 0.1785 |
| Low-continuity games R² | 0.2849 | 0.2860 | 0.2850 |
| Team win % R² | 0.6390 | 0.6379 | 0.6544 |
| Low-continuity teams R² | 0.3775 | 0.3865 | 0.4092 |
| Team net RMSE | 3.1792 | 3.1861 | 3.0839 |
| Veterans-only games R² | 0.1680 | 0.1676 | 0.1684 |
| Veterans-only team wins R² | 0.6137 | 0.6121 | 0.6320 |

Direct game-margin RMSE is **13.9896 original VPM → 13.9928 repaired VPM**,
versus **14.0085 EPM**. Direct game forecast R² becomes 0.1717. The paired
original-minus-repaired game MSE gain is **−0.0882 [−0.2124, +0.0339]**,
and team-net MSE gain is **−0.0437 [−0.1335, +0.0480]**. Neither change is
statistically resolved. Every before/after MSE interval across the seven tests
crosses zero. This estimates the combined repair, not the isolated effect
of the original full-panel means.

Against EPM, repaired game MSE gain is **+0.4384 [−0.8170, +1.6684]**:
no demonstrated EPM win. EPM retains the lower team-net RMSE (**3.0839** versus
**3.1861**), with paired MSE gain **−0.6408 [−1.4009, +0.1193]**. Season-win
association and low-continuity-team point estimates still favor EPM. The full
new `scorecard.csv` includes all dated EPM / lagged DARKO / LEBRON / MAMBA
comparisons, per-season results, veterans and native-coverage diagnostics.

The old v34/v37 artifacts remain preserved and flagged. The known SQ
full-panel-prior path is repaired and the computational temporal checks pass.
This does not remove historical model-selection exposure, certify the
provider's model/publication vintage, or establish an available-before-tipoff
forecast. Nothing is published or promoted, and 2026–27 remains untouched.

### Final verification commands

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  ../vpm/vpm/scripts/validate_clean_sq_lineage.py \
  ../vpm/vpm/results/harness/vpm_clean_sq_preseason_v2 nba_pipeline ../vpm

.venv/bin/python ../vpm/vpm/scripts/validate_field_benchmark.py \
  ../vpm/vpm/results/harness/vpm_field_benchmark_clean_sq_v1

.venv/bin/python ../vpm/vpm/scripts/compare_clean_sq_field.py \
  ../vpm/vpm/results/harness/vpm_field_benchmark_v3 \
  ../vpm/vpm/results/harness/vpm_field_benchmark_clean_sq_v1 \
  ../vpm/vpm/results/harness/vpm_monthly_2019_v1 \
  ../vpm/vpm/results/harness/vpm_clean_sq_preseason_v2
```

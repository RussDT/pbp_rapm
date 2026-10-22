# Expanded annual-snapshot VPM prior

Owner authorized implementation and training on October 22, 2026, including subagents and feature engineering. This continues the existing VPM RESEARCH project. The completed [timing study](VPM_PREGAME_RESULTS.md) did **not** test the full predictive-stat library as VPM inputs. Its frozen artifacts remain unchanged.

## Objective and controls

Train annual-snapshot predictive box-score/advanced stats against following-365-day actual lineup scoring, retaining the full-covariance profiled offensive/defensive residual likelihood. Improve conditional future-game prediction, rather than fit RAPM labels or imitate EPM. Current selected v61 remains selected throughout research; no 2027 outcomes, publication or live switch is authorized.

Keep the completed annual-snapshot + season/home +365-day calibration-game decay recipe as the clean matched control. Preserve residual precisions 2000/4000, joint baseline O10/D10 then fixed-offense conditional D100, Parent4000/8000, TD550, actual FT, SQ12000/48000, reviewed entrant weights and common historical whole-game quarantine. Expand the prior's feature basis and explicitly declared extra-block precisions only.

## Dated evidence and feature admission

Use the full-history **independent** predictive-stat profile first. Re-query saved legal annual OU recipes at exact cohort, deployment and September1,2026 dates. Each query is terminal in an independent replay; inserting query-only dates must not alter later states through season-boundary variance injection. A per-player last-game pregame export is not an annual/current snapshot.

Apply the validated source_v2 observation masks to every primitive/derived dependency affected by unsafe measurements. Existing full-history preparation did not include all these masks. Preserve independently observed tracking/passing measurements when their own source gates pass. Save explicit source dependencies and missingness; unobserved bootstrap states are missing, observed zero is valid. Canonical v2 aliases switch in ending2025 without filling from v1. Record actual parameter fit-end dates, rather than infer legality from directory years.

Saved filter parameters are transported to this newly masked replay, not refitted or called source-clean hyperparameter estimates. Old187CORE fits and baseline inputs stay unchanged. The cross-stat-corrected profile is not silently substituted: its coefficients were fitted under the older observation policy and would require a separately declared arm.

Exporter contract: `export_full_history_annual_snapshots.py`, ordered `query_id/player_id/query_date/query_season/cohort_vintage`, full baseline player universe, wide `lib__` feature packet, long head provenance, explicit feature-schema dependency DAG and receipt. Expected initial catalog:129canonical concepts,71 nonduplicate coherent derived summaries and three declared products. All166 source heads remain provenance, not automatic duplicate model inputs. Final actual admitted counts are recorded per fit.

## Feature transforms and bounded registry

Freeze the old clean numeric inputs, preprocessing, missingness and model-aware fallback behavior exactly. Fit separate additional blocks using only legal training snapshots and pre-cut exposure: explicit catalog admission, robust train-only bounds/scaling, missing flags, exact/near-constant duplicate checks, and target-independent residualization against the corresponding side's old basis. Source metadata or flags alone cannot make a player physically observed. Rows with no measured history for an added block receive a zero block after residualization.

All candidates share one frozen maximal basis per fit cutoff. Remove/add whole fixed blocks; do not change imputation, physical block gates, fallback admission or normalization per candidate while claiming profile-submatrix equivalence. Exact duplicate screening is family-local over the same fixed union, so a coordinate cannot disappear because its retained representative is absent from that family. New fields never erase an old aggregate fallback. Save names, units, source epoch, primitive dependencies, basis order, transforms and hashes.

Initial bounded registry:12 models, comprising clean control; a heavily shrunken additive overlap/source-representation control; new concepts assigned to native sides at four O/D precision-multiplier pairs; both-side new concepts at the same four pairs with stronger opposite-side shrinkage; and two engineered variants with a small declared interaction/hinge bank. Multiplier pairs are (1,1), (10,1), (1,10), (10,10); engineered variants use (10,1)/(10,10). Added block precision scales with admitted group size, with extra shrinkage for opposite-side and engineered terms. The source-representation control is additive, not a pure causal source swap.

Use response-independent group ridge and identical support for total and every Net response. Independent elastic-net selection per lane would break additive closure and is not used. Keep all existing numerical/source/lineup/Net guards.

## Chronological selection

This NEW experiment selects on annual-cut conditional game-margin MSE, unlike the earlier rolling-timing selection. Actual minutes/pace and all declared missingness/veteran/entrant conventions remain common. Annual snapshot training does not imply that deployed rolling player inputs stay frozen.

Fit and score all12 candidates on annual2020–22 development cuts. Prequential nested selection uses a declared strong (10,10) default in outer2020, because no earlier validation forecast is admitted; outer2021 tunes each family's precision only on2020 out-of-time forecasts; outer2022 uses2020–21. Select the family by equal-year outer2020–22 MSE, then select that family's final precision using all2020–22 out-of-time forecasts. Record both adaptive outer scores and final fixed-recipe development scores. Freeze the complete recipe before opening2023–26 diagnostic scores. Tie handling and exact IDs live in the immutable registry.

No new2019 impact fit or change to frozen producer date guards is required. Older cohort snapshots are still legal training covariates. All2020–26 evidence is already spent historically; nesting improves selection discipline, not the claim of an untouched holdout. No “beats EPM” or promotion claim follows from development or a point estimate alone.

## Efficient fitting and validation

Annual features are constant within player/cohort. For residual/nuisance design T, construct A=T D with offensive bases/100, defensive bases/−100 and zero nuisance directions. With K=T′WT+Λ and U=K⁻¹ΛD, profile information is (D−U)′ΛD and RHS is U′T′WY. Retain free-nuisance rows of U. This is an exact evaluation of the same lineup likelihood, avoiding dense possession-by-feature arrays; it is not covariance approximation or label regression.

Before real fits, test nonuniform weights, free nuisance columns, full joint/conditional augmented-solve equivalence, physical Net/total closure, zero/constant responses, and real-prefix clean-control parity against the completed adjoint model. Cache the maximal full-covariance profile once per cutoff; candidate supports/precisions reuse validated submatrices. Source/recipe/parameter/profile hashes remain pinned.

After development selection: fit frozen annual2023–26/current candidates, reproduce all seven annual benchmark tests against matched control/selectedv61/EPM, and evaluate native own/combined/final actual-point errors under both declared intercept conventions. Run rolling replay only for the frozen winner and clean matched control. Verify exact Net/SQ/prior/raw-center/intercept current boards. Report feature/block ablations, coverage, source/optimizer qualifications and paired conditional game/season uncertainty. Preserve all candidates and reject promotion when overall evidence is unresolved or worse.

## Implementation status — October 22

The source export is complete:21,444 exact queries (1,787 players ×12 dates),203 catalog fields and3,559,704 provenance rows for166 source heads. All12 cuts pass independent source/parameter/date/physical-history checks. Nine diverse heads reproduce exactly in raw-prefix replay; perturbing future observations leaves earlier snapshots bit-identical. Source recipes are transported, not refitted. The203 fields include74 redundant derived/product inventory fields that do not automatically enter the canonical-input grid. The separately declared engineered family uses six interactions/four train-only hinges. At the2020 fit, the both-side model has186 offensive/182 defensive basis coordinates; the engineered variant197/193, compared with26/22 in the unchanged baseline. Coordinates include missing flags and are not counts of independent skills.

All36 annual2020–22 development mappers are fitted. The clean control reproduces original H365 coefficients within4.87e-12; worst new normal-equation error1.60e-12. All physical Net responses and the independently fitted total close. Independent2020/2021 cached-profile audits and actual2020 reader/prior stress checks pass, including the inherited2963.46 baseline coordinate. Annual forecast replay is running; **no feature winner or predictive improvement is established yet**. Do not score a partial replay or inspect spent diagnostic scores before selection is frozen.

The first version, `vpm_expanded_annual_v1`/configv70, stopped before any fit because immutable career/diagnostic receipts named a subsequently modified external API TypeScript file. Preserve that failed root and all old receipts. The exact original17,291-byte Git blob was recovered (SHA6910060e…28dd92); the active API file was left unchanged. An independent1,472-file/739-JSON audit verifies the complete affected evidence chain. The NEW `expanded_vpm_source_runner.py` allows only four exact original receipt bytes/input-field/path/digest combinations to resolve that provenance leaf to its byte-identical archive. Existing data/model/source recipes, chronology/math/Net guards and all other pins remain unchanged. The API file is never executed by these research stages.

Active experiment: sibling config `vpm/configs/research_harness_v71_expanded_annual_archived.json`, output `vpm/results/harness/vpm_expanded_annual_archived_v1`. Use the source-runner entry point for every stage; direct legacy readers correctly reject the changed mutable provenance path. Completed development fits must not be duplicated.

From the sibling VPM repository:

```bash
../rapm/.venv/bin/python -m vpm.harness.expanded_vpm_source_runner --config vpm/configs/research_harness_v71_expanded_annual_archived.json --operation forecast --period development
../rapm/.venv/bin/python -m vpm.harness.expanded_vpm_source_runner --config vpm/configs/research_harness_v71_expanded_annual_archived.json --operation select
../rapm/.venv/bin/python -m vpm.harness.expanded_vpm_source_runner --config vpm/configs/research_harness_v71_expanded_annual_archived.json --operation calibrate --labels 2023,2024,2025,2026,current
../rapm/.venv/bin/python -m vpm.harness.expanded_vpm_source_runner --config vpm/configs/research_harness_v71_expanded_annual_archived.json --operation forecast --period spent
```

Then run `--operation diagnostics --period development|spent` and `--operation current --current-stage prepare|boards`. Each must finish with source/recipe/output hash verification and independent result checks. Existing annual query products cannot be used as a daily library; the annual forecast entry point deliberately rejects rolling mode until a separate dated export and registration exists. Keep selectedv61 and2027 unchanged. Both old clean CORE and transported full-history filters carry finite-optimizer-budget qualifications; the new ridge solves use checked direct linear algebra.

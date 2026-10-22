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

Freeze the old clean numeric inputs, preprocessing, missingness and model-aware fallback behavior exactly. The active **v72 bounded raw bank** fits additional coordinates using only legal annual training snapshots and earlier exposure. Numeric features use their declared physical/log1p/logit domain, weighted .005/.995 reference support, weighted center, and scale=max(observed weighted SD,.1 transformed-domain units,maximum support distance/10). Missing indicators have scale floor .1. Thus every added numeric/flag coordinate is bounded by10. The old baseline prefix is deliberately unchanged; this is not a bound on learned impact priors.

Do not regress additions on the unrestricted legacy basis. Correlations with old fields are handled by the joint regularized likelihood. Keep physical-history gates, observed-zero semantics, and response-independent fixed-union, **added-bank-only family-local** duplicate/rank screening. An old baseline coordinate cannot screen or represent an added field. Save domain, unit, dependencies, missingness, support/scale, basis indices and hashes. All responses use the same support and precision; no per-lane feature selection or hidden remainder reconciles Net.

The prior v71 auxiliary-residualization recipe is quarantined after a deployment input-support failure. Before any2023–26 forecast scores were produced, rare baseline BLK/STL missing flags made its selected overlap model produce priors around−786,293/−393,150 points per100 for two sparse-history players. Clean control stayed normal (max22.93) and matched the old model within9.19e-12. Compensated arithmetic preserved the extreme values, proving this was feature extrapolation. Preserve v71 models, selection, failed products and all guards. New v72 retrains the full development grid and freezes a fresh choice; this repair uses predictor support and no later-year scoring outcomes.

Initial bounded registry:12 models, comprising clean control; a heavily shrunken additive overlap/source-representation control; new concepts assigned to native sides at four O/D precision-multiplier pairs; both-side new concepts at the same four pairs with stronger opposite-side shrinkage; and two engineered variants with a small declared interaction/hinge bank. Multiplier pairs are (1,1), (10,1), (1,10), (10,10); engineered variants use (10,1)/(10,10). Added block precision scales with admitted group size, with extra shrinkage for opposite-side and engineered terms. The source-representation control is additive, not a pure causal source swap.

Use response-independent group ridge and identical support for total and every Net response. Independent elastic-net selection per lane would break additive closure and is not used. Keep all existing numerical/source/lineup/Net guards.

## Chronological selection

This NEW experiment selects on annual-cut conditional game-margin MSE, unlike the earlier rolling-timing selection. Actual minutes/pace and all declared missingness/veteran/entrant conventions remain common. Annual snapshot training does not imply that deployed rolling player inputs stay frozen.

Fit and score all12 candidates on annual2020–22 development cuts. Prequential nested selection uses a declared strong (10,10) default in outer2020, because no earlier validation forecast is admitted; outer2021 tunes each family's precision only on2020 out-of-time forecasts; outer2022 uses2020–21. Select the family by equal-year outer2020–22 MSE, then select that family's final precision using all2020–22 out-of-time forecasts. Record both adaptive outer scores and final fixed-recipe development scores. Freeze the complete recipe before opening2023–26 diagnostic scores. Tie handling and exact IDs live in the immutable registry.

No new2019 impact fit or change to frozen producer date guards is required. Older cohort snapshots are still legal training covariates. All2020–26 evidence is already spent historically; nesting improves selection discipline, not the claim of an untouched holdout. No “beats EPM” or promotion claim follows from development or a point estimate alone.

## Efficient fitting and validation

Annual features are constant within player/cohort. For residual/nuisance design T, construct A=T D with offensive bases/100, defensive bases/−100 and zero nuisance directions. With K=T′WT+Λ and U=K⁻¹ΛD, profile information is (D−U)′ΛD and RHS is U′T′WY. Retain free-nuisance rows of U. This is an exact evaluation of the same lineup likelihood, avoiding dense possession-by-feature arrays; it is not covariance approximation or label regression.

Before real fits, test nonuniform weights, free nuisance columns, full joint/conditional augmented-solve equivalence, physical Net/total closure, zero/constant responses, and real-prefix clean-control parity against the completed adjoint model. Cache the maximal full-covariance profile once per cutoff; candidate supports/precisions reuse validated submatrices. Source/recipe/parameter/profile hashes remain pinned.

After development selection: fit frozen annual2023–26/current candidates, reproduce all seven annual benchmark tests against matched control/selectedv61/EPM, and evaluate native own/combined/final actual-point errors under both declared intercept conventions. A rolling extension would require a separately registered daily library; it is not part of this annual experiment. Verify exact Net/SQ/prior/raw-center/intercept current boards. Report feature/block ablations, coverage, source/optimizer qualifications and paired conditional game/season uncertainty. Preserve all candidates and reject promotion when overall evidence is unresolved or worse.

## Implementation status — October22

The common source export is complete:21,444 exact queries (1,787 players ×12 dates),203 catalog fields and3,559,704 provenance rows for166 heads. All12 cuts pass independent source/parameter/date/physical-history checks. Nine diverse heads reproduce in raw-prefix replay; changing future observations leaves earlier queries bit-identical. Source filter parameters are transported, not refitted.129 canonical concepts are eligible;74 derived/product inventory fields do not automatically enter, and six declared interactions/four train-only hinges are separate.

**Active experiment:** sibling `vpm/configs/research_harness_v72_expanded_annual_bounded.json`, output `vpm/results/harness/vpm_expanded_annual_bounded_v1`. Registration verifies8,721 transitive pins. The sole controller `finishing_queue_bounded_v1`, log `vpm/results/expanded_annual_bounded_finishing_v72.log`, runs new2020–22 fits, development annual forecasts/prequential selection, then frozen2023–26/current fits, seven-test benchmarks, current exactNet boards and native diagnostics. Do not duplicate stages. It intentionally leaves final independent report verification as a separate gate.166 focused tests pass; the bounded feature recipe separately passed independent20-test/metamorphic review. Actual v72 predictive results are pending.

Independent failure evidence: expanded_failed_packet_diagnostic_v1/receipt.json SHAae4c9084aeb0e786f10943dc696ba98854617c280c553fb6697088c6464dc796; coefficient-only diagnostic SHA cbbb4e28bd1c094c93e0f21c9b2f6eeddb749afc6629997a151a15930fa99fe3. Bounded-transform final review: expanded_bounded_transform_review_v1/final_receipt.json SHA4c48507d60e29c638376de10543a58de14e41c84d524fcf59b9d75a18507a993. All are exact input pins in the new registration, alongside the preserved failed forecast receipt.

## Preserved experiments and verification

v70 stopped before fitting because four historical career/diagnostic receipts named a subsequently modified external API TypeScript file. Exact original17,291-byte Git bytes (SHA6910060e…28dd92) were recovered; active API and old data/receipts were unchanged. Source admission permits only four exact receipt/field/path/digest combinations to resolve that provenance leaf to its byte-identical archive. All other pins remain strict; API code is never executed by research.

v71 fitted36 development maps and10 later/current maps, all independently matching their saved profile/normal/Net systems. Development prequential selection on3,139 annual games favored the small overlap model13.582260 versus clean control13.584409; conditional paired MSEgain+.058374,95%game interval[-.115476,.224876], unresolved. Broader banks did not win that screen. This development result does not rescue its failed deployment support; **do not deploy or recommend v71**. No complete spent forecast/score was produced.

The original v71 `finishing_queue` was preserved at a verified no-child read-only hash handoff; all ten fits were already complete. `finishing_queue_verified_v1` reused those scientific artifacts, then preserved the first actual spent forecast failure. Neither is active. Independent verifier review confirmed identical admission/implementation maps and8,692 canonical hashes/4.80GB in22.5seconds, replacing59× repeated leaf hashing, with fresh per-call caches, conflicting-digest and mutation checks. v72 reuses this verifier contract, with its own exact executable/config/registration/output pins; scientific transform changes are separately registered and retrained.

For a diagnosed missing v72 stage, the continuation entry is:

```bash
../rapm/.venv/bin/python -m vpm.harness.expanded_vpm_bounded_verification --config vpm/configs/research_harness_v72_expanded_annual_bounded.json --operation forecast --period spent
```

Registered operations also include `calibrate --labels ...`, `select`, `diagnostics --period development|spent`, `current --current-stage prepare|boards` and `native --period development|spent`. These are interfaces for necessary continuation, not permission to duplicate a healthy controller. Preserve failed roots and version repairs instead of overwriting immutable stages. Annual queries cannot substitute for a daily library; rolling mode remains rejected. Selectedv61 and2027 remain unchanged. Historical publication vintages, transported/iteration-limited filters, conditional actual-minute/pace evaluation, fixed-fit/selection intervals and already-spent historical evidence remain explicit limitations. New ridge solves use checked direct linear algebra; no optimal-decay or live-readiness claim.

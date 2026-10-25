# Full-feature VPM history rebuild

Status: all30 source references passed independent clearance, October25,2026, and all12 registered development mappers are produced. Independent actual-model verification is pending; game forecasts and feature selection have not run. The existing reader still displays the completed source-adapted v74 history. Daily forecast/controller integration remains pending.

The owner authorized a replacement after finding that v73/v74 omitted available modern defense fields. The reduction was an implementation choice, not a test showing those fields were unhelpful. Historical coverage required early fallbacks; it did not require dropping observed modern inputs. The replacement keeps an annual-trained forward365-day prior with daily-updated inputs, lineup evidence, ratings and exact components. It does not switch to the separate daily-trained experiment.

## Controls and feature definitions

The original selected modern bank has13 offensive and12 defensive fields, with MPG shared between sides. Defense contains blocks, steals, predictive rim points saved, rim contest volume, defended FGA, shooting fouls, nonoffensive fouls, deflections, offensive fouls drawn, contested defensive rebounds, total shot contests and MPG. In legacy-anchor mode the new bridge reproduces all24 distinct numeric fields and their missingness at2019-10-01,2020-12-01,2025-10-01 and2026-09-01. This is an input/own-prior anchor, not proof that the production cleaned inputs or a new fitted history equal selected v61.

Core, rim and DFGA inputs are the last strictly earlier neutral POSTGAME states, held with a730-day stale mask. They are not date-evolved synthetic pregame OU queries. Source-parameter releases retain the original schedule, including December1,2020 despite the November1,2020 report cut.

The original source has49 independently identified false positive-exposure player/game rows and other typed measurement masks. Exact replication therefore remains a separate legacy anchor. Production source clearance must apply pinned identity and compatible field masks before the same causal estimator; it must disclose transported frozen parameters. A wholesale substitution of a different pregame estimator is not selected-input parity. Original and cleaned caches, receipts and tests remain separate. Impact fitting is blocked until the source mapping and masks pass independent review.

Early history uses only unit-compatible observed core fields. A recorded foul-group sum over total possessions is not relabeled as modern personal fouls over defensive possessions. Features become eligible only with observed nondegenerate support in matured training cohorts. Deployment removes an aggregate only if its replacement is both admitted by the mapper and observed for that player. For personal-foul fallback, both shooting and nonoffensive subtypes must satisfy that rule. Contested twos/threes are admitted together, never as a partial new arm. An observed zero is valid. All-source preprocessing and centering precede native-player reindexing; no player subset is recentered.

The bounded replacement omits the coarse pre2015 all-recorded foul-group predictor, rather than mislabel it as modern personal fouls. Thus the modern bank is restored, while the earliest seasons still have a smaller explicitly named bank. No claim is made that every v73 early-era field is retained.

## Two bounded defense tests

Four registered arms cross total versus split shot contests with raw versus role-decomposed rim savings. All retain the restored modern bank and the same early fallback rules.

1. `full_total_raw`: full bank, total contests, original rim-savings feature.
2. `full_split_raw`: replace total contests with contested twos and threes when both are admitted and observed; otherwise retain total.
3. `full_total_role`: total contests and the role rim decomposition below.
4. `full_split_role`: both changes together.

Contested two- and three-point counts already exist. The modern adapter uses the original archive counts, population and denominator with the sameTD550/K1000 estimator as the selected total. All255,489 original total measurement rows reconcile exactly to observed2PT+3PT counts. Independent full-history OU total/two/three heads also exist but are separate diagnostics; independently fitted predictions need not add and are not spliced into this control. Separate contest types do not establish defensive quality by themselves.

The existing physical rim feature is already

`rim_saved100 = 2 * predicted_rim_contests100 * predicted_suppression`.

Volume is per100 team defensive possessions; suppression is saved makes per rim contest against the earlier league reference. The positional candidate learns a shrunk past-only role mean of suppression, multiplies it by the player's own predicted contest volume, and separates the original feature into role baseline and position-relative quality. These coordinates sum to the original raw control. Known-role players receive the two coordinates; unknown-role players retain a disjoint raw fallback. Keep contest volume. Do not fit raw plus both decomposition coordinates together, zero-center ratings by position, or tune a named player's ranking.

Role means use only each player's latest pre-fit predictor packet and its past decayed rim-contest support. Adding earlier query dates cannot count overlapping support again. The500-contest pooling constant is fixed and unoptimized. Roles use strictly earlier position/minutes, have a730-day stale cap, and are unavailable before the dated2017 source. Historical publication vintages remain uncertified. A previous broader role-aware experiment did not improve overall prediction; this narrower feature still needs a matched test.

## Registered evaluation and rebuild

The sibling configuration is `vpm/configs/research_harness_v77_full_feature_history.json`. A separate independently checked registration must bind its final hash and source-clearance audit before any impact fit; do not edit configuration bytes merely to advance a status. Frozen penalties remain residual2000/4000, joint mapperO10/D10 then conditional fixed-offenseD100, Parent4000/8000, TD550, observed-only2016+SQ12000/48000. No additional calibration-game decay is selected in this feature experiment.

Keep uniform own-only priors across the four arms. Rich selected entrant priors belong only to the separate exact legacy anchor. Existing v73 cohort normals and zero-response arrays may be reused read-only only after source-only identity/census/hash checks; their old compact feature bases and coefficients are not reused. Preserve the29 deduplicated365-day cohort calendar, latest-containing assignment and leap gaps, original annual report schedule, and conservative source parameter releases.

Run annual past-only calibrations and daily-updated development forecasts for2020–22. Select the literal minimum equal-season game MSE among the four arms, with the stated simple-arm tie order; freeze before2023–26 spent diagnostics. The already inspected historical outcomes make these retrospective research comparisons, not a fresh prospective holdout. Annual seven-test diagnostics do not select a different winner. Report conditional paired game/season intervals with fits and selection fixed; preserve actual-minutes/pace and source/universe/entrant-coverage qualifications in comparisons with EPM and selected v61.

After source clearance, the staged continuation is:

1. Complete modern/early source, mask, estimator, split-count, role-prefix and exact selected-input anchors; independently review them.
2. Measure three actual cuts for runtime, peak memory and stored bytes; keep at least15GiB disk free. Stream daily queries and reuse validated source-only geometry rather than duplicate the9.5M-query feature grid.
3. Freeze code/config/source receipts, prepare new cohort bases, fit the four annual maps and run the registered development selection and spent rolling/annual/native/current diagnostics.
4. Produce all30 annual reports and6,108 game/report-date daily products under the frozen winner. Preserve the initial empty1996-11-01 cut and explicitly label every cut before2000-10-01 as a backcast, including2000 game dates.
5. Independently verify all persisted source/prefix/prior/eleven-head algebra/centers/intercepts, annual endpoint parity, predeclared fresh batch solves, the reader/index, final claims and completion certificate before switching the research reader.

Preserve previous successful and failed roots. No2027 outcomes, selected-pointer change, public/live deployment, unsupported superiority or optimal-decay claim. Source filters remain finite-budget upstream estimates, not newly optimized recipes. The old full history remains a reproducible archive throughout this replacement.

The old daily cut folder alone occupies19GiB, while free disk is about29GiB at registration. Blindly duplicating that layout would violate the15GiB reserve. Before full daily production, implement and test a lossless canonical array package with minimal identity/rating metadata: avoid writing the same components both in a wide Parquet and again in NPZ, and avoid duplicate own/combined arrays in an own-only build. Reuse old source-only geometry checkpoints read-only only after matching their inputs/census. Exact raw/centered components and intercepts must remain recoverable and independently verified. The three-cut storage pilot must project the full build plus reader under the available budget; a failed budget is a real blocker, never permission to delete archived evidence.

The compact-storage prototype reproduces three existing v74 boards bit-exactly, including all168 display columns and recomputed external source bases. Its provisional total is7.574GiB including assumed cache/reader/annual allocations. This is a proof on the old layout, not a measured new-bank production budget; repeat the three-cut test with actual new models, full provenance manifests and reader allocations before the6,108-date run.

Production integration must also preserve the native-target/pricing/checkpoint receipt graph. The new emitter's local model/reference/role pins alone are insufficient. The controller verifies the native source closure before and after a stage and binds its immutable registration receipts into each board; the compact manifest may reference that shared graph instead of duplicating its flattened file list. External basis recomputation must use the new full-bank source and role recipe, not the old v74 callback. Authoritative mapper registries enforce latest annual releases, the November2020 exception, separate final-model audits, initial2000-only backcasts and the independently frozen development choice.

## Implementation checkpoint

The focused checks pass:48 RAPM source/physical-feature tests and109 VPM tests. Independent implementation reviews cover source masks/fallbacks, role chronology, full nuisance covariance, literal batch11-head/Parent/SQ/intercept parity, lossless storage and release/audit admission. These are implementation and source-anchor checks, not fitted candidate results or a full-history audit.

Source preparation is complete under sibling `vpm/results/harness/vpm_full_feature_history_v1/source_preparation_v3`, with log `source_prepare_v3.log`; do not relaunch it. Independent clearance verifies all30 cuts,53,518 complete-prefix reference rows,1,766,094 canonical source cells and795 pins. Values and missingness match exactly, with source/parameter chronology, split observations, roles and prefix exposure verified. The receipt is `full_feature_history_source_all_audit_v3/receipt.json`, SHA`aabc1e91ed84fb558222093b723129e9935c89aa8f3518e1d6649dc466ec2747`, under RAPM estimated-skills results. This clears source packets, not fitted models or predictions.

The independent fitting registration binds929 pins and the exact twelve-model development census: four arms at2019-10-01,2020-11-01 and2021-10-01. The queue completed with exit0 under sibling `vpm/results/harness/vpm_full_feature_history_v1/development_calibrations_v1`; do not relaunch it. Producer checks report maximum coefficient Parent-sum gap2.11e-14 and sparse-cross normal error4.58e-14. Independent actual-model audits and authoritative mapper-release registries are required before forecasts. No spent model has been fitted and no prediction-based winner has been selected.

Daily source assembly must also cover the original December1,2020 parameter release, which the November2020 annual report reference does not exercise. Prepare that frozen-parameter source state under a new daily-provider work root and independently check actual query cuts before forecasting. This is source-state replay with unchanged parameters, not source-filter training. Preserve held annual source caches and receipts.

The new development selection consumer uses only complete rolling2020–22 four-arm forecasts, literal equal-season game MSE and the registered tie order. Annual diagnostics cannot select. Its output remains pending independent selection verification; later stages require a separately pinned choice certificate. The existing temporary continuation is active for the unfinished rebuild and stays quiet on unchanged healthy progress.

Preserve v1, which omitted early-cell long provenance, and v2, which stopped before fitting when pandas mixed-date conversion discarded a real source date. The narrow fix uses typed date selection: an observed zero for player202732 on2014-03-03 retains its actual date and positive support. A51-row regression reproduces the old conversion-cache failure; the actual2,051-player2014 packet now reconciles all67,683 canonical cells bit-exactly. No dates, observations or weights were invented. Source-only state caches remain unchanged; only the failed preparation/provenance stage is versioned.

Current source/implementation evidence hashes and remaining gates are recorded in [the readiness evidence](evidence/vpm_full_feature_history_readiness.json). The development comparison, spent diagnostics, full replacement history and reader remain unfinished. The completed source and development-fit roots are immutable; independently validate new stages rather than restarting successful stages.

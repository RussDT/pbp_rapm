# Daily VPM from 1996–97

The complete daily output is produced: **6,108 game/report dates and 9,416,143 player-date rows**, extending the independently audited [annual historical series](VPM_1997_HISTORY.md) from 1996–97 through 2025–26. All 30 annual endpoint comparisons passed and the worker exited successfully. Independent source, full-output, workflow and final reader checks all pass. The final report and separate completion certificate bind the evidence without rewriting the producer marker.

This is an **annual-trained prior with daily-updated VPM**. Annual predictive-stat snapshots train against the following 365 days of actual lineup scoring. Each daily estimate refreshes the predictive stats and earlier lineup evidence under the held annual mapping. It is separate from the experimental daily-trained prior that pairs pregame stats with the same game's outcomes.

The registered calendar contains 6,108 dates: the union of source game dates, admitted lineup game dates and the 30 annual report checkpoints, from November 1, 1996 through October 1, 2026. This covers game dates and report checkpoints, not every offseason calendar day. There are 9,507,828 complete observed-prefix feature queries and 37,845 admitted native games. Every rating excludes all games played on its query date.

## What is being produced

Sibling VPM config: `vpm/configs/research_harness_v74_daily_history_1997.json`. Output root: `vpm/results/harness/vpm_daily_1997_v1`. [Browse the daily history](http://127.0.0.1:8800/?coherent=2#date=2026-10-01). The predictive-stat export is in RAPM `nba_pipeline/results/estimated_skills/vpm_1997_daily_features_v1`.

Each dated board saves Offense, Defense and Total VPM; all standard Net components and TS/TOV/second-chance groups; own and combined priors; source exposure, date and model-vintage flags. Compressed exact arrays also preserve all 11 raw and centered lane heads, raw and centered scalar and lane intercepts, centering offsets, prior bases and both SQ adjustments. The components are emitted with the rating solve, rather than fitted afterward. Float64 values are retained, with no rating clipping or output quantization.

The read-only reader provides all-player/date search, O/D/Total history charts, component layers and full-precision dated Parquet downloads. Its default sample has positive admitted season exposure; an archive toggle reveals other saved estimates, preserving gaps. This is not a roster or complete official appearance census. Sample filters preserve the saved full-universe centers. Raw priors remain available; the independently centered prior display is explicitly labeled and uses full native TD exposure before filtering.

## Model and time contract

This daily series reuses the completed v73 annual mappers and fixed annual predictive-stat recipes. No filter or mapper is refitted and no new hyperparameter search is performed. The source-adapted model keeps uniform own-only priors, actual FG+FT lineup targets, Parent4000/8000, TD550, joint O/D mapper10 followed by fixed-offense defense100, and SQ12000/48000. Defense is positive-good. Audited SQ rows are observed only from2016 onward; missing earlier SQ does not become an observed zero row. Career/background quantities that cannot be identified remain missing.

The input parameter vintage and native NBA season are separate. Native season follows the source game's NBA ending year, including the October2020 bubble. Stat parameters and impact mappers are held on the original annual report schedule: October1, with the November1,2020 exception. A June2026 daily query therefore uses the held2025 filter vintage, while the October1,2026 annual endpoint uses2026. This preserves the annual reconstruction; it is not the full predictive-stat library's separate native-season release schedule. All applicable parameter training dates are checked before their query, including bootstrap recipes.

Every date before October1,2000 borrows the first usable impact mapper, including its feature admission, preprocessing and coefficients. Those dates are explicitly backcasts, including the1999–2000 season's game dates. Their actual stat measurements and lineup evidence still precede the query. Later dates use released, matured mappers. This is a consistent historical reconstruction, not a claim of strict contemporaneous availability for the initial period.

A player needs earlier admitted lineup evidence to receive a native daily rating. The first date, November1,1996, has no earlier lineup history and is explicitly unestimated; a new player's first game may likewise precede their first rating. Complete earlier source-player queries define stat centering before selecting the native lineup universe. Future participants cannot change shared players' priors. Synthetic queries do not advance or mutate the underlying observation clock; actual zero-opportunity source rows retain the frozen source semantics.

## Computation and checks

The daily feature adapter evaluates a query from the last real source observation without adding a synthetic observation to its state. Its current-source pilot reproduces all46,784 original annual queries across30 report dates bit-for-bit, including compact values, missingness, exposure, dates, MPG and provenance.

The incremental lineup solver appends only newly completed admitted games. It retains TD Gram matrices and price-free response moments, recalculating current historical shot prices and the global nonturnover reference at each cut. It does not reread and reconstruct every historical possession design each day. One Parent factorization, plus the two observed-SQ factors when available, serves all component heads. An independent original-target/direct-ridge comparison and all30 actual annual endpoints passed the numerical pilot; the largest discrepancy is about2.8e−10 in accumulated exposure counts, below the unchanged1e−9 parity guard. Rating and lane discrepancies are much smaller.

The completed production replay repeated all30 annual endpoint checks using the newly exported daily packets. It retains per-date manifests,54 immutable sparse-state checkpoints, source hashes and exact chronology flags. The maximum full saved-array endpoint gap is1.128e−10, below the unchanged1e−9 guard; final ratings differ much less. An exclusive controller lock prevents duplicate writers. The frozen producer manifest remains pending independent verification until a separate completion certificate is issued.

The full feature export took608.1seconds. The full same-root rating continuation finished in10,186.3seconds, about2hours50minutes, including native loading and daily rating/component emission. The first10 products were retained bit-identically across its explicit pilot stop/resume. No predictive-stat filter or prior mapper was refitted.

## Interpretation

This is the v73 source-adapted historical series, not a splice of selected modern v61 or the separate daily-trained-versus-EPM experiment. No predictive superiority or optimized decay follows from producing a history or reproducing annual endpoints. Historical publication vintages remain uncertified. The upstream library has4,980 head/year slots:2,476 bootstraps and2,504 finite-budget fitted recipes, only38 reporting convergence and2,466 budget-limited. Current selected v61, prior experiments, public/live systems and2027 outcomes remain unchanged.

## Source preparation and preserved failures

The immutable calendar/query preparation is sibling `vpm/results/harness/vpm_daily_1997_preflight_v3/receipt.json`. The first preparation failed on mixed date types during persistence; a second minimal census omitted the existing scoring quarantine. Both roots are preserved. Version3 normalizes dates and applies the exact per-file excluded-game lists before identity admission, retaining37,845 games. The numerical controller verified exact agreement between this minimal census and the original fully admitted loader before emitting ratings.

## Implementation and production receipts

Task branches in both repositories are `codex/vpm-daily-history-1997`. RAPM `d77c5121` holds the pure daily source adapter and its guarded integer date conversion; sibling VPM `017c905` holds the calendar/preparation/controller/numerical implementation and `cc961a7` the separate reader. Existing frozen harness entry points and scientific files were not edited.

The final-byte independent daily-partition proof is `nba_pipeline/results/estimated_skills/daily_history_1997_daily_append_independent_v2/receipt.json`, SHA `49a780d08e005933814b6a0de0e644e21bd86a1b33320f52818e1657b6f0fe07`. It checks6,108 cuts,6,073 nonempty daily append blocks,7,108,645 native rows and37,845 games against all30 saved annual endpoints. Maximum array/rating difference is1.128e−10 and scalar-intercept difference2.709e−14. The independent review receipt is `review_receipt.json`, SHA `c6bfe56c0f5089d55a6f1bde47d1a3d0dd3532ecdb617a4a81533f733370c155`. The controller requires and pins this final-byte proof before rating emission.

A separate actual first20-date integration check also passes, including the empty November1,1996 cut and the first nonempty daily priors/ratings/components. This tests real cold-start data rather than only annual endpoints.

The full source producer completed9,507,828 queries in109 partitions and608.1seconds, with5,468,449,804 compressed bytes. Its final receipt is `nba_pipeline/results/estimated_skills/vpm_1997_daily_features_v1/receipt.json`, SHA `6b13a3393be07371085441f2d2fd05241f9ea0f7ec1d3187f83be83ba1c44343`. The separate all-partition output/source/date/missingness/exposure/MPG audit passes, including all30 persisted annual packets; `vpm_1997_daily_feature_export_audit_v1/receipt.json` SHA `9a295a78c540dd6081b100dee184277bed623fab1e467556c78ba375a75ba687`. Its log remains `vpm_1997_daily_features_work_v1/export.log`; do not relaunch it. The sole rating controller log is sibling `vpm/results/daily_history_1997_v1.log`. Its initial invocation has completed the real first10 dates (November1–10,1996) and exited0 at its explicit debug limit for a consumer/resume check. The first cut is empty; the next9 contain262–342 native players. Independent first-product and same-recipe resume checks now pass. The full replay has now finished in that same root, with6,108 dates and9,416,143 rows. The real reader is at `http://127.0.0.1:8800/`. Do not relaunch the completed producer. The preserved command documents its same-recipe continuation:

```bash
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  ../rapm/.venv/bin/python -u -m vpm.harness.daily_history_1997 \
  --config vpm/configs/research_harness_v74_daily_history_1997.json \
  --preparation vpm/results/harness/vpm_daily_1997_preflight_v3 \
  --out vpm/results/harness/vpm_daily_1997_v1 --resume
```

The reader has7 passing tests and a preserved3-product preview. The real reader now uses the production root and separate `vpm_daily_1997_reader_v1` index, with completed coverage shown explicitly. Its first10-date source/value audit passes; the independent first10 numerical audit is `daily_history_1997_first10_audit_v2/receipt.json`, SHA `856353d19b4e4a723f5888e2b82f3659dbccc636d79f7773b53e40f46ab633e3`. An independent fresh batch reconstruction matches the saved arrays within2.9e−12 and ratings within9.8e−15. The explicit stop/resume preserves all39 initial registration/load/product files byte-for-byte. The full-product and final coherent consumer checks are recorded below; the original reader and first-product evidence remain preserved.


## Full production output and verification gate

The producer exited0 and wrote its final manifest at11:38:37UTC on October24,2026. It produced6,108 dated manifests,6,107 nonempty boards and30 annual endpoint comparisons; the first cold date is explicitly empty. The final native universe contains2,856 players. The9,416,143 rows include archive-only estimates, not9.4million player appearances. No source/filter/impact-mapper fit was added.

The final manifest is `vpm/results/harness/vpm_daily_1997_v1/manifest.json`, SHA `fa656b591aae87e4f44d0757e577a5fda8be1966b40f293d5029a39dee4fd74c`. Registration SHA is `dc19a7f799f7dd34b2a8e6563b4c3d26416e823a8532e65052d6f6720bd2ae00`. Its `produced_full_daily_history_pending_independent_audit` status and `full_experiment_finished=false` flag are intentional and remain immutable. A separate independently checked certificate will close the output/report verification gate.

Independent verification passed all persisted dates, source/native prefix membership, mapper/vintage chronology, own-prior dot products,11raw/centered Net heads, SQ, TD centers and intercepts, plus all30 annual endpoint tables/arrays. Fresh literal target/direct-ridge/SQ reconstruction passed11 predeclared cuts; it does not independently recompute all6,108 normal equations. Complete reader/index/browser verification passed separately. The final report/certificate gate binds these distinct scopes.

## Comparing this reader with earlier numbers

At the same report date, this daily series reproduces the source-adapted annual history. For example, Jordan's October1,1997 Total VPM is7.208566 in both. A date during the1996–97 season has different predictive stats and less lineup evidence than that completed-season report. Daily changes are expected; a same-date mismatch with the annual history is not.

The historical reconstruction is also a different model series from selected modern v61: it uses the compact historical source definitions and uniform own-only priors, with no invented background/entrant ability. Raw saved priors and separately centered display priors are different layers. Those distinctions must remain visible rather than treating every annual-trained rating as the identical modern model.


## Complete consumer verification

The complete source/index audit passed all9,416,143 float64 rating rows and names across6,108 dates and2,856 players. It confirms742 explicitly borrowed-mapper dates and3,742,582 rows with unidentified background weights kept missing. Boundary component/API checks cover raw and centered final values, raw saved and independently TD-centered own priors, Net/SQ children and scalar intercepts; maximum gap1.967e−13. These11 native heads comprise9 additive lanes and2 reference columns.

Final browser verification found and reproduced one consumer race: a slow background status refresh could restore an old date-picker selection after a newer board loaded. The separate `daily_history_1997_reader_coherent.py` consumer fixes only status request ordering/current-picker selection; every Python data/index/API/validation function remains AST-identical to the preserved original reader. The new reader root is `vpm/results/harness/vpm_daily_1997_reader_v2`; original reader code/root and race evidence remain intact. The producer, model files and saved ratings did not change. Existing7reader checks plus4targeted version/race checks pass.

The final real-browser proof passes slow polling with navigation, out-of-order responses and rapid three-date changes with picker/board agreement. Search, default retained-exposure charts, archive toggles, all four component layers, modern SQ, scalar intercepts and byte-identical Parquet downloads pass without JavaScript errors. Jordan's default chart has819retained daily observations in4disconnected paths; the archive toggle exposes the other saved dates without inventing playing appearances or recentering.

- Complete source/index/API receipt: RAPM `nba_pipeline/results/estimated_skills/daily_history_1997_reader_full_consumer_audit_v2/receipt.json`, SHA `d9d6f18c5fae636d38203df1415ea14cfbf3ad72b4287e6368981221a51e0ccb`.
- Final coherent consumer/browser receipt: `daily_history_1997_reader_full_browser_v2/receipt.json`, SHA `39bb4b1edd008d56cae3b8e7835fcfa9512e9dacb83a06ec3b853d6505c24303`.
- Consumer module SHA `21758e89a9ddb12ee603a20a99049ed9593356fbab8b39af0489bde6d67d7b63`; regression module SHA `568d7a4270d0d265e9c3ca60b9674bf1b04c7631655eaf1d2a6e52c5a685f0ea`.

The full workflow audit separately verifies6,064 registered input pins, all emitted artifact bytes and54checkpoint schedules/state hashes:30,716 unique canonical files and28.151GB. Receipt `daily_history_1997_workflow_audit_v1/receipt.json`, SHA `64611d0cb76bc3adc87e0ea92fe060aaf8ce8b1ed7bde9c137874454dd2fa94b`. The imported read-only historical-provenance verifier was not directly pinned in producer registration; its unchanged committed bytes are separately bound by this audit. No historical TypeScript substitution was used in this production verification. The all-date numerical audit passes separately as recorded below. The final report and completion certificate bind these distinct verification scopes.


## Independent numerical verification and completion

The completed independent numerical audit traversed all6,108 dates and9,416,143 rows, binding24,755 directly audited input files. It recomputed source/native prefix membership, parameter/model releases, own-prior dot products, saved raw/centered lane algebra, SQ corrections, full-native TD centers and scalar intercepts. Maximum own-prior dot-product discrepancy is0; maximum native additive-lane sum gap2.335e−15; centering gap2.220e−15; rating reconstruction gap3.553e−15; and independently accumulated TD exposure gap3.638e−11. These validate saved estimates, not forecast accuracy.

All30 annual full-array/table/scalar comparisons passed within1.128e−10. The11 predeclared independent fresh batch target/direct-ridge/observed-SQ reconstructions passed, with maximum raw-coefficient gap9.701e−15 and relative normal-equation error8.915e−15. Fresh independent normal equations were not recomputed on every date. The numerical audit took560.3seconds. Of742 marked backcast dates,741 have estimated boards, containing401,574 saved rows; the other date is the honest empty cold start.

- Numerical receipt: RAPM `nba_pipeline/results/estimated_skills/daily_history_1997_full_artifact_audit_v2/receipt.json`, SHA `169ea2e3591c07fe166fe96464b60b046c4a58c182497e62c487be6408288374`.
- The original prepared auditor and failed identity-schema attempt remain in `daily_history_1997_full_artifact_audit_v1`. Its direct solver returned canonical numeric-string IDs while persisted IDs are int64. The corrected v2 auditor normalizes only positive canonical integer IDs, rejecting malformed/nonintegral/overflow values;12regressions pass. No model, data, order, tolerance or selected cut changed. Correction receipt SHA `a444c73db6f57abe6128a6dce5db5e385b760cf687ff15b936e63fe29209554f`.
- Focused implementation checks:17daily-source,22calendar/controller/numerics and11reader/version/navigation checks passed. Existing16annual-source bridge checks also passed. These are distinct from the full actual-output audits.

Completion is recorded separately in sibling `vpm/results/harness/vpm_daily_1997_v1/independent_completion.json`, binding this report and independent source, mathematical, workflow and coherent-consumer evidence. The frozen producer's pending-audit marker is intentionally preserved. Selected `vpm/configs/current_model.json` remains SHA `08e5882c463d2ac10eccc8161afd92f14650afd8585634b4008227fde75d05d5`. The finished library is a source-adapted historical research reconstruction; retain selected v61 for the existing current model. No new predictive benchmark, public/live switch or2027outcome was introduced.

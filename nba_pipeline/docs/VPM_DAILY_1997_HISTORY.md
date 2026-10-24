# Daily VPM from 1996–97

The owner requested daily ratings and their components from 1996–97 through 2025–26. The new build is an extension of the independently audited [annual historical series](VPM_1997_HISTORY.md), with the same source-adapted model throughout. It is in progress; the complete daily library is not yet available.

The registered calendar contains 6,108 dates: the union of source game dates, admitted lineup game dates and the 30 annual report checkpoints, from November 1, 1996 through October 1, 2026. This covers game dates and report checkpoints, not every offseason calendar day. There are 9,507,828 complete observed-prefix feature queries and 37,845 admitted native games. Every rating excludes all games played on its query date.

## What is being produced

Sibling VPM config: `vpm/configs/research_harness_v74_daily_history_1997.json`. Planned output root: `vpm/results/harness/vpm_daily_1997_v1`, with a separate local reader. The predictive-stat export is in RAPM `nba_pipeline/results/estimated_skills/vpm_1997_daily_features_v1`.

Each dated board saves Offense, Defense and Total VPM; all standard Net components and TS/TOV/second-chance groups; own and combined priors; source exposure, date and model-vintage flags. Compressed exact arrays also preserve all 11 raw and centered lane heads, raw and centered scalar and lane intercepts, centering offsets, prior bases and both SQ adjustments. The components are emitted with the rating solve, rather than fitted afterward. Float64 values are retained, with no rating clipping or output quantization.

A read-only date/player history interface is being built separately. Reader sample filters must preserve the saved full-universe centers. Raw priors remain available; any centered prior display is explicitly labeled and uses full native TD exposure before filtering.

## Model and time contract

This daily series reuses the completed v73 annual mappers and fixed annual predictive-stat recipes. No filter or mapper is refitted and no new hyperparameter search is performed. The source-adapted model keeps uniform own-only priors, actual FG+FT lineup targets, Parent4000/8000, TD550, joint O/D mapper10 followed by fixed-offense defense100, and SQ12000/48000. Defense is positive-good. Audited SQ rows are observed only from2016 onward; missing earlier SQ does not become an observed zero row. Career/background quantities that cannot be identified remain missing.

The input parameter vintage and native NBA season are separate. Native season follows the source game's NBA ending year, including the October2020 bubble. Stat parameters and impact mappers are held on the original annual report schedule: October1, with the November1,2020 exception. A June2026 daily query therefore uses the held2025 filter vintage, while the October1,2026 annual endpoint uses2026. This preserves the annual reconstruction; it is not the full predictive-stat library's separate native-season release schedule. All applicable parameter training dates are checked before their query, including bootstrap recipes.

Every date before October1,2000 borrows the first usable impact mapper, including its feature admission, preprocessing and coefficients. Those dates are explicitly backcasts, including the1999–2000 season's game dates. Their actual stat measurements and lineup evidence still precede the query. Later dates use released, matured mappers. This is a consistent historical reconstruction, not a claim of strict contemporaneous availability for the initial period.

A player needs earlier admitted lineup evidence to receive a native daily rating. The first date, November1,1996, has no earlier lineup history and is explicitly unestimated; a new player's first game may likewise precede their first rating. Complete earlier source-player queries define stat centering before selecting the native lineup universe. Future participants cannot change shared players' priors. Synthetic queries do not advance or mutate the underlying observation clock; actual zero-opportunity source rows retain the frozen source semantics.

## Computation and checks

The daily feature adapter evaluates a query from the last real source observation without adding a synthetic observation to its state. Its current-source pilot reproduces all46,784 original annual queries across30 report dates bit-for-bit, including compact values, missingness, exposure, dates, MPG and provenance.

The incremental lineup solver appends only newly completed admitted games. It retains TD Gram matrices and price-free response moments, recalculating current historical shot prices and the global nonturnover reference at each cut. It does not reread and reconstruct every historical possession design each day. One Parent factorization, plus the two observed-SQ factors when available, serves all component heads. An independent original-target/direct-ridge comparison and all30 actual annual endpoints passed the numerical pilot; the largest discrepancy is about2.8e−10 in accumulated exposure counts, below the unchanged1e−9 parity guard. Rating and lane discrepancies are much smaller.

The full run must repeat all30 annual endpoint checks using the newly exported daily packets. It will retain per-date manifests, immutable sparse-state checkpoints, source hashes, exact chronology flags and a disk reserve. An exclusive controller lock prevents duplicate writers. A completed producer manifest will remain pending independent verification until a separate completion certificate is issued. Pilot success is not full-output verification.

The full feature export is expected to take tens of minutes. Pilot solves range from about0.08seconds early to5.8seconds on the latest full universe, including component emission; the full rating run is expected to take several hours. These are measured-pilot estimates, not a guaranteed completion time.

## Interpretation

This is the v73 source-adapted historical series, not a splice of selected modern v61 or the separate daily-trained-versus-EPM experiment. No predictive superiority or optimized decay follows from producing a history or reproducing annual endpoints. Historical publication vintages remain uncertified. The upstream library has4,980 head/year slots:2,476 bootstraps and2,504 finite-budget fitted recipes, only38 reporting convergence and2,466 budget-limited. Current selected v61, prior experiments, public/live systems and2027 outcomes remain unchanged.

## Active source preparation

The immutable calendar/query preparation is sibling `vpm/results/harness/vpm_daily_1997_preflight_v3/receipt.json`. The first preparation failed on mixed date types during persistence; a second minimal census omitted the existing scoring quarantine. Both roots are preserved. Version3 normalizes dates and applies the exact per-file excluded-game lists before identity admission, retaining37,845 games. The numerical controller will independently compare this minimal census with the original fully admitted loader before emitting ratings.

## Held implementation and current run

Task branches in both repositories are `codex/vpm-daily-history-1997`. RAPM `d77c5121` holds the pure daily source adapter and its guarded integer date conversion; sibling VPM `017c905` holds the calendar/preparation/controller/numerical implementation and `cc961a7` the separate reader. Existing frozen harness entry points and scientific files were not edited.

The final-byte independent daily-partition proof is `nba_pipeline/results/estimated_skills/daily_history_1997_daily_append_independent_v2/receipt.json`, SHA `49a780d08e005933814b6a0de0e644e21bd86a1b33320f52818e1657b6f0fe07`. It checks6,108 cuts,6,073 nonempty daily append blocks,7,108,645 native rows and37,845 games against all30 saved annual endpoints. Maximum array/rating difference is1.128e−10 and scalar-intercept difference2.709e−14. The independent review receipt is `review_receipt.json`, SHA `c6bfe56c0f5089d55a6f1bde47d1a3d0dd3532ecdb617a4a81533f733370c155`. The controller requires and pins this final-byte proof before rating emission.

A separate actual first20-date integration check also passes, including the empty November1,1996 cut and the first nonempty daily priors/ratings/components. This tests real cold-start data rather than only annual endpoints.

The full source producer completed9,507,828 queries in109 partitions and608.1seconds, with5,468,449,804 compressed bytes. Its final receipt is `nba_pipeline/results/estimated_skills/vpm_1997_daily_features_v1/receipt.json`, SHA `6b13a3393be07371085441f2d2fd05241f9ea0f7ec1d3187f83be83ba1c44343`. The separate all-partition output/source/date/missingness/exposure/MPG audit passes, including all30 persisted annual packets; `vpm_1997_daily_feature_export_audit_v1/receipt.json` SHA `9a295a78c540dd6081b100dee184277bed623fab1e467556c78ba375a75ba687`. Its log remains `vpm_1997_daily_features_work_v1/export.log`; do not relaunch it. The sole rating controller log is sibling `vpm/results/daily_history_1997_v1.log`. Its initial invocation has completed the real first10 dates (November1–10,1996) and exited0 at its explicit debug limit for a consumer/resume check. The first cut is empty; the next9 contain262–342 native players. Independent first-product and same-recipe resume checks now pass. The full replay is ACTIVE in that same root and the real reader is at `http://127.0.0.1:8800/`; do not execute a second controller while its exclusive lock is owned. The command below is for a necessary continuation after the current worker has ended, with the same immutable registration:

```bash
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  ../rapm/.venv/bin/python -u -m vpm.harness.daily_history_1997 \
  --config vpm/configs/research_harness_v74_daily_history_1997.json \
  --preparation vpm/results/harness/vpm_daily_1997_preflight_v3 \
  --out vpm/results/harness/vpm_daily_1997_v1 --resume
```

The reader has7 passing tests and a preserved3-product preview. The real reader now uses the production root and separate `vpm_daily_1997_reader_v1` index, with completed coverage shown explicitly. Its first10-date source/value audit passes; the independent first10 numerical audit is `daily_history_1997_first10_audit_v2/receipt.json`, SHA `856353d19b4e4a723f5888e2b82f3659dbccc636d79f7773b53e40f46ab633e3`. An independent fresh batch reconstruction matches the saved arrays within2.9e−12 and ratings within9.8e−15. The explicit stop/resume preserves all39 initial registration/load/product files byte-for-byte. Full-product, final real consumer and completion audits remain pending.

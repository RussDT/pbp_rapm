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

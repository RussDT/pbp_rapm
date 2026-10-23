# VPM history from 1996–97

Thirty annual completed-season VPM boards are produced, from 1996–97 through 2025–26: 46,322 player-season rows and 2,856 players. This is a separate, source-adapted historical Parent VPM series. It does not replace selected v61, establish a forecasting improvement, or certify historical publication vintages.

The owner requested a similar metric with era-appropriate inputs, accepting some backcasting and dropped unavailable features. The 1996–97, 1997–98 and 1998–99 boards explicitly borrow the first matured mapper, available October 1, 2000. Later boards use only previously matured calibration windows. Player measurement states and Parent lineup observations precede their report dates, including in the three backcasts. Their declared lookahead is the borrowed mapper, including its feature admission, preprocessing and coefficients.

## Deliverables

Sibling VPM root: `vpm/results/harness/vpm_annual_1997_v1`.

Browse the [1996–97 board](../../../vpm/vpm/results/harness/vpm_annual_1997_reader_v2/seasons/1997.html), [latest completed season](../../../vpm/vpm/results/harness/vpm_annual_1997_reader_v2/index.html) or [complete CSV](../../../vpm/vpm/results/harness/vpm_annual_1997_v1/annual_history.csv). The separate reader includes player search, annual O/D/Total history charts, explicit backcast shading, full precision values, grouped Net/TS/TOV/second-chance details and original CSV downloads. Charts and history tables default to years with positive retained season exposure, preserving gaps; an explicit toggle reveals the saved archive-only estimates. The first presentation version is preserved with identical numerical data, superseded by the new display default. The default season filter means positive exposure in admitted games, not a complete official appearance census.

- `annual_history.parquet` and `annual_history.csv`: all 30 reports, including the full centering universe and explicit `active_season` sample flags.
- `cuts/<date>/ratings.parquet` and `ratings.csv`: annual Offense, Defense, Total VPM; own/combined priors; exact standard Net lanes; source exposure and mapper-vintage metadata.
- `cuts/<date>/exact_net.npz` and `intercepts.json`: all 11 raw and centered lane arrays, raw/centered scalar intercepts and offsets, own/combined priors, full-universe TD exposure, and SQ corrections.
- `models/`:26 unique matured-cohort mapper fits, with full covariance geometry, all response coefficients and preprocessing. The 2021 report reuses the preceding matured cohort set because the November 2020 window has not matured by October 1, 2021.
- `cohorts/`:29 deduplicated forward 365-day lineup likelihoods and their complete observed-prefix reference censuses.

The output records maximum points-per-100 Net closure1.025e−12 and prediction-preservation error3.054e−16. These are numerical consistency checks, not evidence of predictive superiority. The independent source/model/30-board audit passed, including all raw lanes, prior arrays, SQ equations, centers and saved scalar intercepts. It reproduced mapper full-covariance information and RHS and conditional coefficients exactly. Its largest centered Parent/Net discrepancy is9.513e−13; all-head Parent relative normal error is4.263e−12.

## Model and chronology

Preserved structure: actualFG+FT lineup scoring; covariance-profiled forward 365-day likelihood with residual penalties2000/4000; joint O/D mapper10 followed by conditional defense100 with offense fixed; Parent4000/8000 and TD550; SQ12000/48000. Defense is positive-good. No new hyperparameter search or H365 calibration-game weighting was added. The latter belongs to a separate completed comparison, not selected v61's structural default.

Each completed ending season is reported October 1, except November 1, 2020 after the delayed season. A row is the rating state at that cut, using all admitted earlier games with TD550; it is not a season-average rating or a likelihood reset for each season. Each training snapshot supplies that date's predictive box-score/advanced stats to the following literal365 days of actual lineup outcomes. This profiles regularized player residuals in the original lineup likelihood, retaining O/D covariance; it is not a regression on shrunken standalone RAPM labels. Require three completely matured windows before chronological mapper emission.

A game belongs to the latest containing registered snapshot, globally for every player. The2020/2021 overlap has 93 games assigned to the latest snapshot. The 1,248 games outside calibration windows comprise 1,242 initial1996–97 warm-up games and six2020Finals games on September30–October11; they remain available to the dated Parent fit. Leap and calendar gaps are recorded without filling. Snapshot feature eligibility, preprocessing and centering use complete observed source-prefix evidence before selecting future target participants. Future-only entrants retain lineup residuals but receive zero own-stat basis. Future playing time cannot change common players' snapshot transformations.

Final ratings are centered on the full retained native player universe using side-specific TD exposure. Saved intercept adjustments preserve each5v5 prediction. Reader filters and season-sample inclusion do not change centering or the saved ratings.

## Era-adapted inputs and missingness

The new compact bridge uses the independently retained1997–2026 predictive-stat library. Shooting makes/attempts and FT forecasts remain coherent, with passing, turnovers, rebounds, steals, blocks and measured fouls. Context is kept separate from physical own-skill usability. Contested-rebound evidence is era-dependent; unavailable heads stay missing. A replacement removes aggregate DREB only after both mapper admission and player-level observation. Aggregate fouls remain unless both admitted shooting/non-offensive replacement fields are observed; non-offensive PF is unavailable in this bridge. Its recorded-foul aggregate includes technicals and uses total possessions, not official personal fouls or defensive possessions.

The bridge differs from the selected neutral estimator. Its own mapper is fitted against exactly those new definitions. Reference-fitted physical support and scale floors bound coordinates before fitting; final player ratings are not clipped. Modern terminal replay is bit-identical to the frozen exporter across166 heads and203 generic fields on the checked three-player November1,2020 sample; this is not a selected-neutral parity claim.

All 30 snapshots contain46,784 queries over2,939 cumulative archive IDs and7,766,144 head-provenance rows. Query membership is not target eligibility: the official-identity/complete-lineup gate determines the2,856-player rating universe. Omitted source IDs and per-cut coverage are recorded. Every query's prefix exposure, unknown rows and literal past365-day regular-season MPG were independently recomputed. Early MPG additionally requires clean whole-game clocks; no minutes or exposure weights are guessed.

A uniform **own-only prior** is used throughout this history. Own and combined priors are identical. Early college/GLeague/draft backgrounds and complete pre-archive careers cannot be identified consistently, so no background ability is invented. Unavailable background contributions/weights remain NaN; observed source exposure>=2000 can prove a zero background weight but is not a new ability feature. Observed exposure is a lower bound, not full career exposure or a debut date. This is an explicit difference from current rich-entrant VPM.

SQ residual solves include only observed, audited2016+ SQ rows, retaining full player columns and absolute TD550 weights. Earlier rows contribute no zero-response observations. Boards without observed SQ have an explicitly inactive correction. This differs from the older historical helper that included inactive zeros in its residual design. The retrospective earlier shot-quality bridge is not substituted.

## Source reconstruction and coverage

| Evidence | Result |
|---|---:|
| RS/PS source pairs, ending 1997–2026 |60|
| New actualFG+FT/Net pairs, ending 1997–99 |6|
| Hash-identical reused2000–26 pairs |54|
| Raw canonical rows / games |7,157,906 /37,979|
| Admitted rows / games |7,108,645 /37,845|
| Early1997–99 admitted rows / games |588,430 /3,284|
| Exact game-scoped1997 alias admissions |15|
| Early scoring quarantines |14|
| Additional early games lost to unresolved identities |9|
| Maximum standard/TD550 row Net error |8.89e−16|

Existing early processed `Net_Diff` includes expected FT, so it was not relabeled actual scoring. Six files were reconstructed through the canonical price-free row helper. Simmons/Booker aliases require exact game/ID keys and matched raw shooting/FT/TOV, canonical source and official identity/team evidence; other administrative identities remain censored. Whole-game scoring quarantine also removes matching pricing games, retaining raw first-chance prices for the admitted game population.

Two failed date wrappers remain preserved. The successful `vpm_historical_targets_1997_dated_v2` uses a locally pinned raw/canonical-agreed game calendar and exact97/98/99 century resolution. It does not change numerator, lineup or target math. All 3,307 early game dates agree between raw and canonical sources. Original54 pairs and source files are unchanged.

Actual January 1, 1997 prefix verification retained 73,754 rows/411 games. Appending later source files and perturbing 7,081,508 future atom rows plus 37,557 pricing rows changed no retained rows, slots, prices or targets; maximum gap 0.

## Interpretation and limits

These are completed-season research estimates, not preseason forecasts, live player availability, a certified roster, or a new model selection. `active_season` means positive exposure in the retained annual target sample; excluded games mean it is not an exhaustive official appearance census. Historical inputs, source vintages and era-dependent admission differ from selected v61. No matched predictive benchmark was run for this history; its numerical checks do not establish a forecasting gain or an era-neutral greatest-player ranking.

The 4,980 saved upstream head/year slots contain 2,476 bootstraps and 2,504 past-only annual fits. Only 38 fits report convergence; 2,466 stop at their fixed optimization budget. These are the full-history library recipes, not the separate 187 neutral CORE fits. No optimal-decay claim follows. Current selected model, prior experiment recipes, public/live wiring and2027 outcomes remain unchanged.

## Reproduction and receipts

From sibling VPM, using the RAPM runtime:

```bash
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  ../rapm/.venv/bin/python -u vpm/run_harness.py \
  --config vpm/configs/research_harness_v73_history_1997.json \
  annual-vpm-history --stage all
```

The controller uses an exclusive lock and immutable, hash-checked checkpoints. Do not duplicate a healthy job or refit completed stages. Registration froze5803 transitive source/code pins before any mapper fit, SHA `c0c5f75ecbc4c23015a9b0bcbef3d4da6a745dc4aa1e37dfb92461c4b22818f4`.

RAPM evidence:

- Target package: `nba_pipeline/results/vpm_historical_targets_1997_dated_v2/manifest.json`, SHA `48da3d93e15d1704f57e8e9f64f0cdc0bc132cd1d00203a1cb3f36c2e3c8bc42`.
- Actual early prefix audit: `nba_pipeline/results/estimated_skills/historical_targets_1997_prefix_audit_v1/receipt.json`, SHA `b954cc123dc6bce5c75223a2fc61484f7f6550274fda589c05983665b49ea4a7`.
- Feature export: `nba_pipeline/results/estimated_skills/vpm_1997_features_v1/receipt.json`, SHA `7df07b66f1d548844dfb3fc6a5668ee57f8170393ee4a17a28cdd72c4c2a9d49`.
- Full feature audit: `nba_pipeline/results/estimated_skills/historical_feature_bridge_full_export_audit_v1/receipt.json`, SHA `2382f31922654a5636d352fcfb26c061c075e4e53a9a084734eff9244d2d129c`.
- Independent model/board audit: `nba_pipeline/results/estimated_skills/vpm_1997_independent_audit_v1/actual/receipt.json`, SHA `104e436f4202217e6e0aeac6ffb5b97325c3e0a3572983202e5d60bb591de2bb`; all 30 boards,26 models, 29 cohorts and 5,972 fresh pre/post pins pass.
- Compact committed [source evidence](evidence/vpm_1997_history_source.json) and [model evidence](evidence/vpm_1997_history_model.json).

Code checks: 32 shared/early-target, 16 feature-bridge, 22 runner/basis and 17 reader tests pass. The independent audit has nine synthetic contract checks and a complete actual-output pass; consumer/browser verification is separate from statistical-model verification. Both are complete: the actual reader preserves every saved value and CSV byte, and browser checks cover search, sample filters, sorting, Net layers, precision tables, CSVs and the corrected default chart gaps.

## Reader verification and completion

The final local entry point is [the 1996–97 view](http://127.0.0.1:8798/seasons/1997.html); the self-contained files remain in sibling `vpm/results/harness/vpm_annual_1997_reader_v2`. The earlier reader is preserved as the numerical source of the presentation-only update. All 31 canonical data payloads and 31 CSVs remain byte-identical across reader versions. Jordan's default history contains 1997, 1998, 2002 and 2003, with disconnected gaps; the archive toggle reveals all saved rows without changing them.

- Final consumer certificate: RAPM `nba_pipeline/results/estimated_skills/annual_history_1997_reader_final_v2/receipt.json`, SHA `344a3eba3c506d1884d41553920494afaf8ea45858fe438515e3b7888cd1dd62`.
- Final reader receipt: `vpm/results/harness/vpm_annual_1997_reader_v2/receipt.json`, SHA `f3eefcb4a289f0b619cef48c382c8b0941e0df4de710c88eef6022134a7a302b`.
- Independent consumer audit: RAPM `nba_pipeline/results/estimated_skills/annual_history_1997_reader_active_consumer_audit_v1/receipt.json`, SHA `53ef0fc37345313d51a96c86367cd638b4a1d74c0668bcabcff137cdf26ff9cc`.
- Actual browser audit: RAPM `nba_pipeline/results/estimated_skills/annual_history_1997_reader_browser_v2/receipt.json`, SHA `28d687dcc1136f1dcf86456077ff2c8f99245b7824530ba7eaf0024d190c652f`. Core reader interactions also retain the first browser audit `annual_history_1997_reader_browser_v1/receipt.json`, SHA `a9922ec1f64a9245681773c4909dbeabaaaa29bcda59a16dc6c1bf2e3c068bf2`; native select-handler navigation used an explicit DOM change-event fallback after the CDP-controlled native popup rejected keys. The v2 archive checkbox was verified by coordinate click.

The scientific producer's immutable manifest intentionally retains its `produced_30_season_history_pending_independent_audit` marker. A separate ignored `independent_completion.json` binds the final report and independent source/model/consumer/browser receipts. Do not rewrite the frozen producer marker to manufacture completion or rerun the successful fits. The result is a verified historical research library for inspection; retain selected v61 for the existing current model.

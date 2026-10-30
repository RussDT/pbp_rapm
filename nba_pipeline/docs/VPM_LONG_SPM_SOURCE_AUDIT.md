# Long-sample statistical prior: source and design audit

Audited October 15, 2026 for the owner-requested separation of predictive
skill estimation, long-sample impact calibration, and forward selection of
lineup/prior strength. This is a feasible implementation contract, not an
EPM replication or a predictive result. Preserve v39/v40, the true no-prior
control, and the failed v43 forward-mean candidate. No 2027 outcomes enter.

## Recommended first control

At each emission cutoff, use actual-point O/D lineup evidence over the
preceding six completed seasons. Pair each player's pooled lineup evidence
with that same player's aggregate observed statistical rates from exactly
those seasons. Fit a statistical mapping while profiling the joint lineup
residual; retain its full normal matrix and covariance structure. Save a
zero-prior pooled RAPM target as an audit, rather than treating its player
coefficients as independent observed labels. Apply the mapping to dated neutral
estimated skills at the emission cutoff. Finally compare fixed prior scales,
lineup penalties and decay chronologically using completed earlier forecast
seasons. Keep the repaired SQ residual correction unchanged.

This deliberately separates *estimating current skills* from *pricing skills
against long-sample impact*. It changes the mapper target and its calibration
features; it does not establish that each feature has passed an independent
incremental forward-impact gate. The existing skill filters predict later
measurements of their own statistics. That is useful evidence, with a
different question from incremental future impact.

## Available sources and years

- `results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet`:
  387,962 rows, seasons 2014–2026, October 29, 2013–June 13, 2026. Raw observed
  makes/attempts, assists, turnovers, rebounds, steals, blocks, fouls and
  side-specific possession exposures; no fitted EPM response needed.
- `results/monthly_vpm_targets_clean_sq_v2/atoms_{year}_{RS|PS}.parquet`:
  seasons 2016–2026. Use actual FG+FT points and native O1–O5/D1–D5 lineups
  for the calibration labels. Do not use SQ probabilities, expected FT, or
  existing forward-label estimates as this long-sample response.
- The new RAPM-owned `scripts/export_long_spm_history.py` adapter reconstructs
  actual FG+FT rows for 2014/2015, RS and PS, on those years' exact canonical
  RAPM boundaries. This permits full six-season 2014–2019 calibration at the
  2019-10-01 cutoff. It disables the optional shooter-stat provider fetch,
  forces actual FT mode and zero luck adjustment, and preserves all canonical
  rows with an explicit `valid_lineup` flag. Its receipt is source evidence,
  not a new fitted model. Earlier forecast warm-ups may still have fewer than
  six seasons.
- `results/estimated_skills/monthly_defense_v1/rim_counts.parquet`:
  314,552 validated player-game rows, seasons 2014–2026. Early cumulative
  observations have already passed consecutive-appearance recovery gates.
- `results/estimated_skills/monthly_defense_v1/dfga_counts.parquet`:
  340,877 validated rows, seasons 2014–2026; use measured matching exposure.
- `results/estimated_skills/vpm_monthly_neutral_v1/features.parquet`:
  127,351 player-cut rows with 97 dates, annual early cuts and monthly cuts
  from January 2019. These are current-state emission inputs. Averaging their
  saved cuts to construct calibration-period statistics would unevenly sample
  the early and modern years and would not match lineup exposure.

The following inventory includes the 2014/2015 warm-up extension. Year labels
are season-ending years. Counts are positive-exposure raw
source players, before exact target-universe intersection; they are not a
roster or a minimum-sample selection recommendation.

| Emission cutoff | Matched calibration seasons | Positive-exposure players | O/D players with ≥2,000 side possessions |
|---|---|---:|---:|
| 2019-10-01 | 2014–2019 | 974 | 598 / 597 |
| 2020-12-01 | 2015–2020 | 1,019 | 601 / 601 |
| 2021-10-01 | 2016–2021 | 1,036 | 602 / 602 |
| 2022-10-01 | 2017–2022 | 1,085 | 614 / 613 |
| 2023-10-01 | 2018–2023 | 1,088 | 622 / 623 |
| 2024-10-01 | 2019–2024 | 1,071 | 610 / 610 |
| 2025-10-01 | 2020–2025 | 1,069 | 609 / 609 |

The latest six-year window has 94.26% measured rim-defense exposure coverage
and 99.07% DFGA coverage against raw line defensive exposure. Report missing
coverage rather than replacing unobserved defensive counts with zero. These
are aggregate coverage rates, not player-level reliability guarantees.

**Important identity gate:** raw game lines contain anchor-only IDs with zero
on-court exposure. In 2020–2025 there are 1,962 raw IDs, but 893 have zero total
offense-plus-defense exposure. Intersect with the exact lineup target universe,
require positive denominator for each rate, and never count these IDs as NBA
training players simply because a raw row exists.

### Completed actual-point warm-up export and scoring gate

`results/long_spm_actual_warmup_v1` contains four immutable exports. Every row
matches the canonical game's event, date and ten lineup slots exactly; all
493,971 rows pass the full unique-player lineup gate.

| Season | Phase | Possession rows | Games |
|---|---|---:|---:|
| 2014 | RS | 231,302 | 1,230 |
| 2014 | PS | 16,214 | 89 |
| 2015 | RS | 231,125 | 1,230 |
| 2015 | PS | 15,330 | 81 |

The separate `scripts/audit_long_spm_history.py` prices exactly deduplicated
raw made FGs and FTs independently of possession construction. The pinned
`score_accounting.json` checks 2,630 games; 2,626 match exactly. Four inherited
canonical-boundary discrepancies are quarantined from **both calibration
lineup evidence and matching-period observed feature aggregation**:

- `0021300316`: canonical actual151 versus raw152 (−1).
- `0041300305`: canonical actual182 versus raw183 (−1).
- `0021400843`: canonical actual189 versus raw191 (−2).
- `0021401185`: canonical actual175 versus raw177 (−2).

The original exports and canonical files remain unchanged. The accounting
receipt pins raw, canonical, export, export-manifest and audit-code hashes.
This is a raw-event scoring check, not a claim that every archived scoreboard
snapshot is complete. The adapter/audit have 21 focused passing tests covering
identity alignment, actual FT enforcement, provider-hook restoration, source
and artifact locks, invalid lineup flags, protected seasons, independent raw
make/miss pricing, exact deduplication and unmatched game populations.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/export_long_spm_history.py
.venv/bin/python nba_pipeline/scripts/audit_long_spm_history.py
.venv/bin/python -m pytest nba_pipeline/tests/test_long_spm_history.py -q
```

## Feature definitions and temporal fences

Aggregate counts before dividing; never average per-game percentages. Match
RS/PS selection and game IDs to the calibration population. Save any
differences between source and target side exposures as an audit, rather than
silently substituting denominators between feature definitions.

Use the existing own-skill units: offensive counts per 100 offensive
possessions; defensive rebounds, steals and blocks per 100 defensive
possessions; fouls per 100 *total* offensive-plus-defensive possessions;
usage = 100 × (FGA + 0.44 FTA + TOV) / offensive possessions. Makes/100 must
equal attempt-rate times make probability. Coherent imputation must preserve
that identity at training and emission.

For rim defense, use only validated row counts and their own measured
defensive exposure. Compute `q` as `(65 + all earlier-day league rim makes) /
(100 + all earlier-day league rim attempts)`, matching `lagged_league` in
`build_predictive_rim_saved.py`. Retain the pre-window 2014-onward history for
this baseline; aggregate `200 * (q * attempts - makes) / exposure` into the
matched window. This is a cumulative earlier-game-day league baseline, not an
immediately preceding-season baseline. Never recover
counts by differencing percentages. Rim suppression, contest rate and saved
points are mechanically related; missingness and joint shrinkage matter.

Every calibration game, source observation, imputation parameter and label
fit must precede the emission cutoff. Same-period outcomes are legitimate
*training* information for retrospective impact calibration, because the
entire matched period is already complete. They may not enter an earlier
forecast. Fit a fresh mapper at each cutoff; never fit 2020–2025 coefficients
and apply them to the 2020 forecast. Actual-only calibration does not require
SQ continuation repricing. The final VPM still uses its existing pre-cut
continuation and repaired SQ contracts.

## Statistical choices and limitations

1. **Pooled target versus player-year panel.** A single six-year coefficient
   per player actually pools lineup evidence. Repeating annual noisy RAPM
   labels in a six-year player-year table does not reproduce it. The pooled
   version averages changing talent and roles; matched aggregate statistics
   should represent that same average. A joint player-year model with temporal
   partial pooling is a later, separate architecture, not a free equivalent.
2. **Shrinkage is still present.** Multi-year labels have lower sampling
   variance, but zero-prior ridge labels remain shrunk. Exposure-weighted
   regression does not undo that shrinkage. Save label penalties and target
   normal equations; do not describe fitted slopes as unbiased causal prices.
3. **Covariance matters.** Teammate/player O/D labels share lineup uncertainty.
   Diagonal possession weights cannot represent it. The accepted implementation
   direction profiles the pooled lineup residual jointly and keeps the full
   normal matrix; a simple regression against saved player labels would be a
   different, weaker control. Joint profiling addresses the correlated solve
   without making observational impact coefficients causal or noise-free.
4. **Preserve the response basis.** Unpenalized season/intercept terms and
   offense/defense signs must match the solver. If covariance corrections are
   added, include the off/def cross block; independent regressions cannot
   pretend correlated target errors disappeared.
5. **Do not multiply repeated evidence.** Reusing overlapping six-year windows
   as independent mapper observations exaggerates sample size. The clean
   first version uses one matched pooled window at each cutoff. Forecast-year
   comparisons remain paired and season-aware.
6. **Same-period → estimated-state transfer is an assumption.** Observed
   long-period rates have different measurement error and context than current
   neutral skill estimates. Forward tuning of scale/precision can reveal a
   poor transfer; it cannot prove every slope is a structural basketball
   constant. Do not add manual 1.5× inflation without a declared comparison.
7. **Feature columns are intentionally redundant.** Raw usage is exactly the
   sum of the three zone attempt rates, 0.44 times FT attempts, and turnovers.
   Rim savings/volume/suppression and shooting makes/attempts/probabilities are
   also linked. The regularized mapping must tolerate these dependencies;
   individual unconstrained coefficients cannot be interpreted in isolation.

Select only using permitted completed development forecast seasons, then
report the already spent 2023–2026 stress seasons. Early short-window boards
must be identified as warm-ups. No claim of untouched historical confirmation,
certified historical publication availability, or EPM superiority follows from
this reconstruction. The first true future confirmation remains 2026–27.

Inspected implementation references: `build_monthly_own_skills.py`,
`assemble_monthly_own_skills.py`, `build_monthly_defensive_skills.py`,
`build_nba_estimated_skills.py`, `build_monthly_vpm_targets.py`, sibling
`harness/own_parent.py`, and immutable sibling config v39.

# Tracking playoff quartiles

Owner-directed exploratory audit, September 13, 2026. Run `.venv/bin/python nba_pipeline/scripts/analyze_tracking_playoff_quartiles.py`. Receipts, source hashes, report.md, quartiles.csv, coverage.csv and player_seasons.parquet go into ignored `results/estimated_skills/tracking_playoff_quartiles/`. No production model changes.

## Contract

Use the tracking-backfill source panel but only measured PA/DFGA observations, never synthetic repairs or inferred omitted zeros. Join exact player/date/season/phase to resolved PBP exposure, reject duplicate exposure keys, require AST/FGA agreement and minutes within 0.5. Missing observations/exposure are not zero. Require at least 95% source-minute coverage in both phases and 200 observed RS possessions. Keep every positive-exposure eligible postseason, reporting its sample size. The contested-rebound endpoint's omissions remain unresolved, with substantial player attrition despite roughly 94–96% overall minutes coverage; selective missingness is a limitation. These gates do not prove every recorded tracking value is correct.

Offensive events use offensive possessions; defensive events use defensive possessions. Possession time is minutes with the ball per100 offensive possessions, not an on-ball percentage. Study 2020–2025; 2019 supplies RS-only warm-up means. No 2026 targets or 2027 data are inspected.

For six stats without fitted next-game skill models, rank all eligible RS players, including non-playoff players, before the league's first playoff game using `(RS count + 200 * 2019 RS mean)/(RS exposure + 200)`. The 200-possession stabilization is a fixed exploratory convention, not tuned or a validated estimated-skill model. Quartiles have equal player counts; ID order breaks ties. Deflections instead inherits the earlier pre-playoff OU skill quartiles. Match those ranks to eligible PS participants. Pool same-player phase contrasts with `E_RS*E_PS/(E_RS+E_PS)`. This weights matched exposure, not seasons equally. Report 2020–2022 and already-spent 2023–2025 separately as descriptive checks.

Intervals resample players with all their seasons, 2,000 draws, fixed seed. They are pointwise and not multiplicity-adjusted. Selection on RS rate can induce regression to the mean even with shrinkage; higher-rate declines alone do not establish a playoff interaction. No new forecast-error results are claimed for the six stats without fitted models. The existing deflection diagnostic remains the sole forecast-error check. A future predictive test must compare against updating baseline forecasts and a regular-season change control before any adjustment is admitted. No significance-driven search or production promotion follows from this audit.

## Initial results

Highest-quartile matched rates, RS→PS, per100 relevant possessions:

| Stat | RS | PS | Change |
|---|---:|---:|---:|
| Potential assists | 15.85 | 14.17 | −1.68 |
| Drives | 17.77 | 17.48 | −0.29 |
| Possession minutes | 7.73 | 7.75 | +0.02 |
| Contested defensive rebounds | 3.71 | 3.27 | −0.44 |
| Contested shots | 17.52 | 17.55 | +0.03 |
| DFGA | 26.51 | 26.50 | −0.01 |
| Deflections | 4.43 | 3.95 | −0.48 |

Potential assists fall across all quartiles; contested DREB rises in the bottom two and declines at the top. Possession minutes fall in the bottom three while remaining flat at the top. Drives show larger declines in the middle quartiles. Top-quartile PA, contested DREB and deflections declines occur in both historical periods. Flat or uncertain patterns remain in the report rather than being dropped. The deflections numbers differ slightly from the earlier audit because this cross-stat report adds the 95% phase coverage gate.

Regression test verifies that recorded zeros remain observations while missing values and unmatched possession denominators do not enter either numerator or exposure.

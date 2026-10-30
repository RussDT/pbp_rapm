# zTS year-to-year stability

Completed September 9, 2026. This is a retrospective audit of the existing
production formula, not a new calibration or an untouched predictive test.

## Finding

**zTS has modestly smaller year-to-year changes, especially when scoring roles
change. It does not consistently improve year-to-year correlation.**

Across 4,450 consecutive-season pairs from 912 players, average absolute
movement falls from 3.039 to 2.915 percentage points versus production rTS:
a 0.125-point / 4.1% reduction. The paired player-bootstrap 95% interval for
the reduction is 0.100–0.150 points. RMSE of annual changes falls 4.1%, from
3.901 to 3.741 points. Average absolute movement is smaller in all 19 annual
transitions, with no formula changes or fitted stability coefficients.

| Metric | Year-to-year Pearson r | Average absolute change, pp | RMS change, pp |
| --- | ---: | ---: | ---: |
| Conventional TS% | 0.638 | 3.055 | 3.914 |
| Production unadjusted TS% | 0.639 | 3.082 | 3.946 |
| Production league-relative TS (rTS) | 0.587 | 3.039 | 3.901 |
| Production zTS | 0.575 | 2.915 | 3.741 |
| Difficulty adjustment alone | 0.904 | 0.650 | 0.855 |

The zTS-minus-rTS correlation difference is -0.012
(95% interval -0.038 to +0.013). A claim of higher overall correlation is
unsupported. Pooled raw TS correlation also includes between-era scoring
levels; rTS is the appropriate comparison for isolating the role adjustment.

## Role-change diagnostic

Role change means the absolute change in the published difficulty adjustment,
in TS percentage points. These groups use both seasons' measured role, not
preseason information. Cutoffs were fixed before the first stability run.

| Difficulty change | Pairs | rTS average change | zTS average change | rTS r | zTS r |
| --- | ---: | ---: | ---: | ---: | ---: |
| At most 0.5 pp | 2,191 | 2.839 | 2.829 | 0.618 | 0.593 |
| Above 0.5 through 1.5 pp | 1,900 | 3.109 | 2.964 | 0.578 | 0.552 |
| Above 1.5 pp | 359 | 3.890 | 3.174 | 0.482 | 0.552 |

Large role changes show an 18.4% reduction in absolute movement, or 0.716 pp
(95% interval 0.513–0.910). Their correlation difference is +0.069
(+0.019 to +0.125). Stable-role absolute movement differs by only 0.010 pp,
with its interval including zero.

This is consistent with the intended role adjustment: difficulty changes
offset some of the observed efficiency changes. It is not a causal test of
role or proof of portable shooting talent. The grouping variable is itself
part of zTS, and role is not randomly assigned.

## Robustness and conflicting evidence

- **Attempt thresholds:** requiring 100, 500, or 1,000 TSA in each season
  retains smaller absolute changes. At 1,000 TSA, correlations are 0.726
  for rTS and 0.743 for zTS, but the difference interval includes zero.
- **Weighting:** harmonic-TSA weighting gives correlations 0.620 versus
  0.630 (difference inconclusive), with a 0.111 pp reduction in absolute
  movement. Equal-season weighting gives a 0.125 pp reduction. These are
  sensitivity checks; equal player-transition weight is primary.
- **Era:** within 2006–2013, rTS/zTS correlations are 0.556/0.599; within
  2014–2025 they are 0.605/0.557. The modern difference is -0.048
  (95% interval -0.079 to -0.017). Both eras have smaller absolute changes.
  The 2013→2014 bridge is included in the full sample but neither era subset.
- **No estimated modern playtype rows:** 473 pairs from 225 players retain
  a 0.140 pp reduction (interval 0.064–0.223), while the correlation
  difference is inconclusive. At most 5% estimated TSA in each season gives
  2,063 pairs and a 0.115 pp reduction; correlation is again lower for zTS.
- **Listed team label:** same-label pairs move 2.949→2.834 pp; different-label
  pairs move 3.230→3.085 pp. These are annual roster-label diagnostics, not
  transaction-aware trade classifications. A one-team label can mask an
  intra-season trade.
- **Season-cluster uncertainty:** resampling the 19 target seasons instead
  of players gives a 0.103–0.148 pp interval for the primary reduction;
  correlation remains inconclusive. This was added after the first inspection.
- **Post-inspection role/quality intersection:** among 133 modern large-role
  pairs with at most 5% estimated TSA each season, movement falls from
  3.920 to 3.249 pp. The reduction interval is 0.363–0.957 pp, while the
  correlation advantage is inconclusive. This is exploratory corroboration,
  not another independent confirmation sample.

## Why smaller movement can coexist with lower correlation

Correlation depends on the spread between players as well as the movement
within players. zTS narrows the cross-player distribution: first-season
standard deviation falls from 4.260 pp for rTS to 4.030 pp for zTS. A smaller
average annual change therefore does not necessarily yield higher correlation.
Simple shrinkage toward a common number could also reduce absolute movement,
so that result alone does not establish better talent estimation.

The exact change accounting is informative:

```text
ΔzTS = ΔrTS + Δdifficulty
mean(ΔzTS²) = mean(ΔrTS²) + 2 mean(ΔrTS × Δdifficulty) + mean(Δdifficulty²)
13.9935     = 15.2199      - 1.9577                       + 0.7313
```

Changing difficulty offsets more efficiency movement than the adjustment
adds. A perfectly constant role label could increase cross-player correlation,
but could not reduce a player's absolute year-to-year movement. These results
support role-related stabilization specifically; they do not identify latent
ability or establish the production coefficient as optimal.

## Data and methodology

- Source: cached public production `player_zts` snapshot in
  `results/historical_scoring_security/databallr_synergy_zts_2006_2026.csv`,
  matched by exact `nba_id + year` to core `csvs/merged_rs_full.csv.gz`.
  All rows are regular season. This uses the available publication panel,
  not a certified complete NBA player census.
- Use 2006–2025, with 9,589 source rows and 9,416 complete usable player-seasons
  before attempt thresholds. Historical 2006–2013 is the observed Synergy
  historical bridge, not the 1997–2005 PBP backcast.
- **Exclude 2026 entirely:** the source snapshots disagree in rTS by up to
  33.83 pp and in minutes by up to 582. That is a vintage mismatch, not a
  basketball result. No selective retention of apparently matching 2026 rows.
- Conventional TS is `100 × Points / (2 × (FGA + 0.44 × FTA))`.
  Production TS uses actual TSA and removes expected technical-FT points:
  `100 × (Points − 0.8 × Technical Free Throw Trips) / (2 × TSA)`.
  Production rTS subtracts its stored season league reference. Keep the two
  TS definitions distinct; do not attribute their difference to playtype.
- zTS is `precise production rTS + published role`, preserving coefficient
  1.0. The only change from published zTS is removal of CSV rounding:
  published closure is within 0.001 pp; source rTS matches within 0.000501 pp.
- Match only consecutive seasons, requiring at least 250 **production TSA**
  in both seasons for the primary sample. The same exact rows and weights
  enter every metric. All rates and differences are in percentage points.
- Primary uncertainty: 2,000 paired bootstrap resamples of player IDs,
  retaining all transitions belonging to a sampled player. Seed 90317.
  Intervals are nominal, not adjusted for the many diagnostic comparisons.
  Separate season clustering checks shared era effects, but neither bootstrap
  is a complete model of teammate dependencies.
- Modern lower-level `playtype_rows_with_estimates.csv` reconstructs role
  within 0.000501 pp and provides estimated-TSA shares. Production repairs
  may use surrounding seasons, including later seasons. Even a no-estimated-
  row subset does not establish that all shared category prices were as-of.

**Within-season split-game repeatability remains unmeasured.** Available
production playtype inputs are player-season aggregates without game IDs.
Applying one full-season difficulty adjustment to both game halves would
share information and artificially reward stability; substituting the PBP
proxy would test a different metric. A valid follow-up needs independently
constructed game-level production playtype inputs and independently priced
halves, or frozen external prices.

No fresh holdout was opened, no coefficients were tuned, and no production
table or consumer was changed. Keep 2026–27 untouched for prospective tests.

## Reproduction

From the RAPM repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python nba_pipeline/scripts/analyze_zts_stability.py
.venv/bin/python -m pytest nba_pipeline/tests/test_zts_stability.py -q
```

The runner accepts `--zts`, `--box`, `--roles`, `--output`, and `--bootstrap`.
Generated artifacts stay gitignored under `results/zts_stability/2006_2025/`:

- `manifest.json`: definitions, source hashes, exclusions, and diagnostic labels.
- `source_audit.csv`: annual source agreement and coverage.
- `player_seasons.csv`, `matched_transitions.csv`: auditable matched input rows.
- `metrics.csv`: correlations, changes, spread, and bootstrap intervals.
- `paired_comparisons.csv`: paired zTS-minus-baseline intervals.
- `change_decomposition.csv`: exact change accounting.

Validation: three focused tests pass for consecutive-season matching, the
distinction between correlation and absolute movement, and paired resampling
with identical comparators; live data identity and snapshot checks also pass.

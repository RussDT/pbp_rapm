# Quarterly predictive VPM from 2014

Owner request, October 20, 2026: investigate MPG, make a decision, then emit
predictive VPM four times annually from 2014, including exact Net lanes. This
is the existing NBA predictive VPM research project, not a second project or
permission to publish. Completed October 21: 52 audited cuts, 105,268
player-date rows and 2,649 players. The historical reader is
[available locally](http://127.0.0.1:8768/quarterly-history.html).

## MPG decision

The matched selected-model test retains MPG. It refits both dated mappings
with MPG and its flag removed and uses the same clean v61 entrant priors,
parent penalties, decay and SQ. Development minimum and historical stress
favor keeping MPG. Native final-possession gain remains unresolved. The raw
appearance count is exposed for reliability inspection; no arbitrary new
MPG shrinkage constant was adopted. See [decision](VPM_MPG_DECISION.md).

## Requested history contract

- Exactly 52 report cuts: January 1, April 1, July 1 and October 1 in each
  calendar year 2014 through 2026. No 2027 outcomes.
- One selected historical view of Offense, Defense and Total, full-precision
  exports, prior values separated from final ratings, and player histories.
- Learned own, population-background and combined priors plus final ratings
  all decompose exactly into Net shot/FT, turnover and second-chance lanes.
  Mid/three EV and residual children stay separately inspectable.
- Standard five-shot midrange-reference convention; positive-good defense;
  full-universe side-specific TD-exposure centering with a
  prediction-preserving intercept. Never recenter a filtered player list.
- Unknown player ID zero, duplicate players within a side and cross-side
  duplicates are excluded from lineup estimation. The coverage receipt must
  quantify those observations; no fictitious player zero is fitted.
- Every lineup ID must have an official positive NBA appearance in that
  source season's RS or PS totals, with no minimum-minute requirement.
  This preserves Chris Smith and playoff-only Luca Vildoza while excluding
  22 unadmitted administrative identities propagated into 183 lineup rows.
- Sources observed strictly before the cut. Fitted prior maps use only
  fully matured 365-day subsequent-lineup training cohorts whose label ends
  are no later than the emission date.

## Availability and early inputs

The current selected OU input packages begin too late to call January 2014 a
replay of that saved board. New fixed-EB historical inputs use the same
physical units at every cut, from observed box/PBP counts with TD550, rate
shrinkage K1000 and percentage shrinkage K100 attempts. Those constants are
declared retrospective design choices, not historically optimized fits.

The old raw PBP descriptions do not support daily STL/BLK extraction.
Official annual integer STL/BLK and full official POSS supply those fields
only after the completed season, using a conservative July availability
boundary. Missing tracking remains missing, never a zero observation. The
rich EB recipe keeps a named total-DREB fallback until an admitted contested
DREB coefficient has an observed state. Aggregate PF likewise remains until
an admitted PF subtype is observed. Raw tracking arrival alone cannot remove
a usable aggregate before its replacement mapper has matured. The same
transformation runs before calibration preprocessing and at deployment;
observed zero is a real replacement measurement.

The legal OU comparison uses the original past-only annual parameter
vintages from October 2015 onward. It requires three fully matured cohorts
before emission. No modern filter parameters are applied to 2014 and called
vintage-safe. Input families are compared on shared forward actual outcomes,
not on how familiar the resulting leaderboard looks.

Early entrants use an NBA population fallback fitted directly through the
past lineup likelihood, transitioning to own priors by 2,000 prior NBA
possessions. This does not claim college/G League measurements exist for
those early dates. Unpenalized background intercepts are identified from
the same likelihood, never averages of zero-shrunk player labels.

## Targets and source gates

RAPM owns [historical target preparation](VPM_HISTORICAL_TARGETS.md) and
[historical inputs](VPM_HISTORICAL_FEATURES.md). The target adapter audits
raw FG/FT scoring and excludes entire mismatched games from both target and
pricing populations. The raw-score audit must identify MISS as a word, so a
made free throw by a player named Missi is not treated as a miss.

Continuation and shot reference counts deliberately use the raw
first-chance population before terminal/lineup filtering in the admitted
games. Reconstructing those prices from only retained 5v5 observations
would omit incomplete-possession continuations and change canonical Net.

Audited clean chronological SQ residuals are available from the 2015–16
season onward (season ending 2016). Earlier
corrections are explicitly inactive; zero correction is not an observed
zero luck value or a fabricated historical tracking estimate. The richer
retrospective shot models are not silently substituted.

## Model and comparison

Sibling immutable config:
`vpm/configs/research_harness_v65_quarterly_2014.json`.
Command: `quarterly-vpm --stage prepare|cohorts|boards|evaluate|assemble|all`.
Output: `vpm/results/harness/vpm_quarterly_2014_v3/`.

From the sibling VPM repository, with the paired RAPM data/runtime present:

```bash
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
  ../rapm/.venv/bin/python -m vpm.harness.cli \
  --config vpm/configs/research_harness_v65_quarterly_2014.json \
  quarterly-vpm --stage all

../rapm/.venv/bin/python vpm/scripts/audit_quarterly_history.py \
  --config vpm/configs/research_harness_v65_quarterly_2014.json --stage all

../rapm/.venv/bin/python ../rapm/nba_pipeline/scripts/historical_targets_native_audit.py \
  --out vpm/results/harness/vpm_quarterly_2014_v3
```

Resume uses hash-verified immutable cut/cohort artifacts. It refuses changed
prepared inputs; a source or methodological change requires a new experiment.

The parent keeps O/D 4000/8000, TD550, SQ residual 12000/48000 and mapper
O10/D100. Candidate recipes are core EB, expanded EB, legal OU expanded and
the same OU own priors with dated rich college/G League/draft backgrounds.
The rich entrant coefficients are re-estimated on the newly admitted actual
RS targets with the corresponding past-only scalers; no archived outcome
coefficients or later lower-league counts are borrowed. Own priors and
established-player combined priors remain unchanged in that increment.
The outcome-profiled prior calibration preserves full lineup covariance and
conditional defensive calibration. Component RHSs add to the parent RHS,
so prior and final Net sums are algebraic, not allocated after fitting.
This is the forward-profiled Parent approach: a dated skill state is trained
against its following 365 days of lineup outcomes. It is not the earlier
six-year observed-stat SPM mapper. The history pools the fully matured annual
cohorts available at each cut, beginning October 2008 for EB and October 2015
for OU, with at least three cohorts required. Later evidence cannot alter an
earlier cut's fitted coefficients.

The declared comparison uses exact original preseason dates and common
actual minutes/pace for 2020–2026. Development 2020–22 selects the early
family among EB recipes and the later family among all legal candidates by
minimum equal-season game MSE. 2023–26 is spent stress, report only. These
are historical reconstructed estimates, not certified historical
publication vintages or pregame availability forecasts. The existing
selected current model and reader remain unchanged.

## Frozen seven-test results

The corrected v65 run exactly reproduces all benchmark game predictions
from v64. The declared development minimum selects `expanded` for the early fallback
and `ou_entrants` once legal OU calibration has three matured cohorts
(October 2018). Development game RMSE is 13.765142 / 13.718452 respectively;
the current selected v61 reference is 13.727018. The early and late recipes
are chosen separately from the same declared 2020–22 comparison, not from
later player appearances or stress outcomes.

Spent 2023–26 equal-season errors, lower is better:

| Test | Current selected mapper100 | Quarterly late recipe |
| --- | ---: | ---: |
| Game margin RMSE, points | 13.888098 | 13.881618 |
| Low-continuity game margin RMSE | 14.466517 | 14.421543 |
| Team win-fraction RMSE | 0.084123 | 0.083628 |
| Low-continuity team win-fraction RMSE | 0.110299 | 0.107293 |
| Team net-rating RMSE | 3.095818 | 3.073285 |
| Veteran-only game margin RMSE | 13.968663 | 13.958417 |
| Veteran-only team win-fraction RMSE | 0.087190 | 0.086660 |

All seven point estimates favor the late historical recipe, but the overall
game-MSE gain 0.179932 has interval [-0.042172, 0.410447], crossing zero.
Low-continuity games, low-continuity team win fraction and veteran-only
games have positive unadjusted paired intervals; the other four do not.
The rich-entrant increment over the same OU own priors improves team-net
MSE by 0.113842 [0.017800, 0.266426], while its overall game increment is
unresolved. This does not authorize replacing the current selected model.
The comparison against selected v61 is a full historical-recipe comparison:
the longer admitted lineup history and refreshed source gates also change.
It does not isolate an individual feature's contribution. The rich-entrant
comparison against `ou` does hold the new history and own priors fixed.

The benchmark retains 8,289 games and 210 team-seasons across 2020–26,
with exact reference game parity and team gap at most 1.78e-15. Independent
minute aggregation, all 98 scalar scores and all 140 paired point estimates
agree within 7.11e-15. Raw results are in `benchmark/scorecard.csv`,
`paired_comparisons.csv`, `selection.json` and `independent_score_checks.json`.
Forecasts use shared observed future minutes and pace; their intervals
condition on fixed fits and this selection. These remain reconstructed,
spent historical comparisons, not a prospective availability forecast or
untouched confirmation.

## Completed validation and outputs

Completed independent model/raw proofs cover all 37 distinct prior maps:
preprocessing, pooled information/RHS and conditional/direct-parent
coefficients reproduce exactly. Component-to-parent coefficient sums differ
by at most 1.27e-14. Fresh raw January2014/October2026 reconstructions cover
all six available endpoint family boards, using 3,340,227 / 6,520,215 admitted
rows and 1,432 / 2,649 players. Parent, centered, lane and prior gaps are at
most 1.34e-14. Exposure sums accumulated by a different sparse path differ
by at most 1.23e-9 on large possession masses. Earliest/latest cohort raw
normal/zero arrays reproduce exactly; all eight rich annual background
profile geometries pass. Receipts: `independent/audit_quarterly_history_models.json`
and `independent/audit_quarterly_history_raw.json`.

The final independent structure audit passes all 52 dates, 170 candidate
boards, 37 prior maps and 441 unique source files. All selected reader
values exactly match saved exports. The selected-reader algebra/reference
maximum is 9.40e-13 across all four bases and three sides. There are
105,268 player-date rows, 2,649 player histories and 582 recent players at
the final cut. The early recipe supplies 19 cuts through July 2018; the late
recipe supplies the remaining 33. Full-precision aggregate and per-date
rating/lane CSVs live under `quarterly-data/`; player histories are JSON.

The native forward comparison independently reconstructs 2,490,186 common
actual-FG+FT possessions over 39 nonempty next-quarter windows. Thirteen
windows have no eligible future data and are explicitly unevaluated. Against
a matched zero-impact-prior control, equal-nonempty-quarter MSE improves by
0.000500102 [0.000405494, 0.000589361], in squared points per possession.
The interval resamples whole games within quarters (15,810 game-quarter
pairs), conditional on frozen models/selection and known future lineups.
It does not establish future availability accuracy or full fitting uncertainty.
Endpoint predictions match within 1.55e-15; representative normal-equation
and prior-response checks match within 7.26e-14. All 110 prepared sources,
110 target source hashes and 108 target artifacts were reverified. Receipts:
`native_forward/summary.json` and
`independent/audit_quarterly_native_sources.json`.

The focused pass is 39 VPM + 32 RAPM tests, 71 total, with nine audit tests
rechecked after an audit-only season-filename filter fix. This is not a claim
about the entire repositories. Browser checks cover early/latest dates,
recent/full universes, all four value bases, search, sort, player lanes and
52-point history. Three downloaded CSVs match saved values within 1.78e-15.
Native select keyboard automation was unsupported in local CDP, so select
changes used DOM input/change events; clicks/search used coordinate/CDP
input. `browser_checks.json` records this explicitly. Five protected current
model/reader files retain identical SHA256s. Port 8768 serves the VPM reader;
8767 currently belongs to an unrelated local server.

Older v34 monthly artifacts and their SQ-lineage warning remain preserved
and are not used as the new history.

## Rejected availability transition and corrected rebuild

The first full v64 fit was stopped during native validation after a semantic
review found that raw tracking arrival could suppress aggregate DREB/PF
fallbacks before the matured mapper admitted replacement coefficients. Its
first January 2014 mapper contains aggregate fallbacks but no contested-DREB
or PF-subtype columns, while 424/433 current states already contain those
new sources. Treating these observations as replacements prematurely drops
known box information.

All v64 fitted outputs, partial native checks and its exact source bundle
remain archived under `vpm_quarterly_2014_v2/`, with an explicit
`rejected_availability_transition.json`. They were never released as the
selected history. v65 refits in a separate immutable v3 directory, uses
model-aware nonoverlapping fallbacks in calibration and deployment, and
adds regression tests for unadmitted replacements, observed zero, partial
subtype admission and nonmutation. Independent full-series verification confirms exactly fourteen changed
candidate boards (the first seven quarters in both early families), every
numeric later board value unchanged, all mapper coefficients/preprocessors
unchanged, and benchmark game-prediction values exactly identical. Early
core/expanded O/D/total ratings now agree within 4.44e-15 before tracking
coefficients mature. Receipt: `independent/availability_transition.json`.

## Reproduction backup and operator handoff

The code/config/templates/tests and dependency recipe actually used are
frozen in sibling `vpm/evidence/quarterly_history_source_v65.zip`, with
`quarterly_history_source_v65_manifest.json` and curated results
`quarterly_history_summary_v65.json`. This intentional source-only backup
contains 179 files, including 67 required pre-existing untracked dependencies,
and 565 external-data hash references. It avoids committing unrelated shared
working-tree edits. Raw data, fitted models and generated tables stay outside
git under the existing backup policy; the paired local data/runtime remain
required for reproduction. Workspace/bundle hashes and ZIP CRC passed.

ZIP SHA256: `b7244dcd481ff8f52c431ebc83d7961b469a848c3bfbf17a9cda8a99d18646f1`.
Manifest SHA256: `3b6c6879fb30610f4766563348f10db388d33d57f03f935766063883db812ef8`.
The source snapshot records dirty repo state and source HEADs VPM63e1043 /
RAPM95278b62; it does not pretend these repositories were clean.

Open `http://127.0.0.1:8768/quarterly-history.html`. The server serves the
existing selected reader folder, with `quarterly-data` linked to immutable
v65 assets. The selected current reader remains `ratings.html` on that port.
The history starts in 2014 under this owner contract; older available lineup
rows train matured priors but are not released as earlier quarterly ratings.
Live 2027 ingestion/refresh adapters and prospective team-win roster/minutes
readiness remain separate outstanding work; this receipt does not certify them.

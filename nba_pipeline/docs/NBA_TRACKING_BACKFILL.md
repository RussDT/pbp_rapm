# NBA tracking game-log backfill

Built September 12, 2026. This is an isolated postgame reconstruction for
estimated-skills preparation, not a new skill model or a RAPM/VPM input.
It repairs potential-assist outages and whole-team defended-attempt outages
in the local Gabriel `player_sheets` archive. Other tracking families are
carried through for inspection but are not backfilled by this command.

## Source and grain

`player_sheets/game_report/year_files/{2014..2026}[ps]_games.csv`, regular
season and playoffs, one played player/date row. The completed panel has
355,995 rows. Player/date uniqueness is required both within files and across
the combined panel. The archive's team IDs can contain traded-player fragments;
team outage rules require at least 180 summed player-minutes to avoid treating
small fragments as complete teams.

The builder tries the corresponding `all_games/*.parquet` copies, requiring
exact player/date plus matching minutes and assists, then preserved daily
endpoint caches for flagged outages. The audit records input hashes. The
first run checked 247 daily cache files and found no recoverable values.
The original files are never edited.

Final coverage: **1,586 potential-assist rows estimated**, with all flagged
PA gaps filled and 354,409 observed values preserved. DFGA has **746 modeled
outage rows**, **11,498 inferred omitted zeros**, and **1,492 unresolved
individual omissions** across all seasons. No cache recovery was available.

## Potential assists

Do **not** assume `POTENTIAL_AST >= AST`. An official assist can disagree with
tracking's potential-assist definition; a low count alone is not corruption.
Preserve these rows and expose `pa_below_ast_flag` for inspection.

Flag missing/negative potential assists, or zero potential assists plus zero
passes with independent activity (an assist, or at least ten touches in at
least five minutes). A team-level passing outage requires 180+ summed minutes,
100+ touches, passes below 35% of touches, and fewer potential assists than
official assists. Within such a team, only zero-passing/zero-potential-assist
rows are flagged; surviving positive observations are retained.

For 2026 this identifies **18 rows** in the March 5 DAL/ORL passing outage.
It preserves the two surviving passing observations and does not overwrite
the hundreds of otherwise normal rows with potential assists below assists.

## DFGA

The archive has no explicit overall DFGA zero rows. Null is therefore a
mixture of endpoint-omitted zeros and missing coverage, not simply a missing
positive count. Never replace every null with the average defended-shot rate.

- Recover overall DFGA from two-point plus three-point attempts only when both
  partitions are present and the sum is at least defended makes.
- A whole-team outage requires 180+ summed minutes and no observed DFGA for
  any player. These rows can receive model estimates.
- Infer an omitted zero only when the player played at most ten minutes,
  both shot partitions are missing, there are no positive defended makes,
  at least five teammates/players have DFGA, the team records at least 20
  defended attempts, and at least 85% of its minutes have DFGA coverage.
  This is explicitly an **inference**, not a recovered observed zero.
- Leave other individual omissions unresolved. In particular, a long stint
  without DFGA is not automatically a zero or a positive model estimate.

The 2026 panel has **86 estimated outage rows**, **1,046 inferred zeros**,
and **29 unresolved individual omissions**. The four outage dates are March
11, March 14, March 28, and April 3 (eight team-games).

## Estimator and timing

The fixed model is histogram gradient boosting: 100 iterations, 15 leaves,
minimum 80 rows per leaf, learning rate 0.06, L2 10, seed 20260912, and no
early stopping. Fit rates per 36 minutes, weighted by minutes, using actual
game minutes; per-36 AST/FGA/FTA/TOV/OREB/DREB/STL/BLK/PF; and a strictly prior
30-appearance tracking rate shrunk with 200 minutes of prior-date league
history. Missing observations contribute neither counts nor exposure to
history. Current tracking counts and other same-family tracking features are
excluded from predictors so an endpoint outage cannot contaminate them.

Each 2015–2026 model is fit before that season's first game. Player history
updates with earlier measured games only; league history excludes the whole
current date. The initial 2014 season uses expanding earlier-date fits for
gaps after at least 500 observed training rows. Unsupported early gaps remain
missing. Same-game assists and other box counts make this a **postgame
imputation**, not a pregame forecast.

## Validation and limits

The 2015–2025 sequential out-of-year checks compare predictions against
observed records with at least five minutes played. Potential-assist pooled
RMSE is **1.843**, versus **2.440** for prior player history and **2.618** for
twice official assists. The model improves on both in every evaluated season.
DFGA pooled RMSE is **3.126**, versus **3.188** for prior history; its benefit
is much smaller. All choices are fixed without scoring 2026 targets.

These checks simulate a missing current observation while earlier observations
remain available. They do not prove accuracy during long, systematically
missing intervals. DFGA validation conditions on recorded counts; omitted
zeros mean it is not a calibrated all-player count model. The estimated
standard deviation uses only earlier validation seasons' residual variance
per minute; it is unavailable for early fits and is not a calibrated interval
for every player or zero-omission case.

Snapshot acquisition timestamps are unavailable. Exact-date joins and causal
model fits do not establish that later source revisions were available live.
Keep the work research-only. Do not score or tune on 2026 outcomes; new skill
model confirmation remains 2027.

## Output contract and reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_tracking_backfill.py
.venv/bin/python -m pytest nba_pipeline/tests/test_tracking_backfill.py -q
```

Generated outputs stay gitignored under
`nba_pipeline/results/estimated_skills/tracking_backfill/`:

- `tracking_backfill.parquet`: original fields, repaired values, estimates,
  source/reason labels, fit cutoffs, uncertainty, and measured flags.
- `repair_ledger.csv`: every row where either target is not simply observed.
- `validation.csv`: held-out season checks through 2025 only.
- `audit.json`: source hashes, coverage counts and date-level validation errors.
- `review.ipynb`: inspectable read-only review of these receipts.

For each target, `*_observed` contains only preserved/recovered measurements;
`*_filled` adds accepted imputation or the labeled DFGA zero inference;
`*_estimate` is the model diagnostic, including rows not eligible for repair.
`*_is_measured` is false for estimates and inferred zeros. A downstream filter
must **not consume the filled column as independent measured evidence**.
Use observation-only filtering as the clean control; any uncertainty-weighted
proxy use needs a separate validation. This builder changes no v3 exports,
skill hyperparameters, public boards, or production solves.

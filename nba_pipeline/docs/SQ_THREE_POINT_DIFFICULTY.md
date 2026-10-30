# qSQ Three-Point Difficulty

## Purpose

The player-facing HTML table uses the SSI `3p_shot_sq.csv` export. It does not
use ShotQuality API `initial_shot_taking` or `initial_ev`; both API fields carry
shooter-talent information and are not pure shot-difficulty inputs.

The display identity is:

```text
raw 3P% + difficulty = effective 3P%
```

Positive difficulty means harder-than-average shots. Negative means easier.

## qSQ scale and calculation

The source contains only three-point attempts and reports `eFG%`, `qSQ`,
`qSI`, and `qSP` on an eFG-percentage scale. The source identity is:

```text
eFG% = qSQ + qSI
```

Because every attempt is a three, convert any eFG-style percentage to the 3P
scale by multiplying by `2/3`:

```text
raw 3P% = eFG% * 2/3
expected 3P% = qSQ * 2/3
shotmaking vs expected = qSI * 2/3
qSP on 3P scale = qSP * 2/3
```

For each season, calculate the attempt-weighted qSQ baseline:

```text
season qSQ = sum(player qSQ * player 3PA) / sum(player 3PA)
difficulty = (season qSQ - player qSQ) * 2/3
effective 3P% = raw 3P% + difficulty
```

The current 2025-26 source contains 474 player rows and 90,688 attempts. Its
attempt-weighted qSQ is `54.174%` eFG, or `36.116%` on the 3P scale.

## Russell v2 comparison

The frozen `RUSSELL_SHOTQUALITY v2` parquet stores total make probability and
includes both date-aware EPM shooter talent and expanding repo-owned shooter
talent. Never calculate `league average - p_v2` and label that difficulty;
doing so would incorrectly classify strong shooters as having easy attempts.

An earlier K=200 shooter-demeaning approximation was rejected. It gave Luka
Doncic `-0.9 pp` difficulty in 2025-26 because it subtracted his persistent
hard-shot context as if that context were weak shooter talent. The viewer no
longer uses that approximation.

The replacement is built by
`nba_pipeline/scripts/build_russell_v2_counterfactual.py`. It reconstructs the
documented v2 season-forward gradient-boosting architecture from the surviving
local PBP, transition, as-of EPM, and career-seed inputs. For each test-season
three, the same fitted model scores two feature rows:

```text
actual score  = actual as-of shooter features + actual shot context
neutral score = league-reference shooter features + identical shot context
```

The neutralized shooter fields are EPM projected 3P%, midrange%, FT%, and 3PA
rate plus the repo-owned expanding zone rate, zone sample size, and jumper
rate. Distance, period/clock, regular-season/playoff state, transition
probability, action/family, and outcome-invariant description features such as
pull-up and step-back remain unchanged. Continuous references are the
test-season shot-weighted means; expanding sample size uses the median.

For regular-season threes:

```text
v2 neutral baseline = mean neutral score over all season threes
v2 CF difficulty = v2 neutral baseline - player mean neutral score
v2 CF effective 3P% = raw 3P% + v2 CF difficulty
```

The identity is attempt-weight centered within each season. In 2025-26, Luka's
694 threes receive `+1.14 pp` v2 CF difficulty, compared with qSQ's `+5.30 pp`.
The sign now passes the hard-shot falsification, but the v2 rebuild remains
materially less sensitive to his shot burden. Among 307 players with at least
100 attempts, qSQ and v2 CF difficulty correlate `r = 0.59` in 2025-26.

This is a current-input architecture reconstruction, not an exact rescore from
the accepted frozen v2 model object, which was not preserved. On 96,813 matched
2025-26 threes, its actual-skill predictions correlate only `0.454` with the
frozen `p_v2` and have probability MAE `0.0513`. The largest known source
difference is the surviving transition backfill/current PBP surface versus the
original transition sidecar/source snapshot. Therefore use the result as a
v2-style counterfactual diagnostic in this viewer; do not overwrite or silently
replace the frozen VPM target.

## ShotQuality-rich counterfactual

`nba_pipeline/scripts/build_sq_rich_three_point_counterfactual.py` adds the
current ShotQuality store's recovered release-time geometry. It does not use
the stale `vpm/data/sq_shots_keyed.parquet` surface. The season-forward contract
is fixed:

- train on 92,320 player-attributed 2024-25 regular-season threes;
- score 90,973 player-attributed 2025-26 regular-season threes;
- calculate the shooter feature from whole-current-game-excluded history with
  200-attempt shrinkage;
- score each test shot with actual shooter history and with the shooter rate and
  exposure set to test-season league references;
- hold period/clock, ShotQuality distance, spacing, safe play descriptors,
  contest, closest/second-closest defender distances and angles, paint count,
  defender position flags, height, and obstruction geometry fixed;
- exclude `initial_ev`, Initial Shot Taking, outcomes as features, and the
  provider's collapsed `shot_x`/`shot_y` coordinates.

The full-season output is a hybrid. A rich model trained only on geometry-valid
2024-25 shots scores geometry-valid 2025-26 shots; a separately trained
context-only model scores the rest. This avoids asking a model trained under
90.9% geometry coverage to extrapolate a missing-tracking branch over nearly
half the next season.

The direct refreshed store contains 91,732 2025-26 regular-season threes and
49,733 with closest-defender geometry (`54.2%`). After excluding the 759 shots
without a player ID, model coverage is 49,733 of 90,973 (`54.7%`). Luka Doncic
has geometry on 407 of 694 threes (`58.6%`). His results are:

| Metric | Difficulty |
|---|---:|
| qSQ | `+5.30 pp` |
| Russell v2 CF rebuild | `+1.14 pp` |
| SQ-rich full-season hybrid | `+1.91 pp` |
| SQ-rich geometry-covered subset | `+2.47 pp` |

The geometry layer is real but does not close the full qSQ gap. On the 2025-26
geometry-covered sample, rich versus context-only points MSE improves by
`0.01098` with a game-clustered 95% interval `[-0.01452, -0.00751]`. Across the
full hybrid sample, Brier improves `0.230846 -> 0.230179` versus context-only,
but remains worse than provider `initial_ev` at `0.229378`. Therefore this is a
useful player-facing context challenger, not a superior total ShotQuality model
and not an admitted VPM target.

## Build and output

Run:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/build_russell_v2_counterfactual.py
OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/build_sq_rich_three_point_counterfactual.py
PYTHONPATH=. .venv/bin/python nba_pipeline/scripts/build_sq_three_point_difficulty_html.py
```

The generator defaults to `/Users/russ/Downloads/3p_shot_sq.csv` and writes a
self-contained page to:

```text
nba_pipeline/results/sq_three_point_difficulty/three_point_difficulty_table.html
```

The page embeds every season from 2013-14 through 2025-26 and provides season,
player-name, minimum-attempt, and sortable-column controls. The main table and
right-side player drawer show qSQ difficulty plus the Russell v2 CF
architecture-rebuild diagnostic for seasons 2020-21 through 2025-26. The
2025-26 rows also show SQ-rich CF difficulty, effective 3P%, and player geometry
coverage. The page can be opened directly without a local web server.

## Validation

- Require all source columns and unique `(SSI Player ID, Season)` rows.
- Require `eFG% = qSQ + qSI` within the source's two-decimal rounding error.
- Weight seasonal qSQ by `Shots Attempted`, never by player rows.
- Preserve the exact displayed `raw + difficulty = effective` identity before
  rounding.
- Do not relabel `initial_shot_taking` or `initial_ev` as shooter-neutral
  difficulty.
- Do not relabel raw talent-aware Russell v2 `p_v2` as difficulty.
- Require actual-versus-neutral scores from the same fitted model, with only
  the declared shooter-talent inputs changed.
- Require attempt-weighted seasonal v2 CF difficulty to equal zero and preserve
  `raw + difficulty = effective` before rounding.
- Keep the architecture rebuild visibly distinct from the accepted frozen v2
  target; its reproduction audit must travel with the output.
- Build SQ-rich results from the refreshed ShotQuality store, never the stale
  keyed parquet, and report geometry coverage beside every player result.
- Use the context-only fallback when closest-defender geometry is absent.
- Keep the rich geometry model descriptive until it beats relevant total-value
  baselines and clears a new frozen VPM target gate.

## Future-oTS usefulness audit

Treat the display's year-to-year stability as descriptive evidence only. The
admission test is whether feature-season qSQ difficulty improves a
chronological prediction of next-season offensive TS RAPM after the existing
box scoring inputs, and then after the existing playtype adjustment.

Translate the three-point percentage adjustment to scoring value as:

```text
qSQ 3P adjustment points/100 = 3PA/100 * difficulty_pp * 3 / 100
qSQ-adjusted TS added/100 = raw TS added/100 + qSQ 3P adjustment points/100
```

The August 2026 development audit trained on only earlier target seasons and
scored the 2024-25 and 2025-26 targets. On the predeclared 100-3PA feature
sample (526 held-out player-seasons), qSQ replacement changed weighted future
oTS RMSE from `1.5080` to `1.5068` (`-0.0012`; player-clustered 95% interval
`[-0.0137, +0.0109]`). After playtype adjustment, RMSE changed from `1.4944`
to `1.4938` (`-0.0006`; interval `[-0.0095, +0.0086]`).

The conclusion is unchanged without a 3PA cutoff (706 held-out
player-seasons): qSQ replacement changed box RMSE from `1.5033` to `1.5062`,
and adding it after playtype changed `1.4934` to `1.4935`. Attempt shrinkage
and minimum-3PA sensitivities changed the sign, and the two target seasons did
not agree. qSQ difficulty correlates only moderately with `playtype_diff`
(`r = 0.45` on the 100-3PA sample), but the non-overlapping information did
not deliver a robust future-oTS gain.

Decision: keep qSQ three-point difficulty as player-facing shooting context;
do not promote it into the oTS/VPM prior from these spent development folds.

# Historical Midrange / Three Shot-Quality Transfer

`scripts/build_historical_shot_quality_transfer.py` is the leakage-safe bridge
from the modern ShotQuality-labeled era to the 1997-present raw-PBP archive.
It is intentionally separate from the retrospective empirical-Bayes model in
`build_pbp_shot_quality.py`.

## Model Contract

The build has two layers, fit separately for `midrange` and `three` attempts:

1. **Modern/rich teacher:** uses `initial_ev`, normalized `sq_desc_*` pre-shot
   features, and historically available PBP fields. When the recovered
   2024-25 rich-feature OOF predictions are present, they are joined as
   leakage-safe soft labels. `dynamic_ev` is excluded.
2. **Historical student:** combines a direct make-probability model with a
   teacher-distilled model. Both use only fields available in historical raw
   PBP: parsed distance/zone/type/modifiers, pregame-only shooter priors,
   event action type, period/game time, and inferred shot-clock/reset context.
   The blend weight is selected on a later training block, never the final
   test games.

The student does not use `initial_ev`, `dynamic_ev`, `sq_desc_*`, shooter or
defender identity, the current made/missed result, or the made-only assist tag
at scoring time. Pregame shooter priors use earlier shots only. The target is
an expected make probability; exported EV is:

```text
historical_sq_ev = historical_sq_make_prob * shot_value
```

Midrange and three-point models have independent estimators and calibration.
They should not be pooled into one calibration curve because their base rates
and feature-response shapes differ.

## Validation Design

Games, not shots, are the split unit. The chronologically latest 20% of games
are held out for the final benchmark. An earlier game block inside the training
sample is used for teacher calibration. This prevents shots from the same game
from appearing on both sides of a split.

The strongest recovered run combines rich 2024-25 OOF teacher predictions
with 2025-26 SQ descriptor/EV rows. Its final test block is the latest 20% of
complete games across that two-season sample. Those games fall late in the
2025-26 ordering, but this is not a whole-season leave-one-season-out test:
earlier 2025-26 games remain in training.

The benchmark reports Brier score, log loss, calibration bias, and improvement
relative to `initial_ev` make probability. A negative improvement is retained
as a failed benchmark; the script does not relabel it as a win.

## Command

Train on every surviving enriched file:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/build_historical_shot_quality_transfer.py
```

By default the command also consumes the rich-feature OOF prediction files
under `results/sq_challenger/` when they exist. Use `--no-rich-oof` for the
descriptor-only audit.

Train on one representative file and score selected historical seasons:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/build_historical_shot_quality_transfer.py \
  --modern-files nba_pipeline/raw_data/NBA26.parquet \
  --score-years 1997-2000 2014 2026 \
  --output-dir nba_pipeline/results/historical_shot_quality \
  --output-name historical_sq_transfer
```

Generated model, benchmark, calibration, held-out prediction, manifest, and
scored-season files belong under `results/` or `validation/`; they are not
curated git inputs.

## Strongest Two-Season Chronological Benchmark

`rich_transfer_v3` combines `104,100` joined rich-feature OOF teacher rows from
2024-25 with the surviving 2025-26 SQ descriptor/EV sample. The final block
contains `483` complete games and is never used for fitting, blend selection,
or calibration.

| Bucket | Model | Test shots | Brier | Log loss | Brier gain vs `initial_ev` | 95% game-cluster CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Midrange | `initial_ev` | 23,546 | 0.249609 | 0.693038 | 0 | — |
| Midrange | SQ/rich teacher | 23,546 | 0.241555 | 0.675948 | +0.008054 | [+0.006560, +0.009508] |
| Midrange | Historical-PBP student | 23,546 | 0.241981 | 0.676893 | **+0.007628** | **[+0.006099, +0.009148]** |
| Three | `initial_ev` | 35,654 | 0.229908 | 0.652303 | 0 | — |
| Three | SQ/rich teacher | 35,654 | 0.229069 | 0.650360 | +0.000838 | [+0.000484, +0.001166] |
| Three | Historical-PBP student | 35,654 | 0.229288 | 0.651047 | **+0.000620** | **[+0.000194, +0.001007]** |

The student's paired log-loss gains are also positive by game: `+0.016145`
for midrange (95% CI `[+0.012849, +0.019404]`) and `+0.001256` for threes
(`[+0.000295, +0.002167]`). The three-point blend assigns 30% weight to the
direct historical outcome model and 70% to teacher distillation; the
midrange selector chooses pure distillation.

## Preserved Failed/Audit Candidates

The first descriptor-only student beat `initial_ev` on midrange but was worse
on threes (`-0.000517` Brier improvement). The v2 student removed that
regression, but its tiny three-point gain had a game-cluster interval spanning
zero. Both artifacts remain under `validation/historical_shot_quality/`.
They are retained because the successful v3 result should not erase failed
candidate history.

## Production v3 Backfill and Historical Competition Audit

`scripts/score_historical_shot_quality_transfer.py` applies the frozen
`rich_transfer_v3_models.joblib` package without retraining it. It streams
1997-2026 chronologically so each shooter prior uses only earlier games, writes
one atomic RS/PS parquet at a time, and resumes only when the model hash, source
fingerprint, schema version, row count, columns, and probability bounds match
the output sidecar.

The scorer normalizes both raw clock schemas before inference:

- Gabriel 1997-2013 stores total quarter seconds (`0..720`) in
  `seconds_remaining_quarter`;
- modern PBP stores a `0..59` seconds component alongside minutes.

Schema v2 fixed an earlier partial backfill that had added Gabriel total seconds
to the minute field a second time. The schema bump forced every stale v1 file
to be rewritten. The completed production run has:

- `60/60` RS/PS output files;
- `4,267,890` midrange/three attempts;
- zero failed files;
- feature missingness, prediction distribution, component-disagreement/support,
  and probability-bin audits;
- recorded-outcome Brier/log-loss fields only where make/miss labels exist.

The canonical output directory is:

```text
results/historical_shot_quality/rich_transfer_v3_scored/
```

Build or safely resume it with:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/score_historical_shot_quality_transfer.py \
  --start-year 1997 \
  --end-year 2026 \
  --season-types all
```

Then run the identical-row competition gate:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/audit_historical_shot_quality_competitors.py
```

The competition audit joins all `4,267,890` v3 rows to
`results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet` with `100%`
coverage for context EV, season/talent EV, lineup-defense EV, and recorded
make/miss outcomes in every season/type/bucket group. It also builds
leave-one-shot-out era/bucket and era/zone empirical references. All comparisons
use identical shots and paired game-cluster intervals.

The repo-owned context, season/talent, lineup-defense, and empirical references
are retrospective: they use other future/full-era outcomes. They are strong
historical research competitors, not leakage-safe deployment baselines.

Against the strongest lineup-defense-adjusted reference, positive values below
mean v3 has lower Brier loss:

| Era | Bucket | v3 Brier gain | 95% game-cluster CI | Result |
| --- | --- | ---: | ---: | --- |
| 1997-2003 | Midrange | -0.008376 | [-0.008665, -0.008101] | v3 loses |
| 1997-2003 | Three | +0.000485 | [+0.000321, +0.000649] | v3 wins |
| 2004-2013 | Midrange | -0.010040 | [-0.010294, -0.009794] | v3 loses |
| 2004-2013 | Three | -0.000861 | [-0.001063, -0.000664] | v3 loses |
| 2014-2023 | Midrange | -0.004923 | [-0.005171, -0.004698] | v3 loses |
| 2014-2023 | Three | +0.000249 | [+0.000164, +0.000336] | v3 wins |
| 2024-2026 | Midrange | +0.001223 | [+0.000953, +0.001493] | v3 wins |
| 2024-2026 | Three | +0.001262 | [+0.001117, +0.001404] | v3 wins |

Across context, season/talent, lineup-defense, and two empirical references, v3
loses `14/40` era/bucket comparisons on both Brier and log loss. The loss map is
not hidden:

- 1997-2003 midrange loses to all five references;
- 2004-2013 midrange loses to the three repo SQ references but beats both
  empirical references;
- 2004-2013 threes lose to the three repo SQ references but beat both empirical
  references;
- 2014-2023 midrange loses to the three repo SQ references but beats both
  empirical references;
- every 2024-2026 comparison is a clear v3 win.

`rich_transfer_v3` is therefore the portable, historically available-feature
model and a strong modern-era transfer. It is not the final retrospective
historical champion. Retrospective v4 and leakage-controlled outer-fold v4 work
are separate lanes; see
[Retrospective Historical ShotQuality v4](./HISTORICAL_SHOT_QUALITY_RETROSPECTIVE_V4.md)
for the former.

## Next Gates

- Recover an additional full rich-feature season and run a true frozen
  leave-one-season-out evaluation.
- Add a historical-only classifier for transferable SQ descriptors and compare
  direct distillation with explicit descriptor prediction.
- Use the v4 lanes, not v3 alone, for the retrospective historical champion
  gate. Keep v3 as the portable prospective surface.
- Do not call any model competition-leading until it wins against the strongest
  identical-row component with leakage controls appropriate to its claim.

# Retrospective Historical ShotQuality v4

`scripts/train_retrospective_historical_shot_quality_v4.py` builds a
retrospective midrange/three stack on top of the portable
`rich_transfer_v3` per-shot surface and the repo-owned full-history PBP
ShotQuality table.

This is a research/audit lane, not the portable deployment model. Preserve
v3 as the prospective transfer contract.

## Exact information contract

The stacker uses:

- v3 direct, distilled, and blended probabilities;
- v3 pregame shooter rate and attempt support;
- distance, action family/modifiers, period, estimated shot clock, and reset
  context;
- exactly one repo-owned retrospective outcome anchor:
  `shot_quality_with_defense_make_prob`.

The canonical context, pregame-prior, career, and shooter-season probabilities
are written and evaluated as competitors, but they are not simultaneous model
features.

That restriction is deliberate. In development, combining the nested
shot-level LOO fields let a nonlinear learner recover the current outcome from
small algebraic differences between the fields, producing impossible Brier
scores near zero. Each field can be a useful standalone retrospective
estimate; their joint differences are not a safe feature set.

The surviving defense anchor is itself retrospective. It excludes the current
shot at each local aggregate, but its nested context/shooter/defender
construction can use other outcomes from the same game and season. Therefore:

- game-group OOF means the **stacker fit** has no train/validation game overlap;
- temporal-forward means the **stacker fit** uses only earlier seasons;
- neither makes the anchor prospective or fully current-game-blind.

The separate safe-crossfit lane must be used for a leakage-controlled claim.

## Outputs

The default output folder is:

```text
results/historical_shot_quality/retrospective_v4/
```

It contains:

- `retrospective_v4_1997_2026.parquet`: joined per-shot probabilities;
- `retrospective_v4_NBA{YY}[_PS].parquet`: season/type slices;
- `retrospective_v4_metrics.csv`: per-era/bucket Brier, log loss, and bias;
- `retrospective_v4_comparisons.csv`: paired improvements and game-cluster
  intervals against every surviving component;
- `retrospective_v4_temporal_split_audit.json`: temporal train/validation
  ranges, row counts, and game-overlap checks;
- `retrospective_v4_coverage.json`: join, row, duplicate, and bounds audit;
- `retrospective_v4_models.joblib`: full-sample stackers for research use.

The full-coverage per-shot stacker surface is
`retrospective_v4_make_prob`, produced by game-group OOF.
`retrospective_v4_temporal_make_prob` is null for 1997 and is the
frozen-stacker comparison for 1998-2026. Both retain the retrospective-anchor
caveat above.

## Command

Run only after all 60 corrected v3 RS/PS files exist:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/train_retrospective_historical_shot_quality_v4.py
```

`--allow-partial` exists only for development. A normal run refuses fewer
than 60 source files.

## Validation requirements

Do not promote v4 based on pooled loss alone. Require:

1. 100% identity join coverage and zero duplicate shot keys;
2. probabilities bounded in `[0, 1]` with no OOF nulls;
3. per-era and per-bucket Brier/log-loss tables;
4. paired complete-game bootstrap intervals against every component;
5. zero stacker train/validation game overlap in temporal audits;
6. explicit “retrospective anchor” labeling everywhere it is displayed.

## Completed 1997-2026 build

The completed three-fold game-group OOF build joins all `4,267,890`
midrange/three attempts from all 60 RS/PS v3 files to the repo-owned
reference:

- midrange: `2,538,759`;
- threes: `1,729,131`;
- identity coverage: `100%`;
- duplicate keys: `0`;
- OOF null/out-of-bounds probabilities: `0` / `0`.

The table below compares v4 with the lowest-Brier surviving component in each
era/bucket. Positive gain means lower v4 Brier. Intervals resample complete
games.

| Era | Bucket | Strongest component | V4 Brier | Component Brier | Gain | 95% game CI |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1997-2003 | Midrange | repo defense LOO | 0.228100 | 0.229621 | +0.001521 | [+0.001411, +0.001624] |
| 1997-2003 | Three | v3 direct | 0.223387 | 0.224180 | +0.000793 | [+0.000695, +0.000883] |
| 2004-2013 | Midrange | repo defense LOO | 0.224938 | 0.228018 | +0.003081 | [+0.002970, +0.003203] |
| 2004-2013 | Three | repo career LOO | 0.223566 | 0.224973 | +0.001407 | [+0.001301, +0.001506] |
| 2014-2023 | Midrange | repo defense LOO | 0.231291 | 0.233678 | +0.002387 | [+0.002284, +0.002499] |
| 2014-2023 | Three | v3 portable | 0.226183 | 0.227627 | +0.001443 | [+0.001356, +0.001539] |
| 2024-2026 | Midrange | v3 direct | 0.239511 | 0.240206 | +0.000695 | [+0.000477, +0.000881] |
| 2024-2026 | Three | v3 direct | 0.228447 | 0.228455 | +0.000008 | [-0.000068, +0.000081] |

V4 has the best point Brier in all eight era/bucket cells. Seven of eight
paired intervals exclude zero; the 2024-2026 three-point result is a
statistical tie with v3 direct, not a proved win. Log loss follows the same
direction, including a negligible `+0.000019` point gain in that tied cell.

The frozen-stacker temporal audit is intentionally less flattering. It beats
the strongest component clearly only for 2004-2013 midrange, ties or narrowly
loses several other cells, and loses materially when the action-code/data
regime changes in 2014. That failure is retained. It shows that the powerful
same-era OOF stack should not be described as a stable prospective model.

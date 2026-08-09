# Historical ShotQuality Safe Crossfit

`scripts/train_historical_sq_safe_crossfit.py` is the leakage-controlled
retrospective research harness for historical midrange and three-point shot
quality. It is intentionally separate from the portable `rich_transfer_v3`
scorer and from the exploratory retrospective-v4 harness.

## Contract

The outer split is grouped by complete games. For every outer fold:

- season/bucket, shot-context, shooter-career, shooter-season, and
  defender-allowed maps are fitted from outer-training games;
- training encodings remove every shot from the row's entire game;
- validation encodings use the outer-training maps and never read validation
  outcomes;
- all encodings are direct empirical-Bayes make rates against fixed
  midrange/three priors, not nested residual encodings;
- a small histogram gradient-boosted stacker combines the frozen v3 score,
  raw PBP features, and the safe encodings; and
- a game-held-out selector inside the outer-training sample chooses a
  bucket/fold-specific blend between v3 and the stacker, without reading the
  outer-validation outcome.

The canonical repo-owned defense-adjusted probability is used only as a
retrospective comparison. It is not fed to the safe model.

## Prototype result

The 1997-98 four-file control contains 256,376 midrange/three shots. Three-fold
game-group crossfit improved the portable v3 model overall
(`0.234168` versus `0.235546` Brier) and on midrange
(`0.237017` versus `0.239726`). It did not beat the full-history canonical
defense baseline (`0.230422` overall), and it regressed three-point Brier
against v3 (`0.227251` versus `0.225392`). This short-window result is a
conservative control, not a competition win.

## Full-history result

The corrected all-history run is under
`results/historical_shot_quality/safe_crossfit_full/`. It contains
4,267,890 identically joined shots and 37,979 unique games. Every one of the
six outer bucket/fold audits reports zero game overlap. The inner
outer-training-only selector chose stack weights `0.75` for every midrange
fold and `0.375` to `0.50` for three-point folds.

The final safe crossfit improves portable v3 Brier by `0.002931` overall
(paired game-cluster 95% interval `[0.002838, 0.003018]`), `0.004817` for
midrange, and `0.000163` for threes. It also beats the strong canonical
defense-adjusted baseline on threes by `0.000325`
(`[0.000279, 0.000384]`) and improves three-point log loss by `0.000937`.

It does **not** beat that canonical baseline on midrange: the Brier deficit is
`0.002417` (`[-0.002493, -0.002320]`). Overall it trails canonical defense by
`0.001306`. This is the remaining modeling gate; the harness should not be
described as a universal historical winner.

Unit tests include:

- whole-game label-flip invariance for training encodings;
- validation-label invariance for every target encoding;
- zero train/validation game overlap;
- learnable-signal recovery; and
- a randomized-label guard that prevents algebraic own-shot reconstruction
  from masquerading as model skill.

## Commands

Prototype:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/train_historical_sq_safe_crossfit.py \
  --years 1997 1998
```

All available v3 files:

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/train_historical_sq_safe_crossfit.py \
  --all-years \
  --output-dir nba_pipeline/results/historical_shot_quality/safe_crossfit_full
```

The output directory contains shot-level OOF predictions, metrics, paired
game-cluster comparisons, a fold-overlap audit, and a manifest spelling out
the target-encoding contract.

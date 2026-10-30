# Mechanism RAPM Future-Alpha Validation

## Decision

Same-window grouped-game CV is not sufficient to publish the current mechanism
penalties. The forward development screen instead fits a trailing five-season
TD550 model and predicts the next non-overlapping two seasons. On the disputed
frequency and rebound targets, this moves the preferred penalty materially up:
the initial future screen chooses `1000/1000`, not the `100`-to-`500`-level
pairs selected by same-window exact-minimum CV.

These results are development evidence, not a silent production promotion.
The candidate set is deliberately compact, 2015-2026 is spent, and 2026-27
remains the untouched confirmation period.

## Design

- Model: zero-prior TD550, RS+PS, smooth rubberband (`alpha_rb=100`), and
  season/phase nuisance effects.
- Training: trailing five complete seasons through cutoff `t`.
- Target: actual possession outcomes in `t+1` and `t+2` for the same mechanism.
- Blocks: non-overlapping future pairs so the same target possession is never
  counted twice. The current primary run uses six blocks from 2015-16 through
  2025-26 where history permits; dunk metrics have four blocks from 2019-20.
- Score: future possession MSE, averaged equally across the two target seasons.
  Future season/phase intercepts are evaluation-only nuisance terms. No future
  outcome enters the player fit.
- Selection: the most-shrunk tested O/D pair within one paired standard error
  of the exact minimum across future blocks.
- Expanded-grid screen: STOP and the new compatible historical transition and
  playtype targets test the full Cartesian grid
  `{100,300,1000,3000,10000}` for offense and defense separately.
- Secondary diagnostic: agreement with a separately fit future two-season RAPM
  at fixed `3000/3000`. This is secondary because its scale depends on the
  target RAPM's own regularization.

The last complete two-year target is 2025-26, so its training cutoff is 2024.
A through-2025 fit cannot honestly be scored on two later complete seasons
until 2027 is available.

## Initial results

| Mechanism | Same-window CV O/D | Future one-SE O/D | Exact future minimum | Blocks |
| --- | ---: | ---: | ---: | ---: |
| Rim frequency | 300 / 100 | 1000 / 1000 | 1000 / 1000 | 6 |
| Midrange frequency | 100 / 1000 | 1000 / 1000 | 1000 / 1000 | 6 |
| Three frequency | 100 / 300 | 1000 / 1000 | 1000 / 1000 | 6 |
| Rim FG% | 500 / 500 | 1000 / 1000 | 1000 / 1000 | 6 |
| Midrange FG% | 1000 / 5000 | 3000 / 5000 | 1000 / 5000 | 6 |
| Three FG% | 2000 / 15000 | 3000 / 15000 | 3000 / 15000 | 6 |
| Rebound rate | 100 / 500 | 1000 / 1000 | 1000 / 1000 | 6 |
| Assisted points | 1000 / 2000 | 3000 / 3000 | 3000 / 3000 | 6 |
| Rim assists | 1000 / 1000 | 1000 / 3000 | 1000 / 3000 | 6 |
| Dunk frequency | 300 / 2000 | 1000 / 1000 | 1000 / 1000 | 4 |
| Dunk assists | 500 / 2000 | 3000 / 3000 | 500 / 3000 | 4 |

The largest clear corrections are the three frequency tables and rebound
rate. Their future-selected models reduce MSE by roughly `0.074%` to `0.105%`
versus the same-window choices, and the low pairs fall outside paired one-SE.
The heavier-shrinkage changes for midrange FG%, three FG%, and dunk assists are
much smaller; treat those as regularization safeguards rather than large
predictive wins.

For the three disputed frequency tables, the literal future-RAPM diagnostic
confirms why it should remain secondary. Correlation with the fixed-`3000/3000`
future two-year RAPM is maximized by `1000/1000` for midrange and three
frequency, while rim frequency's correlation peaks at `300/300`. Coefficient
RMSE mechanically favors `3000/3000` for rim and `10000/10000` for midrange and
three because the target RAPM is itself shrunk at `3000/3000`. The raw future
possession score avoids that circular target-scale preference and selects
`1000/1000` for all three.

The original screen stopped here because STOP then had only 2021-2026
processed inputs, while provider-only playtype proxy points, transition
frequency, and transition-rim frequency had only a 2026 board.

That original limitation is now superseded. Full-definition STOP was rebuilt
from 2014 onward, and separately labeled compatible historical transition and
playtype backcasts were materialized through 1997. The four-cut modern screen
uses 2014-2018 -> 2019-2020 through 2020-2024 -> 2025-2026 and produces:

| Historical target | Current/default O/D | Future one-SE O/D | Exact future minimum | Block winners |
| --- | ---: | ---: | ---: | --- |
| STOP | 1000 / 1000 | 3000 / 3000 | 3000 / 3000 | 3000/3000, 3000/3000, 1000/3000, 1000/3000 |
| Transition frequency backcast | 3000 / 3000 | 3000 / 3000 | 3000 / 3000 | 1000/3000, 3000/3000, 3000/3000, 1000/3000 |
| Transition-rim backcast | 3000 / 3000 | 3000 / 10000 | 1000 / 3000 | 1000/3000, 3000/10000, 1000/3000, 1000/3000 |
| Playtype proxy points backcast | 3000 / 3000 | 3000 / 10000 | 3000 / 10000 | 3000/10000, 3000/10000, 3000/10000, 1000/3000 |

STOP and transition frequency give stable `3000/3000` recommendations. The
transition-rim loss surface is very flat: its conservative one-SE rule chooses
`3000/10000`, while `1000/3000` is the exact minimum in three of four blocks.
Playtype proxy points is the clearest side-specific result: `3000/10000` is
both the exact and one-SE choice and wins three of four blocks. The much lower
first diagnostic result was invalidated after an inherited rim-attempt helper
column exposed a target-dispatch ambiguity; explicit-prefix dispatch, a schema
cleanup, and a regression test now prevent that metric substitution.

The STOP recommendation applies to the exact 2014-2026 definition. The three
`_BACKCAST` recommendations apply to their explicitly named compatible
historical series. They are evidence about reasonable regularization for the
2026 provider boards, not an automatic transfer or relabeling of those exact
provider targets. See
[Historical Mechanism Backfills](./HISTORICAL_MECHANISM_BACKFILLS.md).

## Reproduction

Run a selected mechanism:

```bash
.venv/bin/python nba_pipeline/scripts/validate_mechanism_future_alphas.py \
  --mechanism rim_frequency --max-blocks 6

.venv/bin/python nba_pipeline/scripts/validate_mechanism_future_alphas.py \
  --mechanism stop --max-blocks 4 --full-alpha-grid \
  --output-dir nba_pipeline/results/mechanism_future_alpha_validation_full_grid
```

Rebuild the consolidated selection table from completed receipts:

```bash
.venv/bin/python nba_pipeline/scripts/validate_mechanism_future_alphas.py \
  --summarize-existing
```

Generated receipts live under
`nba_pipeline/results/mechanism_future_alpha_validation/` and
`nba_pipeline/results/mechanism_future_alpha_validation_full_grid/`; both
remain untracked.

## Next gate

Before replacing the current reader alphas:

1. finish the direct future-two-year RAPM agreement readout beyond the three
   disputed frequency mechanisms;
2. refine the grid around each selected family rather than calling this compact
   candidate set globally optimal;
3. inspect coefficient scale and year-block stability;
4. freeze the method and evaluate once on the untouched 2026-27 target.

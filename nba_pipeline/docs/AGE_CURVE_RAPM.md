# Age-Curve RAPM Mode

## Summary

`rapm.py` now supports an optional `--age-curve` mode for `RAPM` only.

This mode does **not** add age as a regression feature. Instead, it uses the existing prior-centered solve and shrinks players toward a learned **age-mean prior** built from `player_stats_with_metrics`.

Scope in v1:
- applies only to the overall `RAPM` solve
- does not change `TS`, `TOV`, `REB`, `BADPASS_TOV`, `SCORING_TOV`, `SECOND_CHANCE`, or `FT_PREMIUM`
- changes `weighted_factors` indirectly because the RAPM regression target changes

## Data Source

The age curve is learned from `player_stats_with_metrics` using:
- regular season rows only: `playoffs = 0`
- rows with `year <= run_end_year`
- `Age`
- `OffPoss`, `DefPoss`
- rolling RAPM fallback columns

Target fallback hierarchy:
- offense: `five_year_orapm -> four_year_orapm -> three_year_orapm -> two_year_orapm -> off_rapm`
- defense: `five_year_drapm -> four_year_drapm -> three_year_drapm -> two_year_drapm -> def_rapm`

The age prior is an **age mean**, not a player-history anchor.

## Curve Construction

For offense and defense separately:
- bucket rows by `floor(Age)`
- compute weighted age means
- weights are `sqrt(OffPoss)` or `sqrt(DefPoss)` times a horizon multiplier
- horizon multipliers: `5Y=5`, `4Y=4`, `3Y=3`, `2Y=2`, `1Y/current=1`
- shrink each bucket toward the global weighted mean with a fixed empirical-Bayes constant
- smooth the shrunk curve locally across adjacent ages
- clip out-of-range ages to the nearest observed endpoint instead of extrapolating

## Player Priors In A Run

For a specific RAPM window:
- compute each player's effective age inside the run window using regular-season rows only
- offense effective age is weighted by `OffPoss`
- defense effective age is weighted by `DefPoss`
- if no in-window age exists, fall back to the latest regular-season age at or before `run_end_year`
- if no age row exists at all, the player gets no age prior and falls back to the normal zero-centered ridge behavior

The evaluated age curve is then passed into the existing offense/defense prior machinery.

This mode remains RAPM-history based. It does not use DARKO `o_dpm` or `d_dpm`
as the age-curve target. DARKO ages are currently used by the separate
`--age-dummies` fixed-effect mode.

## Commands

Single RAPM solve:
```bash
cd nba_pipeline/scripts
python rapm.py RAPM 14 26 ALL --rubberband --age-curve --pure
```

Standard weighted-factors run:
```bash
python 03_run_rapm_analysis.py 14 26 ALL --rubberband --age-curve
```

## Output Naming

When `--age-curve` is enabled:
- RAPM result files include `_age`
- organized result folders include `_age`
- downstream `weighted_factors`, regression outputs, and prediction files from `03_run_rapm_analysis.py` include `_age`

Examples:
- `rapm_14_26_all_pure_rb_age_results.csv`
- `results/rapm_14_26_all_rb_age/`
- `weighted_factors_14_26_all_rb_age.csv`

Non-RAPM metric files keep their standard names inside the `_age` run folder.

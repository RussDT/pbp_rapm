# Randle Strong/Weak RAPM

Standalone research harness for opponent-strength RAPM splits.

This folder intentionally does not depend on the production `rapm.py` strength-split implementation. It tests cleaner parameterizations for the question:

> Which players keep or change impact against stronger opposing lineups?

## Models

`run_randle_strong_weak.py` fits two models on processed `RAPM` parquet rows:

1. `binary_reference`
   - Base offense/defense player effects are the `not_strong` environment.
   - Strong environments add one player-side delta.
   - `off_vs_weak = off_base`
   - `off_vs_strong = off_base + off_strong_delta`
   - `def_vs_weak = def_base`
   - `def_vs_strong = def_base + def_strong_delta`

2. `continuous_slope`
   - Base offense/defense player effects are average opponent strength.
   - Stronger/weaker environment is a same-season z-score of opponent DARKO lineup strength.
   - Exports `*_vs_weak_1sd`, `*_vs_avg`, and `*_vs_strong_1sd`.

Both models use heavier shrinkage on interaction terms through column scaling. This keeps base RAPM and opponent-strength response from being treated as equally stable signals.

## Default Command

```bash
python randle_strong_weak/run_randle_strong_weak.py --start-year 21 --end-year 26 --season-type ALL --base-alpha 1000 --interaction-alpha-mult 4
```

Outputs are written to `randle_strong_weak/outputs/`.

## Two-Stage Residual Harness

`run_two_stage_residual.py` tests the follow-up construction:

1. Fit ordinary six-year RAPM once.
2. Freeze those player values.
3. Subtract the fixed player prediction from each possession.
4. Fit only opponent-strength residual variables.

The current default base file is:

```text
nba_pipeline/results/rapm_21_26_all_pure_a2000_4000_results.csv
```

That stage-one file uses offense alpha `2000` and defense alpha `4000`.

Default command:

```bash
python randle_strong_weak/run_two_stage_residual.py --start-year 21 --end-year 26 --season-type ALL --base-results nba_pipeline/results/rapm_21_26_all_pure_a2000_4000_results.csv
```

Key outputs:

- `randle_strong_weak/outputs_two_stage/validation_summary_21_26_all.csv`
- `randle_strong_weak/outputs_two_stage/diagnostics_21_26_all.json`
- `randle_strong_weak/TWO_STAGE_NOTES.md`
- `randle_strong_weak/two_stage_results_browser.html`

To use frozen RAPM itself as the opponent-strength source:

```bash
python randle_strong_weak/run_two_stage_residual.py --start-year 21 --end-year 26 --season-type ALL --base-results nba_pipeline/results/rapm_21_26_all_pure_a2000_4000_results.csv --strength-source base_rapm --output-dir randle_strong_weak/outputs_two_stage_rapm_strength
python randle_strong_weak/build_two_stage_browser.py --outputs-dir randle_strong_weak/outputs_two_stage_rapm_strength --html-path randle_strong_weak/two_stage_rapm_strength_results_browser.html --title "Two-Stage RAPM-Strength Residuals" --strength-label "Frozen RAPM lineup strength"
```

See `randle_strong_weak/RAPM_STRENGTH_NOTES.md` for the DARKO-vs-RAPM strength comparison.

# Season Intercepts

This harness computes season-level averages from the processed parquet surface so they can be reused as hardcoded seasonal intercept inputs.

## What It Produces

Run:

```bash
cd nba_pipeline/scripts
python build_season_intercepts.py
```

Default output directory:

- `nba_pipeline/results/season_intercepts/raw_season_column_means.csv`
- `nba_pipeline/results/season_intercepts/solver_season_intercepts.csv`
- `nba_pipeline/results/season_intercepts/processed_file_audit.csv`
- `nba_pipeline/results/season_intercepts/season_intercepts.json`

## Artifact Meanings

- `raw_season_column_means.csv`: mean of every numeric non-control parquet column, keyed by source file, season, and season type.
- `solver_season_intercepts.csv`: means expressed in solver space, including signed `Off_Diff` / `Def_Diff` intercepts and alias-derived rows such as `BADPASS_TOV`, `SCORING_TOV`, `ALT_TS`, and `ALT_TOV`.
- `processed_file_audit.csv`: one row per processed parquet showing which season key was used and whether the embedded parquet `Season` value matched the filename.
- `season_intercepts.json`: nested lookup keyed by `season_type -> season_end_year -> metric_key`.

## Season Key Rule

The harness treats the processed filename as the canonical season key.

Example:

- `RAPM26.parquet` resolves to season end year `2026`
- `RAPM26_PS.parquet` resolves to season end year `2026`, season type `PS`

This is deliberate because the embedded parquet `Season` column is not historically consistent across generations of processed files. Some legacy builders wrote `year + 1`, while newer files may already carry a 4-digit end year. The audit file exposes those mismatches instead of silently trusting the parquet value.

## Relationship To `--season-effects`

These artifacts are not the same as `rapm.py --season-effects`.

- `--season-effects` learns nuisance season coefficients inside a specific multi-season regression run.
- `build_season_intercepts.py` precomputes season-level averages directly from each processed parquet.

Use the precomputed harness when you want stable, hardcoded season baselines outside the live solve.

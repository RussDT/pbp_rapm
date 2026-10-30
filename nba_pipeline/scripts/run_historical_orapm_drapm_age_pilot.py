#!/usr/bin/env python3
"""Build the annual O/D panel and longitudinal age-curve pilot package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from nba_pipeline.scripts.build_longitudinal_rapm_age_panel import assemble_age_panel
from nba_pipeline.scripts.fit_longitudinal_rapm_age_curve import (
    fit_longitudinal_curves,
    write_fit_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "nba_pipeline" / "results" / "longitudinal_orapm_drapm_age"
DEFAULT_AGE_SOURCE = REPO_ROOT / "nba_pipeline" / ".cache" / "teammate_help" / "dpm_history.csv"


def annual_result_path(directory: Path, season: int) -> Path:
    return directory / f"rapm_{int(season)}_rs_results.csv"


def solve_annual_rapm(season: int, *, off_alpha: float, def_alpha: float, cores: int) -> pd.DataFrame:
    from nba_pipeline.scripts import rapm as rapm_solver

    rapm_solver.N_CORES = int(cores)
    filename = f"RAPM{str(int(season))[-2:]}.parquet"
    source = rapm_solver.PROCESSED_DIR / filename
    if not source.exists():
        raise FileNotFoundError(f"Missing processed annual RAPM parquet: {source}")
    result = rapm_solver.run_simplified_rapm(
        [filename],
        pure=True,
        off_alpha=float(off_alpha),
        def_alpha=float(def_alpha),
        rubberband=True,
        season_effects=False,
        start_year=int(season),
        end_year=int(season),
        prefix="RAPM",
    )
    if result is None or result.empty:
        raise RuntimeError(f"Annual RAPM solve returned no rows for {season}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("--annual-results-dir", type=Path, default=None)
    parser.add_argument("--age-source", type=Path, default=DEFAULT_AGE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-solves", action="store_true")
    parser.add_argument("--off-alpha", type=float, default=2000.0)
    parser.add_argument("--def-alpha", type=float, default=3000.0)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--age-min", type=int, default=18)
    parser.add_argument("--age-max", type=int, default=40)
    parser.add_argument("--reference-age", type=int, default=27)
    parser.add_argument("--smoothing-alpha", type=float, default=None)
    parser.add_argument("--smoothing-grid", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=20260826)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.end_year < args.start_year:
        raise ValueError("end_year must be at least start_year")
    output_dir = args.output_dir or DEFAULT_RESULTS_DIR / f"{args.start_year}_{args.end_year}"
    annual_dir = args.annual_results_dir or output_dir / "annual_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    annual_dir.mkdir(parents=True, exist_ok=True)

    annual_results = {}
    for season in range(args.start_year, args.end_year + 1):
        path = annual_result_path(annual_dir, season)
        if path.exists():
            annual_results[season] = pd.read_csv(path)
            continue
        if args.skip_solves:
            raise FileNotFoundError(f"Missing reusable annual result: {path}")
        result = solve_annual_rapm(
            season,
            off_alpha=args.off_alpha,
            def_alpha=args.def_alpha,
            cores=args.cores,
        )
        result.to_csv(path, index=False)
        annual_results[season] = result

    age_history = pd.read_csv(args.age_source)
    panel, panel_receipt = assemble_age_panel(annual_results, age_history)
    panel.to_csv(output_dir / "annual_orapm_drapm_panel.csv", index=False)
    panel_receipt.update(
        {
            "start_year": int(args.start_year),
            "end_year": int(args.end_year),
            "off_alpha": float(args.off_alpha),
            "def_alpha": float(args.def_alpha),
            "annual_sample": "regular_season",
            "rubberband": "smooth_local_single_season",
            "age_source": str(args.age_source),
        }
    )
    with (output_dir / "panel_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(panel_receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fit_panel = panel.dropna(subset=["age"]).copy()
    fitted = fit_longitudinal_curves(
        fit_panel,
        age_min=args.age_min,
        age_max=args.age_max,
        reference_age=args.reference_age,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_grid=[float(value) for value in args.smoothing_grid.split(",") if value.strip()],
        cv_folds=args.cv_folds,
        bootstrap_reps=args.bootstrap_reps,
        random_seed=args.random_seed,
    )
    fitted.receipt.update(
        {
            "panel_path": str(output_dir / "annual_orapm_drapm_panel.csv"),
            "annual_results_dir": str(annual_dir),
            "off_alpha": float(args.off_alpha),
            "def_alpha": float(args.def_alpha),
        }
    )
    write_fit_artifacts(fitted, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

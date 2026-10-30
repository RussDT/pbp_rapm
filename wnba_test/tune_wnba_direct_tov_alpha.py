#!/usr/bin/env python3
"""Select the WNBA direct-turnover defensive alpha on season-forward folds.

This tunes only the explanatory ``ALT_TOV_VALUE`` family. Headline VPM keeps
the frozen total/component alphas; the publish bridge reallocates the legacy
completion baseline into shot value and uses this direct solve for TOV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import load_box_features
from wnba_test.clean_decomp_wnba import _load_name_map, _solve_direct_ridge_problem
from wnba_test.tune_wnba_component_alphas import (
    _component_problem,
    _prior_frame,
    _result_vector,
)


DEFAULT_BOX_DIR = ROOT / "wnba_test" / "Results" / "box_prior_true_prior"
DEFAULT_OUTPUT = DEFAULT_BOX_DIR / "direct_tov_alpha"
DEFAULT_GRID = [500, 1000, 2000, 5000, 10000, 20000, 40000, 80000, 160000]
DEVELOPMENT_PAIRS = [(2023, 2024), (2024, 2025)]
REPORT_ONLY_PAIR = (2025, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-dir", type=Path, default=DEFAULT_BOX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--def-alphas",
        default=",".join(str(value) for value in DEFAULT_GRID),
    )
    parser.add_argument("--off-alpha", type=float, default=5000.0)
    parser.add_argument("--prior-scale", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alphas = [float(value) for value in args.def_alphas.split(",") if value]
    name_map = _load_name_map()
    contexts = []
    for rating_year, validation_year in [*DEVELOPMENT_PAIRS, REPORT_ONLY_PAIR]:
        print(f"Preparing {rating_year}->{validation_year}...", flush=True)
        train = _component_problem("ALT_TOV_VALUE", rating_year)
        validation = _component_problem("ALT_TOV_VALUE", validation_year)
        contexts.append(
            {
                "rating_year": rating_year,
                "validation_year": validation_year,
                "train": train,
                "validation": validation,
                "prior": _prior_frame(
                    args.box_dir, rating_year, "ALT_TOV_VALUE"
                ),
                "intercept": float(np.mean(train["y"])),
            }
        )

    rows = []
    for alpha in alphas:
        fold_scores = {}
        for context in contexts:
            result = _solve_direct_ridge_problem(
                context["train"],
                name_map,
                off_alpha=args.off_alpha,
                def_alpha=alpha,
                prior_results=context["prior"],
                prior_scale=args.prior_scale,
            )
            vector = _result_vector(result, context["validation"])
            prediction = (
                np.asarray(context["validation"]["X"] @ vector).reshape(-1)
                + context["intercept"]
            )
            actual = np.asarray(context["validation"]["y"], dtype=float)
            year = context["validation_year"]
            fold_scores[year] = {
                "mse": float(np.mean(np.square(actual - prediction))),
                "correlation": float(np.corrcoef(actual, prediction)[0, 1]),
            }
        rows.append(
            {
                "def_alpha": alpha,
                "development_mse": float(
                    np.mean([fold_scores[year]["mse"] for year in (2024, 2025)])
                ),
                "mse_2024": fold_scores[2024]["mse"],
                "correlation_2024": fold_scores[2024]["correlation"],
                "mse_2025": fold_scores[2025]["mse"],
                "correlation_2025": fold_scores[2025]["correlation"],
                "report_only_mse_2026": fold_scores[2026]["mse"],
                "report_only_correlation_2026": fold_scores[2026]["correlation"],
            }
        )
        print(f"Scored defense alpha {alpha:g}", flush=True)

    grid = pd.DataFrame(rows).sort_values("development_mse").reset_index(drop=True)
    selected = grid.iloc[0]

    current_problem = _component_problem("ALT_TOV_VALUE", 2026)
    current_prior = _prior_frame(args.box_dir, 2026, "ALT_TOV_VALUE")
    current = _solve_direct_ridge_problem(
        current_problem,
        name_map,
        off_alpha=args.off_alpha,
        def_alpha=float(selected["def_alpha"]),
        prior_results=current_prior,
        prior_scale=args.prior_scale,
    )
    current["dTOV"] = -pd.to_numeric(current["def"], errors="coerce")
    box = load_box_features(2026)[["player_id", "steals100"]]
    current = current.merge(box, on="player_id", how="left")
    aja = current.loc[current["player_id"] == "1628932", "dTOV"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "alpha_grid.csv", index=False)
    report = {
        "status": "selected on pre-2026 development; 2026 is report-only",
        "target": "direct ALT_TOV_VALUE possession points",
        "development_pairs": DEVELOPMENT_PAIRS,
        "report_only_pair": REPORT_ONLY_PAIR,
        "off_alpha": args.off_alpha,
        "prior_scale": args.prior_scale,
        "selected_def_alpha": float(selected["def_alpha"]),
        "selected_row": selected.to_dict(),
        "current_2026": {
            "aja_wilson_dtov": float(aja.iloc[0]) if not aja.empty else None,
            "correlation_dtov_with_stabilized_steals": float(
                current["dTOV"].corr(current["steals100"])
            ),
        },
        "interpretation": (
            "This alpha changes only the direct turnover explanation. The "
            "headline VPM solve and additive total remain frozen."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

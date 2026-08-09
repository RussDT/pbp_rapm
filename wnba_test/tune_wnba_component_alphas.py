#!/usr/bin/env python3
"""Joint factor-specific WNBA alpha tuning by coordinate descent.

Each candidate is scored on the sum of all prior-centered component solves
against future actual possession points. Development folds are 2023->2024 and
2024->2025. The frozen 2025 rating is then evaluated on 2026 separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import ADDITIVE_PREFIXES
from wnba_test.clean_decomp_wnba import (
    RESULTS_DIR,
    _attach_date_proxy,
    _files_for_metric,
    _infer_year_from_path,
    _load_clean_total_df,
    _load_name_map,
    _prepare_direct_ridge_problem,
    _prepare_metric_df,
    _solve_direct_ridge_problem,
)


DEFAULT_BOX_DIR = RESULTS_DIR / "box_prior_true_prior"
DEFAULT_OUTPUT = DEFAULT_BOX_DIR / "component_alpha_tuning"
DEFAULT_GRID = [
    1000,
    2000,
    5000,
    10000,
    20000,
    40000,
    80000,
    160000,
    320000,
]
UNIFORM_SEED = {prefix: (5000, 40000) for prefix in ADDITIVE_PREFIXES}
SCREEN_SEED = {
    "ALT_EFG_RIM_FREQ": (20000, 5000),
    "ALT_EFG_RIM_FG": (20000, 40000),
    "ALT_EFG_MID_FREQ": (20000, 10000),
    "ALT_EFG_MID_FG": (20000, 20000),
    "ALT_EFG_THREE_FREQ": (320000, 320000),
    "ALT_EFG_THREE_FG": (320000, 80000),
    "ALT_FT": (320000, 40000),
    "ALT_FC_COMPLETION": (20000, 80000),
    "SECOND_CHANCE_CLEAN": (20000, 20000),
}


def coordinate_descent(
    seed: dict[str, tuple[float, float]],
    components: list[str],
    grid: Iterable[float],
    score: Callable[[dict[str, tuple[float, float]]], float],
    max_passes: int,
) -> tuple[dict[str, tuple[float, float]], float, list[dict[str, object]]]:
    """Discrete joint coordinate descent over component and side alphas."""
    alphas = dict(seed)
    best = float(score(alphas))
    log = [
        {
            "pass": 0,
            "component": "seed",
            "side": "both",
            "alpha": None,
            "score": best,
        }
    ]
    candidates = [float(value) for value in grid]
    for pass_index in range(1, int(max_passes) + 1):
        changed = False
        for component in components:
            for side_index, side_name in ((0, "off"), (1, "def")):
                current = alphas[component]
                best_alpha = current[side_index]
                best_value = best
                for alpha in candidates:
                    if alpha == current[side_index]:
                        continue
                    trial = dict(alphas)
                    trial[component] = (
                        (alpha, current[1])
                        if side_index == 0
                        else (current[0], alpha)
                    )
                    value = float(score(trial))
                    if value < best_value:
                        best_value = value
                        best_alpha = alpha
                if best_alpha != current[side_index]:
                    alphas[component] = (
                        (best_alpha, current[1])
                        if side_index == 0
                        else (current[0], best_alpha)
                    )
                    best = best_value
                    changed = True
                    log.append(
                        {
                            "pass": pass_index,
                            "component": component,
                            "side": side_name,
                            "alpha": best_alpha,
                            "score": best,
                        }
                    )
        if not changed:
            break
    return alphas, best, log


def classify_trust(
    selected_alpha: float,
    max_alpha: float,
    development_gain: float,
    confirmation_gain: float,
    minimum_gain_x1e6: float = 1.0,
) -> str:
    if (
        selected_alpha >= max_alpha
        or development_gain <= float(minimum_gain_x1e6)
    ):
        return "prior_only"
    if confirmation_gain > float(minimum_gain_x1e6):
        return "supported"
    return "mixed"


def _component_problem(prefix: str, year: int) -> dict[str, object]:
    files = [
        path
        for path in _files_for_metric(prefix, [year % 100], "ALL")
        if path.exists()
    ]
    if not files:
        raise FileNotFoundError(f"No {prefix} files for {year}")
    frames = [
        _attach_date_proxy(pd.read_csv(path), _infer_year_from_path(path))
        for path in files
    ]
    prepared = _prepare_metric_df(pd.concat(frames, ignore_index=True), prefix)
    return _prepare_direct_ridge_problem(prepared)


def _prior_frame(box_dir: Path, year: int, prefix: str) -> pd.DataFrame:
    readable = pd.read_csv(
        box_dir / f"box_priors_{year}.csv",
        dtype={"player_id": str},
    )
    readable["player_id"] = (
        pd.to_numeric(readable["player_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    return pd.DataFrame(
        {
            "player_id": readable["player_id"],
            "off": pd.to_numeric(
                readable[f"box_o{prefix}"], errors="coerce"
            ).fillna(0.0),
            "def": -pd.to_numeric(
                readable[f"box_d{prefix}"], errors="coerce"
            ).fillna(0.0),
        }
    )


def _result_vector(
    result: pd.DataFrame,
    validation_problem: dict[str, object],
) -> np.ndarray:
    player_to_col = dict(validation_problem["player_to_col"])
    vector = np.zeros(len(player_to_col), dtype=float)
    for player, offense, defense in result[
        ["player_id", "off", "def"]
    ].itertuples(index=False, name=None):
        player_id = str(player)
        off_key = f"{player_id}_off"
        def_key = f"{player_id}_def"
        if off_key in player_to_col:
            vector[player_to_col[off_key]] = float(offense) / 100.0
        if def_key in player_to_col:
            vector[player_to_col[def_key]] = float(defense) / 100.0
    return vector


class JointAlphaEvaluator:
    def __init__(
        self,
        box_dir: Path,
        fold_pairs: list[tuple[int, int]],
        prior_scale: float,
    ) -> None:
        self.box_dir = box_dir
        self.fold_pairs = fold_pairs
        self.prior_scale = float(prior_scale)
        self.name_map = _load_name_map()
        self.contexts = []
        for rating_year, validation_year in fold_pairs:
            print(
                f"Preparing joint-alpha fold {rating_year}->{validation_year}...",
                flush=True,
            )
            total_train = _prepare_direct_ridge_problem(
                _load_clean_total_df([rating_year % 100], "ALL")
            )
            validation_type = "RS" if validation_year == 2026 else "ALL"
            total_validation = _prepare_direct_ridge_problem(
                _load_clean_total_df(
                    [validation_year % 100],
                    validation_type,
                )
            )
            component_problems = {
                prefix: _component_problem(prefix, rating_year)
                for prefix in ADDITIVE_PREFIXES
            }
            priors = {
                prefix: _prior_frame(box_dir, rating_year, prefix)
                for prefix in ADDITIVE_PREFIXES
            }
            self.contexts.append(
                {
                    "rating_year": rating_year,
                    "validation_year": validation_year,
                    "component_problems": component_problems,
                    "priors": priors,
                    "validation_problem": total_validation,
                    "intercept": float(np.mean(total_train["y"])),
                }
            )
        self.solve_cache: dict[tuple[int, str, float, float], np.ndarray] = {}
        self.score_log: list[dict[str, object]] = []

    def component_vector(
        self,
        context_index: int,
        prefix: str,
        off_alpha: float,
        def_alpha: float,
    ) -> np.ndarray:
        key = (context_index, prefix, float(off_alpha), float(def_alpha))
        if key not in self.solve_cache:
            context = self.contexts[context_index]
            result = _solve_direct_ridge_problem(
                context["component_problems"][prefix],
                self.name_map,
                off_alpha=float(off_alpha),
                def_alpha=float(def_alpha),
                prior_results=context["priors"][prefix],
                prior_scale=self.prior_scale,
            )
            self.solve_cache[key] = _result_vector(
                result,
                context["validation_problem"],
            )
        return self.solve_cache[key]

    def score(
        self,
        alphas: dict[str, tuple[float, float]],
        log: bool = True,
    ) -> float:
        fold_mses = []
        for context_index, context in enumerate(self.contexts):
            problem = context["validation_problem"]
            coefficient_sum = np.zeros(problem["X"].shape[1], dtype=float)
            for prefix, (off_alpha, def_alpha) in alphas.items():
                coefficient_sum += self.component_vector(
                    context_index,
                    prefix,
                    off_alpha,
                    def_alpha,
                )
            prediction = (
                float(context["intercept"])
                + problem["X"] @ coefficient_sum
            )
            fold_mses.append(
                float(
                    np.mean(
                        np.square(
                            np.asarray(problem["y"], dtype=float)
                            - np.asarray(prediction, dtype=float)
                        )
                    )
                )
            )
        value = float(np.mean(fold_mses))
        if log:
            self.score_log.append(
                {
                    "alphas": json.dumps(
                        {key: list(pair) for key, pair in alphas.items()},
                        sort_keys=True,
                    ),
                    "mean_mse": value,
                    "fold_mses": json.dumps(fold_mses),
                }
            )
        return value


def _parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-dir", type=Path, default=DEFAULT_BOX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--grid",
        default=",".join(str(value) for value in DEFAULT_GRID),
    )
    parser.add_argument("--prior-scale", type=float, default=1.5)
    parser.add_argument("--max-passes", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid = _parse_grid(args.grid)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    development = JointAlphaEvaluator(
        args.box_dir,
        [(2023, 2024), (2024, 2025)],
        args.prior_scale,
    )
    seeds = {
        "uniform": UNIFORM_SEED,
        "screen": SCREEN_SEED,
    }
    seed_results = {}
    best_label = ""
    best_config: dict[str, tuple[float, float]] | None = None
    best_score = float("inf")
    all_steps = []
    for label, seed in seeds.items():
        print(f"Starting coordinate descent from {label} seed...", flush=True)
        started = time.time()
        selected, score, steps = coordinate_descent(
            seed,
            ADDITIVE_PREFIXES,
            grid,
            development.score,
            args.max_passes,
        )
        seed_results[label] = {
            "final_alphas": {
                key: list(pair) for key, pair in selected.items()
            },
            "development_mse": score,
            "seconds": round(time.time() - started, 3),
        }
        for step in steps:
            all_steps.append({"seed": label, **step})
        if score < best_score:
            best_label = label
            best_config = selected
            best_score = score
    assert best_config is not None

    uniform_mse = development.score(UNIFORM_SEED)
    max_alpha = max(grid)
    max_shrink = {
        prefix: (max_alpha, max_alpha) for prefix in ADDITIVE_PREFIXES
    }
    max_shrink_mse = development.score(max_shrink)

    confirmation = JointAlphaEvaluator(
        args.box_dir,
        [(2025, 2026)],
        args.prior_scale,
    )
    confirmation_mse = confirmation.score(best_config)
    uniform_confirmation_mse = confirmation.score(UNIFORM_SEED)
    max_shrink_confirmation_mse = confirmation.score(max_shrink)

    trust_rows = []
    for prefix in ADDITIVE_PREFIXES:
        for side_index, side_name in ((0, "off"), (1, "def")):
            selected_alpha = best_config[prefix][side_index]
            ablated = dict(best_config)
            current = ablated[prefix]
            ablated[prefix] = (
                (max_alpha, current[1])
                if side_index == 0
                else (current[0], max_alpha)
            )
            development_ablation = development.score(ablated)
            confirmation_ablation = confirmation.score(ablated)
            development_gain = (development_ablation - best_score) * 1e6
            confirmation_gain = (
                confirmation_ablation - confirmation_mse
            ) * 1e6
            trust_rows.append(
                {
                    "component": prefix,
                    "side": side_name,
                    "selected_alpha": selected_alpha,
                    "max_grid_alpha": max_alpha,
                    "development_mse_gain_vs_max_shrink_x1e6": development_gain,
                    "confirmation_2026_mse_gain_vs_max_shrink_x1e6": confirmation_gain,
                    "trust": classify_trust(
                        selected_alpha,
                        max_alpha,
                        development_gain,
                        confirmation_gain,
                    ),
                }
            )

    report = {
        "status": "complete",
        "method": "joint component-sum coordinate descent",
        "trust_minimum_mse_gain_x1e6": 1.0,
        "development_folds": [[2023, 2024], [2024, 2025]],
        "confirmation_fold": [2025, 2026],
        "selection_uses_2026": False,
        "prior_scale": args.prior_scale,
        "grid": grid,
        "winning_seed": best_label,
        "selected_alphas": {
            key: list(pair) for key, pair in best_config.items()
        },
        "development": {
            "selected_mse": best_score,
            "uniform_5000_40000_mse": uniform_mse,
            "max_shrink_mse": max_shrink_mse,
            "selected_gain_vs_uniform_x1e6": (
                uniform_mse - best_score
            )
            * 1e6,
        },
        "confirmation_2026": {
            "selected_mse": confirmation_mse,
            "uniform_5000_40000_mse": uniform_confirmation_mse,
            "max_shrink_mse": max_shrink_confirmation_mse,
            "selected_gain_vs_uniform_x1e6": (
                uniform_confirmation_mse - confirmation_mse
            )
            * 1e6,
        },
        "seeds": seed_results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    pd.DataFrame(trust_rows).to_csv(
        output_dir / "factor_trust.csv",
        index=False,
    )
    pd.DataFrame(all_steps).to_csv(
        output_dir / "coordinate_steps.csv",
        index=False,
    )
    pd.DataFrame(
        development.score_log + confirmation.score_log
    ).to_csv(output_dir / "score_evaluations.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    print(pd.DataFrame(trust_rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

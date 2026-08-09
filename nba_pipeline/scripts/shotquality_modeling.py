"""Shared, leakage-aware utilities for ShotQuality-era model experiments."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


EPS = 1e-6


def normalize_game_id(values: pd.Series) -> pd.Series:
    """Return stable ten-character NBA game ids."""
    return (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(10)
    )


def parse_descriptor_tags(value: object) -> frozenset[str]:
    """Parse a ShotQuality descriptor list without executing arbitrary input."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return frozenset()
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(item).strip() for item in value if str(item).strip())
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return frozenset()
    if not isinstance(parsed, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(item).strip() for item in parsed if str(item).strip())


def clip_probability(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def calibration_error(
    y_true: Iterable[int],
    probability: Iterable[float],
    *,
    bins: int = 10,
) -> float:
    """Shot-weighted expected calibration error with quantile bins."""
    y = np.asarray(y_true, dtype=float)
    p = clip_probability(probability)
    if len(y) == 0:
        return float("nan")
    quantiles = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return float(abs(y.mean() - p.mean()))
    bucket = np.digitize(p, quantiles[1:-1], right=True)
    total = 0.0
    for idx in range(len(quantiles) - 1):
        mask = bucket == idx
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return float(total)


def probability_metrics(
    y_true: Iterable[int],
    probability: Iterable[float],
    *,
    shot_value: float | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = clip_probability(probability)
    metrics = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ece_10_quantile": calibration_error(y, p, bins=10),
        "mean_prediction": float(p.mean()),
    }
    if shot_value is not None:
        actual_points = y.astype(float) * float(shot_value)
        expected_points = p * float(shot_value)
        metrics["points_mae"] = float(np.mean(np.abs(actual_points - expected_points)))
        metrics["points_rmse"] = float(
            np.sqrt(np.mean(np.square(actual_points - expected_points)))
        )
    return metrics


def paired_cluster_bootstrap(
    y_true: Iterable[int],
    challenger_probability: Iterable[float],
    baseline_probability: Iterable[float],
    groups: Iterable[object],
    *,
    iterations: int = 2000,
    seed: int = 20260723,
) -> dict[str, Mapping[str, float]]:
    """Paired game-cluster bootstrap for challenger-minus-baseline loss."""
    y = np.asarray(y_true, dtype=float)
    challenger = clip_probability(challenger_probability)
    baseline = clip_probability(baseline_probability)
    group_values = pd.Series(groups, dtype="string").fillna("UNKNOWN").to_numpy()

    loss_differences = {
        "brier": np.square(challenger - y) - np.square(baseline - y),
        "log_loss": (
            -(y * np.log(challenger) + (1.0 - y) * np.log(1.0 - challenger))
            + y * np.log(baseline)
            + (1.0 - y) * np.log(1.0 - baseline)
        ),
    }
    unique_groups, inverse = np.unique(group_values, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(unique_groups), size=(iterations, len(unique_groups))
    )

    output: dict[str, Mapping[str, float]] = {}
    for metric_name, differences in loss_differences.items():
        sums = np.bincount(inverse, weights=differences)
        sampled_sum = sums[sampled_indices].sum(axis=1)
        sampled_count = counts[sampled_indices].sum(axis=1)
        draws = sampled_sum / sampled_count
        output[metric_name] = {
            "challenger_minus_baseline": float(differences.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "bootstrap_iterations": int(iterations),
            "cluster_count": int(len(unique_groups)),
        }
    return output


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

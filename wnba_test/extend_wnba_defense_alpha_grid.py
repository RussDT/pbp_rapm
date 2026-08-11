#!/usr/bin/env python3
"""Preregistered defense-alpha grid extension on development folds only.

Binding contract: nba_pipeline/docs/WNBA_RTSS_CONFIRMATION_PREREG.md, section
"Defense-alpha grid extension". Evaluate UNIFORM defense alphas
(40000, 60000, 80000, 120000, 160000) with offense fixed at 5000 and prior
scale 1.5, scored with the joint component-sum future-possession objective and
fold construction imported from ``tune_wnba_component_alphas`` (the
``JointAlphaEvaluator``) on the development forward folds 2023->2024 and
2024->2025 ONLY. 2026 is never read. Selection rule: lowest pooled development
MSE, but a challenger must beat 40000 by more than the tuner's trust minimum
(1.0e-6 absolute MSE); ties or within-noise keep 40000 (status quo bias is
intentional). Writes ``report.json`` and ``decision.json`` under
``wnba_test/Results/alpha_grid_extension/``; the 2027 live config and the
confirmation harness read ``decision.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import ADDITIVE_PREFIXES
from wnba_test.clean_decomp_wnba import RESULTS_DIR
from wnba_test.tune_wnba_component_alphas import (
    DEFAULT_BOX_DIR,
    JointAlphaEvaluator,
)

DEFENSE_GRID = [40000.0, 60000.0, 80000.0, 120000.0, 160000.0]
OFFENSE_ALPHA = 5000.0
PRIOR_SCALE = 1.5
STATUS_QUO_DEF_ALPHA = 40000.0
DEVELOPMENT_FOLDS = [(2023, 2024), (2024, 2025)]
# Tuner trust convention (trust_minimum_mse_gain_x1e6 = 1.0): a pooled-MSE
# gain of 1.0e-6 or less is within noise and does not displace the status quo.
MINIMUM_GAIN_MSE = 1.0e-6
DEFAULT_OUTPUT = RESULTS_DIR / "alpha_grid_extension"
TUNER_REPORT_PATH = (
    RESULTS_DIR / "box_prior_true_prior" / "component_alpha_tuning" / "report.json"
)
DECISION_RULE = (
    "Lowest pooled development MSE over folds 2023->2024 and 2024->2025 with "
    "uniform defense alpha in (40000, 60000, 80000, 120000, 160000), offense "
    "fixed at 5000, prior scale 1.5; a challenger replaces 40000 only if it "
    "beats the 40000 pooled MSE by more than 1.0e-6 absolute MSE (the tuner's "
    "trust minimum). Ties or within-noise keep 40000. 2026 excluded from "
    "selection."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-dir", type=Path, default=DEFAULT_BOX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for rating_year, validation_year in DEVELOPMENT_FOLDS:
        if rating_year >= 2026 or validation_year >= 2026:
            raise AssertionError(
                "confirmation-season fence: development folds must not touch 2026"
            )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started_total = time.time()
    prep_started = time.time()
    evaluator = JointAlphaEvaluator(args.box_dir, DEVELOPMENT_FOLDS, PRIOR_SCALE)
    prep_seconds = time.time() - prep_started

    rows: list[dict[str, object]] = []
    for def_alpha in DEFENSE_GRID:
        alphas = {
            prefix: (OFFENSE_ALPHA, def_alpha) for prefix in ADDITIVE_PREFIXES
        }
        eval_started = time.time()
        pooled = float(evaluator.score(alphas))
        seconds = time.time() - eval_started
        fold_mses = [
            float(value)
            for value in json.loads(str(evaluator.score_log[-1]["fold_mses"]))
        ]
        rows.append(
            {
                "def_alpha": float(def_alpha),
                "fold_mses": {
                    f"{rating}->{validation}": fold_mses[index]
                    for index, (rating, validation) in enumerate(DEVELOPMENT_FOLDS)
                },
                "pooled_dev_mse": pooled,
                "seconds": round(seconds, 3),
            }
        )
        print(
            f"def_alpha={def_alpha:.0f}: pooled_dev_mse={pooled:.10f} "
            f"({seconds:.1f}s)",
            flush=True,
        )

    pooled_by_alpha = {row["def_alpha"]: row["pooled_dev_mse"] for row in rows}
    baseline_mse = float(pooled_by_alpha[STATUS_QUO_DEF_ALPHA])
    for row in rows:
        row["gain_vs_40000_x1e6"] = (
            baseline_mse - float(row["pooled_dev_mse"])
        ) * 1e6

    challengers = {
        alpha: mse
        for alpha, mse in pooled_by_alpha.items()
        if alpha != STATUS_QUO_DEF_ALPHA
    }
    # Deterministic tie-break among challengers: lower alpha (less shrinkage,
    # closer to status quo) first.
    best_challenger = min(challengers, key=lambda a: (challengers[a], a))
    margin = baseline_mse - float(challengers[best_challenger])
    beat_40000 = margin > MINIMUM_GAIN_MSE
    selected_def_alpha = (
        float(best_challenger) if beat_40000 else STATUS_QUO_DEF_ALPHA
    )

    baseline_reproduction: dict[str, object] = {
        "tuner_report_path": str(TUNER_REPORT_PATH),
        "tuner_uniform_5000_40000_dev_mse": None,
        "this_run_40000_dev_mse": baseline_mse,
        "abs_difference": None,
    }
    if TUNER_REPORT_PATH.exists():
        tuner_report = json.loads(TUNER_REPORT_PATH.read_text())
        tuner_uniform = float(
            tuner_report["development"]["uniform_5000_40000_mse"]
        )
        baseline_reproduction["tuner_uniform_5000_40000_dev_mse"] = tuner_uniform
        baseline_reproduction["abs_difference"] = abs(
            tuner_uniform - baseline_mse
        )

    report = {
        "status": "complete",
        "prereg": (
            "nba_pipeline/docs/WNBA_RTSS_CONFIRMATION_PREREG.md"
            "#defense-alpha-grid-extension"
        ),
        "method": (
            "joint component-sum future-possession MSE via "
            "tune_wnba_component_alphas.JointAlphaEvaluator (imported, not "
            "reimplemented); uniform defense alpha across the nine additive "
            "components, offense fixed"
        ),
        "fold_definition": {
            "folds": [list(pair) for pair in DEVELOPMENT_FOLDS],
            "per_fold_construction": (
                "rating year Y (season type ALL): nine prior-centered "
                "component ridge solves around box_priors_{Y}.csv with prior "
                "scale 1.5; intercept = mean clean-total possession points of "
                "year Y; validation year Y+1 (season type ALL): prediction = "
                "intercept + X_validation @ sum(component coefficient "
                "vectors), MSE against actual possession points"
            ),
            "components": list(ADDITIVE_PREFIXES),
            "selection_uses_2026": False,
        },
        "offense_alpha": OFFENSE_ALPHA,
        "prior_scale": PRIOR_SCALE,
        "defense_grid": [float(value) for value in DEFENSE_GRID],
        "box_dir": str(args.box_dir),
        "results": rows,
        "baseline_reproduction": baseline_reproduction,
        "prep_seconds": round(prep_seconds, 3),
        "total_seconds": round(time.time() - started_total, 3),
    }
    decision = {
        "selected_def_alpha": selected_def_alpha,
        "beat_40000": bool(beat_40000),
        "margin": margin,
        "margin_x1e6": margin * 1e6,
        "best_challenger_def_alpha": float(best_challenger),
        "baseline_40000_pooled_dev_mse": baseline_mse,
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_rule": DECISION_RULE,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

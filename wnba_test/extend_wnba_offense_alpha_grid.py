#!/usr/bin/env python3
"""Offense-alpha grid extension on development folds only.

Companion to ``extend_wnba_defense_alpha_grid.py``. That script resolved the
upper defense edge (40000 held; the best challenger, 60000, lost by 6.3e-6).
The live offense alpha ``5000`` sits on the *lower* edge of the audited
``true_prior_grid.csv`` sweep, which only tested (5000, 7500, 10000), so the
question of whether less offensive shrinkage helps was never asked.

Extend downward: (1000, 2000, 3000, 4000, 5000) with defense fixed at 40000
and prior scale 1.5, scored with the joint component-sum future-possession
objective and fold construction imported from ``tune_wnba_component_alphas``
(the ``JointAlphaEvaluator``) on the development forward folds 2023->2024 and
2024->2025 ONLY. 2026 is never read.

Selection rule mirrors the defense extension exactly: lowest pooled
development MSE, but a challenger must beat 5000 by more than the tuner's
trust minimum (1.0e-6 absolute MSE); ties or within-noise keep 5000 (status
quo bias is intentional).
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

# Two-sided. A downward-only sweep cannot close the edge: pooled MSE falls
# monotonically as alpha rises toward 5000, so the upward arm is required to
# show 5000 is an interior minimum rather than a new boundary.
OFFENSE_GRID = [
    1000.0,
    2000.0,
    3000.0,
    4000.0,
    5000.0,
    6000.0,
    7500.0,
    10000.0,
    15000.0,
    20000.0,
]
DEFENSE_ALPHA = 40000.0
PRIOR_SCALE = 1.5
STATUS_QUO_OFF_ALPHA = 5000.0
DEVELOPMENT_FOLDS = [(2023, 2024), (2024, 2025)]
MINIMUM_GAIN_MSE = 1.0e-6
DEFAULT_OUTPUT = RESULTS_DIR / "alpha_grid_extension_offense"
DECISION_RULE = (
    "Lowest pooled development MSE over folds 2023->2024 and 2024->2025 with "
    "uniform offense alpha in (1000, 2000, 3000, 4000, 5000), defense fixed "
    "at 40000, prior scale 1.5; a challenger replaces 5000 only if it beats "
    "the 5000 pooled MSE by more than 1.0e-6 absolute MSE (the tuner's trust "
    "minimum). Ties or within-noise keep 5000. 2026 excluded from selection."
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
    for off_alpha in OFFENSE_GRID:
        alphas = {
            prefix: (off_alpha, DEFENSE_ALPHA) for prefix in ADDITIVE_PREFIXES
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
                "off_alpha": float(off_alpha),
                "fold_mses": {
                    f"{rating}->{validation}": fold_mses[index]
                    for index, (rating, validation) in enumerate(DEVELOPMENT_FOLDS)
                },
                "pooled_dev_mse": pooled,
                "seconds": round(seconds, 3),
            }
        )
        print(
            f"off_alpha={off_alpha:.0f}: pooled_dev_mse={pooled:.10f} "
            f"({seconds:.1f}s)",
            flush=True,
        )

    pooled_by_alpha = {row["off_alpha"]: row["pooled_dev_mse"] for row in rows}
    baseline_mse = float(pooled_by_alpha[STATUS_QUO_OFF_ALPHA])
    for row in rows:
        row["gain_vs_5000_x1e6"] = (
            baseline_mse - float(row["pooled_dev_mse"])
        ) * 1e6

    challengers = {
        alpha: mse
        for alpha, mse in pooled_by_alpha.items()
        if alpha != STATUS_QUO_OFF_ALPHA
    }
    # Deterministic tie-break among challengers: higher alpha (more
    # shrinkage, closer to status quo) first.
    best_challenger = min(challengers, key=lambda a: (challengers[a], -a))
    margin = baseline_mse - float(challengers[best_challenger])
    beat_5000 = margin > MINIMUM_GAIN_MSE
    selected_off_alpha = (
        float(best_challenger) if beat_5000 else STATUS_QUO_OFF_ALPHA
    )

    report = {
        "status": "complete",
        "companion": (
            "wnba_test/extend_wnba_defense_alpha_grid.py resolved the upper "
            "defense edge in the same cycle"
        ),
        "method": (
            "joint component-sum future-possession MSE via "
            "tune_wnba_component_alphas.JointAlphaEvaluator (imported, not "
            "reimplemented); uniform offense alpha across the nine additive "
            "components, defense fixed"
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
        "defense_alpha": DEFENSE_ALPHA,
        "prior_scale": PRIOR_SCALE,
        "offense_grid": [float(value) for value in OFFENSE_GRID],
        "box_dir": str(args.box_dir),
        "results": rows,
        "prep_seconds": round(prep_seconds, 3),
        "total_seconds": round(time.time() - started_total, 3),
    }
    decision = {
        "selected_off_alpha": selected_off_alpha,
        "beat_5000": bool(beat_5000),
        "margin": margin,
        "margin_x1e6": margin * 1e6,
        "best_challenger_off_alpha": float(best_challenger),
        "baseline_5000_pooled_dev_mse": baseline_mse,
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

import pandas as pd

from nba_pipeline.scripts.analyze_longitudinal_age_curve_sensitivity import (
    compare_age_curves,
)


def test_convergence_receipt_uses_core_supported_ages_and_net_tolerance():
    base = pd.DataFrame(
        {
            "age": [19, 20, 27, 39, 40],
            "orapm_age_effect": [-1.0, -0.8, 0.0, -3.8, -4.0],
            "drapm_age_effect": [-0.9, -0.7, 0.0, -0.9, -1.0],
            "supported_age": [False, True, True, True, False],
        }
    )
    close = base.copy()
    close["orapm_age_effect"] += [0.5, 0.04, 0.0, 0.08, 0.5]
    close["drapm_age_effect"] += [0.5, 0.03, 0.0, 0.04, 0.5]

    comparison, receipt = compare_age_curves(
        {"alpha20": base, "alpha5": close},
        reference_label="alpha5",
        convergence_label="alpha20",
        net_tolerance=0.20,
    )

    assert receipt["converged"]
    assert receipt["max_abs_net_difference"] == 0.12
    assert set(comparison["model"]) == {"alpha20", "alpha5"}


from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_historical_five_man_epm_fit import (
    FIXED_OFFENSE_RAW_COEFFICIENTS,
    expand_fit,
    fit_ridge,
)


def test_fixed_shortfall_coefficient_cannot_flip_positive() -> None:
    frame = pd.DataFrame(
        {
            "role_demand_shortfall": [0.00, 0.01, 0.05, 0.20],
            "usage_hhi": [0.20, 0.21, 0.22, 0.23],
            "target": [0.0, 0.0, 1.0, 5.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        }
    )
    fit = fit_ridge(
        frame,
        ["role_demand_shortfall", "usage_hhi"],
        "target",
        "weight",
        alpha=10.0,
        fixed_raw_coefficients=FIXED_OFFENSE_RAW_COEFFICIENTS,
    )
    index = fit.feature_names.index("role_demand_shortfall")
    raw_coefficient = fit.coefficients[index] / fit.scale[index]
    assert np.isclose(raw_coefficient, -15.0)
    contribution = fit.contributions(frame)["role_demand_shortfall"]
    assert contribution.iloc[-1] < contribution.iloc[0]


def test_expanded_fit_keeps_predictions_and_zeroes_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "role_demand_shortfall": [0.00, 0.02, 0.10, 0.20],
            "usage_hhi": [0.20, 0.22, 0.24, 0.26],
            "onball_sum": [0.8, 0.9, 1.0, 1.1],
            "target": [0.0, -0.5, -1.0, -2.0],
            "weight": [100.0, 110.0, 120.0, 130.0],
        }
    )
    fit = fit_ridge(
        frame,
        ["role_demand_shortfall", "usage_hhi"],
        "target",
        "weight",
        alpha=10.0,
        fixed_raw_coefficients=FIXED_OFFENSE_RAW_COEFFICIENTS,
    )
    expanded = expand_fit(
        fit,
        frame,
        features=["role_demand_shortfall", "usage_hhi", "onball_sum"],
        weight="weight",
    )
    np.testing.assert_allclose(expanded.predict(frame), fit.predict(frame))
    assert expanded.coefficients[-1] == 0.0

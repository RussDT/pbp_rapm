import numpy as np
import pandas as pd

from nba_pipeline.scripts.rapm import (
    build_age_lookup_from_source,
    compute_fixed_age_poly_offsets,
    load_age_value_curves,
)


def test_frozen_positive_good_age_table_maps_to_solver_offense_and_raw_defense(tmp_path):
    curve_path = tmp_path / "age_curve.csv"
    pd.DataFrame(
        {
            "age": [20, 27, 30],
            "orapm_age_effect": [-2.0, 0.0, -0.6],
            "drapm_age_effect": [-1.0, 0.0, -0.3],
        }
    ).to_csv(curve_path, index=False)
    curves, metadata = load_age_value_curves(curve_path)
    age_rows = []
    for player_id in range(1, 11):
        age_rows.append(
            {
                "nba_id": player_id,
                "year": 2026,
                "Age": 30.0 if player_id <= 5 else 27.0,
            }
        )
    age_lookup = build_age_lookup_from_source(pd.DataFrame(age_rows))
    frame = pd.DataFrame(
        {
            "season": [2026],
            "home_poss": [1],
            **{f"h{i}": [i] for i in range(1, 6)},
            **{f"a{i}": [i + 5] for i in range(1, 6)},
        }
    )

    offsets, summary = compute_fixed_age_poly_offsets(
        frame,
        curves,
        age_lookup,
        reference_age=27,
        min_age=metadata["age_min"],
        max_age=metadata["age_max"],
    )

    # Five age-30 offensive players contribute 5 * -0.6 points/100.
    np.testing.assert_allclose(offsets, [-0.03])
    assert summary["mapped_slots"] == 10
    assert curves["def"](30.0) == 0.3  # solver defense is raw points allowed


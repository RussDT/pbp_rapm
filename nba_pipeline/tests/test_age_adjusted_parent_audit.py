import pandas as pd

from nba_pipeline.scripts.build_age_adjusted_parent_audit import build_parent_audit


def test_parent_audit_flips_defense_and_exposes_reference_age_extrapolation():
    results = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["Older", "Prime"],
            "off": [5.0, 2.0],
            "def": [-3.0, 1.0],
            "net_rapm": [8.0, 1.0],
            "possessions": [4000, 2000],
            "off_poss": [2000, 1000],
            "def_poss": [2000, 1000],
        }
    )
    panel = pd.DataFrame(
        {
            "player_id": ["1", "1", "2"],
            "season": [2024, 2025, 2025],
            "age": [34.0, 36.0, 27.0],
            "off_poss": [1000, 3000, 1000],
            "def_poss": [2000, 2000, 1000],
        }
    )
    curve = pd.DataFrame(
        {
            "age": [27, 34, 36],
            "orapm_age_effect": [0.0, -2.0, -4.0],
            "drapm_age_effect": [0.0, -1.0, -2.0],
            "supported_age": [True, True, False],
        }
    )

    audit, receipt = build_parent_audit(results, panel, curve, reference_age=27.0)
    older = audit.loc[audit["player_id"] == "1"].iloc[0]
    prime = audit.loc[audit["player_id"] == "2"].iloc[0]

    assert older["orapm_age27"] == 5.0
    assert older["drapm_age27"] == 3.0
    assert older["net_age27"] == 8.0
    assert older["off_mean_age"] == 35.5
    assert older["def_mean_age"] == 35.0
    assert older["off_age_adjustment_to_27"] == 3.5
    assert older["def_age_adjustment_to_27"] == 1.5
    assert older["net_age_adjustment_to_27"] == 5.0
    assert older["net_at_observed_age"] == 3.0
    assert older["nearest_age_to_reference"] == 7.0
    assert older["reference_age_support"] == "long_extrapolation"
    assert older["off_supported_exposure_share"] == 0.25
    assert older["def_supported_exposure_share"] == 0.5

    assert prime["reference_age_support"] == "near_reference"
    assert prime["net_age_adjustment_to_27"] == 0.0
    assert receipt["result_players"] == 2
    assert receipt["players_with_age_audit"] == 2
    assert receipt["defense_output_sign"] == "positive_good"

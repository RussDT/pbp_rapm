import pandas as pd

from nba_pipeline.scripts.evaluate_age_curve_candidates import evaluate_candidates


def test_matching_age_curve_beats_zero_on_consecutive_season_changes():
    panel = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "season": [2024, 2025, 2024, 2025],
            "age": [27.0, 28.0, 27.0, 28.0],
            "orapm": [2.0, 1.0, 0.0, -1.0],
            "drapm": [1.0, 0.5, 0.0, -0.5],
            "off_poss": [1000] * 4,
            "def_poss": [1000] * 4,
        }
    )
    curve = pd.DataFrame(
        {
            "age": [27, 28],
            "orapm_age_effect": [0.0, -1.0],
            "drapm_age_effect": [0.0, -0.5],
        }
    )

    scores, receipt = evaluate_candidates(panel, {"matching": curve})
    matching_net = scores[(scores["candidate"] == "matching") & (scores["target"] == "net")].iloc[0]
    zero_net = scores[(scores["candidate"] == "zero") & (scores["target"] == "net")].iloc[0]

    assert matching_net["weighted_mse"] == 0.0
    assert zero_net["weighted_mse"] == 2.25
    assert matching_net["mse_improvement_vs_zero"] == 2.25
    assert receipt["consecutive_pairs"] == 2
    assert receipt["players"] == 2

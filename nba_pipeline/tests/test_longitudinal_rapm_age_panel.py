import pandas as pd

from nba_pipeline.scripts.build_longitudinal_rapm_age_panel import assemble_age_panel


def test_age_panel_uses_latest_season_age_and_flips_defense_positive_good():
    annual_results = {
        2010: pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["One", "Two"],
                "off": [1.5, -0.5],
                "def": [-2.0, 1.0],
                "off_poss": [1000, 800],
                "def_poss": [990, 790],
            }
        ),
        2011: pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["One", "Two"],
                "off": [1.7, -0.2],
                "def": [-1.8, 0.8],
                "off_poss": [1100, 900],
                "def_poss": [1090, 890],
            }
        ),
    }
    age_history = pd.DataFrame(
        {
            "nba_id": [1, 1, 1, 2, 2],
            "season": [2010, 2010, 2011, 2010, 2011],
            "date": ["2010-02-01", "2010-05-01", "2011-05-01", "2010-05-01", "2011-05-01"],
            "age": [24.7, 25.0, 26.0, 29.0, 30.0],
        }
    )

    panel, receipt = assemble_age_panel(annual_results, age_history)

    one_2010 = panel[(panel["player_id"] == "1") & (panel["season"] == 2010)].iloc[0]
    assert one_2010["age"] == 25.0
    assert one_2010["orapm"] == 1.5
    assert one_2010["drapm"] == 2.0
    assert receipt["age_coverage"] == 1.0


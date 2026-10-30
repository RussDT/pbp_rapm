import numpy as np
import pandas as pd
import pytest

from nba_pipeline.scripts.process_rapm_blocks import clean_chance_utils as chance


@pytest.mark.parametrize("stats_available", [True, False])
def test_missing_ft_percentage_uses_actuals_in_parent_and_ft_child(monkeypatch, stats_available):
    stats = pd.DataFrame({"PlayerID": [1], "FTPerc": [0.8], "ThreePerc": [0.35]})
    monkeypatch.setattr(chance, "fetch_player_stats_supabase", lambda *args: stats if stats_available else None)
    # Player 1 has a known percentage; player 2 is absent from the stats panel.
    events = pd.DataFrame({
        "player1_id": [1, 1, 2, 2],
        "home_description": ["Free Throw 1 of 2", "MISS Free Throw 2 of 2"] * 2,
        "visitor_description": [None] * 4,
        "event_type": ["FreeThrow"] * 4,
        "Home_Action_Score": [1.0, 0.0, 1.0, 0.0],
        "Away_Action_Score": [0.0] * 4,
    })
    scored = chance._attach_rapm_scoring(events, "NBA26.parquet", 26, "test")
    expected = [0.8, 0.8, 1.0, 0.0] if stats_available else [1.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(scored["ActualNet"], expected)
    np.testing.assert_allclose(
        scored["FC_FT_Action_Score"] + scored["FC_FT_SQ_BASELINE_Action_Score"],
        expected,
    )

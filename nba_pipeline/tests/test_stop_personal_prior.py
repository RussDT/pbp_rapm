import sys
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tune_stop_personal_prior import aggregate_personal_stop_prior


def test_personal_stop_prior_pools_counts_and_uses_negative_good_defense():
    history = pd.DataFrame(
        [
            {"nba_id": 1, "Name": "A", "year": 2024, "DefPoss": 100, "STOPS_100": 4.0, "credited_stops": 4.0},
            {"nba_id": 1, "Name": "A", "year": 2025, "DefPoss": 300, "STOPS_100": 2.0, "credited_stops": 6.0},
            {"nba_id": 2, "Name": "B", "year": 2024, "DefPoss": 400, "STOPS_100": 1.0, "credited_stops": 4.0},
        ]
    )

    prior = aggregate_personal_stop_prior(history, 2024, 2025).set_index("nba_id")

    assert prior.loc[1, "personal_stop_pct"] == 2.5
    assert prior.loc[1, "league_stop_pct"] == 1.75
    assert prior.loc[1, "relative_stop_pct"] == 0.75
    assert prior.loc[1, "prior_value"] == -0.75
    assert prior.loc[2, "prior_value"] == 0.75

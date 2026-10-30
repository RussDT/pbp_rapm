import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tune_stop_td_epm_prior as module


def _history():
    rows = []
    for year in [2021, 2022, 2023]:
        rows.extend(
            [
                {
                    "nba_id": 1,
                    "Name": "Specialist",
                    "year": year,
                    "DefPoss": 1000.0,
                    "OFFD_100": 1.0,
                    "d_Blocks_Per100": 2.0,
                    "block_recov_percent": 0.75,
                    "v2_charges_drawn_total": 4.0,
                    "Pos2": "SG",
                    "Defensive Role": "Point of Attack",
                },
                {
                    "nba_id": 2,
                    "Name": "Baseline",
                    "year": year,
                    "DefPoss": 1000.0,
                    "OFFD_100": 0.0,
                    "d_Blocks_Per100": 1.0,
                    "block_recov_percent": 0.25,
                    "v2_charges_drawn_total": 0.0,
                    "Pos2": "C",
                    "Defensive Role": "Anchor Big",
                },
            ]
        )
    frame = pd.DataFrame(rows)
    frame["offd_count"] = frame["OFFD_100"] * frame["DefPoss"] / 100.0
    frame["block_count"] = frame["d_Blocks_Per100"] * frame["DefPoss"] / 100.0
    frame["recovered_block_count"] = frame["block_count"] * frame["block_recov_percent"]
    frame["charge_count"] = frame["v2_charges_drawn_total"]
    frame["league_offd_rate"] = 0.5
    return frame


def test_td_panel_preserves_repeat_offd_signal_and_recovery_counts():
    panel = module.build_td_mechanism_panel(_history(), 2023, 700).set_index("nba_id")
    assert panel.loc[1, "td_offd_rate"] == 1.0
    assert panel.loc[2, "td_offd_rate"] == 0.0
    assert panel.loc[1, "above_league_year_share"] == 1.0
    assert panel.loc[2, "above_league_year_share"] == 0.0
    assert np.isclose(panel.loc[1, "td_recovery_rate"], 0.75)
    assert np.isclose(panel.loc[2, "td_recovery_rate"], 0.25)


def test_recovery_shrinkage_respects_player_block_sample():
    panel = module.build_td_mechanism_panel(_history(), 2023, 700)
    prediction = module.recovery_prediction(panel, pseudo_blocks=20.0)
    specialist = prediction[panel["nba_id"].to_numpy() == 1][0]
    baseline = prediction[panel["nba_id"].to_numpy() == 2][0]
    assert specialist > baseline
    assert 0.25 < baseline < specialist < 0.75


def test_charges_are_auxiliary_not_added_to_expected_stop_identity():
    assert "td_charge_rate" in module.OFFD_NUMERIC
    assert all("charge" not in column for column in module.ARM_COLUMNS["epm_steals_recovered_blocks_offd"])

import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tune_stop_feature_prior.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("tune_stop_feature_prior", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_trailing_panel_pools_rates_and_uses_latest_role():
    rows = []
    for year, poss, stop, dfga, role in [
        (2021, 100.0, 2.0, 10.0, "Chaser"),
        (2022, 300.0, 4.0, 20.0, "Helper"),
        (2023, 600.0, 3.0, 30.0, None),
    ]:
        rows.append(
            {
                "nba_id": 1,
                "Name": "A",
                "year": year,
                "DefPoss": poss,
                "STOPS_100": stop,
                "dfga/100": dfga,
                "Pos2": "SG",
                "Defensive Role": role,
            }
        )
        rows.append(
            {
                "nba_id": 2,
                "Name": "B",
                "year": year,
                "DefPoss": poss,
                "STOPS_100": 1.0,
                "dfga/100": 5.0,
                "Pos2": "C",
                "Defensive Role": "Anchor Big",
            }
        )
    panel = MODULE.build_trailing_feature_panel(pd.DataFrame(rows), 2023).set_index("nba_id")
    assert panel.loc[1, "STOPS_100"] == 3.2
    assert panel.loc[1, "dfga/100"] == 25.0
    assert panel.loc[1, "Defensive_Role_filled"] == "Helper"
    assert panel.loc[1, "source_def_poss"] == 1000.0
    assert panel.loc[1, "source_relative_stop_pct"] > 0


def test_mapper_examples_do_not_use_same_year_features_as_target():
    rows = []
    for year in range(2014, 2018):
        for player in [1, 2]:
            rows.append(
                {
                    "nba_id": player,
                    "Name": str(player),
                    "year": year,
                    "DefPoss": 1000.0,
                    "STOPS_100": float(player + year % 2),
                    "Pos2": "PG" if player == 1 else "C",
                    "Defensive Role": "Point of Attack" if player == 1 else "Anchor Big",
                }
            )
    examples = MODULE.build_mapper_examples(
        pd.DataFrame(rows), first_source_end=2016, last_target_year=2017
    )
    assert set(examples["source_end"]) == {2016}
    assert set(examples["target_year"]) == {2017}

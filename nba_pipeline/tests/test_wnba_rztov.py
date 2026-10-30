from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_test import build_wnba_rztov as module


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


def test_score_box_season_closes_shot_and_turnover_ledgers(monkeypatch) -> None:
    model_rows = []
    raw_rows = []
    for player_id, off_poss, tsa, scoring_tov, badpass_tov in [
        ("1", 1000.0, 20.0, 2.0, 1.0),
        ("2", 2000.0, 18.0, 1.0, 0.5),
    ]:
        row = {
            "season": 2026,
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "off_poss": off_poss,
        }
        for feature in set(
            module.ASSIST_SHARE_FEATURES + module.POTENTIAL_ASSIST_FEATURES
        ):
            row[feature] = 0.0
        model_rows.append(row)
        raw_rows.append(
            {
                "season": 2026,
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "off_poss": off_poss,
                "fga100_raw": tsa - 2.0,
                "tsa100_raw": tsa,
                "tov100_raw": scoring_tov + badpass_tov,
                "sc_tov100_raw": scoring_tov,
                "bp_tov100_raw": badpass_tov,
                "turnovers": 10.0,
            }
        )

    monkeypatch.setattr(
        module, "load_model_box", lambda _season: pd.DataFrame(model_rows)
    )
    monkeypatch.setattr(
        module, "load_raw_box", lambda _season: pd.DataFrame(raw_rows)
    )
    board = module.score_box_season(
        2026,
        _ConstantModel(0.25),
        _ConstantModel(5.0),
        {"self": 0.10, "assisted": 0.05, "potential_assist": 0.20},
    )

    assert np.allclose(board["estimated_self_created_tsa100"], [15.0, 13.5])
    assert np.allclose(board["estimated_assisted_tsa100"], [5.0, 4.5])
    assert np.allclose(board["shot_volume_identity_gap"], 0.0)
    assert np.allclose(
        board["scoring_excess100"] + board["badpass_excess100"],
        board["turnover_excess100"],
    )
    assert np.allclose(
        board["rztov_pct"],
        100.0 * board["turnover_excess100"] / board["plays100"],
    )
    assert abs(
        np.average(board["scoring_excess100"], weights=board["off_poss"])
    ) < 1e-12
    assert abs(
        np.average(board["badpass_excess100"], weights=board["off_poss"])
    ) < 1e-12


def test_weighted_metrics_match_perfect_prediction() -> None:
    metrics = module._weighted_metrics(
        np.array([1.0, 2.0, 4.0]),
        np.array([1.0, 2.0, 4.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert abs(metrics["correlation"] - 1.0) < 1e-12


def test_score_box_history_labels_fixed_retrospective_contract(monkeypatch) -> None:
    def fake_score(
        season: int,
        _share_model: object,
        _potast_model: object,
        _prices: dict[str, float],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "season": [season],
                "player_id": [str(season)],
                "shot_volume_identity_gap": [0.0],
            }
        )

    monkeypatch.setattr(module, "score_box_season", fake_score)
    history = module.score_box_history(
        [2022, 2023, 2025, 2026],
        object(),
        object(),
        {},
    )

    assert history["historical_score_status"].tolist() == [
        "pre_training_retrospective_score",
        "training_period_retrospective_score",
        "training_period_retrospective_score",
        "current_unlabeled_score",
    ]
    assert history["role_model_training_seasons"].eq("2023-2024-2025").all()
    assert history["turnover_price_training_seasons"].eq("2023-2024-2025").all()
    assert not history.duplicated(["season", "player_id"]).any()


def test_raw_box_uses_canonical_foul_trip_tsa(tmp_path, monkeypatch):
    pd.DataFrame({"EntityId": [1], "Name": ["Player"], "OffPoss": [100],
                  "FG2A": [8], "FG3A": [2], "FTA": [12],
                  "TwoPtShootingFoulsDrawn": [3], "ThreePtShootingFoulsDrawn": [2],
                  "2pt And 1 Free Throw Trips": [1], "3pt And 1 Free Throw Trips": [1],
                  "NonShootingFoulsDrawn": [2]}).to_csv(tmp_path / "2026_pbp.csv", index=False)
    monkeypatch.setattr(module, "BOX_DATA_DIR", tmp_path)
    assert module.load_raw_box(2026).iloc[0]["tsa100_raw"] == 15

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts import build_nba_rztov_box_backcast as module


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


def _raw_rows() -> pd.DataFrame:
    lineup = {
        "home_player1": 1,
        "home_player2": 2,
        "home_player3": 3,
        "home_player4": 4,
        "home_player5": 5,
        "away_player1": 6,
        "away_player2": 7,
        "away_player3": 8,
        "away_player4": 9,
        "away_player5": 10,
    }
    rows = [
        {
            **lineup,
            "event_type": 1,
            "player1_id": 1,
            "player1_name": "Shooter",
            "player2_id": np.nan,
            "home_description": "Shooter 3PT Jump Shot (3 PTS) (Passer 1 AST)",
            "neutral_description": None,
            "visitor_description": None,
        },
        {
            **lineup,
            "event_type": 2,
            "player1_id": 1,
            "player1_name": "Shooter",
            "player2_id": np.nan,
            "home_description": "MISS Shooter Driving Layup",
            "neutral_description": None,
            "visitor_description": None,
        },
        {
            **lineup,
            "event_type": 3,
            "player1_id": 1,
            "player1_name": "Shooter",
            "player2_id": np.nan,
            "home_description": "Shooter Free Throw 1 of 2",
            "neutral_description": None,
            "visitor_description": None,
        },
        {
            **lineup,
            "event_type": 5,
            "player1_id": 2,
            "player1_name": "Passer",
            "player2_id": np.nan,
            "home_description": "Passer Bad Pass Turnover",
            "neutral_description": None,
            "visitor_description": None,
        },
        {
            **lineup,
            "event_type": 1,
            "player1_id": 2,
            "player1_name": "Passer",
            "player2_id": np.nan,
            "home_description": "Passer Jump Shot (2 PTS)",
            "neutral_description": None,
            "visitor_description": None,
        },
        {
            **lineup,
            "event_type": 6,
            "player1_id": 2,
            "player1_name": "Passer",
            "player2_id": np.nan,
            "home_description": "Passer Personal Foul",
            "neutral_description": None,
            "visitor_description": None,
        },
    ]
    return pd.DataFrame(rows)


def test_raw_feature_builder_resolves_assist_and_smooths_made_share() -> None:
    possessions = pd.DataFrame(
        {
            "O1": [1, 1],
            "O2": [2, 2],
            "O3": [3, 3],
            "O4": [4, 4],
            "O5": [5, 5],
        }
    )
    features, audit = module.build_season_features_from_frames(
        _raw_rows(), possessions, 2020
    )
    shooter = features.set_index("player_id").loc[1]
    passer = features.set_index("player_id").loc[2]

    assert audit["assist_tags"] == 1
    assert audit["resolved_tags"] == 1
    assert shooter["fga"] == 2
    assert shooter["tsa100"] == 100.0 * (2.0 + 0.44) / 2.0
    assert shooter["official_assisted_make_share_raw"] == 1.0
    assert 0.0 < shooter["official_assisted_make_share_smoothed"] < 1.0
    assert passer["assists100"] == 50.0
    assert passer["assist_points100"] == 150.0
    assert passer["badpass_tov100"] == 50.0


def test_score_role_panel_preserves_tsa_closure(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "season": [1997, 2014],
            "player_id": [1, 2],
            "player_name": ["Old", "Modern"],
            "off_poss": [1000.0, 1000.0],
            "tsa100": [20.0, 25.0],
            "canonical_tsa100": [22.0, 27.0],
            "x": [0.0, 0.0],
        }
    )
    observed_path = tmp_path / "observed.csv"
    pd.DataFrame(
        {
            "nba_id": [2],
            "year": [2014],
            "off_poss": [1000.0],
            "v_self": [10.0],
            "v_ast": [15.0],
            "v_potast": [8.0],
        }
    ).to_csv(observed_path, index=False)
    scored = module.score_role_panel(
        features,
        _ConstantModel(0.4),
        ["x"],
        _ConstantModel(7.0),
        ["x"],
        observed_path,
    )

    assert np.allclose(scored["estimated_assisted_tsa100"], [8.8, 10.8])
    assert np.allclose(scored["estimated_self_created_tsa100"], [13.2, 16.2])
    assert np.allclose(scored["shot_volume_identity_gap"], 0.0)
    assert scored.loc[0, "estimate_status"] == "pre_label_backcast_research_only"
    assert scored["qualified_role_estimate"].eq(1).all()


def test_chronological_predictions_never_train_on_test_or_future_seasons() -> None:
    panel = pd.DataFrame(
        {
            "season": np.repeat(np.arange(2014, 2019), 2),
            "player_id": np.tile([1, 2], 5),
            "label_off_poss": 1000.0,
            "feature": np.arange(10, dtype=float),
            "target": np.arange(10, dtype=float) / 10.0,
        }
    )
    predictions = module.chronological_predictions(
        panel,
        "target",
        ["feature"],
        alpha=1.0,
        lower=0.0,
        upper=1.0,
    )

    assert predictions["test_season"].tolist() == [2017, 2017, 2018, 2018]
    assert (predictions["train_last"] < predictions["test_season"]).all()
    assert predictions.loc[predictions["test_season"].eq(2017), "train_last"].eq(2016).all()


def test_weighted_metrics_are_exact_for_perfect_prediction() -> None:
    metrics = module.weighted_metrics([1.0, 2.0], [1.0, 2.0], [2.0, 3.0])
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert abs(metrics["correlation"] - 1.0) < 1e-12


def test_joint_price_solve_recovers_three_target_two_lane_prices() -> None:
    rng = np.random.default_rng(4)
    rows = 300
    v_self = rng.uniform(4.0, 25.0, rows)
    v_ast = rng.uniform(3.0, 20.0, rows)
    v_potast = 0.4 * v_self + rng.uniform(0.0, 20.0, rows)
    sc_100 = rng.uniform(0.5, 4.0, rows)
    bp_100 = rng.uniform(0.2, 3.0, rows)
    true_prices = np.array([0.05, 0.08, 0.11])
    sc_excess = sc_100 - true_prices[0] * v_self - true_prices[1] * v_ast
    bp_excess = bp_100 - true_prices[2] * v_potast
    frame = pd.DataFrame(
        {
            "poss": rng.uniform(2000.0, 12000.0, rows),
            "sc_100": sc_100,
            "bp_100": bp_100,
            "act": sc_100 + bp_100,
            "v_self": v_self,
            "v_ast": v_ast,
            "v_potast": v_potast,
            "y_sc": 0.8 * sc_excess + rng.normal(0.0, 0.02, rows),
            "y_bp": 0.9 * bp_excess + rng.normal(0.0, 0.02, rows),
            "y_tov": 0.7 * (sc_excess + bp_excess) + rng.normal(0.0, 0.02, rows),
        }
    )
    fitted = module._joint_price_solve(frame)

    assert fitted["converged"] == 1
    assert np.allclose(
        [fitted["self"], fitted["ast"], fitted["potast"]], true_prices, atol=0.002
    )


def test_score_ledger_uses_earliest_fit_and_closes_both_lanes() -> None:
    panel = pd.DataFrame(
        {
            "nba_id": [1, 2, 1, 2],
            "name": ["A", "B", "A", "B"],
            "year": [1997, 1997, 2001, 2001],
            "OffPoss": [1000.0, 1500.0, 1200.0, 1400.0],
            "tsa100": [25.0, 30.0, 26.0, 29.0],
            "v_self": [10.0, 20.0, 12.0, 18.0],
            "v_ast": [15.0, 10.0, 14.0, 11.0],
            "v_potast": [5.0, 12.0, 7.0, 10.0],
            "n_sc": [20.0, 30.0, 25.0, 32.0],
            "n_bp": [10.0, 24.0, 12.0, 20.0],
            "sc_100": [2.0, 2.0, 25 / 12, 32 / 14],
            "bp_100": [1.0, 1.6, 1.0, 20 / 14],
            "act": [3.0, 3.6, 37 / 12, 52 / 14],
            "involve": [30.0, 42.0, 33.0, 39.0],
        }
    )
    fits = {2001: {"self": 0.05, "ast": 0.08, "potast": 0.10}}
    ledger, _ = module.score_ledger(panel, fits, "box_estimated_rztov")

    assert ledger.loc[ledger["year"].eq(1997), "be_source"].eq(2001).all()
    assert ledger.loc[ledger["year"].eq(1997), "early_window_inherited"].eq(1).all()
    assert np.allclose(ledger["excess"], ledger["sc_excess"] + ledger["bp_excess"])
    assert np.allclose(ledger["tov_saved_100"], -ledger["excess"])
    for _, group in ledger.groupby("year"):
        assert abs(np.average(group["sc_excess"], weights=group["OffPoss"])) < 1e-12
        assert abs(np.average(group["bp_excess"], weights=group["OffPoss"])) < 1e-12


def test_comparison_metrics_report_perfect_rank_stability() -> None:
    panel = pd.DataFrame(
        {
            "nba_id": [1, 2, 3], "rztov": [-2.0, 0.5, 3.0],
            "poss": [2000.0, 3000.0, 4000.0], "qualified": [1, 1, 1],
        }
    )
    metrics = module.comparison_metrics(panel, panel)
    assert metrics["rmse"] == 0.0
    assert metrics["rank_correlation"] == 1.0


def test_raw_tsa_counts_trips_not_free_throws():
    raw = _raw_rows().iloc[[0]].copy()
    rows = []
    for description in ["Free Throw 1 of 1", "Free Throw 1 of 2", "Free Throw 2 of 2",
                        "Free Throw 1 of 3", "Free Throw 2 of 3", "Free Throw 3 of 3",
                        "Free Throw Technical 1 of 1"]:
        row = raw.iloc[0].copy()
        row["event_type"] = 3
        row["home_description"] = description
        rows.append(row)
    events = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)
    possessions = pd.DataFrame({f"O{i}": [i] for i in range(1, 6)})
    result, _ = module.build_season_features_from_frames(events, possessions, 2020)
    assert result.set_index("player_id").loc[1, "pbp_tsa"] == 3


def test_canonical_tsa_excludes_playoffs_and_one_shot_foul():
    raw = pd.DataFrame({
        'game_id': ['0022500001']*7 + ['0042500001'],
        'event_num': list(range(8)),
        'event_type': [1,3,3,3,3,3,3,1],
        'player1_id': [42]*8,
        'home_description': ['Made shot','Free Throw 1 of 1','Free Throw 1 of 2',
                             'Free Throw 2 of 2','Free Throw 1 of 3',
                             'Free Throw 2 of 3','Free Throw 3 of 3','Playoff shot'],
    })
    counts=module.canonical_tsa_counts(raw)
    assert counts.loc[42,'pbp_tsa']==3
    assert counts.loc[42,'fga']==1
    assert counts.loc[42,'fta']==6


def test_integer_game_ids_preserve_historical_regular_season():
    rows=pd.DataFrame({'game_id':[29600001,49600001]})
    assert module.regular_season_rows(rows).game_id.tolist()==[29600001]

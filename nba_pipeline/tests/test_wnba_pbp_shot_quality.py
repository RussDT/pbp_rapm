import pandas as pd

from wnba_test.build_wnba_pbp_shot_quality import (
    IMPACT_TARGET_ZONES,
    MODEL_FEATURE_COLUMNS,
    ShotQualityConfig,
    build_classic_shot_rows,
    build_shot_rows,
    fit_context_model,
    predict_context_ev,
    select_impact_target_shots,
    season_forward_splits,
)


def _raw_shots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["1022400001", "1022400001"],
            "event_num": [10, 20],
            "period": [1, 1],
            "seconds_remaining_quarter": [500.0, 450.0],
            "personid": [1, 2],
            "playername": ["Shooter One", "Shooter Two"],
            "isfieldgoal": [1, 1],
            "shotresult": ["Made", "Missed"],
            "shotvalue": [3, 2],
            "shotdistance": [25.2, 2.0],
            "xlegacy": [230.0, 10.0],
            "ylegacy": [80.0, 5.0],
            "actiontype": ["Made Shot", "Missed Shot"],
            "subtype": ["Jump Shot", "Layup"],
            "description": [
                "S. One 25' 3PT step back (3 PTS) (P. Guard 1 AST)",
                "MISS S. Two driving Layup",
            ],
        }
    )


def _classic_raw_shots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["1021500001"] * 4,
            "EVENTNUM": [10, 20, 30, 40],
            "EVENTMSGTYPE": [1, 2, 2, 1],
            "EVENTMSGACTIONTYPE": [79, 1, 5, 3],
            "PERIOD": [1, 1, 1, 1],
            "PCTIMESTRING": ["8:30", "4:05", "1:15", "0:45"],
            "HOMEDESCRIPTION": [
                "Player One 17' Pullup Jump Shot (2 PTS)",
                None,
                "MISS Player Three 2' Layup",
                "Player Four 3' Hook Shot (2 PTS)",
            ],
            "VISITORDESCRIPTION": [
                None,
                "MISS Player Two 25' 3PT Jump Shot",
                None,
                None,
            ],
            "PLAYER1_ID": [1, 2, 3, 4],
            "PLAYER1_NAME": [
                "Player One",
                "Player Two",
                "Player Three",
                "Player Four",
            ],
        }
    )


def test_build_shot_rows_uses_only_pre_outcome_model_features():
    shots = build_shot_rows(_raw_shots(), 2024, "RS")

    assert shots["actual_points"].tolist() == [3.0, 0.0]
    assert shots["shot_zone"].tolist() == ["three", "rim"]
    assert shots["impact_target_eligible"].tolist() == [True, False]
    assert shots["desc_step_back"].tolist() == [True, False]
    assert shots["desc_driving"].tolist() == [False, True]
    assert {
        "shotresult",
        "actiontype",
        "description",
        "is_make",
        "actual_points",
        "made_assisted_tag",
    }.isdisjoint(MODEL_FEATURE_COLUMNS)


def test_impact_target_scope_is_midrange_and_three_only():
    shots = build_shot_rows(_raw_shots(), 2024, "RS")
    eligible = select_impact_target_shots(shots)

    assert IMPACT_TARGET_ZONES == {
        "short_mid",
        "long_mid",
        "deep_two",
        "three",
    }
    assert eligible["shot_zone"].tolist() == ["three"]
    assert not eligible["shot_zone"].eq("rim").any()


def test_classic_adapter_recovers_common_midrange_three_features():
    shots = build_classic_shot_rows(_classic_raw_shots(), 2015, "RS")

    assert shots["shot_zone"].tolist() == [
        "long_mid",
        "three",
        "rim",
        "short_mid",
    ]
    assert shots["impact_target_eligible"].tolist() == [True, True, False, True]
    assert shots["shot_distance"].tolist() == [17.0, 25.0, 2.0, 3.0]
    assert shots["seconds_remaining_quarter"].tolist() == [
        510.0,
        245.0,
        75.0,
        45.0,
    ]
    assert not shots["geometry_available"].any()
    assert {
        "xlegacy",
        "ylegacy",
        "location_side",
        "corner_three",
    }.isdisjoint(MODEL_FEATURE_COLUMNS)


def test_context_prediction_does_not_use_validation_outcomes():
    train = build_shot_rows(_raw_shots(), 2024, "RS")
    test_made = build_shot_rows(_raw_shots().iloc[[0]], 2025, "RS")
    raw_missed = _raw_shots().iloc[[0]].copy()
    raw_missed["shotresult"] = "Missed"
    raw_missed["description"] = "MISS S. One 25' 3PT step back"
    test_missed = build_shot_rows(raw_missed, 2025, "RS")
    model = fit_context_model(
        train,
        ShotQualityConfig(zone_shrink=10.0, context_shrink=10.0),
    )

    made_prediction = predict_context_ev(model, test_made)
    missed_prediction = predict_context_ev(model, test_missed)

    assert made_prediction.tolist() == missed_prediction.tolist()


def test_season_forward_splits_never_train_on_validation_year():
    splits = season_forward_splits([2022, 2023, 2024, 2025, 2026])

    assert splits == [
        ([2022], 2023),
        ([2022, 2023], 2024),
        ([2022, 2023, 2024], 2025),
        ([2022, 2023, 2024, 2025], 2026),
    ]
    assert all(max(train_years) < validation_year for train_years, validation_year in splits)

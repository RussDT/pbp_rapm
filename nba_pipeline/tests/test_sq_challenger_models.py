import numpy as np
import pandas as pd
import pytest

from nba_pipeline.scripts.shotquality_modeling import (
    normalize_game_id,
    parse_descriptor_tags,
)
from nba_pipeline.scripts.train_sq_shot_value_challenger import (
    build_feature_matrix,
    segment_rows,
    validate_feature_names,
)
from nba_pipeline.scripts.train_sq_transition_classifier import (
    build_historical_feature_matrix,
    prepare_sequence_frame,
)


def test_normalize_game_id_and_descriptor_parser_are_stable():
    ids = normalize_game_id(pd.Series([22400001, "0022400002", 22400003.0]))
    assert ids.tolist() == ["0022400001", "0022400002", "0022400003"]
    assert parse_descriptor_tags("['transition', 'lay_up']") == {
        "transition",
        "lay_up",
    }
    assert parse_descriptor_tags("__import__('os').system('false')") == set()


def test_shot_value_feature_contract_excludes_outcome_and_ev_outputs():
    frame = pd.DataFrame(
        {
            "made": [True, False],
            "action_type": ["2pt", "3pt"],
            "SHOT_ZONE_BASIC": ["Mid-Range", "Above the Break 3"],
            "play_descriptors": [
                "['pull_up', 'jumpshot']",
                "['catch_and_shoot', 'jumpshot']",
            ],
            "ev.initial_ev.value": [0.82, 1.05],
            "contest_score": [0.4, 0.7],
            "shot_x": [20.0, 25.0],
            "shot_y": [10.0, 8.0],
        }
    )
    assert segment_rows(frame, "midrange").tolist() == [True, False]
    assert segment_rows(frame, "three").tolist() == [False, True]

    feature_only = build_feature_matrix(
        frame, include_initial_ev=False, shot_value=2
    )
    anchored = build_feature_matrix(
        frame, include_initial_ev=True, shot_value=2
    )
    assert "baseline_initial_make_prob" not in feature_only
    assert anchored["baseline_initial_make_prob"].tolist() == [0.41, 0.525]
    assert not any("made" in column.lower() for column in anchored)
    assert not any("dynamic_ev" in column.lower() for column in anchored)
    with pytest.raises(ValueError):
        validate_feature_names(["ev.dynamic_ev.value"], include_initial_ev=False)


def test_transition_sequence_is_clock_first_not_event_number_first():
    raw = pd.DataFrame(
        {
            "game_id": ["0022400001"] * 3,
            "event_num": [5, 99, 7],
            "event_type": [5, 4, 2],
            "event_action_type": [0, 0, 1],
            "period": [1, 1, 1],
            "minute_remaining_quarter": [10, 9, 9],
            "seconds_remaining_quarter": [0, 55, 50],
            "home_description": [
                "Turnover",
                "Rebound",
                "MISS Player 15' Jump Shot",
            ],
            "neutral_description": [None, None, None],
            "visitor_description": [None, None, None],
        }
    )
    ordered = prepare_sequence_frame(raw)
    shot = ordered.loc[ordered["event_num"].eq(7)].iloc[0]
    assert shot["prev1_event_type"] == 4
    assert shot["elapsed_since_prev1"] == 5
    assert shot["last_change_event_type"] == 4
    assert shot["seconds_since_last_change"] == 5


def test_clock_normalization_supports_total_and_component_second_schemas():
    common = {
        "event_type": [12, 2, 12, 2],
        "event_action_type": [0, 1, 0, 1],
        "period": [1, 1, 1, 1],
        "home_description": [
            None,
            "MISS Old 15' Jump Shot",
            None,
            "MISS Modern 15' Jump Shot",
        ],
        "neutral_description": [None] * 4,
        "visitor_description": [None] * 4,
    }
    raw = pd.DataFrame(
        {
            **common,
            "game_id": ["old", "old", "modern", "modern"],
            "event_num": [1, 2, 1, 2],
            "minute_remaining_quarter": [12, 11, 12, 11],
            "seconds_remaining_quarter": [720, 699, 0, 39],
            "_raw_source": ["NBA97.parquet"] * 2 + ["NBA25.parquet"] * 2,
        }
    )
    prepared = prepare_sequence_frame(raw)
    shot_clock = prepared.loc[
        prepared["event_type"].eq(2), ["game_id", "clock_sec"]
    ].set_index("game_id")["clock_sec"]
    assert shot_clock["old"] == 699
    assert shot_clock["modern"] == 699


def test_clock_normalization_prefers_canonical_time_quarter():
    raw = pd.DataFrame(
        {
            "game_id": ["old"],
            "event_num": [1],
            "event_type": [2],
            "event_action_type": [1],
            "period": [1],
            "time_quarter": ["11:39"],
            "minute_remaining_quarter": [99],
            "seconds_remaining_quarter": [999],
            "home_description": ["MISS Player 15' Jump Shot"],
            "neutral_description": [None],
            "visitor_description": [None],
        }
    )
    assert prepare_sequence_frame(raw).iloc[0]["clock_sec"] == 699


def test_transition_features_use_historical_pregame_fields_only():
    raw = pd.DataFrame(
        {
            "game_id": ["0022400001"] * 2,
            "event_num": [4, 7],
            "event_type": [5, 2],
            "event_action_type": [0, 1],
            "period": [1, 1],
            "minute_remaining_quarter": [11, 11],
            "seconds_remaining_quarter": [50, 44],
            "home_description": [
                "Player Turnover (P1.T1)",
                "MISS Player 26' 3PT Pullup Jump Shot",
            ],
            "neutral_description": [None, None],
            "visitor_description": [None, None],
        }
    )
    prepared = prepare_sequence_frame(raw)
    features = build_historical_feature_matrix(prepared)
    assert "shot_distance_ft" in features
    assert "shot_action::3pt" in features
    assert "prev1_event_type::5" in features
    assert all("transition" not in column.lower() for column in features)
    assert all("made" not in column.lower() for column in features)
    assert np.isfinite(features.fillna(0).to_numpy()).all()

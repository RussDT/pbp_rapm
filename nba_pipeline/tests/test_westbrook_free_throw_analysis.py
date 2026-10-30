from __future__ import annotations

import pandas as pd

from nba_pipeline.scripts.analyze_westbrook_free_throws import (
    _assign_trips,
    _build_pre_event_score,
    normalize_game_id,
    parse_clock_seconds,
    parse_ft_description,
)


def test_ft_description_categories_and_outcomes() -> None:
    made = parse_ft_description("Westbrook Free Throw 1 of 2 (12 PTS)")
    assert (made.made, made.declared_num, made.declared_total, made.explicit_kind) == (
        1,
        1,
        2,
        "ordinary",
    )

    missed_technical = parse_ft_description("MISS Westbrook Free Throw Technical")
    assert missed_technical.made == 0
    assert missed_technical.declared_num is None
    assert missed_technical.explicit_kind == "technical"

    clear_path = parse_ft_description("MISS Westbrook Free Throw Clear Path 2 of 2")
    assert (clear_path.declared_num, clear_path.declared_total, clear_path.explicit_kind) == (
        2,
        2,
        "clear_path",
    )

    malformed_miss = parse_ft_description("Free Throw 2 of 2")
    assert malformed_miss.made == 0
    assert malformed_miss.ambiguous_no_outcome_marker


def test_clock_parser_handles_historical_total_seconds_and_modern_components() -> None:
    assert parse_clock_seconds("07:18", 7, 438) == 438
    assert parse_clock_seconds("11:17", 11, 17) == 677
    assert parse_clock_seconds(None, 7, 438) == 438
    assert parse_clock_seconds(None, 11, 17) == 677


def test_sequence_proven_one_of_one_is_repaired_to_first_of_two() -> None:
    rows = pd.DataFrame(
        [
            {
                "game_id": "0021700736",
                "period": 4,
                "clock_sec": 455.0,
                "explicit_kind": "ordinary",
                "declared_num": 1,
                "declared_total": 1,
            },
            {
                "game_id": "0021700736",
                "period": 4,
                "clock_sec": 455.0,
                "explicit_kind": "ordinary",
                "declared_num": 2,
                "declared_total": 2,
            },
        ]
    )
    result, audit = _assign_trips(rows)
    assert result["trip_id"].nunique() == 1
    assert result["trip_total"].tolist() == [2, 2]
    assert result["trip_complete"].all()
    assert result["label_anomaly"].all()
    assert audit["sequence_repaired_first_rows"] == 1


def test_technical_is_not_folded_into_same_clock_ordinary_trip() -> None:
    rows = pd.DataFrame(
        [
            {
                "game_id": "0021600001",
                "period": 2,
                "clock_sec": 120.0,
                "explicit_kind": "technical",
                "declared_num": None,
                "declared_total": None,
            },
            {
                "game_id": "0021600001",
                "period": 2,
                "clock_sec": 120.0,
                "explicit_kind": "ordinary",
                "declared_num": 1,
                "declared_total": 2,
            },
            {
                "game_id": "0021600001",
                "period": 2,
                "clock_sec": 120.0,
                "explicit_kind": "ordinary",
                "declared_num": 2,
                "declared_total": 2,
            },
        ]
    )
    result, _ = _assign_trips(rows)
    assert result["trip_id"].nunique() == 2
    assert result["trip_total"].tolist() == [1, 2, 2]


def test_post_scoring_zero_zero_placeholder_is_not_a_real_score_reset() -> None:
    rows = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 4,
            "away_score": [0, 2, 0, 3],
            "home_score": [0, 0, 0, 0],
        }
    )
    result = _build_pre_event_score(rows)
    assert result["pre_away_score"].tolist() == [0, 0, 2, 2]


def test_game_id_normalization() -> None:
    assert normalize_game_id(217000736) == "0217000736"
    assert normalize_game_id("0021700736") == "0021700736"

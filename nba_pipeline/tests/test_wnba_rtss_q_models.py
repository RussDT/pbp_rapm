import json

import pandas as pd
import pytest

from wnba_test.train_wnba_rtss_q_models import (
    CONFIRMATION_SEASON,
    DEVELOPMENT_SEASONS,
    add_prior_game_shooter_history,
    extract_game,
    q_make_probability,
)


def test_q_score_is_expected_efg_not_three_point_make_probability():
    assert q_make_probability(50.0, 2.0) == pytest.approx(0.50)
    assert q_make_probability(50.0, 3.0) == pytest.approx(1.0 / 3.0)


def test_extractor_uses_release_time_allowlist_and_fences_2026(tmp_path):
    payload = {
        "shots": [
            {
                "id": "shot",
                "gameId": "game",
                "season": 2025,
                "shooterId": "player",
                "outcome": True,
                "three": True,
                "qSQ": 51.0,
                "qSP": 54.0,
                "assisted": True,
                "assistOpp": True,
                "blocked": False,
                "fouled": True,
                "rimDepth": 2.0,
                "complexOutcome": "made",
            }
        ]
    }
    path = tmp_path / "basketball-markings.json"
    path.write_text(json.dumps(payload))
    row = extract_game(str(path))[0]
    assert row["qSQ_prob"] == pytest.approx(0.34)
    assert row["qSP_prob"] == pytest.approx(0.36)
    for forbidden in (
        "assisted",
        "assistOpp",
        "blocked",
        "fouled",
        "rimDepth",
        "complexOutcome",
    ):
        assert forbidden not in row

    payload["shots"][0]["season"] = CONFIRMATION_SEASON
    path.write_text(json.dumps(payload))
    assert extract_game(str(path)) == []
    assert CONFIRMATION_SEASON not in DEVELOPMENT_SEASONS


def test_shooter_history_excludes_every_shot_in_current_game():
    frame = pd.DataFrame(
        [
            {
                "shooter_id": "p",
                "game_id": "g1",
                "game_epoch": 1.0,
                "is_make": 1,
                "qSQ_prob": 0.5,
            },
            {
                "shooter_id": "p",
                "game_id": "g1",
                "game_epoch": 1.0,
                "is_make": 0,
                "qSQ_prob": 0.5,
            },
            {
                "shooter_id": "p",
                "game_id": "g2",
                "game_epoch": 2.0,
                "is_make": 1,
                "qSQ_prob": 0.25,
            },
        ]
    )
    result = add_prior_game_shooter_history(frame, shrink_attempts=2.0)
    first = result[result["game_id"] == "g1"]
    second = result[result["game_id"] == "g2"]
    assert first["prior_attempts"].eq(0).all()
    assert first["shooter_prior_residual"].eq(0.0).all()
    assert second["prior_attempts"].eq(2).all()
    # Game one residual is (1 - .5 - .5) = 0, so game two still receives zero.
    assert second["shooter_prior_residual"].eq(0.0).all()

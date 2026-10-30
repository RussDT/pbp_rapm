from pathlib import Path

import numpy as np
import pandas as pd

from nba_pipeline.scripts.calibrate_shot_clock_estimates import (
    apply_calibrator,
    fit_and_benchmark_calibrator,
    join_player_shot_log_labels,
    parse_clock_seconds,
)


def test_parse_clock_seconds_accepts_integer_and_decimal_seconds():
    result = parse_clock_seconds(pd.Series(["10:58", "00:03.20", None]))
    assert result.iloc[0] == 658
    assert result.iloc[1] == 3.2
    assert np.isnan(result.iloc[2])


def test_shot_ordinal_join_rejects_inconsistent_outcome(tmp_path: Path):
    rows = []
    for event_num, clock, event_type, description in [
        (10, "10:00", 1, "Home Guard 15' Jump Shot (2 PTS)"),
        (20, "08:00", 2, "MISS Home Guard 25' 3PT Jump Shot"),
    ]:
        minute, second = [int(value) for value in clock.split(":")]
        row = {
            "game_id": "0021400001",
            "event_num": event_num,
            "event_type": event_type,
            "event_action_type": 1,
            "period": 1,
            "time_quarter": clock,
            "seconds_remaining_quarter": second,
            "home_description": description,
            "visitor_description": "",
            "player1_id": 100,
            "player1_name": "Home Guard",
            "player1_team_id": 1,
            "player1_team_abbreviation": "HOM",
        }
        for idx in range(1, 6):
            row[f"home_player{idx}"] = idx
            row[f"away_player{idx}"] = idx + 10
        rows.append(row)
    raw_path = tmp_path / "NBA15.parquet"
    pd.DataFrame(rows).to_parquet(raw_path, index=False)

    labels = pd.DataFrame(
        {
            "GAME_ID": [21400001, 21400001],
            "PERIOD": [1, 1],
            "GAME_CLOCK": ["9:58", "7:58"],
            "SHOT_CLOCK": [12.4, 8.1],
            "SHOT_NUMBER": [1, 2],
            "PTS_TYPE": [2, 3],
            # Second row intentionally disagrees with the raw miss.
            "SHOT_RESULT": ["made", "made"],
            "player_id": [100, 100],
        }
    )
    label_path = tmp_path / "shot_logs.csv"
    labels.to_csv(label_path, index=False)

    mapped, audit = join_player_shot_log_labels(raw_path, label_path)
    assert len(mapped) == 1
    assert mapped.iloc[0]["event_num"] == 10
    assert audit["ordinal_join_candidates"] == 2
    assert audit["valid_joined_labels"] == 1


def test_game_heldout_residual_model_improves_systematic_reset_bias():
    rng = np.random.default_rng(7)
    rows = []
    reason_correction = {
        "made_fg": 4.0,
        "def_rebound": 1.2,
        "turnover": 2.0,
        "off_rebound": -1.4,
    }
    for game_idx in range(30):
        for shot_idx in range(32):
            reason = list(reason_correction)[shot_idx % 4]
            reset_len = 24
            raw_estimate = float(2 + shot_idx % 19)
            actual = np.clip(
                raw_estimate + reason_correction[reason] + rng.normal(0, 0.35),
                0,
                reset_len,
            )
            rows.append(
                {
                    "game_id": f"00214{game_idx:05d}",
                    "period": 1 + shot_idx % 4,
                    "event_num": shot_idx + 1,
                    "clock_sec": 700 - shot_idx * 18,
                    "shot_clock_est": raw_estimate,
                    "shot_clock_raw": raw_estimate,
                    "elapsed_since_reset": reset_len - raw_estimate,
                    "reset_len": reset_len,
                    "rule_reset_len": reset_len,
                    "reset_reason": reason,
                    "reset_time_quarter": "10:00",
                    "confidence": "high",
                    "needs_review": False,
                    "shot_clock_actual": actual,
                }
            )
    package, heldout, benchmark, _ = fit_and_benchmark_calibrator(pd.DataFrame(rows))
    raw_mae = benchmark.loc[benchmark["model"].eq("raw_rule_estimate"), "mae"].iloc[0]
    calibrated = benchmark.loc[
        benchmark["model"].eq("game_heldout_residual_calibrator")
    ].iloc[0]
    assert calibrated["mae"] < raw_mae * 0.5
    assert 0.75 <= calibrated["interval_90_coverage"] <= 1.0
    assert set(heldout["game_id"]).isdisjoint(
        {f"00214{idx:05d}" for idx in range(24)}
    )
    rescored = apply_calibrator(pd.DataFrame(rows).head(20), package)
    assert "shot_clock_est" in rescored
    assert rescored["shot_clock_calibrated"].between(0, rescored["reset_len"]).all()
    assert {
        "shot_clock_calibrated_lower",
        "shot_clock_calibrated_upper",
    }.issubset(rescored.columns)

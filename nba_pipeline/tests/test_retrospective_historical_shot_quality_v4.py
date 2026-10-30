from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.train_retrospective_historical_shot_quality_v4 import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    fit_game_group_oof,
    fit_temporal_forward_predictions,
    prepare_model_frame,
)


def _synthetic_rows() -> pd.DataFrame:
    rows = []
    for game_number in range(6):
        for shot_number in range(20):
            bucket = "midrange" if shot_number % 2 == 0 else "three"
            made = int((game_number + shot_number) % 3 == 0)
            rows.append(
                {
                    "game_id": f"g{game_number}",
                    "season_end_year": 2000 + game_number // 3,
                    "season_type": "RS",
                    "model_bucket": bucket,
                    "is_make": made,
                    "shooter_id": 100 + shot_number % 5,
                    "shot_distance_ft": 16.0 if bucket == "midrange" else 25.0,
                    "shot_zone": "long_mid" if bucket == "midrange" else "three",
                    "shot_family": "jumper",
                    "modifier_bundle": "none",
                    "event_action_type_str": "1",
                    "shot_clock_est": 12.0,
                    "shot_clock_fraction": 0.5,
                    "elapsed_since_reset": 12.0,
                    "effective_reset_len": 24.0,
                    "shot_clock_reset_reason": "def_rebound",
                    "shot_clock_confidence": "high",
                    "shooter_pregame_make_rate": 0.38,
                    "shooter_pregame_log_attempts": 4.0,
                    "historical_sq_direct_make_prob": 0.38,
                    "historical_sq_distilled_make_prob": 0.39,
                    "historical_sq_component_gap": 0.01,
                    "historical_sq_make_prob": 0.39,
                    "period": 2,
                    "shot_quality_with_defense_make_prob": 0.38,
                }
            )
    return pd.DataFrame(rows)


def test_exactly_one_canonical_loo_anchor_is_a_model_feature():
    assert "shot_quality_with_defense_make_prob" in NUMERIC_FEATURES
    assert "shot_context_make_prob" not in NUMERIC_FEATURES
    assert "shot_quality_prior_make_prob" not in NUMERIC_FEATURES
    assert "shot_quality_make_prob" not in NUMERIC_FEATURES
    assert "shot_quality_season_make_prob" not in NUMERIC_FEATURES
def test_random_game_group_oof_cannot_reconstruct_current_outcome():
    rng = np.random.default_rng(7)
    rows = pd.concat([_synthetic_rows()] * 6, ignore_index=True)
    rows["game_id"] = [f"g{index // 20}" for index in range(len(rows))]
    rows["is_make"] = rng.binomial(1, 0.4, size=len(rows))
    rows["shooter_id"] = rng.integers(1, 40, size=len(rows))
    rows["shot_quality_with_defense_make_prob"] = 0.4
    featured = prepare_model_frame(rows, rare_minimum=1)
    probability, _ = fit_game_group_oof(featured, folds=3, random_seed=17)
    brier = np.mean((featured["is_make"].to_numpy() - probability) ** 2)
    # Algebraically leaked LOO stacks score near zero on this fixture. A
    # current-game-excluded stack should remain near the Bernoulli baseline.
    assert brier > 0.15
    assert set(NUMERIC_FEATURES).issubset(featured.columns)
    assert set(CATEGORICAL_FEATURES).issubset(featured.columns)


def test_temporal_predictions_never_train_on_validation_or_future_seasons():
    rows = pd.concat([_synthetic_rows()] * 6, ignore_index=True)
    rows["season_end_year"] = np.repeat(
        [1997, 1998, 2004, 2014, 2024, 2026],
        len(rows) // 6,
    )
    rows["game_id"] = [
        f"{season}_{index}"
        for index, season in enumerate(rows["season_end_year"].astype(int))
    ]
    featured = prepare_model_frame(rows, rare_minimum=1)
    probability, audit = fit_temporal_forward_predictions(featured, random_seed=5)
    assert np.isnan(probability[featured["season_end_year"].eq(1997)]).all()
    assert np.isfinite(probability[featured["season_end_year"].gt(1997)]).all()
    assert all(record["train_through"] < record["validation_start"] for record in audit)
    assert all(record["train_validation_game_overlap"] == 0 for record in audit)

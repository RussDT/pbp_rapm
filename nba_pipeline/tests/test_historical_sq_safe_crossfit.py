from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from nba_pipeline.scripts.train_historical_sq_safe_crossfit import (
    ENCODING_FEATURES,
    add_fold_target_encodings,
    direct_encode_fold,
    fit_safe_game_crossfit,
)


def _encoding_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = pd.DataFrame(
        {
            "_game_key": ["g1", "g1", "g2", "g2", "g3", "g4"],
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g4"],
            "season_end_year": [1997] * 6,
            "season_type": ["rs"] * 6,
            "model_bucket": ["midrange"] * 6,
            "shot_zone": ["long_mid"] * 6,
            "shot_family": ["jump"] * 6,
            "event_action_type_str": ["0"] * 6,
            "modifier_bundle": ["plain"] * 6,
            "distance_bin": ["9"] * 6,
            "shooter_id": [1, 1, 1, 1, 2, 3],
            "is_make": [1, 0, 1, 1, 0, 0],
            "D1": [10] * 6,
            "D2": [11] * 6,
            "D3": [12] * 6,
            "D4": [13] * 6,
            "D5": [14] * 6,
        }
    )
    validation = fit.iloc[:2].copy()
    validation["_game_key"] = "heldout"
    validation["game_id"] = "heldout"
    validation["is_make"] = [0, 1]
    return fit, validation


def test_direct_fit_encoding_excludes_whole_current_game() -> None:
    fit, validation = _encoding_rows()
    train_probability, support, _, _ = direct_encode_fold(
        fit,
        validation,
        keys=["season_end_year", "season_type", "model_bucket"],
        shrink=10.0,
    )
    # The two shots in g1 see the exact same map, despite opposite own outcomes.
    assert train_probability[0] == train_probability[1]
    assert support[0] == support[1] == 4

    flipped = fit.copy()
    flipped.loc[flipped["_game_key"].eq("g1"), "is_make"] = [0, 1]
    flipped_probability, _, _, _ = direct_encode_fold(
        flipped,
        validation,
        keys=["season_end_year", "season_type", "model_bucket"],
        shrink=10.0,
    )
    np.testing.assert_allclose(train_probability[:2], flipped_probability[:2])


def test_validation_encodings_ignore_validation_outcomes() -> None:
    fit, validation = _encoding_rows()
    _, encoded = add_fold_target_encodings(fit, validation)
    randomized = validation.copy()
    randomized["is_make"] = [1, 1]
    _, randomized_encoded = add_fold_target_encodings(fit, randomized)
    np.testing.assert_allclose(
        encoded[ENCODING_FEATURES].to_numpy(dtype=float),
        randomized_encoded[ENCODING_FEATURES].to_numpy(dtype=float),
    )


def _synthetic_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for bucket_index, bucket in enumerate(["midrange", "three"]):
        for game in range(24):
            for shot in range(12):
                favorable = (shot % 4) < 2
                made = int(favorable if (game + shot) % 5 else not favorable)
                rows.append(
                    {
                        "_game_key": f"{bucket}-g{game}",
                        "game_id": f"{bucket}-g{game}",
                        "season_end_year": 1997 + game % 2,
                        "season_type": "rs",
                        "model_bucket": bucket,
                        "shot_zone": "open_like" if favorable else "hard_like",
                        "shot_family": "jump",
                        "modifier_bundle": "plain",
                        "event_action_type_str": "0",
                        "shot_clock_reset_reason": "possession_start",
                        "shot_clock_confidence": "high",
                        "shooter_id": 100 + shot % 6,
                        "is_make": made,
                        "D1": 200 + game % 10,
                        "D2": 220 + game % 10,
                        "D3": 240 + game % 10,
                        "D4": 260 + game % 10,
                        "D5": 280 + game % 10,
                        "historical_sq_make_prob": 0.72 if favorable else 0.18,
                        "historical_sq_direct_make_prob": 0.70 if favorable else 0.20,
                        "historical_sq_distilled_make_prob": 0.74 if favorable else 0.16,
                        "historical_sq_component_gap": 0.04,
                        "shot_distance_ft": 16.0 + bucket_index * 8,
                        "shot_clock_est": 12.0,
                        "shot_clock_fraction": 0.5,
                        "elapsed_since_reset": 12.0,
                        "effective_reset_len": 24.0,
                        "period": 1 + game % 4,
                        "distance_bin": "8" if bucket == "midrange" else "12",
                    }
                )
    return pd.DataFrame(rows)


def test_safe_crossfit_does_not_collapse_to_random() -> None:
    rows = _synthetic_rows()
    prediction, encodings, audit = fit_safe_game_crossfit(
        rows,
        folds=3,
        min_samples_leaf=15,
    )
    assert all(record["game_overlap"] == 0 for record in audit)
    assert prediction.std() > 0.08
    assert not encodings.isna().any().any()
    y = rows["is_make"].to_numpy()
    fixed = rows["model_bucket"].map({"midrange": 0.41, "three": 0.36}).to_numpy()
    assert brier_score_loss(y, prediction) < brier_score_loss(y, fixed) - 0.05


def test_random_labels_cannot_be_reconstructed_by_encodings() -> None:
    rows = _synthetic_rows()
    rng = np.random.default_rng(20260723)
    rows["is_make"] = rng.binomial(1, 0.5, size=len(rows))
    # Remove all intentional raw/upstream signal. Any near-perfect result would
    # therefore expose own-shot or own-game target reconstruction.
    rows["shot_zone"] = "same"
    rows["historical_sq_make_prob"] = 0.5
    rows["historical_sq_direct_make_prob"] = 0.5
    rows["historical_sq_distilled_make_prob"] = 0.5
    rows["historical_sq_component_gap"] = 0.0
    rows["shot_distance_ft"] = 20.0
    prediction, _, _ = fit_safe_game_crossfit(
        rows,
        folds=3,
        min_samples_leaf=25,
    )
    assert brier_score_loss(rows["is_make"], prediction) > 0.20

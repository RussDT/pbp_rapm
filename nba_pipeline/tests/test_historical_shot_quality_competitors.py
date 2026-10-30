import numpy as np
import pandas as pd

from nba_pipeline.scripts.audit_historical_shot_quality_competitors import (
    add_empirical_probabilities,
    competitor_comparisons,
    paired_game_cluster_interval,
)


def test_cluster_interval_is_positive_for_better_candidate():
    y = np.array([1, 0, 1, 0, 1, 0])
    candidate = np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2])
    baseline = np.repeat(0.5, len(y))
    games = np.array(["g1", "g1", "g2", "g2", "g3", "g3"])
    interval = paired_game_cluster_interval(
        y, candidate, baseline, games, repetitions=200
    )
    assert interval["brier_improvement_ci_low"] > 0
    assert interval["log_loss_improvement_ci_low"] > 0


def test_empirical_probability_excludes_current_shot():
    rows = pd.DataFrame(
        {
            "era": ["1997-2003"] * 3,
            "model_bucket": ["three"] * 3,
            "shot_zone": ["three"] * 3,
            "is_make": [1, 0, 0],
        }
    )
    scored = add_empirical_probabilities(rows, prior_strength=0)
    assert scored["era_bucket_empirical_probability"].tolist() == [0.0, 0.5, 0.5]


def test_comparison_flags_losing_era():
    rows = pd.DataFrame(
        {
            "era": ["1997-2003"] * 6,
            "model_bucket": ["three"] * 6,
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "is_make": [1, 0, 1, 0, 1, 0],
            "historical_sq_make_prob": [0.5] * 6,
            "context_probability": [0.8, 0.2, 0.8, 0.2, 0.8, 0.2],
            "season_talent_probability": [0.5] * 6,
            "defense_adjusted_probability": [0.5] * 6,
            "era_bucket_empirical_probability": [0.5] * 6,
            "era_zone_empirical_probability": [0.5] * 6,
        }
    )
    comparison = competitor_comparisons(rows, bootstrap_repetitions=100)
    context = comparison[comparison["competitor"].eq("repo_shot_context")].iloc[0]
    assert bool(context["v3_loses_brier"])
    assert bool(context["v3_loses_log_loss"])

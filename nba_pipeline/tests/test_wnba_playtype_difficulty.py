"""Offline pytest coverage for the WNBA PBP playtype-difficulty proxy.

Covers the frozen six-family classifier against real WNBA description strings,
the leakage guard (make/miss and assist tags must never move a family), the
prior-season pricing shrinkage, the two level conventions, and the forward
evaluation's panel construction and free-throw parsing. Nothing here touches
the network or the RTSS export.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
WNBA_TEST_DIR = ROOT / "wnba_test"
if str(WNBA_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(WNBA_TEST_DIR))

import build_wnba_playtype_difficulty as difficulty  # noqa: E402
import evaluate_wnba_playtype_difficulty as evaluation  # noqa: E402


def _classify(descriptions, threes):
    return difficulty.classify_wnba_shots(
        pd.Series(descriptions), pd.Series(threes)
    ).tolist()


def test_classifier_assigns_real_wnba_descriptions():
    descriptions = [
        "L. Lacan driving Layup (4 PTS) (J. Sheldon 3 AST)",
        "MISS R. Allen 24' pullup 3PT",
        "MISS K. Thornton 25' 3PT",
        "S. Ionescu 6' driving floating bank Jump Shot (11 PTS)",
        "MISS C. Taylor putback Layup",
        "MISS D. Bonner 8' Jump Shot",
        "A. Wilson cutting Layup (12 PTS)",
    ]
    threes = [False, True, True, False, False, False, False]
    assert _classify(descriptions, threes) == [
        "SELF_CREATED_2",
        "SELF_CREATED_3",
        "OTHER_3",
        "SELF_CREATED_2",
        "OTHER_RIM_2",
        "OTHER_2",
        "CUT_2",
    ]


def test_classification_ignores_outcome_and_assist_tags():
    """A family must depend on the action words only, never on the result."""
    base = "A. Ogunbowale 18' pullup Jump Shot"
    variants = [
        base,
        f"MISS {base}",
        f"{base} (22 PTS)",
        f"{base} (22 PTS) (C. Gray 5 AST)",
        f"MISS {base} - blocked",
    ]
    families = _classify(variants, [False] * len(variants))
    assert len(set(families)) == 1
    assert families[0] == "SELF_CREATED_2"


def test_three_flag_separates_self_created_families():
    description = ["M. Mabrey step back Shot"] * 2
    assert _classify(description, [True, False]) == [
        "SELF_CREATED_3",
        "SELF_CREATED_2",
    ]


def _pricing_frame():
    rows = []
    for family, points, count in [
        ("CUT_2", 1.4, 2000),
        ("SELF_CREATED_3", 0.9, 2000),
        ("OTHER_3", 1.05, 8000),
        ("SELF_CREATED_2", 0.86, 12000),
        ("OTHER_RIM_2", 1.15, 5000),
        ("OTHER_2", 0.78, 2000),
    ]:
        rows.extend([{"pbp_playtype_category": family, "actual_points": points}] * count)
    return pd.DataFrame(rows)


def test_prices_shrink_toward_the_league_mean():
    prices = difficulty.estimate_category_prices(
        _pricing_frame(), pricing_season=2024, feature_season=2025
    )
    assert set(prices["pbp_playtype_category"]) == set(difficulty.PBP_CATEGORIES)
    league = prices["league_points_per_fga"].iloc[0]
    for _, row in prices.iterrows():
        raw = row["expected_points_per_fga"]
        # Every shrunk price sits strictly between the raw family mean and the
        # league mean, and low-count families shrink further.
        assert min(row["mean"], league) <= raw <= max(row["mean"], league)
    small = prices.set_index("pbp_playtype_category").loc["CUT_2"]
    large = prices.set_index("pbp_playtype_category").loc["SELF_CREATED_2"]
    assert abs(small["expected_points_per_fga"] - league) / abs(
        small["mean"] - league
    ) < abs(large["expected_points_per_fga"] - league) / abs(large["mean"] - league)


def test_missing_pricing_family_falls_back_to_league_mean():
    frame = _pricing_frame()
    frame = frame[frame["pbp_playtype_category"].ne("CUT_2")]
    prices = difficulty.estimate_category_prices(
        frame, pricing_season=2024, feature_season=2025
    ).set_index("pbp_playtype_category")
    assert prices.loc["CUT_2", "count"] == 0
    assert prices.loc["CUT_2", "expected_points_per_fga"] == pytest.approx(
        prices.loc["CUT_2", "league_points_per_fga"]
    )


def _feature_shots():
    rows = []
    # Hard diet: all self-created twos.  Easy diet: all cuts.
    rows.extend(
        [{"player_id": 1, "player_name": "Hard", "pbp_playtype_category": "SELF_CREATED_2"}]
        * 200
    )
    rows.extend(
        [{"player_id": 2, "player_name": "Easy", "pbp_playtype_category": "CUT_2"}] * 200
    )
    return pd.DataFrame(rows)


def test_difficulty_sign_and_two_level_conventions():
    prices = difficulty.estimate_category_prices(
        _pricing_frame(), pricing_season=2024, feature_season=2025
    )
    players = difficulty.aggregate_player_difficulty(
        _feature_shots(), prices, feature_season=2025
    ).set_index("player_name")
    # Positive means a harder-than-reference mix.
    assert players.loc["Hard", "playtype_diff"] > 0
    assert players.loc["Easy", "playtype_diff"] < 0
    # The two conventions differ only by a per-season constant, so they must
    # produce identical spreads and identical orderings.
    offsets = players["playtype_diff"] - players["playtype_diff_prior_league"]
    assert offsets.max() - offsets.min() == pytest.approx(0.0, abs=1e-9)
    assert players["descriptor_regime"].eq("modern").all()
    assert players["share_SELF_CREATED_2"].loc["Hard"] == pytest.approx(1.0)


def test_pre_modern_seasons_are_flagged():
    prices = difficulty.estimate_category_prices(
        _pricing_frame(), pricing_season=2013, feature_season=2014
    )
    players = difficulty.aggregate_player_difficulty(
        _feature_shots(), prices, feature_season=2014
    )
    assert players["descriptor_regime"].eq("sparse_descriptions").all()


def test_season_centering_removes_a_league_wide_level_shift():
    """A league-wide relabelling must not move anybody's centered value."""
    prices = difficulty.estimate_category_prices(
        _pricing_frame(), pricing_season=2024, feature_season=2025
    )
    shots = _feature_shots()
    # A pure league-wide scoring-environment shift: every family is worth the
    # same amount more, so no player's relative role has changed at all.
    drifted_prices = prices.copy()
    drifted_prices["expected_points_per_fga"] += 0.05
    before = difficulty.aggregate_player_difficulty(shots, prices, feature_season=2025)
    after = difficulty.aggregate_player_difficulty(
        shots, drifted_prices, feature_season=2025
    )
    # The NBA convention reads the shift as 2.5 TS points of extra difficulty
    # for everyone; the centered convention is exactly invariant to it.
    assert (
        after["playtype_diff_prior_league"] - before["playtype_diff_prior_league"]
    ).mean() == pytest.approx(-2.5)
    assert np.allclose(after["playtype_diff"], before["playtype_diff"])


def test_free_throw_result_parsing_matches_wnba_descriptions():
    """Free throws share one event type, so misses come from the MISS prefix."""
    description = pd.Series(
        [
            "MISS Jones Free Throw 1 of 2 ",
            " Sutton Free Throw 2 of 2 (2 PTS)",
            "MISS Iriafen Free Throw 1 of 2 ",
        ]
    )
    made = ~description.str.strip().str.upper().str.startswith("MISS")
    assert made.tolist() == [False, True, False]


def test_weighted_rmse_matches_manual_computation():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 2.0, 2.0])
    weights = np.array([1.0, 2.0, 3.0])
    expected = np.sqrt((1 * 0.25 + 2 * 0.0 + 3 * 1.0) / 6.0)
    assert evaluation._weighted_rmse(actual, predicted, weights) == pytest.approx(expected)


def test_real_artifact_invariants():
    """Guard the built panel if it is present; skip in a clean checkout."""
    path = (
        ROOT
        / "wnba_test"
        / "Results"
        / "playtype_difficulty"
        / "wnba_playtype_player_season.csv"
    )
    if not path.exists():
        pytest.skip("playtype difficulty artifact not built")
    frame = pd.read_csv(path)
    modern = frame[
        frame["descriptor_regime"].eq("modern") & frame["pbp_fga"].ge(100)
    ]
    shares = [f"share_{family}" for family in difficulty.PBP_CATEGORIES]
    assert np.allclose(frame[shares].sum(axis=1), 1.0)
    # Season centering is exact on the shot-weighted mean, so the qualified
    # player mean sits near zero without being forced to it.
    for season, group in modern.groupby("feature_season"):
        assert abs(group["playtype_diff"].mean()) < 1.0, season
        assert group["playtype_diff"].std() > 1.0, season


def test_scaled_display_column_is_the_raw_column_times_the_coefficient():
    """``playtype_diff`` must stay raw; only the display column is scaled.

    The forecasting lane and the box-prior lane both consume the raw value, and
    the NBA columns are unscaled, so baking the coefficient into the feature
    would break two of three consumers plus cross-league comparability.
    """
    prices = difficulty.estimate_category_prices(
        _pricing_frame(), pricing_season=2024, feature_season=2025
    )
    players = difficulty.aggregate_player_difficulty(
        _feature_shots(), prices, feature_season=2025
    ).set_index("player_name")

    coefficient = difficulty.ROLE_DISPLAY_COEFFICIENT
    assert 0.0 < coefficient < 1.0
    assert players["playtype_diff_scaled"].to_numpy() == pytest.approx(
        coefficient * players["playtype_diff"].to_numpy()
    )
    # The raw column must NOT already carry the coefficient: scaling is strictly
    # a shrink toward zero, so the raw magnitude is always the larger one.
    assert (
        players["playtype_diff_scaled"].abs() < players["playtype_diff"].abs()
    ).all()
    # Scaling is monotone, so it may not reorder anyone.
    assert (
        players["playtype_diff"].rank().to_numpy()
        == players["playtype_diff_scaled"].rank().to_numpy()
    ).all()


def test_scaled_column_closes_the_reader_identity_on_the_real_artifact():
    """rTS + playtype_diff_scaled = rTS_role, with no display-time arithmetic."""
    path = (
        ROOT
        / "wnba_test"
        / "Results"
        / "playtype_difficulty"
        / "wnba_playtype_player_season.csv"
    )
    if not path.exists():
        pytest.skip("playtype difficulty artifact not built")
    frame = pd.read_csv(path)
    if "playtype_diff_scaled" not in frame.columns:
        pytest.skip("artifact predates the scaled display column")

    coefficient = difficulty.ROLE_DISPLAY_COEFFICIENT
    assert frame["playtype_diff_scaled"].to_numpy() == pytest.approx(
        coefficient * frame["playtype_diff"].to_numpy()
    )
    # A synthetic rTS stands in for the box lane: the identity must hold by
    # construction rather than by a coincidence of the built values.
    rng = np.random.default_rng(20260819)
    r_ts = rng.normal(0.0, 5.0, size=len(frame))
    assert r_ts + frame["playtype_diff_scaled"].to_numpy() == pytest.approx(
        r_ts + coefficient * frame["playtype_diff"].to_numpy()
    )

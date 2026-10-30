import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nba_pipeline.scripts import rapm
from nba_pipeline.scripts.rubberband_model import (
    build_rubberband_basis,
    compute_pre_possession_clock_state,
)


def test_pre_possession_clock_uses_previous_possession_end_and_period_start():
    frame = pd.DataFrame(
        {
            "game_id": ["1", "1", "1", "1", "1", "2"],
            "period": [1, 1, 1, 2, 5, 1],
            "time_quarter": ["11:31", "11:10", "00:00", "11:42", "04:40", "11:35"],
        }
    )
    state = compute_pre_possession_clock_state(frame)
    assert state["pre_period_clock_seconds"].tolist() == [720, 691, 670, 720, 300, 720]
    assert state["seconds_remaining_game"].tolist() == [2880, 2851, 2830, 2160, 300, 2880]
    assert state["is_overtime"].tolist() == [0, 0, 0, 0, 1, 0]


def test_sparse_historical_score_snapshots_are_carried_forward_within_game():
    frame = pd.DataFrame(
        {
            "game_id": ["1", "1", "1", "1", "2", "2"],
            "away_score": [0, 2, 0, 2, 0, 0],
            "home_score": [0, 0, 0, 2, 0, 3],
        }
    )
    away, home, observed = rapm.reconstruct_cumulative_game_scores(frame)

    np.testing.assert_allclose(away, [0, 2, 2, 2, 0, 0])
    np.testing.assert_allclose(home, [0, 0, 0, 2, 0, 3])
    assert observed.tolist() == [False, True, False, True, False, True]


def test_home_offense_inference_ignores_ambiguous_score_jumps_and_propagates_rosters():
    game_ids = np.array(["1"] * 5)
    away_score = np.array([2, 2, 4, 4, 6], dtype=float)
    home_score = np.array([0, 0, 2, 4, 4], dtype=float)
    away_a = [1, 2, 3, 4, 5]
    away_b = [1, 2, 3, 4, 6]
    home_a = [11, 12, 13, 14, 15]
    home_b = [11, 12, 13, 14, 16]
    offense = np.array([away_a, home_a, away_a, home_b, away_b])
    defense = np.array([home_a, away_a, home_a, away_b, home_b])

    inferred, resolved = rapm.infer_home_offense_from_score_state(
        game_ids,
        offense,
        defense,
        away_score,
        home_score,
    )

    # Row three has an ambiguous 2-2 scoreboard jump; lineup continuity still
    # identifies the offense. Substitute players 6 and 16 are propagated too.
    assert inferred.tolist() == [False, True, False, True, False]
    assert resolved.tolist() == [True] * 5


def test_clock_aware_bases_are_exactly_zero_at_tied_score():
    frame = pd.DataFrame(
        {
            "score_diff": [0.0, 0.0],
            "period": [2, 5],
            "seconds_remaining_game": [1800.0, 120.0],
            "regulation_progress": [0.375, 1.0],
            "is_overtime": [0, 1],
            "overtime_progress": [0.0, 0.6],
        }
    )
    for model in ("pressure", "smooth"):
        basis = build_rubberband_basis(frame, model)
        assert basis.matrix.shape[0] == 2
        assert np.max(np.abs(basis.matrix.toarray())) == 0.0


def test_smooth_basis_is_continuous_across_old_bucket_boundary():
    frame = pd.DataFrame(
        {
            "score_diff": [9.999999, 10.000001],
            "period": [4, 4],
            "seconds_remaining_game": [300.0, 300.0],
            "regulation_progress": [1 - 300 / 2880, 1 - 300 / 2880],
            "is_overtime": [0, 0],
            "overtime_progress": [0.0, 0.0],
        }
    )
    basis = build_rubberband_basis(frame, "smooth").matrix.toarray()
    assert np.max(np.abs(basis[1] - basis[0])) < 1e-5


def test_legacy_basis_preserves_24_feature_contract():
    frame = pd.DataFrame({"score_diff": [-20, -8, 0, 8, 20], "period": [1, 2, 3, 4, 5]})
    basis = build_rubberband_basis(frame, "legacy")
    assert basis.matrix.shape == (5, 24)
    assert len(basis.feature_names) == 24
    assert basis.matrix.getnnz(axis=1).tolist() == [1, 1, 0, 1, 1]


def test_joint_generalized_solver_matches_closed_form_ridge():
    dense = np.array(
        [
            [1.0, 0.0, 0.2],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.8],
            [1.0, 1.0, 1.0],
        ]
    )
    target = np.array([0.1, 0.4, -0.2, 0.3])
    mapping = {"1_off": 0, "2_def": 1, "rb_smooth_test": 2}
    penalties = np.diag([2.0, 3.0, 5.0])
    expected = np.linalg.solve(dense.T @ dense + penalties, dense.T @ target)
    actual = rapm.joint_generalized_rapm(
        csr_matrix(dense),
        target,
        mapping,
        ["1", "2"],
        alpha_offense=2.0,
        alpha_defense=3.0,
        rubberband=True,
        alpha_rb=5.0,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_joint_generalized_solver_preserves_weights_priors_and_unpenalized_nuisance():
    dense = np.array(
        [
            [1.0, 0.0, 0.2, 0.0],
            [1.0, 1.0, 0.5, 0.0],
            [0.0, 1.0, 0.8, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.4, 1.0],
            [0.0, 1.0, 0.7, 1.0],
        ]
    )
    target = np.array([0.1, 0.4, -0.2, 0.3, 0.25, -0.1])
    weights = np.array([1.0, 0.8, 1.2, 0.7, 1.3, 0.9])
    mapping = {
        "1_off": 0,
        "2_def": 1,
        "rb_smooth_test": 2,
        "season_fe_2026_rs": 3,
    }
    penalty = np.diag([2.0, 3.0, 5.0, 0.0])
    prior = np.array([0.3, -0.2, 0.0, 0.0])
    weight_matrix = np.diag(weights)
    expected = np.linalg.solve(
        dense.T @ weight_matrix @ dense + penalty,
        dense.T @ weight_matrix @ target + penalty @ prior,
    )
    actual = rapm.joint_generalized_rapm(
        csr_matrix(dense),
        target,
        mapping,
        ["1", "2"],
        alpha_offense=2.0,
        alpha_defense=3.0,
        off_prior={"1": 0.3},
        def_prior={"2": -0.2},
        weights=weights,
        rubberband=True,
        alpha_rb=5.0,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_output_suffix_marks_legacy_only():
    assert rapm.build_output_suffix("all", pure=True, rubberband=True) == "all_pure_rb"
    assert (
        rapm.build_output_suffix(
            "all",
            pure=True,
            rubberband=True,
            rubberband_model="legacy",
        )
        == "all_pure_rb_legacy"
    )

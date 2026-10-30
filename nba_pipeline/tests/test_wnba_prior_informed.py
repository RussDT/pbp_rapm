import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from wnba_test.build_prior_informed_wnba import (
    _age_state,
    _reconcile_display_priors,
)
from wnba_test.clean_decomp_wnba import (
    DISPLAY_PREFIXES,
    _solve_ridge_with_prior,
)


def test_prior_centered_ridge_matches_closed_form_diagonal_case():
    design = csr_matrix(np.eye(2))
    target = np.array([2.0, -1.0])
    weights = np.ones(2)
    penalties = np.array([3.0, 4.0])
    prior = np.array([1.0, -2.0])

    actual = _solve_ridge_with_prior(
        design,
        target,
        weights,
        penalties,
        prior,
    )
    expected = (target + penalties * prior) / (1.0 + penalties)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_annual_state_decay_respects_player_specific_gap():
    state = pd.DataFrame(
        {
            "player_id": ["1", "2"],
            "player_name": ["A", "B"],
            "off": [4.0, 4.0],
            "def": [-2.0, -2.0],
            "rating_year": [2025, 2024],
        }
    )

    aged = _age_state(state, 2026, persistence=0.5)

    np.testing.assert_allclose(aged["off"], [2.0, 1.0])
    np.testing.assert_allclose(aged["def"], [-1.0, -0.5])


def test_completion_prior_reconciliation_restores_display_identity():
    player_ids = ["1", "2"]
    priors = {}
    for prefix in DISPLAY_PREFIXES:
        priors[prefix] = pd.DataFrame(
            {
                "player_id": player_ids,
                "player_name": ["A", "B"],
                "off": [0.1, -0.2],
                "def": [-0.05, 0.1],
            }
        )
    total = pd.DataFrame(
        {
            "player_id": player_ids,
            "player_name": ["A", "B"],
            "off": [2.0, -1.0],
            "def": [-1.5, 0.5],
        }
    )

    reconciled = _reconcile_display_priors(priors, total)
    additive_prefixes = [
        "ALT_EFG_RIM_FREQ",
        "ALT_EFG_RIM_FG",
        "ALT_EFG_MID_FREQ",
        "ALT_EFG_MID_FG",
        "ALT_EFG_THREE_FREQ",
        "ALT_EFG_THREE_FG",
        "ALT_FT",
        "ALT_FC_COMPLETION",
        "SECOND_CHANCE_CLEAN",
    ]
    for side in ("off", "def"):
        component_sum = sum(
            reconciled[prefix].set_index("player_id")[side]
            for prefix in additive_prefixes
        )
        expected = total.set_index("player_id")[side]
        np.testing.assert_allclose(component_sum.sort_index(), expected.sort_index())

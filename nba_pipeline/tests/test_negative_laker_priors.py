import pandas as pd

from nba_pipeline.scripts.build_negative_laker_priors import build_negative_laker_priors
from nba_pipeline.scripts.evaluate_negative_laker_stabilization import evaluate_grid
from nba_pipeline.scripts.rapm import build_output_suffix


def test_negative_laker_prior_uses_observed_net_for_eligibility_and_age27_components():
    curve = pd.DataFrame(
        {
            "age": [20, 27, 38],
            "orapm_age_effect": [-2.0, 0.0, -3.0],
            "drapm_age_effect": [-1.0, 0.0, -1.0],
        }
    )
    rows = pd.DataFrame(
        [
            {
                "nba_id": 1,
                "player_id": "oldfr01",
                "player_name": "Old Fringe",
                "year": 2025,
                "age": 38,
                "team_abbreviation": "AAA",
                "playoffs": 0,
                "minutes": 300,
                "orapm": -5.0,
                "drapm": 0.5,
                "rapm": -4.5,
            },
            {
                "nba_id": 1,
                "player_id": "oldfr01",
                "player_name": "Old Fringe",
                "year": 2025,
                "age": 38,
                "team_abbreviation": "BBB",
                "playoffs": 0,
                "minutes": 200,
                "orapm": -4.0,
                "drapm": 0.0,
                "rapm": -4.0,
            },
            {
                "nba_id": 2,
                "player_id": "veter01",
                "player_name": "Veteran",
                "year": 2025,
                "age": 27,
                "team_abbreviation": "CCC",
                "playoffs": 0,
                "minutes": 1100,
                "orapm": -1.0,
                "drapm": 0.0,
                "rapm": -1.0,
            },
            {
                "nba_id": 3,
                "player_id": "posit01",
                "player_name": "Positive",
                "year": 2025,
                "age": 27,
                "team_abbreviation": "DDD",
                "playoffs": 0,
                "minutes": 100,
                "orapm": 1.0,
                "drapm": 0.0,
                "rapm": 1.0,
            },
        ]
    )

    off, defense, audit, receipt = build_negative_laker_priors(
        rows,
        curve,
        start_year=1997,
        end_year=2026,
        minute_threshold=1000,
        prior_scale=0.5,
        stabilization_minutes=0,
    )

    old = audit.loc[audit["player_id"] == "1"].iloc[0]
    assert old["career_minutes"] == 500
    assert old["career_rapm"] == -4.3
    assert bool(old["eligible"])
    # Observed O=-4.6 and D=+0.3. Removing age-38 effects yields O=-1.6,
    # D=+1.3; scale 0.5 produces -0.8/+0.65 age-27 priors.
    assert old["off_prior_age27"] == -0.8
    assert old["def_prior_age27_positive_good"] == 0.65
    assert old["def_prior_solver_raw"] == -0.65
    assert off.to_dict("records") == [{"player_id": "1", "prior_value": -0.8}]
    assert defense.to_dict("records") == [{"player_id": "1", "prior_value": -0.65}]
    assert receipt["eligible_players"] == 1


def test_prior_label_makes_grid_outputs_distinct():
    suffix = build_output_suffix(
        "all",
        pure=True,
        off_prior={"1": -0.01},
        def_prior={"1": 0.01},
        prior_label="neglaker_m1000_k500_s1",
    )
    assert suffix == "all_pure_offdefprior_neglaker_m1000_k500_s1"


def test_forward_grid_selects_reliability_stabilization():
    transitions = pd.DataFrame(
        {
            "history_minutes": [100.0],
            "history_off": [-4.0],
            "history_def": [-2.0],
            "history_net": [-6.0],
            "target_off": [-2.0],
            "target_def": [-1.0],
            "target_net": [-3.0],
            "target_minutes": [500.0],
        }
    )

    grid = evaluate_grid(transitions, [0.0, 100.0, 500.0])
    selected = grid.loc[grid["off_def_mse_sum"].idxmin()]
    assert selected["stabilization_minutes"] == 100.0
    assert selected["off_def_mse_sum"] == 0.0

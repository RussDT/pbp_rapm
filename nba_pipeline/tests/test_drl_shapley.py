from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "nba_pipeline" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_drl_shapley as mod  # noqa: E402


def make_bundle(
    *,
    action_ids: list[int] | None = None,
    game_ids: list[int] | None = None,
    game_date_ord: list[int] | None = None,
    season_end_year: list[int] | None = None,
    rewards: list[float] | None = None,
    gamma: list[float] | None = None,
    next_index: list[int] | None = None,
    terminal: list[bool] | None = None,
    pre_margin: list[float] | None = None,
    final_margin: list[float] | None = None,
    player_ids: np.ndarray | None = None,
    side_ids: np.ndarray | None = None,
    team_home: list[int] | None = None,
    team_away: list[int] | None = None,
) -> mod.ArrayBundle:
    n_rows = len(action_ids or game_ids or game_date_ord or [0])
    numeric = np.zeros((n_rows, len(mod.NUMERIC_FEATURES)), dtype=np.float32)
    numeric[:, mod.NUMERIC_FEATURES.index("home_possession_flag")] = 1.0
    numeric[:, mod.NUMERIC_FEATURES.index("seconds_remaining_game")] = 120.0

    if player_ids is None:
        player_ids = np.tile(np.arange(1, 11, dtype=np.int64), (n_rows, 1))
    if side_ids is None:
        side_ids = np.tile(np.array([1] * 5 + [0] * 5, dtype=np.int64), (n_rows, 1))
    pos_ids = np.zeros((n_rows, 10), dtype=np.int64)

    output_meta = pd.DataFrame(
        {
            "global_index": np.arange(n_rows, dtype=np.int64),
            "game_id": np.array(game_ids or [1] * n_rows, dtype=np.int64),
            "game_date": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n_rows), unit="D"),
            "event_num": np.arange(1, n_rows + 1, dtype=np.int64),
            "season_end_year": np.array(season_end_year or [2026] * n_rows, dtype=np.int64),
            "action_class": [mod.ACTION_CLASSES[action_ids[i]] for i in range(n_rows)] if action_ids else ["make_2"] * n_rows,
            "reward_home": np.array(rewards or [0.0] * n_rows, dtype=np.float32),
            "gamma": np.array(gamma or [1.0] * n_rows, dtype=np.float32),
            "pre_margin_home": np.array(pre_margin or [0.0] * n_rows, dtype=np.float32),
            "terminal_final_margin_home": np.array(final_margin or [0.0] * n_rows, dtype=np.float32),
            "primary_player_id": np.full(n_rows, 1, dtype=np.int64),
            "secondary_player_id": np.full(n_rows, 2, dtype=np.int64),
            "tertiary_player_id": np.full(n_rows, 3, dtype=np.int64),
            "team_id_home": np.array(team_home or [100] * n_rows, dtype=np.int64),
            "team_id_away": np.array(team_away or [200] * n_rows, dtype=np.int64),
            "score_bucket": ["within_5"] * n_rows,
            "time_bucket": ["0_3_min"] * n_rows,
        }
    )
    for idx, col in enumerate(mod.LINEUP_COLS):
        output_meta[col] = player_ids[:, idx]
    for idx, col in enumerate(mod.NUMERIC_FEATURES):
        output_meta[col] = numeric[:, idx]

    player_name_map = {pid: f"P{pid}" for pid in range(1, 11)}
    player_team_map = {pid: (100 if pid <= 5 else 200) for pid in range(1, 11)}
    player_pos_group = {pid: "PG" for pid in range(1, 11)}

    return mod.ArrayBundle(
        numeric=numeric,
        numeric_scaled=numeric.copy(),
        player_ids=player_ids.astype(np.int64),
        side_ids=side_ids.astype(np.int64),
        pos_ids=pos_ids,
        rewards=np.array(rewards or [0.0] * n_rows, dtype=np.float32),
        gamma=np.array(gamma or [1.0] * n_rows, dtype=np.float32),
        next_index=np.array(next_index or [-1] * n_rows, dtype=np.int64),
        terminal=np.array(terminal or [True] * n_rows, dtype=bool),
        pre_margin_home=np.array(pre_margin or [0.0] * n_rows, dtype=np.float32),
        terminal_final_margin_home=np.array(final_margin or [0.0] * n_rows, dtype=np.float32),
        terminal_remaining_margin=np.array(final_margin or [0.0] * n_rows, dtype=np.float32),
        home_win=np.zeros(n_rows, dtype=np.int64),
        action_id=np.array(action_ids or [mod.ACTION_TO_ID["make_2"]] * n_rows, dtype=np.int64),
        primary_player_id=np.full(n_rows, 1, dtype=np.int64),
        secondary_player_id=np.full(n_rows, 2, dtype=np.int64),
        tertiary_player_id=np.full(n_rows, 3, dtype=np.int64),
        season_end_year=np.array(season_end_year or [2026] * n_rows, dtype=np.int64),
        game_date_ord=np.array(game_date_ord or [738900 + i for i in range(n_rows)], dtype=np.int32),
        game_id=np.array(game_ids or [1] * n_rows, dtype=np.int64),
        event_num=np.arange(1, n_rows + 1, dtype=np.int64),
        team_id_home=np.array(team_home or [100] * n_rows, dtype=np.int64),
        team_id_away=np.array(team_away or [200] * n_rows, dtype=np.int64),
        output_meta=output_meta,
        player_vocab=np.arange(1, 11, dtype=np.int64),
        player_name_map=player_name_map,
        player_team_map=player_team_map,
        player_pos_group=player_pos_group,
    )


def test_build_transition_dataframe_keeps_next_index_inside_game(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        [
            {
                "game_id": "0022500001",
                "event_num": 1,
                "event_type": 1,
                "event_action_type": 0,
                "period": 1,
                "time_quarter": "12:00",
                "seconds_remaining_quarter": 720,
                "home_description": "Home Shot (2 PTS)",
                "visitor_description": None,
                "neutral_description": None,
                "home_score": 2,
                "away_score": 0,
                "score_margin": 2,
                "player1_id": 1001,
                "player1_name": "Home One",
                "player1_team_id": 10,
                "player2_id": pd.NA,
                "player2_name": None,
                "player2_team_id": pd.NA,
                "player3_id": pd.NA,
                "player3_name": None,
                "player3_team_id": pd.NA,
                **{f"home_player{i}": 1000 + i for i in range(1, 6)},
                **{f"away_player{i}": 2000 + i for i in range(1, 6)},
            },
            {
                "game_id": "0022500001",
                "event_num": 2,
                "event_type": 1,
                "event_action_type": 0,
                "period": 1,
                "time_quarter": "11:50",
                "seconds_remaining_quarter": 710,
                "home_description": None,
                "visitor_description": "Away Shot (2 PTS)",
                "neutral_description": None,
                "home_score": 2,
                "away_score": 2,
                "score_margin": 0,
                "player1_id": 2001,
                "player1_name": "Away One",
                "player1_team_id": 20,
                "player2_id": pd.NA,
                "player2_name": None,
                "player2_team_id": pd.NA,
                "player3_id": pd.NA,
                "player3_name": None,
                "player3_team_id": pd.NA,
                **{f"home_player{i}": 1000 + i for i in range(1, 6)},
                **{f"away_player{i}": 2000 + i for i in range(1, 6)},
            },
            {
                "game_id": "0022500002",
                "event_num": 1,
                "event_type": 1,
                "event_action_type": 0,
                "period": 1,
                "time_quarter": "12:00",
                "seconds_remaining_quarter": 720,
                "home_description": "Home Shot (2 PTS)",
                "visitor_description": None,
                "neutral_description": None,
                "home_score": 2,
                "away_score": 0,
                "score_margin": 2,
                "player1_id": 3001,
                "player1_name": "Home Two",
                "player1_team_id": 30,
                "player2_id": pd.NA,
                "player2_name": None,
                "player2_team_id": pd.NA,
                "player3_id": pd.NA,
                "player3_name": None,
                "player3_team_id": pd.NA,
                **{f"home_player{i}": 3000 + i for i in range(1, 6)},
                **{f"away_player{i}": 4000 + i for i in range(1, 6)},
            },
            {
                "game_id": "0022500002",
                "event_num": 2,
                "event_type": 13,
                "event_action_type": 0,
                "period": 1,
                "time_quarter": "11:45",
                "seconds_remaining_quarter": 705,
                "home_description": None,
                "visitor_description": None,
                "neutral_description": "End Period",
                "home_score": 2,
                "away_score": 0,
                "score_margin": 2,
                "player1_id": 0,
                "player1_name": None,
                "player1_team_id": 0,
                "player2_id": pd.NA,
                "player2_name": None,
                "player2_team_id": pd.NA,
                "player3_id": pd.NA,
                "player3_name": None,
                "player3_team_id": pd.NA,
                **{f"home_player{i}": 3000 + i for i in range(1, 6)},
                **{f"away_player{i}": 4000 + i for i in range(1, 6)},
            },
        ]
    )
    raw_path = tmp_path / "NBA26.parquet"
    raw.to_parquet(raw_path, index=False)
    monkeypatch.setattr(mod, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        mod,
        "load_game_dates",
        lambda: pd.DataFrame(
            {
                "GAME_ID": [22500001, 22500002],
                "date": pd.to_datetime(["2025-10-01", "2025-10-02"]),
                "season": [2025, 2025],
            }
        ),
    )

    df = mod.build_transition_dataframe(train_start=26, train_end=26, max_games_per_season=0, aliases={})

    assert df["next_index"].tolist() == [1, -1, 3, -1]
    assert df["terminal"].tolist() == [0, 1, 0, 1]
    valid = df["next_index"] >= 0
    next_games = df.loc[df.loc[valid, "next_index"].astype(int), "game_id"].to_numpy()
    assert np.all(df.loc[valid, "game_id"].to_numpy() == next_games)


def test_forward_chaining_split_falls_back_to_game_ids_when_dates_collapse():
    bundle = make_bundle(
        action_ids=[mod.ACTION_TO_ID["make_2"]] * 4,
        game_ids=[10, 10, 20, 20],
        game_date_ord=[738900, 738900, 738900, 738900],
        season_end_year=[2026, 2026, 2026, 2026],
    )

    train_idx, valid_idx = mod.forward_chaining_split(bundle, output_year=26, validation_frac=0.5)

    assert len(train_idx) > 0
    assert len(valid_idx) > 0
    assert set(train_idx).isdisjoint(set(valid_idx))
    assert set(bundle.game_id[valid_idx]) == {20}


def test_project_distribution_matches_terminal_and_td_expectations():
    next_probs = torch.zeros((2, len(mod.SUPPORT)), dtype=torch.float32)
    next_probs[0, 45] = 1.0  # margin +5
    next_probs[1, 40] = 1.0  # margin 0
    rewards = torch.tensor([1.0, -3.25], dtype=torch.float32)
    gamma = torch.tensor([0.5, 0.9], dtype=torch.float32)
    terminal = torch.tensor([False, True])

    target = mod.project_distribution(next_probs, rewards, gamma, mod.SUPPORT_TENSOR, terminal)
    expected = (target * mod.SUPPORT_TENSOR.unsqueeze(0)).sum(dim=1)

    assert torch.allclose(target.sum(dim=1), torch.ones(2), atol=1e-6)
    assert expected[0].item() == pytest.approx(3.5, abs=0.2)
    assert expected[1].item() == pytest.approx(-3.25, abs=0.2)


def test_sanitize_probability_tensor_normalizes_bad_rows():
    probs = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [float("nan"), 1.2, -0.2],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out = mod.sanitize_probability_tensor(probs)

    assert torch.all(out >= 0.0)
    assert torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-6)
    assert out[2].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)


def test_classify_event_uses_real_three_point_markers_not_score_totals():
    base_row = {
        "event_type": 1,
        "event_action_type": 0,
        "player1_id": 1,
        "player1_team_id": 100,
        "player2_id": pd.NA,
        "player2_team_id": pd.NA,
        "player3_id": pd.NA,
        "player3_team_id": pd.NA,
        "neutral_description": None,
        "is_transition": False,
        **{f"home_player{i}": i for i in range(1, 6)},
        **{f"away_player{i}": 100 + i for i in range(1, 6)},
    }

    made_two = pd.Series(
        {
            **base_row,
            "home_description": "Nurkic 13' Floating Jump Shot (3 PTS)",
            "visitor_description": None,
            "home_score": 3.0,
            "away_score": 0.0,
        }
    )
    made_three = pd.Series(
        {
            **base_row,
            "home_description": "George 26' 3PT Jump Shot (6 PTS)",
            "visitor_description": None,
            "home_score": 6.0,
            "away_score": 0.0,
        }
    )

    two_event = mod.classify_event(made_two, offense_side="home", aliases={})
    three_event = mod.classify_event(made_three, offense_side="home", aliases={})

    assert two_event["action_class"] == "make_2"
    assert three_event["action_class"] == "make_3"


def test_shapley_attributor_enforces_efficiency():
    torch.manual_seed(7)
    attributor = mod.ShapleyAttributor(num_players=20, embedding_dim=16, hidden_size=32, num_heads=4)
    numeric = torch.randn(3, len(mod.NUMERIC_FEATURES))
    player_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 3, dtype=torch.long)
    side_ids = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 3, dtype=torch.long)
    pos_ids = torch.zeros((3, 10), dtype=torch.long)
    target_value = torch.tensor([1.25, -0.5, 3.0], dtype=torch.float32)

    phi = attributor(numeric, player_ids, side_ids, pos_ids, target_value)

    assert phi.shape == (3, 10)
    assert torch.allclose(phi.sum(dim=1), target_value, atol=1e-5)


def test_allocate_action_credits_preserves_delta_v_for_offense_and_defense():
    bundle = make_bundle(
        action_ids=[mod.ACTION_TO_ID["make_2"], mod.ACTION_TO_ID["steal"]],
        game_ids=[1, 1],
        next_index=[1, -1],
        terminal=[False, True],
    )
    output_indices = np.array([0, 1], dtype=np.int64)
    phi_current = np.zeros((2, 10), dtype=np.float32)
    phi_next = np.array(
        [
            [0.6, 0.4] + [0.0] * 8,
            [0.3, -0.1, 0.2, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    delta_v_raw = np.array([2.0, -1.5], dtype=np.float32)
    delta_v_stabilized = np.array([1.0, -0.5], dtype=np.float32)

    class DummyAllocator:
        def eval(self):
            return self

        def __call__(self, numeric, candidate_player_ids, candidate_side_ids, candidate_pos_ids, action_ids, mask):
            weights = torch.zeros(candidate_player_ids.shape, dtype=torch.float32, device=numeric.device)
            weights[:, 0] = 0.75
            weights[:, 1] = 0.25
            return weights

    credits = mod.allocate_action_credits(
        bundle=bundle,
        output_indices=output_indices,
        phi_current=phi_current,
        phi_next=phi_next,
        delta_v_raw=delta_v_raw,
        delta_v_stabilized=delta_v_stabilized,
        allocator=DummyAllocator(),
        args=SimpleNamespace(batch_size=16),
        device=torch.device("cpu"),
    )

    grouped = credits.groupby("global_index", as_index=False).agg(
        total_raw=("credit_raw", "sum"),
        total_stabilized=("credit_stabilized", "sum"),
    )
    assert grouped["total_raw"].tolist() == pytest.approx(delta_v_raw.tolist())
    assert grouped["total_stabilized"].tolist() == pytest.approx(delta_v_stabilized.tolist())


def test_build_state_value_table_tracks_state_and_event_gaps():
    bundle = make_bundle(
        action_ids=[mod.ACTION_TO_ID["make_2"]],
        rewards=[2.0],
        gamma=[0.9],
        terminal=[True],
        pre_margin=[1.0],
        final_margin=[4.0],
    )
    phi_current = np.array([[0.7] * 10], dtype=np.float32)
    phi_next = np.zeros((1, 10), dtype=np.float32)

    state_values = mod.build_state_value_table(
        bundle=bundle,
        output_indices=np.array([0], dtype=np.int64),
        remaining_current=np.array([7.0], dtype=np.float32),
        remaining_next=np.array([0.0], dtype=np.float32),
        delta_v_raw=np.array([-5.0], dtype=np.float32),
        delta_v_stabilized=np.array([-2.0], dtype=np.float32),
        entropy=np.array([0.1], dtype=np.float32),
        phi_current=phi_current,
        phi_next=phi_next,
    )
    state_values = mod.attach_event_credit_sums(
        state_values,
        pd.DataFrame(
            [
                {"global_index": 0, "credit_raw": -2.0, "credit_stabilized": -1.0},
                {"global_index": 0, "credit_raw": -3.0, "credit_stabilized": -1.0},
            ]
        ),
    )

    row = state_values.iloc[0]
    assert row["phi_sum_current"] == pytest.approx(7.0)
    assert row["state_efficiency_gap"] == pytest.approx(0.0, abs=1e-6)
    assert row["next_state_efficiency_gap"] == pytest.approx(0.0, abs=1e-6)
    assert row["event_credit_raw_sum"] == pytest.approx(-5.0)
    assert row["event_credit_stabilized_sum"] == pytest.approx(-2.0)
    assert row["event_conservation_gap_raw"] == pytest.approx(0.0, abs=1e-6)
    assert row["event_conservation_gap_stabilized"] == pytest.approx(0.0, abs=1e-6)


def test_player_value_decomposition_routes_secondary_scoring_credit_to_playmaking():
    ratings = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "P1",
                "team_id": 100,
                "pos_group": "PG",
                "states_total": 10,
                "states_off": 5,
                "states_def": 5,
                "state_value_total": 0.2,
                "state_value_off": 0.1,
                "state_value_def": 0.1,
                "state_total_per100": 2.0,
                "state_off_per100": 2.0,
                "state_def_per100": 2.0,
                "shrunk_state_total_per100": 2.0,
                "shrunk_state_off_per100": 2.0,
                "shrunk_state_def_per100": 2.0,
                "total_per100": 2.0,
                "off_per100": 2.0,
                "def_per100": 2.0,
                "shrunk_total_per100": 2.0,
                "shrunk_off_per100": 2.0,
                "shrunk_def_per100": 2.0,
            },
            {
                "player_id": 2,
                "player_name": "P2",
                "team_id": 100,
                "pos_group": "PG",
                "states_total": 10,
                "states_off": 5,
                "states_def": 5,
                "state_value_total": 0.1,
                "state_value_off": 0.05,
                "state_value_def": 0.05,
                "state_total_per100": 1.0,
                "state_off_per100": 1.0,
                "state_def_per100": 1.0,
                "shrunk_state_total_per100": 1.0,
                "shrunk_state_off_per100": 1.0,
                "shrunk_state_def_per100": 1.0,
                "total_per100": 1.0,
                "off_per100": 1.0,
                "def_per100": 1.0,
                "shrunk_total_per100": 1.0,
                "shrunk_off_per100": 1.0,
                "shrunk_def_per100": 1.0,
            },
        ]
    )
    credits = pd.DataFrame(
        [
            {"global_index": 1, "player_id": 1, "action_class": "make_3", "role": "primary", "credit_raw": 1.2, "credit_stabilized": 1.2},
            {"global_index": 1, "player_id": 2, "action_class": "make_3", "role": "secondary", "credit_raw": 0.5, "credit_stabilized": 0.5},
            {"global_index": 2, "player_id": 1, "action_class": "turnover_bad_pass", "role": "primary", "credit_raw": -0.2, "credit_stabilized": -0.2},
        ]
    )

    out = mod.build_player_value_decomposition(None, ratings, credits)

    p1 = out.set_index("player_id").loc[1]
    p2 = out.set_index("player_id").loc[2]
    assert p1["scoring"] == pytest.approx(1.2)
    assert p1["turnovers"] == pytest.approx(-0.2)
    assert p1["event_credit_stabilized_total"] == pytest.approx(1.0)
    assert p1["bucket_total_stabilized"] == pytest.approx(1.0)
    assert p1["bucket_gap_stabilized"] == pytest.approx(0.0)
    assert p2["playmaking"] == pytest.approx(0.5)
    assert p2["event_credit_stabilized_total"] == pytest.approx(0.5)
    assert p2["bucket_total_stabilized"] == pytest.approx(0.5)
    assert p2["bucket_gap_stabilized"] == pytest.approx(0.0)


def test_build_reconciliation_report_passes_clean_contracts():
    state_values = pd.DataFrame(
        [
            {
                "state_efficiency_gap": 0.0,
                "next_state_efficiency_gap": 0.0,
                "event_conservation_gap_raw": 0.0,
                "event_conservation_gap_stabilized": 0.0,
            }
        ]
    )
    player_totals = pd.DataFrame([{"bucket_gap_stabilized": 0.0}])

    report = mod.build_reconciliation_report(state_values, player_totals)

    assert report["all_contracts_pass"] is True
    assert report["state_current_efficiency"]["max_abs"] == pytest.approx(0.0)
    assert report["event_conservation_raw"]["max_abs"] == pytest.approx(0.0)
    assert report["player_bucket_reconciliation"]["max_abs"] == pytest.approx(0.0)


def test_build_synergy_outputs_handles_empty_pair_table():
    bundle = make_bundle(
        action_ids=[mod.ACTION_TO_ID["make_2"]],
        game_ids=[55],
        final_margin=[5.0],
        team_home=[100],
        team_away=[200],
    )
    phi_current = np.array([[0.1] * 10], dtype=np.float32)
    ratings = pd.DataFrame([{"player_id": 1, "player_name": "P1", "team_id": 100, "pos_group": "PG", "shrunk_total_per100": 1.0}])

    pair_df, team_df = mod.build_synergy_outputs(
        bundle=bundle,
        output_indices=np.array([0], dtype=np.int64),
        phi_current=phi_current,
        ratings=ratings,
        args=SimpleNamespace(min_pair_possessions=500, permutation_tests=10, seed=7),
    )

    assert list(pair_df.columns) == [
        "team_id",
        "player_i",
        "player_j",
        "player_i_name",
        "player_j_name",
        "possessions_together",
        "possessions_apart_i",
        "possessions_apart_j",
        "synergy_total",
        "synergy_offense",
        "synergy_defense",
        "p_value",
        "fdr_significant",
    ]
    assert len(pair_df) == 0
    assert "actual_wins" in team_df.columns


def test_build_synergy_outputs_uses_true_together_and_apart_samples():
    player_ids = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 11, 3, 4, 5, 6, 7, 8, 9, 10],
            [12, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ],
        dtype=np.int64,
    )
    bundle = make_bundle(
        action_ids=[mod.ACTION_TO_ID["make_2"]] * 4,
        game_ids=[1, 1, 1, 1],
        final_margin=[5.0, 5.0, 5.0, 5.0],
        player_ids=player_ids,
        team_home=[100, 100, 100, 100],
        team_away=[200, 200, 200, 200],
    )
    phi_current = np.array(
        [
            [1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ratings = pd.DataFrame(
        [
            {"player_id": 1, "player_name": "P1", "team_id": 100, "pos_group": "PG", "shrunk_total_per100": 1.0},
            {"player_id": 2, "player_name": "P2", "team_id": 100, "pos_group": "SG", "shrunk_total_per100": 1.0},
        ]
    )

    pair_df, _ = mod.build_synergy_outputs(
        bundle=bundle,
        output_indices=np.arange(4, dtype=np.int64),
        phi_current=phi_current,
        ratings=ratings,
        args=SimpleNamespace(min_pair_possessions=1, permutation_tests=25, seed=7),
    )

    pair = pair_df[(pair_df["team_id"] == 100) & (pair_df["player_i"] == 1) & (pair_df["player_j"] == 2)].iloc[0]
    assert pair["possessions_together"] == 2
    assert pair["possessions_apart_i"] == 1
    assert pair["possessions_apart_j"] == 1
    assert pair["synergy_total"] == pytest.approx(1.1)
    assert pair["synergy_offense"] == pytest.approx(1.1)
    assert pair["synergy_defense"] == pytest.approx(0.0)
    assert 0.0 <= pair["p_value"] <= 1.0

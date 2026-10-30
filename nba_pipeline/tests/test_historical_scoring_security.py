from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_pipeline.scripts.build_historical_scoring_security import (
    VALUE_FIT_INTERCEPT,
    VALUE_FIT_SLOPE,
    PRICE_TARGET_ALPHA_TAG,
    _fit_origin_price,
    apply_role_sources,
    fit_turnover_point_prices,
    build_board,
    calibrate_role_proxy,
    resolve_historical_player_names,
    validate_observed_role_coverage,
    validate_identities,
)


def test_origin_price_recovers_known_turnover_value() -> None:
    saved = np.array([-1.0, -0.5, 0.25, 1.0, 2.0])
    weight = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    target = 0.72 * saved
    price, metrics = _fit_origin_price(saved, target, weight)
    assert np.isclose(price, 0.72)
    assert np.isclose(metrics["rmse"], 0.0)
    assert np.isclose(metrics["r2"], 1.0)


def test_lane_prices_read_low_alpha_price_targets(tmp_path) -> None:
    """Prices come from the near-unpenalized target files, never the a2000/4000 components."""
    assert PRICE_TARGET_ALPHA_TAG == "a20_40"
    ledger = pd.DataFrame(
        {
            "nba_id": np.repeat([1, 2, 3, 4], 5),
            "year": np.tile(np.arange(2022, 2027), 4),
            "OffPoss": 1000.0,
            "sc_excess": np.repeat([-1.0, -0.5, 0.5, 1.0], 5),
            "bp_excess": np.repeat([0.5, -1.0, 1.0, -0.5], 5),
        }
    )
    for lane, column, price in (("scoring", "sc_excess", 0.9), ("badpass", "bp_excess", 1.0)):
        saved = -ledger.groupby("nba_id")[column].mean()
        frame = pd.DataFrame({"player_id": saved.index, "off": price * saved.values, "off_poss": 5000})
        frame.to_csv(tmp_path / f"alt_{lane}_tov_net_value_22_26_all_rb_se_a20_40_results.csv", index=False)
        shrunk = frame.assign(off=0.5 * frame["off"])
        shrunk.to_csv(tmp_path / f"alt_{lane}_tov_net_value_22_26_all_rb_se_a2000_4000_results.csv", index=False)
    prices = fit_turnover_point_prices(ledger, tmp_path, 2026, 2026, bootstrap_draws=5, seed=1)
    assert np.isclose(prices.loc[0, "scoring_point_price"], 0.9)
    assert np.isclose(prices.loc[0, "badpass_point_price"], 1.0)


def test_observed_role_sources_override_backcast_and_close_zts() -> None:
    scoring = pd.DataFrame(
        {
            "nba_id": [1, 2, 3],
            "year": [2005, 2008, 2018],
            "rts": [1.0, 2.0, 3.0],
            "tsa100": [20.0, 20.0, 20.0],
            "playtype_diff": [np.nan, np.nan, 1.5],
        }
    )
    proxy = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "feature_season": [2005, 2008, 2018],
            "pbp_fga": [500, 500, 500],
            "pbp_difficulty_pp": [0.2, 0.3, 0.4],
            "estimated_role": [0.5, 0.6, 0.7],
            "pricing_status": ["prior_season"] * 3,
        }
    )
    anchor = pd.DataFrame(
        {
            "nba_id": [2, 3],
            "year": [2008, 2018],
            "role": [1.25, 2.25],
            "zts": [3.25, 5.25],
        }
    )
    output = apply_role_sources(scoring, proxy, anchor)
    assert output["role_difficulty"].tolist() == [0.5, 1.25, 2.25]
    assert output["role_source"].tolist() == [
        "pbp_backcast_calibrated_on_databallr_2006_2013",
        "databallr_synergy_observed",
        "databallr_synergy_observed",
    ]
    assert np.allclose(output["zts"], output["rts"] + output["role_difficulty"])
    expected_add = 2.0 * 20.0 * (output["zts"] + 4.0) / 100.0
    assert np.allclose(
        output["scoring_value100"],
        VALUE_FIT_INTERCEPT + VALUE_FIT_SLOPE * expected_add,
    )


def test_missing_observed_era_role_never_uses_historical_backcast() -> None:
    scoring = pd.DataFrame(
        {
            "nba_id": [1],
            "year": [2026],
            "rts": [3.5],
            "tsa100": [28.0],
            "playtype_diff": [np.nan],
        }
    )
    proxy = pd.DataFrame(
        {
            "player_id": [1],
            "feature_season": [2026],
            "pbp_fga": [1_000],
            "pbp_difficulty_pp": [3.3],
            "estimated_role": [-0.65],
            "pricing_status": ["prior_season"],
        }
    )
    observed = pd.DataFrame(columns=["nba_id", "year", "role", "zts"])

    output = apply_role_sources(scoring, proxy, observed)

    assert pd.isna(output.loc[0, "role_difficulty"])
    assert output.loc[0, "role_source"] == "databallr_synergy_observed_missing"


def test_refreshed_modern_role_keeps_scoring_context_on_same_snapshot() -> None:
    scoring = pd.DataFrame(
        {
            "nba_id": [1627750],
            "year": [2026],
            "rts": [3.553],
            "tsa100": [28.451],
            "playtype_diff": [np.nan],
            "scoring_context_source": ["core_merged_exact"],
        }
    )
    proxy = pd.DataFrame(
        {
            "player_id": [1627750],
            "feature_season": [2026],
            "pbp_fga": [1_360],
            "pbp_difficulty_pp": [3.324],
            "estimated_role": [-0.654],
            "pricing_status": ["prior_season"],
        }
    )
    observed = pd.DataFrame(
        {
            "nba_id": [1627750],
            "year": [2026],
            "role": [2.367],
            "zts": [6.659],
            "rts": [4.292],
            "tsa100": [27.727],
        }
    )

    output = apply_role_sources(scoring, proxy, observed)

    assert np.isclose(output.loc[0, "rts"], 4.292)
    assert np.isclose(output.loc[0, "tsa100"], 27.727)
    assert np.isclose(output.loc[0, "role_difficulty"], 2.367)
    assert np.isclose(output.loc[0, "zts"], 6.659)
    assert pd.isna(output.loc[0, "estimated_role"])
    assert output.loc[0, "scoring_context_source"] == "databallr_zts_observed"


def test_zero_observed_coverage_for_modern_season_raises() -> None:
    scoring = pd.DataFrame(
        {
            "nba_id": [1, 2],
            "year": [2026, 2026],
            "role_source": [
                "databallr_synergy_observed_missing",
                "databallr_synergy_observed_missing",
            ],
        }
    )

    with np.testing.assert_raises_regex(ValueError, "2026"):
        validate_observed_role_coverage(scoring)


def test_role_mapper_is_compositional_and_self_created_three_monotonic() -> None:
    patterns = [
        (0.04, 0.28, 0.18, 0.40, 0.04),
        (0.08, 0.18, 0.22, 0.38, 0.08),
        (0.18, 0.08, 0.25, 0.25, 0.18),
        (0.25, 0.03, 0.18, 0.18, 0.28),
    ]
    proxy_rows = []
    anchor_rows = []
    player_id = 1
    for year in range(2006, 2014):
        raw_roles = []
        season_rows = []
        for cut, self_three, other_three, self_two, other_rim in patterns:
            role = (
                -4.0 * cut
                + 3.0 * self_three
                + 1.5 * other_three
                + 2.0 * self_two
                - 5.0 * other_rim
            )
            row = {
                "player_id": player_id,
                "feature_season": year,
                "pbp_fga": 500,
                "pbp_difficulty_pp": role * 0.4,
                "share_cut_2": cut,
                "share_self_created_3": self_three,
                "share_other_3": other_three,
                "share_self_created_2": self_two,
                "share_other_rim_2": other_rim,
            }
            season_rows.append(row)
            raw_roles.append(role)
            player_id += 1
        centered_roles = np.asarray(raw_roles) - np.mean(raw_roles)
        for row, role in zip(season_rows, centered_roles):
            proxy_rows.append(row)
            anchor_rows.append(
                {
                    "nba_id": row["player_id"],
                    "year": year,
                    "role": role,
                    "zts": role,
                    "rts": 0.0,
                    "tsa100": 20.0,
                    "minutes": 1_000,
                }
            )

    for self_three in (0.10, 0.30):
        proxy_rows.append(
            {
                "player_id": player_id,
                "feature_season": 2005,
                "pbp_fga": 500,
                "pbp_difficulty_pp": 0.0,
                "share_cut_2": 0.10,
                "share_self_created_3": self_three,
                "share_other_3": 0.20,
                "share_self_created_2": 0.30,
                "share_other_rim_2": 0.10,
            }
        )
        player_id += 1

    calibrated, _, receipt = calibrate_role_proxy(
        pd.DataFrame(proxy_rows), pd.DataFrame(anchor_rows)
    )
    probe = calibrated[calibrated["feature_season"].eq(2005)].sort_values(
        "share_self_created_3"
    )

    assert receipt["selected_family"] == "share_monotonic"
    assert "pbp_difficulty_pp" not in receipt["selected_features"]
    assert receipt["standardized_role_coefficients"]["share_self_created_3"] >= 0.0
    assert receipt["standardized_role_coefficients"]["share_cut_2"] <= 0.0
    assert probe["estimated_role"].iloc[1] >= probe["estimated_role"].iloc[0]


def test_historical_player_names_resolve_by_exact_id_and_year() -> None:
    scoring = pd.DataFrame(
        {
            "nba_id": [893, 397, 893],
            "year": [1997, 1998, 2002],
            "player_name": ["Jordan", "Miller", "Michael Jordan"],
        }
    )
    lookup = pd.DataFrame(
        {
            "nba_id": [893, 397],
            "year": [1997, 1998],
            "player_name": ["Michael Jordan", "Reggie Miller"],
        }
    )

    resolved = resolve_historical_player_names(scoring, lookup)

    assert resolved["player_name"].tolist() == [
        "Michael Jordan",
        "Reggie Miller",
        "Michael Jordan",
    ]


def test_composite_board_preserves_additive_lanes() -> None:
    scoring = pd.DataFrame(
        {
            "nba_id": [1, 2],
            "year": [2001, 2020],
            "rts": [2.0, 1.0],
            "role_difficulty": [1.0, 0.5],
            "zts": [3.0, 1.5],
            "scoring_value100": [2.4, 1.0],
        }
    )
    ledger = pd.DataFrame(
        {
            "nba_id": [1, 2],
            "year": [2001, 2020],
            "OffPoss": [5000.0, 800.0],
            "sc_excess": [-0.4, -0.4],
            "bp_excess": [0.2, 0.2],
            "rztov": [-1.0, -1.0],
            "tov_saved_100": [0.2, 0.2],
            "qualified": [1, 0],
            "be_source": [2001, 2020],
            "turnover_source": ["box_estimated", "observed"],
        }
    )
    prices = pd.DataFrame(
        {
            "ledger_source": ["box_estimated", "observed"],
            "window_hi": [2001, 2020],
            "scoring_point_price": [0.7, 0.5],
            "badpass_point_price": [0.8, 0.75],
        }
    )
    board = build_board(scoring, ledger, prices)
    assert np.isclose(board.loc[0, "scoring_tov_value100"], 0.28)
    assert np.isclose(board.loc[0, "badpass_tov_value100"], -0.16)
    assert np.isclose(board.loc[0, "scoring_impact_no_badpass100"], 2.68)
    assert np.isclose(board.loc[0, "turnover_value100"], 0.12)
    assert np.isclose(board.loc[0, "scoring_security_value100"], 2.52)
    # Each ledger is priced by its own lane fit, never the other ledger's.
    assert np.isclose(board.loc[1, "scoring_tov_value100"], 0.20)
    assert np.isclose(board.loc[1, "badpass_tov_value100"], -0.15)
    assert board["qualified_scoring_security"].tolist() == [1, 0]
    gaps = validate_identities(board)
    assert max(gaps.values()) < 1e-12

    unpriced = ledger.assign(turnover_source=["box_estimated", "unknown"])
    with pytest.raises(ValueError, match="no matching lane point price"):
        build_board(scoring, unpriced, prices)

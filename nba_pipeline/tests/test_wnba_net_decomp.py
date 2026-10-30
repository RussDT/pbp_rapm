import numpy as np
import pandas as pd

from wnba_test.build_wnba_box_priors import (
    _derive_additive_priors,
    feature_columns,
)
from wnba_test.build_wnba_net_decomp import NET_ADDITIVE_PREFIXES
from wnba_test.clean_decomp_wnba import (
    _prepare_metric_df,
    validate_net_first_chance_alias_identity,
)


def _first_chance_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Is_Turnover": [0, 0, 0, 1],
            "Is_BadPass_TOV": [0, 0, 0, 1],
            "Net_Diff": [2.0, 0.0, 1.0, 0.0],
            "FC_TS_NET_Diff": [2.4, 0.3, 1.2, 0.0],
            "FC_FT_Diff": [0.0, 0.3, 0.0, 0.0],
            "FC_NET_RIM_EFG_FREQ_Diff": [0.2, 0.0, 0.0, 0.0],
            "FC_NET_RIM_EFG_FG_Diff": [0.5, 0.0, 0.0, 0.0],
            "FC_NET_MID_EFG_FREQ_Diff": [0.0, 0.0, -0.1, 0.0],
            "FC_NET_MID_EFG_FG_Diff": [0.0, 0.0, 0.2, 0.0],
            "FC_NET_THREE_EFG_FREQ_Diff": [0.0, 0.0, 0.0, 0.0],
            "FC_NET_THREE_EFG_FG_Diff": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _prior_frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["1", "2"],
            "player_name": ["A", "B"],
            "off": [value, -value],
            "def": [-value / 2.0, value / 2.0],
        }
    )


def test_net_first_chance_aliases_close_on_turnovers_and_shots():
    frame = _first_chance_rows()

    assert validate_net_first_chance_alias_identity(frame) < 1e-12

    direct_tov = _prepare_metric_df(frame, "ALT_TOV_NET_VALUE")["Off_Diff"]
    assert direct_tov.iloc[-1] == -frame.loc[:2, "FC_TS_NET_Diff"].mean()
    assert direct_tov.iloc[:3].eq(0.0).all()


def test_net_second_chance_uses_kappa_adjusted_target():
    frame = pd.DataFrame({"SC_NET_CONT_Diff": [0.4, -0.2]})
    solved = _prepare_metric_df(frame, "SECOND_CHANCE_NET")

    np.testing.assert_allclose(solved["Off_Diff"], [0.4, -0.2])
    np.testing.assert_allclose(solved["Def_Diff"], [-0.4, 0.2])


def test_net_prior_family_uses_matching_features_and_closes_total():
    assert feature_columns("ALT_EFG_NET_RIM_FG", "def") == feature_columns(
        "ALT_EFG_RIM_FG", "def"
    )
    base = {
        prefix: _prior_frame(index / 10.0)
        for index, prefix in enumerate(
            [
                "ALT_EFG_NET_RIM_FREQ",
                "ALT_EFG_NET_RIM_FG",
                "ALT_EFG_NET_MID_FREQ",
                "ALT_EFG_NET_MID_FG",
                "ALT_EFG_NET_THREE_FREQ",
                "ALT_EFG_NET_THREE_FG",
                "ALT_FT",
                "ALT_FT_FREQ",
                "ALT_FC_NET_COMPLETION",
                "ALT_TOV_NET_VALUE",
                "ALT_BADPASS_TOV_NET_VALUE",
                "SECOND_CHANCE_NET",
            ],
            start=1,
        )
    }
    priors = _derive_additive_priors(
        base,
        additive_prefixes=NET_ADDITIVE_PREFIXES,
    )

    for side in ("off", "def"):
        expected = sum(priors[prefix][side] for prefix in NET_ADDITIVE_PREFIXES)
        np.testing.assert_allclose(priors["RAPM"][side], expected)
        np.testing.assert_allclose(
            priors["ALT_TOV_NET_VALUE"][side],
            priors["ALT_BADPASS_TOV_NET_VALUE"][side]
            + priors["ALT_SCORING_TOV_NET_VALUE"][side],
        )

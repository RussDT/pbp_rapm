import pandas as pd
import pytest

from wnba_test.build_wnba_metric_history import (
    _add_reader_columns,
    _validate_history,
)


def _history_fixture():
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["High volume", "Low volume", "Second"],
            "off": [3.0, 8.0, 2.0],
            "def": [1.0, 1.0, 1.0],
            "net_rapm": [4.0, 9.0, 3.0],
            "possessions": [2000, 100, 1500],
            "off_poss": [1000, 50, 750],
            "def_poss": [1000, 50, 750],
            "oRESID": [0.0, 0.0, 0.0],
            "dRESID": [0.0, 0.0, 0.0],
            "RESID": [0.0, 0.0, 0.0],
        }
    )


def test_history_reader_columns_keep_full_and_qualified_ranks_distinct():
    history = _add_reader_columns(
        _history_fixture(),
        year=2022,
        season_type="ALL",
        model_variant="pure_warmup_no_prior",
        prior_table=None,
    )
    by_name = history.set_index("player_name")

    assert by_name.loc["Low volume", "rank"] == 1
    assert pd.isna(by_name.loc["Low volume", "qualified_rank"])
    assert by_name.loc["High volume", "qualified_rank"] == 1
    assert by_name.loc["Second", "qualified_rank"] == 2
    assert not bool(by_name.loc["Low volume", "qualified"])
    assert by_name.loc["High volume", "box_prior_source"] == "none_2022_warmup"
    assert by_name.loc["High volume", "delta_net_rapm"] == 4.0


def test_history_validation_rejects_duplicate_player_seasons():
    history = _add_reader_columns(
        _history_fixture(),
        year=2022,
        season_type="ALL",
        model_variant="pure_warmup_no_prior",
        prior_table=None,
    )
    duplicated = pd.concat([history, history.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate_player_season_keys"):
        _validate_history(duplicated)

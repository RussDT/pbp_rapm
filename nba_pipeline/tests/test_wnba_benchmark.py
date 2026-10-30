import pandas as pd

from wnba_test.benchmark_wnba_metrics import static_rating_map


def test_static_rating_map_can_convert_direct_solver_defense_to_positive_good(
    tmp_path,
):
    path = tmp_path / "ratings.csv"
    pd.DataFrame(
        {
            "player_id": [1],
            "off": [2.5],
            "def": [-1.25],
        }
    ).to_csv(path, index=False)

    offense, defense = static_rating_map(path, defense_multiplier=-1.0)

    assert offense == {"1": 2.5}
    assert defense == {"1": 1.25}

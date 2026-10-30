import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_vpm_own_skills import snapshot_rows
from estimated_skills_context import running_group_residual


def test_snapshot_excludes_cut_date_and_keeps_zero_and_missing_distinct():
    states = pd.DataFrame(dict(player_id=[1, 1, 2], game_date=pd.to_datetime(['2024-01-01', '2024-01-02', '2020-01-01']),
                               value=[0., 100., 3.], state_variance=[1., 1., 1.], observed_exposure=[10., 20., 5.]))
    got = snapshot_rows(states, ['2024-01-02']).set_index('player_id')
    assert got.loc[1, 'value'] == 0
    assert not got.loc[1, 'is_missing']
    assert pd.isna(got.loc[2, 'value']) and got.loc[2, 'is_missing']


def test_context_normalization_cannot_change_with_future_exposure():
    sub = pd.DataFrame(dict(season=[2024, 2024, 2026], opp_team_id=[1, 1, 1],
                            game_date=pd.to_datetime(['2024-01-01', '2024-01-02', '2026-01-01']), game_id=['a', 'b', 'c']))
    old = running_group_residual(sub.iloc[:2], np.array([.1, .2]), np.array([50., 50.]), 5., reference_end=2025)
    new = running_group_residual(sub, np.array([.1, .2, 1000.]), np.array([50., 50., 1e9]), 5., reference_end=2025)
    np.testing.assert_array_equal(old, new[:2])

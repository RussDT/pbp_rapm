import numpy as np
import pandas as pd

from nba_pipeline.scripts.process_rapm_blocks.clean_chance_utils import (
    DECOMP_SHOT_ACTION_COLUMNS,
    build_decomp_shot_action_components,
)
from nba_pipeline.scripts.rapm import (
    compute_alt_first_chance_baselines,
    detect_file_type_and_prepare,
    recenter_player_coefficients_by_side_possessions,
)


def test_decomp_shot_children_close_to_parent_for_every_action_type():
    index = pd.RangeIndex(5)
    actual = pd.Series([2.0, 0.0, 3.0, 0.0, 0.0], index=index)
    is_rim = pd.Series([True, False, False, False, False], index=index)
    is_mid = pd.Series([False, True, False, False, False], index=index)
    is_three = pd.Series([False, False, True, False, False], index=index)
    is_fga = is_rim | is_mid | is_three
    ft_baseline = pd.Series([0.0, 0.0, 0.0, 0.54, 0.0], index=index)

    out = build_decomp_shot_action_components(
        actual_net=actual,
        is_fga=is_fga,
        is_rim_fga=is_rim,
        is_mid_fga=is_mid,
        is_three_fga=is_three,
        ft_sq_baseline=ft_baseline,
        rim_avg_pts=1.25,
        mid_avg_pts=0.85,
        three_avg_pts=1.12,
    )

    expected_parent = pd.Series([2.0, 0.0, 3.0, 0.54, 0.0], index=index)
    np.testing.assert_allclose(out[DECOMP_SHOT_ACTION_COLUMNS].sum(axis=1), expected_parent)
    np.testing.assert_allclose(out['FC_DECOMP_EFG_Action_Score'], expected_parent)
    np.testing.assert_allclose(out['FC_DECOMP_MID_VALUE_Action_Score'], 0.0)


def test_mid_component_carries_the_counterfactual_removed_from_frequency():
    index = pd.RangeIndex(3)
    actual = pd.Series([2.0, 0.0, 3.0], index=index)
    is_rim = pd.Series([True, False, False], index=index)
    is_mid = pd.Series([False, True, False], index=index)
    is_three = pd.Series([False, False, True], index=index)

    out = build_decomp_shot_action_components(
        actual_net=actual,
        is_fga=pd.Series(True, index=index),
        is_rim_fga=is_rim,
        is_mid_fga=is_mid,
        is_three_fga=is_three,
        ft_sq_baseline=pd.Series(0.0, index=index),
        rim_avg_pts=1.25,
        mid_avg_pts=0.85,
        three_avg_pts=1.12,
    )

    np.testing.assert_allclose(out['FC_DECOMP_MID_FG_Action_Score'], [0.85, 0.0, 0.85])
    np.testing.assert_allclose(out['FC_DECOMP_RIM_FREQ_Action_Score'], [0.40, 0.0, 0.0])
    np.testing.assert_allclose(out['FC_DECOMP_THREE_FREQ_Action_Score'], [0.0, 0.0, 0.27])


def test_turnover_baseline_imputation_preserves_decomp_parent_child_identity():
    children = {
        'DECOMP_RIM_FREQ': 'FC_DECOMP_RIM_FREQ_Diff',
        'DECOMP_RIM_FG': 'FC_DECOMP_RIM_FG_Diff',
        'DECOMP_MID_FG': 'FC_DECOMP_MID_FG_Diff',
        'DECOMP_THREE_FREQ': 'FC_DECOMP_THREE_FREQ_Diff',
        'DECOMP_THREE_FG': 'FC_DECOMP_THREE_FG_Diff',
    }
    df = pd.DataFrame(
        {
            'Net_Diff': [1.0, 2.0, 0.0],
            'Is_Turnover': [0, 0, 1],
            'FC_DECOMP_RIM_FREQ_Diff': [0.2, 0.1, 0.0],
            'FC_DECOMP_RIM_FG_Diff': [0.3, -0.1, 0.0],
            'FC_DECOMP_MID_FG_Diff': [0.4, 0.8, 0.0],
            'FC_DECOMP_THREE_FREQ_Diff': [0.05, 0.2, 0.0],
            'FC_DECOMP_THREE_FG_Diff': [0.05, 1.0, 0.0],
        }
    )
    child_columns = list(children.values())
    df['FC_DECOMP_EFG_Diff'] = df[child_columns].sum(axis=1)
    prefixes = ['DECOMP_EFG', *children]
    baselines = compute_alt_first_chance_baselines(df, requested_prefixes=prefixes)

    prepared = {
        prefix: detect_file_type_and_prepare(
            df,
            prefix=prefix,
            alt_baselines=baselines,
        )['Off_Diff']
        for prefix in prefixes
    }

    np.testing.assert_allclose(
        prepared['DECOMP_EFG'],
        sum(prepared[prefix] for prefix in children),
    )


def test_solver_recenters_each_side_with_its_own_possession_weights():
    players = ['1', '2']
    columns = {'1_off': 0, '2_off': 1, '1_def': 2, '2_def': 3}
    beta = np.array([3.0, -1.0, 2.0, -4.0])
    off_poss = {'1': 90, '2': 10}
    def_poss = {'1': 20, '2': 80}

    recenter_player_coefficients_by_side_possessions(
        beta, columns, players, off_poss, def_poss
    )

    assert abs(beta[0] * 90 + beta[1] * 10) < 1e-12
    assert abs(beta[2] * 20 + beta[3] * 80) < 1e-12

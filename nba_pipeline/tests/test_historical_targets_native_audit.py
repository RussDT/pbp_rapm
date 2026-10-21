"""Independent native endpoint and equal-quarter bootstrap audit helpers."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

def test_independent_native_audit_uses_signed_raw_coefficients_and_known_players():
    from historical_targets_native_audit import raw_lineup_prediction
    d = pd.DataFrame([{**{f'O{i}': i for i in range(1, 6)},
                       **{f'D{i}': i + 5 for i in range(1, 6)}}])
    beta = np.r_[np.full(10, .02), np.full(10, -.01)]
    value, known = raw_lineup_prediction(d, np.arange(1, 11), beta, 1.)
    assert known.all() and value[0] == pytest.approx(1.05)
    d.loc[0, 'D5'] = 99
    value, known = raw_lineup_prediction(d, np.arange(1, 11), beta, 1.)
    assert not known.any() and np.isnan(value[0])


def test_independent_native_bootstrap_weights_quarters_equally():
    from historical_targets_native_audit import independent_bootstrap
    pairs = pd.DataFrame(dict(estimate_date=['a', 'b', 'b'], rows=[100, 1, 1],
                              loss_gain_sum=[100., -2., -2.]))
    result = independent_bootstrap(pairs, replicates=100)
    assert result == dict(mse_gain=-.5, ci_low=-.5, ci_high=-.5)


def test_independent_control_equation_check_rejects_a_changed_saved_coefficient():
    from historical_targets_native_audit import normal_equation_check
    from test_vpm_net_lane_targets import sample
    d, g = sample()
    for side, offset in [('O', 0), ('D', 5)]:
        for j in range(1, 6):
            d[f'{side}{j}'] = j + offset
    d['event_num'] = np.arange(len(d))
    d['sq_active'] = False
    for zone in ['mid', 'three']:
        d[zone + '_gross_residual'] = 0.
    vectors = {key: np.zeros(20) for key in ['actual_beta', 'mid_weak', 'mid_strong', 'three_weak', 'three_strong']}
    board = pd.DataFrame(dict(td_off_possessions=np.r_[np.full(5, len(d)), np.zeros(5)],
                              td_def_possessions=np.r_[np.zeros(5), np.full(5, len(d))]))
    for field in ['raw_ovpm', 'raw_dvpm', 'combined_prior_off', 'combined_prior_def']:
        board[field] = 0.
    gaps = normal_equation_check(d, g, '2020-02-01', np.arange(1, 11), vectors, board)
    assert max(gaps.values()) < 1e-12
    board.loc[0, 'raw_ovpm'] = 1.
    with pytest.raises(ValueError, match='not matched prior responses'):
        normal_equation_check(d, g, '2020-02-01', np.arange(1, 11), vectors, board)
    board.loc[0, 'raw_ovpm'] = 0.
    vectors['actual_beta'][0] = .1
    with pytest.raises(ValueError, match='normal equation failed'):
        normal_equation_check(d, g, '2020-02-01', np.arange(1, 11), vectors, board)


def test_rejected_methodology_cannot_receive_final_native_acceptance(tmp_path):
    import json
    from historical_targets_native_audit import run
    (tmp_path / 'rejected_availability_transition.json').write_text(json.dumps(
        dict(status='rejected_methodology_before_release')))
    with pytest.raises(ValueError, match='Rejected experiment'):
        run(tmp_path)

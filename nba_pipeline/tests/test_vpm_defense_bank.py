import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_vpm_defense_bank import summarize, shooting_foul_events


def measurement_frame():
    return pd.DataFrame(dict(player_id=[1, 1, 2, 1], game_id=['a', 'b', 'c', 'd'],
        game_date=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-03']),
        numerator=[2., 0., 4., 900.], denominator=[100.] * 4, feature=['def_deflections100'] * 4))


def test_snapshot_excludes_current_date_and_future_and_preserves_missing():
    d = measurement_frame()
    a, p = summarize(d, [1, 2, 3], '2020-01-03', half_life=550)
    changed = d.copy()
    changed.loc[changed.game_date.ge('2020-01-03'), 'numerator'] = 1e9
    b, _ = summarize(changed, [1, 2, 3], '2020-01-03', half_life=550)
    pd.testing.assert_frame_equal(a, b)
    assert np.isnan(a.loc[2, 'def_deflections100'])
    assert p.last_observation_date.lt(p.estimate_date).all()
    # A covered zero increases exposure rather than becoming missing.
    assert a.loc[0, 'def_deflections100__exposure'] > 199


def test_calibration_uses_only_exact_matched_games_and_sums_counts():
    d = measurement_frame()
    a, _ = summarize(d, [1, 2, 3], '2020-02-01', game_ids={'a', 'b'})
    assert a.loc[0, 'def_deflections100'] == 1.
    assert a.loc[0, 'def_deflections100__exposure'] == 200.
    assert np.isnan(a.loc[1, 'def_deflections100'])


def test_shooting_foul_labels_work_when_archive_subtypes_are_all_zero():
    raw = pd.DataFrame(dict(event_type=[6, 6, 6, 6, 1], event_action_type=[0] * 5,
        player1_id=[1, 2, 3, 4, 5],
        home_description=['Brown S.FOUL (P1.T1)', 'Brown P.FOUL (P2.T2)', None, 'Shooting Foul', 'S.FOUL'],
        neutral_description=[None] * 5,
        visitor_description=[None, None, 'Reed S.FOUL (P1.T1)', None, None]))
    actual = shooting_foul_events(raw)
    assert actual.player1_id.tolist() == [1, 3, 4]

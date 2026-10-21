"""Chronology, exposure, coherent shooting and source-clock contracts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from historical_features import cuts, normalize_clock, eb_feature, feature_snapshot


def sample_lines():
    d = pd.DataFrame(dict(player_id=[1,1,2,3], game_id=['a','b','c','d'],
        game_date=pd.to_datetime(['2013-12-01','2014-01-01','2013-12-02','2013-12-03']),
        season=[2014]*4, season_type=['rs']*4, counts_observed=[True,True,True,False],
        minutes=[30.,48.,10.,np.nan], off_poss=[50.]*4, def_poss=[50.]*4,
        rim_fga=[5.,50.,0.,np.nan], rim_fgm=[3.,50.,0.,np.nan],
        mid_fga=[2.,20.,2.,np.nan], mid_fgm=[1.,20.,1.,np.nan],
        three_fga=[3.,30.,3.,np.nan], three_fgm=[1.,30.,1.,np.nan],
        fta=[2.,10.,0.,np.nan], ftm=[2.,10.,0.,np.nan]))
    for c in ['pts','ast','tov','oreb','dreb','stl','blk','pf_committed']:
        d[c] = [0.,10.,0.,np.nan]
    d['total_poss'] = d.off_poss+d.def_poss
    d['usage_events'] = d.rim_fga+d.mid_fga+d.three_fga+.44*d.fta+d.tov
    return d


def test_quarterly_schedule_has_every_requested_cut_and_training_anchors():
    dates = cuts()
    report = pd.date_range('2014-01-01','2026-10-01',freq='QS')
    assert len(report)==52 and set(report).issubset(dates)
    assert pd.Timestamp('2008-10-01') in dates
    assert pd.Timestamp('2020-12-01') in dates
    assert pd.Timestamp('2026-09-01') in dates
    assert max(dates)==pd.Timestamp('2026-10-01')


def test_clock_ignores_already_full_historical_seconds():
    d = pd.DataFrame(dict(time_quarter=['12:00','11:47'],
        minute_remaining_quarter=[12,11],seconds_remaining_quarter=[720,707]))
    n = normalize_clock(d)
    assert (n.minute_remaining_quarter*60+n.seconds_remaining_quarter).tolist()==[720.,707.]
    assert d.seconds_remaining_quarter.tolist()==[720,707]


def test_bad_clock_fails_closed():
    with pytest.raises(ValueError,match='clock'):
        normalize_clock(pd.DataFrame(dict(time_quarter=['bad'])))


def test_same_day_and_future_observations_cannot_change_snapshot():
    d = sample_lines()
    cut = pd.Timestamp('2014-01-01')
    first, p = feature_snapshot(d,cut)
    d.loc[d.game_date.ge(cut), ['rim_fga','rim_fgm','minutes']] = 99999
    second, q = feature_snapshot(d,cut)
    pd.testing.assert_frame_equal(first,second)
    pd.testing.assert_frame_equal(p,q)
    assert p.last_observation_date.lt(cut).all()


def test_measured_zero_updates_exposure_but_unknown_counts_stay_missing():
    d,p = feature_snapshot(sample_lines(),'2014-01-01')
    d = d.set_index('player_id')
    assert d.loc[1,'own_blk_rate']==0 and d.loc[2,'own_blk_rate']==0
    assert d.loc[1,'own_blk_rate_missing']==0
    assert np.isnan(d.loc[3,'own_blk_rate']) and d.loc[3,'own_blk_rate_missing']==1
    assert 3 not in set(p.loc[p.feature.eq('own_blk_rate'),'player_id'])


def test_shooting_makes_are_coherent_and_zero_attempts_stay_zero():
    d,_ = feature_snapshot(sample_lines(),'2014-01-01')
    d = d.set_index('player_id')
    for zone,attempt in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),
                         ('three','three_fga_rate'),('ft','fta_rate')]:
        good = d['own_'+zone+'_pct'].notna()
        np.testing.assert_allclose(d.loc[good,'own_'+zone+'_makes100'],
            d.loc[good,'own_'+attempt]*d.loc[good,'own_'+zone+'_pct'])
        assert (d['own_'+zone+'_makes100'].dropna()<=d['own_'+attempt].dropna()+1e-12).all()
    assert np.isnan(d.loc[2,'own_rim_pct'])
    assert d.loc[2,'own_rim_makes100']>=0  # EB attempt-rate contains population prior.


def test_mpg_uses_only_recent_rs_positive_minute_appearances():
    x = sample_lines()
    extra = x.iloc[[0,0,0]].copy()
    extra['game_date'] = pd.to_datetime(['2012-12-31','2013-12-04','2013-12-05'])
    extra['game_id']=['old','playoff','zero']
    extra['season_type']=['rs','ps','rs']
    extra['minutes']=[40.,48.,0.]
    d,_ = feature_snapshot(pd.concat([x,extra]),'2014-01-01')
    d = d.set_index('player_id')
    assert d.loc[1,'ctx_mpg']==30 and d.loc[1,'ctx_games']==1


def test_weighted_rate_uses_count_sums_and_prefix_population():
    d = sample_lines()
    v,e,last,league = eb_feature(d,[1,2,3],pd.Timestamp('2014-01-01'),'rim_fga','off_poss',1000)
    assert np.isnan(v[2])
    weights = np.exp2(-np.array([31,30])/550.)
    expected_league = 5*weights[0]/(50*weights.sum())
    assert league==pytest.approx(expected_league)
    assert v[0]==pytest.approx(100*(5*weights[0]+1000*league)/(50*weights[0]+1000))


def test_missing_entire_tracking_era_is_nan_not_zero():
    columns=['player_id','game_id','game_date','feature','numerator','denominator']
    empty=pd.DataFrame(columns=columns)
    empty['game_date']=pd.to_datetime(empty.game_date)
    d,_=feature_snapshot(sample_lines(),'2014-01-01',measurements=empty)
    assert d.def_contested_dreb100.isna().all()
    assert d.def_contested_dreb100_missing.eq(1).all()


def test_annual_defense_not_used_until_july_and_uses_full_official_exposure():
    lines = sample_lines()
    lines[['stl','blk']] = np.nan
    annual = pd.DataFrame(dict(player_id=[1],game_date=pd.to_datetime(['2013-06-30']),
        available_date=pd.to_datetime(['2013-07-01']),stl=[100.],blk=[50.],annual_poss=[5000.]))
    early,_ = feature_snapshot(lines,'2013-06-30',annual_defense=annual)
    assert early.empty  # No known player's observed game precedes this cut.
    d,p = feature_snapshot(lines,'2014-01-01',annual_defense=annual)
    d = d.set_index('player_id')
    assert d.loc[1,'own_stl_rate']==pytest.approx(2.)
    assert d.loc[1,'own_blk_rate']==pytest.approx(1.)
    assert np.isnan(d.loc[2,'own_stl_rate'])
    rows=p.loc[p.feature.eq('own_stl_rate')]
    assert rows.estimator.str.contains('annual_official_poss').all()


def test_future_annual_totals_do_not_change_earlier_snapshot():
    lines=sample_lines();lines[['stl','blk']]=np.nan
    annual=pd.DataFrame(dict(player_id=[1,1],
        game_date=pd.to_datetime(['2013-06-30','2014-06-30']),
        available_date=pd.to_datetime(['2013-07-01','2014-07-01']),
        stl=[100.,100000.],blk=[50.,100000.],annual_poss=[5000.,5000.]))
    first,_=feature_snapshot(lines,'2014-01-01',annual_defense=annual)
    second,_=feature_snapshot(lines,'2014-01-01',annual_defense=annual.iloc[[0]])
    pd.testing.assert_frame_equal(first,second)


def test_july1_annual_availability_boundary_for_known_player():
    lines=sample_lines();lines['game_date']=pd.Timestamp('2013-01-01');lines[['stl','blk']]=np.nan
    annual=pd.DataFrame(dict(player_id=[1],game_date=pd.to_datetime(['2013-06-30']),
        available_date=pd.to_datetime(['2013-07-01']),stl=[100.],blk=[50.],annual_poss=[5000.]))
    before,_=feature_snapshot(lines,'2013-06-30',annual_defense=annual)
    after,_=feature_snapshot(lines,'2013-07-01',annual_defense=annual)
    assert before.own_stl_rate.isna().all()
    assert after.set_index('player_id').loc[1,'own_stl_rate']==pytest.approx(2.)

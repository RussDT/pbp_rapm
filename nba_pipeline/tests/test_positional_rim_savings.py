import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from positional_rim_savings import (fit_positional_baseline, transform_positional_baseline,
                                   mapper_rim_basis, CANDIDATE_COLUMNS,
                                   REGISTERED_POOLING_ATTEMPTS)


def packet(date='2020-01-01'):
    return pd.DataFrame(dict(player_id=[1, 2, 3], estimate_date=pd.Timestamp(date),
        own_rim_saved100=[1.6, -.8, 0.], own_rim_contests100=[8., 4., 2.],
        own_rim_suppression=[.1, -.1, 0.], own_rim_suppression__prior_exposure=[100., 10., 40.],
        rim_last_observation_date=pd.Timestamp(date)-pd.Timedelta(days=1),
        role_available_at=pd.Timestamp(date)-pd.Timedelta(days=2),
        role_source_last_date=pd.Timestamp(date)-pd.Timedelta(days=3),
        role_guard=[1., 0., .5], role_wing=[0., 1., 0.], role_big=[0., 0., .5]))


def fit(d=None, cut='2020-02-01'):
    return fit_positional_baseline(packet() if d is None else d, cut, pooling_attempts=50.)


def test_role_quality_baseline_shrinks_and_exact_raw_control_closes():
    d=packet();m=fit(d);pooled=(100*.1+10*(-.1))/150
    assert m['pooled_suppression']==pooled
    assert m['role_support']==[120.,10.,20.]
    assert m['role_mean_suppression'][0]==pytest.approx((10+50*pooled)/170)
    z=transform_positional_baseline(d,m,mode='training')
    np.testing.assert_array_equal(z.rim_saved100_raw_control,d.own_rim_saved100)
    np.testing.assert_allclose(z.rim_role_quality_baseline_saved100+
        z.rim_position_relative_quality_saved100,z.rim_saved100_raw_control,atol=1e-15)
    assert m['uses_outcomes'] is False and m['modifies_ratings'] is False
    json.dumps(m,allow_nan=False)


def test_future_predictor_perturbation_and_target_fields_cannot_change_baseline():
    old=packet();future=packet('2020-03-01');future['own_rim_suppression']=1e50
    mixed=pd.concat([old,future],ignore_index=True)
    mixed['future_lineup_points']=1e100;mixed['future_def_possessions']=1e200
    assert fit(mixed)==fit(old)
    assert fit(old.assign(final_def_rating=-999.))==fit(old)


def test_training_subset_order_invariant_but_tamper_or_foreign_query_rejected():
    d=packet();m=fit(d)
    a=transform_positional_baseline(d.iloc[[2,0]],m,mode='training')
    assert a.player_id.tolist()==[3,1]
    bad=d.copy();bad.loc[0,'own_rim_suppression']=.2;bad.loc[0,'own_rim_saved100']=3.2
    with pytest.raises(ValueError,match='exact fitted'):transform_positional_baseline(bad,m,mode='training')
    bad=d.copy();bad.loc[0,'player_id']=999
    with pytest.raises(ValueError,match='exact fitted'):transform_positional_baseline(bad,m,mode='training')
    with pytest.raises(ValueError,match='after deployment'):transform_positional_baseline(d,m)


def test_missing_role_and_missing_quality_stay_missing_raw_control_unchanged():
    m=fit();d=packet('2020-02-01');d.loc[0,['role_guard','role_wing','role_big']]=np.nan
    d.loc[1,'own_rim_suppression__prior_exposure']=0
    d.loc[2,['own_rim_saved100','own_rim_contests100','own_rim_suppression']]=np.nan
    z=transform_positional_baseline(d,m)
    assert not z.position_relative_supported.any()
    assert z.rim_role_quality_baseline_saved100.isna().all()
    pd.testing.assert_series_equal(z.rim_saved100_raw_when_role_unknown,
                                  d.own_rim_saved100,check_names=False)
    pd.testing.assert_series_equal(z.rim_saved100_raw_control,d.own_rim_saved100,check_names=False)


def test_sparse_and_unseen_position_shrink_to_training_pool_not_current_packet():
    d=packet();d[['role_guard','role_wing','role_big']]=[1.,0.,0.];m=fit(d)
    assert m['role_support'][2]==0
    assert m['role_mean_suppression'][2]==pytest.approx(m['pooled_suppression'])
    query=packet('2020-02-01');query[['role_guard','role_wing','role_big']]=[0.,0.,1.]
    z=transform_positional_baseline(query,m)
    np.testing.assert_allclose(z.rim_role_quality_baseline_saved100,
                               2*query.own_rim_contests100*m['pooled_suppression'])
    more=pd.concat([query,query.assign(player_id=[4,5,6])],ignore_index=True)
    doubled=transform_positional_baseline(more,m)
    pd.testing.assert_frame_equal(z,doubled.iloc[:3].reset_index(drop=True))


def test_no_early_position_training_support_is_explicitly_inactive():
    d=packet('2015-01-01');d[['role_guard','role_wing','role_big']]=np.nan
    d[['role_available_at','role_source_last_date']]=pd.NaT
    m=fit(d,cut='2015-02-01');assert not m['active']
    z=transform_positional_baseline(packet('2015-03-01'),m)
    assert z.rim_position_relative_quality_saved100.isna().all()
    np.testing.assert_array_equal(z.rim_saved100_raw_control,[1.6,-.8,0.])
    np.testing.assert_array_equal(z.rim_saved100_raw_when_role_unknown,[1.6,-.8,0.])
    empty=fit_positional_baseline(d.iloc[:0],'2015-02-01',pooling_attempts=50.)
    assert empty['pooled_suppression'] is None and not empty['active']
    json.dumps(empty,allow_nan=False)


def test_volume_remains_physical_no_position_volume_equalization():
    m=fit();a=packet('2020-02-01').iloc[[0]].copy();b=a.copy()
    b.player_id=4;b.own_rim_contests100*=2;b.own_rim_saved100*=2
    z=transform_positional_baseline(pd.concat([a,b],ignore_index=True),m)
    for c in ['rim_saved100_raw_control','rim_role_quality_baseline_saved100',
              'rim_position_relative_quality_saved100']:
        assert z.loc[1,c]==pytest.approx(2*z.loc[0,c])


@pytest.mark.parametrize('column',['rim_last_observation_date','role_available_at','role_source_last_date'])
def test_same_day_or_future_source_provenance_rejected(column):
    d=packet();d[column]=d.estimate_date
    with pytest.raises(ValueError,match='Same-day/future'):fit(d)


@pytest.mark.parametrize('kind',['partial_role','zero_roles','identity','negative_support','units','duplicate','protected'])
def test_invalid_identity_roles_opportunity_units_and_protected_queries_fail_closed(kind):
    d=packet()
    if kind=='partial_role':d.loc[0,'role_guard']=np.nan
    elif kind=='zero_roles':d.loc[0,['role_guard','role_wing','role_big']]=0
    elif kind=='identity':d.loc[0,'player_id']=0
    elif kind=='negative_support':d.loc[0,'own_rim_suppression__prior_exposure']=-1
    elif kind=='units':d.loc[0,'own_rim_saved100']=.8
    elif kind=='duplicate':d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
    elif kind=='protected':d.estimate_date=pd.Timestamp('2027-01-01')
    with pytest.raises(ValueError):fit(d)


def test_disjoint_fallback_preserves_raw_span_missing_and_observed_zero():
    m=fit();d=packet('2020-02-01')
    d.loc[1,['role_guard','role_wing','role_big']]=np.nan
    d.loc[2,'own_rim_suppression__prior_exposure']=0
    z=transform_positional_baseline(d,m)
    assert pd.isna(z.loc[0,'rim_saved100_raw_when_role_unknown'])
    assert z.loc[1,'rim_saved100_raw_when_role_unknown']==-.8
    assert z.loc[2,'rim_saved100_raw_when_role_unknown']==0.
    assert z.loc[1:2,['rim_role_quality_baseline_saved100',
        'rim_position_relative_quality_saved100']].isna().all().all()
    np.testing.assert_allclose(z[CANDIDATE_COLUMNS].sum(axis=1,min_count=1),d.own_rim_saved100)
    assert not set(CANDIDATE_COLUMNS)&{'rim_saved100_raw_control'}
    assert REGISTERED_POOLING_ATTEMPTS==500.


def test_model_aware_whole_block_fallback_identical_calibration_deployment():
    m=fit();d=packet('2020-02-01');z=transform_positional_baseline(d,m)
    old=mapper_rim_basis(z,role_decomposition_admitted=False)
    assert list(old)==['rim_saved100_raw_control']
    np.testing.assert_array_equal(old.iloc[:,0],d.own_rim_saved100)
    new=mapper_rim_basis(z,role_decomposition_admitted=True)
    assert list(new)==CANDIDATE_COLUMNS
    np.testing.assert_allclose(new.sum(axis=1,min_count=1),d.own_rim_saved100)
    with pytest.raises(ValueError,match='admission'):mapper_rim_basis(z,role_decomposition_admitted=None)


def test_stale_role_provenance_rejected_instead_of_using_latest_static_position():
    d=packet();d.role_source_last_date=pd.Timestamp('2016-01-01')
    with pytest.raises(ValueError,match='staleness'):fit(d)


def test_earlier_query_density_does_not_recount_cumulative_attempt_support():
    d=packet();d.rim_last_observation_date=pd.Timestamp('2019-12-15')
    d.role_available_at=pd.Timestamp('2019-12-14')
    d.role_source_last_date=pd.Timestamp('2019-12-13')
    old=d.copy();old.estimate_date=pd.Timestamp('2019-12-20')
    # Same old measurement/attempt mass repeated under another earlier query.
    repeated=old.copy();repeated.estimate_date=pd.Timestamp('2019-12-19')
    other=old.copy();other.estimate_date=pd.Timestamp('2019-12-21')
    other.own_rim_suppression=.8
    other.own_rim_saved100=2*other.own_rim_contests100*.8
    dense=pd.concat([d,old,repeated,other],ignore_index=True).sample(frac=1,random_state=9)
    a=fit(d);b=fit(dense)
    for name in ['pooled_suppression','role_mean_suppression','role_support',
                 'latest_player_packets','quality_supported_fit_players','known_role_training_rows']:
        assert a[name]==b[name]
    assert b['training_rows']==12 and len(b['training_row_hashes'])==12
    assert b['latest_player_packets']==3
    z=transform_positional_baseline(dense,b,mode='training')
    np.testing.assert_allclose(z[CANDIDATE_COLUMNS].sum(axis=1,min_count=1),dense.own_rim_saved100)


def test_latest_missing_packet_does_not_recover_an_earlier_observed_baseline_weight():
    old=packet();new=packet('2020-01-02')
    new.own_rim_suppression__prior_exposure=0
    m=fit(pd.concat([old,new],ignore_index=True))
    assert m['quality_supported_training_rows']==3
    assert m['quality_supported_fit_players']==0 and not m['active']
    assert m['role_support']==[0.,0.,0.]


def test_latest_packet_uses_parsed_date_order_not_string_sort_order():
    old=packet('2019-12-20');new=packet('2020-01-02')
    old.own_rim_suppression=.8;old.own_rim_saved100=2*old.own_rim_contests100*.8
    d=pd.concat([new,old],ignore_index=True)
    expected=fit(d)
    d.estimate_date=d.estimate_date.dt.strftime('%m/%d/%Y')
    actual=fit(d)
    assert actual==expected


def test_fractional_support_density_invariance_is_bit_exact_without_zero_padding():
    d=pd.concat([packet().iloc[[i%3]] for i in range(16)],ignore_index=True)
    d.player_id=np.arange(1,17)
    d.own_rim_suppression__prior_exposure=np.random.RandomState(11).uniform(.01,1000,16)
    d.rim_last_observation_date=pd.Timestamp('2019-12-15')
    d.role_available_at=pd.Timestamp('2019-12-14')
    d.role_source_last_date=pd.Timestamp('2019-12-13')
    old=d.copy();old.estimate_date=pd.Timestamp('2019-12-20')
    a=fit(d);b=fit(pd.concat([d,old],ignore_index=True))
    # This fixed vector exposes the original zero-padded reduction roundoff.
    w=d.own_rim_suppression__prior_exposure.to_numpy()
    padded=np.zeros(32);padded[1::2]=w
    assert w.sum()!=padded.sum()
    for name in ['pooled_suppression','role_mean_suppression','role_support',
                 'supported_fit_attempt_mass']:
        assert a[name]==b[name]

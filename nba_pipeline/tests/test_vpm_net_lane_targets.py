import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vpm_net_lane_targets import targets,compact,LANES

def sample():
    d=pd.DataFrame(dict(game_id=[1]*7,game_date=pd.to_datetime(['2020-01-01']*7),source_season_end_year=[2020]*7,season_phase=['RS']*7,actual_points=[2,2,0,3,2,1,0],First_Chance_Diff=[2,0,0,3,2,1,0],Second_Chance_Diff=[0,2,0,0,0,0,0],Is_Turnover=[0,0,0,0,0,0,1],rim_attempts=[1,1,0,0,0,0,0],rim_points=[2,0,0,0,0,0,0],mid_attempts=[0,0,1,0,0,0,0],mid_points=[0]*7,three_attempts=[0,0,0,1,0,0,0],three_points=[0,0,0,3,0,0,0],ft_points=[0,0,0,0,2,1,0],ft_reference_weight=[0,0,0,0,1,0,0],FC_MISS_RIM=[0,1,0,0,0,0,0],FC_MISS_MID=[0,0,1,0,0,0,0],FC_MISS_THREE=[0]*7))
    g=pd.DataFrame(dict(game_id=[1],game_date=pd.to_datetime(['2020-01-01']),source_season_end_year=[2020],season_phase=['RS'],rim_attempts=[2],rim_fg_points=[2],rim_misses=[1],rim_points=[2],mid_attempts=[1],mid_fg_points=[0],mid_misses=[1],mid_points=[0],three_attempts=[1],three_fg_points=[3],three_misses=[0],three_points=[0]));return d,g

def test_continuation_payback_ft_and_turnover_closure():
    d,g=sample();y,m=targets(d,g);np.testing.assert_allclose(y[:,:9].sum(axis=1),d.actual_points,atol=1e-12);assert y[1,0]==2 and y[1,8]==0;assert y[5,6]==1;assert y[4,2]==1.25 and y[4,6]==.75
    assert y[-1,7]==-m['net_turnover_price'];assert y[-1,:7].sum()==pytest.approx(m['net_turnover_price'])

def test_weighted_imputation_and_compact_rollup_preserve_parent():
    d,g=sample();w=np.arange(1,8,dtype=float);y,m=targets(d,g,w);np.testing.assert_allclose(y[:,:9].sum(axis=1),d.actual_points,atol=1e-12);np.testing.assert_allclose(compact(y).sum(axis=1),d.actual_points,atol=1e-12);np.testing.assert_allclose(y[-1,:7],np.average(y[:-1,:7],weights=w[:-1],axis=0),atol=1e-12)

def test_missing_or_future_population_join_cannot_become_zero():
    d,g=sample();g.loc[0,'rim_attempts']=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):targets(d,g)
    d,g=sample();g.game_id=2
    with pytest.raises(ValueError,match='Pricing misses'):targets(d,g)

def test_nonzero_midrange_reference_changes_frequency_not_total():
    d,g=sample()
    # Unfiltered pricing includes two mid attempts worth two FG points in total.
    g.loc[0,['mid_attempts','mid_fg_points','mid_misses','mid_points']]=[2,2,1,0]
    y,_=targets(d,g);five=compact(y)
    assert y[0,0]==pytest.approx(2.) and five[0,0]==pytest.approx(1.)
    assert y[3,4]==pytest.approx(3.) and five[3,3]==pytest.approx(2.)
    assert five[0,2]==pytest.approx(1.) and five[3,2]==pytest.approx(1.)
    np.testing.assert_allclose(five.sum(axis=1),d.actual_points,atol=1e-12)

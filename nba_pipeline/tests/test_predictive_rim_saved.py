import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_predictive_rim_saved import rolling,lagged_league


def panel():
    return pd.DataFrame({'player_id':[1,1,1], 'game_date':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03']), 'rim_dfga':[5.,0.,5.], 'rim_dfgm':[1.,0.,4.], 'def_poss':[50.,50.,50.], 'league_pct':[.65,.65,.65]})


def test_no_current_or_future_outcome_in_prediction():
    d=panel();a=rolling(d,365,730,400,.1)
    d.loc[2,'rim_dfgm']=0
    b=rolling(d,365,730,400,.1)
    np.testing.assert_array_equal(a[:,:3],b[:,:3])
    assert a[2,5]!=b[2,5]


def test_joint_identity_and_zero_attempts():
    d=panel();a=rolling(d,365,730,400,.1)
    np.testing.assert_allclose(a[:,2],2*a[:,0]*a[:,1])
    np.testing.assert_allclose(a[:,5],2*a[:,3]*a[:,4])
    assert a[1,3]<a[1,0]  # measured zero contests updates volume
    assert a[1,4]==a[1,1]  # no shot means no finishing update
    assert a[0,2]==0


def test_league_excludes_whole_current_date():
    d=panel();d.loc[1,'game_date']=d.loc[0,'game_date']
    a=lagged_league(d);d.loc[0,'rim_dfgm']=5
    b=lagged_league(d)
    np.testing.assert_array_equal(a[:2],b[:2]);assert a[2]!=b[2]

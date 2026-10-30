import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from research_shooting_fusion import states,residual_filter,losses

def panel():
    return pd.DataFrame({'player_id':[1]*3,'game_date':pd.to_datetime(['2020-01-01','2020-01-03','2020-01-05']),'three_fgm':[1,0,2],'three_fga':[2,0,4],'ftm':[1,0,1],'fta':[2,0,2],'off_poss':[50,50,50],'team':[1,1,2]})

def test_pregame_features_do_not_see_current_or_future_results():
    d=panel();a=states(d,1e-6,1)
    d.loc[1:,['three_fgm','three_fga','ftm','fta']]=20
    b=states(d,1e-6,1)
    np.testing.assert_allclose(a[:2],b[:2])

def test_zero_attempt_game_updates_volume_not_shooting_mean():
    a=states(panel(),1e-6,1)
    assert a[2,0]==a[1,0]
    assert a[2,6]<a[1,6]

def test_constant_anchor_matches_base_filter_and_joint_includes_zeros():
    d=panel();a=states(d,1e-6,1)
    p,_=residual_filter(d,np.full(3,.35),1e-6,1)
    np.testing.assert_allclose(p,a[:,0])
    d['volume']=10
    l,w=losses(d,np.full(3,.4),'makes_mse')
    assert l[1]>0 and w[1]==50

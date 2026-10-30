import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
from research_three_pct_volume_prior import volume_features

def test_volume_is_prior_only_and_zero_attempt_games_inform_it():
    d=pd.DataFrame({'player_id':[1,1,1],'three_fga':[0,0,10],'off_poss':[100,100,100]})
    v=volume_features(d,.05)
    assert list(v.round(5))==[5.,3.33333,2.5]
    d.loc[2,'three_fga']=999
    assert volume_features(d,.05).equals(v)

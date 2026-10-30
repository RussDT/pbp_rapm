import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
import numpy as np
from research_team_game_seasonality import schedule,filtered

def test_team_game_count_does_not_reset_when_player_misses_game():
    d=pd.DataFrame({'player_id':[1,2,2,1,2],'game_id':['a','a','b','c','c'],'game_date':pd.to_datetime(['2024-01-01','2024-01-01','2024-01-03','2024-01-05','2024-01-05']),'season':[2024]*5,'season_type':['rs']*5})
    ctx=d[['player_id','game_id']].assign(team_id=10,is_home=1)
    x=schedule(d,ctx)
    assert x.loc[x.player_id.eq(1),'team_game'].tolist()==[1,3]

def test_known_context_is_removed_from_state_updates():
    d=pd.DataFrame({'player_id':[1,1],'game_date':pd.to_datetime(['2024-01-01','2024-01-03']),'three_fga':[10,10],'three_fgm':[4,4]})
    pred,neutral=filtered(d,np.full(2,.35),np.full(2,.05),1e-7,1.)
    np.testing.assert_allclose(pred,.4);np.testing.assert_allclose(neutral,.35)

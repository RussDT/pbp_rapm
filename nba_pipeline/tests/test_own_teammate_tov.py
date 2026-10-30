import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from build_own_teammate_tov import design,stats,board


def sample():
 d=pd.DataFrame({'actor':[1,0,2,0],'Is_Turnover':[1,0,1,0],'Season':[2024]*4,'game_id':[1,1,2,2]})
 for i in range(1,6):d[f'O{i}']=i;d[f'D{i}']=i+5
 return d


def test_role_statistics_equal_five_focal_rows():
 d=sample();O,D,ids=design(d);n=len(ids);H=[];y=[]
 for k,row in d.iterrows():
  for pid in range(1,6):
   z=-O[k].toarray().ravel()/5;z[pid-1]+=1;H.append(z);y.append(-float(row.actor==pid)+row.Is_Turnover/5)
 H=np.array(H);y=np.array(y);G,B,c=stats(d,O,ids)
 np.testing.assert_allclose(G,H.T@H,atol=1e-12);np.testing.assert_allclose(B,H.T@y);np.testing.assert_allclose(c,y@y)


def test_children_close_and_all_possessions_count():
 r=board(sample(),300)
 np.testing.assert_allclose(r.own_tov_impact+r.teammate_tov_suppression,r.total_otov,atol=1e-12)
 assert r.loc[r.player_id.eq(1),'tov_possessions'].iloc[0]==4
 assert r.loc[r.player_id.eq(1),'own_turnovers_per100'].iloc[0]==25
 assert r.loc[r.player_id.eq(1),'own_tov_impact'].iloc[0]<r.loc[r.player_id.eq(3),'own_tov_impact'].iloc[0]

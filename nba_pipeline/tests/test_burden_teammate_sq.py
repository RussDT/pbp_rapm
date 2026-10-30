import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from build_burden_adjusted_teammate_sq import adjust,FEATURES


def sample():
 rng=np.random.default_rng(52);n=60
 d=pd.DataFrame(rng.normal(size=(n,4)),columns=FEATURES)
 d['player_id']=np.arange(n);d['teammate_exposure']=1000;d['own_role_fga']=200;d['shot_share']=.25
 d['teammate_sq_conditional']=2*d.role_shot_share+rng.normal(size=n)
 return d


def test_player_does_not_train_own_baseline_and_identity_closes():
 d=sample();a,_=adjust(d);d.loc[0,'teammate_sq_conditional']+=1000;b,_=adjust(d)
 assert a.loc[0,'burden_baseline_conditional']==b.loc[0,'burden_baseline_conditional']
 np.testing.assert_allclose(a.burden_baseline_conditional+a.teammate_surplus_conditional,d.teammate_sq_conditional-np.r_[1000,np.zeros(59)])
 np.testing.assert_allclose(a.burden_baseline+a.teammate_surplus,.75*a.teammate_sq_conditional)


def test_low_exposure_is_missing_not_neutral():
 d=sample();d.loc[0,'own_role_fga']=0;r,_=adjust(d)
 assert not r.loc[0,'burden_eligible'] and pd.isna(r.loc[0,'teammate_surplus'])

"""Algebra and time-boundary checks for the monthly skill engines."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_nba_estimated_skills import SkillSpec, skill_subpanel
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2
from monthly_ou_filter import FastOU
from build_monthly_defensive_skills import RimEngine
from build_predictive_rim_saved import rolling, lagged_league


def observations():
    d=pd.DataFrame(dict(player_id=[1]*5+[2]*4,
        game_date=pd.to_datetime(['2018-01-01','2018-01-04','2018-10-20','2018-11-01','2019-01-05',
                                 '2018-02-02','2018-02-03','2018-11-02','2019-01-02']),
        season=[2018,2018,2019,2019,2019,2018,2018,2019,2019],
        attempts=[9,20,1,8,14,6,4,11,8],makes=[6,4,1,3,7,2,1,4,6],
        def_poss=[55,70,7,30,40,30,20,65,43]))
    d['game_id']=np.arange(len(d)).astype(str)
    return d


def test_compiled_ou_matches_original_with_all_corrections(tmp_path):
    sub=skill_subpanel(observations(),SkillSpec('test','makes','attempts','pct'))
    p=[800.,.02,20.,.005,.001,.2];mean=.55;n=len(sub)
    init=np.linspace(.4,.7,n);shift=np.linspace(-.03,.02,n);trans=np.linspace(-.02,.01,n)
    context=np.linspace(.01,-.02,n);lam=.8
    expected=run_filter_v2(LayoutV2(sub),sub.obs_rate.to_numpy(),p,mean,
                           mu0=init,level_shift=shift,transient=trans+lam*context)
    actual=FastOU(sub,tmp_path).run(sub.obs_rate.to_numpy(),p,mean,init,shift,trans,context,lam)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)


def test_ou_future_observations_do_not_change_previous_states(tmp_path):
    d=observations();spec=SkillSpec('test','makes','attempts','pct');p=[800.,.02,20.,.005,.001,.2]
    prefix=skill_subpanel(d.loc[d.game_date.lt('2019-01-01')],spec)
    changed=d.copy();changed.loc[changed.game_date.ge('2019-01-01'),'makes']=0
    full=skill_subpanel(changed,spec)
    expected=FastOU(prefix,tmp_path).run(prefix.obs_rate.to_numpy(),p,.55)
    actual=FastOU(full,tmp_path).run(full.obs_rate.to_numpy(),p,.55)
    mask=full.game_date.lt('2019-01-01')
    for a,b in zip(actual,expected):np.testing.assert_allclose(a[mask],b,rtol=0,atol=1e-14)


def test_unsorted_ou_histories_fail(tmp_path):
    sub=skill_subpanel(observations(),SkillSpec('test','makes','attempts','pct'))
    with pytest.raises(ValueError,match='chronological'):FastOU(sub.iloc[::-1],tmp_path)


def test_rim_vectorized_states_match_count_recurrence_and_date_prefix():
    d=observations().rename(columns={'attempts':'rim_dfga','makes':'rim_dfgm'})
    d['league_pct']=lagged_league(d)
    a=RimEngine(d).run(365,730,100,.075)
    np.testing.assert_allclose(a,rolling(d,365,730,100,.075),rtol=0,atol=1e-14)
    prefix=d.loc[d.game_date.lt('2019-01-01')].copy();prefix['league_pct']=lagged_league(prefix)
    np.testing.assert_allclose(a[d.game_date.lt('2019-01-01')],RimEngine(prefix).run(365,730,100,.075),rtol=0,atol=1e-14)
    np.testing.assert_allclose(a[:,5],2*a[:,3]*a[:,4],rtol=0,atol=1e-14)

from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_vpm_dated_roles import snapshot,role_membership,ROLES


def positions():
    d=pd.DataFrame(dict(year=[2017,2018,2019,2018],nba_id=[1,1,1,2],position=['C','PF','PG','SG'],minutes=[2000,1000,1000,500],source_last_game_date=pd.to_datetime(['2017-04-01','2018-04-01','2019-04-01','2018-04-01']),earliest_use_date=pd.to_datetime(['2017-04-02','2018-04-02','2019-04-02','2018-04-02'])))
    return d


def test_only_available_two_seasons_mix_by_minutes_and_recency():
    d=snapshot(positions(),[1,2,3],'2018-10-01').set_index('player_id')
    np.testing.assert_allclose(d.loc[1,ROLES].astype(float),[0,.5,.5]);np.testing.assert_allclose(d.loc[2,ROLES].astype(float),[1,0,0]);assert d.loc[3,ROLES].isna().all();assert not d.loc[3,'role_known']


def test_missing_before_archive_and_stale_positions_remain_missing():
    assert not snapshot(positions(),[1,2],'2015-10-01').role_known.any()
    assert not snapshot(positions(),[1,2],'2026-09-01').role_known.any()
    d=snapshot(positions(),[1],'2017-04-02');assert not d.role_known.any()


def test_future_rows_and_impact_outputs_do_not_affect_past_roles():
    x=positions();a=snapshot(x,[1,2,3],'2018-10-01');x['bpm']=1e99;x.loc[x.year.eq(2019),'position']='C'
    pd.testing.assert_frame_equal(a,snapshot(x,[1,2,3],'2018-10-01'))
    np.testing.assert_allclose(role_membership('PF-C'),[0,.5,.5])

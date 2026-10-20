"""Daily operational handoff must fail closed without changing research recipes."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import manage_estimated_skills as M


def config(tmp_path):
    (tmp_path/'parameter.json').write_text('{"fixed":1}')
    c=dict(schema_version=1,id='test',inventory_as_of='2026-10-20',season_end_year=2027,profiles={'current':'current'},research_lanes=[],families=[dict(id='core',profile='current',frozen_files=['parameter.json'],live_adapter=None,remaining_work=['wire'],fields=[dict(id='current:rate',key='rate',label='Rate',unit='/100',mapper=[],status='active',dependencies=[])])])
    p=tmp_path/'operations.json';M.write(p,c);return c,p


def test_catalog_routes_every_field_and_dependency_once():
    c=M.validate(M.read(M.CONFIG));r=M.inventory(c)
    assert r['profiles']=={'current_neutral':59,'standalone_incumbent':28,'retained_research':1}
    assert len(r['research_lanes'])==20
    assert all(x['parameters_available'] for x in r['rows'])
    assert all(x['daily_adapter']=='not_wired' for x in r['rows'])


def test_freeze_copies_immutable_recipes_and_detects_tampering(tmp_path):
    c,p=config(tmp_path);bundle=tmp_path/'bundle';M.freeze(c,p,bundle,2027,tmp_path)
    (tmp_path/'parameter.json').write_text('{"fixed":2}')
    _,m=M.verify_bundle(bundle);h=m['files']['parameter.json'];assert (bundle/'objects'/h).read_text()=='{"fixed":1}'
    with pytest.raises(FileExistsError):M.freeze(c,p,bundle,2027,tmp_path)
    (bundle/'objects'/h).write_text('corrupt')
    with pytest.raises(ValueError,match='changed'):M.verify_bundle(bundle)


def fixture_receipt(tmp_path):
    c,p=config(tmp_path);bundle=tmp_path/'bundle';M.freeze(c,p,bundle,2027,tmp_path)
    inbox=tmp_path/'inbox';inbox.mkdir();artifact=inbox/'states.parquet'
    d=pd.DataFrame(dict(player_id=[1,2],stat_key=['rate','rate'],value=[0.,np.nan],last_observation_date=['2026-10-21',None],missing_reason=['','no measured observations']))
    d.to_parquet(artifact,index=False)
    r=dict(family='core',season=2027,as_of='2026-10-22',parameter_bundle_sha256=M.sha(bundle/'manifest.json'),source_status='complete',expected_player_ids=[1,2],completed_game_census_checked=True,identities_checked=True,missingness_checked=True,artifact='states.parquet',artifact_sha256=M.sha(artifact))
    M.write(inbox/'core.json',r);return bundle,inbox,r,d


def test_missing_adapter_never_reads_archive_as_live(tmp_path):
    c,p=config(tmp_path);bundle=tmp_path/'bundle';M.freeze(c,p,bundle,2027,tmp_path)
    r=M.check(bundle,tmp_path/'absent','2026-10-22',tmp_path/'report')
    assert r['status']=='blocked' and r['families'][0]['reason']=='daily adapter receipt missing'
    assert not r['states_updated_by_this_command']


def test_valid_receipt_preserves_zero_and_missing(tmp_path):
    bundle,inbox,r,d=fixture_receipt(tmp_path);out=M.check(bundle,inbox,'2026-10-22',tmp_path/'report')
    assert out['status']=='passed' and out['families'][0]['missing']==1 and out['families'][0]['players']==2
    pd.testing.assert_frame_equal(pd.read_parquet(inbox/'states.parquet'),d)


@pytest.mark.parametrize('change',[{'as_of':'2026-09-01'},{'season':2026},{'parameter_bundle_sha256':'wrong'},{'source_status':'delayed'},{'completed_game_census_checked':False}])
def test_stale_wrong_recipe_or_incomplete_feed_is_blocked(tmp_path,change):
    bundle,inbox,r,d=fixture_receipt(tmp_path);r.update(change);M.write(inbox/'core.json',r)
    assert M.check(bundle,inbox,'2026-10-22',tmp_path/'report')['status']=='blocked'


@pytest.mark.parametrize('kind',['future','duplicate','missing_reason','nonfinite','missing_field','missing_player_field','bad_id'])
def test_bad_state_rows_fail_before_publication(tmp_path,kind):
    bundle,inbox,r,d=fixture_receipt(tmp_path)
    if kind=='future':d.loc[0,'last_observation_date']='2026-10-22'
    elif kind=='duplicate':d=pd.concat([d,d.iloc[:1]],ignore_index=True)
    elif kind=='missing_reason':d.loc[1,'missing_reason']=''
    elif kind=='nonfinite':d.loc[0,'value']=np.inf
    elif kind=='missing_field':d['stat_key']='unexpected'
    elif kind=='missing_player_field':d=pd.concat([d,pd.DataFrame([dict(player_id=3,stat_key='other',value=1.,last_observation_date='2026-10-21',missing_reason='')])],ignore_index=True)
    else:
        d['player_id']=d.player_id.astype(float);d.loc[0,'player_id']=1.5
    d.to_parquet(inbox/'states.parquet',index=False);r['artifact_sha256']=M.sha(inbox/'states.parquet');M.write(inbox/'core.json',r)
    assert M.check(bundle,inbox,'2026-10-22',tmp_path/'report')['status']=='blocked'

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/build_fresh_bam_features.py'
spec=importlib.util.spec_from_file_location('fresh_bam_features',PATH)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def fixture():
    native=pd.DataFrame(dict(year=[2018]*4,nba_id_off=[1,2,3,4],nba_id_def=[10,10,20,20],
        MATCHUP_FGA=[10.,10.,10.,10.],MATCHUP_TOV=[2.,0.,0.,0.],MATCHUP_FTA=[0.,0.,0.,0.],
        PLAYER_PTS=[11.,2.,8.,8.],PARTIAL_POSS=[100.,50.,100.,100.],
        pair_source_counts_valid=True,pair_identity_verified=True))
    lines=pd.DataFrame(dict(season=[2018]*4,season_type=['rs']*4,game_id=['a']*4,player_id=[1,2,3,4],
        pts=[20.,10.,10.,10.],fga=[20.,10.,10.,10.],fta=[0.]*4))
    pos=pd.DataFrame(dict(year=[2018,2018,2018],nba_id=[1,3,4],position=['PG','SF','C'],
        earliest_use_date=pd.to_datetime(['2018-04-12']*3)))
    # Entire population, including player99 who never occurs in matchup pairs.
    cohort=pd.DataFrame(dict(year=[2018]*3,nba_id_off=[1,3,99],Offensive_Score=[2.,1.,9.]))
    metadata={2018:dict(median=2.,fifth_percentile=1.1,cohort_players=3)}
    ends=pd.Series({2018:pd.Timestamp('2018-04-11')})
    return native,M.actual_ts(lines),cohort,metadata,pos,ends


def test_fresh_native_formula_uses_actual_ts_tov_nonlinear_fullcohort_load():
    pairs=M.dated_pairs(*fixture())
    row=pairs.loc[pairs.nba_id_off.eq(1)].iloc[0]
    expected=-10*(11/(2*(10+1.05*2))-.5)*2*((10+1.05*2)/20)*(1+.3025)
    np.testing.assert_allclose(row.dated_bam_pair,expected,atol=1e-12)
    assert row.literal_adjusted_load==2 and row.all_defender_fga==20
    assert row.dated_boost==.3025
    unknown=pairs.loc[pairs.nba_id_off.eq(2)].iloc[0]
    assert np.isnan(unknown.dated_bam_pair) and not unknown.pair_verified


def test_missing_creation_player_uses_fullcohort_fifth_not_median_or_zero():
    native,offensive,cohort,metadata,pos,ends=fixture()
    native.loc[native.nba_id_off.eq(4),'PLAYER_PTS']=1.
    d=M.dated_pairs(native,offensive,cohort,metadata,pos,ends)
    row=d.loc[d.nba_id_off.eq(4)].iloc[0]
    assert not row.creation_load_observed
    assert row.Offensive_Score==1.1 and row.literal_adjusted_load==1.1
    np.testing.assert_allclose(row.dated_bam_pair,-10*(.05-.5)*1.1*(10/20)*1.125)


def test_actual_ts_uses_all_rs_counts_rounds_three_excludes_ps_and_future():
    d=pd.DataFrame(dict(season=[2018,2018,2018,2026],season_type=['rs','rs','ps','rs'],
        game_id=['a','b','c','d'],player_id=[1]*4,pts=[1.,2.,999.,999.],fga=[1.,2.,1.,1.],fta=[1.,2.,0.,0.]))
    result=M.actual_ts(d)
    assert len(result)==1
    assert result.source_ts.iloc[0]==round(3/(2*(3+.44*3)),3)
    assert result.actual_pts.iloc[0]==3
    pd.testing.assert_frame_equal(result,M.actual_ts(d.iloc[:3]))
    with pytest.raises(ValueError,match='Ambiguous'):
        M.actual_ts(pd.concat([d,d.iloc[[0]]]))


def test_source_year_positions_no_bpm_knowledge_floor_or_later_position():
    args=list(fixture());later=args[4].copy();later.year=2025;later.position='C'
    a=M.dated_pairs(*args)
    args[4]=pd.concat([args[4],later]);b=M.dated_pairs(*args)
    pd.testing.assert_frame_equal(a,b)
    assert a.loc[a.nba_id_off.eq(1),'pair_verified'].all()


def test_missing_actual_ts_and_invalid_identity_counts_remain_missing():
    args=list(fixture());args[1]=args[1].loc[args[1].nba_id_off.ne(1)]
    args[0].loc[args[0].nba_id_off.eq(3),'pair_identity_verified']=False
    args[0].loc[args[0].nba_id_off.eq(4),'pair_source_counts_valid']=False
    pairs=M.dated_pairs(*args)
    assert pairs.dated_bam_pair.isna().all()
    seasons,_=M.H.normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    assert seasons.season_z.isna().all() and seasons.season_exposure.eq(0).all()


def test_weighted_z_then_calendar_decay_then_single_k1000():
    pairs=M.dated_pairs(*fixture());seasons,_=M.H.normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    good=seasons.season_exposure.gt(0)
    np.testing.assert_allclose(np.average(seasons.loc[good,'season_z'],weights=seasons.loc[good,'season_exposure']),0,atol=1e-12)
    f,p=M.H.lagged_snapshots(seasons,['2018-10-01'])
    decay=2**(-(pd.Timestamp('2018-10-01')-pd.Timestamp('2018-04-11')).days/550)
    for row in f.itertuples():
        season=seasons.loc[seasons.player_id.eq(row.player_id)].iloc[0]
        exposure=season.season_exposure*decay
        np.testing.assert_allclose(row.matchup_dated_bam,season.season_z*exposure/(exposure+1000),atol=1e-12)
    assert p.parameter_fit_end.lt(pd.Timestamp('2018-10-01')).all()


def test_source_gate_requires_all_native_years_and_correct_aggregate_hashes(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'YEARS',[2018])
    native=fixture()[0];path=tmp_path/'native_2018_rs_pairs_verified.parquet';native.to_parquet(path,index=False)
    gate=tmp_path/'gate.json';gate.write_text(json.dumps(dict(status='blocked',verified_years=[])))
    _,_,complete=M.load_native(tmp_path,gate);assert not complete
    gate.write_text(json.dumps(dict(status='passed_native_source_gate',verified_years=[2018],artifacts={path.name:M.sha(path)})))
    _,_,complete=M.load_native(tmp_path,gate);assert complete
    gate.write_text(json.dumps(dict(status='passed_native_source_gate',verified_years=[2018],artifacts={path.name:'wrong'})))
    with pytest.raises(ValueError,match='checksum'):
        M.load_native(tmp_path,gate)


def test_full_cohort_creation_sum_retains_literal_team_stints_and_diagnostic_separately():
    rows=pd.DataFrame(dict(year=[2018]*7,play_type=['Isolation','Isolation','Postup','PRBallHandler','Isolation','Postup','Isolation'],
        PLAYER_ID=[1,1,1,1,99,99,100],TEAM_ID=[10,20,10,10,10,10,10],
        PPP=[1.,2.,1.,1.,1.,1.,0.],POSS=[2.,2.,1.,3.,10.,10.,99.],GP=[20.,60.,80.,80.,80.,80.,80.]))
    cohort,meta=M.creation_loads(rows)
    one=cohort.loc[cohort.nba_id_off.eq(1)].iloc[0]
    assert one.Offensive_Score==10. # 2+4+1+3, no GP reweighting.
    assert one.gp_weighted_creation_load_diagnostic==7.5 # (2*20+4*60)/80+1+3
    assert one.source_teams==2
    assert set(cohort.nba_id_off)=={1,99} # zero PPP excluded.
    assert meta[2018]['median']==15. and meta[2018]['fifth_percentile']==10.5
    assert meta[2018]['multi_team_players']==1
    with pytest.raises(ValueError,match='Duplicate'):
        M.creation_loads(pd.concat([rows,rows.iloc[[0]]]))


def test_raw_synergy_request_response_year_and_complete_type_checks(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'YEARS',[2018]);receipts=[]
    for play in M.PLAYTYPES:
        raw=tmp_path/f'{play}.json';meta=tmp_path/f'{play}.meta.json'
        raw.write_text(json.dumps(dict(resultSets=[dict(headers=['PLAYER_ID','PPP','POSS','SEASON_ID','PLAY_TYPE','TYPE_GROUPING'],
            rowSet=[[1,1.,2.,'22017',play,'offensive']])])))
        params=dict(LeagueID='00',PerMode='PerGame',PlayerOrTeam='P',SeasonYear='2017-18',SeasonType='Regular Season',PlayType=play,TypeGrouping='offensive')
        meta.write_text(json.dumps(dict(parameters=params,sha256=M.sha(raw))))
        receipts.append(dict(year=2018,play_type=play,raw=str(raw),metadata=str(meta),sha256=M.sha(raw),metadata_sha256=M.sha(meta)))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(status='completed_official_RS_inputs',receipts=receipts)))
    rows,_=M.load_synergy(tmp_path);assert len(rows)==3
    bad=json.loads(meta.read_text());bad['parameters']['PerMode']='Totals';meta.write_text(json.dumps(bad))
    receipts[-1]['metadata_sha256']=M.sha(meta)
    manifest.write_text(json.dumps(dict(status='completed_official_RS_inputs',receipts=receipts)))
    with pytest.raises(ValueError,match='request-certified'):
        M.load_synergy(tmp_path)
    manifest.write_text(json.dumps(dict(status='completed_official_RS_inputs',receipts=receipts[:2])))
    with pytest.raises(ValueError,match='missing complete'):
        M.load_synergy(tmp_path)


def test_full_cohort_medians_do_not_change_when_matchup_opponents_are_filtered():
    rows=pd.DataFrame(dict(year=[2018]*3,play_type=['Isolation']*3,PLAYER_ID=[1,2,99],
        TEAM_ID=[10]*3,PPP=[1.]*3,POSS=[1.,3.,20.],GP=[80.]*3))
    cohort,meta=M.creation_loads(rows)
    args=list(fixture());args[2]=cohort;args[3]=meta
    d=M.dated_pairs(*args)
    args[0]=args[0].iloc[[0]]
    subset=M.dated_pairs(*args)
    assert d.median_fallback_load.eq(3.).all() and subset.median_fallback_load.eq(3.).all()
    assert d.fifth_percentile_load.eq(1.2).all()


def test_future_source_year_perturbation_cannot_change_earlier_cut():
    pairs=M.dated_pairs(*fixture());seasons,_=M.H.normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    later=seasons.copy();later.year=2025;later.source_rs_end=pd.Timestamp('2025-04-13')
    later.position_available_at=pd.Timestamp('2025-04-14');later.season_z=9999.
    a,_=M.H.lagged_snapshots(seasons,['2018-10-01'])
    b,_=M.H.lagged_snapshots(pd.concat([seasons,later]),['2018-10-01'])
    pd.testing.assert_frame_equal(a,b)


def test_independent_synergy_receipt_hashes_are_bound(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'YEARS',[2018])
    path=tmp_path/'source.json';data=tmp_path/'response.json';data.write_text('{}')
    assert not M.synergy_admission(path)[0]
    path.write_text(json.dumps(dict(status='passed_fresh_official_synergy_source_audit',years=[2018],
        play_types=list(M.PLAYTYPES),source_hashes={str(data):M.sha(data)})))
    assert M.synergy_admission(path)[0]
    data.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checksum'):
        M.synergy_admission(path)

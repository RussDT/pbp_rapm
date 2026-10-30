import numpy as np
import pandas as pd
from nba_pipeline.scripts.playoff_availability import (
    count_participation, predict_from_history, add_participation_adjustment,
    career_participation_summary,
)


def fixture():
    rows=[]
    for year,n in [(2021,4),(2022,0),(2023,2),(2024,3)]:
        for pid,played in [(1,n),(2,4)]:
            rows.append(dict(season=year,player_id=pid,playoff_games_played=played,
                team_playoff_games=4,playoff_participation_actual=played/4,
                playoff_participation_source='actual'))
    evidence=pd.DataFrame(rows)
    seasons=evidence[['season','player_id']].assign(title_lift_over_contract_v2=.2,
        title_lift_over_contract=-.1,playoff_adjusted_title_lift_over_contract_v2=.3)
    return seasons,evidence


def test_raw_appearances_dedupe_and_keep_brief_lineup_only_players():
    games=pd.DataFrame([dict(game_id='0042400101',home_team_abbr='MIL',away_team_abbr='IND'),
                        dict(game_id='0042400102',home_team_abbr='MIL',away_team_abbr='IND')])
    rows=[]
    for gid in [42400101,42400101,42400102]:
        r={'game_id':gid}
        r.update({f'home_player{i}':i for i in range(1,6)})
        r.update({f'away_player{i}':10+i for i in range(1,6)})
        rows.append(r)
    result=count_participation(pd.DataFrame(rows),games)
    assert result.set_index('player_id').loc[1,'playoff_games_played']==2
    assert result.set_index('player_id').loc[11,'team_abbr']=='IND'


def test_prediction_uses_game_weighted_other_seasons_and_league_fallback():
    training=pd.DataFrame(dict(player_id=[1,1,2],playoff_games_played=[4,0,2],team_playoff_games=[4,16,2]))
    pred,history,league=predict_from_history(pd.DataFrame({'player_id':[1,99]}),training,4)
    assert np.isclose(league,6/22)
    assert np.isclose(pred[0],(4+4*(6/22))/24)
    assert pred[1]==league
    assert history.tolist()==[20,0]


def test_outer_season_outcomes_cannot_change_own_prediction_or_selected_k():
    seasons,evidence=fixture()
    first,_=add_participation_adjustment(seasons,evidence)
    evidence.loc[evidence.season.eq(2024),'playoff_games_played']=0
    evidence.loc[evidence.season.eq(2024),'playoff_participation_actual']=0
    second,_=add_participation_adjustment(seasons,evidence)
    cols=['playoff_participation_estimated','playoff_participation_prior_games']
    pd.testing.assert_frame_equal(first.loc[first.season.eq(2024),cols],second.loc[second.season.eq(2024),cols])


def test_actual_zero_nonplayoff_estimate_unknown_missing_and_career_season_sum():
    seasons,evidence=fixture()
    extra=seasons.iloc[:2].copy(); extra['season']=2025
    seasons=pd.concat([seasons,extra],ignore_index=True)
    other=evidence.iloc[:2].copy(); other['season']=2025
    other[['playoff_games_played','team_playoff_games','playoff_participation_actual']]=np.nan
    other['playoff_participation_source']=['estimated_nonplayoff','unresolved_roster']
    evidence=pd.concat([evidence,other],ignore_index=True)
    out,_=add_participation_adjustment(seasons,evidence)
    assert out.loc[out.season.eq(2022)&out.player_id.eq(1),'participation_scaled_title_lift_v2'].iloc[0]==0
    row=out.loc[out.season.eq(2025)&out.player_id.eq(1)].iloc[0]
    assert row.playoff_participation_used==row.playoff_participation_estimated
    assert np.isnan(out.loc[out.season.eq(2025)&out.player_id.eq(2),'participation_scaled_title_lift_v2'].iloc[0])
    career=career_participation_summary(out).set_index('player_id')
    assert np.isclose(career.loc[1,'participation_scaled_title_lift_v2_championships'],out.loc[out.player_id.eq(1),'participation_scaled_title_lift_v2'].sum())


def test_uncertified_zero_roster_proxies_do_not_train_the_model():
    seasons,evidence=fixture()
    evidence.loc[evidence.season.eq(2022),'playoff_participation_source']='inferred_roster'
    first,_=add_participation_adjustment(seasons,evidence)
    evidence.loc[evidence.season.eq(2022),'playoff_games_played']=4
    evidence.loc[evidence.season.eq(2022),'playoff_participation_actual']=1
    second,_=add_participation_adjustment(seasons,evidence)
    pd.testing.assert_series_equal(first.playoff_participation_estimated,second.playoff_participation_estimated)


def test_verified_entire_postseason_absentee_and_team_assignment_proxy(tmp_path):
    import json
    from nba_pipeline.scripts.playoff_availability import build_participation_evidence
    games=pd.DataFrame([dict(season=2024,game_id='0042400101',home_team_abbr='MIL',away_team_abbr='IND')])
    row={'game_id':42400101}
    row.update({f'home_player{i}':i for i in range(1,6)})
    row.update({f'away_player{i}':10+i for i in range(1,6)})
    pd.DataFrame([row]).to_parquet(tmp_path/'NBA24_PS.parquet')
    cert=tmp_path/'roster.json'; cert.write_text(json.dumps([{'season':2024,'player_id':99,'team_abbr':'MIL','source_url':'https://example.test/report'}]))
    seasons=pd.DataFrame(dict(season=[2024]*4,player_id=[99,88,77,66],team_abbr=['MIL','MIL','DET','BOS']))
    regular=pd.DataFrame(dict(season=[2024]*3,player_id=[88,77,66],team_abbr=['MIL','DET','LAL'],last_game_date=['2024-04-12']*3,player_exposures=[50]*3))
    result=build_participation_evidence(seasons,games,regular,tmp_path,cert).set_index('player_id')
    assert result.loc[99,'playoff_participation_actual']==0
    assert result.loc[99,'playoff_participation_source']=='actual'
    assert result.loc[88,'playoff_participation_source']=='inferred_roster'
    assert result.loc[77,'playoff_participation_source']=='estimated_nonplayoff'
    assert result.loc[66,'playoff_participation_source']=='unresolved_roster'


def test_app_patch_preserves_other_model_rows_and_metadata(tmp_path):
    import gzip,json
    from nba_pipeline.scripts.refresh_playoff_participation import patch_app_participation
    from nba_pipeline.scripts.playoff_availability import CAREER_FIELDS,SEASON_FIELDS
    (tmp_path/'players').mkdir()
    def put(path,payload):
        with gzip.open(path,'wt') as f:json.dump(payload,f)
    untouched={'player_id':9,'season':1999,'darko_title_lift_over_contract':.9}
    existing={'player_id':1,'season':2024,'darko_title_lift_over_contract':.7}
    put(tmp_path/'players.json.gz',[{'player_id':1,'darko_contract_seasons_covered':12}])
    put(tmp_path/'season-players.json.gz',[existing.copy(),untouched.copy()])
    put(tmp_path/'players/01.json.gz',{'seasons':[existing.copy()], 'teammates':[{'sentinel':'preserve'}]})
    (tmp_path/'metadata.json').write_text(json.dumps({'model':{'darko':'keep'}}))
    career=pd.DataFrame([dict(player_id=1,**{c:0 for c in CAREER_FIELDS})])
    seasons=pd.DataFrame([dict(player_id=1,season=2024,**{c:None for c in SEASON_FIELDS})])
    patch_app_participation(tmp_path,career,seasons,{'sentinel':'new'})
    with gzip.open(tmp_path/'season-players.json.gz','rt') as f:rows=json.load(f)
    assert rows[0]['darko_title_lift_over_contract']==.7
    assert rows[1]==untouched
    with gzip.open(tmp_path/'players/01.json.gz','rt') as f:payload=json.load(f)
    assert payload['teammates']==[{'sentinel':'preserve'}]
    assert json.loads((tmp_path/'metadata.json').read_text())['model']['darko']=='keep'

"""Complete current NBA team accounting diagnostic, without altering player ratings.

Uses all player role rows, season-rate allocation by team possession shares,
all-player league centering, and a separate league-average team-turnover baseline.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from nba_pipeline.scripts.build_nba_rztov_box_backcast import regular_season_rows
OUT=ROOT/'nba_pipeline/results/rztov_team_page_accounting'

def main():
    year=2026
    base=ROOT/'nba_pipeline/results/nba_rztov_box_backcast'
    roles=pd.read_csv(base/'nba_rztov_box_role_backcast_1997_2026.csv')
    roles=roles[roles.season.eq(year)].rename(columns={'player_id':'pid','estimated_self_created_tsa100':'self','estimated_assisted_tsa100':'assisted','estimated_potential_assists100':'potast'})
    if not roles.tsa_source.eq('same_sample_RS_raw_PBP_FGA_plus_multishot_FT_trips').all():
        raise ValueError('Rebuild role board on the same RS PBP sample before team aggregation')
    price=pd.read_csv(base/'historical_box_price_path_1997_2026.csv')
    # Use the exact current window's price fields from the ledger, already exposed.
    ledger=pd.read_csv(base/'nba_box_estimated_rztov_1997_2026_diagnostic.csv')
    p=price[price.window_hi.eq(year)].iloc[0].rename({'self':'be_self','ast':'be_ast','potast':'be_potast'})
    roles['sc_raw']=roles.self*p.be_self+roles.assisted*p.be_ast
    roles['bp_raw']=roles.potast*p.be_potast
    raw=pd.read_parquet(ROOT/'nba_pipeline/raw_data/NBA26.parquet')
    raw=regular_season_rows(raw.drop_duplicates()).copy()
    raw['game_id']=raw.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    raw['pid']=pd.to_numeric(raw.player1_id,errors='coerce').fillna(0).astype('int64')
    raw['team']=pd.to_numeric(raw.player1_team_id,errors='coerce').fillna(0).astype('int64')
    # Team turnovers encode the franchise in player1_id, with player1_team_id=0.
    raw.loc[raw.pid.gt(1000000000)&raw.team.eq(0),'team']=raw.pid
    mapping=raw[raw.pid.between(1,999999999)&raw.team.gt(1000000000)][['game_id','pid','team']].drop_duplicates()
    # Add players who never authored an event using observed home/away lineups.
    extra=[]
    for side in ['home','away']:
        cols=[f'{side}_player{i}' for i in range(1,6)]
        membership=raw[cols].eq(raw.pid,axis=0).any(axis=1)
        side_votes=raw.loc[membership & raw.pid.between(1,999999999)&raw.team.gt(1000000000),['game_id','team']].value_counts().rename('n').reset_index()
        side_team=side_votes.sort_values('n',ascending=False).drop_duplicates('game_id')[['game_id','team']]
        roster=raw[['game_id']+cols].melt(id_vars='game_id',value_name='pid').drop(columns='variable').drop_duplicates()
        roster['pid']=pd.to_numeric(roster.pid,errors='coerce').fillna(0).astype('int64')
        roster=roster[roster.pid.between(1,999999999)].merge(side_team,on='game_id',validate='many_to_one')
        extra.append(roster)
    expanded=pd.concat(extra).drop_duplicates()
    expanded=expanded[~expanded.duplicated(['game_id','pid'],keep=False)]
    missing=expanded.merge(mapping[['game_id','pid']],on=['game_id','pid'],how='left',indicator=True)
    mapping=pd.concat([mapping,missing[missing._merge.eq('left_only')][['game_id','pid','team']]]).drop_duplicates()
    if mapping.duplicated(['game_id','pid']).any(): raise ValueError('Ambiguous player-game team')
    poss=pd.read_parquet(ROOT/'nba_pipeline/processed/RAPM26.parquet',columns=['game_id']+[f'O{i}' for i in range(1,6)])
    poss=regular_season_rows(poss)
    poss['game_id']=poss.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    poss['possession_id']=np.arange(len(poss))
    slots=poss.melt(id_vars=['game_id','possession_id'],value_name='pid').drop(columns='variable')
    slots['pid']=pd.to_numeric(slots.pid,errors='coerce').fillna(0).astype('int64')
    slots=slots.merge(mapping,on=['game_id','pid'],how='left',validate='many_to_one')
    votes=slots.groupby(['possession_id','team']).size().rename('votes').reset_index()
    winner=votes.sort_values('votes',ascending=False).drop_duplicates('possession_id')
    if len(winner)!=len(poss) or winner.votes.lt(3).any(): raise ValueError('Unresolved offensive team')
    if votes.groupby('possession_id').team.nunique().gt(1).any(): raise ValueError('Conflicting offensive teams')
    slots=slots.drop(columns='team').merge(winner[['possession_id','team']],on='possession_id',validate='many_to_one')
    exposure=slots.groupby(['pid','team']).size().rename('poss').reset_index()
    exposure['share']=exposure.poss/exposure.groupby('pid').poss.transform('sum')
    teamposs=winner.groupby('team').size().rename('team_possessions')
    joined=exposure.merge(roles,on='pid',how='left',validate='many_to_one')
    if joined.off_poss.isna().any(): raise ValueError('Missing role for player-slot')
    if not np.allclose(joined.groupby('pid').poss.sum().sort_index(), roles.set_index('pid').off_poss.reindex(joined.pid.unique()).sort_index()): raise ValueError('Player exposure mismatch')
    desc=raw[['home_description','visitor_description','neutral_description']].fillna('').astype(str).agg(' '.join,axis=1)
    et=pd.to_numeric(raw.event_type,errors='coerce')
    fga=et.isin([1,2])
    trip=et.eq(3)&desc.str.contains(r'\b1 of [23]\b',case=False,regex=True)&~desc.str.contains('technical',case=False)
    shots=raw.loc[fga|trip,['pid','team']].copy()
    if shots.team.le(1000000000).any(): raise ValueError('Unassigned shooting opportunity team')
    stint_tsa=shots.groupby(['pid','team']).size().rename('team_tsa_count').reset_index()
    joined=joined.merge(stint_tsa,on=['pid','team'],how='outer',validate='one_to_one')
    if joined.poss.isna().any(): raise ValueError('Shot without resolved player-team exposure')
    joined['team_tsa_count']=joined.team_tsa_count.fillna(0)
    if not np.isclose(joined.team_tsa_count.sum(),roles.canonical_tsa_count.sum()): raise AssertionError('TSA source closure')
    joined['self']=100*joined.team_tsa_count*(1-joined.estimated_assisted_share)/joined.poss
    joined['assisted']=100*joined.team_tsa_count*joined.estimated_assisted_share/joined.poss
    joined['sc_raw']=joined.self*p.be_self+joined.assisted*p.be_ast
    mask=pd.to_numeric(raw.event_type,errors='coerce').eq(5)&~desc.str.contains('no turnover',case=False)
    events=raw.loc[mask,['pid','team']].copy()
    if events.team.le(1000000000).any(): raise ValueError('Unassigned turnover team')
    events['player_tov']=events.pid.between(1,999999999).astype(int)
    events['team_tov']=1-events.player_tov
    events['bp']=desc[mask].str.contains('bad pass',case=False).astype(int)*events.player_tov
    events['sc']=events.player_tov-events.bp
    events['below500_tov']=(~events.pid.isin(ledger[ledger.year.eq(year)].nba_id)).astype(int)*events.player_tov
    total=events.groupby('team')[['player_tov','team_tov','sc','bp','below500_tov']].sum().join(teamposs)
    sc_center=(events.sc.sum()-(roles.sc_raw*roles.off_poss/100).sum())/(roles.off_poss.sum()/100)
    bp_center=(events.bp.sum()-(roles.bp_raw*roles.off_poss/100).sum())/(roles.off_poss.sum()/100)
    joined['sc_expected']=(joined.sc_raw+sc_center)*joined.poss/100
    joined['bp_expected']=(joined.bp_raw+bp_center)*joined.poss/100
    total=total.join(joined.groupby('team')[['sc_expected','bp_expected']].sum())
    total=total.join(joined.groupby('team').team_tsa_count.sum())
    total['tsa_per100']=100*total.team_tsa_count/total.team_possessions
    total['player_expected']=total.sc_expected+total.bp_expected
    team_tov_rate=100*total.team_tov.sum()/total.team_possessions.sum()
    total['team_expected']=team_tov_rate*total.team_possessions/100
    total['actual_total']=total.player_tov+total.team_tov
    total['expected_total']=total.player_expected+total.team_expected
    total['fewer_than_expected']=total.expected_total-total.actual_total
    for c in ['player_tov','team_tov','player_expected','team_expected','actual_total','expected_total','fewer_than_expected']:
        total[c+'_per100']=100*total[c]/total.team_possessions
    if not np.isclose(total.actual_total.sum(),len(events)): raise AssertionError('Count closure')
    if not np.isclose(total.expected_total.sum(),total.actual_total.sum()): raise AssertionError('League baseline closure')
    OUT.mkdir(parents=True,exist_ok=True)
    total.reset_index().to_csv(OUT/'nba_2026_teams.csv',index=False)
    joined.to_csv(OUT/'nba_2026_player_team_allocations.csv',index=False)
    manifest={'status':'team_page_diagnostic_not_published','season':year,'all_player_rows':len(roles),'team_count':len(total),'possessions':int(total.team_possessions.sum()),'actual_turnovers':int(total.actual_total.sum()),'player_turnovers':int(total.player_tov.sum()),'team_turnovers':int(total.team_tov.sum()),'team_turnover_baseline_per100':team_tov_rate,'sc_center':sc_center,'bp_center':bp_center,'pricing':'incumbent 2022-2026 prices; all-player league recentering','team_baseline':'league-average team/unassigned turnovers per resolved possession, separate from workload','allocation':'actual same-sample player-team TSA split at estimated season assisted share; potential assists use estimated season rate on exact team exposure','reference':'local PBP and resolved possessions, not independent official box-score certification'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(total.loc[[1610612738,1610612757]].to_string());print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

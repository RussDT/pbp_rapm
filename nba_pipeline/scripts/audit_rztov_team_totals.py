"""Read-only NBA/WNBA turnover accounting and team expectation audit.

Team allowances allocate season-level player rates by PBP team possession shares.
They are diagnostics, not refitted team-stint role estimates or official box totals.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'nba_pipeline/results/rztov_team_audit'

def key(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)

def audit(league, year, board):
    if league == 'NBA':
        path = ROOT / f'nba_pipeline/raw_data/NBA{year%100:02d}.parquet'
        wanted = ['game_id','event_num','event_type','player1_id','player1_team_id','home_description','visitor_description','neutral_description']
        raw = pd.read_parquet(path, columns=[c for c in wanted if c in pq.read_schema(path).names]).drop_duplicates()
        poss = pd.read_parquet(ROOT / f'nba_pipeline/processed/RAPM{year%100:02d}.parquet',columns=['game_id']+[f'O{i}' for i in range(1,6)])
    else:
        raw = pd.read_csv(ROOT / f'wnba_test/GabrielNormalizedClean/WNBA{year%100:02d}.csv',low_memory=False).drop_duplicates()
        poss = pd.read_csv(ROOT / f'wnba_test/Processed/RAPM{year%100:02d}.csv',usecols=['game_id']+[f'O{i}' for i in range(1,6)])
    raw['game_id'] = key(raw.game_id); poss['game_id'] = key(poss.game_id)
    raw['pid'] = pd.to_numeric(raw.player1_id,errors='coerce').fillna(0).astype('int64')
    raw['team'] = pd.to_numeric(raw.player1_team_id,errors='coerce').fillna(0).astype('int64')
    valid = raw.pid.between(1,999999999)
    mapping = raw.loc[valid & raw.team.gt(1000000000),['game_id','pid','team']].drop_duplicates()
    ambiguous = mapping.duplicated(['game_id','pid'],keep=False)
    mapping = mapping[~ambiguous]
    slots = poss.melt(id_vars='game_id',value_name='pid').drop(columns='variable')
    slots['pid'] = pd.to_numeric(slots.pid,errors='coerce').fillna(0).astype('int64')
    slots = slots.merge(mapping,on=['game_id','pid'],how='left',validate='many_to_one')
    slots['team'] = slots.team.fillna(0).astype('int64')
    exposure = slots.groupby(['pid','team']).size().rename('team_poss').reset_index()
    exposure['share'] = exposure.team_poss / exposure.groupby('pid').team_poss.transform('sum')
    desc = raw[[c for c in ['home_description','visitor_description','neutral_description'] if c in raw]].fillna('').astype(str).agg(' '.join,axis=1)
    tov = pd.to_numeric(raw.event_type,errors='coerce').eq(5) & ~desc.str.contains('no turnover',case=False)
    events = raw.loc[tov,['pid','team']].copy()
    events['player_tov'] = events.pid.between(1,999999999).astype(int)
    events['team_unassigned_tov'] = 1-events.player_tov
    events['badpass'] = (desc[tov].str.contains('bad pass',case=False) & events.player_tov.eq(1)).astype(int)
    events['scoring'] = events.player_tov-events.badpass
    events['total_tov'] = 1
    totals = events.groupby('team')[['total_tov','player_tov','team_unassigned_tov','badpass','scoring']].sum()
    b=board.copy()
    joined = exposure.merge(b,on='pid',how='left',validate='many_to_one')
    joined['allocated_poss'] = joined.off_poss * joined.share
    for target,rate in [('board_actual','actual'),('allowance_count','allowance'),('sc_allow_count','sc_allow'),('bp_allow_count','bp_allow'),('self_count','self'),('assisted_count','assisted'),('potast_count','potast')]:
        joined[target] = joined[rate]*joined.allocated_poss/100
    cols=['board_actual','allowance_count','sc_allow_count','bp_allow_count','self_count','assisted_count','potast_count']
    totals=totals.join(joined.groupby('team')[cols].sum(),how='outer').fillna(0)
    if 'role_error100' in joined:
        common = joined[joined.role_error100.notna()].copy()
        common['role_error_count'] = common.role_error100 * common.allocated_poss / 100
        common['common_tsa'] = (common.self + common.assisted) * common.allocated_poss / 100
        common['common_allowance'] = common.allowance * common.allocated_poss / 100
        role = common.groupby('team')[['role_error_count','common_tsa','common_allowance','allocated_poss']].sum()
        role['role_error_count_centered'] = role.role_error_count - role.allocated_poss * role.role_error_count.sum()/role.allocated_poss.sum()
        role['role_error_pct_allowance'] = 100*role.role_error_count_centered/role.common_allowance
        totals = totals.join(role)

    covered_events=events[events.pid.isin(b.pid)].groupby('team').player_tov.sum()
    totals['covered_event_tov']=covered_events
    totals['covered_event_tov']=totals.covered_event_tov.fillna(0)
    totals['unmodeled_player_tov']=totals.player_tov-totals.covered_event_tov
    totals['team_excess_vs_allowance']=totals.covered_event_tov-totals.allowance_count
    totals['allowance_per100_tsa']=100*totals.allowance_count/(totals.self_count+totals.assisted_count)
    totals['self_share']=totals.self_count/(totals.self_count+totals.assisted_count)
    totals['potast_per_tsa']=totals.potast_count/(totals.self_count+totals.assisted_count)
    totals['league']=league;totals['season']=year
    source_player=events.groupby('pid').player_tov.sum()
    b['source_tov']=b.pid.map(source_player).fillna(0)
    b['board_tov']=b.actual*b.off_poss/100
    b['source_count_gap']=b.board_tov-b.source_tov
    b['league']=league;b['season']=year
    summary={'league':league,'season':year,'board_rows':len(b),'source_tov':int(tov.sum()),'source_player_tov':int(events.player_tov.sum()),'source_team_tov':int(events.team_unassigned_tov.sum()),'unmodeled_player_tov':int(events.loc[~events.pid.isin(b.pid),'player_tov'].sum()),'board_actual':float(b.board_tov.sum()),'board_allowance':float((b.allowance*b.off_poss/100).sum()),'max_player_source_gap':float(b.source_count_gap.abs().max()),'source_gap_abs_sum':float(b.source_count_gap.abs().sum()),'unknown_team_slots':int(slots.team.eq(0).sum()),'board_players_without_slots':int((~b.pid.isin(exposure.pid)).sum()),'traded_players':int(exposure[exposure.team.ne(0)].groupby('pid').team.nunique().gt(1).sum()),'lane_gap':float((b.actual-b.sc-b.bp).abs().max())}
    print(summary,flush=True)
    return totals.reset_index(),b,summary

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    n=pd.read_csv(ROOT/'nba_pipeline/results/nba_rztov_box_backcast/nba_box_estimated_rztov_1997_2026_diagnostic.csv').rename(columns={'nba_id':'pid','year':'season','OffPoss':'off_poss','act':'actual','v_self':'self','v_ast':'assisted','v_potast':'potast','sc_100':'sc','bp_100':'bp'})
    r=pd.read_csv(ROOT/'nba_pipeline/results/nba_rztov_box_backcast/nba_rztov_box_role_backcast_1997_2026.csv').rename(columns={'player_id':'pid'})
    n=n.merge(r[['pid','season','actual_assist_opp_share','actual_potential_assists100']],on=['pid','season'],validate='one_to_one')
    n['role_error100']=(n.be_self-n.be_ast)*(n.self-(1-n.actual_assist_opp_share)*n.tsa100)+n.be_potast*(n.potast-n.actual_potential_assists100)
    w=pd.read_csv(ROOT/'wnba_test/Results/wnba_rztov/wnba_box_estimated_rztov_2022_2026.csv').rename(columns={'player_id':'pid','tov100_raw':'actual','turnover_allowance100':'allowance','scoring_allowance100':'sc_allow','badpass_allowance100':'bp_allow','estimated_self_created_tsa100':'self','estimated_assisted_tsa100':'assisted','estimated_potential_assists100':'potast','sc_tov100_raw':'sc','bp_tov100_raw':'bp'})
    rtss=pd.read_csv(ROOT/'wnba_test/Results/rtss_prior_research/player_season_features.csv').rename(columns={'player_id':'pid'})
    w=w.merge(rtss[['pid','season','rtss_shot_assist_opp_share','rtss_potential_assist_rate100']],on=['pid','season'],how='left',validate='one_to_one')
    prices=json.loads((ROOT/'wnba_test/Results/wnba_rztov/manifest.json').read_text())['break_evens']
    w['role_error100']=(prices['self']-prices['assisted'])*(w.self-(1-w.rtss_shot_assist_opp_share)*w.tsa100_raw)+prices['potential_assist']*(w.potast-w.rtss_potential_assist_rate100)
    teams=[];players=[];summaries=[]
    for league,df in [('NBA',n),('WNBA',w)]:
        for year,b in df.groupby('season'):
            t,p,s=audit(league,int(year),b);teams.append(t);players.append(p);summaries.append(s)
    team_rows=pd.concat(teams)
    team_rows.to_csv(OUT/'team_seasons.csv',index=False)
    for league in ['NBA','WNBA']:
        matched=team_rows[team_rows.league.eq(league)&team_rows.common_allowance.gt(0)]
        matched.to_csv(OUT/f'{league.lower()}_team_role_estimation_effect.csv',index=False)
    pd.concat(players).to_csv(OUT/'player_source_checks.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'season_summary.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'status':'diagnostic','team_allocation':'season rates allocated by observed PBP possession shares; not team-stint role estimates','reference':'local PBP events, not independent official box totals','seasons':len(summaries)},indent=2))
if __name__=='__main__': main()

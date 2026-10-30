"""Possession evidence before player attribution: one game, six on-court factors.

No historical player ratings, fitted box coefficients, or team-remainder
redistribution enters the event ledger. On-court results are shared observations,
not additive individual credit. dTOV actor prices are accounting allocations,
not fitted skill coefficients or causal estimates.
"""
import json
import html
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from . import build_player_game_net_points as g
except ImportError:
    import build_player_game_net_points as g

ROOT=g.ROOT
OUT=ROOT/'results/player_game_attribution_audit'
GAME='0042500301'

def split_dtov(is_turnover, price, baseline, actor):
    total=-(-price*int(is_turnover)-baseline)
    direct=price if is_turnover and actor is not None else 0.
    unidentified=price if is_turnover and actor is None else 0.
    return dict(team_points=total,identified_actor_points=direct,unidentified_event_points=unidentified,baseline_points=baseline)

def turnover_lane(first_chance_turnover):
    return 'dTOV' if first_chance_turnover else 'dSC'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=ROOT/'results/player_game_net_points'
    manifest=json.loads((base/'manifest.json').read_text())
    rawpath=ROOT/'raw_data/NBA26_PS.parquet'
    raw=g.normalize_raw(pd.read_parquet(rawpath))
    fc=pd.read_parquet(ROOT/'processed_netcont_lane/FIRST_CHANCE26_PS.parquet')
    sc=pd.read_parquet(ROOT/'processed_netcont_lane/SECOND_CHANCE_CLEAN26_PS.parquet')
    for frame in [fc,sc]:frame['game_id']=g.team.normalize_game_id(frame.game_id)
    rows,price=g.component_rows(fc,sc);baselines=rows[['TS','TOV','SC']].mean()
    np.testing.assert_allclose(price,manifest['turnover_price'])
    meta=g.team.build_game_metadata(raw)
    rows['offense_team']=g.team.assign_offense_teams(rows,meta)
    rows=rows.loc[rows.game_id.eq(GAME)].reset_index(drop=True)
    raw=raw.loc[raw.game_id.eq(GAME)].reset_index(drop=True)
    rows=g.restore_terminal_misses(rows,raw,meta)
    sq=g.team.load_russell_v2_shots(g.team.DEFAULT_RUSSELL_SHOTQUALITY_PATH,rawpath.name)
    shots=g.attach_all_shots(rows,raw,sq)
    m=meta[GAME]
    # Explicit lookup avoids pandas renaming underscore-prefixed tuple fields.
    for _,shot in shots.iterrows():
        roster=m.home_players if rows.iloc[int(shot['_row'])].offense_team==m.home_team else m.away_players
        if int(shot.player1_id) not in roster:raise ValueError('Shot/possession side mismatch')
    threes=shots.loc[shots._zone.eq('three')].groupby(['_row','lane']).residual.sum()
    old=pd.read_parquet(base/'observed_events.parquet')
    old=old.loc[old.game_id.eq(GAME)&old.kind.eq('steal')].copy()
    first_tov=rows.set_index('event_num').Is_Turnover
    if not old.event_num.isin(first_tov.index).all():raise ValueError('Steal has no possession terminal')
    old['accounting_factor']=old.event_num.map(first_tov).astype(bool).map(turnover_lane)
    actor_map=old.set_index('event_num').player_id.to_dict()
    if old.event_num.duplicated().any():raise ValueError('Multiple steal actors')
    boxes=pd.read_parquet(ROOT/'results/player_game_box_net_points/players.parquet')
    boxes=boxes.loc[boxes.game_id.eq(GAME)].copy()
    rawlookup=raw.set_index('event_num');ledger=[];dtov=[];exposure=[]
    for idx,row in rows.iterrows():
        event=rawlookup.loc[row.event_num]
        defense=m.away_team if row.offense_team==m.home_team else m.home_team
        desc=' '.join(str(event.get(c) or '') for c in ['home_description','visitor_description'])
        turnover=bool(row.get('Is_Turnover',0))
        actor=actor_map.get(int(row.event_num)) if turnover else None
        defenders=[int(row[f'D{i}']) for i in range(1,6)]
        if actor is not None and actor not in defenders:raise ValueError('Steal actor absent from possession lineup')
        category='steal' if actor is not None else ('offensive_foul_drawer_unknown' if turnover and ('Offensive Foul' in desc or 'Offensive Charge' in desc) else ('other_turnover_actor_unknown' if turnover else 'no_turnover'))
        split=split_dtov(turnover,price,float(baselines.TOV),actor)
        dtov.append(dict(game_id=GAME,event_num=int(row.event_num),period=int(row.period),clock=row.time_quarter,team=defense,category=category,actor_id=actor,description=desc,**split))
        for side,team,cols,sign in [('o',row.offense_team,'O',1),('d',defense,'D',-1)]:
            pids=[int(row[f'{cols}{i}']) for i in range(1,6)]
            if len(set(pids))!=5:raise ValueError('Nonunique lineup')
            for pid in pids:exposure.append(dict(player_id=pid,side=side,event_num=int(row.event_num)))
            for lane in ['TS','TOV','SC']:
                actual=sign*(float(row[lane])-float(baselines[lane]))
                adjustment=float(threes.get((idx,lane),0)) if side=='d' else 0.
                ledger.append(dict(game_id=GAME,event_num=int(row.event_num),team=team,factor=side+lane,actual_points=actual,opponent_three_adjustment=adjustment,adjusted_points=actual+adjustment,lineup=','.join(map(str,pids))))
    ledger=pd.DataFrame(ledger);dtov=pd.DataFrame(dtov)
    np.testing.assert_allclose(dtov.team_points,dtov.identified_actor_points+dtov.unidentified_event_points+dtov.baseline_points,atol=1e-12)
    oncourt=[]
    for r in ledger.itertuples():
        for pid in r.lineup.split(','):oncourt.append(dict(player_id=int(pid),factor=r.factor,actual_points=r.actual_points,adjusted_points=r.adjusted_points))
    oc=pd.DataFrame(oncourt).groupby(['player_id','factor'])[['actual_points','adjusted_points']].sum().reset_index()
    exp=pd.DataFrame(exposure).groupby(['player_id','side']).size()
    oc['possessions']=[int(exp.loc[(r.player_id,r.factor[0])]) for r in oc.itertuples()]
    oc['on_court_per100']=100*oc.adjusted_points/oc.possessions
    oc=oc.merge(boxes[['player_id','player_name','team','MIN']],on='player_id',validate='many_to_one')
    teams=pd.read_parquet(base/'teams.parquet');teams=teams.loc[teams.game_id.eq(GAME)]
    checks=[]
    for t in teams.itertuples():
        for factor in g.FACTORS:
            observed=float(ledger.loc[ledger.team.eq(t.team)&ledger.factor.eq(factor),'adjusted_points'].sum())
            bridge=0.
            if factor=='oTS':bridge=t.ft_conversion_bridge
            if factor=='dTS':bridge=-float(teams.loc[teams.team.eq(t.opponent),'ft_conversion_bridge'].iloc[0])
            target=float(getattr(t,factor));gap=observed+bridge-target
            if abs(gap)>1e-8:raise ValueError(f'Team factor mismatch {t.team}/{factor}: {gap}')
            checks.append(dict(team=t.team,factor=factor,ledger_points=observed,unallocated_ft_bridge=bridge,team_target=target,gap=gap))
    checks=pd.DataFrame(checks)
    for name,frame in [('possession_ledger',ledger),('dtov_ledger',dtov),('on_court',oc),('team_checks',checks)]:
        frame.to_csv(OUT/f'{name}.csv',index=False);frame.to_parquet(OUT/f'{name}.parquet',index=False)
    summary=dtov.groupby('team')[['identified_actor_points','unidentified_event_points','baseline_points','team_points']].sum().reset_index()
    def table(d):return '<div class="scroll">'+d.to_html(index=False,escape=True,float_format=lambda v:f'{v:.3f}')+'</div>'
    content=['<!doctype html><meta charset="utf-8"><title>One game: attribution evidence</title><style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:16px;background:#faf9f5;color:#182533}p{line-height:1.6;max-width:1000px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}h2{margin-top:38px}.note{background:#fff0ce;padding:18px}</style><h1>NYK–CLE · May 19, 2026</h1><p>One-game attribution evidence. This is not a new player ranking or a validated fantasy score.</p>',
    f'<h2>Start with dTOV</h2><p>A turnover deletes {price:.4f} points in this factor’s season accounting. Each defensive possession carries the league baseline of {baselines.TOV:.4f} points. Identified steals get an explicit event allocation; unidentified turnover credit stays unassigned. This price is not a regression coefficient or a claim of sole causal responsibility.</p>',table(summary),
    '<h2>Every turnover</h2>',table(dtov.loc[dtov.category.ne('no_turnover'),['event_num','period','clock','team','category','actor_id','identified_actor_points','unidentified_event_points','description']]),
    '<h2>Why all steals cannot be used as the dTOV feature</h2><p>Five of 21 steals in this game end second-chance sequences. Net DECOMP records those outcomes in dSC. First-chance steals belong to dTOV. The old all-steals dTOV regression mixed these populations. Second-chance steals are counted separately below; no unverified dSC event price is assigned.</p><h2>Player box and identified first-chance steal credit</h2>']
    direct=dtov.groupby('actor_id').identified_actor_points.sum()
    comparison=boxes[['player_id','player_name','team','MIN','PTS','AST','TOV','STL','BLK','OREB','DREB','dTOV_box_points','dTOV']].copy()
    comparison['first_chance_steals']=comparison.player_id.map(old.loc[old.accounting_factor.eq('dTOV')].groupby('player_id').size()).fillna(0)
    comparison['second_chance_steals']=comparison.player_id.map(old.loc[old.accounting_factor.eq('dSC')].groupby('player_id').size()).fillna(0)
    comparison['identified_steal_event_points']=comparison.player_id.map(direct).fillna(0)
    comparison=comparison.rename(columns={'dTOV':'old_dtov_with_team_adjustment','dTOV_box_points':'old_dtov_box_prior'})
    np.testing.assert_allclose(comparison.identified_steal_event_points,comparison.first_chance_steals*price,atol=1e-9)
    np.testing.assert_allclose(comparison.STL,comparison.first_chance_steals+comparison.second_chance_steals)
    old[['event_num','team','player_id','accounting_factor']].to_csv(OUT/'steal_factor_mapping.csv',index=False)
    content += [table(comparison.drop(columns='player_id')),'<p class="note">The old box prior and its team adjustment are comparison controls, not BPM. Identified event points omit the baseline and unidentified/shared credit; they are not a complete player dTOV score.</p>',
    '<h2>Six-factor results during each player’s possessions</h2><p>These are on-court observations, shared by all five teammates. They must not be summed as individual contribution. Exposure uses canonical possession lineups, not exact event-time lineups when substitutions occur inside a possession. Actual box minutes are shown separately. The FT conversion bridge remains team-level and is not silently distributed.</p>',table(oc[['player_name','team','factor','MIN','possessions','actual_points','adjusted_points','on_court_per100']]),
    '<h2>Team reconciliation</h2>',table(checks),'<p>Opponent-strength adjustment: not implemented. Opponent-three conversion adjustment is a separate accounting choice, not opponent strength. No claim of improvement over BPM has been established.</p><p><a href="dtov_ledger.csv">dTOV events</a> · <a href="possession_ledger.csv">All factor observations</a> · <a href="on_court.csv">Player exposures</a></p>']
    (OUT/'review.html').write_text('\n'.join(content))
    (OUT/'manifest.json').write_text(json.dumps(dict(game=GAME,turnover_price=price,baselines=baselines.to_dict(),max_factor_gap=float(checks.gap.abs().max()),status='attribution evidence only',bpm_comparison='not implemented; old box model is not BPM',steals_total=len(old),steals_second_chance=int(old.accounting_factor.eq('dSC').sum()),offd='unidentified; no invented drawer',opponent_strength='not implemented',lineups='canonical possession lineups; not exact event-time exposure'),indent=2))
    print(summary.to_string(index=False));print(OUT/'review.html')

if __name__=='__main__':main()

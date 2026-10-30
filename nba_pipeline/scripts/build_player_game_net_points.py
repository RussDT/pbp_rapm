#!/usr/bin/env python3
"""Research six-factor completed-game credit, with explicit opponent-3P bridge.

Historical Net DECOMP anchors + observed actor signals are projected onto exact
team component totals. This is a constrained attribution rule, not fitted
single-game RAPM or a validated causal allocation.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from . import build_team_game_net_decomp as team
    from .build_vpm_target_swap import _classify_fgas
    from .process_rapm_blocks.common import sort_raw_pbp_chronologically
except ImportError:
    import build_team_game_net_decomp as team
    from build_vpm_target_swap import _classify_fgas
    from process_rapm_blocks.common import sort_raw_pbp_chronologically

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ['oTS', 'oTOV', 'oSC', 'dTS', 'dTOV', 'dSC']


def normalize_raw(raw):
    """Merge complementary split descriptions only; reject conflicting values."""
    raw = raw.drop_duplicates().copy()
    raw['game_id'] = team.normalize_game_id(raw.game_id)
    duplicate = raw.duplicated(['game_id','event_num'], keep=False)
    merged = []
    for key, group in raw.loc[duplicate].groupby(['game_id','event_num'], sort=False):
        record = {}
        for col in raw.columns:
            values = group[col].dropna().unique()
            if len(values)>1:
                raise ValueError(f'Conflicting raw event {key}: {col}')
            record[col] = values[0] if len(values) else None
        merged.append(record)
    if merged:
        raw = pd.concat([raw.loc[~duplicate],pd.DataFrame(merged)],ignore_index=True)
    return sort_raw_pbp_chronologically(raw).reset_index(drop=True)


def rebound_events(raw):
    """Read changes in explicit cumulative personal Off/Def rebound counts."""
    text=raw.home_description.fillna('')+' '+raw.visitor_description.fillna('')
    counts=text.str.extract(r'REBOUND \(Off:(\d+) Def:(\d+)\)',flags=re.I)
    mask=raw.event_type.eq(4) & counts.notna().all(axis=1)
    out=raw.loc[mask,['game_id','event_num','player1_id']].copy()
    out[['orb_total','drb_total']]=counts.loc[mask].astype(int).to_numpy()
    out['orb']=0
    out['drb']=0
    for _,player in out.groupby(['game_id','player1_id'],sort=False):
        previous=np.zeros(2,dtype=int)
        runs=player[['orb_total','drb_total']].ne(player[['orb_total','drb_total']].shift()).any(axis=1).cumsum()
        for _,run in player.groupby(runs,sort=False):
            total=run[['orb_total','drb_total']].iloc[0].to_numpy(int)
            delta=total-previous
            if np.array_equal(delta,[len(run),0]):
                out.loc[run.index,'orb']=1
            elif np.array_equal(delta,[0,len(run)]):
                out.loc[run.index,'drb']=1
            else:
                raise ValueError('Personal rebound counts are inconsistent')
            previous=total
    return out


def normalized_name(name):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(name).lower()) if not unicodedata.combining(c)).strip()


def project_credit(anchor, observed, exposure, target):
    """Minimum squared displacement weighted by exposure, subject to sum=target."""
    anchor, observed, exposure = [np.asarray(x, dtype=float) for x in (anchor, observed, exposure)]
    if not all(np.isfinite(x).all() for x in (anchor, observed, exposure)) or not np.isfinite(target):
        raise ValueError('Nonfinite allocation input')
    if (exposure < 0).any() or exposure.sum() <= 0:
        raise ValueError('Allocation requires positive total exposure')
    shared = (target - (anchor + observed).sum()) * exposure / exposure.sum()
    return anchor + observed + shared, shared


def adjusted_defense(opponent_components, three_variance):
    """Remove only opponent 3P conversion, retaining its TS/SC lane."""
    result=-opponent_components.copy()
    for lane in ('TS','SC'):
        result[lane]+=float(three_variance.get(lane,0.))
    return result


def prior_factors(frame, year):
    if frame.player_id.duplicated().any():
        raise ValueError('Duplicate prior players')
    if frame.Latest_Year.max() >= year:
        raise ValueError('Prior must end before the game season')
    if not frame.DECOMP_PRICING.eq('net_continuation_kappa').all():
        raise ValueError('Only Net DECOMP priors are compatible')
    out = frame[['player_id', 'player_name']].copy()
    atoms = ['RIM_FREQ', 'RIM_FG', 'MID_FG', 'THREE_FREQ', 'THREE_FG']
    for side in ('o', 'd'):
        out[side+'TS'] = frame[[side+'DECOMP_'+a for a in atoms]].sum(axis=1) + frame[side+'DECOMP_FT']
        out[side+'TOV'] = frame[side+'DECOMP_TOV_VALUE']
        out[side+'SC'] = frame[side+'DECOMP_SC']
    if not np.isfinite(out[FACTORS].to_numpy()).all():
        raise ValueError('Missing/nonfinite prior factors')
    return out.set_index('player_id')


def component_rows(fc, sc):
    rows = team._validate_and_merge(fc, sc)
    non = rows.Is_Turnover.eq(0)
    price = float(rows.loc[non, 'FC_TS_NET_Diff'].mean())
    if not np.isfinite(price) or price <= 0:
        raise ValueError('Invalid turnover price')
    rows['TS'] = np.where(non, rows.FC_TS_NET_Diff, price)
    rows['TOV'] = np.where(non, 0., -price)
    rows['SC'] = rows.SC_NET_CONT_Diff
    rows['modeled_points'] = rows.Net_Diff + rows.SC_GROSS_Diff
    np.testing.assert_allclose(rows[['TS','TOV','SC']].sum(axis=1), rows.modeled_points, atol=1e-9, rtol=0)
    if not np.isfinite(rows[['TS','TOV','SC']].to_numpy()).all():
        raise ValueError('Nonfinite component rows')
    return rows, price


def restore_terminal_misses(rows, raw, metadata):
    """Restore omitted scoreless end-period possessions, including OREB chains."""
    fga, ordered = _classify_fgas(raw)
    keys = ordered[['game_id','event_num','period','_raw_position']]
    terminals = rows[['game_id','event_num','period','FC_CONT_KAPPA_Diff']].merge(keys,on=['game_id','event_num','period'],validate='one_to_one')
    lookup = ordered.set_index(['game_id','event_num'])
    extra=[]
    kappa_samples={}
    tails=[]
    for (gid,period), events in ordered.groupby(['game_id','period']):
        shots=fga.loc[fga.game_id.eq(gid) & fga._raw_position.isin(events._raw_position)]
        t=terminals.loc[terminals.game_id.eq(gid) & terminals.period.eq(period)].sort_values('_raw_position')
        if t.empty:
            raise ValueError(f'Period has no canonical possessions: {gid}/{period}')
        idx=np.searchsorted(t._raw_position.to_numpy(),shots._raw_position.to_numpy())
        seen=set()
        for (_,shot),j in zip(shots.iterrows(),idx):
            if j<len(t) and j not in seen:
                seen.add(j)
                k=float(t.iloc[j].FC_CONT_KAPPA_Diff)
                if k>0:
                    kappa_samples.setdefault(shot['_zone'],[]).append(k)
        tail=shots.loc[shots._raw_position.gt(t._raw_position.max())]
        if len(tail):
            tails.append((gid,period,tail,events))
    for gid,period,tail,events in tails:
        m=metadata[gid]
        actors=[int(lookup.loc[(gid,e)].player1_id) for e in tail.event_num]
        home=actors[0] in m.home_players
        offense=m.home_players if home else m.away_players
        if any(p not in offense for p in actors) or tail._actual_pts.sum()!=0:
            raise ValueError(f'Unmodeled scoring or mixed-offense tail: {gid}/{period}')
        after=events.loc[events._raw_position.gt(tail._raw_position.min())]
        if not after.event_type.isin([0,2,4,6,8,9,13]).all():
            raise ValueError(f'Unmodeled live tail: {gid}/{period}')
        rebounds=after.loc[after.event_type.eq(4)]
        opposing=m.away_players if home else m.home_players
        if (rebounds.player1_id.isin(opposing)&rebounds.time_quarter.ne('00:00')).any():
            raise ValueError(f'Defensive rebound in omitted tail: {gid}/{period}')
        first=tail.iloc[0]; last=tail.iloc[-1]
        event=lookup.loc[(gid,last.event_num)]
        first_event=lookup.loc[(gid,first.event_num)]
        # Recover the SAME source-file zone price; no new continuation model.
        values=kappa_samples.get(first['_zone'],[])
        counts=pd.Series(np.round(values,10)).value_counts()
        if counts.empty or counts.iloc[0]/len(values)<0.95:
            raise ValueError(f'Cannot recover dominant source zone kappa: {first["_zone"]}')
        k=float(np.median([v for v in values if abs(v-counts.index[0])<1e-8]))
        data=dict(game_id=gid,game_id_norm=gid,event_num=int(last.event_num),period=int(period),time_quarter=event.time_quarter,game_date=rows.loc[rows.game_id.eq(gid),'game_date'].iloc[0],Has_FC_FGA=1,Is_Turnover=0,TS=k,TOV=0.,SC=-k,modeled_points=0.,offense_team=m.home_team if home else m.away_team,restored_buzzer_miss=True)
        for i in range(1,6):
            data[f'O{i}']=first_event[f'{"home" if home else "away"}_player{i}']
            data[f'D{i}']=first_event[f'{"away" if home else "home"}_player{i}']
        extra.append(data)
    rows=rows.copy()
    rows['restored_buzzer_miss']=False
    if extra:
        rows=pd.concat([rows,pd.DataFrame(extra)],ignore_index=True)
    return rows


def attach_all_shots(rows, raw, sq):
    """Map all FGAs to the next terminal within the SAME game and period."""
    shots, ordered = _classify_fgas(raw)
    keys = ordered[['game_id','event_num','period','_raw_position','player1_id']]
    if keys.duplicated(['game_id','event_num']).any():
        raise ValueError('Conflicting raw event keys; repair source before attribution')
    terms = rows[['game_id','event_num','period']].copy()
    terms['_row'] = np.arange(len(rows))
    terms = terms.merge(keys.drop(columns=['player1_id']), on=['game_id','event_num','period'], validate='one_to_one')
    if len(terms) != len(rows):
        raise ValueError('Missing raw terminal keys')
    shots = shots.merge(keys[['game_id','event_num','period','player1_id']], on=['game_id','event_num'], validate='one_to_one')
    parts = []
    for (gid, period), group in shots.groupby(['game_id','period']):
        t = terms.loc[terms.game_id.eq(gid) & terms.period.eq(period)].sort_values('_raw_position')
        idx = np.searchsorted(t._raw_position.to_numpy(), group._raw_position.to_numpy())
        if (idx >= len(t)).any():
            raise ValueError(f'FGA after final possession terminal: {gid}/{period}')
        group = group.copy()
        group['_row'] = t._row.to_numpy()[idx]
        parts.append(group)
    shots = pd.concat(parts).sort_values('_raw_position')
    shots['first'] = ~shots.duplicated('_row')
    shots = shots.merge(sq, on=['game_id','event_num'], how='left', validate='one_to_one')
    ev = shots[team.DEFAULT_RUSSELL_SHOTQUALITY_COLUMN]
    if ev.isna().any() or (~ev.between(0, np.where(shots._zone.eq('three'),3,2))).any():
        raise ValueError('Missing or invalid shot-quality values')
    shots['residual'] = shots._actual_pts - ev
    shots['lane'] = np.where(shots['first'], 'TS', 'SC')
    expected = rows.Has_FC_FGA.astype(bool).to_numpy()
    actual = np.zeros(len(rows), dtype=bool)
    actual[shots.loc[shots['first'],'_row'].to_numpy(int)] = True
    if not np.array_equal(expected, actual):
        raise ValueError('First-FGA mapping differs from canonical Has_FC_FGA: '+str(rows.loc[expected != actual,['game_id','event_num','Has_FC_FGA']].to_dict('records')))
    return shots


def build(year, season_type, prior_path, sq_path, prior_scale=1., team_filter=None, game_filter=None):
    suffix = f'{year%100:02d}' + ('_PS' if season_type == 'ps' else '')
    raw_path = ROOT/'raw_data'/f'NBA{suffix}.parquet'
    fc_path = ROOT/'processed_netcont_lane'/f'FIRST_CHANCE{suffix}.parquet'
    sc_path = ROOT/'processed_netcont_lane'/f'SECOND_CHANCE_CLEAN{suffix}.parquet'
    raw = normalize_raw(pd.read_parquet(raw_path))
    fc, sc = pd.read_parquet(fc_path), pd.read_parquet(sc_path)
    for frame in (fc,sc):
        frame['game_id'] = team.normalize_game_id(frame.game_id)
    rows, price = component_rows(fc, sc)
    rebounds = rebound_events(raw)
    orb_price = float(rows.SC_GROSS_Diff.sum()/rebounds.orb.sum())
    drb_price = float(rows.loc[rows.FC_CONT_KAPPA_Diff.gt(0),'FC_CONT_KAPPA_Diff'].mean())
    baselines = rows[['TS','TOV','SC']].mean()
    metadata = team.build_game_metadata(raw)
    rows['offense_team'] = team.assign_offense_teams(rows, metadata)
    priors = prior_factors(pd.read_csv(prior_path), year)
    sq = team.load_russell_v2_shots(sq_path, raw_path.name)
    ids = [gid for gid,m in metadata.items() if (not team_filter or team_filter in (m.home_team,m.away_team)) and (not game_filter or gid in {g.zfill(10) for g in game_filter.split(',')})]
    if not ids:
        raise ValueError('No games selected')
    rows = rows.loc[rows.game_id.isin(ids)].reset_index(drop=True)
    raw = raw.loc[raw.game_id.isin(ids)].reset_index(drop=True)
    rows = restore_terminal_misses(rows, raw, metadata)
    shots = attach_all_shots(rows, raw, sq)
    # Only explicit previous-season observations enter FT actor expectations.
    prev = pd.read_parquet(ROOT/'raw_data'/f'NBA{(year-1)%100:02d}.parquet')
    ft = prev.loc[pd.to_numeric(prev.event_type).eq(3)].drop_duplicates().copy()
    desc = ft.home_description.fillna('') + ' ' + ft.visitor_description.fillna('')
    ft['made'] = ~desc.str.contains('MISS', case=False)
    ft_hist = ft.groupby('player1_id').made.agg(['sum','count'])
    league_ft = float(ft.made.mean())
    ft_pct = (ft_hist['sum'] + 50*league_ft)/(ft_hist['count']+50)
    players, teams, events = [], [], []
    for gid in ids:
        m = metadata[gid]
        gr = rows.loc[rows.game_id.eq(gid)]
        rr = raw.loc[raw.game_id.eq(gid)]
        gs = shots.loc[shots.game_id.eq(gid)]
        sides = [(m.home_team,m.home_players,m.home_score,m.away_team,m.away_score), (m.away_team,m.away_players,m.away_score,m.home_team,m.home_score)]
        for name, roster, score, opponent, opp_score in sides:
            own = gr.loc[gr.offense_team.eq(name)]
            other = gr.loc[gr.offense_team.eq(opponent)]
            exp_o = pd.Series(own[team.OFFENSE_COLS].to_numpy().ravel()).value_counts()
            exp_d = pd.Series(other[[f'D{i}' for i in range(1,6)]].to_numpy().ravel()).value_counts()
            pids = sorted(set(exp_o.index)|set(exp_d.index))
            if any(pd.isna(p) or int(p)<=0 or p not in roster for p in pids):
                raise ValueError(f'Invalid player exposure: {gid}/{name}')
            pids = [int(p) for p in pids]
            ix = {p:i for i,p in enumerate(pids)}
            observed = np.zeros((len(pids),6))
            own_shots = gs.loc[gs.player1_id.isin(roster)]
            opp_shots = gs.loc[~gs.player1_id.isin(roster)]
            # Validate every shot's possession-side attribution before using it.
            for _, shot in own_shots.iterrows():
                if rows.iloc[int(shot['_row'])].offense_team != name:
                    raise ValueError(f'Shooter/possession side mismatch {gid}/{shot.event_num}')
                pid = int(shot.player1_id)
                if pid not in ix:
                    raise ValueError(f'Shooter has no terminal exposure: {gid}/{pid}')
                factor = 'o'+shot.lane
                observed[ix[pid],FACTORS.index(factor)] += shot.residual
                events.append(dict(game_id=gid,team=name,player_id=pid,event_num=int(shot.event_num),factor=factor,kind='fga_conversion',points=float(shot.residual)))
            text = rr.home_description.fillna('')+' '+rr.visitor_description.fillna('')
            fts = rr.loc[pd.to_numeric(rr.event_type).eq(3) & rr.player1_id.isin(roster)]
            for idx, event in fts.iterrows():
                pid = int(event.player1_id)
                value = float(not ('MISS' in text.loc[idx].upper())) - float(ft_pct.get(pid, league_ft))
                if pid not in ix:
                    raise ValueError(f'FT shooter has no exposure: {gid}/{pid}')
                observed[ix[pid],0] += value
                events.append(dict(game_id=gid,team=name,player_id=pid,event_num=int(event.event_num),factor='oTS',kind='ft_conversion',points=value))
            # Charge only a personal turnover explicitly attached to a canonical TOV terminal.
            tov = own.loc[own.Is_Turnover.eq(1),['game_id','event_num']].merge(rr[['game_id','event_num','event_type','player1_id']],on=['game_id','event_num'],validate='one_to_one')
            for event in tov.itertuples():
                pid = int(event.player1_id)
                if int(event.event_type)==5 and pid in ix:
                    observed[ix[pid],1] -= price
                    events.append(dict(game_id=gid,team=name,player_id=pid,event_num=int(event.event_num),factor='oTOV',kind='personal_turnover',points=-price))
            # Opportunity-value actor anchors; team projection prevents double counting.
            game_rebounds=rebounds.loc[rebounds.game_id.eq(gid)&rebounds.player1_id.isin(roster)]
            for event in game_rebounds.itertuples():
                pid=int(event.player1_id)
                if pid not in ix:
                    raise ValueError(f'Rebounder has no exposure: {gid}/{pid}')
                factor='oSC' if event.orb else 'dSC'
                value=orb_price if event.orb else drb_price
                observed[ix[pid],FACTORS.index(factor)]+=value
                events.append(dict(game_id=gid,team=name,player_id=pid,event_num=int(event.event_num),factor=factor,kind='offensive_rebound' if event.orb else 'defensive_rebound',points=value))
            # A steal must identify one player by exact game-local source name.
            name_rows=rr.loc[rr.player1_id.isin(roster),['player1_id','player1_name']].dropna().drop_duplicates()
            aliases={}
            for event in name_rows.itertuples():
                aliases.setdefault(normalized_name(event.player1_name),set()).add(int(event.player1_id))
            side_description='home_description' if name==m.home_team else 'visitor_description'
            for event in rr.loc[rr.event_type.eq(5)].itertuples():
                description=getattr(event,side_description)
                match=re.match(r'^\s*(.*?)\s+STEAL\s*\(',str(description),re.I)
                candidates=aliases.get(normalized_name(match.group(1)),set()) if match else set()
                if len(candidates)==1:
                    pid=next(iter(candidates))
                    if pid in ix:
                        observed[ix[pid],4]+=price
                        events.append(dict(game_id=gid,team=name,player_id=pid,event_num=int(event.event_num),factor='dTOV',kind='steal',points=price))
            own_fg = float(own_shots._actual_pts.sum())
            actual_ft = float((~text.loc[fts.index].str.contains('MISS',case=False)).sum())
            if abs(own_fg+actual_ft-score)>1e-9:
                raise ValueError(f'Raw scoring does not match final score: {gid}/{name}: {own_fg+actual_ft} != {score}')
            bridge = score - float(own.modeled_points.sum())
            opp_bridge = opp_score - float(other.modeled_points.sum())
            own_targets = own[['TS','TOV','SC']].sum() - len(own)*baselines
            opp_targets = other[['TS','TOV','SC']].sum() - len(other)*baselines
            own_targets['TS'] += bridge
            opp_targets['TS'] += opp_bridge
            variance = opp_shots.loc[opp_shots._zone.eq('three')].groupby('lane').residual.sum()
            adjusted_def = adjusted_defense(opp_targets,variance)
            targets = np.r_[own_targets.to_numpy(),adjusted_def.to_numpy()]
            exposures = np.array([[float(exp_o.get(p,0))]*3+[float(exp_d.get(p,0))]*3 for p in pids])
            rates = priors.reindex(pids)[FACTORS].fillna(0.).to_numpy()
            anchor = prior_scale*rates*exposures/100.
            credit = np.zeros_like(anchor)
            shared = np.zeros_like(anchor)
            for j in range(6):
                credit[:,j],shared[:,j] = project_credit(anchor[:,j],observed[:,j],exposures[:,j],targets[j])
            # Show an equal-history control; direct observed action signals stay fixed.
            control = np.column_stack([project_credit(np.zeros(len(pids)),observed[:,j],exposures[:,j],targets[j])[0] for j in range(6)])
            names = rr.loc[rr.player1_id.isin(pids)].dropna(subset=['player1_name']).drop_duplicates('player1_id').set_index('player1_id').player1_name
            for i,pid in enumerate(pids):
                record = dict(game_id=gid,game_date=str(own.game_date.iloc[0]),team=name,opponent=opponent,player_id=pid,player_name=str(names.get(pid,priors.player_name.get(pid,str(pid)))),off_possessions=exposures[i,0],def_possessions=exposures[i,3],prior_missing=pid not in priors.index)
                for j,factor in enumerate(FACTORS):
                    record[factor]=credit[i,j]
                    record[factor+'_prior_points']=anchor[i,j]
                    record[factor+'_observed_points']=observed[i,j]
                    record[factor+'_shared_points']=shared[i,j]
                record.update(offense=credit[i,:3].sum(),defense=credit[i,3:].sum(),game_net_points=credit[i].sum(),zero_prior_net_points=control[i].sum(),prior_effect=credit[i].sum()-control[i].sum())
                players.append(record)
            baseline_bridge = float((len(own)-len(other))*baselines.sum())
            removed = float(variance.sum())
            total = float(credit.sum())
            gap = total + baseline_bridge - removed - (score-opp_score)
            if abs(gap)>1e-8:
                raise ValueError(f'Score closure failed: {gap}')
            teams.append(dict(game_id=gid,game_date=str(own.game_date.iloc[0]),team=name,opponent=opponent,points=score,opponent_points=opp_score,actual_margin=score-opp_score,game_net_points=total,opponent_3p_variance=removed,opponent_3p_variance_TS=float(variance.get('TS',0)),opponent_3p_variance_SC=float(variance.get('SC',0)),unequal_possessions_baseline=baseline_bridge,ft_conversion_bridge=bridge,closure_gap=gap,**dict(zip(FACTORS,targets))))
    paths=[raw_path,fc_path,sc_path,prior_path,sq_path,ROOT/'raw_data'/f'NBA{(year-1)%100:02d}.parquet']
    manifest=dict(restored_buzzer_misses=int(rows.restored_buzzer_miss.sum()),status='research prototype; allocation strengths not validated',year=year,season_type=season_type,prior_scale=prior_scale,prior_season_end=year-1,turnover_price=price,orb_opportunity_price=orb_price,drb_completion_price=drb_price,baselines=baselines.to_dict(),sq_timing='retrospective player-season Russell v2; not pregame or pure contest quality',allocation='exposure-weighted constrained projection of historical priors plus observed actor signals',limitations=['Creation and shot-defense allocation remain shared; rebound/steal opportunity prices and prior strength are unvalidated attribution assumptions','Terminal lineup exposure; within-possession substitutions are not event exposure','FT team bridge is exact only after raw scoring audit; individual FT expectation uses previous season with 50-attempt smoothing','Historical anchors are not opponent-strength corrections; values describe team game contribution','Zero-prior sensitivity is not an uncertainty interval'],inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths])
    manifest['selected_game_ids']=ids
    manifest['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['prior_sha256']=hashlib.sha256(prior_path.read_bytes()).hexdigest()
    return pd.DataFrame(players),pd.DataFrame(teams),pd.DataFrame(events),manifest


def write_review(players, teams, path):
    chunks=['<!doctype html><meta charset="utf-8"><title>Six-factor Game Net Points</title><style>body{font:15px system-ui;margin:32px;color:#17212b;background:#faf9f5}table{border-collapse:collapse;width:100%;margin:18px 0 36px}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}td:nth-child(1),th:nth-child(1){text-align:left}h2{margin-top:40px}p{max-width:1000px;line-height:1.5}</style><h1>Six-factor Game Net Points</h1><p>Research prototype. Actual offensive outcomes; opponent three-point conversion replaced by Russell v2 expectation. Positive is good. Historical Net DECOMP priors plus observed shot/FT conversion, turnovers, steals and rebound opportunities; shared credit uses exposure-weighted reconciliation. Creation and shot defense remain prior-guided; opportunity prices are research assumptions. Prior effect shows sensitivity to removing history, not statistical uncertainty.</p>']
    for gid,g in players.groupby('game_id',sort=True):
        t=teams.loc[teams.game_id.eq(gid)]
        chunks.append('<h2>'+html.escape(str(g.game_date.iloc[0])+' · '+ ' / '.join(g.team.unique())+' · '+gid)+'</h2>')
        chunks.append(t[['team','actual_margin','game_net_points','opponent_3p_variance','unequal_possessions_baseline','closure_gap']].to_html(index=False,float_format=lambda x:f'{x:+.2f}'))
        chunks.append(g.sort_values('game_net_points',ascending=False)[['player_name','team',*FACTORS,'game_net_points','prior_effect','prior_missing']].to_html(index=False,float_format=lambda x:f'{x:+.2f}'))
    chunks.append('<p>Score identity: player net points + unequal-possession baseline − opponent 3P variance = actual margin. Offense and defense use asymmetric outcomes, so adjusted margins need not be opposites.</p>')
    path.write_text('\n'.join(chunks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year',type=int,default=2026)
    p.add_argument('--season-type',choices=['rs','ps'],default='ps')
    p.add_argument('--prior',type=Path)
    p.add_argument('--shot-quality',type=Path,default=team.DEFAULT_RUSSELL_SHOTQUALITY_PATH)
    p.add_argument('--prior-scale',type=float,default=1.)
    p.add_argument('--team',type=str)
    p.add_argument('--game-id',type=str,help='One ID or comma-separated IDs; default all eligible source games')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'player_game_net_points')
    a=p.parse_args()
    prior=a.prior or ROOT/'master_results'/'net_decomp_rolling'/f'weighted_factors_decomp_net_{(a.year-5)%100:02d}_{(a.year-1)%100:02d}_all_rb_se_a2000_4000.csv'
    if not np.isfinite(a.prior_scale) or a.prior_scale<0:
        p.error('prior-scale must be finite and nonnegative')
    players,teams,events,manifest=build(a.year,a.season_type,prior,a.shot_quality,a.prior_scale,a.team.upper() if a.team else None,a.game_id.zfill(10) if a.game_id else None)
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('players',players),('teams',teams),('observed_events',events)]:
        frame.to_csv(out/f'{name}.csv',index=False)
        frame.to_parquet(out/f'{name}.parquet',index=False)
    manifest.update(player_games=len(players),team_games=len(teams),max_closure_gap=float(teams.closure_gap.abs().max()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_review(players,teams,out/'review.html')
    print(json.dumps({k:manifest[k] for k in ['player_games','team_games','max_closure_gap']},indent=2))
    print(out/'review.html')

if __name__=='__main__':
    main()

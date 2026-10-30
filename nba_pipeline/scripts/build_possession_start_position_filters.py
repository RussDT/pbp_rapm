#!/usr/bin/env python3
"""Shooter-position filters for the possession-start and continuation analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from miss_branch_balance import add_miss_balance
from build_possession_start_report import _base_processing, season_raw_path, RIM_ACTION_TYPES, MISS_DISTANCE_RE, CATEGORY_ORDER

OUT = Path(__file__).resolve().parents[1] / 'results/possession_start_report'
POSITION_GROUP = {'PG':'Guards','SG':'Guards','SF':'Forwards','PF':'Forwards','C':'Bigs'}

def build(first_chance_only=False, year=26, team_filters=False):
    raw = _base_processing(season_raw_path(year,'RS')).reset_index(drop=True)
    raw['row_idx']=np.arange(len(raw))
    duplicated = raw.duplicated(['game_id','event_num'], keep=False)
    assert not raw.loc[duplicated,'event_type'].isin(['MAKE','MISS','Rebound','FreeThrow','Turnover']).any(), 'Ambiguous live event keys'
    p=pd.read_parquet(OUT/f'possession_start_possessions_{year}_rs.parquet')
    for label in ['start','end']:
        lookup=raw.drop_duplicates(['game_id','event_num'],keep='first' if label=='start' else 'last').set_index(['game_id','event_num']).row_idx
        p[label+'_idx']=lookup.reindex(pd.MultiIndex.from_arrays([p.game_id,p[label+'_event_num']])).to_numpy()
        assert p[label+'_idx'].notna().all()
    intervals=p[['game_id','poss_group','poss_offense','start_idx','end_idx','off_team_abbr']].rename(columns={'game_id':'parsed_game','poss_offense':'parsed_offense'}).sort_values('start_idx')
    events=pd.merge_asof(raw,intervals,left_on='row_idx',right_on='start_idx')
    events=events.loc[events.game_id.eq(events.parsed_game)&events.row_idx.le(events.end_idx)].copy()
    positions=pd.read_json(OUT/f'shooter_positions_{2000+year}.json').sort_values('minutes',ascending=False).drop_duplicates('nba_id')
    position=positions.set_index('nba_id').position
    events['shooter_group']=pd.to_numeric(events.player1_id,errors='coerce').map(position).map(POSITION_GROUP).fillna('Unknown')
    # The terminal event before a possession start matches the report's existing labels.
    prior=(p.start_idx-1).astype(int)
    safe=prior.clip(lower=0)
    same_game=raw.game_id.iloc[safe].to_numpy()==p.game_id.to_numpy()
    types=raw.event_type.iloc[safe].to_numpy()
    text=(raw.home_description.fillna('')+' '+raw.visitor_description.fillna('')+' '+raw.neutral_description.fillna('')).str.strip()
    previous_text=text.iloc[safe].to_numpy()
    labeled=p[[c for c in CATEGORY_ORDER if c in p]].any(axis=1)&~p.Timeout
    assert (previous_text[labeled&same_game]==p.previous_terminal_event_text.fillna('').to_numpy()[labeled&same_game]).all()
    actor_idx=np.where(types=='Rebound',safe-1,safe).clip(min=0)
    actor_ids=pd.to_numeric(raw.player1_id.iloc[actor_idx],errors='coerce').to_numpy()
    p['shooter_group']=pd.Series(actor_ids,index=p.index).map(position).map(POSITION_GROUP).fillna('Unknown')
    p.loc[~same_game,'shooter_group']='Unknown'
    # First chance ends at the first own rebound of a live miss. Ignore
    # administrative rebounds on non-final/technical/flagrant free throws.
    own_event_text=pd.Series(np.where(events.parsed_offense.eq('Home'),events.home_description,events.visitor_description),index=events.index).fillna('')
    prev_own_text=pd.Series(np.where(events.parsed_offense.eq('Home'),events.Prev_home_desc,events.Prev_visitor_desc),index=events.index).fillna('')
    own_rebound=events.event_type.eq('Rebound')&own_event_text.str.contains('REBOUND',case=False,regex=False)&prev_own_text.str.contains(r'MISS|Putback',case=False,regex=True)&~prev_own_text.str.contains(r'1 of [23]|2 of 3|Technical|Flagrant',case=False,regex=True)
    events['prior_own_rebounds']=own_rebound.astype(int).groupby([events.game_id,events.poss_group]).cumsum()-own_rebound.astype(int)
    events['first_chance']=events.prior_own_rebounds.eq(0)
    first_chance_lookup=events.set_index('row_idx').first_chance
    p['source_first_chance']=first_chance_lookup.reindex(actor_idx).fillna(False).to_numpy()
    p['source_shooting_team']=events.set_index('row_idx').off_team_abbr.reindex(actor_idx).to_numpy()
    # Own continuation: exact saved missed-shot event keys and actual later points.

    misses=pd.read_parquet(OUT/f'own_miss_continuation_detail_{year}_rs.parquet')
    misses=misses.merge(events.loc[events.event_type.eq('MISS'),['game_id','event_num','shooter_group','first_chance','off_team_abbr']],on=['game_id','event_num'],how='left',validate='one_to_one')
    assert misses.shooter_group.notna().all()
    # A loss frequency is valid only when every priced defensive rebound is
    # tied to an actual miss in the all-miss denominator.
    p['source_shot_event_num']=raw.event_num.iloc[actor_idx].to_numpy()
    pair_audits=[]
    for category in ['At Rim Block','At Rim Miss (Unblocked)','Missed 2PT','Missed FG','Short Mid Range Miss','Long Mid Range Miss','Missed 3PT']:
        source=p.loc[p[category],['game_id','source_shot_event_num']].rename(columns={'source_shot_event_num':'event_num'})
        joined=source.merge(misses[['game_id','event_num',category]],on=['game_id','event_num'],how='left',validate='one_to_one')
        failed=joined.loc[~joined[category].eq(True)]
        pair_audits.append({'category':category,'opponent_rows':len(source),'matched_rows':len(source)-len(failed),'excluded_rows':len(failed)})
        if len(failed):
            failed.to_csv(OUT/(f'unmatched_{year}_'+category.lower().replace(' ','_').replace('(','').replace(')','')+'.csv'),index=False)
        p.loc[p[category], 'paired_'+category]=joined[category].eq(True).to_numpy()

    owntext=pd.Series(np.where(events.parsed_offense.eq('Home'),events.home_description,events.visitor_description),index=events.index).fillna('')
    shots=events.loc[events.event_type.isin(['MAKE','MISS'])&owntext.ne('')].copy()
    shottext=shots.home_description.fillna('')+' '+shots.visitor_description.fillna('')
    three=shottext.str.contains('3PT',case=False,regex=False)
    dist=pd.to_numeric(shottext.str.extract(r"(?:^|\s)(\d+)'",expand=False),errors='coerce')
    rim=~three&(pd.to_numeric(shots.event_action_type,errors='coerce').isin(RIM_ACTION_TYPES)|shottext.str.contains(r'layup|dunk| tip ',case=False,regex=True)|dist.le(4))
    shots['zone']=np.select([three,rim],['Three','Rim'],default='Mid')
    shots['made']=shots.event_type.eq('MAKE')
    shots['fine_zone']=np.select([three,rim,dist.between(5,14),dist.ge(15)],['Three','Rim','Short mid','Long mid'],default='Unclassified mid')
    made_dist=pd.to_numeric(pd.Series(previous_text,index=p.index).str.extract(r"(?:^|\s)(\d+)'",expand=False),errors='coerce')
    p['Made Short Mid FG']=p['Made Mid FG']&made_dist.between(5,14)
    p['Made Long Mid FG']=p['Made Mid FG']&made_dist.ge(15)

    baseline=p.off_points.mean()
    cohort_audit={'all_attempts':len(shots),'first_chance_attempts':int(shots.first_chance.sum()),'later_chance_attempts':int((~shots.first_chance).sum()),'first_chance_misses':int(misses.first_chance.sum())}
    if first_chance_only:
        p=p.loc[p.source_first_chance].copy()
        misses=misses.loc[misses.first_chance].copy()
        shots=shots.loc[shots.first_chance].copy()
        assert shots.prior_own_rebounds.eq(0).all()

    hidden={'Above The Break 3 Miss','Corner 3 Miss','Made FG'}
    nonshot={'Live Ball Turnover','Deadball','Timeout'}
    groups=['All','Guards','Forwards','Bigs','Unknown']
    payload={'baseline':baseline,'season':f'{1999+year}–{year:02d} NBA regular season','position_source':f'Databallr player_stats Pos2, {2000+year} RS','mapping':POSITION_GROUP,'groups':{},'first_chance_only':first_chance_only,'cohort_audit':cohort_audit}
    team_names=sorted(shots.off_team_abbr.dropna().unique()) if team_filters else []
    payload['teams']={}
    for team,name in [(team,name) for team in ['All',*team_names] for name in groups]:
        selected=p if name=='All' else p.loc[p.shooter_group.eq(name)]
        m=misses if name=='All' else misses.loc[misses.shooter_group.eq(name)]
        f=shots if name=='All' else shots.loc[shots.shooter_group.eq(name)]
        if team!='All':
            selected=selected.loc[selected.source_shooting_team.eq(team)]
            m=m.loc[m.off_team_abbr.eq(team)]
            f=f.loc[f.off_team_abbr.eq(team)]
        rows=[]
        for category in CATEGORY_ORDER:
            if category in hidden or (name!='All' and category in nonshot): continue
            sub=selected.loc[selected[category]]
            own=m.loc[m[category]] if category in m else None
            paired=sub.loc[sub['paired_'+category].eq(True)] if 'paired_'+category in sub else None
            rows.append({'category':category,'n':len(sub),'ppp':float(sub.off_points.mean()) if len(sub) else None,'delta':float(sub.off_points.mean()-baseline) if len(sub) else None,'own':float(own.remaining_own_points.mean()) if own is not None and len(own) else None,'misses':len(own) if own is not None else None,'paired_n':len(paired) if paired is not None else None,'paired_ppp':float(paired.off_points.mean()) if paired is not None and len(paired) else None})
        zone=[]
        for z in ['Rim','Mid','Three']:
            sub=f.loc[f.zone.eq(z)]
            zone.append({'zone':z,'attempts':len(sub),'makes':int(sub.made.sum()),'fg':float(sub.made.mean()) if len(sub) else None})
        weighted=[]
        for z,made_cat,miss_cats in [('Rim','Made Rim FG',['At Rim Miss (Unblocked)','At Rim Block']),('Short mid','Made Short Mid FG',['Short Mid Range Miss']),('Long mid','Made Long Mid FG',['Long Mid Range Miss']),('Three','Made 3FG',['Missed 3PT'])]:
            attempts=f.loc[f.fine_zone.eq(z)]
            made=selected.loc[selected[made_cat]]
            missed=selected.loc[selected[miss_cats].any(axis=1)]
            fg=float(attempts.made.mean()) if len(attempts) else None
            made_ppp=float(made.off_points.mean()) if len(made) else None
            miss_ppp=float(missed.off_points.mean()) if len(missed) else None
            blend=fg*made_ppp+(1-fg)*miss_ppp if all(v is not None for v in [fg,made_ppp,miss_ppp]) else None
            weighted.append({'zone':z,'attempts':len(attempts),'makes':int(attempts.made.sum()),'fg':fg,'made_ppp':made_ppp,'miss_ppp':miss_ppp,'made_n':len(made),'miss_n':len(missed),'weighted_ppp':blend})
        target=payload['groups'] if team=='All' else payload['teams'].setdefault(team,{})
        target[name]={'rows':rows,'zones':zone,'weighted_zones':weighted,'attempts':len(f),'players':int(f.player1_id.nunique())}

    known=shots.shooter_group.ne('Unknown')
    payload['pairing_audit']=pair_audits
    payload['coverage']={'attempts':len(shots),'classified':int(known.sum()),'pct':float(known.mean()),'unknown_players':int(shots.loc[~known,'player1_id'].nunique())}
    assert sum(payload['groups'][g]['attempts'] for g in groups[1:])==len(shots)
    old=pd.read_csv(OUT/f'possession_start_with_continuation_{year}_rs.csv').set_index('category') if not first_chance_only else None
    for r in ([] if first_chance_only else payload['groups']['All']['rows']):
        assert r['n']==old.loc[r['category'],'possessions']
        if r['own'] is not None: assert abs(r['own']-old.loc[r['category'],'own_points_per_miss'])<1e-12
    for cat in CATEGORY_ORDER:
        if cat in nonshot: continue
        assert sum(int(p.loc[p.shooter_group.eq(g),cat].sum()) for g in groups[1:])==int(p[cat].sum())
    if team_filters:
        assert sum(t['All']['attempts'] for t in payload['teams'].values())==payload['groups']['All']['attempts']
        for category in CATEGORY_ORDER:
            if category in nonshot or category in hidden: continue
            league=next(r for r in payload['groups']['All']['rows'] if r['category']==category)
            children=[next(r for r in t['All']['rows'] if r['category']==category) for t in payload['teams'].values()]
            assert sum(r['n'] for r in children)==league['n']
            if league['misses'] is not None: assert sum(r['misses'] for r in children)==league['misses']
    add_miss_balance(payload)
    packed=json.dumps(payload,allow_nan=False)
    suffix=('' if year==26 else f'_{year}')+('_first_chance' if first_chance_only else '')+('_teams' if team_filters else '')
    (OUT/f'possession_start_position_filters{suffix}.json').write_text(packed)
    template=Path(__file__).with_name('templates').joinpath('possession_start_filters.html').read_text()
    (OUT/f'possession_start_filters{suffix}.html').write_text(template.replace('__POSITION_DATA__',packed))
    balance_template=Path(__file__).with_name('templates').joinpath('possession_miss_balance.html')
    if balance_template.exists():
        (OUT/f'possession_miss_balance{suffix}.html').write_text(balance_template.read_text().replace('__POSITION_DATA__',packed))
    shots.loc[~known,['player1_id','player1_name']].drop_duplicates().to_csv(OUT/f'position_filter_unknown_players_{year}.csv',index=False)
    print(json.dumps({'coverage':payload['coverage'],'rim':{g:next(r for r in payload['groups'][g]['rows'] if r['category']=='At Rim Miss (Unblocked)') for g in groups}},indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--first-chance-only',action='store_true')
    parser.add_argument('--year',type=int,default=26)
    parser.add_argument('--team-filters',action='store_true')
    args=parser.parse_args()
    build(args.first_chance_only,args.year,args.team_filters)

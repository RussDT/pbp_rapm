"""Observed event credits on immutable actual-Net possession boundaries.

This adapter allocates no unidentified credit and estimates no coefficients.
Canonical Net tariffs price state transitions. A rebound identity can affect
only second chance; it is never evidence of who forced the preceding miss.
"""
from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd

from process_rapm_blocks.common import RIM_ACTION_TYPES, sort_raw_pbp_chronologically

LANES = ['rim_freq','rim_conversion','mid_freq','mid_conversion','three_freq',
         'three_conversion','ft','tov','second_chance',
         'mid_reference_on_rim','mid_reference_on_three']
RAW_COLUMNS = ['game_id','event_num','event_type','event_action_type','period',
               'minute_remaining_quarter','seconds_remaining_quarter',
               'home_description','visitor_description',
               'player1_id','player1_name','player1_team_id',
               'player2_id','player2_name','player2_team_id',
               'player3_id','player3_name','player3_team_id'] + [f'{s}_player{i}' for s in ['away','home'] for i in range(1,6)]


def name_key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(value).lower())
                   if not unicodedata.combining(c)).strip()


def normalize(raw):
    raw = raw.copy().drop_duplicates()
    raw['game_id'] = pd.to_numeric(raw.game_id, errors='raise').astype('int64')
    # Complementary split descriptions may merge only without contradictions.
    repeated = raw.duplicated(['game_id','event_num'],keep=False)
    merged = []
    for key,part in raw.loc[repeated].groupby(['game_id','event_num'],sort=False):
        record = {}
        for c in raw.columns:
            values = part[c].dropna().unique()
            if len(values)>1:
                raise ValueError(f'Conflicting event {key}: {c}')
            record[c] = values[0] if len(values) else None
        merged.append(record)
    if merged:
        raw = pd.concat([raw.loc[~repeated],pd.DataFrame(merged)],ignore_index=True)
    raw = sort_raw_pbp_chronologically(raw)
    raw['event_type'] = pd.to_numeric(raw.event_type,errors='raise').astype(int)
    text = raw.home_description.fillna('')+' '+raw.visitor_description.fillna('')
    three = text.str.contains('3PT',case=False,regex=False)
    rim = pd.to_numeric(raw.event_action_type,errors='coerce').isin(RIM_ACTION_TYPES)
    rim |= text.str.contains(r'\blayup\b|\bdunk\b|\btip\b',case=False,regex=True)
    raw['zone'] = np.where(three,'three',np.where(rim,'rim','mid'))
    miss = text.str.contains(r'\bMISS\b',case=False,regex=True)
    raw['points'] = np.where(raw.event_type.eq(1),np.where(three,3.,2.),
                            np.where(raw.event_type.eq(3)&~miss,1.,0.))
    ft = raw.event_type.eq(3)
    technical = text.str.contains(r'Technical|Flagrant|Clear Path|Transition',case=False,regex=True)
    one = text.str.contains(r'\b1 of 1\b',regex=True)
    two = text.str.contains(r'\b[12] of 2\b',regex=True)
    threes = text.str.contains(r'\b[123] of 3\b',regex=True)
    raw['ft_reference'] = np.where(ft&~technical&~one,
                                  np.where(threes,1/3,np.where(two,.5,0.)),0.)
    return raw


def possession_game(raw, reference, players):
    """A separate event-owned actuals ledger, without legacy lane/owner repair.

    Close on defensive rebound, turnover, final FT or period; a made FGA waits
    for its same-clock and-one. A new offensive actor also proves a change of
    possession. Retained technical FTs are zero-exposure administrative rows.
    Every action keeps its raw event key and raw ten-player lineup.
    """
    game=int(raw.game_id.iloc[0]);roster=players.set_index('player_id')
    team_by_player=roster.team_id.to_dict();team_home=players.groupby('team_id').is_home.first().to_dict()
    teamids=set(map(int,players.team_id.unique()));records=raw.to_dict('records')
    raw=raw.copy();raw['_possession_row']=-1
    groups=[];active=[];owner=None;pending=None;period=None;last_miss=False;reasons=[]

    def finish(reason):
        nonlocal active,owner,pending,last_miss
        if active:
            groups.append((active,owner,1.,reason))
        active=[];owner=None;pending=None;last_miss=False

    for i,e in enumerate(records):
        typ=int(e['event_type']);per=int(e['period'])
        if period is not None and per!=period:finish('period_change')
        period=per
        pid=int(e['player1_id']) if pd.notna(e['player1_id']) else 0
        text=str(e['home_description'] or '')+' '+str(e['visitor_description'] or '')
        actor_team=team_by_player.get(pid,pid if pid in teamids else None)
        one_shot=typ==3 and bool(re.search(r'\b1 of 1\b',text))
        associated_basket=(one_shot and pending is not None and pid==pending[0])
        # Away-from-play/take-foul penalty shots are often called simply
        # "Free Throw 1 of 1" in this archive. The live sequence distinguishes
        # a basket's and-one from a retained one-shot penalty. Clock drift
        # between a basket and its and-one must not create a second possession.
        prior_foul=(next((r for r in reversed(records[:i]) if int(r['event_type']) in [1,2,5,6]),None) if typ==3 else None)
        foul_text=('' if prior_foul is None else str(prior_foul['home_description'] or '')+' '+str(prior_foul['visitor_description'] or ''))
        explicit_penalty=bool(re.search(r'Technical|Defensive\s*3\s*Sec|Flagrant|Clear Path|Transition',text,re.I))
        explicit_penalty|=bool(typ==3 and re.search(r'AWAY.FROM.PLAY|T\.FOUL|FLAGRANT|CLEAR.PATH|TRANSITION',foul_text,re.I))
        retained_penalty=typ==3 and (explicit_penalty or (one_shot and not associated_basket))
        if retained_penalty:
            if actor_team is None:raise ValueError('Unidentified administrative FT team')
            # A retained penalty does not replace a field-goal opportunity.
            e['ft_reference']=0.;raw.loc[i,'ft_reference']=0.
            groups.append(([i],actor_team,0.,'administrative_free_throw'))
            continue
        if typ in [1,2,3,5]:
            if actor_team is None:
                # Team turnover descriptions identify a side without inventing
                # a personal actor. Ambiguous neutral turnovers fail.
                if typ!=5:raise ValueError('Unidentified live scoring actor')
                h=bool(str(e['home_description'] or '').strip())
                a=bool(str(e['visitor_description'] or '').strip())
                if h==a:
                    if owner is None:raise ValueError('Unidentified team turnover owner')
                    actor_team=owner
                else:
                    actor_team=next(int(t) for t,v in team_home.items() if bool(v)==h)
            and_one=associated_basket and not explicit_penalty
            if owner is not None and (actor_team!=owner or (pending is not None and not and_one)):
                finish('next_live_owner' if actor_team!=owner else 'made_basket_complete')
            if owner is None:owner=actor_team
            active.append(i)
            if typ==1:
                pending=(pid,e['minute_remaining_quarter'],e['seconds_remaining_quarter'])
                last_miss=False
            elif typ==2:last_miss=True
            elif typ==5:finish('turnover')
            elif typ==3:
                final=bool(re.search(r'\b(1 of 1|2 of 2|3 of 3)\b',text))
                retained=bool(re.search('Flagrant|Clear Path|Transition',text,re.I))
                if final and not retained and float(e['points'])==1:finish('final_made_free_throw')
                elif final:last_miss=True;pending=None
        elif typ==4 and owner is not None:
            active.append(i)
            if actor_team is None:
                h=bool(str(e['home_description'] or '').strip());a=bool(str(e['visitor_description'] or '').strip())
                if h!=a:actor_team=next(int(t) for t,v in team_home.items() if bool(v)==h)
            # A dead-ball rebound between FTs does not change possession.
            future=next((v for v in records[i+1:] if int(v['event_type']) in [1,2,3,5,13]),None)
            intermediate_ft=(future is not None and int(future['event_type'])==3
                             and future['minute_remaining_quarter']==e['minute_remaining_quarter']
                             and future['seconds_remaining_quarter']==e['seconds_remaining_quarter'])
            if last_miss and actor_team is not None and actor_team!=owner and not intermediate_ft:
                finish('defensive_rebound')
        elif typ==13:
            if owner is not None:active.append(i)
            finish('end_of_period')
        elif owner is not None:active.append(i)
    finish('end_of_game')
    atoms=[];slot_rows=[]
    for j,(indices,team_id,exposure,reason) in enumerate(groups):
        # Use the last live action/rebound lineup, never a repaired slot set.
        meaningful=[i for i in indices if int(records[i]['event_type']) in [1,2,3,4,5]]
        if not meaningful:continue
        e=records[meaningful[-1]]
        away=[int(e[f'away_player{k}']) for k in range(1,6)]
        home=[int(e[f'home_player{k}']) for k in range(1,6)]
        if len(set(away+home))!=10 or any(p not in team_by_player for p in away+home):
            raise ValueError('Unresolved raw terminal lineup')
        if any(bool(roster.loc[p,'is_home']) for p in away) or any(not bool(roster.loc[p,'is_home']) for p in home):
            raise ValueError('Raw lineup team mismatch')
        off,defense=(home,away) if bool(team_home[team_id]) else (away,home)
        row=dict(game_id=game,event_num=int(e['event_num']),period=int(e['period']),
                 game_date=reference.game_date.iloc[0],source_season_end_year=int(reference.source_season_end_year.iloc[0]),
                 season_phase='RS',actual_points=sum(float(records[i]['points']) for i in indices),
                 possession_exposure=exposure,terminal_reason=reason,off_home=1. if bool(team_home[team_id]) else -1.)
        row.update(dict(zip([f'{s}{k}' for s in ['O','D'] for k in range(1,6)],off+defense)))
        # j may skip empty non-action groups, so use the actual appended index.
        raw.loc[indices,'_possession_row']=len(atoms);atoms.append(row)
    atoms=pd.DataFrame(atoms)
    scoring=raw.event_type.isin([1,2,3])
    if raw.loc[scoring,'_possession_row'].lt(0).any():raise ValueError('Unassigned scoring action')
    expected=reference.actual_points.sum()
    if atoms.actual_points.sum()!=expected:raise ValueError('Raw/canonical full-game score differs')
    # Independent per-team/player actual shooting identity; this checks the
    # exact observed-count source used for player names and native admission.
    points=raw.loc[scoring].groupby('player1_id').points.sum()
    observed=players.set_index('player_id').observed_points
    if not points.reindex(observed.index,fill_value=0.).eq(observed).all():
        raise ValueError('Raw/observed personal scoring differs')
    index={int(p):i for i,p in enumerate(players.player_id)}
    slots=np.array([[index[int(row[c])] for c in [f'{s}{k}' for s in ['O','D'] for k in range(1,6)]] for row in atoms.to_dict('records')])
    return raw,atoms,slots


def reconstruct_game(raw, native, players, slots):
    """Reconstruct new price-free event atoms on existing terminal boundaries.

    Frozen legacy lane atoms are not reused or overwritten. Raw event scoring
    must match the existing actual-point target on EVERY row, with one exact
    offensive team for every scoring action. This explicitly audits the finer
    source contract needed by actor attribution, beyond whole-game closure.
    """
    teams=dict(players[['player_id','team_id']].astype(int).itertuples(index=False,name=None))
    rr=raw.to_dict('records');lookup={int(r['event_num']):i for i,r in enumerate(rr)}
    terminal=sorted((lookup[int(r.event_num)],j) for j,r in enumerate(native.itertuples()))
    result=native.copy();previous=-1;zone_counts={z:[0.,0.,0.,0.] for z in ['rim','mid','three']}
    for end,j in terminal:
        row=native.iloc[j];own=int(players.iloc[slots[j,0]].team_id)
        ev=([r for r in rr if int(r['_possession_row'])==j] if '_possession_row' in raw
            else [r for r in rr[previous+1:end+1] if int(r['period'])==int(row.period)])
        previous=end
        fields={c:0. for c in ['First_Chance_Diff','Second_Chance_Diff','ft_points','ft_reference_weight','Is_Turnover']}
        for z in zone_counts:
            fields.update({z+'_attempts':0.,z+'_points':0.,'FC_MISS_'+z.upper():0.})
        seen_miss=False;first_zone=None;turnover=False
        for e in ev:
            typ=int(e['event_type']);pid=int(e['player1_id']) if pd.notna(e['player1_id']) else None
            if typ in [1,2,3]:
                if teams.get(pid)!=own:
                    raise ValueError(f'Raw scoring actor/possession ownership mismatch at {e["event_num"]}')
                fields['Second_Chance_Diff' if seen_miss else 'First_Chance_Diff']+=float(e['points'])
                if not seen_miss:
                    if typ in [1,2]:
                        z=e['zone'];fields[z+'_attempts']+=1;fields[z+'_points']+=float(e['points'])
                        if typ==2:
                            fields['FC_MISS_'+z.upper()]+=1;seen_miss=True;first_zone=z
                    else:
                        fields['ft_points']+=float(e['points']);fields['ft_reference_weight']+=float(e['ft_reference'])
            elif typ==5 and not seen_miss:
                actor_team=teams.get(pid,pid if pid==own else None)
                if actor_team is not None and actor_team!=own:
                    raise ValueError('Turnover belongs to other possession side')
                turnover=True
        actual=fields['First_Chance_Diff']+fields['Second_Chance_Diff']
        if actual!=float(row.actual_points):
            raise ValueError(f'Raw/native possession points differ at {row.event_num}: {actual} != {row.actual_points}')
        if turnover:
            # Net's deletion lane is a possession lost BEFORE a scoring
            # opportunity. A turnover after a missed FT/retained scoring
            # opportunity must not charge a second opportunity price.
            if not (actual or sum(fields[z+'_attempts'] for z in zone_counts) or fields['ft_reference_weight']):
                fields['Is_Turnover']=1.
        for c,v in fields.items():result.at[j,c]=v
        for z in zone_counts:
            zone_counts[z][0]+=fields[z+'_attempts'];zone_counts[z][1]+=fields[z+'_points']
            zone_counts[z][2]+=fields['FC_MISS_'+z.upper()]
            if first_zone==z:zone_counts[z][3]+=fields['Second_Chance_Diff']
    if '_possession_row' not in raw and sum(float(e['points']) for e in rr[previous+1:]):
        raise ValueError('Raw scoring after final native terminal')
    counts=dict(game_id=int(native.game_id.iloc[0]),game_date=native.game_date.iloc[0],
                source_season_end_year=int(native.source_season_end_year.iloc[0]),season_phase='RS')
    for z,(n,p,m,sc) in zone_counts.items():
        counts.update({z+'_attempts':n,z+'_fg_points':p,z+'_misses':m,z+'_points':sc})
    return result,counts


def game_events(raw, atoms, players, slots, pricing):
    """Return independently source-checked direct actor credits for one game.

    Unresolved steal/block names remain explicitly shared. Shooting actors must
    resolve exactly. All scoring, FC/SC classification, attempts, misses and FT
    reference weights must reproduce every native possession before admission.
    """
    game = int(atoms.game_id.iloc[0])
    if raw.game_id.nunique()!=1 or int(raw.game_id.iloc[0])!=game:
        raise ValueError('Wrong raw game')
    pidx = {int(p):i for i,p in enumerate(players.player_id)}
    teams = {int(p):int(t) for p,t in players[['player_id','team_id']].itertuples(index=False,name=None)}
    home = {int(p):bool(h) for p,h in players[['player_id','is_home']].itertuples(index=False,name=None)}
    aliases = {}
    for n in [1,2,3]:
        for p,name in raw[[f'player{n}_id',f'player{n}_name']].dropna().drop_duplicates().itertuples(index=False,name=None):
            if int(p) in pidx:
                aliases.setdefault(name_key(name),set()).add(int(p))
    records = raw.to_dict('records')
    positions = {int(r['event_num']):i for i,r in enumerate(records)}
    terminals = []
    for j,r in enumerate(atoms.itertuples()):
        if int(r.event_num) not in positions:
            raise ValueError('Missing exact raw terminal')
        pos = positions[int(r.event_num)]
        if int(records[pos]['period'])!=int(r.period):
            raise ValueError('Terminal period mismatch')
        terminals.append((pos,j))
    terminals.sort()
    if len({p for p,j in terminals})!=len(terminals):
        raise ValueError('Repeated possession terminal')
    actions=[]; diagnostics=[]; previous=-1
    zone_price = pricing['net_zone_reference']; kappa = pricing['kappa']
    survival = float(pricing['net_turnover_price'])
    fg_reference = float(pricing['gross_fga_reference'])

    def resolved(description, kind, team_id):
        match = re.match(r'^\s*(.*?)\s+'+kind+r'\s*\(',str(description),re.I)
        ids = aliases.get(name_key(match.group(1)),set()) if match else set()
        ids = {p for p in ids if teams[p]==team_id}
        return next(iter(ids)) if len(ids)==1 else None

    for end,j in terminals:
        atom = atoms.iloc[j]; ix = slots[j]
        offense = int(players.iloc[ix[0]].team_id)
        defense = int(players.iloc[ix[5]].team_id)
        offense_home = bool(players.iloc[ix[0]].is_home)
        events = ([r for r in records if int(r['_possession_row'])==j] if '_possession_row' in raw
                  else [r for r in records[previous+1:end+1] if int(r['period'])==int(atom.period)])
        previous=end
        seen_miss=False; state=0.; fc_points=0.; sc_points=0.
        attempts={z:0. for z in ['rim','mid','three']}
        fgpoints={z:0. for z in attempts}; misses={z:0. for z in attempts}
        ft_points=0.; ft_reference=0.; sc_transition=0.; outside=0

        def add(side,pid,lane,value,kind,event):
            nonlocal outside
            if pid is None or abs(value)<1e-15:
                return
            if int(pid) not in pidx or teams[int(pid)] != (offense if side=='o' else defense):
                raise ValueError('Actor is not an exact game/side identity')
            qi=pidx[int(pid)]
            exposure=players.iloc[qi]['native_off_poss' if side=='o' else 'native_def_poss']
            if exposure<=0:
                raise ValueError('Actor has no native side exposure')
            if qi not in ix[:5 if side=='o' else 10] or (side=='d' and qi not in ix[5:]):
                outside+=1
            actions.append(dict(row=int(j),game_id=game,event_num=int(event['event_num']),
                                player_id=int(pid),side=side,lane=lane,value=float(value),kind=kind))

        for event in events:
            typ=int(event['event_type']); pid=event['player1_id']
            pid=int(pid) if pd.notna(pid) else None
            actor_team=teams.get(pid)
            points=float(event['points']);zone=event['zone']
            if typ in [1,2,3] and actor_team!=offense:
                raise ValueError(f'Scoring actor/possession side mismatch at {event["event_num"]}')
            if typ in [1,2]:
                if not seen_miss:
                    attempts[zone]+=1;fgpoints[zone]+=points;fc_points+=points
                    missed=typ==2
                    misses[zone]+=missed
                    value=points+float(kappa[zone])*missed-float(zone_price[zone])
                    add('o',pid,zone+'_conversion',value,'first_chance_conversion',event)
                    if missed:
                        desc=event['visitor_description' if offense_home else 'home_description']
                        blocker=resolved(desc,'BLOCK',defense)
                        add('d',blocker,zone+'_conversion',-value,'blocked_first_chance',event)
                        seen_miss=True;state=float(kappa[zone])
                else:
                    sc_points+=points
                    new=float(kappa[zone]) if typ==2 else 0.
                    delta=points+new-state
                    add('o',pid,'second_chance',delta,'continuation_shot',event)
                    # Defended shot responsibility is shared, including SC shots.
                    sc_transition+=delta;state=new
            elif typ==3:
                if not seen_miss:
                    fc_points+=points;ft_points+=points
                    ref=float(event['ft_reference']);ft_reference+=ref
                    add('o',pid,'ft',points-fg_reference*ref,'first_chance_free_throw',event)
                else:
                    # FT-trip misses do not start a continuation state in Net.
                    delta=points-state;state=0.;sc_transition+=delta;sc_points+=points
                    add('o',pid,'second_chance',delta,'continuation_free_throw',event)
            elif typ==4 and seen_miss:
                # Team/admin rebounds carry no identified actor; state still
                # transitions by exact team identity if the admin ID is a team.
                rebound_team=actor_team
                if rebound_team is None and pid in [offense,defense]:
                    rebound_team=pid
                if rebound_team==offense:
                    delta=survival-state
                    add('o',pid if pid in pidx else None,'second_chance',delta,'offensive_rebound_transition',event)
                    sc_transition+=delta;state=survival
                elif rebound_team==defense:
                    delta=-state
                    add('d',pid if pid in pidx else None,'second_chance',state,'defensive_rebound_completion',event)
                    sc_transition+=delta;state=0.
            elif typ==5:
                desc=event['visitor_description' if offense_home else 'home_description']
                stealer=resolved(desc,'STEAL',defense)
                if seen_miss:
                    delta=-state
                    add('o',pid if actor_team==offense else None,'second_chance',delta,'continuation_turnover',event)
                    add('d',stealer,'second_chance',state,'continuation_steal',event)
                    sc_transition+=delta;state=0.
                elif int(atom.Is_Turnover)==1:
                    add('o',pid if actor_team==offense else None,'tov',-survival,'personal_turnover',event)
                    add('d',stealer,'tov',survival,'steal',event)
        # Unidentified terminal completion stays in the shared ledger.
        sc_transition-=state
        fields={'First_Chance_Diff':fc_points,'Second_Chance_Diff':sc_points,
                'ft_points':ft_points,'ft_reference_weight':ft_reference}
        for z in attempts:
            fields.update({z+'_attempts':attempts[z],z+'_points':fgpoints[z],
                           'FC_MISS_'+z.upper():misses[z]})
        gap=max(abs(float(atom[c])-v) for c,v in fields.items())
        if gap>1e-9:
            failures={c:[float(atom[c]),v] for c,v in fields.items() if abs(float(atom[c])-v)>1e-9}
            raise ValueError(f'Event/native action mismatch at {atom.event_num}: {failures}')
        initial=sum(float(kappa[z])*misses[z] for z in misses)
        if abs(sc_transition-(sc_points-initial))>1e-9:
            raise ValueError('Second-chance state transitions do not telescope')
        diagnostics.append(dict(row=j,event_num=int(atom.event_num),action_gap=gap,
                                sc_state_gap=sc_transition-(sc_points-initial),
                                cross_terminal_lineup_actor_terms=outside))
    # Scoreless tails are retained as source limitations, never fabricate an
    # extra possession or admit omitted scoring.
    tail=records[previous+1:]
    if '_possession_row' not in raw and sum(float(r['points']) for r in tail)>0:
        raise ValueError('Scoring after final admitted possession')
    return pd.DataFrame(actions,columns=['row','game_id','event_num','player_id','side','lane','value','kind']),pd.DataFrame(diagnostics)

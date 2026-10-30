"""One-cut current-season extension with frozen annual skill parameters.

Historical files stay immutable. Current cloud counters append only to a local
copy. Context/correction training windows stop at the existing 2026 recipe.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
S = R/'scripts'
E = R/'results/estimated_skills'


def write(path, text):
    path=Path(path)
    with path.open('x') as f:f.write(text)


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError('Expected one explicit source seam: '+old)
    return text.replace(old,new)


def prepare(root, *, native_scale=False):
    root=Path(root).resolve();cut=pd.Timestamp('2026-10-30').as_unit('s')
    refresh=json.loads((root/'refresh.json').read_bytes())
    if refresh['cut']!='2026-10-30' or refresh['player_missing_game_ids'] or refresh['team_missing_game_ids']:
        raise ValueError('Complete current cloud core snapshot required')
    suffix='_native_scale' if native_scale else ''
    source=root/('current_source'+suffix);source.mkdir(exist_ok=False)
    code=root/('current_code'+suffix);code.mkdir(exist_ok=False)
    old=pd.read_parquet(E/'monthly_source_2014_2026/player_game_lines.parquet')
    cloud=pd.read_parquet(root/'player_core.parquet')
    fresh=pd.DataFrame(index=cloud.index,columns=old.columns)
    mapping={'player_id':'person_id','rim_fga':'at_rim_fga','rim_fgm':'at_rim_fgm',
             'three_fga':'fg3a','three_fgm':'fg3m','pf_committed':'fouls',
             'off_fouls_committed':'foul_offensive'}
    for name in old:
        original=mapping.get(name,name)
        if original in cloud:fresh[name]=cloud[original].to_numpy()
    fresh['mid_fga']=cloud.short_midrange_fga+cloud.long_midrange_fga
    fresh['mid_fgm']=cloud.short_midrange_fgm+cloud.long_midrange_fgm
    fresh['minutes']=cloud.seconds_pbp/60
    fresh['game_id']=cloud.game_id.astype('int64').astype(str).str.zfill(10)
    fresh['season_type']='rs';fresh['season']=2027
    names={int(p.stem):json.loads(p.read_bytes()).get('name') for p in (root/'objects/player').glob('*.json') if p.stem.isdigit()}
    fresh['player_name']=cloud.person_id.map(names)
    if native_scale:
        legacy=pd.read_parquet(root/'legacy_current_counts.parquet').set_index(['player_id','game_id'])
        keys=pd.MultiIndex.from_frame(fresh[['player_id','game_id']])
        legacy=legacy.reindex(keys)
        if legacy.fga.isna().any():raise ValueError('Every current identity needs an explicit raw counter row')
        # Retain the selected source's rim/zone definitions and observed FT counts.
        # Cloud assist/steal/block identities replace unavailable legacy text tags.
        for name in ['fga','fgm','rim_fga','rim_fgm','mid_fga','mid_fgm','three_fga',
                     'three_fgm','fta','ftm','pts','tov','minutes']:
            fresh[name]=legacy[name].to_numpy()
        native=pd.read_parquet(root/'native_source/RAPM27_current.parquet')
        for side,slots in [('off',[f'O{i}' for i in range(1,6)]),('def',[f'D{i}' for i in range(1,6)])]:
            long=native[['game_id',*slots]].melt(id_vars='game_id',value_name='player_id')
            long.game_id=long.game_id.astype('int64').astype(str).str.zfill(10)
            counts=long.groupby(['player_id','game_id']).size()
            fresh[side+'_poss']=counts.reindex(keys,fill_value=0).to_numpy()
    clean=cloud.lineups_clean.eq(True)&cloud.provably_wrong_rows.eq(0)&cloud.rows_without_10_players.eq(0)
    fresh.loc[~clean,['off_poss','def_poss']]=np.nan
    for num,den in [('rim_fgm','rim_fga'),('mid_fgm','mid_fga'),('three_fgm','three_fga'),('ftm','fta')]:
        if not (fresh[num].ge(0)&fresh[num].le(fresh[den])).all():raise ValueError('Invalid observed current counts')
    if not pd.to_datetime(fresh.game_date).lt(cut).all():raise ValueError('Same/future day source')
    lines=pd.concat([old,fresh],ignore_index=True)
    if lines.duplicated(['player_id','game_id']).any():raise ValueError('Repeated current/history game')
    lines.to_parquet(source/'player_game_lines.parquet',index=False)
    ctx=pd.read_parquet(E/'monthly_vintage_v1/game_context.parquet')
    newctx=pd.DataFrame(dict(game_id=fresh.game_id,player_id=fresh.player_id.astype('int64'),
        is_home=cloud.is_home.astype(int),team_id=cloud.team_id,opp_team_id=cloud.opponent_id))
    pd.concat([ctx,newctx],ignore_index=True).to_parquet(source/'game_context.parquet',index=False)
    age=pd.read_parquet(E/'age_lookup.parquet');known=age.loc[age.season.eq(2026)].copy()
    known['season']=2027;known['age']=known.age+1
    pd.concat([age,known],ignore_index=True).to_parquet(source/'age_lookup.parquet',index=False)

    # Isolated copies change only the current source paths and scope fences.
    context=(S/'estimated_skills_context.py').read_text()
    context=replace_once(context,'hist = ab[ab.season < S] if i else ab[ab.season == S]',
        'hist = ab[ab.season < min(S, 2026)] if i else ab[ab.season == S]')
    context=replace_once(context,'m = (seasons < S) & has_prior & np.isfinite(resid)',
        'm = (seasons < min(S, 2026)) & has_prior & np.isfinite(resid)')
    write(code/'current_skill_context.py',context)
    corrections=(S/'build_nba_estimated_skills_v2_stageb.py').read_text()
    corrections=replace_once(corrections,'train = [s for s in seasons_all if s < S]',
        'train = [s for s in seasons_all if s < min(S, 2026)]')
    write(code/'current_skill_corrections.py',corrections)
    own=(S/'build_monthly_own_skills.py').read_text()
    own=replace_once(own,'ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(R))+')')
    own=replace_once(own,"SOURCE=ROOT/'results/estimated_skills/monthly_source_2014_2026'",'SOURCE=Path('+repr(str(source))+')')
    own=replace_once(own,"AGE=ROOT/'results/estimated_skills/age_lookup.parquet'",'AGE=Path('+repr(str(source/'age_lookup.parquet'))+')')
    own=replace_once(own,'import estimated_skills_context as X','import current_skill_context as X')
    own=replace_once(own,'from build_nba_estimated_skills_v2_stageb import build_corrections',
        'from current_skill_corrections import build_corrections')
    own=replace_once(own,"ctx=pd.read_parquet(out/'game_context.parquet')",'ctx=pd.read_parquet(Path('+repr(str(source/'game_context.parquet'))+'))')
    write(code/'current_monthly_own.py',own)
    selected=(S/'selected_state_history.py').read_text()
    selected=replace_once(selected,'ROOT = Path(__file__).resolve().parents[1]','ROOT = Path('+repr(str(R))+')')
    selected=replace_once(selected,'import build_monthly_own_skills as C','import current_monthly_own as C')
    selected=selected.replace('between(1997,2026)','between(1997,2027)').replace('self.lines.season.gt(2026)','self.lines.season.gt(2027)')
    selected=selected.replace("'2026-10-01'","'2026-10-30'").replace("'2026-10-02'","'2026-10-30'")
    selected=selected.replace('protected_2027_inputs=0','current_season_observation_rows=int(self.lines.season.eq(2027).sum())')
    write(code/'current_selected_state.py',selected)
    sys.path.insert(0,str(code));sys.path.insert(1,str(S))
    import current_selected_state as selected_module
    selected=selected_module.SelectedStateHistory(root/('state_cache'+suffix),source_mode='identity_cleaned')
    selected._pin(source/'game_context.parquet');selected._pin(root/'refresh.json')
    selected._source_inputs=dict(selected._inputs)
    previous=R.parent.parent/'vpm/vpm/results/harness/vpm_full_feature_history_v1/compact_history_fast_v1/cuts/2026-10-01'
    # All prior player identities remain available; no future eligibility list.
    previous_reference=E/'full_feature_history_fast_prepared_features_v1/full6107_v1/2026-10-01/reference.parquet'
    prior=pd.read_parquet(previous_reference)
    wide,long,proof=selected.query_reference(cut,2027,extra_player_ids=prior.player_id.to_numpy('int64'))
    wide.to_parquet(root/('current_reference_core'+suffix+'.parquet'),index=False)
    long.to_parquet(root/('current_reference_provenance'+suffix+'.parquet'),index=False)
    write(root/('current_reference_core'+suffix+'.json'),json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='current_core_features_ready',rows=len(wide),new_observation_rows=len(fresh),
        latest_source=str(pd.to_datetime(fresh.game_date).max().date()),new_mapper_fits=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--native-scale',action='store_true');args=p.parse_args()
    prepare(args.root,native_scale=args.native_scale)

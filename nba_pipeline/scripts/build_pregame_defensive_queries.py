"""Batched defensive skill queries with explicit game-day season metadata.

The caller supplies each query's date and NBA ending season. Calendar dates
never infer a season: the October2020 bubble remains season2020, and the
completed-season September2026 cut remains season2026. Frozen filter recipes
and measurement semantics are imported without changing their saved exporter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_monthly_own_skills as C
import build_pregame_defensive_skills as F
from build_pregame_neutral_skills import compile_kernel,replay

ROOT=Path(__file__).resolve().parents[1]
FEATURES=F.TRACKING+['own_rim_contests100','own_rim_suppression','own_rim_saved100','own_dfga100','ctx_mpg']
QUERY_PREFIX='__pregame_query__'


def prepare_queries(queries):
    required=['player_id','query_date','query_season']
    if not set(required).issubset(queries):raise ValueError('Need explicit player_id/query_date/query_season')
    d=queries.copy().reset_index(drop=True)
    d['query_date']=pd.to_datetime(d.query_date,utc=True).dt.tz_localize(None)
    if d[required].isna().any().any() or not d.query_date.eq(d.query_date.dt.normalize()).all():
        raise ValueError('Queries need complete identities and midnight game dates')
    for field in ['player_id','query_season']:
        numeric=pd.to_numeric(d[field],errors='raise')
        if not np.isfinite(numeric).all() or not np.equal(numeric,np.floor(numeric)).all():
            raise ValueError('Integer query identity/season required')
        d[field]=numeric.astype(np.int64)
    if not d.player_id.gt(0).all() or not d.query_season.between(2014,2026).all():
        raise ValueError('Invalid query identity or out-of-scope NBA season')
    if not d.query_date.between(C.vintage_cut(2015),pd.Timestamp('2026-09-01')).all():
        raise ValueError('Query outside legal annual vintage/completed source period')
    if 'query_id' not in d:d['query_id']=[str(i) for i in range(len(d))]
    if d.query_id.isna().any():raise ValueError('Missing query ID')
    d['query_id']=d.query_id.astype(str)
    if d.query_id.duplicated().any():raise ValueError('Query IDs must be unique')
    conflict=d.groupby(['player_id','query_date']).query_season.nunique().gt(1)
    if conflict.any():raise ValueError('Conflicting seasons for one player/date query batch')
    cuts=[(year,C.vintage_cut(year)) for year in range(2015,2026)]
    d['parameter_vintage']=[max(y for y,cut in cuts if cut<=date) for date in d.query_date]
    return d


def load_inputs(source):
    queries,rim,dfga,measurements,audit=F.source_inputs(source)
    return dict(source=Path(source),queries=queries,rim=rim,dfga=dfga,measurements=measurements,
        source_reconciliation=audit,models_root=F.DEFENSE/'models')


def combined_rows(queries,real_rows):
    """One real observation row plus any number of pure queries per player/day."""
    max_date=queries.query_date.max()
    columns=F.KEY+['game_date','season','season_type','minutes']
    real=real_rows.loc[real_rows.game_date.le(max_date),columns].copy()
    if real.game_id.astype(str).str.startswith(QUERY_PREFIX).any():raise ValueError('Reserved synthetic identity in source')
    real['_query_position']=-1
    extra=queries[['player_id','query_date','query_season','query_id']].rename(
        columns={'query_date':'game_date','query_season':'season'}).copy()
    extra['game_id']=QUERY_PREFIX+extra.query_id
    extra['season_type']='query';extra['minutes']=np.nan
    extra['_query_position']=np.arange(len(extra))
    q=pd.concat([real,extra[columns+['_query_position']]],ignore_index=True)
    q=q.sort_values(['player_id','game_date','game_id'],kind='stable').reset_index(drop=True)
    conflict=q.groupby(['player_id','game_date']).season.nunique().gt(1)
    if conflict.any():raise ValueError('Explicit query season conflicts with actual same-date NBA season')
    if q.duplicated(F.KEY).any():raise ValueError('Duplicate combined query/observation identity')
    return q


def query_vintage(queries,year,inputs,work_dir):
    """Return all requested player-day states in one frozen-vintage replay."""
    requested=prepare_queries(queries)
    if requested.empty:raise ValueError('Empty defensive query block')
    if not requested.parameter_vintage.eq(year).all():raise ValueError('Queries span different legal annual vintages')
    model_path=Path(inputs['models_root'])/f'{year}.json'
    model=json.loads(model_path.read_text());cut=C.vintage_cut(year)
    if pd.Timestamp(model['fit_cut'])!=cut or int(model['vintage'])!=year:
        raise ValueError('Defensive annual parameter identity mismatch')
    for side in ['rim','dfga']:
        if pd.Timestamp(model[side]['fit_end'])>=cut:raise ValueError('Future defensive parameter fit')
    q=combined_rows(requested,inputs['queries'])
    selected=q._query_position.ge(0).to_numpy()
    positions=q.loc[selected,'_query_position'].to_numpy(int)
    result=requested.iloc[positions].copy().reset_index(drop=True)
    def add(name,value,exposure,last):
        result[name]=np.asarray(value)[selected]
        result[name+'_missing']=~np.isfinite(result[name])
        result[name+'__prior_exposure']=np.asarray(exposure)[selected]
        result[name+'__last_observation_day']=np.asarray(last)[selected]
        if name!='ctx_mpg':result[name+'__observed_this_game']=False
    for name in F.TRACKING:
        observations=inputs['measurements'].loc[inputs['measurements'].feature.eq(name)]
        value,exposure,last,_=F.eb_queries(q,observations)
        add(name,value,exposure,last)
    arr,last,_=F.rim_queries(q,inputs['rim'],model['rim'])
    for name,col,exposure in [('own_rim_contests100',0,3),('own_rim_suppression',1,4),('own_rim_saved100',2,4)]:
        add(name,arr[:,col],arr[:,exposure],last)
    observations=q[F.KEY].merge(inputs['dfga'][F.KEY+['dfga_value','def_poss']],on=F.KEY,
        how='left',validate='one_to_one')
    valid=np.isfinite(observations.dfga_value)&observations.dfga_value.ge(0)&observations.def_poss.gt(0)
    frame=q[['player_id','season']].copy()
    frame['day']=(q.game_date-pd.Timestamp('1970-01-01')).dt.days
    frame['exposure']=observations.def_poss.where(valid,0.)
    frame['obs_rate']=observations.dfga_value.div(observations.def_poss).where(valid)
    mean=model['dfga']['mean'];zeros=np.zeros(len(q))
    values=replay(frame,model['dfga']['params'],mean,np.full(len(q),mean),zeros,zeros,compile_kernel(work_dir))
    add('own_dfga100',np.where(values[4]>0,100*np.maximum(values[0],0),np.nan),values[4],values[5].astype(int))
    result['own_dfga100__state_variance']=10000*values[1][selected]
    mpg,games,last=F.mpg_queries(q)
    add('ctx_mpg',mpg,games,last);result['ctx_games']=games[selected]
    result['estimate_date']=result.query_date
    result['dfga_parameter_fit_end']=pd.Timestamp(model['dfga']['fit_end'])
    result['rim_parameter_fit_end']=pd.Timestamp(model['rim']['fit_end'])
    days=(result.query_date-pd.Timestamp('1970-01-01')).dt.days.to_numpy()
    for name in [c for c in result if c.endswith('__last_observation_day')]:
        last=result[name].to_numpy()
        if not ((last==-1)|(last<days)).all():raise ValueError('Query included current/future observation')
    result['_output_position']=positions
    result=result.sort_values('_output_position',kind='stable').drop(columns='_output_position').reset_index(drop=True)
    receipt=dict(status='passed_batched_explicit_season_defensive_queries',year=year,queries=len(result),
        replay_rows=len(q),source_observation_census_rows=int((~selected).sum()),
        model_sha256=C.sha(model_path),query_only=True,season='explicit caller metadata, never calendar inferred',
        parameter_fit_cut=str(cut.date()),measurement_fence='all same-date queries precede updates',
        availability={name:int(result[name].notna().sum()) for name in FEATURES},
        code_sha256=C.sha(Path(__file__)))
    return result,receipt


def source_files(source,query_path):
    files=[Path(__file__),Path(F.__file__),Path(C.__file__),Path(__file__).with_name('build_pregame_neutral_skills.py'),
        Path(__file__).with_name('pregame_neutral_kernel.cpp'),Path(source)/'player_game_lines.parquet',
        Path(source)/'manifest.json',Path(query_path),F.DEFENSE/'rim_counts.parquet',
        F.DEFENSE/'dfga_counts.parquet',F.MEASUREMENTS]
    files+=sorted((F.DEFENSE/'models').glob('*.json'))
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    queries=prepare_queries(pd.read_parquet(args.queries))
    lock=dict(source_hashes={str(p.resolve()):C.sha(p) for p in source_files(args.source,args.queries)},
        query_season='explicit NBA ending season',full_recipe_ready=False,selected_model_changed=False)
    lock_path=args.out/'run_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:
        raise ValueError('Changed query/source/code requires a new output version')
    F.write_json(lock_path,lock);inputs=load_inputs(args.source)
    F.write_json(args.out/'source_reconciliation.json',inputs['source_reconciliation'])
    for year,group in queries.groupby('parameter_vintage',sort=True):
        target=args.out/'vintages'/f'{year}.parquet';stamp=target.with_suffix('.json')
        if stamp.exists():
            previous=json.loads(stamp.read_text())
            if previous['run_lock_sha256']!=C.sha(lock_path) or previous['output_sha256']!=C.sha(target):
                raise ValueError('Cached batch differs from lock')
            print(json.dumps(dict(year=int(year),status='cached')),flush=True);continue
        frame,receipt=query_vintage(group,int(year),inputs,args.out/'kernel')
        target.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(target,index=False)
        receipt.update(output_sha256=C.sha(target),run_lock_sha256=C.sha(lock_path))
        F.write_json(stamp,receipt)
        print(json.dumps(dict(year=int(year),queries=len(frame),status=receipt['status'])),flush=True)


if __name__=='__main__':main()

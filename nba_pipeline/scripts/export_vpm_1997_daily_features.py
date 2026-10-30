"""Pure daily queries of the audited v73 compact predictive-stat estimator.

Annual report recipes are HELD between report cuts (2020-Nov1 exception),
bootstrap1997 before the first report. This is not the library's next-game
ending-season recipe calendar, selected-neutral parity, or a parameter refit.
The new C++ query path never mutates source states. Real zero-opportunity and
same-day source batches retain frozen full-history kernel semantics.
"""
from __future__ import annotations
import argparse,ctypes,fcntl,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from build_full_skill_history import Engine
from full_skill_history_specs import SKILLS
import export_vpm_1997_features as A
from export_full_history_annual_snapshots import strict_observations,validate_recipe,sha,write_json,feature_schema

PROFILE='v73_compact_daily_pure_query_held_report_parameters_v1'
LOWER=pd.Timestamp('1996-11-01');UPPER=pd.Timestamp('2026-10-01')
KEYS=A.KEYS+['parameter_vintage']
CPP=Path(__file__).with_name('vpm_1997_daily_query_kernel.cpp')
COMPACT_SCHEMA=A.compact_schema(feature_schema())
RECORDS={r['name']:r for r in COMPACT_SCHEMA['records'] if r.get('source')}
HEAD_NAMES=sorted({h for r in RECORDS.values() for v in r['source_heads_by_era'].values() for h in v})
SPECS=[s for s in SKILLS if s.name in HEAD_NAMES]


def parameter_vintage(cut):
    cut=pd.Timestamp(cut)
    if not LOWER<=cut<=UPPER:raise ValueError('Historical date1996-Nov1 through2026-Oct1 only; no2027')
    legal=[y for y in range(1997,2027) if A.annual_report_cut(y)<=cut]
    return max(legal) if legal else 1997


def normalize_queries(frame):
    q=frame.copy()
    if not set(A.KEYS)<=set(q) or q.query_id.isna().any() or q.query_id.duplicated().any():raise ValueError('Exact unique query identity required')
    q['query_date']=pd.to_datetime(q.query_date,utc=True,errors='raise').dt.tz_localize(None)
    if q.query_date.isna().any() or not q.query_date.eq(q.query_date.dt.normalize()).all() or not q.query_date.between(LOWER,UPPER).all():raise ValueError('Date-only protected historical query bounds')
    for c in ['player_id','query_season']:
        z=pd.to_numeric(q[c],errors='raise')
        if not np.isfinite(z).all() or not np.equal(z,np.floor(z)).all():raise ValueError('Exact integer identity/season required')
        q[c]=z.astype('int64')
    if q.player_id.le(0).any() or not q.query_season.between(1997,2026).all():raise ValueError('Positive canonical IDs and no2027 seasons')
    q['cohort_vintage']=pd.to_numeric(q.cohort_vintage,errors='raise').astype('Int64')
    # Validate bounds/IDs above, then evaluate the frozen calendar once per
    # distinct date rather than once per full-universe player query.
    vintage_by_date={cut:parameter_vintage(cut) for cut in q.query_date.drop_duplicates()}
    expected=q.query_date.map(vintage_by_date).astype('int64')
    if 'parameter_vintage' in q and not pd.to_numeric(q.parameter_vintage).eq(expected).all():raise ValueError('Parameter vintage violates held annual report calendar')
    q['parameter_vintage']=expected
    if q.duplicated(['player_id','query_date','query_season']).any() or q.groupby('query_date').query_season.nunique().gt(1).any():raise ValueError('Ambiguous same-date query identities/seasons')
    return q[KEYS].reset_index(drop=True)


def build_calendar(panel,target_calendar=None,*,include_reports=True):
    """Union source/retained-target game days and annual endpoints; no2027."""
    dates=panel[['game_date','season']].drop_duplicates().rename(columns={'game_date':'query_date','season':'query_season'})
    if target_calendar is not None:
        t=target_calendar[['game_date','season']].drop_duplicates().rename(columns={'game_date':'query_date','season':'query_season'})
        dates=pd.concat([dates,t],ignore_index=True)
    if include_reports:
        dates=pd.concat([dates,pd.DataFrame([dict(query_date=A.annual_report_cut(y),query_season=y) for y in range(1997,2027)])],ignore_index=True)
    dates.query_date=pd.to_datetime(dates.query_date)
    dates=dates.drop_duplicates().sort_values('query_date').reset_index(drop=True)
    if dates.groupby('query_date').query_season.nunique().gt(1).any() or not dates.query_season.between(1997,2026).all() or not dates.query_date.between(LOWER,UPPER).all():raise ValueError('Conflicting/protected daily calendar')
    dates['parameter_vintage']=dates.query_date.map(parameter_vintage)
    return dates


def build_queries(panel,calendar):
    """Vectorized complete earlier source-prefix IDs; never future participants."""
    first=panel.groupby('player_id').game_date.min();parts=[]
    for day in calendar.itertuples():
        ids=np.sort(first.index[first.lt(day.query_date)].to_numpy('int64'))
        if len(ids)==0:continue
        prefix=f'daily:{day.query_date.date()}:{int(day.query_season)}:'
        parts.append(pd.DataFrame(dict(query_id=prefix+pd.Series(ids).astype(str),player_id=ids,query_date=day.query_date,query_season=int(day.query_season),cohort_vintage=int(day.parameter_vintage),parameter_vintage=int(day.parameter_vintage))))
    return normalize_queries(pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=KEYS))


def compile_kernel(out):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True);path=folder/('daily_query_'+sha(CPP)[:16]+'.so')
    if not path.exists():subprocess.run(['c++','-O3','-std=c++11','-shared','-fPIC',str(CPP),'-o',str(path)],check=True)
    return path


class QueryEngine:
    """Shared source/query geometry; three float64 outputs per query/head."""
    def __init__(self,panel,queries,library):
        self.q=normalize_queries(queries)
        if not len(self.q):raise ValueError('Nonempty query batch required')
        source=panel.loc[panel.game_date.lt(self.q.query_date.max())].reset_index(drop=True)
        self.source=source
        s=source[['player_id','game_date','season']].copy();s['_source']=np.arange(len(s));s['_query']=-1
        z=pd.DataFrame(dict(player_id=self.q.player_id,game_date=self.q.query_date,season=self.q.query_season,_source=-1,_query=np.arange(len(self.q))))
        full=pd.concat([s,z],ignore_index=True).sort_values(['player_id','game_date','_query'],kind='stable')
        self.ids=np.ascontiguousarray(pd.factorize(full.player_id)[0],dtype='int64');self.nplayers=int(self.ids.max())+1
        self.day=np.ascontiguousarray((full.game_date-pd.Timestamp('1970-01-01')).dt.total_seconds()/86400,dtype='float64')
        self.season=np.ascontiguousarray(full.season,dtype='float64');self.query=np.ascontiguousarray(full._query,dtype='int64')
        self.positions=full._source.to_numpy('int64');self.real=self.positions>=0
        self.library=ctypes.CDLL(str(library));self.fn=self.library.daily_skill_queries
        d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags='C_CONTIGUOUS')
        self.fn.argtypes=[ctypes.c_long,ctypes.c_long,i,d,d,d,d,i,d,ctypes.c_double,ctypes.c_int,d,d,d];self.fn.restype=None

    def run(self,spec,recipe,original_library):
        # Reuse the frozen measurement/epoch gates; only query evaluation differs.
        if len(self.source)==0:
            n=len(self.q)
            return dict(prediction=np.full(n,np.nan),prior_exposure=np.zeros(n),last_day=np.full(n,-1.),source_supported=np.zeros(n,bool),player_observed=np.zeros(n,bool))
        old=Engine(self.source,spec,original_library)
        n=len(self.ids);ex=np.zeros(n);y=np.full(n,np.nan)
        ex[self.real]=old.exposure[self.positions[self.real]];y[self.real]=old.y[self.positions[self.real]]
        out=[np.empty(len(self.q),dtype='float64') for _ in range(3)]
        self.fn(n,self.nplayers,self.ids,self.day,self.season,np.ascontiguousarray(ex),np.ascontiguousarray(y),self.query,np.ascontiguousarray(recipe['parameters'],dtype='float64'),float(recipe['mean']),int(spec.kind=='pct'),*out)
        prediction,exposure,last=out;prediction*=1 if spec.kind=='pct' else 100
        first=old.day[old.exposure>0].min(initial=np.inf)
        supported=self.q.query_season.between(spec.first_season,spec.last_season).to_numpy() & ((self.q.query_date-pd.Timestamp('1970-01-01')).dt.days.to_numpy()>first)
        observed=(exposure>0)&(last>=0);prediction[~(supported&observed)]=np.nan
        if np.any((last>=0)&(last>=(self.q.query_date-pd.Timestamp('1970-01-01')).dt.days.to_numpy())):raise ValueError('Current-day observation leaked into daily query')
        return dict(prediction=prediction,prior_exposure=exposure,last_day=last,source_supported=supported,player_observed=observed)


class SupportIndex:
    """Exact prefix exposure and [cut−365days,cut) positive-RS-minute means."""
    def __init__(self,support):
        self.groups={int(pid):g.sort_values('game_date',kind='stable').reset_index(drop=True) for pid,g in support.groupby('player_id',sort=False)}
    def query(self,queries):
        q=normalize_queries(queries);out=pd.DataFrame(index=q.index)
        numeric=['archive_off_possessions_lower_bound','archive_def_possessions_lower_bound','archive_exposure_unknown_rows','archive_observed_games','archive_positive_playing_games','ctx_games']
        for c in numeric:out[c]=0.
        for c in ['first_archive_observation_date','last_archive_observation_date','last_archive_positive_observation_date','ctx_mpg__last_observation_date']:out[c]=pd.NaT
        out['ctx_mpg']=np.nan
        for pid,part in q.groupby('player_id',sort=False):
            g=self.groups.get(int(pid))
            if g is None:continue
            days=g.game_date.to_numpy('datetime64[ns]');cut=part.query_date.to_numpy('datetime64[ns]');end=np.searchsorted(days,cut,side='left');ix=part.index
            off=g.off_poss.to_numpy(float);de=g.def_poss.to_numpy(float);go=np.isfinite(off)&(off>=0);gd=np.isfinite(de)&(de>=0)
            positive=(g.off_poss.gt(0)|g.def_poss.gt(0)|g.minutes.gt(0)).to_numpy()
            for c,val in [('archive_off_possessions_lower_bound',np.where(go,off,0)),('archive_def_possessions_lower_bound',np.where(gd,de,0)),('archive_exposure_unknown_rows',~(go&gd)),('archive_positive_playing_games',positive)]:out.loc[ix,c]=np.r_[0,np.cumsum(val)][end]
            out.loc[ix,'archive_observed_games']=end
            present=end>0
            if present.any():
                out.loc[ix[present],'first_archive_observation_date']=days[0];out.loc[ix[present],'last_archive_observation_date']=days[end[present]-1]
            pday=days[positive];pe=np.searchsorted(pday,cut,side='left');has=pe>0
            if has.any():out.loc[ix[has],'last_archive_positive_observation_date']=pday[pe[has]-1]
            minutes=g.loc[g.playoffs.eq(False)&g.minute_observed&g.minutes.gt(0)];md=minutes.game_date.to_numpy('datetime64[ns]');mv=minutes.minutes.to_numpy(float)
            lo=np.searchsorted(md,cut-np.timedelta64(365,'D'),side='left');hi=np.searchsorted(md,cut,side='left');count=hi-lo;cache={}
            means=np.full(len(count),np.nan)
            for j,(a,b) in enumerate(zip(lo,hi)):
                if b>a:
                    key=(int(a),int(b))
                    if key not in cache:cache[key]=float(np.mean(mv[a:b]))
                    means[j]=cache[key]
            out.loc[ix,'ctx_games']=count;out.loc[ix,'ctx_mpg']=means
            has=count>0
            if has.any():out.loc[ix[has],'ctx_mpg__last_observation_date']=md[hi[has]-1]
        out['ctx_mpg__missing']=out.ctx_games.eq(0);out['ctx_mpg__days_since_observation']=(q.query_date-out.ctx_mpg__last_observation_date).dt.total_seconds()/86400
        for c in numeric[2:]:out[c]=out[c].astype('int64')
        return out


def day_dates(last):
    """Exact integer source-day dates; −1 is the only missing sentinel.

    Avoid float-with-NaN unit conversion: one actual prefreeze pandas2.3.3 /
    NumPy2.0.2 pilot raised an intermittent overflow there. Its complete modern
    replay contained only valid integer days/−1, and2,200 subsequent exact-array
    conversions did not reproduce the exception. This conversion never rounds
    or invents a date and rejects corrupt/noninteger/out-of-range source days.
    """
    last=np.asarray(last,dtype='float64');limit=int((UPPER-pd.Timestamp('1970-01-01')).days)
    if not np.isfinite(last).all() or not np.equal(last,np.floor(last)).all() or np.any((last<0)&(last!=-1)) or np.any(last>limit):raise ValueError('Invalid exact source observation day')
    ns=np.full(len(last),np.iinfo(np.int64).min,dtype='int64');good=last>=0
    ns[good]=last[good].astype('int64')*86400000000000
    return pd.DatetimeIndex(ns.view('datetime64[ns]'))


def compact_from_heads(queries,heads,support_index):
    q=normalize_queries(queries);v={h:r['prediction'] for h,r in heads.items()};d={}
    for z in A.ZONES:
        d[z+'_fga100']=v[z+'_fga100'];d[z+'_pct']=v[z+'_pct'];d[z+'_fgm100']=v[z+'_fga100']*v[z+'_pct']
    for tag,zones in [('mid',A.ZONES[1:3]),('three',A.ZONES[3:])]:
        for stat in ['fga100','fgm100']:d[tag+'_'+stat]=sum(d[z+'_'+stat].astype(float) for z in zones)
        den=d[tag+'_fga100'];d[tag+'_pct']=np.divide(d[tag+'_fgm100'],den,out=np.full(len(q),np.nan),where=den>0)
    for h in ['fta100','ft_pct','tov100','oreb100','dreb100','stl100','blk100','foul_shooting100']:d[h]=v[h]
    d['ftm100']=v['fta100']*v['ft_pct'];d['ast100']=sum(v['assists_'+z+'100'].astype(float) for z in A.ZONES)
    d['usage_events100']=sum(d[z+'_fga100'].astype(float) for z in A.ZONES)+.44*v['fta100']+v['tov100']
    d['all_fouls100_total_possessions']=sum(v['foul_'+k+'100'].astype(float) for k in ['shooting','offensive','technical','flagrant','loose_ball','double','personal_group'])
    d['dreb_contest100']=np.where(q.query_season.le(2024),v['dreb_contest100__v1'],v['dreb_contest100__v2'])
    columns={}
    for name,r in RECORDS.items():
        prediction=d[r['source'][5:]];exposure=np.zeros(len(q));last=np.full(len(q),-1.);legal=np.zeros(len(q),bool)
        for era,take in [('through_2024',q.query_season.le(2024).to_numpy()),('from_2025',q.query_season.ge(2025).to_numpy())]:
            deps=[heads[h] for h in r['source_heads_by_era'][era]]
            e=np.stack([a['prior_exposure'] for a in deps],axis=1).min(axis=1);raw_days=np.stack([a['last_day'] for a in deps],axis=1);days=raw_days.max(axis=1);days[(raw_days<0).any(axis=1)]=-1
            okay=np.stack([a['source_supported']&a['player_observed'] for a in deps],axis=1).all(axis=1)
            exposure[take]=e[take];last[take]=days[take];legal[take]=okay[take]
        lastdate=day_dates(last)
        observed=legal&np.isfinite(prediction)&(exposure>0)&~pd.isna(lastdate)
        columns[name]=np.where(observed,prediction,np.nan);columns[name+'__exposure']=exposure;columns[name+'__last_observation_date']=lastdate
        columns[name+'__missing']=~observed;columns[name+'__days_since_observation']=(q.query_date-lastdate).dt.total_seconds()/86400
    for name in A.UNAVAILABLE:
        columns[name]=np.full(len(q),np.nan);columns[name+'__exposure']=np.zeros(len(q));columns[name+'__last_observation_date']=pd.Series(pd.NaT,index=q.index);columns[name+'__missing']=np.ones(len(q),bool);columns[name+'__days_since_observation']=np.full(len(q),np.nan)
    out=pd.concat([q,pd.DataFrame(columns),support_index.query(q)],axis=1);out['estimate_date']=q.query_date
    return out


class DailyFeatureStore:
    """Read-only fixed-source store; no held annual values or model refits."""
    def __init__(self,full_history_root,validated_source_root,work_root,*,admission_paths=()):
        self.root=Path(full_history_root).resolve();self.validated=Path(validated_source_root).resolve();self.work=Path(work_root).resolve()
        lock=json.loads((self.root/'run_lock.json').read_text());self.inputs={}
        def pin(p,expected=None):
            p=Path(p).resolve();h=sha(p)
            if expected is not None and h!=expected:raise ValueError('Frozen source hash mismatch: '+str(p))
            if str(p) in self.inputs and self.inputs[str(p)]!=h:raise ValueError('Conflicting daily input pins')
            self.inputs[str(p)]=h;return h
        self._pin=pin
        pin(self.root/'run_lock.json');pin(self.root/'observations.parquet',lock['observations_sha256']);pin(self.root/'registry.json',lock['registry_sha256'])
        folder=Path(__file__).parent
        for p in [Path(__file__),CPP,folder/'export_vpm_1997_features.py',folder/'export_full_history_annual_snapshots.py',folder/'build_full_skill_history.py',folder/'export_full_skill_history.py',folder/'full_skill_history_specs.py',folder/'prepare_full_skill_history.py',folder.parent/'config/full_skill_history_sources.json',self.validated/'manifest.json',self.validated/'player_game_lines.parquet',*map(Path,admission_paths)]:pin(p)
        pin(folder/'build_full_skill_history.py',lock['code_sha256'])
        pin(folder/'full_skill_history_kernel.cpp',lock['kernel_sha256'])
        source_manifest=json.loads((self.validated/'manifest.json').read_text())
        for name,digest in source_manifest['artifacts'].items():pin(self.validated/name,digest)
        self.original_library=self.root/'kernel'/('full_skill_'+lock['kernel_sha256'][:16]+'.so');pin(self.original_library)
        original=pd.read_parquet(self.root/'observations.parquet');A.validate_panel(original)
        validated=pd.read_parquet(self.validated/'player_game_lines.parquet')
        panel,self.masks=strict_observations(original,validated)
        support,self.minute_audit=A.observation_support(panel,validated);self.support=SupportIndex(support)
        columns=list(dict.fromkeys(['player_id','game_id','game_date','season','day',*[c for s in SPECS for c in [s.numerator,s.denominator]]]))
        self.panel=panel[columns].copy();self.calendar_source=panel[['player_id','game_date','season']].copy()
        if self.panel.groupby(['player_id','game_date']).season.nunique().gt(1).any():raise ValueError('Same-date source season ambiguity')
        self.library=compile_kernel(self.work/'kernel');pin(self.library);self.recipes={};self.registry=[]
        lock_sha=sha(self.root/'run_lock.json')
        for year in range(1997,2027):
            self.recipes[year]={}
            for spec in SPECS:
                path=self.root/'models'/str(year)/(spec.name+'.json');digest=pin(path);r=json.loads(path.read_text())
                if r.get('skill')!=spec.name or r.get('run_lock_sha256')!=lock_sha:raise ValueError('Annual parameter source identity changed')
                validate_recipe(r,year,A.annual_report_cut(year));self.recipes[year][spec.name]=r
                self.registry.append(dict(head_name=spec.name,parameter_vintage=year,model_path=str(path.resolve()),model_sha256=digest,parameter_bootstrap=r['status']=='bootstrap',parameter_last_training_observation=r.get('last_training_observation'),parameters=r['parameters'],mean=r['mean'],status=r['status']))
        self.verify_inputs()

    def verify_inputs(self):
        for p,h in self.inputs.items():
            if sha(p)!=h:raise ValueError('Daily source/code changed: '+p)

    def query_heads(self,queries):
        q=normalize_queries(queries);heads={h:{k:np.full(len(q),False if k in ['source_supported','player_observed'] else np.nan,dtype=bool if k in ['source_supported','player_observed'] else float) for k in ['prediction','prior_exposure','last_day','source_supported','player_observed']} for h in HEAD_NAMES}
        for vintage,batch in q.groupby('parameter_vintage',sort=True):
            ix=batch.index.to_numpy();batch=batch.reset_index(drop=True);engine=QueryEngine(self.panel,batch,self.library)
            for spec in SPECS:
                recipe=self.recipes[int(vintage)][spec.name]
                validate_recipe(recipe,int(vintage),batch.query_date.min())
                values=engine.run(spec,recipe,self.original_library)
                for k,v in values.items():heads[spec.name][k][ix]=v
        return heads

    def query(self,queries,*,include_heads=True):
        q=normalize_queries(queries)
        if len(q)==0:return pd.DataFrame(columns=[*KEYS,*A.COMPACT_SOURCES,*A.UNAVAILABLE,'ctx_mpg','estimate_date'])
        heads=self.query_heads(q);out=compact_from_heads(q,heads,self.support)
        if include_heads:
            primitive={}
            for name,r in heads.items():
                for key in ['prediction','prior_exposure','last_day','source_supported','player_observed']:primitive['head__'+name+'__'+key]=r[key]
            out=pd.concat([out,pd.DataFrame(primitive)],axis=1)
        return out

    def export(self,queries,out,*,batch_rows=100000,minimum_free_bytes=15*1024**3,resume=False):
        """Exclusive export/resume: never race a healthy registered writer."""
        out=Path(out).resolve();lock=out.with_name(out.name+'.export.lock')
        with lock.open('a+') as held:
            try:fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('Another healthy daily export holds '+str(lock)) from exc
            try:return self._export_locked(queries,out,batch_rows=batch_rows,minimum_free_bytes=minimum_free_bytes,resume=resume)
            finally:fcntl.flock(held.fileno(),fcntl.LOCK_UN)

    def _export_locked(self,queries,out,*,batch_rows=100000,minimum_free_bytes=15*1024**3,resume=False):
        """Exact registered resume; a partial unreceipted partition fails closed.

        Completed files are never overwritten. A killed atomic write remains a
        .partial diagnostic; use a new version for that partition, not deletion.
        """
        out=Path(out).resolve();q=normalize_queries(queries)
        if batch_rows<1:raise ValueError('Positive bounded batch size required')
        if shutil.disk_usage(out.parent).free<minimum_free_bytes:raise OSError('At least15GiB reserve required before export')
        self.verify_inputs()
        schema=dict(COMPACT_SCHEMA,profile=PROFILE,query_keys=KEYS,parameter_calendar='hold latest annual report recipe;2020-Nov1;bootstrap1997 beforefirstreport',query_season='explicit native NBA ending season; not parameter vintage')
        registry=dict(profile=PROFILE,records=self.registry)
        def atomic_json(path,value):
            temporary=path.with_name(path.name+'.partial')
            if path.exists() or temporary.exists():raise FileExistsError('Never overwrite complete or partial artifact: '+str(path))
            write_json(temporary,value);temporary.replace(path)
        def atomic_parquet(path,frame):
            temporary=path.with_name(path.name+'.partial')
            if path.exists() or temporary.exists():raise FileExistsError('Never overwrite complete or partial partition: '+str(path))
            frame.to_parquet(temporary,index=False,compression='zstd');temporary.replace(path)
        expected=dict(status='registered_daily_fixed_source_export_before_partitions',profile=PROFILE,input_hashes=self.inputs,rows=len(q),maximum_batch_rows=batch_rows,numeric_storage='float64_no_quantization',minimum_free_bytes=minimum_free_bytes)
        if out.exists():
            if not resume:raise FileExistsError('Existing daily root requires explicit --resume')
            registration=json.loads((out/'registration.json').read_text())
            for key,value in expected.items():
                if registration.get(key)!=value:raise ValueError('Resume registration mismatch: '+key)
            if sha(out/'queries.parquet')!=registration['query_sha256']:raise ValueError('Resume query bytes changed')
            pd.testing.assert_frame_equal(pd.read_parquet(out/'queries.parquet'),q,check_exact=True)
            for name,h in registration['metadata_hashes'].items():
                if sha(out/name)!=h:raise ValueError('Resume metadata changed: '+name)
            if json.loads((out/'compact_schema.json').read_text())!=schema or json.loads((out/'head_parameter_registry.json').read_text())!=registry:raise ValueError('Resume schema/parameter registry changed')
            if list(out.rglob('*.partial')):raise ValueError('Incomplete atomic partition preserved; new partition/version required')
        else:
            out.mkdir(parents=True)
            atomic_parquet(out/'queries.parquet',q);atomic_json(out/'compact_schema.json',schema);atomic_json(out/'head_parameter_registry.json',registry)
            registration=dict(expected,query_sha256=sha(out/'queries.parquet'),metadata_hashes={n:sha(out/n) for n in ['compact_schema.json','head_parameter_registry.json']})
            atomic_json(out/'registration.json',registration)
        parts=[];start=time.monotonic();completed=(out/'receipt.json').exists()
        for season,batch in q.groupby('query_season',sort=True):
            batch=batch.sort_values(['query_date','player_id'],kind='stable').reset_index(drop=True)
            for i in range(0,len(batch),batch_rows):
                piece=batch.iloc[i:i+batch_rows].reset_index(drop=True)
                dest=out/'partitions'/str(int(season))/f'{i//batch_rows:04d}.parquet';meta=dest.with_suffix('.json')
                if dest.exists() or meta.exists():
                    if not (dest.exists() and meta.exists()):raise ValueError('Unreceipted/incomplete partition preserved; version required: '+str(dest))
                    p=json.loads(meta.read_text())
                    if p['path']!=str(dest) or p['sha256']!=sha(dest) or p['rows']!=len(piece) or p['bytes']!=dest.stat().st_size:raise ValueError('Completed partition hash/identity changed')
                    got=pd.read_parquet(dest,columns=KEYS);pd.testing.assert_frame_equal(got,piece,check_exact=True)
                    if p['first_date']!=str(piece.query_date.min().date()) or p['last_date']!=str(piece.query_date.max().date()):raise ValueError('Completed partition date receipt changed')
                else:
                    if completed:raise ValueError('Final receipt exists but partition missing')
                    if shutil.disk_usage(out).free<minimum_free_bytes:raise OSError('15GiB reserve reached; preserve partial immutable root')
                    now=time.monotonic();frame=self.query(piece);dest.parent.mkdir(parents=True,exist_ok=True)
                    atomic_parquet(dest,frame)
                    p=dict(path=str(dest),sha256=sha(dest),rows=len(frame),bytes=dest.stat().st_size,first_date=str(piece.query_date.min().date()),last_date=str(piece.query_date.max().date()),seconds=time.monotonic()-now)
                    atomic_json(meta,p);print(json.dumps(dict(stage='daily_feature_partition',**p)),flush=True)
                parts.append(p)
        expected_paths={p['path'] for p in parts}
        if {str(p) for p in (out/'partitions').rglob('*.parquet')}!=expected_paths:raise ValueError('Unexpected extra partition census')
        self.verify_inputs()
        if completed:
            result=json.loads((out/'receipt.json').read_text())
            if result['partitions']!=parts or result['input_hashes']!=self.inputs:raise ValueError('Final receipt changed registered partition/source identity')
            for p,h in result['output_hashes'].items():
                if sha(p)!=h:raise ValueError('Final output bytes changed')
            return result
        result=dict(status='passed_daily_compact_pure_feature_queries',profile=PROFILE,query_rows=len(q),query_dates=int(q.query_date.nunique()),source_heads=len(SPECS),compact_features=24,parameter_calendar='held annual report vintage;2020-Nov1 exception;bootstrap1997 beforefirstreport',query_season_separate_from_parameter_vintage=True,synthetic_queries_never_mutate_source_state=True,actual_source_zero_opportunity_rows_preserve_frozen_semantics=True,parameter_transport_not_refit=True,cross_stat_coefficients_used=False,source_v2_masks_retained=True,mask_changes=self.masks,minute_admission=self.minute_audit,source_exposure_not_complete_career=True,numeric_storage='float64_zstd_no_quantization',selected_model_changed=False,protected_2027_inputs=0,total_bytes=sum(p['bytes'] for p in parts),seconds=time.monotonic()-start,partitions=parts,input_hashes=self.inputs,output_hashes={str(p.resolve()):sha(p) for p in out.rglob('*') if p.is_file()})
        atomic_json(out/'receipt.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full-history-root',type=Path,required=True);p.add_argument('--validated-source-root',type=Path,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-rows',type=int,default=100000);p.add_argument('--resume',action='store_true')
    a=p.parse_args();q=pd.read_parquet(a.queries);store=DailyFeatureStore(a.full_history_root,a.validated_source_root,a.work_root);store.inputs[str(a.queries.resolve())]=sha(a.queries);r=store.export(q,a.out,batch_rows=a.batch_rows,resume=a.resume);print(json.dumps({k:r[k] for k in ['status','query_rows','query_dates','total_bytes','seconds']}))
if __name__=='__main__':main()

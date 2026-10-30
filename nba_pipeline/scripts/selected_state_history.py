"""Original selected-VPM dated input bank, without changing its estimators.

Core/rim/DFGA states are neutral POSTGAME states held until the next measured
state (730-day stale mask), not date-evolved synthetic OU observations. The
tracking population is the original selected measurement population. No fits.
"""
from __future__ import annotations
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'selected_state_full_bank_history_v1'
OFF = ['own_rim_makes100','own_rim_fga_rate','own_mid_makes100','own_mid_fga_rate',
       'own_three_makes100','own_three_fga_rate','own_ft_makes100','own_fta_rate',
       'own_ast_rate','own_tov_rate','own_oreb_rate','own_usg_pct','ctx_mpg']
DEF = ['own_blk_rate','own_stl_rate','own_rim_saved100','own_rim_contests100','own_dfga100',
       'def_shooting_pf100','def_nonoff_pf100','def_deflections100','def_off_fouls_drawn100',
       'def_contested_dreb100','def_contested_shots100','ctx_mpg']
TRACKING = ['def_'+s+'100' for s in ['deflections','contested_dreb','contested_shots',
                                    'charges_drawn','off_fouls_drawn','shooting_pf','nonoff_pf']]
EARLY_COMPATIBLE = {
 'own_rim_fga_rate':'hist__rim_fga100','own_rim_makes100':'hist__rim_fgm100','own_rim_pct':'hist__rim_pct',
 'own_mid_fga_rate':'hist__mid_fga100','own_mid_makes100':'hist__mid_fgm100','own_mid_pct':'hist__mid_pct',
 'own_three_fga_rate':'hist__three_fga100','own_three_makes100':'hist__three_fgm100','own_three_pct':'hist__three_pct',
 'own_fta_rate':'hist__fta100','own_ft_makes100':'hist__ftm100','own_ft_pct':'hist__ft_pct',
 'own_ast_rate':'hist__ast100','own_tov_rate':'hist__tov100','own_oreb_rate':'hist__oreb100',
 'own_usg_pct':'hist__usage_events100','own_stl_rate':'hist__stl100','own_blk_rate':'hist__blk100',
 'own_dreb_rate':'hist__dreb100', 'def_contested_dreb100':'hist__contested_dreb100'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def normalize_schedule(frame):
    required=['query_id','player_id','query_date','query_season']
    if not set(required)<=set(frame):raise ValueError('Explicit query identity/date/NBA season required')
    q=frame[required].copy().reset_index(drop=True)
    if q.empty or q.query_id.isna().any() or q.query_id.duplicated().any():raise ValueError('Unique nonempty query IDs required')
    for c in ['player_id','query_season']:
        x=pd.to_numeric(q[c],errors='raise').to_numpy(float)
        if not np.isfinite(x).all() or (x!=np.floor(x)).any():raise ValueError('Exact integral query IDs/seasons required')
        q[c]=x.astype(np.int64)
    if q.player_id.le(0).any() or not q.query_season.between(1997,2026).all():raise ValueError('Only positive IDs/completed 1997–2026 NBA seasons')
    q.query_date=pd.to_datetime(q.query_date,errors='raise')
    if q.query_date.dt.tz is not None or q.query_date.isna().any() or not q.query_date.eq(q.query_date.dt.normalize()).all():raise ValueError('Naive calendar dates required')
    if not q.query_date.between('1996-11-01','2026-10-01').all():raise ValueError('Outside registered completed-data cuts')
    if q.duplicated(['player_id','query_date']).any() or q.groupby('query_date').query_season.nunique().gt(1).any():raise ValueError('One row/player/date and one explicit NBA season/date')
    return q


def vintage_at(cut):
    cut=pd.Timestamp(cut)
    legal=[y for y in range(2015,2026) if pd.Timestamp(f'{y}-12-01' if y==2020 else f'{y}-10-01')<=cut]
    return max(legal) if legal else None


def held_snapshot(states, players, cut, *, max_age_days=730):
    """Pure terminal lookup; queries never update a source event or its clock."""
    cut=pd.Timestamp(cut);ids=pd.Index(np.asarray(players,np.int64),name='player_id')
    if ids.duplicated().any() or (ids<=0).any():raise ValueError('Unique positive query addresses required')
    s=states.copy();s.game_date=pd.to_datetime(s.game_date)
    if s.duplicated(['player_id','game_id']).any():raise ValueError('Duplicate source states')
    s=s.loc[s.game_date.lt(cut)].sort_values(['player_id','game_date'],kind='stable').groupby('player_id',sort=False).tail(1).set_index('player_id').reindex(ids)
    s['last_observation_date']=s.game_date;s['age_days']=(cut-s.game_date).dt.days
    s['is_missing']=s.age_days.gt(max_age_days)|s.value.isna()
    s.loc[s.is_missing,['value','state_variance']]=np.nan
    s['estimate_date']=cut
    return s.reset_index()


def tracking_snapshot(measurements,players,cut,*,half_life=550.,prior=1000.):
    """Literal original selected TD550/K1000, full-prefix league population."""
    d=measurements.loc[pd.to_datetime(measurements.game_date).lt(pd.Timestamp(cut))].copy()
    result=pd.DataFrame({'player_id':np.asarray(players,np.int64)});history=[]
    for field in TRACKING:
        g=d.loc[d.feature.eq(field)].copy();w=np.exp2(-(pd.Timestamp(cut)-pd.to_datetime(g.game_date)).dt.total_seconds()/86400/half_life)
        g['n']=g.numerator*w;g['e']=g.denominator*w
        sums=g.groupby('player_id')[['n','e']].sum().reindex(result.player_id)
        league=float(g.n.sum()/g.e.sum()) if len(g) and g.e.sum()>0 else 0.
        value=(100*(sums.n+prior*league)/(sums.e+prior)).where(sums.e.gt(0))
        result[field]=value.to_numpy();result[field+'__exposure']=sums.e.to_numpy()
        last=g.groupby('player_id').game_date.max().reindex(result.player_id)
        history.append(pd.DataFrame({'player_id':result.player_id,'value':value.to_numpy(),'is_missing':value.isna().to_numpy(),'feature':field,'last_observation_date':last.to_numpy(),'observed_exposure':sums.e.to_numpy()}))
    return result,pd.concat(history,ignore_index=True)


MASK_FIELDS=('ast','blk','dreb','fga','fgm','fta','ftm','mid_fga','mid_fgm','oreb','pts','rim_fga','rim_fgm','stl','three_fga','three_fgm','tov')
FOREIGN_SHA='d154b01686edffa6b65b49485c983f15780b27b425d9d7e6d9155d1c88106384'
MASK_SHA='4623f58273c8144b912c4e326f4ef4554cbd0e22f36ff9901e0ad174a646552b'

def _keys(frame):
    return pd.MultiIndex.from_arrays([frame.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10),frame.player_id.astype(np.int64)])

def mask_legacy_source(lines,foreign,masks,*,context=None,require_full_census=False):
    """Only declared identities/count-fields; never copy replacement counts/E.

    Duplicate field masks retain all reasons in the coverage report. Unknown
    legacy identities are reported absent, not synthesized or assigned zeros.
    """
    if lines.duplicated(['player_id','game_id']).any():raise ValueError('Duplicate original source identity')
    fk=_keys(foreign)
    if fk.duplicated().any():raise ValueError('Duplicate phantom identities')
    keep=~_keys(lines).isin(fk)
    if require_full_census:
        if len(foreign)!=49 or foreign.game_id.nunique()!=19 or (~keep).sum()!=49:raise ValueError('Independent phantom census must remain exactly49rows/19games')
        if not ((lines.loc[~keep,'off_poss']+lines.loc[~keep,'def_poss'])>0).all():raise ValueError('Phantom census requires positive legacy exposure')
    d=lines.loc[keep].copy();index=_keys(d);report=[]
    if not set(masks.field)<=set(MASK_FIELDS):raise ValueError('Unregistered field map; no foul or denominator substitutions')
    for field in MASK_FIELDS:
        if field not in d:raise ValueError('Original primitive count field missing: '+field)
        g=masks.loc[masks.field.eq(field)].copy();keys=_keys(g);location=index.get_indexer(keys);matched=location>=0;unique=np.unique(location[matched])
        values=d.iloc[location[matched]][field].to_numpy(float);source=g.loc[matched,'source_value'].to_numpy(float)
        report.append(dict(original_field=field,mask_field=field,records=len(g),unique_identity_fields=int(keys.nunique()),matched_records=int(matched.sum()),unmatched_records=int((~matched).sum()),unmatched_reason='identity_absent_from_retained_legacy_source',masked_original_fields=len(unique),original_finite=int(np.isfinite(d.iloc[unique][field].to_numpy(float)).sum()),original_vs_validated_counter_different=int((~np.isclose(values,source,equal_nan=True)).sum()),reasons=g.reason.value_counts().to_dict()))
        d.iloc[unique,d.columns.get_loc(field)]=np.nan
    if 'usg_events' in d:d['usg_events']=d.fga+.44*d.fta+d.tov
    ctx=None if context is None else context.loc[~_keys(context).isin(fk)].copy()
    return d,ctx,dict(phantom_rows_removed=int((~keep).sum()),phantom_games=int(lines.loc[~keep,'game_id'].nunique()),field_map=report,denominators_changed=False,validated_counter_values_copied=False)


def decayed_rim_quality_support(frame,half_life):
    """Post-observation quality support using the selected suppression half-life."""
    if not np.isfinite(half_life) or half_life<=0:raise ValueError('Positive quality half-life required')
    result=np.zeros(len(frame));days=(pd.to_datetime(frame.game_date)-pd.Timestamp('1970-01-01')).dt.days.to_numpy();values=frame.rim_dfga.to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Valid observed rim contest counts required')
    for indices in frame.groupby('player_id',sort=False).indices.values():
        last=None;mass=0.
        for i in indices:
            if last is not None and days[i]<last:raise ValueError('Source events must remain chronological within player')
            mass=mass*(2**(-(days[i]-last)/half_life) if last is not None else 0.)+values[i];result[i]=mass;last=days[i]
    return result


def mpg_snapshot(lines,players,cut):
    cut=pd.Timestamp(cut);g=lines.loc[pd.to_datetime(lines.game_date).lt(cut)&pd.to_datetime(lines.game_date).ge(cut-pd.Timedelta(days=365))&lines.season_type.eq('rs')&lines.minutes.gt(0)]
    x=g.groupby('player_id').agg(ctx_mpg=('minutes','mean'),ctx_games=('game_id','size'),last_observation_date=('game_date','max')).reindex(players)
    return x.reset_index(names='player_id')


def compatible_early_packet(frame):
    """Only matching physical concepts; never relabel total-possession foul counts.

    Input is an explicitly supplied, already prefix-audited historical packet.
    Every mapped value still carries its different source/filter provenance.
    """
    d=frame.copy();result=d[[c for c in ['query_id','player_id','query_date','query_season','estimate_date'] if c in d]].copy();rows=[]
    for dest,source in EARLY_COMPATIBLE.items():
        if source not in d:continue
        ex=source+'__exposure';last=source+'__last_observation_date'
        if ex not in d or last not in d:raise ValueError('Early physical field lacks genuine support/date: '+source)
        dates=pd.to_datetime(d[last]);cut=pd.to_datetime(d['query_date'] if 'query_date' in d else d['estimate_date'])
        good=d[source].notna()&pd.to_numeric(d[ex]).gt(0)&dates.notna()&dates.lt(cut)
        if (dates.notna()&dates.ge(cut)).any():raise ValueError('Early source reaches query date')
        result[dest]=d[source].where(good)
        rows.append(pd.DataFrame({'player_id':d.player_id,'estimate_date':cut,'feature':dest,'source_column':source,'source_epoch':'early_full_history_physical_bridge','estimator':'independent_full_history_transport_not_selected_OU','last_observation_date':dates,'observed_exposure':d[ex],'is_missing':~good}))
    if 'ctx_mpg' in d:
        for col in ['ctx_games','ctx_mpg__last_observation_date']:
            if col not in d:raise ValueError('Early MPG support missing')
        cut=pd.to_datetime(d['query_date'] if 'query_date' in d else d['estimate_date']);last=pd.to_datetime(d.ctx_mpg__last_observation_date)
        if (last.notna()&last.ge(cut)).any():raise ValueError('Early MPG reaches query date')
        good=d.ctx_games.gt(0)&last.lt(cut)&d.ctx_mpg.notna();result['ctx_mpg']=d.ctx_mpg.where(good)
        rows.append(pd.DataFrame({'player_id':d.player_id,'estimate_date':cut,'feature':'ctx_mpg','source_column':'ctx_mpg','source_epoch':'early_validated_RS_minutes','estimator':'literal_prior365_RS_positive_appearance_mean','last_observation_date':last,'observed_exposure':d.ctx_games,'is_missing':~good}))
    return result,pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def combine_eras(selected_packet,early_packet):
    """Fill only pre-Oct2015 missing selected concepts from an audited bridge.

    Existing original-tracking observations take precedence, even early. No
    early estimator fills a modern missing/stale original-selected state.
    """
    keys=['player_id','estimate_date'];a=selected_packet.copy();b=early_packet.copy()
    for frame in [a,b]:
        if 'estimate_date' not in frame and 'query_date' in frame:frame['estimate_date']=pd.to_datetime(frame.query_date)
        frame['estimate_date']=pd.to_datetime(frame.estimate_date)
        if frame.duplicated(keys).any():raise ValueError('Duplicate era source query')
    cols=[c for c in list(EARLY_COMPATIBLE)+['ctx_mpg'] if c in b]
    out=a.merge(b[keys+cols],on=keys,how='left',suffixes=('','__early'),validate='one_to_one')
    early=out.estimate_date.lt('2015-10-01')
    for c in cols:
        if c not in a:out[c]=out[c].where(early)
        else:
            out[c]=out[c].where(~(early&out[c].isna()),out[c+'__early']);out=out.drop(columns=c+'__early')
    for c in [c for c in out if c.startswith('own_') and not c.endswith('_missing')]:out[c+'_missing']=out[c].isna().astype(int)
    return out


class SelectedStateHistory:
    """Frozen original sources; one real-event replay per annual vintage/skill.

    No filter fitting, synthetic source rows, future-ID eligibility, clean-source
    substitution, or full-history materialization. Cache reuse verifies inputs.
    """
    def __init__(self,cache_root,*,pipeline_root=ROOT,source_mode="legacy_anchor"):
        import build_monthly_own_skills as C
        import build_monthly_defensive_skills as D
        if Path(pipeline_root).resolve()!=Path(C.ROOT).resolve():raise ValueError('Original selected source paths must remain explicit/canonical')
        if source_mode not in ['legacy_anchor','identity_cleaned']:raise ValueError('Explicit registered selected source mode required')
        self.source_mode=source_mode;self.foreign=self.masks=None;self.cleaning_audit=None
        self.C,self.D=C,D;self.root=Path(pipeline_root);self.cache=Path(cache_root).resolve();self.cache.mkdir(parents=True,exist_ok=True)
        self.core=C.OUT;self.defense=D.OUT;self._states={};self._panels={};self._inputs={};self._models={}
        self.lines=self._read_pinned(C.SOURCE/'player_game_lines.parquet',pd.read_parquet);self.lines.game_date=pd.to_datetime(self.lines.game_date)
        if self.lines.season.gt(2026).any() or self.lines.game_date.ge('2026-10-02').any():raise ValueError('Protected/future source')
        self.measurement_path=self.root/'results/estimated_skills/defense_bank_v51_shooting_pf_repaired/measurements.parquet'
        self.measurements=self._read_pinned(self.measurement_path,pd.read_parquet);self.measurements.game_date=pd.to_datetime(self.measurements.game_date)
        if self.measurements.game_date.ge('2026-10-02').any():raise ValueError('Protected tracking source')
        for p in [Path(__file__),C.SOURCE/'player_game_lines.parquet',C.OUT/'game_context.parquet',C.AGE,self.measurement_path,
                  Path(C.__file__),Path(D.__file__),Path(C.B.__file__),Path(C.X.__file__),Path(D.T.__file__),
                  self.defense/'rim_counts.parquet',self.defense/'dfga_counts.parquet',C.KERNEL_SOURCE,
                  inspect.getsourcefile(C.build_corrections),inspect.getsourcefile(C.FastOU),
                  inspect.getsourcefile(C.X.expanding_linear_context)]:self._pin(p)
        if source_mode=='identity_cleaned':
            foreign_path=self.root/'results/estimated_skills/pregame_source_validated_v1/foreign_legacy_player_games.csv';mask_path=self.root/'results/estimated_skills/pregame_source_validated_v2/observation_masks.csv'
            if sha(foreign_path)!=FOREIGN_SHA or sha(mask_path)!=MASK_SHA:raise ValueError('Exact independently admitted identity/mask bytes required')
            self.foreign=self._read_pinned(foreign_path,lambda p:pd.read_csv(p,dtype={'game_id':str}),expected=FOREIGN_SHA);self.masks=self._read_pinned(mask_path,lambda p:pd.read_csv(p,dtype={'game_id':str}),expected=MASK_SHA)
            self.lines,_,self.cleaning_audit=mask_legacy_source(self.lines,self.foreign,self.masks,require_full_census=True)
            foreign_keys=_keys(self.foreign);tracking_hit=_keys(self.measurements).isin(foreign_keys)
            hits={'tracking_rows':int(tracking_hit.sum()),'tracking_by_feature':self.measurements.loc[tracking_hit,'feature'].value_counts().to_dict(),'contested_shots_total_rows':int((tracking_hit&self.measurements.feature.eq('def_contested_shots100')).sum())}
            self.measurements=self.measurements.loc[~tracking_hit].copy()
            for name,filename in [('rim_rows','rim_counts.parquet'),('dfga_rows','dfga_counts.parquet')]:
                source=self._read_pinned(self.defense/filename,lambda p:pd.read_parquet(p,columns=['player_id','game_id']));hits[name]=int(_keys(source).isin(foreign_keys).sum())
            self.cleaning_audit['other_source_identity_hits']=hits
            for p in [foreign_path,mask_path,self.root/'results/estimated_skills/pregame_source_validated_v1/manifest.json',self.root/'results/estimated_skills/pregame_source_validated_v2/manifest.json']:self._pin(p)
        self._source_inputs=dict(self._inputs)
        self.source_dates=self.lines.groupby('game_date').season.agg(['min','max'])
        if self.source_dates['min'].ne(self.source_dates['max']).any():raise ValueError('Ambiguous native NBA season/date')

    def _read_pinned(self,path,reader,*,expected=None):
        before=self._pin(path)
        if expected is not None and before!=expected:raise ValueError('Independently admitted source bytes differ before read')
        data=reader(Path(path))
        after=self._pin(path)
        if before!=after or (expected is not None and after!=expected):raise ValueError('Source changed across read')
        return data

    def _pin(self,path):
        p=str(Path(path).resolve());h=sha(p)
        if p in self._inputs and self._inputs[p]!=h:raise ValueError('Selected input changed during adapter use: '+p)
        self._inputs[p]=h;return h

    def _model(self,year,skill):
        p=self.core/'models'/str(year)/(skill+'.json') if skill!='defensive' else self.defense/'models'/f'{year}.json'
        self._pin(p);model=json.loads(p.read_text());self._models[str(p.resolve())]=sha(p);return model

    def _end(self,year):
        return self.C.vintage_cut(year+1) if year<2025 else pd.Timestamp('2026-10-02')

    def _core_states(self,year,skill,model):
        C=self.C;end=self._end(year)
        if year not in self._panels:self._panels={year:C.load_panel(C.OUT,end)}
        df,ctx=self._panels[year]
        if self.source_mode=='identity_cleaned':df,ctx,_=mask_legacy_source(df,self.foreign,self.masks,context=ctx)
        spec=next(s for s in C.B.SKILLS if s.name==skill);sub=C.B.skill_subpanel(df.loc[np.isfinite(df[spec.numer])&df[spec.numer].ge(0)].copy(),spec) if self.source_mode=='identity_cleaned' else C.B.skill_subpanel(df,spec)
        C.B.WARMUP_SEASON=int(sub.season.min());league,_=C.B.league_center(sub,spec);centers=league if model['centered'] else np.zeros(len(sub))
        mr=0. if model['centered'] else model['mean_rate'];obs=sub.obs_rate.to_numpy();response=obs-centers;engine=C.FastOU(sub,self.cache/'kernel')
        base=engine.run(response,model['stage_a']['params'],mr)[0]+centers;residual=np.where(sub.step_k.to_numpy()>0,obs-base,np.nan)
        absent=C.X.absent_teammates(df,ctx) if spec.kind=='rate' else None
        cols=C.context_columns(sub,spec,base,residual,league,absent,model['context_c'],model['reference_year'])
        context,_=C.X.expanding_linear_context(cols,residual,sub.exposure.to_numpy(),sub.season.to_numpy(),sub.step_k.to_numpy()>0,sorted(sub.season.unique())[1:])
        initial,shift,trans,_=C.build_corrections(sub,spec,pd.read_parquet(C.AGE),model['mean_rate'],sorted(sub.season.unique()))
        p=model['params'];_,_,post,var=engine.run(response,p[:6],mr,initial-centers,shift,trans,context,p[6])
        values=post+centers;bounded=np.clip(values,0,1) if spec.kind=='pct' else np.maximum(values,0);scale=1 if spec.kind=='pct' else 100
        states=sub[['player_id','game_id','game_date']].copy();states['value']=scale*bounded;states['state_variance']=var*scale**2;states['observed_exposure']=sub.groupby('player_id').exposure.cumsum()
        return states

    def _defensive_states(self,year,model):
        D=self.D;end=self._end(year);d=pd.read_parquet(self.defense/'dfga_counts.parquet');d=d.loc[d.game_date.lt(end)].copy()
        if self.source_mode=='identity_cleaned':d=d.loc[~_keys(d).isin(_keys(self.foreign))].copy()
        sub=D.T.subpanel(d,D.T.STATS['dfga']);m=model['dfga'];engine=self.C.FastOU(sub,self.cache/'kernel');_,_,post,var=engine.run(sub.obs_rate.to_numpy(),m['params'],m['mean'])
        states=sub[['player_id','game_id','game_date']].copy();states['value']=100*np.maximum(post,0);states['state_variance']=10000*var;states['observed_exposure']=sub.groupby('player_id').exposure.cumsum();states['feature']='own_dfga100';parts=[states]
        d=pd.read_parquet(self.defense/'rim_counts.parquet');d=d.loc[d.game_date.lt(end)].copy()
        if self.source_mode=='identity_cleaned':d=d.loc[~_keys(d).isin(_keys(self.foreign))].copy()
        d['league_pct']=D.lagged_league(d);m=model['rim'];s=m['selected'];values=D.RimEngine(d).run(m['volume_half'],s['half_life'],s['prior_attempts'],m['mean']);quality=decayed_rim_quality_support(d,s['half_life'])
        for name,col,ex in [('own_rim_saved100',5,'rim_dfga'),('own_rim_contests100',3,'def_poss'),('own_rim_suppression',4,'rim_dfga')]:
            states=d[['player_id','game_id','game_date']].copy();states['value']=values[:,col];states['state_variance']=np.nan;states['observed_exposure']=d.groupby('player_id')[ex].cumsum();states['feature']=name;states['quality_prior_exposure']=quality;parts.append(states)
        return pd.concat(parts,ignore_index=True)

    def poststates(self,year,skill):
        key=(int(year),skill)
        if key in self._states:return self._states[key]
        model=self._model(year,skill);model_path=self.core/'models'/str(year)/(skill+'.json') if skill!='defensive' else self.defense/'models'/f'{year}.json'
        inputs=dict(self._source_inputs);inputs[str(model_path.resolve())]=sha(model_path)
        fingerprint=hashlib.sha256(json.dumps(dict(source_mode=self.source_mode,inputs=inputs),sort_keys=True).encode()).hexdigest()
        folder=self.cache/str(year);folder.mkdir(exist_ok=True);path=folder/(skill+'.parquet');receipt=folder/(skill+'.json')
        with (folder/(skill+'.lock')).open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            if receipt.exists():
                m=json.loads(receipt.read_text())
                if m['input_fingerprint']!=fingerprint or m['output_sha256']!=sha(path):raise ValueError('Stale/changed poststate cache; use a new root')
                states=pd.read_parquet(path)
            else:
                if path.exists():raise ValueError('Orphan poststate cache')
                states=self._defensive_states(year,model) if skill=='defensive' else self._core_states(year,skill,model)
                tmp=path.with_suffix(f'.{os.getpid()}.tmp');states.to_parquet(tmp,index=False);tmp.replace(path)
                _write(receipt,dict(schema=SCHEMA,source_mode=self.source_mode,input_fingerprint=fingerprint,input_hashes=inputs,output_sha256=sha(path),rows=len(states),new_fits=0))
        if self._states and next(iter(self._states))[0]!=year:self._states.clear()
        self._states[key]=states;return states

    def query_reference(self,cut,query_season,extra_player_ids=()):
        cut=pd.Timestamp(cut);ids=np.union1d(self.lines.loc[self.lines.game_date.lt(cut),'player_id'].to_numpy(np.int64),np.asarray(extra_player_ids,np.int64))
        q=pd.DataFrame(dict(query_id=[f'{cut.date()}:{i}' for i in ids],player_id=ids,query_date=cut,query_season=int(query_season)))
        return self.query(q)

    def query(self,schedule):
        q=normalize_schedule(schedule);packets=[];provenance=[]
        for cut,queries in q.groupby('query_date',sort=True):
            seasons=queries.query_season.unique();season=int(seasons[0]);known=self.source_dates.loc[cut] if cut in self.source_dates.index else None
            if known is not None and int(known['min'])!=season:raise ValueError('NBA season does not match source game-date metadata')
            ids=queries.player_id.to_numpy();packet=queries.copy();packet['estimate_date']=cut;year=vintage_at(cut)
            prefix=self.lines.loc[self.lines.game_date.lt(cut)]
            for side in ['off','def']:
                packet['source_prefix_'+side+'_exposure']=prefix.groupby('player_id')[side+'_poss'].sum().reindex(ids,fill_value=0).to_numpy()
            packet['source_prefix_exposure_estimator']='original_legacy_counter_prefix_lower_bound_'+self.source_mode;packet['source_prefix_exposure_is_complete_career']=False
            source_dates=prefix.groupby('player_id').game_date.agg(['min','max','size']).reindex(ids)
            packet['source_prefix_first_observation_date']=source_dates['min'].to_numpy();packet['source_prefix_last_observation_date']=source_dates['max'].to_numpy();packet['source_prefix_observed_games']=source_dates['size'].fillna(0).to_numpy()
            fields={};histories={}
            if year is not None:
                for skill in [s.name for s in self.C.B.SKILLS]+['defensive']:
                    model=self._model(year,skill);state=self.poststates(year,skill)
                    for feature,g in state.groupby('feature',sort=False) if skill=='defensive' else [('own_'+skill,state)]:
                        fit_end=pd.Timestamp(model['dfga' if feature=='own_dfga100' else 'rim']['fit_end']) if skill=='defensive' else pd.Timestamp(model['fit_end'])
                        if fit_end>=cut:raise ValueError('Parameter fit reaches query cut')
                        snap=held_snapshot(g,ids,cut);fields[feature]=snap.value.to_numpy()
                        if feature=='own_rim_suppression':
                            packet['own_rim_suppression__prior_exposure']=snap.quality_prior_exposure.where(~snap.is_missing).to_numpy();packet['rim_last_observation_date']=snap.last_observation_date.to_numpy()
                        snap['feature']=feature;snap['parameter_fit_end']=fit_end;snap['parameter_vintage']=year;snap['parameter_model_sha256']=sha(self.defense/'models'/f'{year}.json' if skill=='defensive' else self.core/'models'/str(year)/(skill+'.json'));snap['source_epoch']='selected_original_2014_2026';snap['estimator']='original_annual_neutral_postgame_held_730';histories[feature]=snap;provenance.append(snap)
            for name,values in fields.items():packet[name]=values
            tracking,h=tracking_snapshot(self.measurements,ids,cut)
            for c in tracking:
                if c!='player_id':packet[c]=tracking[c].to_numpy()
            h['estimate_date']=cut;h['source_epoch']='selected_original_tracking_measurements';h['estimator']='fixed_TD550_K1000';provenance.append(h)
            mpg=mpg_snapshot(self.lines,ids,cut);packet['ctx_mpg']=mpg.ctx_mpg.to_numpy();packet['ctx_games']=mpg.ctx_games.to_numpy()
            h=mpg.rename(columns={'ctx_games':'observed_exposure'});h['value']=h.ctx_mpg;h['is_missing']=h.value.isna();h['feature']='ctx_mpg';h['estimate_date']=cut;h['source_epoch']='selected_original_minutes';h['estimator']='literal_prior365_RS_positive_appearance_mean';provenance.append(h)
            for zone,attempt in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),('three','three_fga_rate'),('ft','fta_rate')]:
                if 'own_'+attempt in packet:
                    packet[f'own_{zone}_makes100']=packet['own_'+attempt]*packet[f'own_{zone}_pct']
                    ra,rb=histories['own_'+attempt],histories[f'own_{zone}_pct']
                    h=ra.copy();h['feature']=f'own_{zone}_makes100';h['value']=packet[f'own_{zone}_makes100'].to_numpy();h['is_missing']=h.value.isna();h['state_variance']=np.nan
                    h['last_observation_date']=pd.concat([ra.last_observation_date,rb.last_observation_date],axis=1).max(axis=1)
                    h['observed_exposure']=np.minimum(ra.observed_exposure.to_numpy(float),rb.observed_exposure.to_numpy(float));h['exposure_semantics']='minimum_dependency_support_gate_not_weight_mixed_denominators';h['derived_inputs']='own_'+attempt+' * '+f'own_{zone}_pct';provenance.append(h)
            for name in list(dict.fromkeys(OFF+DEF+['own_dreb_rate','own_pf_committed_rate'])):
                if name not in packet:packet[name]=np.nan
            for name in [c for c in packet if c.startswith('own_')]:packet[name+'_missing']=packet[name].isna().astype(int)
            if 'own_rim_suppression' not in packet:packet['own_rim_suppression']=np.nan
            if 'own_rim_suppression__prior_exposure' not in packet:packet['own_rim_suppression__prior_exposure']=np.nan
            if 'rim_last_observation_date' not in packet:packet['rim_last_observation_date']=pd.NaT
            finite=packet[['own_rim_saved100','own_rim_contests100','own_rim_suppression']].notna().all(axis=1)
            np.testing.assert_allclose(packet.loc[finite,'own_rim_saved100'],2*packet.loc[finite,'own_rim_contests100']*packet.loc[finite,'own_rim_suppression'],atol=1e-10,rtol=0)
            packet['feature_source_policy']='full_selected_bank_original_poststate_and_tracking';packets.append(packet)
        wide=pd.concat(packets,ignore_index=True);long=pd.concat(provenance,ignore_index=True,sort=False)
        long['is_missing']=~np.isfinite(pd.to_numeric(long.value,errors='raise'))
        long=long.merge(q.rename(columns={'query_date':'estimate_date'})[['query_id','player_id','estimate_date']],on=['player_id','estimate_date'],how='left',validate='many_to_one')
        for p,h in self._inputs.items():
            if sha(p)!=h:raise ValueError('Selected input changed during queries: '+p)
        receipt=dict(status='passed_original_selected_full_bank_queries',schema=SCHEMA,query_rows=len(wide),query_dates=int(wide.query_date.nunique()),input_hashes=dict(self._inputs),model_hashes=dict(self._models),new_fits=0,query_rows_update_states=False,selected_neutral_hold_not_pregame_evolution=True,source_mode=self.source_mode,identity_and_typed_count_mask_control=self.cleaning_audit,parameter_transport_on_changed_measurements=self.source_mode!='legacy_anchor',modern_source_clean_substitution=False,protected_2027_inputs=0)
        return wide,long,receipt

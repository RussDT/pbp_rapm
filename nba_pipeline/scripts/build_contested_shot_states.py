"""Validated contested-shot splits: original selected EB and independent OU.

The modern path preserves the original total's measurements, population and
TD550/K1000 estimator. The full-history OU path is a separate diagnostic; its
three independently filtered predictions need not add. No impact fit, source
repair, selected-pointer change or retrieval occurs here.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import export_vpm_1997_daily_features as D
from build_full_skill_history import Engine
from full_skill_history_specs import SKILLS
from export_full_history_annual_snapshots import sha,write_json,validate_recipe

PROFILE='validated_contested_shot_splits_original_EB_and_separate_OU_v1'
HEADS=['contested_shots100','contested_shots_2pt100','contested_shots_3pt100']
RAW=['report.hustle_contested_shots','report.hustle_contested_shots_2pt','report.hustle_contested_shots_3pt']
ARCHIVE=['hustle_CONTESTED_SHOTS','hustle_CONTESTED_SHOTS_2PT','hustle_CONTESTED_SHOTS_3PT']
MODERN=['def_'+h for h in HEADS]
SPECS=[next(s for s in SKILLS if s.name==h) for h in HEADS]


def normalize_source(frame):
    p=frame.copy()
    required={'player_id','game_id','game_date','season','def_poss',*RAW}
    if not required<=set(p):raise ValueError('Contested source identity/count/exposure fields required')
    p['game_date']=pd.to_datetime(p.game_date,errors='raise')
    ids=pd.to_numeric(p.player_id,errors='raise');season=pd.to_numeric(p.season,errors='raise')
    if (not np.isfinite(ids).all() or ids.le(0).any() or not np.equal(ids,np.floor(ids)).all()
        or not season.between(1997,2026).all() or not np.equal(season,np.floor(season)).all()
        or p.game_date.isna().any() or not p.game_date.eq(p.game_date.dt.normalize()).all()
        or not p.game_date.between(D.LOWER,D.UPPER-pd.Timedelta(days=1)).all()):
        raise ValueError('Protected/future/noncanonical contested source')
    p['player_id']=ids.astype('int64');p['season']=season.astype('int64')
    p['game_id']=p.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    if p.duplicated(['player_id','game_id']).any():raise ValueError('Ambiguous contested player/game source')
    p['day']=(p.game_date-pd.Timestamp('1970-01-01')).dt.days
    return p


def audit_observations(frame):
    p=normalize_source(frame);x=p[RAW].to_numpy(float);finite=np.isfinite(x)
    if not np.array_equal(finite,finite[:,[0]].repeat(3,axis=1)):raise ValueError('Partial contested two/three/total observation')
    if np.isinf(x).any():raise ValueError('Infinite contested source count')
    good=finite.all(axis=1)
    if ((x[good]<0).any() or not np.equal(x[good],np.floor(x[good])).all()
        or not np.equal(x[good,0],x[good,1]+x[good,2]).all()):raise ValueError('Invalid counts or contested2+3 total closure')
    admitted=good&np.isfinite(p.def_poss)&p.def_poss.gt(0)
    return dict(rows=len(p),observed_triplets=int(good.sum()),admissible_positive_denominator_rows=int(admitted.sum()),
        observed_but_unusable_denominator_rows=int((good&~admitted).sum()),maximum_raw_split_closure_gap=0.,
        first_observation=str(p.loc[good,'game_date'].min().date()) if good.any() else None,
        last_observation=str(p.loc[good,'game_date'].max().date()) if good.any() else None,
        zeros={h:int(np.sum(x[:,j]==0)) for j,h in enumerate(HEADS)},
        observed_by_season={str(y):int(n) for y,n in p.loc[good].groupby('season').size().items()},
        denominator='validated full_history_v2 def_poss; independent OU diagnostic only',
        definition_url='https://www.nba.com/stats/help/glossary',
        definition='Hustle closeout with raised hand before an opponent shot releases, split by two-point versus three-point shot. Not nearest-defender FGA or a rim-distance split.')


def modern_split_measurements(measurements,archive):
    """Attach exact original-source splits, retaining total denominator/population."""
    m=measurements.loc[measurements.feature.eq(MODERN[0])].copy()
    m['game_date']=pd.to_datetime(m.game_date);m['game_id']=m.game_id.astype(str).str.zfill(10)
    if m.duplicated(['player_id','game_id']).any() or m.duplicated(['player_id','game_date']).any():raise ValueError('Ambiguous original selected contested source')
    a=archive.copy();a['game_date']=pd.to_datetime(a.game_date)
    if a.duplicated(['player_id','game_date']).any():raise ValueError('Ambiguous original archive split identity')
    if ('season' in a and not a.season.between(1997,2026).all()) or not a.game_date.between(D.LOWER,D.UPPER-pd.Timedelta(days=1)).all():raise ValueError('Protected/future original archive')
    z=m.merge(a[['player_id','game_date',*ARCHIVE]],on=['player_id','game_date'],how='left',validate='one_to_one',indicator=True)
    x=z[ARCHIVE].to_numpy(float)
    if ((z._merge!='both').any() or not np.isfinite(x).all() or (x<0).any()
        or not np.equal(x,np.floor(x)).all() or not np.equal(x[:,0],x[:,1]+x[:,2]).all()
        or not np.equal(x[:,0],z.numerator.to_numpy(float)).all()):
        raise ValueError('Original selected total does not exactly match observed archive splits; retain aggregate control')
    if not (np.isfinite(z.denominator)&z.denominator.gt(0)).all():raise ValueError('Original selected denominator unavailable')
    result=z[['player_id','game_id','game_date','denominator']].copy()
    for j,h in enumerate(MODERN):result[h]=x[:,j]
    result['day']=(result.game_date-pd.Timestamp('1970-01-01')).dt.days
    return result,dict(rows=len(result),missing_or_mismatched_original_total_rows=0,maximum_raw_split_closure_gap=0.,
        denominator_policy='EXACT original selected measurement.denominator; never substitute API/full_history exposure',
        source_population_policy='EXACT original selected total measurement rows; no inferred observations',
        first_date=str(result.game_date.min().date()) if len(result) else None,last_date=str(result.game_date.max().date()) if len(result) else None)


def modern_query_states(measurements,queries):
    """Pure-date TD550/K1000 split queries; queries never commit any state.

    All three heads share physical observations, exposure and population.
    Consequently their EB predictions add naturally, without reconciliation.
    """
    q=D.normalize_queries(queries);d=measurements.copy()
    if d.duplicated(['player_id','game_id']).any():raise ValueError('Repeated modern measurement')
    d['game_date']=pd.to_datetime(d.game_date);x=d[MODERN].to_numpy(float);e=d.denominator.to_numpy(float)
    ids=pd.to_numeric(d.player_id,errors='raise')
    if not (np.isfinite(ids)&ids.gt(0)&np.equal(ids,np.floor(ids))).all():raise ValueError('Noncanonical modern source identity')
    if (not d.game_date.between(D.LOWER,D.UPPER-pd.Timedelta(days=1)).all() or not np.isfinite(x).all()
        or (x<0).any() or not np.equal(x[:,0],x[:,1]+x[:,2]).all()
        or not (np.isfinite(e)&(e>0)).all()):raise ValueError('Invalid/protected modern split observations')
    allids=pd.Index(np.union1d(q.player_id,d.player_id));position={int(p):i for i,p in enumerate(allids)}
    pn=np.zeros((len(allids),3));pe=np.zeros(len(allids));last=np.full(len(allids),-1,dtype='int64');gn=np.zeros(3);ge=0.;global_day=None
    result=np.full((len(q),3),np.nan);exposure=np.zeros(len(q));lastout=np.full(len(q),-1,dtype='int64')
    obs={pd.Timestamp(day):g for day,g in d.groupby('game_date',sort=False)};requests={pd.Timestamp(day):g for day,g in q.groupby('query_date',sort=False)}
    for day in sorted(set(obs)|set(requests)):
        number=int((day-pd.Timestamp('1970-01-01')).days)
        if day in requests:
            g=requests[day];p=np.array([position[int(v)] for v in g.player_id]);ix=g.index.to_numpy();decay=np.exp2(-(number-last[p])/550.);mass=pe[p]*decay
            estimate=100*(pn[p]*decay[:,None]+1000*(gn/ge if ge>0 else np.zeros(3)))/(mass[:,None]+1000)
            result[ix]=np.where((mass>0)[:,None],estimate,np.nan);exposure[ix]=mass;lastout[ix]=last[p]
        if day in obs:
            g=obs[day];p=np.array([position[int(v)] for v in g.player_id]);u=np.unique(p)
            decay=np.exp2(-(number-last[u])/550.);pn[u]*=decay[:,None];pe[u]*=decay
            np.add.at(pn,p,g[MODERN].to_numpy(float));np.add.at(pe,p,g.denominator.to_numpy(float));last[u]=number
            decay=1. if global_day is None else np.exp2(-(number-global_day)/550.);gn=gn*decay+g[MODERN].sum().to_numpy(float);ge=ge*decay+float(g.denominator.sum());global_day=number
    out=q.copy();lastdate=D.day_dates(lastout)
    for j,h in enumerate(MODERN):
        out[h]=result[:,j];out[h+'__exposure']=exposure;out[h+'__last_observation_date']=lastdate
        out[h+'__missing']=~np.isfinite(result[:,j]);out[h+'__days_since_observation']=(q.query_date-lastdate).dt.total_seconds()/86400
    finite=np.isfinite(result).all(axis=1)
    if np.max(np.abs(result[finite,0]-result[finite,1]-result[finite,2]),initial=0)>1e-9:raise ValueError('Shared-support TD550/K1000 split predictions fail closure')
    return out


def ou_query_states(panel,queries,recipes,query_library,original_library):
    """Independent saved OU transport, explicitly separate from selected EB."""
    p=normalize_source(panel);audit_observations(p);q=D.normalize_queries(queries);out=q.copy();provenance=[]
    predictions={h:np.full(len(q),np.nan) for h in HEADS}
    for h in HEADS:
        out['lib__'+h]=np.full(len(q),np.nan);out['lib__'+h+'__exposure']=np.zeros(len(q));out['lib__'+h+'__last_observation_date']=pd.Series(pd.NaT,index=out.index,dtype='datetime64[ns]')
    for vintage,batch in q.groupby('parameter_vintage',sort=True):
        ix=batch.index.to_numpy();engine=D.QueryEngine(p,batch.reset_index(drop=True),query_library)
        for spec in SPECS:
            recipe=recipes[int(vintage)][spec.name];validate_recipe(recipe,int(vintage),batch.query_date.min())
            v=engine.run(spec,recipe,original_library);predictions[spec.name][ix]=v['prediction']
            name='lib__'+spec.name
            for column,fill in [(name,np.nan),(name+'__exposure',0.)]:
                if column not in out:out[column]=fill
            out.loc[ix,name]=v['prediction'];out.loc[ix,name+'__exposure']=v['prior_exposure']
            dates=D.day_dates(v['last_day']);out.loc[ix,name+'__last_observation_date']=pd.Series(dates,index=ix)
            row=batch.copy();row['head_name']=spec.name
            for k,values in v.items():row[k]=values
            row['last_observation_date']=dates;row['parameter_year']=int(vintage);row['model_sha256']=recipe.get('_sha256','');row['parameter_bootstrap']=recipe['status']=='bootstrap';row['parameter_last_training_observation']=recipe.get('last_training_observation');provenance.append(row)
    for head in HEADS:
        name='lib__'+head;out[name+'__missing']=out[name].isna();out[name+'__days_since_observation']=(q.query_date-pd.to_datetime(out[name+'__last_observation_date'])).dt.total_seconds()/86400
    out['diag__contested_shots_sum_parts100']=predictions[HEADS[1]]+predictions[HEADS[2]]
    out['diag__independent_total_minus_sum_parts100']=predictions[HEADS[0]]-out.diag__contested_shots_sum_parts100
    return out,pd.concat(provenance,ignore_index=True) if provenance else pd.DataFrame(columns=[*D.KEYS,'head_name','prediction','prior_exposure','last_day','source_supported','player_observed','last_observation_date','parameter_year','model_sha256','parameter_bootstrap','parameter_last_training_observation'])


def annual_packet_features(packet,queries):
    """Exact saved annual OU values; no estimator substitution or held postgame."""
    q=D.normalize_queries(queries)
    columns=['player_id','query_date','query_season',*[c for h in HEADS for c in ['lib__'+h,'lib__'+h+'__exposure','lib__'+h+'__last_observation_date']]]
    p=packet[columns].copy();p['query_date']=pd.to_datetime(p.query_date)
    if not p.query_season.between(1997,2026).all() or not p.query_date.between(D.LOWER,D.UPPER).all():raise ValueError('Protected/future saved annual packet')
    if p.duplicated(['player_id','query_date','query_season']).any():raise ValueError('Ambiguous saved annual packet')
    out=q.merge(p,on=['player_id','query_date','query_season'],how='left',validate='one_to_one',indicator=True)
    if out._merge.ne('both').any():raise ValueError('Query not present in saved annual universe; explicit purequery required')
    out=out.drop(columns='_merge')
    for h in HEADS:
        name='lib__'+h;last=pd.to_datetime(out[name+'__last_observation_date']);finite=out[name].notna()
        if (finite&(~np.isfinite(out[name])|~np.isfinite(out[name+'__exposure'])|out[name+'__exposure'].le(0)|last.isna()|last.ge(out.query_date))).any():raise ValueError('Annual packet lacks strict physical history')
        out[name+'__missing']=~finite;out[name+'__days_since_observation']=(out.query_date-last).dt.total_seconds()/86400
    out['diag__contested_shots_sum_parts100']=out['lib__'+HEADS[1]]+out['lib__'+HEADS[2]]
    out['diag__independent_total_minus_sum_parts100']=out['lib__'+HEADS[0]]-out.diag__contested_shots_sum_parts100
    return out


class ContestedShotStore:
    """Pinned, local source store. Writes only caller-owned query work/output."""
    def __init__(self,full_history_root,validated_source_root,measurement_path,archive_root,work_root):
        self.root=Path(full_history_root).resolve();self.work=Path(work_root).resolve();self.inputs={}
        def pin(path,expected=None):
            path=Path(path).resolve();digest=sha(path)
            if expected is not None and expected!=digest:raise ValueError('Frozen contested source hash mismatch: '+str(path))
            if str(path) in self.inputs and self.inputs[str(path)]!=digest:raise ValueError('Conflicting contested source pin')
            self.inputs[str(path)]=digest;return digest
        self.pin=pin;lock=json.loads((self.root/'run_lock.json').read_text());pin(self.root/'run_lock.json');pin(self.root/'observations.parquet',lock['observations_sha256']);pin(self.root/'registry.json',lock['registry_sha256'])
        scripts=Path(__file__).parent
        for name in ['build_contested_shot_states.py','export_vpm_1997_daily_features.py','vpm_1997_daily_query_kernel.cpp','export_vpm_1997_features.py','export_full_history_annual_snapshots.py','full_skill_history_specs.py','build_vpm_defense_bank.py','build_pregame_defensive_skills.py']:pin(scripts/name)
        pin(scripts.parent/'config/full_skill_history_sources.json');pin(scripts/'export_full_skill_history.py');pin(scripts/'build_full_skill_history.py',lock['code_sha256']);pin(scripts/'full_skill_history_kernel.cpp',lock['kernel_sha256'])
        cols=['player_id','game_id','game_date','season','day','off_poss','def_poss',*RAW]
        self.panel=normalize_source(pd.read_parquet(self.root/'observations.parquet',columns=cols));self.source_audit=audit_observations(self.panel)
        validated=Path(validated_source_root).resolve();manifest=json.loads((validated/'manifest.json').read_text());pin(validated/'manifest.json')
        for name,digest in manifest['artifacts'].items():pin(validated/name,digest)
        val=pd.read_parquet(validated/'player_game_lines.parquet',columns=['player_id','game_id','game_date','off_poss','def_poss']);val['game_id']=val.game_id.astype(str).str.zfill(10)
        if val.duplicated(['player_id','game_id']).any() or pd.to_datetime(val.game_date).ge(D.UPPER).any():raise ValueError('Invalid validated source identities/dates')
        z=val.merge(self.panel[['player_id','game_id','game_date','off_poss','def_poss']],on=['player_id','game_id'],how='left',validate='one_to_one',suffixes=('_validated','_original'),indicator=True)
        if z._merge.ne('both').any() or not pd.to_datetime(z.game_date_validated).equals(pd.to_datetime(z.game_date_original)):raise ValueError('Validated source date/identity differs')
        for side in ['off_poss','def_poss']:
            if not np.array_equal(z[side+'_validated'].to_numpy(),z[side+'_original'].to_numpy(),equal_nan=True):raise ValueError('Validated tracking exposure differs')
        measurement_path=Path(measurement_path).resolve();pin(measurement_path);measurements=pd.read_parquet(measurement_path)
        rows=[]
        for year in range(2017,2027):
            for suffix in ['', 'ps']:
                path=Path(archive_root).resolve()/f'{year}{suffix}_games.csv';pin(path)
                f=pd.read_csv(path,usecols=['PLAYER_ID','date',*ARCHIVE],low_memory=False).rename(columns={'PLAYER_ID':'player_id'})
                f['game_date']=pd.to_datetime(f.date.astype(str),format='%Y%m%d');f['season']=year;rows.append(f)
        self.modern,self.modern_audit=modern_split_measurements(measurements,pd.concat(rows,ignore_index=True))
        self.original_library=self.root/'kernel'/('full_skill_'+lock['kernel_sha256'][:16]+'.so');pin(self.original_library)
        self.query_library=D.compile_kernel(self.work/'kernel');pin(self.query_library)
        self.recipes={}
        for year in range(1997,2027):
            self.recipes[year]={}
            for name in HEADS:
                path=self.root/'models'/str(year)/(name+'.json');digest=pin(path);recipe=json.loads(path.read_text())
                if recipe.get('skill')!=name or recipe.get('run_lock_sha256')!=sha(self.root/'run_lock.json'):raise ValueError('Contested OU parameter identity differs')
                validate_recipe(recipe,year,D.A.annual_report_cut(year));recipe['_sha256']=digest;self.recipes[year][name]=recipe
        self.verify_inputs()

    def verify_inputs(self):
        for path,digest in self.inputs.items():
            if sha(path)!=digest:raise ValueError('Contested source changed: '+path)

    def query(self,queries,*,include_ou=True):
        self.verify_inputs();q=D.normalize_queries(queries);modern=modern_query_states(self.modern,q)
        if include_ou:
            library,provenance=ou_query_states(self.panel,q,self.recipes,self.query_library,self.original_library)
            modern=pd.concat([modern,library.drop(columns=D.KEYS)],axis=1)
        else:provenance=pd.DataFrame()
        self.verify_inputs();return modern,provenance

    def annual(self,packet_root,queries):
        packet_root=Path(packet_root).resolve();receipt=json.loads((packet_root/'receipt.json').read_text());self.pin(packet_root/'receipt.json')
        path=packet_root/'feature_packet.parquet';self.pin(path,receipt['output_hashes'][str(path)])
        out=annual_packet_features(pd.read_parquet(path),queries);self.verify_inputs();return out


def schema():
    return dict(profile=PROFILE,modern_estimator=dict(half_life_days=550.,prior_defensive_possessions=1000.,source='original selected measurements + exact matching original year-file split counts',fields=MODERN,units='per100 original selected measurement defensive possessions',raw_and_predicted_split_closure=True,zero_is_observed=True,no_history_missing=True),
        diagnostic_OU=dict(fields=['lib__'+h for h in HEADS],units='per100 validated full_history defensive possessions',parameter_transport_not_refit=True,independent_total_not_sum=True,derived_sum_and_gap_are_diagnostics_not_reconciliation=True),
        candidate_policy='Replace aggregate only if BOTH splits admitted by mapper AND observed per player; otherwise retain aggregate. Never all3 redundant predictors in primary split arm.',
        query_calendar='Explicit native ending season independent of held annual report parameter vintage;2020-Nov1 exception',historical_publication_vintages_certified=False,protected_2027_inputs=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);base=Path(__file__).resolve().parents[1]/'results/estimated_skills'
    parser.add_argument('--history-root',type=Path,default=base/'full_history_v2');parser.add_argument('--validated-source',type=Path,default=base/'pregame_source_validated_v2')
    parser.add_argument('--measurements',type=Path,default=base/'defense_bank_v51_shooting_pf_repaired/measurements.parquet');parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parents[2]/'player_sheets/game_report/year_files')
    parser.add_argument('--queries',type=Path);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--modern-only',action='store_true');args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():raise FileExistsError('Never overwrite contested adapter evidence/output')
    out.mkdir(parents=True)
    store=ContestedShotStore(args.history_root,args.validated_source,args.measurements,args.archive,out/'work')
    write_json(out/'schema.json',schema());write_json(out/'source_audit.json',dict(full_history=store.source_audit,modern_original_selected=store.modern_audit))
    if args.queries:
        store.pin(args.queries);features,provenance=store.query(pd.read_parquet(args.queries),include_ou=not args.modern_only)
        features.to_parquet(out/'features.parquet',index=False,compression='zstd');provenance.to_parquet(out/'provenance.parquet',index=False,compression='zstd')
    store.verify_inputs();outputs={str(p.resolve()):sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'receipt.json',dict(status='passed_validated_contested_count_adapter',profile=PROFILE,rows=len(features) if args.queries else 0,no_refits=True,selected_pointer_changed=False,protected_2027_inputs=0,input_hashes=store.inputs,output_hashes=outputs))
    print(json.dumps(dict(status='passed_validated_contested_count_adapter',receipt=str(out/'receipt.json'),sha256=sha(out/'receipt.json'))))


if __name__=='__main__':main()

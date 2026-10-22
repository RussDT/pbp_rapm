"""Source-adapted1997–2026 predictive-stat snapshots for historical Parent VPM.

Saved independent OU parameters/means are transported, never refitted. Each
query is a separate terminal missing-measurement replay with strict date and
physical-history gates. This is not selected-neutral parity or a career model.
The frozen modern exporter and full_history_v2 remain unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from export_full_history_annual_snapshots import (Engine,SKILLS,ZONES,derived,
    audit_derived,sha,write_json,strict_observations,validate_recipe,feature_schema)

PROFILE='historical_independent_transport_source_v2_1997_v1'
KEYS=['query_id','player_id','query_date','query_season','cohort_vintage']
BOUND=pd.Timestamp('2026-10-01')
MIN_CUT=pd.Timestamp('1997-01-01')

def annual_report_cut(year):
    if int(year)!=year or not 1997<=int(year)<=2026:
        raise ValueError('Completed ending season1997–2026 only; no2027')
    return pd.Timestamp(f'{int(year)}-11-01' if year==2020 else f'{int(year)}-10-01')


def normalize_queries(frame):
    q=frame.copy()
    if not set(KEYS)<=set(q) or q.query_id.isna().any() or q.query_id.duplicated().any():
        raise ValueError('Explicit unique query IDs and complete query keys required')
    q['query_date']=pd.to_datetime(q.query_date,utc=True,errors='raise').dt.tz_localize(None)
    if q.query_date.isna().any() or not q.query_date.eq(q.query_date.dt.normalize()).all() or not q.query_date.between(MIN_CUT,BOUND).all():
        raise ValueError('Date-only historical queries1997 through October1,2026 required')
    for name in ['player_id','query_season']:
        x=pd.to_numeric(q[name],errors='raise')
        if x.isna().any() or not np.isfinite(x).all() or not np.equal(x,np.floor(x)).all():
            raise ValueError('Exact integer player/season required')
        q[name]=x.astype(np.int64)
    if q.player_id.le(0).any() or not q.query_season.between(1997,2026).all():
        raise ValueError('Positive canonical archive identities and no2027 query/recipe required')
    q['cohort_vintage']=pd.to_numeric(q.cohort_vintage,errors='raise').astype('Int64')
    if q.duplicated(['player_id','query_date','query_season']).any() or q.groupby('query_date').query_season.nunique().gt(1).any():
        raise ValueError('Repeated or ambiguous explicit query identity')
    return q[KEYS].reset_index(drop=True)


def validate_panel(panel):
    if not set(['player_id','game_id','game_date','season'])<=set(panel):raise ValueError('Source identities required')
    if not pd.to_numeric(panel.season).between(1997,2026).all() or pd.to_datetime(panel.game_date).ge(BOUND).any():
        raise ValueError('Protected/future source observation')
    ids=pd.to_numeric(panel.player_id)
    if not np.isfinite(ids).all() or ids.le(0).any() or not np.equal(ids,np.floor(ids)).all() or panel.duplicated(['player_id','game_id']).any():
        raise ValueError('Unique positive canonical archive source identities required')


def build_queries(panel,seasons=range(1997,2027)):
    """Cumulative observed archive IDs, without importing later player addresses."""
    validate_panel(panel);rows=[]
    for year in seasons:
        cut=annual_report_cut(year)
        prefix=panel.loc[pd.to_datetime(panel.game_date).lt(cut)&panel.season.le(year)]
        for player in np.sort(prefix.player_id.unique()):
            rows.append(dict(query_id=f'history:{cut.date()}:{int(year)}:{int(player)}',player_id=int(player),
                query_date=cut,query_season=int(year),cohort_vintage=int(year)))
    return normalize_queries(pd.DataFrame(rows,columns=KEYS))

def replay_snapshot(panel, queries, specs, recipes, library):
    """One independent terminal snapshot; no source update from query rows."""
    q = normalize_queries(queries)
    if q.query_date.nunique() != 1 or q.query_season.nunique() != 1:
        raise ValueError("Replay snapshots separately to avoid query side effects")
    cut, year = q.query_date.iloc[0], int(q.query_season.iloc[0])
    prefix = panel.loc[pd.to_datetime(panel.game_date).lt(cut)].copy()
    prefix["_query_row"] = -1
    query = pd.DataFrame(index=np.arange(len(q)), columns=prefix.columns)
    query["player_id"] = q.player_id.to_numpy()
    query["game_date"] = cut
    query["season"] = year
    query["day"] = int((cut - pd.Timestamp("1970-01-01")).days)
    query["_query_row"] = np.arange(len(q))
    full = pd.concat([prefix, query], ignore_index=True).sort_values(
        ["player_id", "game_date", "_query_row"], kind="stable").reset_index(drop=True)
    take = np.flatnonzero(full._query_row.ge(0).to_numpy())
    order = np.argsort(full._query_row.iloc[take].to_numpy())
    rows, values = [], {}
    for spec in specs:
        recipe = recipes[spec.name]
        validate_recipe(recipe, year, cut)
        engine = Engine(full, spec, library)
        result = engine.run(recipe["parameters"], recipe["mean"])
        prediction = result[0][take][order] * (1 if spec.kind == "pct" else 100)
        exposure = result[3][take][order]
        day = result[5][take][order]
        first = engine.day[engine.exposure > 0].min(initial=np.inf)
        supported = bool(spec.first_season <= year <= spec.last_season and first < query.day.iloc[0])
        observed = (exposure > 0) & (day >= 0)
        legal = supported & observed
        last = pd.to_datetime(np.where(day >= 0, day, np.nan), unit="D", origin="1970-01-01")
        if (pd.Series(last).dropna() >= cut).any():
            raise ValueError("Current/future measurement in common snapshot")
        values[spec.name] = np.where(legal, prediction, np.nan)
        r = q.copy()
        r["head_name"] = spec.name
        r["prediction"] = values[spec.name]
        r["source_supported"] = supported
        r["player_observed"] = observed
        r["prior_exposure"] = exposure
        r["last_observation_date"] = last
        r["parameter_year"] = year
        r["parameter_last_training_observation"] = pd.to_datetime(recipe.get("last_training_observation"))
        r["parameter_bootstrap"] = recipe["status"] == "bootstrap"
        r["model_sha256"] = recipe.get("_sha256", "")
        rows.append(r)
    return pd.DataFrame(values), pd.concat(rows, ignore_index=True)


def make_features(values, provenance, queries, schema):
    q = normalize_queries(queries)
    frame = values.copy()
    frame["season"] = q.query_season.to_numpy()
    frame = derived(frame)
    frame["assist_points_created100"] = 2 * sum(frame["assists_" + z + "100"] for z in ZONES[:3]) + 3 * sum(frame["assists_" + z + "100"] for z in ZONES[3:])
    frame["unassisted_fg_points100"] = 2 * frame.two_unassisted_fgm100 + 3 * frame.three_unassisted_fgm100
    frame["defended_fg_points100"] = 2 * frame.two_pt_def_fgm100 + 3 * frame.three_pt_def_fgm100
    audit_derived(frame)
    lookup = {head: f.set_index("query_id").reindex(q.query_id) for head, f in provenance.groupby("head_name", sort=False)}
    columns = {}
    for r in schema["records"]:
        name = r["name"][5:]
        prediction = frame[name].to_numpy(float)
        exposure = np.zeros(len(q))
        last = np.full(len(q), np.datetime64("NaT"), dtype="datetime64[ns]")
        legal = np.zeros(len(q), bool)
        for era, take in [("through_2024", q.query_season.le(2024).to_numpy()), ("from_2025", q.query_season.ge(2025).to_numpy())]:
            heads = r["source_heads_by_era"][era]
            parts = [lookup[h] for h in heads]
            x = np.stack([p.prior_exposure.to_numpy(float) for p in parts], axis=1)
            days = np.stack([p.last_observation_date.to_numpy(dtype="datetime64[ns]") for p in parts], axis=1)
            available = np.stack([(p.source_supported & p.player_observed).to_numpy(bool) for p in parts], axis=1).all(axis=1)
            exposure[take] = x.min(axis=1)[take]
            last[take] = days.max(axis=1)[take]
            legal[take] = available[take]
        columns[r["name"]] = np.where(legal, prediction, np.nan)
        columns[r["exposure_column"]] = exposure
        columns[r["last_observation_column"]] = last
    return pd.concat([q, pd.DataFrame(columns)], axis=1)



COMPACT_SOURCES={
 'hist__rim_fga100':'lib__at_rim_fga100','hist__rim_fgm100':'lib__at_rim_fgm100','hist__rim_pct':'lib__at_rim_pct',
 'hist__mid_fga100':'lib__mid_fga100','hist__mid_fgm100':'lib__mid_fgm100','hist__mid_pct':'lib__mid_pct',
 'hist__three_fga100':'lib__three_fga100','hist__three_fgm100':'lib__three_fgm100','hist__three_pct':'lib__three_pct',
 'hist__fta100':'lib__fta100','hist__ftm100':'lib__ftm100','hist__ft_pct':'lib__ft_pct',
 'hist__ast100':'lib__ast100','hist__tov100':'lib__tov100','hist__oreb100':'lib__oreb100',
 'hist__usage_events100':'lib__usage_events100','hist__stl100':'lib__stl100','hist__blk100':'lib__blk100',
 'hist__dreb100':'lib__dreb100','hist__pf100_total':'lib__all_fouls100_total_possessions',
 'hist__shooting_pf100_total':'lib__foul_shooting100','hist__contested_dreb100':'lib__dreb_contest100'}
UNAVAILABLE=['hist__nonoff_pf100_total']


def compact_schema(generic):
    lookup={r['name']:r for r in generic['records']};records=[]
    for name,source in COMPACT_SOURCES.items():
        native='def' if name in ['hist__stl100','hist__blk100','hist__dreb100','hist__pf100_total','hist__shooting_pf100_total','hist__contested_dreb100'] else 'off'
        unit='fraction' if name.endswith('_pct') else 'per_100_total_possessions' if name.endswith('_total') else 'per_100_defensive_possessions' if native=='def' else 'per_100_offensive_possessions'
        description=('Sum of seven recorded PBP foul groups, including technicals; not certified official personal PF' if name=='hist__pf100_total' else 'Measured shooting-foul subtype; not total or defensive-possession PF' if name=='hist__shooting_pf100_total' else 'FGA + 0.44 * FTA + TOV per100 offensive possessions; approximate usage events, not USG percent' if name=='hist__usage_events100' else 'Saved independent production forecast; no selected-neutral equivalence')
        records.append(dict(name=name,source=source,native_side=native,units=unit,description=description,
            exposure_column=name+'__exposure',last_observation_column=name+'__last_observation_date',
            missing_column=name+'__missing',date_lag_column=name+'__days_since_observation',
            source_heads_by_era=lookup[source]['source_heads_by_era']))
    records.append(dict(name=UNAVAILABLE[0],native_side='def',units='per_100_total_possessions',source=None,
        status='unavailable_pending_exact_nonoff_personal_foul_admission',
        description='No invented split or subtraction of incompatible foul definitions'))
    records.append(dict(name='ctx_mpg',native_side='both',units='minutes_per_positive_minute_RS_appearance',
        exposure_column='ctx_games',last_observation_column='ctx_mpg__last_observation_date',missing_column='ctx_mpg__missing',
        date_lag_column='ctx_mpg__days_since_observation',recipe='literal_past365_days_regular_season_validated_positive_minutes_mean'))
    return dict(schema_version=1,schema='vpm_1997_compact_physical_features_v1',records=records,
        aggregate_rebound_and_foul_preserved=True,replacement_policy='Mapper admission AND individual observed replacement required; compact exporter never drops aggregates',
        prior_exposure_is_source_support_not_career=True,selected_neutral_parity_claim=False,
        source_exposure_metadata_not_model_covariates=True,
        provenance_columns=['estimate_date','archive_off_possessions_lower_bound','archive_def_possessions_lower_bound',
            'archive_exposure_unknown_rows','archive_observed_games','archive_positive_playing_games',
            'first_archive_observation_date','last_archive_observation_date','last_archive_positive_observation_date'])


def compact_features(wide,generic_schema,source_support):
    q=normalize_queries(wide[KEYS]);columns={}
    declared={r['name'] for r in generic_schema['records']}
    if not set(COMPACT_SOURCES.values())<=declared:raise ValueError('Compact sources absent from pinned generic schema')
    for name,source in COMPACT_SOURCES.items():
        value=wide[source].to_numpy(float);exposure=wide[source+'__exposure'].to_numpy(float)
        last=pd.to_datetime(wide[source+'__last_observation_date'])
        observed=np.isfinite(value)&np.isfinite(exposure)&(exposure>0)&last.notna().to_numpy()
        if (last[observed]>=q.query_date[observed]).any():raise ValueError('Compact physical observation reaches its query')
        columns[name]=np.where(observed,value,np.nan);columns[name+'__exposure']=exposure
        columns[name+'__last_observation_date']=last;columns[name+'__missing']=~observed
        columns[name+'__days_since_observation']=(q.query_date-last).dt.total_seconds()/86400
    for name in UNAVAILABLE:
        columns[name]=np.nan;columns[name+'__exposure']=0.;columns[name+'__last_observation_date']=pd.NaT
        columns[name+'__missing']=True;columns[name+'__days_since_observation']=np.nan
    out=pd.concat([q,pd.DataFrame(columns,index=q.index)],axis=1);out['estimate_date']=q.query_date
    metadata=[]
    for cut,group in q.groupby('query_date',sort=False):
        prefix=source_support.loc[pd.to_datetime(source_support.game_date).lt(cut)]
        histories={int(player):g for player,g in prefix.groupby('player_id',sort=False)}
        for row in group.itertuples():
            p=histories.get(int(row.player_id),prefix.iloc[:0])
            off=p.off_poss.to_numpy(float);de=p.def_poss.to_numpy(float)
            goodoff=np.isfinite(off)&(off>=0);goodde=np.isfinite(de)&(de>=0)
            minutes=p.loc[p.playoffs.eq(False) & p.game_date.ge(cut-pd.Timedelta(days=365)) & p.minute_observed & p.minutes.gt(0)]
            last=minutes.game_date.max() if len(minutes) else pd.NaT
            positive=(p.off_poss.gt(0)|p.def_poss.gt(0)|p.minutes.gt(0))
            metadata.append(dict(query_id=row.query_id,ctx_mpg=float(minutes.minutes.mean()) if len(minutes) else np.nan,
                ctx_games=len(minutes),ctx_mpg__last_observation_date=last,ctx_mpg__missing=not len(minutes),
                ctx_mpg__days_since_observation=(cut-last).total_seconds()/86400 if len(minutes) else np.nan,
                archive_off_possessions_lower_bound=float(off[goodoff].sum()),archive_def_possessions_lower_bound=float(de[goodde].sum()),
                archive_exposure_unknown_rows=int((~(goodoff&goodde)).sum()),archive_observed_games=len(p),
                archive_positive_playing_games=int(positive.sum()),first_archive_observation_date=p.game_date.min(),
                last_archive_observation_date=p.game_date.max(),last_archive_positive_observation_date=p.loc[positive,'game_date'].max()))
    return out.merge(pd.DataFrame(metadata),on='query_id',how='left',validate='one_to_one',sort=False)


def observation_support(panel,validated):
    """Literal minutes, with conservative early two-team clock validation.

    Modern minutes use the independently validated v2 masks exactly. Earlier
    minutes require all game rows clean/issue-free and both team sums equal an
    NBA48+5k clock. This is an observation admission, not a minutes forecast.
    """
    columns=['player_id','game_id','game_date','season','playoffs','team_id','seconds_pbp','lineups_clean','known_issue','off_poss','def_poss']
    p=panel[columns].copy().reset_index(drop=True);p.game_id=p.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    p.game_date=pd.to_datetime(p.game_date);seconds=p.seconds_pbp.to_numpy(float)
    clean=p.lineups_clean.eq(True)&p.known_issue.fillna('').astype(str).str.strip().eq('')&np.isfinite(seconds)&(seconds>=0)
    admitted=pd.Series(False,index=p.index);invalid_early_games=0
    for game,g in p.loc[p.season.lt(2014)].groupby('game_id',sort=False):
        by=g.groupby('team_id').seconds_pbp.sum();durations=by.to_numpy(float)/300
        good=(len(by)==2 and np.all(np.asarray(by.index,float)>0) and clean.loc[g.index].all()
            and np.isfinite(durations).all() and np.max(durations)-np.min(durations)<1e-8
            and np.min(durations)>=48 and np.all(abs((durations-48)/5-np.rint((durations-48)/5))<1e-8))
        if good:good=bool(g.seconds_pbp.le(durations[0]*60+1e-8).all())
        admitted.loc[g.index]=good;invalid_early_games+=not good
    p['minutes']=np.where(admitted,seconds/60,np.nan)
    p['minute_observed']=admitted.to_numpy();p['minute_source']=np.where(admitted,'early_clean_two_team_clock','unadmitted_early_clock_or_source_issue')
    v=validated[['player_id','game_id','game_date','minutes']].copy();v.game_id=v.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    v.game_date=pd.to_datetime(v.game_date)
    if v.duplicated(['player_id','game_id']).any():raise ValueError('Duplicate validated minute source')
    index=pd.MultiIndex.from_frame(p[['player_id','game_id']]);positions=index.get_indexer(pd.MultiIndex.from_frame(v[['player_id','game_id']]))
    if (positions<0).any() or not np.array_equal(p.game_date.iloc[positions].to_numpy(),v.game_date.to_numpy()):raise ValueError('Validated minute identity/date mismatch')
    finite=v.minutes.notna().to_numpy();actual=v.minutes.to_numpy(float)
    if np.any(finite&((actual<0)|~np.isfinite(actual))):raise ValueError('Invalid validated minutes')
    if not np.array_equal(p.seconds_pbp.iloc[positions].to_numpy()[finite]/60,actual[finite]):raise ValueError('Modern seconds/minutes definitions differ')
    # No early-clock inference is allowed to fill an absent modern validated row.
    modern=p.season.ge(2014);p.loc[modern,['minutes']]=np.nan;p.loc[modern,'minute_observed']=False;p.loc[modern,'minute_source']='unadmitted_modern_minutes'
    p.loc[positions,'minutes']=actual;p.loc[positions,'minute_observed']=finite;p.loc[positions,'minute_source']='validated_source_v2_minutes'
    p['minute_source_issue']=p.known_issue.notna()|~p.lineups_clean.eq(True)
    receipt=dict(early_minute_rule='all_rows_clean_no_known_issue_both_teams_same48plus5k_clock',
        early_excluded_games=int(invalid_early_games),early_admitted_rows=int(p.loc[p.season.lt(2014),'minute_observed'].sum()),
        modern_validated_rows=int(finite.sum()),modern_seconds_minutes_gap=0.,missing_minutes_never_zero_filled=True,
        mpg='strictly_past365day_RS_positive_minute_appearance_mean_no_shrinkage')
    return p.drop(columns=['seconds_pbp','lineups_clean','known_issue']),receipt


def export_queries(full_history_root,validated_source_root,queries,out,*,admission_paths=()):
    root,source,out=map(lambda p:Path(p).resolve(),[full_history_root,validated_source_root,out])
    if out.exists():raise FileExistsError('Use a new immutable historical snapshot root')
    q=normalize_queries(queries)
    if not all(pd.Timestamp(row.query_date)==annual_report_cut(int(row.query_season)) for row in q.itertuples()):
        raise ValueError('Export only registered annual report cuts; replay API handles independent diagnostic dates')
    lock=json.loads((root/'run_lock.json').read_text())
    for name,key in [('observations.parquet','observations_sha256'),('registry.json','registry_sha256')]:
        if sha(root/name)!=lock[key]:raise ValueError('Frozen library source/registry changed')
    library=root/'kernel'/('full_skill_'+lock['kernel_sha256'][:16]+'.so')
    folder=Path(__file__).parent
    files=[root/'run_lock.json',root/'observations.parquet',root/'registry.json',library,
        source/'manifest.json',source/'player_game_lines.parquet',Path(__file__),
        folder/'export_full_history_annual_snapshots.py',folder/'build_full_skill_history.py',
        folder/'full_skill_history_specs.py',folder/'export_full_skill_history.py',folder/'full_skill_history_kernel.cpp',
        folder.parent/'config/full_skill_history_sources.json',*[Path(p).resolve() for p in admission_paths]]
    if sha(folder/'full_skill_history_kernel.cpp')!=lock['kernel_sha256']:raise ValueError('Frozen replay kernel source changed')
    inputs=dict(queries.attrs.get('input_hashes',{}))
    for p in files:
        name,digest=str(p.resolve()),sha(p)
        if name in inputs and inputs[name]!=digest:raise ValueError('Conflicting historical source pins')
        inputs[name]=digest
    for name,digest in inputs.items():
        if not Path(name).is_absolute() or sha(name)!=digest:raise ValueError('Query/source input changed')
    original=pd.read_parquet(root/'observations.parquet');validated=pd.read_parquet(source/'player_game_lines.parquet')
    validate_panel(original);panel,masks=strict_observations(original,validated)
    expected=build_queries(original,sorted(q.query_season.unique()))
    pd.testing.assert_frame_equal(q,expected,check_exact=True)
    support,minute_audit=observation_support(panel,validated)
    registry=[];recipes={}
    for year in sorted(q.query_season.unique()):
        recipes[int(year)]={}
        for spec in SKILLS:
            path=root/'models'/str(int(year))/(spec.name+'.json');r=json.loads(path.read_text())
            if r.get('skill')!=spec.name or r['run_lock_sha256']!=sha(root/'run_lock.json'):raise ValueError('Frozen annual recipe identity changed')
            r['_sha256']=sha(path);inputs[str(path.resolve())]=r['_sha256'];recipes[int(year)][spec.name]=r
            registry.append(dict(head_name=spec.name,parameter_year=int(year),model_path=str(path.resolve()),model_sha256=r['_sha256'],parameter_last_training_observation=r.get('last_training_observation'),parameter_bootstrap=r['status']=='bootstrap'))
    schema=feature_schema(registry)
    schema['metadata'].update(historical_report_dates='Oct1 except Nov1,2020',query_season='explicit completed ending season; same saved recipe ending year',
        original_modern_exporter_unchanged=True,extended_terminal_replay_contract=PROFILE,source_observation_bound=str(BOUND.date()))
    packets=[];heads=[]
    for _,batch in q.groupby('query_date',sort=True):
        batch=batch.reset_index(drop=True);values,long=replay_snapshot(panel,batch,SKILLS,recipes[int(batch.query_season.iloc[0])],library)
        packets.append(make_features(values,long,batch,schema));heads.append(long)
        print(json.dumps(dict(stage='historical_terminal_snapshot',date=str(batch.query_date.iloc[0].date()),queries=len(batch))),flush=True)
    wide=pd.concat(packets,ignore_index=True).set_index('query_id').reindex(q.query_id).reset_index()
    long=pd.concat(heads,ignore_index=True);compact=compact_features(wide,schema,support)
    if not wide[KEYS].equals(q) or not compact[KEYS].equals(q) or len(long)!=len(q)*len(SKILLS):raise ValueError('Historical ordered query census changed')
    for name,digest in inputs.items():
        if sha(name)!=digest:raise ValueError('Input changed during historical replay')
    out.mkdir(parents=True)
    for name,frame in [('feature_packet',wide),('head_provenance',long),('compact_features',compact),('queries',q),('source_observations',support)]:frame.to_parquet(out/(name+'.parquet'),index=False)
    write_json(out/'feature_schema.json',schema);write_json(out/'compact_schema.json',compact_schema(schema))
    coverage=[]
    for cut,frame in compact.groupby('query_date',sort=True):
        for name in [*COMPACT_SOURCES,*UNAVAILABLE,'ctx_mpg']:
            coverage.append(dict(query_date=str(cut.date()),feature=name,finite_rows=int(frame[name].notna().sum()),query_rows=len(frame)))
    pd.DataFrame(coverage).to_csv(out/'compact_coverage.csv',index=False)
    receipt=dict(status='passed_historical_independent_terminal_and_compact_features',profile=PROFILE,
        query_rows=len(q),query_dates=int(q.query_date.nunique()),source_heads=len(SKILLS),generic_features=len(schema['records']),
        compact_feature_columns=[*COMPACT_SOURCES,*UNAVAILABLE,'ctx_mpg'],feature_schema_sha256=sha(out/'feature_schema.json'),compact_schema_sha256=sha(out/'compact_schema.json'),
        completed_ending_seasons=sorted(map(int,q.query_season.unique())),report_cut_policy='Oct1 completed ending season except2020Nov1',
        source_universe='Cumulative positive canonical archive player IDs observed strictly before each report, never future addresses; query membership is not NBA target eligibility',
        parameter_transport_not_refit=True,cross_stat_coefficients_used=False,parameter_dates_checked_individually=True,
        source_v2_statistical_masks_retained=True,strict_mask_changes=masks,pre2014_original_library_admission_retained=True,
        minute_admission=minute_audit,career_weights_or_background_ability_invented=False,
        source_exposure_is_observed_lower_bound_not_complete_career=True,source_exposure_metadata_not_model_features=True,
        nonoff_pf_unavailable=True,aggregate_dreb_and_pf_always_exported=True,
        model_aware_fallback_required='DREB replaced only admitted+observed contested DREB; PF replaced only BOTH admitted+observed shooting and nonoff bundle',
        selected_neutral_parity_claim=False,full_archive_bootstrap_and_finite_optimizer_budget_retained=True,
        source_publication_vintages_certified=False,source_library_exporters_or_models_modified=False,protected_2027_inputs=0,
        input_hashes=inputs,output_hashes={str(p.resolve()):sha(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'receipt.json',receipt);return receipt


def parse_seasons(value):
    years=list(range(int(value.split(':')[0]),int(value.split(':')[1])+1)) if ':' in value else list(map(int,value.split(',')))
    if not years or len(set(years))!=len(years):raise ValueError('Unique explicit completed seasons required')
    for year in years:annual_report_cut(year)
    return sorted(years)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-history-root',type=Path,required=True);p.add_argument('--validated-source-root',type=Path,required=True)
    p.add_argument('--seasons',required=True,help='Explicit comma list or1997:2026; no implicit full build')
    p.add_argument('--identity-admission',type=Path);p.add_argument('--alias-admission',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();panel=pd.read_parquet(args.full_history_root/'observations.parquet',columns=['player_id','game_id','game_date','season'])
    q=build_queries(panel,parse_seasons(args.seasons))
    result=export_queries(args.full_history_root,args.validated_source_root,q,args.out,admission_paths=[x for x in [args.identity_admission,args.alias_admission] if x is not None])
    print(json.dumps({k:result[k] for k in ['status','query_rows','query_dates','completed_ending_seasons']}),flush=True)


if __name__=='__main__':main()

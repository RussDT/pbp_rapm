"""Emit the owner-requested 2026-10-30 board from a saved history checkpoint."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
E=R/'results/estimated_skills'
V=R.parents[1]/'vpm'
H=V/'vpm/results/harness/vpm_full_feature_history_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def write_code(path, source):
    if path.exists():
        if path.read_text()!=source:raise ValueError('Existing generated source differs: '+str(path))
        return
    with path.open('x')as handle:handle.write(source)


def main(root):
    start=time.monotonic();root=Path(root).resolve();cut=pd.Timestamp('2026-10-30').as_unit('s')
    if shutil.disk_usage(root).free<15*2**30:raise OSError('Preserve runtime reserve')
    refresh=json.loads((root/'refresh.json').read_bytes())
    if refresh['cut']!=str(cut.date()) or refresh['manifest_games']!=71:
        raise ValueError('Exact completed current input snapshot required')
    scores=pd.read_csv(root/'native_source/score_comparison.csv')
    if len(scores)!=71 or scores.difference.ne(0).any():raise ValueError('All current game points must match')
    pointer=V/'vpm/configs/current_model.json';pointer_sha=sha(pointer)
    if pointer_sha!='08e5882c463d2ac10eccc8161afd92f14650afd8585634b4008227fde75d05d5':
        raise ValueError('Selected pointer changed')
    code=root/'current_code_native_scale'
    sys.path.insert(0,str(code));sys.path.insert(1,str(R/'scripts'))
    from current_selected_state import normalize_schedule
    import build_contested_shot_states as contested
    from selected_state_history import _keys
    from vpm.harness.full_bank_history_storage import read_compact_board as read_historical_board
    storage=(V/'vpm/harness/full_bank_history_storage.py').read_text()
    substitutions=[
        ("SCHEMA = 'lossless_own_only_raw_Net_metadata_and_explicit_derived_views_v1'","SCHEMA = 'lossless_own_only_raw_Net_current_20261030_v1'"),
        ("metadata.get('protected_2027_inputs') != 0","metadata.get('current_season_games') != 71 or metadata.get('status') != 'produced_current_full_feature_vpm_pending_independent_closeout'"),
        ("not pd.Timestamp('1996-11-01') <= cut <= pd.Timestamp('2026-10-01')","cut != pd.Timestamp('2026-10-30')"),
        ("years.between(1997, 2026).all()","years.eq(2027).all()")]
    for old,new in substitutions:
        if storage.count(old)!=1:raise ValueError('Known current storage seam differs')
        storage=storage.replace(old,new)
    storage=storage.replace('Only declared own-only completed-history boards supported','Exact current own-only board metadata required').replace('Outside completed-history calendar;2027 protected','Only the requested October30 current board is supported').replace('Saved row season outside completed history;2027 protected','Current board requires explicit query season2027')
    storage_path=code/'current_board_storage.py';write_code(storage_path,storage)
    storage_module=load('vpm.harness.current_board_storage_20261030',storage_path)
    write_compact_board=storage_module.write_compact_board;read_compact_board=storage_module.read_compact_board
    array_signature=storage_module.array_signature

    reference_path=root/'current_reference_core_native_scale.parquet'
    reference=pd.read_parquet(reference_path)
    if not reference.query_season.eq(2027).all() or not pd.to_datetime(reference.estimate_date).eq(cut).all():
        raise ValueError('Explicit current source season/date required')
    prior_path=E/'full_feature_history_fast_prepared_features_v1/full6107_v1/2026-10-01/reference.parquet'
    prior=pd.read_parquet(prior_path).set_index('player_id')
    cloud=pd.read_parquet(root/'player_core.parquet')
    # Archive metadata counts actual past appearances; it never grants eligibility.
    rows=reference.set_index('player_id');now=cloud.rename(columns={'person_id':'player_id'}).groupby('player_id')
    for side in ['off','def']:
        name='archive_'+side+'_possessions_lower_bound'
        rows[name]=prior[name].reindex(rows.index).fillna(0)+now[side+'_poss'].sum().reindex(rows.index).fillna(0)
    for name in ['archive_observed_games','archive_positive_playing_games']:
        counts=now.size() if name=='archive_observed_games' else cloud.loc[cloud.seconds_pbp.gt(0)].groupby('person_id').size()
        rows[name]=prior[name].reindex(rows.index).fillna(0)+counts.reindex(rows.index).fillna(0)
    rows['archive_exposure_unknown_rows']=prior.archive_exposure_unknown_rows.reindex(rows.index).fillna(0)
    for name,which in [('first_archive_observation_date','min'),('last_archive_observation_date','max'),('last_archive_positive_observation_date','max')]:
        take=cloud if name!='last_archive_positive_observation_date' else cloud.loc[cloud.seconds_pbp.gt(0)]
        dates=take.groupby('person_id').game_date.agg(which).reindex(rows.index)
        together=pd.concat([pd.to_datetime(prior[name].reindex(rows.index)),pd.to_datetime(dates)],axis=1)
        rows[name]=together.min(axis=1) if which=='min' else together.max(axis=1)
    reference=rows.reset_index()
    for side in ['off','def']:reference['reference_'+side+'_exposure']=reference['source_prefix_'+side+'_exposure'].fillna(0.)

    # The original TD550/K1000 split estimator, with the current query scope.
    measurements=pd.read_parquet(E/'defense_bank_v51_shooting_pf_repaired/measurements.parquet')
    foreign=pd.read_csv(E/'pregame_source_validated_v1/foreign_legacy_player_games.csv',dtype={'game_id':str})
    measurements=measurements.loc[~_keys(measurements).isin(_keys(foreign))].copy()
    archive=[]
    archive_root=R.parent/'player_sheets/game_report/year_files'
    for year in range(2017,2027):
        for suffix in ['', 'ps']:
            p=archive_root/f'{year}{suffix}_games.csv'
            frame=pd.read_csv(p,usecols=['PLAYER_ID','date',*contested.ARCHIVE],low_memory=False).rename(columns={'PLAYER_ID':'player_id'})
            frame['game_date']=pd.to_datetime(frame.date.astype(str),format='%Y%m%d');frame['season']=year;archive.append(frame)
    modern,_=contested.modern_split_measurements(measurements,pd.concat(archive,ignore_index=True))
    function=inspect.getsource(contested.modern_query_states)
    old='q=D.normalize_queries(queries);d=measurements.copy()'
    if function.count(old)!=1:raise ValueError('Known split-query seam differs')
    function=function.replace(old,'q=normalize_schedule(queries);d=measurements.copy()')
    split_path=code/'current_split_query.py'
    write_code(split_path,'from build_contested_shot_states import *\nfrom current_selected_state import normalize_schedule\n'+function)
    split_module=load('current_split_query',split_path)
    splits=split_module.modern_query_states(modern,reference[['query_id','player_id','query_date','query_season']])
    np.testing.assert_allclose(reference.def_contested_shots100,splits.def_contested_shots100,atol=1e-10,rtol=0,equal_nan=True)
    for column in splits:
        if column.startswith('def_contested_shots_'):reference[column]=splits[column].to_numpy()
    complete=root/'current_reference_complete.parquet'
    if complete.exists():pd.testing.assert_frame_equal(reference,pd.read_parquet(complete),check_exact=True)
    else:reference.to_parquet(complete,index=False)

    # Preserve every numerical operation, changing only the explicit current fence.
    daily=(V/'vpm/harness/daily_history_1997_numerics.py').read_text()
    daily=daily.replace("pd.Timestamp('2026-10-01')","pd.Timestamp('2026-10-30')")
    daily=daily.replace('between(1997,2026)','between(1997,2027)')
    daily_path=code/'current_daily_numerics.py'
    write_code(daily_path,daily)
    daily_name='vpm.harness.current_daily_numerics_20261030';daily_module=load(daily_name,daily_path)
    bank=(V/'vpm/harness/full_bank_history_numerics.py').read_text()
    bank=bank.replace('from .daily_history_1997_numerics import','from .current_daily_numerics_20261030 import')
    bank=bank.replace('between(1997,2026)','between(1997,2027)').replace('1997<=int(query_season)<=2026','1997<=int(query_season)<=2027')
    head,tail=bank.rsplit('protected_2027_inputs=0',1)
    bank=head+'current_season_games=int(self.games.source_season_end_year.eq(2027).sum())'+tail
    bank_path=code/'current_bank_numerics.py'
    write_code(bank_path,bank)
    geometry=load('vpm.harness.current_bank_numerics_20261030',bank_path)
    engine=geometry.FullBankHistoryGeometry.resume(H/'compact_history_fast_v1/checkpoints/2026-10-01')
    if engine.rows!=7108645 or len(engine.games)!=37845:raise ValueError('Original complete history checkpoint required')
    native=root/'native_source/targets';atoms=pd.read_parquet(native/'atoms_2027_RS.parquet');games=pd.read_parquet(native/'games_2027_RS.parquet')
    atoms['sq_active']=False;atoms['mid_gross_residual']=np.nan;atoms['three_gross_residual']=np.nan
    atoms=atoms.sort_values('game_date',kind='stable').reset_index(drop=True);games=games.sort_values('game_date',kind='stable').reset_index(drop=True)
    pins={str(p.resolve()):sha(p) for p in [root/'refresh.json',complete,native/'atoms_2027_RS.parquet',native/'games_2027_RS.parquet',daily_path,bank_path,storage_path,Path(__file__)]}
    engine.append(atoms,games,as_of=cut,input_hashes=pins)
    registry=E/'full_feature_history_early_model_audit_v1/registries/full_split_raw.json'
    release=json.loads(registry.read_bytes())['releases']['2026-10-01'];model_path=Path(release['model_path'])
    if sha(model_path)!=release['model_sha256'] or sha(release['audit_path'])!=release['audit_sha256']:raise ValueError('Frozen audited current mapper changed')
    model=json.loads(model_path.read_bytes());model_pins={str(registry):sha(registry),str(model_path):release['model_sha256'],release['audit_path']:release['audit_sha256']}
    for name,digest in model['artifacts'].items():
        p=(model_path.parent/name).resolve()
        if not p.is_relative_to(model_path.parent) or sha(p)!=digest:raise ValueError('Frozen mapper artifact changed')
        model_pins[str(p)]=digest
    metadata=dict(model_path=str(model_path),model_sha256=release['model_sha256'],mapper_asof='2026-10-01',historical_backcast=False,
        input_hashes=model_pins,source_input_hashes=pins,reference_sha256=geometry.reference_fingerprint(reference))
    old_board=read_historical_board(H/'compact_history_fast_v1/cuts/2026-10-01')
    names=dict(zip(old_board.ratings.player_id.astype(int),old_board.ratings.player_name))
    for p in (root/'objects/player').glob('*.json'):
        if p.stem.isdigit():names[int(p.stem)]=json.loads(p.read_bytes()).get('name')
    board=engine.emit(reference,model,metadata,as_of=cut,query_season=2027,names=names,role_source=None)
    board.metadata.update(status='produced_current_full_feature_vpm_pending_independent_closeout',
        current_season_games=71,current_season_last_game='2026-10-29',current_tracking_reports_available=False,
        current_season_shot_quality_available=False,tracking_policy='Retain earlier observed states under unchanged estimator and stale masks; no invented current observations',
        new_mapper_fits=0,selected_pointer_changed=False)
    recipe=dict(schema='current_complete_source_reference_basis_v1',input_hashes=pins,
                description='Original full-source preprocessing and frozen mapper basis before native player reindexing, current October30 query',
                reference_path=str(complete),reference_sha256=geometry.reference_fingerprint(reference),model_sha256=release['model_sha256'])
    def basis(saved,ids):
        if saved!=recipe:raise ValueError('Current basis recipe changed')
        _,value,_=geometry.source_reference_basis(reference,model,ids,cut,role_source=None)
        return dict(off_basis=value['off'],def_basis=value['def'])
    destination=root.parent/'current_refresh_20261030_board_v1'
    write_compact_board(board.ratings,board.arrays,board.intercepts,board.metadata,destination,
        input_hashes={**pins,**model_pins},basis_recompute_recipe=recipe,minimum_free_gib=15)
    decoded=read_compact_board(destination,basis_provider=basis)
    pd.testing.assert_frame_equal(board.ratings.reset_index(drop=True),decoded.ratings.reset_index(drop=True),check_exact=True)
    for name in board.arrays:
        if array_signature(board.arrays[name])!=array_signature(decoded.arrays[name]):raise ValueError('Current persisted array differs')
    if sha(pointer)!=pointer_sha:raise ValueError('Selected pointer changed')
    recent=pd.to_datetime(decoded.ratings.last_archive_positive_observation_date).ge(cut-pd.Timedelta(days=365))
    top=decoded.ratings.loc[recent].sort_values('vpm',ascending=False)
    top.to_csv(root/'current_rankings.csv',index=False)
    write(root/'result.json',dict(status='produced_current_refresh_pending_independent_closeout',estimate_date=str(cut.date()),
        games_appended=71,rows_appended=len(atoms),native_games=len(engine.games),players=len(board.ratings),
        recent_players=int(recent.sum()),columns=len(board.ratings.columns),arrays=len(board.arrays),
        model_sha256=release['model_sha256'],new_mapper_fits=0,new_objective_tuning=0,
        current_tracking_reports_available=False,current_season_sq_available=False,
        normal_equation_error=board.metadata['normal_equation_error'],native_net_gap=board.metadata['maximum_native_lane_sum_gap'],
        board_path=str(destination),board_manifest_sha256=sha(destination/'manifest.json'),selected_pointer_changed=False,seconds=time.monotonic()-start))
    print(top[['player_name','ovpm','dvpm','vpm']].head(40).to_json(orient='records'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);main(p.parse_args().root)

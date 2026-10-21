"""Build immutable identity-clean v2 EB/OU historical feature packages.

Admission is exact season-specific official RS+PS positive-GP identity, with
no minutes/opportunities threshold. Original OU own states are preserved;
their population and copied fixed defensive/MPG fields are identity cleaned.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from historical_features import ROOT, OUT as SOURCE, cuts, sha, write_json, feature_snapshot, annual_defense_observations
from build_predictive_rim_saved import lagged_league

OUT=ROOT/'results/estimated_skills/vpm_historical_features_v2'
OU_SOURCE=ROOT/'results/estimated_skills/vpm_historical_features_ou_v1'
OU_OUT=ROOT/'results/estimated_skills/vpm_historical_features_ou_v2'
ADMISSION=ROOT/'results/vpm_historical_targets_v1/identity_admission.json'


def identity_filter(frame,admission,*,season='season'):
    sets={int(y):set(m['admitted_ids']) for y,m in admission['seasons'].items()}
    allowed=np.zeros(len(frame),bool)
    for year,rows in frame.groupby(season).indices.items():
        if int(year) not in sets:raise ValueError('Identity admission missing season')
        allowed[rows]=frame.iloc[rows].player_id.isin(sets[int(year)]).to_numpy()
    return allowed


def dated_name(frame,admission):
    result=frame.copy()
    for year,rows in result.groupby('season').indices.items():
        names=admission['seasons'][str(int(year))]['names']
        values=result.iloc[rows].player_id.astype(int).astype(str).map(names)
        if values.isna().any() or values.str.match(r'^ID_\d+$').any():
            raise ValueError('Admitted official identity lacks a real name')
        result.iloc[rows,result.columns.get_loc('player_name')]=values.to_numpy()
    return result


def filter_tracking(frame,game_seasons,admission,exclude_games=()):
    d=frame.copy();d['game_id']=d.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    d['game_date']=pd.to_datetime(d.game_date)
    d=d.loc[~d.game_id.isin(exclude_games)].copy()
    if 'season' not in d:
        d=d.merge(game_seasons,on='game_id',how='left',validate='many_to_one')
    if d.season.isna().any():raise ValueError('Tracking observation lacks canonical source season')
    return d.loc[identity_filter(d,admission)].copy()


def validate_hashes(root,meta):
    for name,digest in meta.get('artifact_hashes',meta.get('artifacts',{})).items():
        if sha(root/name)!=digest:raise ValueError('Frozen input artifact changed: '+name)


def build(out=OUT,ou_out=OU_OUT,source=SOURCE,ou_source=OU_SOURCE,admission_path=ADMISSION):
    out,ou_out,source,ou_source=map(Path,[out,ou_out,source,ou_source])
    admission= json.loads(Path(admission_path).read_text())
    if admission['status']!='passed':raise ValueError('Official season identity census is not final')
    for path,digest in admission['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('Official identity source changed')
    old=json.loads((source/'manifest.json').read_text());validate_hashes(source,old)
    out.mkdir(parents=True,exist_ok=True);ou_out.mkdir(parents=True,exist_ok=True)
    original=pd.read_parquet(source/'player_game_lines.parquet')
    keep=identity_filter(original,admission)
    rejected=original.loc[~keep]
    rejected.groupby(['season','player_id']).agg(player_game_rows=('game_id','size'),
        off_possessions=('off_poss','sum'),def_possessions=('def_poss','sum'),
        minutes=('minutes','sum'),first_observation=('game_date','min'),last_observation=('game_date','max')).reset_index().to_parquet(out/'identity_rejected_measurements.parquet',index=False)
    lines=dated_name(original.loc[keep].copy(),admission)
    lines.to_parquet(out/'player_game_lines.parquet',index=False)
    game_seasons=original[['game_id','season']].drop_duplicates()
    if game_seasons.game_id.duplicated().any():raise ValueError('A game spans multiple source seasons')
    base=ROOT/'results/estimated_skills'
    paths=[base/'defense_bank_v51_shooting_pf_repaired/measurements.parquet',
        base/'monthly_defense_v1/rim_counts.parquet',base/'monthly_defense_v1/dfga_counts.parquet']
    quarantine=set(old['quarantined_games'])
    measured,rim,dfga=[filter_tracking(pd.read_parquet(path),game_seasons,admission,quarantine) for path in paths]
    for d in [measured,rim,dfga]:d.drop(d.loc[d.game_id.isin(quarantine)].index,inplace=True)
    rim['league_pct']=lagged_league(rim)
    annual,annual_paths,annual_audit=annual_defense_observations()
    frames=[];provenance=[];coverage=[]
    for cut in cuts():
        d,p=feature_snapshot(lines,cut,measured,rim,dfga,annual)
        frames.append(d);provenance.append(p)
        coverage.append(dict(estimate_date=str(cut.date()),players=len(d),available={c:int(d[c].notna().sum())
            for c in d if (c.startswith('own_') or c.startswith('def_')) and not c.endswith('_missing')}))
        print('Identity-clean historical features',str(cut.date()),len(d),flush=True)
    wide=pd.concat(frames,ignore_index=True);long=pd.concat(provenance,ignore_index=True)
    wide.to_parquet(out/'features.parquet',index=False);long.to_parquet(out/'provenance.parquet',index=False)
    meta=dict(old,status='passed_identity_clean_historical_fixed_eb_input_candidate',
        rows=len(wide),provenance_rows=len(long),coverage=coverage,
        identity_policy='Exact source-season official RS union PS with GP>0; no MIN or POSS floor',
        identity_admission=str(admission_path),identity_admission_sha256=sha(admission_path),
        removed_identity_rows=len(rejected),removed_identity_ids=int(rejected.player_id.nunique()),
        annual_defense_audit=annual_audit)
    meta['source_hashes']={str(p):sha(p) for p in [source/'player_game_lines.parquet',source/'manifest.json',
        Path(admission_path),Path(__file__),Path(__file__).with_name('historical_features.py'),*paths,*annual_paths]}
    meta['artifact_hashes']={n:sha(out/n) for n in ['features.parquet','provenance.parquet','player_game_lines.parquet','identity_rejected_measurements.parquet']}
    write_json(out/'manifest.json',meta)
    # Preserve original annual-OU own states numerically. Remove only player/date
    # rows not known through admitted real appearances before that cut.
    old_ou=json.loads((ou_source/'manifest.json').read_text());validate_hashes(ou_source,old_ou)
    own_provenance=pd.read_parquet(ou_source/'provenance.parquet')
    own_provenance=own_provenance.loc[own_provenance.feature.str.startswith('own_')].copy()
    keys=wide[['player_id','estimate_date']].drop_duplicates()
    own_provenance=own_provenance.merge(keys,on=['player_id','estimate_date'],how='inner',validate='many_to_one')
    own=own_provenance.pivot(index=['player_id','estimate_date'],columns='feature',values='value').reset_index()
    for zone,attempt in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),('three','three_fga_rate'),('ft','fta_rate')]:
        own[f'own_{zone}_makes100']=own[f'own_{attempt}']*own[f'own_{zone}_pct']
    dates=own.estimate_date.unique();copy=[c for c in wide if not c.startswith('own_') and c!='feature_recipe']
    ou=wide.loc[wide.estimate_date.isin(dates),copy].merge(own,on=['player_id','estimate_date'],how='left',validate='one_to_one')
    for c in list(ou):
        if c.startswith('own_'):ou[c+'_missing']=ou[c].isna().astype(int)
    ou['feature_recipe']='original_annual_ou_legal_vintage_with_identity_clean_tracking_mpg'
    extras=long.loc[(long.feature.str.startswith('def_')|long.feature.eq('ctx_mpg'))&long.estimate_date.isin(dates)].copy()
    extras['model_version']=extras.estimator
    op=pd.concat([own_provenance,extras],ignore_index=True)
    ou.to_parquet(ou_out/'features.parquet',index=False);op.to_parquet(ou_out/'provenance.parquet',index=False)
    om=dict(old_ou,status='passed_identity_clean_legal_annual_ou_replay_input_candidate',rows=len(ou),
        identity_policy=meta['identity_policy'],identity_admission=str(admission_path),
        identity_admission_sha256=sha(admission_path),own_estimates='Original v1 OU values held exactly; only unadmitted player/date populations removed')
    om['sources']={str(p):sha(p) for p in [ou_source/'features.parquet',ou_source/'provenance.parquet',
        ou_source/'manifest.json',out/'features.parquet',out/'provenance.parquet',out/'manifest.json',
        Path(admission_path),Path(__file__)]}
    om['artifacts']={n:sha(ou_out/n) for n in ['features.parquet','provenance.parquet']}
    write_json(ou_out/'manifest.json',om)
    print(json.dumps(dict(status='passed',eb_rows=len(wide),ou_rows=len(ou),removed_rows=len(rejected),
        removed_ids=int(rejected.player_id.nunique()))),flush=True)
    return meta,om


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--ou-out',type=Path,default=OU_OUT);ap.add_argument('--admission',type=Path,default=ADMISSION)
    args=ap.parse_args();build(args.out,args.ou_out,admission_path=args.admission)

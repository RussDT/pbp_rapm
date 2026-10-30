"""Verify and combine annual-vintage skill packages into monthly VPM features."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_monthly_own_skills import OUT as CORE, ROOT, sha, write_json
from build_monthly_defensive_skills import OUT as DEFENSE
from build_nba_estimated_skills import SKILLS

OUT=ROOT/'results/estimated_skills/vpm_monthly_neutral_v1'


def assemble(core=CORE,defense=DEFENSE,out=OUT):
    expected_cuts=set(pd.date_range('2019-01-01','2026-09-01',freq='MS'))|{pd.Timestamp(f'{y}-10-01') for y in range(2015,2026)}
    files=[core/'snapshots'/str(y)/(s.name+'.parquet') for y in range(2015,2026) for s in SKILLS]
    files += [defense/'snapshots'/f'{y}.parquet' for y in range(2015,2026)]
    missing=[str(p) for p in files if not p.exists()]
    if missing:raise FileNotFoundError(f'{len(missing)} skill packages remain: {missing[:8]}')
    frames=[];models={}
    for path in files:
        d=pd.read_parquet(path)
        if not d.last_observation_date.lt(d.estimate_date).all() or not d.parameter_fit_end.lt(d.estimate_date).all():
            raise ValueError(f'Observation/parameter date fence failed: {path}')
        if not d.parameter_fit_cut.le(d.estimate_date).all():raise ValueError('Model not yet fitted at cut')
        for year in d.vintage.unique():
            model=defense/'models'/f'{year}.json' if path.parent.name=='snapshots' else core/'models'/str(year)/(path.stem+'.json')
            digest=sha(model)
            if not d.loc[d.vintage.eq(year),'model_sha256'].eq(digest).all():raise ValueError('Snapshot model checksum mismatch')
            models[str(model)]=digest
        frames.append(d)
    long=pd.concat(frames,ignore_index=True)
    if long.duplicated(['player_id','estimate_date','feature']).any():raise ValueError('Duplicate monthly player skill')
    for feature,g in long.groupby('feature'):
        if set(g.estimate_date)!=expected_cuts:raise ValueError(f'Missing feature cuts: {feature}')
    wide=long.pivot(index=['player_id','estimate_date'],columns='feature',values='value').reset_index()
    definitions={c:dict(units='fraction' if c.endswith('_pct') or c.endswith('_suppression') else 'per 100 source possessions',
                        provenance='feature_provenance.parquet') for c in wide if c.startswith('own_')}
    for zone,attempts in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),('three','three_fga_rate'),('ft','fta_rate')]:
        wide[f'own_{zone}_makes100']=wide[f'own_{attempts}']*wide[f'own_{zone}_pct']
        definitions[f'own_{zone}_makes100']=dict(formula=f'own_{attempts} * own_{zone}_pct',units='makes per 100 offensive possessions')
        if not wide[f'own_{zone}_pct'].dropna().between(0,1).all():raise ValueError('Invalid shooting percentage')
    for c in list(wide.columns):
        if c.startswith('own_'):wide[c+'_missing']=wide[c].isna().astype(int)
    complete=wide.dropna(subset=['own_rim_saved100','own_rim_contests100','own_rim_suppression'])
    gap=float((complete.own_rim_saved100-2*complete.own_rim_contests100*complete.own_rim_suppression).abs().max())
    if gap>1e-10:raise ValueError('Rim saved/volume/suppression identity failed')
    out.mkdir(parents=True,exist_ok=True)
    long.to_parquet(out/'feature_provenance.parquet',index=False);wide.to_parquet(out/'features.parquet',index=False)
    manifest=dict(status='chronological_skill_features_not_completed_monthly_ratings',schema_version=1,
        cuts=[str(c.date()) for c in sorted(expected_cuts)],rows=len(wide),provenance_rows=len(long),
        observed_features=int(long.feature.nunique()),feature_definitions=definitions,model_count=len(models),models=models,
        sources={str(p):sha(p) for p in [*files,core/'source_lock.json',defense/'source_lock.json',Path(__file__)]},
        artifacts={n:sha(out/n) for n in ['features.parquet','feature_provenance.parquet']},
        rim_identity_max_gap=gap,observation_rule='strictly before cut; neutral latest state held across idle days',
        parameter_rule='annual vintage fit only on observations before its preseason cut; preceding-season corrections',
        historical_parameter_replay=False,nested_chronological_fit=True,fit_end='per-feature provenance',
        max_state_age_days=730,context_in_prior=False,confirmation='2027 protected',
        limitations=['Architecture and search settings are retrospective research choices, not historically published models',
                     'Source and biographical records are retrospective archives, not certified publication vintages',
                     'Latent variances are not calibrated confidence intervals; rim-state variance is unavailable'])
    write_json(out/'manifest.json',manifest)
    print(json.dumps(dict(output=str(out),rows=len(wide),cuts=len(expected_cuts),models=len(models))))
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--core',type=Path,default=CORE)
    ap.add_argument('--defense',type=Path,default=DEFENSE);ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();assemble(args.core,args.defense,args.out)

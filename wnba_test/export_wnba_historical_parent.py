#!/usr/bin/env python3
"""Compatible historical WNBA count/parent-label adapter (no model selection).

Annual counts become available only after Dec 31. LUMA supplies an explicitly
compatible aggregate-stint parent likelihood, never event-level DECOMP/SQ.
"""
from pathlib import Path
import sys,json,unicodedata,re,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wnba_test.build_luma_historical_metrics import load_luma_season,build_design,solve_block


def norm(x):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def export(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=out/'manifest.json'
    if receipt.exists():
        for source,expected in json.loads(receipt.read_text())['sources'].items():
            if digest(source)!=expected:raise ValueError('Frozen historical source changed; choose a new output directory')
    manifest=json.loads((ROOT/'wnba_test/Results/luma_historical_our_priors/manifest.json').read_text())
    luma=Path(manifest['source_root']);cw=pd.read_csv(luma/'data/crosswalk.csv',dtype=str).fillna('')
    cw['normalized']=cw.display_name.map(norm)
    source_dir=ROOT/'nba_pipeline/external/gabriel_wnba/data'
    fields={'rim_fga':['AtRimFGA'],'rim_fgm':['AtRimFGM'],'mid_fga':['ShortMidRangeFGA','LongMidRangeFGA'],'mid_fgm':['ShortMidRangeFGM','LongMidRangeFGM'],'three_fga':['FG3A'],'three_fgm':['FG3M'],'fta':['FTA'],'ftm':['FtPoints'],'ast':['Assists'],'tov':['Turnovers'],'oreb':['OffRebounds'],'dreb':['DefRebounds'],'stl':['Steals'],'blk':['Blocks'],'pf':['Fouls']}
    counts=[];labels=[];audit=[];sources={str(luma/'data/crosswalk.csv'):digest(luma/'data/crosswalk.csv')}
    for year in range(2009,2022):
        path=source_dir/f'{year}_pbp.csv'
        if not path.exists():raise FileNotFoundError(path)
        sources[str(path)]=digest(path);raw=pd.read_csv(path,low_memory=False)
        if raw.EntityId.duplicated().any():raise ValueError('nonunique annual identity')
        f=pd.DataFrame({'person_id':raw.EntityId.astype(int),'player_name':raw.Name,'season':year,'season_type':'rs','game_id':-year,'game_date':pd.Timestamp(year,12,31),'annual_aggregate':True})
        for k,cols in fields.items():
            if not set(cols).issubset(raw):raise ValueError((year,k,'missing counts'))
            f[k]=raw[cols].apply(pd.to_numeric,errors='coerce').sum(axis=1)
        f['possessions']=(raw.OffPoss+raw.DefPoss)/2
        if (f.possessions<0).any():raise ValueError('negative exposure')
        f=f[f.possessions.gt(0)].copy();counts.append(f)
        if year==2009:continue
        active=cw[(pd.to_numeric(cw.first_season)<=year)&(pd.to_numeric(cw.last_season)>=year)].copy()
        raw['normalized']=raw.Name.map(norm)
        # Exact normalized names must be unique in both providers. A numeric
        # crosswalk may support that evidence but cannot override a conflict.
        left=active[~active.normalized.duplicated(keep=False)]
        right=raw[~raw.normalized.duplicated(keep=False)]
        joined=left.merge(right[['EntityId','normalized']],on='normalized',validate='one_to_one')
        numeric=pd.to_numeric(joined.wnba_stats_id,errors='coerce')
        conflict=numeric.notna() & numeric.ne(joined.EntityId)
        joined=joined[~conflict]
        mapping=dict(zip(joined.luma_id,joined.EntityId.astype(int)))
        sd=load_luma_season(luma,year,('rs',))
        for r in sd.audit['files']:sources[r['path']]=r['sha256']
        x,players,w,_=build_design(sd.observations)
        result=solve_block(x,players,w,(sd.observations.points/sd.observations.possessions).to_numpy(),off_alpha=2000,def_alpha=5000)
        p=len(players)
        for i,player in enumerate(players):
            if player in mapping:
                labels.append(dict(player_id=mapping[player],season=year,off=float(result.beta[i,0]),**{'def':float(result.beta[p+i,0])},off_poss=float(result.off_exposure[i]),def_poss=float(result.def_exposure[i]),source='luma_aggregate_oliver_possessions'))
        audit.append(dict(season=year,label_players=p,mapped=sum(p in mapping for p in players),numeric_name_conflicts=int(conflict.sum()),raw_feature_rows=len(raw),retained_stint_sides=len(sd.observations)))
        print(year,audit[-1],flush=True)
    pd.concat(counts,ignore_index=True).to_parquet(out/'annual_counts.parquet',index=False)
    pd.DataFrame(labels).to_parquet(out/'historical_labels.parquet',index=False)
    pd.DataFrame(audit).to_csv(out/'identity_audit.csv',index=False)
    report=dict(status='compatible_research',availability='Annual RS counts embargoed through December 30; available December 31 for next-year snapshots',sources=sources,label_contract='RS actual points / Oliver possessions, alpha2000/5000, no prior, exposure centered; compatible with modern event labels, not identical measurement')
    (out/'manifest.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'wnba_test/Results/predictive_historical_parent_v1');a=p.parse_args();export(a.output_dir)

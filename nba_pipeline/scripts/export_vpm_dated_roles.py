"""Dated position-history mixtures for a role-aware VPM prior; no impact inputs."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
from fetch_matchup_bpm import parse_advanced

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/estimated_skills/vpm_dated_roles_v1'
POSITIONS=ROOT/'external/matchup_bpm_v1/player_seasons.parquet'
OWN=ROOT/'results/estimated_skills/vpm_monthly_neutral_v1/features.parquet'
COLS=['year','nba_id','position','minutes','source_last_game_date','earliest_use_date']
ROLES=['role_guard','role_wing','role_big']
DATES=sorted([f'{y}-10-01' for y in range(2015,2026) if y!=2020]+['2020-12-01','2026-09-01'])


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def role_membership(position):
    labels=re.split(r'[-/]',str(position).upper());mapping={'PG':0,'SG':0,'G':0,'SF':1,'PF':1,'F':1,'C':2}
    if any(x not in mapping for x in labels):raise ValueError('Unrecognized role position: '+str(position))
    p=np.zeros(3)
    for x in labels:p[mapping[x]]+=1/len(labels)
    return p


def snapshot(positions,players,cut):
    cut=pd.Timestamp(cut)
    d=positions.loc[pd.to_datetime(positions.earliest_use_date).lt(cut)&pd.to_datetime(positions.source_last_game_date).ge(cut-pd.Timedelta(days=730))].copy()
    assert not d.duplicated(['year','nba_id']).any()
    d=d.sort_values(['nba_id','year']).groupby('nba_id').tail(2).copy()
    d['w']=d.minutes*.5**(d.groupby('nba_id').year.transform('max')-d.year)
    a=np.array([role_membership(x) for x in d.position]) if len(d) else np.empty((0,3))
    for i,r in enumerate(ROLES):d[r]=a[:,i]*d.w
    g=d.groupby('nba_id');shares=g[ROLES].sum().div(g.w.sum(),axis=0)
    out=pd.DataFrame(index=pd.Index(players,name='player_id'));out=out.join(shares)
    out['role_source_years']=g.year.apply(lambda x:'|'.join(map(str,x))).reindex(out.index)
    out['role_source_last_date']=g.source_last_game_date.max().reindex(out.index)
    out['role_available_at']=g.earliest_use_date.max().reindex(out.index)
    out['role_known']=out.role_guard.notna();out['estimate_date']=cut
    np.testing.assert_allclose(out.loc[out.role_known,ROLES].sum(axis=1),1,atol=1e-12)
    assert pd.to_datetime(out.role_available_at).dropna().lt(cut).all()
    assert pd.to_datetime(out.role_source_last_date).dropna().lt(cut).all()
    return out.reset_index()


def run(out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    positions=pd.read_parquet(POSITIONS,columns=COLS);positions.nba_id=positions.nba_id.astype(int)
    assert positions.minutes.gt(0).all() and not positions.duplicated(['year','nba_id']).any()
    own=pd.read_parquet(OWN,columns=['player_id','estimate_date']);own.estimate_date=pd.to_datetime(own.estimate_date)
    sources=[POSITIONS,POSITIONS.parent/'manifest.json',OWN,Path(__file__),Path(__file__).with_name('fetch_matchup_bpm.py')]
    # Independent direct archived HTML verification of selected position/minutes.
    crosswalk=pd.read_parquet(POSITIONS,columns=['year','nba_id','bref_id']);verified=0
    for year in range(2017,2027):
        path=POSITIONS.parent/'raw'/f'NBA_{year}_advanced.html';sources.append(path)
        direct=parse_advanced(path.read_text(),year)[['year','bref_id','position','minutes']]
        ref=crosswalk.loc[crosswalk.year.eq(year)].merge(direct,on=['year','bref_id'],validate='one_to_one')
        old=positions.loc[positions.year.eq(year)].set_index('nba_id').reindex(ref.nba_id)
        np.testing.assert_array_equal(old.position,ref.position);np.testing.assert_array_equal(old.minutes,ref.minutes);verified+=len(ref)
    frames=[];coverage=[];checked=0
    for value in DATES:
        cut=pd.Timestamp(value);ids=own.loc[own.estimate_date.eq(cut),'player_id'].to_numpy();d=snapshot(positions,ids,cut)
        truncated=snapshot(positions.loc[pd.to_datetime(positions.earliest_use_date).lt(cut)],ids,cut);pd.testing.assert_frame_equal(d,truncated)
        # A literal per-player source aggregation verifies mixture and missingness.
        for p in d.itertuples():
            h=positions.loc[positions.nba_id.eq(p.player_id)&pd.to_datetime(positions.earliest_use_date).lt(cut)&pd.to_datetime(positions.source_last_game_date).ge(cut-pd.Timedelta(days=730))].sort_values('year').tail(2)
            if h.empty:assert not p.role_known
            else:
                w=h.minutes.to_numpy()*.5**(h.year.max()-h.year.to_numpy());expected=sum(role_membership(x)*z for x,z in zip(h.position,w))/w.sum()
                np.testing.assert_allclose(expected,[p.role_guard,p.role_wing,p.role_big],atol=1e-12);checked+=1
        frames.append(d);coverage.append(dict(estimate_date=value,players=len(d),known=int(d.role_known.sum()),missing=int((~d.role_known).sum())))
        print('Dated roles',value,len(d),'known',int(d.role_known.sum()),flush=True)
    data=pd.concat(frames,ignore_index=True);data.to_parquet(out/'features.parquet',index=False)
    manifest=dict(status='passed',role_columns=ROLES,admitted_source_columns=COLS,forbidden_impact_fields_excluded=True,source_years=list(range(2017,2027)),source_rows=verified,literal_player_mixtures_checked=checked,coverage=coverage,protected_outcomes=0,historical_publication_vintage_certified=False,source_hashes={str(p):sha(p) for p in sources},artifact_hashes={'features.parquet':sha(out/'features.parquet')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Role source audit passed',flush=True)

if __name__=='__main__':run()

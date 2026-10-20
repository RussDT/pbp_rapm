"""Additional dated summaries for forward-trained VPM; raw measurements stay RAPM-owned."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from build_vpm_defense_bank import summarize, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/estimated_skills/vpm_forward_expanded_v1'

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/estimated_skills/defense_bank_v51_shooting_pf_repaired'
    meta=json.loads((source/'manifest.json').read_text())
    audit=json.loads((source/'independent_audit.json').read_text())
    assert audit['status']=='passed' and audit['independent_source_shooting_events']==327958
    for name,h in meta['artifact_hashes'].items():assert sha(source/name)==h
    lp=ROOT/'results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet'
    own=ROOT/'results/estimated_skills/vpm_monthly_neutral_v1/features.parquet'
    lines=pd.read_parquet(lp);lines.game_date=pd.to_datetime(lines.game_date)
    assert lines.season.max()<=2026
    measured=pd.read_parquet(source/'measurements.parquet')
    dates=[f'{y}-10-01' for y in range(2015,2026) if y!=2020]+['2020-12-01','2026-09-01']
    packets=[];histories=[];maximum=0.;cells=0
    previous=pd.read_parquet(source/'features.parquet')
    own_dates=set(pd.to_datetime(pd.read_parquet(own,columns=['estimate_date']).estimate_date))
    for value in sorted(dates):
        cut=pd.Timestamp(value);assert cut in own_dates
        players=np.sort(lines.loc[lines.game_date.lt(cut),'player_id'].unique())
        d,p=summarize(measured,players,cut,half_life=550)
        d['estimate_date']=cut
        h=lines.loc[lines.game_date.lt(cut)&lines.game_date.ge(cut-pd.Timedelta(days=365))&lines.season_type.eq('rs')&lines.minutes.gt(0)]
        g=h.groupby('player_id').agg(ctx_mpg=('minutes','mean'),ctx_games=('game_id','size'),last=('game_date','max'))
        d=d.merge(g[['ctx_mpg','ctx_games']],on='player_id',how='left',validate='one_to_one')
        if len(g):
            histories.append(pd.DataFrame(dict(player_id=g.index,feature='ctx_mpg',estimate_date=cut,last_observation_date=g['last'].to_numpy(),measured_exposure=g.ctx_games.to_numpy())))
        histories.append(p);packets.append(d)
        # Independent literal weighted-rate calculation for every exported cell.
        prior=measured.loc[measured.game_date.lt(cut)].copy()
        for feature in meta['features']:
            z=prior.loc[prior.feature.eq(feature)].copy();w=.5**((cut-z.game_date).dt.total_seconds()/86400/550)
            z['n']=w*z.numerator;z['e']=w*z.denominator
            totals=z.groupby('player_id')[['n','e']].sum();league=z.n.sum()/z.e.sum() if z.e.sum()>0 else 0.
            expected=(100*(totals.n+1000*league)/(totals.e+1000)).reindex(d.player_id).to_numpy()
            np.testing.assert_allclose(d[feature],expected,atol=1e-10,rtol=0,equal_nan=True)
            if np.isfinite(expected).any():maximum=max(maximum,float(np.nanmax(abs(d[feature]-expected))))
            cells+=len(d)
        old=previous.loc[previous.estimate_date.eq(cut)].set_index('player_id')
        if len(old):
            now=d.set_index('player_id').reindex(old.index)
            for feature in meta['features']:np.testing.assert_allclose(now[feature],old[feature],atol=1e-10,rtol=0,equal_nan=True)
        print('Forward feature snapshot',value,len(d),flush=True)
    pd.concat(packets,ignore_index=True).to_parquet(OUT/'features.parquet',index=False)
    p=pd.concat(histories,ignore_index=True);assert p.last_observation_date.lt(p.estimate_date).all()
    p.to_parquet(OUT/'provenance.parquet',index=False)
    paths=[lp,own,source/'manifest.json',source/'independent_audit.json',source/'measurements.parquet',Path(__file__),Path(__file__).with_name('build_vpm_defense_bank.py')]
    result=dict(status='passed',source_hashes={str(p):sha(p) for p in paths},artifact_hashes={p:sha(OUT/p) for p in ['features.parquet','provenance.parquet']},
        dates=sorted(dates),independently_reconstructed_cells=cells,maximum_gap=maximum,shooting_foul_source_events=327958,
        mpg='strictly prior365-day RS minutes per actual positive-minute appearance; missing without appearances',protected_outcomes=0)
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':run()

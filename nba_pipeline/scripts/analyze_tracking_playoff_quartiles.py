"""Observed tracking playoff quartiles. Exploratory; no predictive promotion."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/estimated_skills/tracking_playoff_quartiles'
SPECS={
 'potential_assists':('potential_ast_observed','off_poss'),
 'drives':('DRIVES','off_poss'),
 'possession_minutes':('TIME_OF_POSS','off_poss'),
 'contested_dreb':('DREB_CONTEST','def_poss'),
 'contested_shots':('hustle_CONTESTED_SHOTS','def_poss'),
 'dfga':('dfga_observed','def_poss'),
 'deflections':('hustle_DEFLECTIONS','def_poss')}


def bootstrap_change(x,draws=2000):
    ids,players=pd.factorize(x.player_id)
    a=np.bincount(ids,weights=x.weight*x.change100)
    w=np.bincount(ids,weights=x.weight)
    rng=np.random.default_rng(20260913)
    samples=[]
    for _ in range(draws):
        n=rng.multinomial(len(players),np.full(len(players),1/len(players)))
        samples.append(n@a/(n@w))
    return np.percentile(samples,[2.5,97.5])


def aggregate_phase(x,field,exposure):
    valid=x[field].notna() & x[field].ge(0) & x[exposure].gt(0)
    d=x.assign(value=x[field].where(valid,0),used_exposure=x[exposure].where(valid,0),used_minutes=x.MIN.where(valid,0),used=valid.astype(int))
    a=d.groupby('player_id').agg(count=('value','sum'),exposure=('used_exposure','sum'),covered_minutes=('used_minutes','sum'),minutes=('MIN','sum'),games=('used','sum'))
    a['coverage']=a.covered_minutes/a.minutes
    return a


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    t=pd.read_parquet(ROOT/'results/estimated_skills/tracking_backfill/tracking_backfill.parquet')
    t=t[t.season.between(2019,2025)].rename(columns={'PLAYER_ID':'player_id'})
    t['season_type']=np.where(t.playoffs,'ps','rs')
    t['game_date']=pd.to_datetime(t.date.astype(str),format='%Y%m%d')
    p=pd.read_parquet(ROOT/'results/estimated_skills/player_game_lines.parquet')
    keys=['player_id','season','season_type','game_date']
    p=p[~p.duplicated(keys,keep=False)]
    assert not t.duplicated(keys).any()
    d=t.merge(p[keys+['off_poss','def_poss','minutes','ast','fga']],on=keys,how='left',validate='one_to_one')
    match=d.MIN.sub(d.minutes).abs().le(.5)&d.AST.eq(d.ast)&d.FGA.eq(d.fga)
    d.loc[~match,['off_poss','def_poss']]=np.nan
    # Never put synthetic repair values into the observed comparisons.
    for stat in ['potential_ast','dfga']:
        d.loc[~d[stat+'_is_measured'],stat+'_observed']=np.nan
    elite=pd.read_csv(ROOT/'results/estimated_skills/deflections_playoff_context/elite_player_seasons.csv')
    records=[];coverage=[]
    for stat,(field,exposure) in SPECS.items():
        warm=d[d.season.eq(2019)&d.season_type.eq('rs')]
        ok=warm[field].ge(0)&warm[exposure].gt(0)
        mean=warm.loc[ok,field].sum()/warm.loc[ok,exposure].sum()
        for season in range(2020,2026):
            s=d[d.season.eq(season)]
            cutoff=s.loc[s.season_type.eq('ps'),'game_date'].min()
            r=s[s.season_type.eq('rs')&s.game_date.lt(cutoff)]
            p=s[s.season_type.eq('ps')]
            a=aggregate_phase(r,field,exposure); b=aggregate_phase(p,field,exposure)
            for phase,z in [('rs',a),('ps',b)]:
                coverage.append({'stat':stat,'season':season,'phase':phase,'players':len(z),'minutes_coverage':z.covered_minutes.sum()/z.minutes.sum(),'players_95pct_coverage':int(z.coverage.ge(.95).sum())})
            a=a[a.coverage.ge(.95)&a.exposure.ge(200)].copy()
            # Fixed 200-possession shrinkage to 2019 RS mean; not tuned to playoffs.
            a['rank_rate100']=100*(a['count']+200*mean)/(a.exposure+200)
            a['quartile']=pd.qcut(a.rank_rate100.rank(method='first'),4,labels=[1,2,3,4]).astype(int)
            b=b[b.coverage.ge(.95)&b.exposure.gt(0)]
            z=a.join(b,how='inner',lsuffix='_rs',rsuffix='_ps').reset_index()
            z['stat']=stat;z['season']=season
            z['rs100']=100*z.count_rs/z.exposure_rs;z['ps100']=100*z.count_ps/z.exposure_ps
            z['change100']=z.ps100-z.rs100
            z['weight']=z.exposure_rs*z.exposure_ps/(z.exposure_rs+z.exposure_ps)
            if stat=='deflections':
                ranks=elite[elite.season.eq(season)][['player_id','quartile','skill100']]
                z=z.drop(columns=['quartile','rank_rate100']).merge(ranks,on='player_id',validate='one_to_one').rename(columns={'skill100':'rank_rate100'})
            records.append(z)
    pairs=pd.concat(records,ignore_index=True)
    summaries=[]
    for period,years in [('all',[2020,2021,2022,2023,2024,2025]),('development',[2020,2021,2022]),('spent_stress',[2023,2024,2025])]:
        for (stat,q),x in pairs[pairs.season.isin(years)].groupby(['stat','quartile']):
            lo,hi=bootstrap_change(x)
            summaries.append({'period':period,'stat':stat,'quartile':q,'player_seasons':len(x),'players':x.player_id.nunique(),'ps_possessions':x.exposure_ps.sum(),'rs100':np.average(x.rs100,weights=x.weight),'ps100':np.average(x.ps100,weights=x.weight),'change100':np.average(x.change100,weights=x.weight),'ci_low':lo,'ci_high':hi})
    result=pd.DataFrame(summaries)
    result.to_csv(OUT/'quartiles.csv',index=False);pairs.to_parquet(OUT/'player_seasons.parquet',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    lines=['# Tracking playoff quartiles','', 'Exploratory 2020–2025 matched-player comparisons. Every rate is per100 relevant possessions; possession time is minutes/100. Intervals are pointwise player-cluster bootstrap intervals, not corrected for multiple comparisons.','', '| Stat | Quartile | RS | PS | Change [95% interval] | Player-seasons |','|---|---:|---:|---:|---:|---:|']
    for x in result[result.period.eq('all')].itertuples():
        lines.append(f'| {x.stat} | {x.quartile} | {x.rs100:.2f} | {x.ps100:.2f} | {x.change100:+.2f} [{x.ci_low:+.2f}, {x.ci_high:+.2f}] | {x.player_seasons} |')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    inputs=[Path(__file__), ROOT/'results/estimated_skills/tracking_backfill/tracking_backfill.parquet', ROOT/'results/estimated_skills/player_game_lines.parquet', ROOT/'results/estimated_skills/deflections_playoff_context/elite_player_seasons.csv']
    (OUT/'dependencies.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2))
    print(result[result.period.eq('all')].to_string(index=False))

if __name__=='__main__':main()

"""Descriptive seasonal shooting patterns; never infer provider internals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/estimated_skills/league_seasonality'
SPECS={'three_pct':('three_fgm','three_fga','p_fg3pct'),'ft_pct':('ftm','fta','p_ftpct'),'rim_pct':('rim_fgm','rim_fga','p_fgpct_rim'),'mid_pct':('mid_fgm','mid_fga','p_fgpct_mid')}

def fit_fe(d,ycol,draws=1000):
    # Weighted player-season demeaning; first regular-season sixth is reference.
    x=np.column_stack([d.phase.eq(k).to_numpy().astype(float) for k in range(1,7)])
    y=d[ycol].to_numpy();w=d.weight.to_numpy();g=pd.factorize(list(zip(d.player_id,d.season)))[0]
    den=np.bincount(g,weights=w)
    xc=x-np.column_stack([np.bincount(g,weights=w*x[:,j])[g]/den[g] for j in range(6)])
    yc=y-np.bincount(g,weights=w*y)[g]/den[g]
    ids,players=pd.factorize(d.player_id);n=len(players)
    A=np.zeros((n,6,6));b=np.zeros((n,6))
    for j in range(6):
        b[:,j]=np.bincount(ids,weights=w*xc[:,j]*yc,minlength=n)
        for k in range(6):A[:,j,k]=np.bincount(ids,weights=w*xc[:,j]*xc[:,k],minlength=n)
    def solve(counts):return np.linalg.lstsq(np.einsum('i,ijk->jk',counts,A),counts@b,rcond=None)[0]
    beta=solve(np.ones(n));rng=np.random.default_rng(20260914)
    samples=np.array([solve(rng.multinomial(n,np.full(n,1/n))) for _ in range(draws)])
    lo,hi=np.percentile(samples,[2.5,97.5],axis=0)
    return [(0,0.,0.,0.)]+[(k,float(beta[k-1]),float(lo[k-1]),float(hi[k-1])) for k in range(1,7)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lines=pd.read_parquet(ROOT/'results/estimated_skills/player_game_lines.parquet')
    lines=lines[lines.season.between(2019,2025)].copy()
    lines=lines[~lines.duplicated(['player_id','game_date'],keep=False)]
    calendar=lines[lines.season_type.eq('rs')].groupby('season').game_date.agg(['min','max'])
    lines['phase']=np.minimum(5,np.floor(6*(lines.game_date-lines.season.map(calendar['min'])).dt.days/(lines.season.map(calendar['max'])-lines.season.map(calendar['min'])).dt.days.clip(lower=1)).astype(int))
    lines.loc[lines.season_type.eq('ps'),'phase']=6
    pred=pd.read_parquet(ROOT/'results/estimated_skills/predictions.parquet')
    output=[];means=[]
    for skill,(num,den,field) in SPECS.items():
        d=lines[lines[den].gt(0)].copy();d['weight']=d[den];d['rate']=100*d[num]/d[den]
        p=pred[pred.skill.eq(skill)][['player_id','game_id','pred_rate']]
        d=d.merge(p,on=['player_id','game_id'],validate='one_to_one');d['residual']=d.rate-100*d.pred_rate
        for sample,z in [('standard_seasons',d[~d.season.isin([2020,2021])]),('all_seasons',d)]:
            for target in ['rate','residual']:
                for phase,e,lo,hi in fit_fe(z,target):output.append(dict(skill=skill,sample=sample,target=target,phase=phase,effect_pp=e,lo=lo,hi=hi))
        for (year,phase),z in d.groupby(['season','phase']):means.append(dict(skill=skill,season=year,phase=phase,rate=np.average(z.rate,weights=z.weight),residual=np.average(z.residual,weights=z.weight),attempts=z.weight.sum()))
    pd.DataFrame(output).to_csv(OUT/'controlled_effects.csv',index=False);pd.DataFrame(means).to_csv(OUT/'season_phase_rates.csv',index=False)
    # Common-player panels of game-agnostic EPM, equal player weights.
    snaps=[];seen=set()
    for folder in ['epm','epm_title']:
        for path in sorted((ROOT/'cache'/folder).glob('*go0.csv')):
            if path.name in seen:continue
            seen.add(path.name)
            e=pd.read_csv(path)
            if not {'requested_date','date','season','player_id',* [v[2] for v in SPECS.values()]}.issubset(e.columns):continue
            e=e[e.season.isin(calendar.index)]
            if e.empty:continue
            e['effective']=pd.to_datetime(e.date)
            e=e[e.effective.ge(e.season.map(calendar['min']))&e.effective.le(e.season.map(calendar['max']))]
            if e.empty:continue
            snaps.append(e)
    epm=[]
    if snaps:
        e=pd.concat(snaps).sort_values('requested_date').drop_duplicates(['season','effective','player_id'],keep='first')
        for year,z in e.groupby('season'):
            if z.effective.nunique()<3:continue
            counts=z.groupby('player_id').effective.nunique();ids=counts[counts.eq(z.effective.nunique())].index;z=z[z.player_id.isin(ids)]
            for date,g in z.groupby('effective'):
                for skill,(_,_,field) in SPECS.items():epm.append(dict(season=int(year),date=str(date.date()),skill=skill,mean_pct=100*g[field].mean(),players=len(g),mode='game_agnostic'))
    pd.DataFrame(epm).to_csv(OUT/'epm_fixed_player_series.csv',index=False)
    # Render an offline static chart when matplotlib is available.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=pd.DataFrame(output);fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,skill in zip(axes.flat,SPECS):
        for target,label in [('rate','Observed, player-season adjusted'),('residual','Error vs our pregame estimate')]:
            z=r[(r.skill==skill)&(r['sample']=='standard_seasons')&(r.target==target)]
            ax.plot(z.phase,z.effect_pp,marker='o',label=label);ax.fill_between(z.phase,z.lo,z.hi,alpha=.15)
        ax.axhline(0,color='gray',lw=.6);ax.set_title(skill);ax.set_xticks(range(7),['RS1','RS2','RS3','RS4','RS5','RS6','PS']);ax.set_ylabel('Percentage points vs RS1')
    axes[0,0].legend(fontsize=8);fig.suptitle('Seasonal shooting pattern • 2019, 2022–2025\nPlayer-season controls; player-cluster 95% intervals; descriptive')
    fig.tight_layout();fig.savefig(OUT/'seasonality.png',dpi=160)
    print(r[(r['sample']=='standard_seasons')&r.phase.isin([3,5,6])].to_string(index=False))
    print('epm rows',len(epm))
if __name__=='__main__':main()

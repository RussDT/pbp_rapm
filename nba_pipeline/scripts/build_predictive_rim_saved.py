"""Rolling six-foot contest savings; lagged league baseline, no provider normal%."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from build_deflections_next_game import compare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/estimated_skills/predictive_rim_saved'
SOURCE = ROOT/'results/estimated_skills/rim_defense_matchups/player_games.parquet'


def lagged_league(d):
    daily = d.groupby('game_date')[['rim_dfgm','rim_dfga']].sum().sort_index()
    previous = daily.cumsum().shift(1).fillna(0)
    return d.game_date.map((previous.rim_dfgm+65)/(previous.rim_dfga+100)).to_numpy()


def rolling(d, volume_half, suppression_half, k, mean_volume):
    """Prediction before update; missing observations excluded by caller.

    Discount sufficient statistics, then add fixed prior exposure. Denominators
    reflect opportunities, never unweighted game percentages.
    """
    out = np.empty((len(d),6))
    ids = d.player_id.to_numpy(); days=d.game_date.to_numpy().astype('datetime64[D]').astype(int)
    n=d.rim_dfga.to_numpy(); m=d.rim_dfgm.to_numpy(); e=d.def_poss.to_numpy(); q=d.league_pct.to_numpy()
    last_id=None
    for i in range(len(d)):
        if ids[i] != last_id:
            vn=ve=sn=sr=0.; previous=days[i]; last_id=ids[i]
        gap=days[i]-previous
        vd=2**(-gap/volume_half); sd=2**(-gap/suppression_half)
        vn*=vd; ve*=vd; sn*=sd; sr*=sd
        volume=(vn+1000*mean_volume)/(ve+1000)
        suppression=np.clip(sr/(sn+k),q[i]-1,q[i])
        out[i,:3]=[100*volume,suppression,200*volume*suppression]
        vn+=n[i];ve+=e[i];sn+=n[i];sr+=q[i]*n[i]-m[i]
        volume=(vn+1000*mean_volume)/(ve+1000)
        suppression=np.clip(sr/(sn+k),q[i]-1,q[i])
        out[i,3:]=[100*volume,suppression,200*volume*suppression]
        previous=days[i]
    return out


def mse(d,pred,target,weight,seasons):
    return float(np.mean([np.average((pred[d.season.eq(s)]-target[d.season.eq(s)])**2,weights=weight[d.season.eq(s)]) for s in seasons]))


def refresh_frozen(source, out, frozen):
    """Rebuild states after a count repair without tuning or opening new scores."""
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    config = json.loads(frozen.read_text())
    full = pd.read_parquet(source)
    d = full.loc[full.rim_counts_valid].sort_values(['player_id', 'game_date']).copy()
    d['league_pct'] = lagged_league(d)
    d['target100'] = 200 * (d.league_pct*d.rim_dfga-d.rim_dfgm)/d.def_poss
    arr = rolling(d, config['volume_half_life_days'], config['selected']['half_life_days'],
                  config['selected']['prior_attempts'], config['mean_volume_2019'])
    for i,c in enumerate(['pred_contests100','pred_suppression','pred_saved100',
                          'post_contests100','post_suppression','post_saved100']):
        d[c] = arr[:, i]
    d['pred_allowed_pct'] = d.league_pct-d.pred_suppression
    out.mkdir(parents=True)
    keep = ['player_id','tracking_name','game_id','game_date','season','season_type','def_poss',
            'rim_dfga','rim_dfgm','league_pct','target100','pred_contests100','pred_suppression',
            'pred_saved100','post_contests100','post_suppression','post_saved100','pred_allowed_pct']
    d[keep].to_parquet(out/'dated_predictions.parquet',index=False)
    latest = d.groupby('player_id').tail(1).copy()
    latest['estimated_rim_points_saved100'] = latest.post_saved100
    latest.to_csv(out/'current_states.csv', index=False)
    config.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  repair_status='cumulative counts recovered; frozen parameters reused; no new scoring',
                  frozen_parameter_source=str(frozen),
                  frozen_parameter_source_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                  historical_leakage_free=False)
    (out/'frozen.json').write_text(json.dumps(config,indent=2))
    coverage=d.groupby(['season','season_type']).agg(rows=('game_id','size'),games=('game_id','nunique'))
    coverage.to_csv(out/'coverage.csv')
    print(coverage.to_string())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    full=pd.read_parquet(SOURCE)
    d=full.loc[full.rim_counts_valid].copy().sort_values(['player_id','game_date']).reset_index(drop=True)
    d['league_pct']=lagged_league(d)
    d['target100']=200*(d.league_pct*d.rim_dfga-d.rim_dfgm)/d.def_poss
    d['exposure']=d.def_poss
    warm=d.season.eq(2019)
    mean_volume=float(d.loc[warm,'rim_dfga'].sum()/d.loc[warm,'def_poss'].sum())
    dev=[2023]; stress=[2024,2025]
    volumes=[]
    for h in [90,365,730]:
        arr=rolling(d,h,730,400,mean_volume)
        loss=mse(d,arr[:,0],100*d.rim_dfga/d.def_poss,d.def_poss,dev)
        volumes.append({'half_life_days':h,'mse':loss})
    vh=min(volumes,key=lambda r:r['mse'])['half_life_days']
    grid=[]
    for h in [365,730,1460]:
        for k in [100,400,1600]:
            name=f'h{h}_k{k}'; arr=rolling(d,vh,h,k,mean_volume)
            d[name]=arr[:,2]
            grid.append({'name':name,'half_life_days':h,'prior_attempts':k,'mse':mse(d,arr[:,2],d.target100,d.def_poss,dev)})
    best=min(grid,key=lambda r:r['mse'])
    devrows=d[d.season.isin(dev)].copy()
    eligible=[]
    for row in grid:
        c=compare(devrows,best['name'],row['name'],draws=300)
        if c['delta_mse']<=c['delta_se']+1e-12: eligible.append(row)
    selected=max(eligible,key=lambda r:(r['prior_attempts'],r['half_life_days']))
    d['zero']=0.
    d['candidate']=d[selected['name']]
    gate=compare(devrows.assign(candidate=devrows[selected['name']],zero=0.),'candidate','zero',draws=1000)
    accepted=gate['delta_mse']>gate['delta_se']
    config={'development':dev,'stress':stress,'2026':'export only','2027':'protected','volume_grid':volumes,'suppression_grid':grid,'selected':selected,'volume_half_life_days':vh,'volume_prior_possessions':1000,'mean_volume_2019':mean_volume,'dev_gate_vs_zero':gate,'development_gate_passed':accepted,'retained':False,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'definition':'neutral league-relative six-foot contest savings; not provider opponent-normal savings'}
    (OUT/'frozen.json').write_text(json.dumps(config,indent=2))
    arr=rolling(d,vh,selected['half_life_days'],selected['prior_attempts'],mean_volume)
    for i,c in enumerate(['pred_contests100','pred_suppression','pred_saved100','post_contests100','post_suppression','post_saved100']): d[c]=arr[:,i]
    d['pred_allowed_pct']=d.league_pct-d.pred_suppression
    # Benchmark no personalized finishing suppression on the identical contest-volume prediction.
    d['pred_allowed_points100']=2*d.pred_contests100*d.pred_allowed_pct
    d['zero_allowed_points100']=2*d.pred_contests100*d.league_pct
    d['actual_allowed_points100']=200*d.rim_dfgm/d.def_poss
    test=d[d.season.isin(stress)].copy()
    report={'saved':compare(test,'pred_saved100','zero',draws=1000),
        'allowed_points':compare(test.assign(target100=test.actual_allowed_points100),'pred_allowed_points100','zero_allowed_points100',draws=1000),
        'player_cluster':compare(test,'pred_saved100','zero',cluster='player',draws=1000),
        'yearly':[compare(g,'pred_saved100','zero',draws=300) for _,g in test.groupby('season')]}
    (OUT/'stress_report.json').write_text(json.dumps(report,indent=2))
    keep=['player_id','tracking_name','game_id','game_date','season','season_type','def_poss','rim_dfga','rim_dfgm','league_pct','target100','pred_contests100','pred_suppression','pred_saved100','post_contests100','post_suppression','post_saved100','pred_allowed_pct']
    d[keep].to_parquet(OUT/'dated_predictions.parquet',index=False)
    latest=d.groupby('player_id').tail(1).query('season == 2026').copy()
    latest['estimated_rim_points_saved100']=latest.post_saved100
    latest['status']='research candidate; historical stress only'
    coverage=full.query('season == 2026').assign(covered=lambda x:x.def_poss.where(x.rim_counts_valid,0)).groupby('player_id')[['covered','def_poss']].sum()
    latest=latest.join((coverage.covered/coverage.def_poss).rename('coverage2026'),on='player_id')
    latest['observed_games']=d.groupby('player_id').size().reindex(latest.player_id).to_numpy()
    cols=['player_id','tracking_name','game_date','estimated_rim_points_saved100','post_contests100','post_suppression','observed_games','coverage2026']
    latest=latest[cols].sort_values('estimated_rim_points_saved100',ascending=False)
    latest.to_csv(OUT/'current_2026.csv',index=False)
    (OUT/'index.html').write_text('<meta charset="utf-8"><title>Predictive rim points saved</title><style>body{font:15px system-ui;background:#111;color:#eee;margin:30px}td,th{padding:8px}th{position:sticky;top:0;background:#222}</style><h1>Rolling predictive rim points saved / 100</h1><p>Research · positive is better · six-foot contests relative to lagged league finishing. Postgame estimates as of each player’s latest covered game; not advanced to today. Missing rows do not update the model. Coverage is a fraction. This differs from opponent-adjusted observed savings.</p><input placeholder="Search player" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))">'+latest.to_html(index=False,float_format=lambda x:f'{x:.3f}'))
    print(json.dumps({'selected':selected,'volume_half':vh,'dev':gate,'stress':report['saved'],'allowed_points':report['allowed_points']},indent=2),flush=True)
    print(latest.head(12).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--frozen',type=Path,help='Reuse parameters; rebuild states without tuning/scoring')
    args=parser.parse_args()
    if args.frozen:
        refresh_frozen(args.source,args.out,args.frozen)
    else:
        SOURCE,OUT=args.source,args.out
        main()


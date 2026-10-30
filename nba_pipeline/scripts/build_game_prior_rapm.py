"""One-game joint O/D ridge centered on explicit provisional game-box anchors.

Fits theta in points/possession: ||y-X theta||² + alpha||theta-mu||².
No intercept, no silent team projection, no opponent-strength claim. Actual and
opponent-three-EV are separate joint solves; headline uses actual O and EV D.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/game_prior_rapm'
AUDIT=ROOT/'results/player_game_attribution_audit'
BOX=ROOT/'results/player_game_box_net_points'
ALPHAS=[100.,500.,2000.]

def solve(x,y,mu,alpha):
    if alpha<=0 or not np.isfinite(alpha):raise ValueError('Positive finite alpha required')
    x=np.asarray(x,float);y=np.asarray(y,float);mu=np.asarray(mu,float)
    if not all(np.isfinite(v).all() for v in [x,y,mu]):raise ValueError('Nonfinite solve input')
    delta=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@(y-x@mu))
    return mu+delta

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ledger=pd.read_parquet(AUDIT/'possession_ledger.parquet')
    boxes=pd.read_parquet(BOX/'players.parquet')
    gids=ledger.game_id.unique()
    if len(gids)!=1:raise ValueError('One-game audit required')
    boxes=boxes.loc[boxes.game_id.eq(gids[0])].sort_values('player_id').reset_index(drop=True)
    if boxes.player_id.duplicated().any():raise ValueError('Duplicate players')
    pids=boxes.player_id.astype(int).tolist();ix={p:i for i,p in enumerate(pids)};n=len(pids)
    records=[];residuals=[];terms=[];fits=[]
    for lane in ['TS','TOV','SC']:
        off=ledger.loc[ledger.factor.eq('o'+lane)].set_index(['game_id','event_num'])
        defense=ledger.loc[ledger.factor.eq('d'+lane)].set_index(['game_id','event_num'])
        if off.index.has_duplicates or defense.index.has_duplicates:raise ValueError('Duplicate possession')
        defense=defense.reindex(off.index)
        if defense.lineup.isna().any():raise ValueError('Missing defensive lineup')
        x=np.zeros((len(off),2*n))
        for j,(ol,dl) in enumerate(zip(off.lineup,defense.lineup)):
            for text,offset,sign in [(ol,0,1),(dl,n,-1)]:
                lineup=[int(p) for p in text.split(',')]
                if len(set(lineup))!=5:raise ValueError('Invalid lineup')
                for pid in lineup:x[j,offset+ix[pid]]=sign
        exposure=np.abs(x).sum(axis=0)
        prior_points=np.r_[boxes['o'+lane+'_box_points'],boxes['d'+lane+'_box_points']].astype(float)
        if np.any((exposure==0)&(np.abs(prior_points)>1e-10)):raise ValueError('Prior credit without exposure')
        mu=np.divide(prior_points,exposure,out=np.zeros_like(prior_points),where=exposure>0)
        np.testing.assert_allclose(exposure,np.r_[boxes.off_possessions,boxes.def_possessions])
        actual=off.actual_points.to_numpy(float)
        np.testing.assert_allclose(actual,-defense.actual_points.to_numpy(float),atol=1e-10)
        # Same measured possessions; EV changes only opponent-three conversion.
        expected=-defense.adjusted_points.to_numpy(float)
        for alpha in ALPHAS:
            for variant,y in [('actual',actual),('three_ev',expected)]:
                theta=solve(x,y,mu,alpha);pred=x@theta;error=y-pred
                normal_gap=float(np.max(np.abs(x.T@error-alpha*(theta-mu))))
                if normal_gap>1e-8:raise ValueError('Normal-equation failure')
                fits.append(dict(lane=lane,alpha=alpha,variant=variant,possessions=len(y),normal_equation_gap=normal_gap,prior_sse=float(np.sum((y-x@mu)**2)),fit_sse=float(error@error),penalty=float(alpha*np.sum((theta-mu)**2))))
                for i,pid in enumerate(pids):
                    for side,offset in [('o',0),('d',n)]:
                        k=offset+i
                        records.append(dict(player_id=pid,player_name=boxes.iloc[i].player_name,team=boxes.iloc[i].team,alpha=alpha,variant=variant,factor=side+lane,possessions=exposure[k],prior_points=prior_points[k],lineup_adjustment=(theta[k]-mu[k])*exposure[k],final_points=theta[k]*exposure[k],prior_rate100=mu[k]*100,final_rate100=theta[k]*100,headline=(side=='o' and variant=='actual') or (side=='d' and variant=='three_ev')))
                for j,(key,r) in enumerate(off.iterrows()):
                    residuals.append(dict(event_num=key[1],team=r.team,lane=lane,alpha=alpha,variant=variant,target=y[j],prediction=pred[j],residual=error[j]))
    players=pd.DataFrame(records);residual=pd.DataFrame(residuals);fit=pd.DataFrame(fits)
    headline=players.loc[players.headline].copy()
    totals=headline.groupby(['alpha','player_id','player_name','team'])[['prior_points','lineup_adjustment','final_points']].sum().reset_index()
    # Context model accounting: own O - opponent D predicts own scoring;
    # contribution totals must never be mislabeled as observed team margin.
    response=residual.groupby(['alpha','variant','team','lane'])[['target','prediction','residual']].sum().reset_index()
    np.testing.assert_allclose(response.target,response.prediction+response.residual,atol=1e-9)
    np.testing.assert_allclose(players.final_points,players.prior_points+players.lineup_adjustment,atol=1e-9)
    for name,frame in [('player_factors',players),('headline_totals',totals),('possession_residuals',residual),('team_response_accounting',response),('fit_checks',fit)]:
        frame.to_csv(OUT/f'{name}.csv',index=False);frame.to_parquet(OUT/f'{name}.parquet',index=False)
    def table(d):return '<div class="scroll">'+d.to_html(index=False,float_format=lambda x:f'{x:.3f}',escape=True)+'</div>'
    content=['<!doctype html><meta charset="utf-8"><title>Prior-centered game RAPM experiment</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:15px;background:#faf9f5;color:#172333}p{line-height:1.6;max-width:1050px}.scroll{overflow:auto}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}table{border-collapse:collapse;white-space:nowrap}.note{background:#fff0ce;padding:18px}h2{margin-top:40px}summary{cursor:pointer}</style><h1>NYK–CLE · Prior-centered game solve</h1><p class="note">Allocation experiment, not a validated player metric. The starting game-box priors are the existing disputed single-season mapping: the all-steals/first-chance dTOV mismatch and missing OFFD remain. No claim of beating BPM. Alpha values are sensitivities, not selected settings.</p><p>Each possession is modeled as the sum of five offensive effects minus five defensive effects. Ridge pulls every effect toward its game-box prior. Larger alpha leaves less room for the game’s lineup results to move credit. Player totals below use actual offense and three-EV defense from separately labeled joint solves.</p>']
    for alpha in ALPHAS:
        content += [f'<h2>Alpha {alpha:g}</h2>',table(totals.loc[totals.alpha.eq(alpha)].sort_values('final_points',ascending=False).drop(columns=['alpha','player_id']))]
        content += ['<details><summary>All six components: prior → adjustment → final</summary>',table(headline.loc[headline.alpha.eq(alpha),['player_name','team','factor','possessions','prior_points','lineup_adjustment','final_points','final_rate100']]),'</details>']
    content += ['<h2>What has—and has not—been distributed</h2><p>These are jointly inferred player effects conditional on the opposing game effects. They are not externally opponent-neutral ratings. Only prior totals plus lineup changes form player credit; residuals stay unassigned. To reconstruct an offense’s modeled factor, subtract the opposing defensive contribution from its offensive contribution. The table below shows target = prediction + residual for each consistent solve.</p><p>The source audit keeps the FT conversion bridge at team level; it is absent from the possession fit and has not been silently distributed. Canonical possession lineups also do not resolve substitutions inside possessions. Actual-versus-three-EV models are separate and their hybrid headline is not a single fitted score identity.</p>',table(response),'<details><summary>Numerical checks and objective</summary>',table(fit),'</details><p><a href="player_factors.csv">Every factor and rate</a> · <a href="possession_residuals.csv">Every possession residual</a> · <a href="manifest.json">Model contract</a></p>']
    (OUT/'review.html').write_text('\n'.join(content))
    import hashlib
    sources=[Path(__file__),AUDIT/'possession_ledger.parquet',AUDIT/'manifest.json',BOX/'players.parquet',BOX/'models.json']
    (OUT/'manifest.json').write_text(json.dumps(dict(game_id=str(gids[0]),alphas=ALPHAS,objective='sum((y-X theta)^2)+alpha*sum((theta-mu)^2); theta points/possession; X +O/-D',prior='existing provisional game-specific box points divided by exact side exposure; no historical player rating',intercept=False,constraints='none; residuals explicitly unassigned',opponent_adjustment='joint within-game separation, not pregame neutral-opponent adjustment',headline='actual-fit O plus three-EV-fit D; both full fits exported',status='solver experiment; priors and allocation not validated',inputs=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]),indent=2)+'\n')
    print(totals.to_string(index=False));print(OUT/'review.html')

if __name__=='__main__':main()

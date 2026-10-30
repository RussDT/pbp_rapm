"""Exploratory skill-dependent playoff diagnostic; no model selection."""
import json
import numpy as np
import pandas as pd
from analyze_deflections_playoff_context import OUT, BASE, prior


def main():
    cfg = json.loads((BASE/'frozen.json').read_text())
    sub = prior.subpanel(pd.read_parquet(prior.OUT/'panel.parquet'))
    sub = sub[sub.season.le(2025)].copy()
    model = cfg['models'][cfg['selected']]
    pred, _, post, _ = prior.engine(model['engine'], sub, model['params'], cfg['mean_rate'])
    sub['prediction100'], sub['state100'] = 100*pred, 100*post
    sub['count'] = sub.obs_rate*sub.exposure
    sub['pred_count'] = pred*sub.exposure
    records = []
    for season, season_rows in sub.groupby('season'):
        if season < 2020: continue
        ps = season_rows[season_rows.season_type.eq('ps')]
        if ps.empty: continue
        cutoff = ps.game_date.min()
        rs = season_rows[season_rows.season_type.eq('rs') & season_rows.game_date.lt(cutoff)]
        agg = rs.groupby('player_id').agg(rs_count=('count','sum'),rs_exposure=('exposure','sum'))
        last = rs.sort_values('game_date').groupby('player_id').tail(1).set_index('player_id')
        agg['skill100'] = last.state100
        agg['player_name'] = last.player_name
        agg = agg[agg.rs_exposure.ge(200)].copy()
        # Ranking universe includes eligible non-playoff players.
        agg['quartile'] = pd.qcut(agg.skill100.rank(method='first'),4,labels=[1,2,3,4]).astype(int)
        pg = ps.groupby('player_id').agg(ps_count=('count','sum'),ps_exposure=('exposure','sum'),ps_games=('game_id','nunique'),pred_count=('pred_count','sum'))
        a = agg.join(pg,how='inner').reset_index()
        a['season'] = season
        a['rs100'] = 100*a.rs_count/a.rs_exposure
        a['ps100'] = 100*a.ps_count/a.ps_exposure
        a['dynamic_pred100'] = 100*a.pred_count/a.ps_exposure
        a['change100'] = a.ps100-a.rs100
        a['error100'] = a.ps100-a.dynamic_pred100
        a['weight'] = a.rs_exposure*a.ps_exposure/(a.rs_exposure+a.ps_exposure)
        records.append(a)
    d = pd.concat(records,ignore_index=True)
    def summary(x):
        w=x.weight
        return {'player_seasons':len(x),'players':int(x.player_id.nunique()), **{k:float(np.average(x[k],weights=w)) for k in ['skill100','rs100','ps100','change100','error100']}}
    # Continuous interaction with season intercepts; player-cluster bootstrap.
    X=np.column_stack([pd.get_dummies(d.season).to_numpy(),d.skill100.to_numpy()])
    y=d.error100.to_numpy(); w=d.weight.to_numpy()
    def fit(ww):
        return float(np.linalg.lstsq(X*np.sqrt(ww[:,None]),y*np.sqrt(ww),rcond=None)[0][-1])
    ids, unique=pd.factorize(d.player_id);rng=np.random.default_rng(20260912)
    slopes=[]
    for _ in range(2000):
        counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))
        slopes.append(fit(w*counts[ids]))
    report={'quartiles':{int(q):summary(x) for q,x in d.groupby('quartile')},'residual_slope_per_skill100':fit(w),'player_cluster_ci95':np.percentile(slopes,[2.5,97.5]).tolist(),'by_period':{name:{int(q):summary(x) for q,x in d[d.season.isin(years)].groupby('quartile')} for name,years in [('development',[2020,2021,2022]),('spent_stress',[2023,2024,2025])]}}
    names=d[d.player_name.str.contains('Caruso|Thybulle',case=False)].copy()
    report['named_players']=names[['player_name','season','rs100','ps100','skill100','dynamic_pred100','ps_exposure','ps_games']].to_dict('records')
    report['named_pooled']={n:summary(x) for n,x in names.groupby('player_name')}
    OUT.mkdir(exist_ok=True,parents=True)
    d.to_csv(OUT/'elite_player_seasons.csv',index=False)
    (OUT/'elite_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

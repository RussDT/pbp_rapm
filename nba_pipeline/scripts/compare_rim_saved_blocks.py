"""Fixed blocks-only control for rolling rim savings; prior-season mapper fits."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from build_predictive_rim_saved import SOURCE, OUT
from build_deflections_next_game import compare


def main():
    full=pd.read_parquet(SOURCE).sort_values(['player_id','game_date'])
    warm=full.season.eq(2019)
    mean=full.loc[warm,'blk'].sum()/full.loc[warm,'def_poss'].sum()
    records=[]
    for pid,g in full.groupby('player_id',sort=False):
        count=poss=0.;last=g.game_date.iloc[0]
        for r in g.itertuples():
            decay=2**(-(r.game_date-last).days/365)
            count*=decay;poss*=decay
            records.append((r.game_id,pid,100*(count+1000*mean)/(poss+1000)))
            count+=r.blk;poss+=r.def_poss;last=r.game_date
    features=pd.DataFrame(records,columns=['game_id','player_id','pregame_blocks100'])
    d=pd.read_parquet(OUT/'dated_predictions.parquet').merge(features,on=['game_id','player_id'],validate='one_to_one')
    d['blocks_only']=np.nan;d['blocks_plus_rim']=np.nan
    for y in [2023,2024,2025]:
        train=d.season.lt(y);test=d.season.eq(y)
        for name,cols in [('blocks_only',['pregame_blocks100']),('blocks_plus_rim',['pregame_blocks100','pred_saved100'])]:
            model=Ridge(alpha=1000).fit(d.loc[train,cols],d.loc[train,'target100'],sample_weight=d.loc[train,'def_poss'])
            d.loc[test,name]=model.predict(d.loc[test,cols])
    t=d[d.season.isin([2024,2025])].assign(exposure=lambda x:x.def_poss)
    report={'rim_vs_blocks':compare(t,'pred_saved100','blocks_only',draws=1000), 'joint_vs_blocks':compare(t,'blocks_plus_rim','blocks_only',draws=1000), 'status':'fixed diagnostic; no retuning or model swap'}
    (OUT/'blocks_control.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

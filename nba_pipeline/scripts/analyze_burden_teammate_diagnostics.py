import sys
sys.path.insert(0,'nba_pipeline/scripts')
import build_burden_adjusted_teammate_sq as m
import pandas as pd,numpy as np,json
from sklearn.model_selection import KFold
p=m.OUT;d=pd.read_csv(p/'all_results.csv');d=d[d.window.eq('2022-2026')].reset_index(drop=True);rows=[]
full=m.FEATURES.copy()
for label,cols in [('share_only',['role_shot_share']),('share_talent',['role_shot_share','teammate_talent_proxy']),('share_difficulty',['role_shot_share','own_difficulty_pp','share_x_difficulty']),('full',full)]:
 m.FEATURES=cols;r,s=m.adjust(d);rows.append({'model':label,**s})
pd.DataFrame(rows).to_csv(p/'ablation.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
m.FEATURES=full
r=d[d.burden_eligible].copy();X=r[full].to_numpy();y=r.teammate_sq_conditional.to_numpy();w=r.teammate_exposure.to_numpy();sensitivity=[]
for alpha in [1,10,100,1000]:
 pred=np.zeros(len(r))
 for tr,va in KFold(5,shuffle=True,random_state=924).split(X):pred[va]=m.predict(m.fit(X[tr],y[tr],w[tr],alpha),X[va])
 sensitivity.append({'alpha':alpha,'rank_correlation_with_selected':m.corr(y-pred,r.teammate_surplus_conditional),'rmse':float(np.sqrt(np.average((y-pred)**2,weights=w)))})
pd.DataFrame(sensitivity).to_csv(p/'penalty_sensitivity.csv',index=False);print(sensitivity)

"""Single-parent oTS control on the exact six-part shooting substrate."""
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import build_v2_six_shooting_split as model


def fit_matched(df):
 X,ids=model.design(df);n=len(ids)
 O=X[:,:n]+X[:,n:2*n];D=X[:,2*n:]
 A=hstack([O,D],format='csr')
 # Identical complete-season expected-FT target and season centering as the board.
 Y=df[model.TARGETS].to_numpy()
 y=(Y-model.baseline(df,Y,np.arange(len(df)))).sum(axis=1)
 G=(A.T@A).toarray();G.flat[::len(G)+1]+=np.repeat([3000.,5000.],n)
 from scipy.linalg import cho_factor,cho_solve
 b=cho_solve(cho_factor(G,lower=True,check_finite=False),A.T@y,check_finite=False)
 exposures=np.asarray(O.sum(axis=0)).ravel()
 center=np.average(b[:n],weights=exposures)
 result=pd.DataFrame({'player_id':ids,'normal_ots':100*(b[:n]-center),'normal_ots_opportunities':exposures.astype(int)})
 components=cho_solve(cho_factor(G,lower=True,check_finite=False),A.T@(Y-model.baseline(df,Y,np.arange(len(df)))),check_finite=False)[:n]
 components-=np.average(components,weights=exposures,axis=0)
 for i,target in enumerate(model.TARGETS):result['normal_ots_'+target]=100*components[:,i]
 assert np.allclose(result[['normal_ots_'+t for t in model.TARGETS]].sum(axis=1),result.normal_ots,atol=1e-10)
 return result


def main():
 df=pd.read_parquet(model.OUT/'opportunities.parquet');boards=[]
 for year in [0,2022,2023,2024,2025,2026]:
  sub=df if not year else df[df.Season.eq(year)].reset_index(drop=True)
  r=fit_matched(sub);r['window']='2022-2026' if not year else str(year)
  r['normal_ots_source']='matched_single_parent_o3000_d5000';boards.append(r)
 pd.concat(boards).to_csv(model.OUT/'matched_normal_ots.csv',index=False)
 (model.OUT/'matched_normal_ots_manifest.json').write_text(json.dumps({'target':'SQ + make + FT premium = actual FG + expected FT','sample':'identical opportunities.parquet','centering':'same training-season means; offensive exposure-weighted display centering','off_alpha':3000,'def_alpha':5000,'units':'points per 100 on-court team shooting opportunities','purpose':'matched descriptive single-parent comparison; not the independently tuned actual-point chronological control'},indent=2))
if __name__=='__main__':main()

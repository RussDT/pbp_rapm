"""Parent-anchored own/teammate turnover decomposition, RS 2022-2026.

Five focal rows per possession: own target=-I(focal committed turnover).
Joint mean/contrast parameterization gives own=(parent+4*contrast)/5,
teammate=4*(parent-contrast)/5. No outcome-conditioned opportunity denominator.
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,diags,hstack
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import GroupKFold
import build_v2_six_shooting_split as base
ROOT=base.ROOT;OUT=ROOT/'results/own_teammate_tov_2022_2026_rs'


def prepare():
 parts=[];audits=[]
 for year in range(2022,2027):
  d=base.unique(base.norm(pd.read_parquet(ROOT/f'processed/TOV{year%100}.parquet')))
  raw=base.unique(base.norm(pd.read_parquet(ROOT/f'raw_data/NBA{year%100}.parquet')))
  events=raw.loc[raw.event_type.eq(5),base.KEY+['player1_id']].rename(columns={'player1_id':'actor'})
  d=d.merge(events,on=base.KEY,how='left',validate='one_to_one')
  line=d[base.OFF+base.DEF].to_numpy(dtype=float);valid=np.isfinite(line).all(axis=1)&(line>0).all(axis=1)&(np.diff(np.sort(line,axis=1),axis=1)>0).all(axis=1)
  is_tov=d.Is_Turnover.eq(1).to_numpy();resolved=(line[:,:5]==d.actor.to_numpy()[:,None]).sum(axis=1)==1
  keep=valid&(~is_tov|resolved)
  audits.append({'season':year,'input_possessions':len(d),'input_tov':int(is_tov.sum()),'invalid_lineup':int((~valid).sum()),'unresolved_tov_valid_lineup':int((valid&is_tov&~resolved).sum()),'retained_possessions':int(keep.sum()),'retained_tov':int((keep&is_tov).sum())})
  d=d[keep].copy();d['Season']=year;d['actor']=d.actor.fillna(0).astype(int);parts.append(d[base.KEY+['Season',*base.OFF,*base.DEF,'actor','Is_Turnover']])
 pd.DataFrame(audits).to_csv(OUT/'coverage.csv',index=False);d=pd.concat(parts,ignore_index=True);d.to_parquet(OUT/'possessions.parquet',index=False);return d


def design(d):
 ids=np.unique(d[base.OFF+base.DEF].to_numpy(dtype=int));n=len(ids);rr=np.repeat(np.arange(len(d)),5)
 O=coo_matrix((np.ones(len(rr)),(rr,np.searchsorted(ids,d[base.OFF].to_numpy(dtype=int).ravel()))),shape=(len(d),n)).tocsr()
 D=coo_matrix((np.ones(len(rr)),(rr,np.searchsorted(ids,d[base.DEF].to_numpy(dtype=int).ravel()))),shape=(len(d),n)).tocsr()
 return O,D,ids


def stats(d,O,ids,w=None):
 w=np.ones(len(d)) if w is None else w;t=d.Is_Turnover.to_numpy(dtype=float);n=len(ids)
 exposure=np.asarray(O.T@w).ravel();G=np.diag(exposure)-(O.T@O.multiply(w[:,None])).toarray()/5
 take=t==1;own=np.bincount(np.searchsorted(ids,d.actor.to_numpy()[take]),weights=w[take],minlength=n)
 B=-own+np.asarray(O.T@(w*t)).ravel()/5
 return G,B,float(.8*np.sum(w*t))


def solve(G,B,alpha):
 A=G.copy();A.flat[::len(A)+1]+=alpha
 return cho_solve(cho_factor(A,lower=True,check_finite=False),B,check_finite=False)


def select(d,O,ids):
 grid=[1.,10.,100.,300.,1000.,3000.,10000.,30000.,100000.,1e12];loss=[]
 for tr,va in GroupKFold(3).split(d,groups=d.game_id):
  G,B,_=stats(d.iloc[tr],O[tr],ids);v=d.iloc[va];counts=v.groupby('game_id').game_id.transform('size').to_numpy();w=1/counts;w/=w.sum()
  V,b,c=stats(v,O[va],ids,w)
  loss.append([float(c-2*(beta:=solve(G,B,a))@b+beta@V@beta) for a in grid])
 loss=np.array(loss);means=loss.mean(axis=0);best=means.argmin();delta=loss-loss[:,[best]];se=delta.std(axis=0,ddof=1)/np.sqrt(3);eligible=means-means[best]<=se+1e-12;chosen=max(a for a,ok in zip(grid,eligible) if ok)
 pd.DataFrame({'alpha':grid,'mse':means,'paired_se':se,'selected':np.array(grid)==chosen}).to_csv(OUT/'role_alpha_cv.csv',index=False)
 return chosen,{'selected_alpha':chosen,'minimum_alpha':grid[best],'criterion':'three grouped-game folds, equal game weight, five focal own-turnover squared errors; largest alpha within paired one-SE','parent_off_alpha':3000,'parent_def_alpha':5000}


def board(d,alpha):
 O,D,ids=design(d);n=len(ids);A=hstack([O,D],format='csr');y=-d.Is_Turnover.to_numpy(dtype=float);means=pd.Series(y).groupby(d.Season.to_numpy()).transform('mean').to_numpy();G=(A.T@A).toarray();G.flat[::len(G)+1]+=np.repeat([3000.,5000.],n)
 parent=cho_solve(cho_factor(G,lower=True,check_finite=False),A.T@(y-means),check_finite=False)[:n]
 C,B,_=stats(d,O,ids);contrast=solve(C,B,alpha);exp=np.asarray(O.sum(axis=0)).ravel()
 parent-=np.average(parent,weights=exp);contrast-=np.average(contrast,weights=exp)
 own=(parent+4*contrast)/5;team=4*(parent-contrast)/5
 raw=np.bincount(np.searchsorted(ids,d.loc[d.Is_Turnover.eq(1),'actor']),minlength=n)
 return pd.DataFrame({'player_id':ids,'own_tov_impact':own*100,'teammate_tov_suppression':team*100,'total_otov':parent*100,'tov_role_contrast':contrast*100,'tov_possessions':exp.astype(int),'own_turnovers':raw,'own_turnovers_per100':np.divide(raw*100,exp,out=np.full(n,np.nan),where=exp>0)})


def main():
 OUT.mkdir(parents=True,exist_ok=True);d=prepare();O,D,ids=design(d);alpha,meta=select(d,O,ids);results=[]
 for year in [0,2022,2023,2024,2025,2026]:
  sub=d if not year else d[d.Season.eq(year)].reset_index(drop=True);r=board(sub,alpha);r['window']='2022-2026' if not year else str(year);results.append(r)
 pd.concat(results).to_csv(OUT/'all_results.csv',index=False);(OUT/'selection.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
if __name__=='__main__':main()

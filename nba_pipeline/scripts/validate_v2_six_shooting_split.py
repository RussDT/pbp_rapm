"""Nested chronological diagnostic for six-part shooting; never touches 2027."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import GroupKFold
import build_v2_six_shooting_split as model
ROOT=model.ROOT
OUT=ROOT/'results/v2_six_shooting_premium_validation'

def fit(G,B,penalty):
 A=G.copy(order='F');A.flat[::len(A)+1]+=penalty
 return cho_solve(cho_factor(A,lower=True,overwrite_a=True,check_finite=False),B,check_finite=False)

def usage(X,n):
 own=np.asarray(X[:,:n].sum(axis=0)).ravel();exposure=own+np.asarray(X[:,n:2*n].sum(axis=0)).ravel()
 return np.divide(own,exposure,out=np.full(n,.2),where=exposure>0)

def control_design(X,n,u,role):
 S=X[:,:n];O=S+X[:,n:2*n];D=X[:,2*n:]
 return hstack([O,D,S-O.multiply(u)],format='csr') if role else hstack([O,D],format='csr')

def tune_control(df,X,n,role):
 grid=[30,300,3000,30000,1e12] if role else [100,300,1000,3000,10000,30000]
 Y=np.column_stack([df.actual.to_numpy(),np.zeros((len(df),2))]);losses=np.zeros((3,len(grid)))
 for f,(tr,va) in enumerate(GroupKFold(3).split(df,groups=df.game_id)):
  u=usage(X[tr],n);M=control_design(X,n,u,role);A=M[tr];V=M[va]
  yc=(Y-model.baseline(df,Y,tr)).sum(axis=1);G=(A.T@A).toarray();B=A.T@yc[tr]
  for k,a in enumerate(grid):
   pen=np.repeat([3000,5000,a] if role else [a,5000],n)
   pred=V@fit(G,B,pen)
   errors=pd.DataFrame({'g':df.game_id.to_numpy()[va],'e':(yc[va]-pred)**2}).groupby('g').e.mean()
   losses[f,k]=errors.mean()
 mean=losses.mean(axis=0);best=int(mean.argmin());delta=losses-losses[:,[best]]
 se=delta.std(axis=0,ddof=1)/np.sqrt(3);eligible=(mean-mean[best])<=se+1e-12
 return max(a for a,e in zip(grid,eligible) if e)

def main():
 OUT.mkdir(exist_ok=True,parents=True)
 df=pd.read_parquet(model.OUT/'opportunities.parquet');X,ids=model.design(df);n=len(ids)
 allgames=[];choices=[]
 for year in range(2023,2027):
  tr=np.flatnonzero(df.Season.to_numpy()<year);va=np.flatnonzero(df.Season.to_numpy()==year)
  train=df.iloc[tr].reset_index(drop=True);test=df.iloc[va].reset_index(drop=True);Xt=X[tr];Xv=X[va]
  model.OUT=OUT/f'train_through_{year-1}';model.OUT.mkdir(exist_ok=True)
  print('CHRONOLOGICAL TEST',year,flush=True)
  selection=model.cv(train,Xt,ids,[300.,1000.,3000.,10000.,30000.,100000.,1000000.,1e12],[1.,10.,30.,100.,300.,1000.,10000.,100000.])
  Y=train[model.TARGETS].to_numpy();center=Y-model.baseline(train,Y,np.arange(len(train)))
  # Forecast the new season from the last completed season's mean, never test outcomes.
  reference=Y[train.Season.eq(year-1)].mean(axis=0).sum()
  G=(Xt.T@Xt).toarray();B=Xt.T@center;u=usage(Xt,n)
  beta=sum(model.solve_gram(G,B[:,i],n,selection['selected_pairs'][name]) for i,name in enumerate(model.TARGETS))
  predictions={'six_known_shooter':reference+Xv@beta}
  O=Xv[:,:n]+Xv[:,n:2*n];D=Xv[:,2*n:]
  predictions['six_lineup_only']=reference+O@(u*beta[:n]+(1-u)*beta[n:2*n])+D@beta[2*n:]
  actual_Y=np.column_stack([train.actual.to_numpy(),np.zeros((len(train),2))])
  actual_center=(actual_Y-model.baseline(train,actual_Y,np.arange(len(train)))).sum(axis=1)
  actual_reference=train.loc[train.Season.eq(year-1),'actual'].mean()
  choice={'test_season':year,'training_last_season':year-1,'six_alphas':selection['selected_pairs']}
  for role in [False,True]:
   a=tune_control(train,Xt,n,role);M=control_design(Xt,n,u,role);V=control_design(Xv,n,u,role)
   pen=np.repeat([3000,5000,a] if role else [a,5000],n)
   b=fit((M.T@M).toarray(),M.T@actual_center,pen)
   if role:
    predictions['centered_known_shooter']=actual_reference+V@b
    predictions['centered_lineup_only']=actual_reference+O@b[:n]+D@b[n:2*n]
    choice['centered_role_alpha']=a
   else:
    predictions['plain_lineup_rapm']=actual_reference+V@b;choice['plain_off_alpha']=a
  errors=pd.DataFrame({'game_id':test.game_id,'test_season':year})
  for name,pred in predictions.items():errors[name]=(test.actual.to_numpy()-pred)**2
  bygame=errors.groupby(['test_season','game_id']).mean().reset_index();allgames.append(bygame);choices.append(choice)
  pd.DataFrame(choices).to_json(OUT/'choices.json',orient='records',indent=2)
  print(choice,flush=True);print(bygame.drop(columns=['game_id','test_season']).mean().to_dict(),flush=True)
 games=pd.concat(allgames,ignore_index=True);games.to_csv(OUT/'game_errors.csv',index=False)
 names=[c for c in games.columns if c not in ['game_id','test_season']]
 games.groupby('test_season')[names].mean().to_csv(OUT/'season_mse.csv')
 contrasts=[('six_known_shooter','centered_known_shooter'),('six_lineup_only','plain_lineup_rapm'),('six_lineup_only','centered_lineup_only'),('centered_lineup_only','plain_lineup_rapm')]
 rng=np.random.default_rng(20260918);rows=[]
 for a,b in contrasts:
  parts=[g[a].to_numpy()-g[b].to_numpy() for _,g in games.groupby('test_season')]
  samples=np.mean([np.mean(p[rng.integers(0,len(p),size=(2000,len(p)))],axis=1) for p in parts],axis=0)
  rows.append({'challenger':a,'reference':b,'mse_delta':float(np.mean([p.mean() for p in parts])),'ci_low':float(np.quantile(samples,.025)),'ci_high':float(np.quantile(samples,.975)),'seasons_won':sum(p.mean()<0 for p in parts),'test_seasons':len(parts)})
 result=pd.DataFrame(rows);result.to_csv(OUT/'paired_comparisons.csv',index=False);print(result.to_string(index=False),flush=True)
if __name__=='__main__':main()

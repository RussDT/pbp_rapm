"""Descriptive, cross-fitted burden baseline; never causal playmaking credit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import build_v2_six_shooting_split as model
ROOT=model.ROOT;OUT=ROOT/'results/burden_teammate_sq_2022_2026_rs'
FEATURES=['role_shot_share','own_difficulty_pp','share_x_difficulty','teammate_talent_proxy']
ALPHAS=[1.,10.,100.,1000.]


def prepare_panel():
 p=pd.read_parquet(ROOT/'results/td_overall_playtype_mix/attempt_panel_2019_2026.parquet')
 p['game_id']=pd.to_numeric(p.game_id);p['player_id']=p.Focal_Player.astype(int)
 # Prior-season bucket-adjusted accuracy excess, averaged over a fixed league mix.
 # This is a coarse talent proxy, not a purified talent estimate or current teammate shot mix.
 parts=[]
 for year in range(2022,2027):
  prior=p[p.Season.eq(year-1)];now=p[p.Season.eq(year)].copy()
  league=prior.groupby('mix_category').actual_points.agg(['mean','count'])
  z=prior.groupby(['player_id','mix_category']).actual_points.agg(['sum','count'])
  z['excess']=(z['sum']-z['count']*z.index.get_level_values(1).map(league['mean']))/(z['count']+100.)
  z['weighted']=z.excess*z.index.get_level_values(1).map(league['count']/league['count'].sum())
  talent=z.groupby(level=0).weighted.sum()*100
  now['talent']=now.player_id.map(talent).fillna(0.)
  now['talent_known']=now.player_id.isin(talent.index)
  parts.append(now)
 return pd.concat(parts,ignore_index=True)


def features(df,p):
 pieces=[]
 for year,d in df.groupby('Season'):
  a=p[p.Season.eq(year)&p.game_id.isin(d.game_id.unique())]
  ids=np.unique(d[model.OFF+model.DEF].to_numpy(dtype=int));n=len(ids)
  own=np.bincount(np.searchsorted(ids,d.shooter),minlength=n)
  exposure=np.bincount(np.searchsorted(ids,d[model.OFF].to_numpy(dtype=int).ravel()),minlength=n)
  u=np.divide(own,exposure,out=np.zeros(n),where=exposure>0)
  ownrole=a.groupby('player_id').lag_difficulty_pp.agg(['mean','count'])
  # Role cache has four non-shooters in O1..O4; aggregate recipient prior talent.
  rec=[]
  for col in ['O1','O2','O3','O4']:
   rec.append(pd.DataFrame({'player_id':a[col].to_numpy(dtype=int),'talent':a.talent.to_numpy(),'known':a.talent_known.to_numpy()}))
  tm=pd.concat(rec).groupby('player_id').agg(talent=('talent','mean'),known=('known','mean'))
  r=pd.DataFrame({'player_id':ids,'exposure':exposure,'teammate_exposure':exposure-own,'role_shot_share':u,'Season':year})
  r['own_difficulty_pp']=r.player_id.map(ownrole['mean']);r['own_role_fga']=r.player_id.map(ownrole['count']).fillna(0)
  r['teammate_talent_proxy']=r.player_id.map(tm.talent);r['talent_known_share']=r.player_id.map(tm.known)
  r['share_x_difficulty']=(r.role_shot_share-.2)*r.own_difficulty_pp
  pieces.append(r)
 annual=pd.concat(pieces)
 # Retain season-specific role interactions before opportunity-weighted pooling.
 cols=FEATURES+['talent_known_share']
 for c in cols:annual[c]=annual[c]*annual.teammate_exposure
 grouped=annual.groupby('player_id');r=grouped[['exposure','teammate_exposure','own_role_fga',*cols]].sum(min_count=1)
 for c in cols:r[c]=r[c]/r.teammate_exposure.replace(0,np.nan)
 return r.reset_index()


def sq_board(df,penalty_scale=1.):
 X,ids=model.design(df);n=len(ids);Y=df[model.TARGETS].to_numpy();base=model.baseline(df,Y,np.arange(len(df)))
 b=model.solve_gram((X.T@X).toarray(),X.T@(Y[:,0]-base[:,0]),n,(30*penalty_scale,1000*penalty_scale),5000*penalty_scale)
 own=np.bincount(np.searchsorted(ids,df.shooter),minlength=n);exp=np.bincount(np.searchsorted(ids,df[model.OFF].to_numpy(dtype=int).ravel()),minlength=n)
 u=np.divide(own,exp,out=np.zeros(n),where=exp>0);center=np.average(u*b[:n]+(1-u)*b[n:2*n],weights=exp)
 return pd.DataFrame({'player_id':ids,'teammate_sq_conditional':100*(b[n:2*n]-center),'shot_share':u})


def fit(X,y,w,alpha):
 scaler=StandardScaler().fit(X,sample_weight=w)
 reg=Ridge(alpha=alpha).fit(scaler.transform(X),y,sample_weight=w/w.mean())
 return scaler,reg


def predict(fitted,X):return fitted[1].predict(fitted[0].transform(X))


def select(X,y,w):
 folds=list(KFold(3,shuffle=True,random_state=193).split(X));loss=[]
 for alpha in ALPHAS:
  loss.append([np.average((y[va]-predict(fit(X[tr],y[tr],w[tr],alpha),X[va]))**2,weights=w[va]) for tr,va in folds])
 loss=np.array(loss);best=loss.mean(axis=1).argmin();delta=loss-loss[best]
 eligible=delta.mean(axis=1)<=delta.std(axis=1,ddof=1)/np.sqrt(3)+1e-12
 return max(a for a,ok in zip(ALPHAS,eligible) if ok)


def adjust(frame):
 r=frame.copy();ok=(r.teammate_exposure>=500)&(r.own_role_fga>=100)&r[FEATURES].notna().all(axis=1)
 r['burden_eligible']=ok;r['burden_baseline_conditional']=np.nan
 d=r[ok];X=d[FEATURES].to_numpy();y=d.teammate_sq_conditional.to_numpy();w=d.teammate_exposure.to_numpy();pred=np.zeros(len(d));alphas=[]
 # Entire player is held out within this window; target is never used in its own baseline fit.
 for tr,va in KFold(5,shuffle=True,random_state=924).split(X):
  alpha=select(X[tr],y[tr],w[tr]);alphas.append(alpha);pred[va]=predict(fit(X[tr],y[tr],w[tr],alpha),X[va])
 r.loc[ok,'burden_baseline_conditional']=pred
 r['teammate_surplus_conditional']=r.teammate_sq_conditional-r.burden_baseline_conditional
 r['burden_baseline']=(1-r.shot_share)*r.burden_baseline_conditional
 r['teammate_surplus']=(1-r.shot_share)*r.teammate_surplus_conditional
 rmse=float(np.sqrt(np.average((y-pred)**2,weights=w)));null=float(np.sqrt(np.average((y-np.average(y,weights=w))**2,weights=w)))
 return r,{'eligible':int(ok.sum()),'alphas':alphas,'player_heldout_rmse':rmse,'mean_only_rmse':null,'weighted_r2':1-rmse**2/null**2}


def corr(a,b):return float(spearmanr(a,b).statistic)


def main():
 OUT.mkdir(parents=True,exist_ok=True);df=pd.read_parquet(model.OUT/'opportunities.parquet');p=prepare_panel()
 original=pd.read_csv(model.OUT/'all_results.csv',dtype={'window':str});boards=[];audit={}
 for year in [0,2022,2023,2024,2025,2026]:
  window='2022-2026' if not year else str(year);d=df if not year else df[df.Season.eq(year)].reset_index(drop=True)
  parent=original[original.window.eq(window)][['player_id','player_name','teammate_sq_conditional','shot_share']]
  board,stats=adjust(parent.merge(features(d,p),on='player_id',validate='one_to_one'));board['window']=window;boards.append(board);audit[window]=stats;print(window,stats,flush=True)
 result=pd.concat(boards);result.to_csv(OUT/'all_results.csv',index=False)
 stability=[];halfboards=[]
 for seed in [41,73,109]:
  rng=np.random.default_rng(seed);left=[]
  for _,d in df.groupby('Season'):
   games=np.sort(d.game_id.unique());rng.shuffle(games);left.extend(games[:len(games)//2])
  halves=[]
  for label,mask in [('A',df.game_id.isin(left)),('B',~df.game_id.isin(left))]:
   d=df[mask].reset_index(drop=True);r,_=adjust(sq_board(d,.5).merge(features(d,p),on='player_id'));r['seed']=seed;r['half']=label;halfboards.append(r);halves.append(r[r.burden_eligible])
  m=halves[0].merge(halves[1],on='player_id',suffixes=('_a','_b'));raw=corr(m.teammate_sq_conditional_a,m.teammate_sq_conditional_b);res=corr(m.teammate_surplus_conditional_a,m.teammate_surplus_conditional_b)
  stability.append({'seed':seed,'players':len(m),'raw_teammate_spearman':raw,'surplus_spearman':res,'surplus_sign_agreement':float((np.sign(m.teammate_surplus_conditional_a)==np.sign(m.teammate_surplus_conditional_b)).mean())});print(stability[-1],flush=True)
 pd.concat(halfboards).to_csv(OUT/'split_game_boards.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'stability.csv',index=False)
 # Independent creation measures are descriptive checks, never fitted baseline features.
 creation=pd.read_csv(ROOT/'results/pbp_playtype_difficulty/2023_2026/pbp_playtype_future_ots_analysis.csv')
 creation['window']=creation.feature_season.astype(str);m=result.merge(creation[['player_id','window','potential_assists_per100','rim_assists_per100']],on=['player_id','window']).dropna(subset=['teammate_surplus_conditional','potential_assists_per100','rim_assists_per100'])
 checks=[]
 for season,g in m.groupby('window'):
  for metric in ['potential_assists_per100','rim_assists_per100']:
   checks.append({'season':season,'measure':metric,'players':len(g),'raw_spearman':corr(g.teammate_sq_conditional,g[metric]),'surplus_spearman':corr(g.teammate_surplus_conditional,g[metric])})
 pd.DataFrame(checks).to_csv(OUT/'creation_checks.csv',index=False)
 audit['method']={'features':FEATURES,'difficulty':'prior-season priced seven-family PBP proxy; NOT production zTS difficulty','talent':'prior-season bucket-adjusted FG accuracy excess, 100-FGA shrinkage per bucket, fixed prior league mix; missing players neutral','crossfit':'five held-out player folds; nested three-fold alpha selection','stability':'three season-stratified random game half splits; half-sized SQ ridge penalties to preserve penalty/exposure ratio','limits':'descriptive residual, not causal attribution; retrospective transition labels and fixed lag-price/skill inputs shared across half splits; not future confirmation; no 2027'}
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2));print('DONE',OUT,flush=True)
if __name__=='__main__':main()

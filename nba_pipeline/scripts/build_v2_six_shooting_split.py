#!/usr/bin/env python3
"""Five-season RS shooting decomposition; joint six-offense-alpha game CV.

Three targets share one opportunity sample: v2 mid/three EV + actual rim,
mid/three actual-minus-v2, and DECOMP expected FT premium. Each uses shooter, nonshooter,
and defensive player blocks. Penalties for the six offensive blocks are
selected jointly against their summed actual-point prediction.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import GroupKFold
try:
 from .run_assist_points_hierarchy import build_player_name_map
 from .process_rapm_blocks.common import RIM_ACTION_TYPES, fetch_player_stats_supabase
except ImportError:
 from run_assist_points_hierarchy import build_player_name_map
 from process_rapm_blocks.common import RIM_ACTION_TYPES, fetch_player_stats_supabase
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v2_six_shooting_premium_2022_2026_rs'
TARGETS=['sq','make','ft']
KEY=['game_id','event_num']
OFF=[f'O{i}' for i in range(1,6)];DEF=[f'D{i}' for i in range(1,6)]

def norm(df):
 df=df.copy()
 for c in KEY:df[c]=pd.to_numeric(df[c],errors='raise').astype('int64')
 return df

def unique(df):
 df=df.drop_duplicates()
 return df.loc[~df.duplicated(KEY,keep=False)].copy()

def ft_baseline_units(text):
 """Canonical DECOMP replacement-shot allocation per FT event."""
 zero=text.str.contains(r'Technical|\b1 of 1\b',case=False,regex=True)
 return np.where(zero,0.,np.where(text.str.contains(r'\b[123] of 3\b'),1/3,np.where(text.str.contains(r'\b[12] of 2\b'),.5,0.)))

def training_targets(df,train):
 """Re-price FT targets using training-only shooter-season rates and FGA value."""
 if 'ft_baseline_units' not in df:return df[TARGETS].to_numpy()
 t=df.iloc[train]
 rates=t.groupby(['Season','shooter'])[['ft_actual','fta']].sum()
 rates['rate']=rates.ft_actual/rates.fta.replace(0,np.nan)
 rate=pd.MultiIndex.from_frame(df[['Season','shooter']]).map(rates.rate).to_numpy(dtype=float)
 expected=np.where(np.isfinite(rate),rate*df.fta.to_numpy(),df.ft_actual.to_numpy())
 fg=t.groupby('Season')[['fg_points','fga']].sum();price=fg.fg_points/fg.fga
 replacement=df.Season.map(price).to_numpy()*df.ft_baseline_units.to_numpy()
 return np.column_stack([df.sq.to_numpy()-df.ft_baseline.to_numpy()+replacement,df['make'].to_numpy(),expected-replacement])

def prepare():
 ev=unique(norm(pd.read_parquet(ROOT.parent.parent/'vpm/vpm/data/russell_sq_v2_ev_21_26.parquet')))
 ev=ev.rename(columns={'shooter_id':'ev_shooter','made':'ev_made','zone':'ev_zone'})
 allrows=[];audits=[]
 for year in range(2022,2027):
  raw=norm(pd.read_parquet(ROOT/f'raw_data/NBA{year%100}.parquet')).drop_duplicates()
  badkeys=raw.loc[raw.duplicated(KEY,keep=False),KEY].drop_duplicates()
  # Conflicting raw keys cannot be attached safely; retain their position as invalid actions.
  raw=raw.merge(badkeys.assign(conflict=True),on=KEY,how='left')
  raw['clock']=raw.time_quarter.str.split(':').apply(lambda x:60*int(x[0])+float(x[1]))
  raw=raw.sort_values(['game_id','period','clock','event_num'],ascending=[True,True,False,True]).reset_index(drop=True)
  ends=unique(norm(pd.read_parquet(ROOT/f'processed/TS{year%100}.parquet')))
  ends['episode']=np.arange(len(ends))
  raw=raw.merge(ends[KEY+['episode']],on=KEY,how='left',validate='many_to_one')
  raw['episode']=raw.groupby(['game_id','period']).episode.bfill()
  actions=raw[raw.event_type.isin([1,2,3])].copy()
  text=actions.home_description.fillna('')+' '+actions.visitor_description.fillna('')
  actions['is_fg']=actions.event_type.isin([1,2]);actions['is_ft']=actions.event_type.eq(3)
  actions['zone']=np.where(text.str.contains('3PT',case=False),'three',np.where(actions.event_action_type.isin(RIM_ACTION_TYPES)|text.str.contains(r'\blayup\b|\bdunk\b|\btip\b',case=False),'rim','mid'))
  actions['made']=np.where(actions.is_fg,actions.event_type.eq(1),~text.str.contains('MISS',case=False))
  actions['points']=actions.made*np.where(actions.is_ft,1,np.where(actions.zone.eq('three'),3,2))
  descriptions=actions.home_description.fillna('')+' '+actions.visitor_description.fillna('')
  rates_path=OUT/f'ft_rates_{year}.csv'
  rates=pd.read_csv(rates_path) if rates_path.exists() else fetch_player_stats_supabase(year,False)
  if rates is None or rates.empty:raise RuntimeError(f'Missing canonical FT rates for {year}')
  rates.to_csv(rates_path,index=False)
  pct=pd.to_numeric(rates.set_index('PlayerID').FTPerc,errors='coerce')
  assert pct.dropna().between(0,1).all()
  expected=actions.player1_id.map(pct).fillna(actions.points)
  actions['ft_actual']=np.where(actions.is_ft,actions.points,0.)
  actions['fg_points']=np.where(actions.is_fg,actions.points,0.)
  actions['ft_expected']=np.where(actions.is_ft,expected,0.)
  actions['ft_baseline_units']=np.where(actions.is_ft,ft_baseline_units(descriptions),0.)
  actions['ft_baseline']=actions.ft_baseline_units*actions.loc[actions.is_fg,'points'].mean()
  actions=actions.merge(ev[KEY+['ev_shooter','ev_made','ev_zone','ev_pts']],on=KEY,how='left',validate='many_to_one')
  mid3=actions.is_fg & actions.zone.ne('rim')
  match=actions.player1_id.eq(actions.ev_shooter)&actions.made.eq(actions.ev_made)&actions.zone.eq(actions.ev_zone)&actions.ev_pts.notna()
  actions['bad']=actions.conflict.fillna(False)|(mid3 & ~match)|actions.player1_id.le(0)|actions.player1_id.isna()
  actions['sq']=np.where(mid3,actions.ev_pts,np.where(actions.is_fg,actions.points,0))+actions.ft_baseline
  actions['make']=np.where(mid3,actions.points-actions.ev_pts,0)
  actions['ft']=actions.ft_expected-actions.ft_baseline
  group=actions.dropna(subset=['episode']).groupby('episode')
  counts=group.agg(shooter=('player1_id','first'),shooters=('player1_id','nunique'),fga=('is_fg','sum'),fta=('is_ft','sum'),bad=('bad','max'),actual=('points','sum'),sq=('sq','sum'),make=('make','sum'),ft=('ft','sum'),ft_actual=('ft_actual','sum'),fg_points=('fg_points','sum'),ft_expected=('ft_expected','sum'),ft_baseline_units=('ft_baseline_units','sum'),ft_baseline=('ft_baseline','sum'))
  # Single-shooter episodes include FG+and-one and pure FT trips, but reject
  # mixed technical/foreign-side scoring and multiple-FGA parser segments.
  eligible=(counts.shooters==1)&(counts.fga<=1)&(~counts.bad)
  counts=counts[eligible]
  rows=ends.merge(counts,left_on='episode',right_index=True,validate='one_to_one')
  line=rows[OFF+DEF].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
  valid=np.isfinite(line).all(axis=1)&(line>0).all(axis=1)&(np.diff(np.sort(line,axis=1),axis=1)>0).all(axis=1)
  valid&=(line[:,:5]==rows.shooter.to_numpy()[:,None]).sum(axis=1)==1
  rows=rows[valid].copy(); rows['Season']=year
  rows['expected_total']=rows.fg_points+rows.ft_expected
  assert np.allclose(rows.sq+rows['make']+rows.ft,rows.expected_total,atol=1e-12)
  audit={'season':year,'raw_fga':int(actions.is_fg.sum()),'raw_fta':int(actions.is_ft.sum()),'raw_games':int(raw.game_id.nunique()),'retained_opportunities':len(rows),'retained_fga':int(rows.fga.sum()),'retained_fta':int(rows.fta.sum()),'retained_games':int(rows.game_id.nunique()),'missing_v2_mid_three':int((mid3&~match).sum()),'rejected_mixed_or_invalid_episodes':int((~eligible).sum()),'invalid_lineup_episodes':int((~valid).sum()),'row_closure_max':float(np.max(abs(rows.sq+rows['make']+rows.ft-rows.expected_total)))}
  audits.append(audit);allrows.append(rows[['game_id','event_num','Season','shooter',*OFF,*DEF,'fga','fta','sq','make','ft','actual','ft_actual','fg_points','ft_expected','ft_baseline_units','ft_baseline','expected_total']])
  print(audit,flush=True)
 df=pd.concat(allrows,ignore_index=True);df.to_parquet(OUT/'opportunities.parquet',index=False);pd.DataFrame(audits).to_csv(OUT/'coverage.csv',index=False)
 return df

def design(df):
 ids=np.unique(df[OFF+DEF].to_numpy(dtype=np.int64));n=len(ids);nr=len(df)
 shooter=df.shooter.to_numpy(dtype=np.int64)
 o=df[OFF].to_numpy(dtype=np.int64);d=df[DEF].to_numpy(dtype=np.int64)
 selfcol=np.searchsorted(ids,shooter)
 tmcols=n+np.searchsorted(ids,o);dv=2*n+np.searchsorted(ids,d)
 mask=o!=shooter[:,None]
 rr=np.repeat(np.arange(nr),5)
 X=coo_matrix((np.ones(nr*10),(np.concatenate([np.arange(nr),rr[mask.ravel()],rr]),np.concatenate([selfcol,tmcols.ravel()[mask.ravel()],dv.ravel()]))),shape=(nr,3*n)).tocsr()
 assert np.all(X.sum(axis=1)==10)
 return X,ids

def baseline(df,Y,train):
 means=pd.DataFrame(Y[train],columns=TARGETS).groupby(df.Season.to_numpy()[train]).mean()
 return means.loc[df.Season].to_numpy()

def solve_gram(G,B,n,pair,def_alpha=5000):
 penalty=np.repeat([pair[0],pair[1],def_alpha],n)
 A=G.copy(order='F');A.flat[::A.shape[0]+1]+=penalty
 return cho_solve(cho_factor(A,lower=True,overwrite_a=True,check_finite=False),B,check_finite=False)

def cv(df,X,ids,grid,own_grid=None):
 n=len(ids);Y=df[TARGETS].to_numpy();pairs=[(a,b) for a in (own_grid or grid) for b in grid];C=len(pairs)
 folds=list(GroupKFold(3).split(df,groups=df.game_id))
 fold_losses=[];fold_component=[]
 for f,(tr,va) in enumerate(folds):
  print('FOLD',f+1,'train',len(tr),'validation',len(va),'features',3*n,flush=True)
  fold_Y=training_targets(df,tr);base=baseline(df,fold_Y,tr);yc=fold_Y-base;train=X[tr];valid=X[va]
  G=(train.T@train).toarray();B=train.T@yc[tr]
  games=df.game_id.to_numpy()[va];_,gi,counts=np.unique(games,return_inverse=True,return_counts=True)
  weights=1/counts[gi];weights=weights/weights.sum();root=np.sqrt(weights)
  P=np.lib.format.open_memmap(OUT/f'cv_predictions_fold{f}.npy',mode='w+',dtype='float32',shape=(3,C,len(va)))
  component=[]
  for k,pair in enumerate(pairs):
   start=time.time();beta=solve_gram(G,B,n,pair);pred=valid@beta
   P[:,k,:]=pred.T
   component.append(np.sum(weights[:,None]*(yc[va]-pred)**2,axis=0))
   if k%10==0:print(' pair',k+1,'/',C,pair,'seconds',round(time.time()-start,2),flush=True)
  P.flush()
  actual=df.actual.to_numpy()[va]-base[va].sum(axis=1)
  # Quadratic sufficient statistics enumerate all C^3 six-alpha combinations.
  Z=[np.asarray(P[t],dtype=float)*root[None,:] for t in range(3)]
  a=actual*root
  linear=[np.sum(z*z,axis=1)-2*z@a for z in Z]
  crosses=[Z[0]@Z[1].T,Z[0]@Z[2].T,Z[1]@Z[2].T]
  loss=np.dot(a,a)+linear[0][:,None,None]+linear[1][None,:,None]+linear[2][None,None,:]+2*(crosses[0][:,:,None]+crosses[1][:,None,:]+crosses[2][None,:,:])
  fold_losses.append(loss);fold_component.append(component)
  np.save(OUT/f'joint_loss_fold{f}.npy',loss)
  del Z,P,G,B,train,valid
 losses=np.asarray(fold_losses);means=losses.mean(axis=0);best=np.unravel_index(means.argmin(),means.shape)
 delta=losses-losses[(slice(None),*best)][:,None,None,None]
 se=delta.std(axis=0,ddof=1)/np.sqrt(3)
 eligible=(means-means[best])<=se+1e-12
 strength=np.log(np.array(pairs)).sum(axis=1)
 score=strength[:,None,None]+strength[None,:,None]+strength[None,None,:]
 maximum=np.max(score[eligible]);tied=eligible & (score>=maximum-1e-10)
 selected=np.unravel_index(np.where(tied,means,np.inf).argmin(),means.shape)
 audit=[]
 for t,name in enumerate(TARGETS):
  component=np.asarray(fold_component)[:,:,t]
  for k,pair in enumerate(pairs):
   audit.append({'target':name,'own_alpha':pair[0],'teammate_alpha':pair[1],'component_mse':component[:,k].mean(),'selected_joint':k==selected[t],'joint_minimum':k==best[t]})
 pd.DataFrame(audit).to_csv(OUT/'alpha_candidates.csv',index=False)
 report={'grid':grid,'own_grid':own_grid or grid,'candidates_per_target':C,'joint_combinations':C**3,'minimum_pairs':{t:pairs[best[i]] for i,t in enumerate(TARGETS)},'selected_pairs':{t:pairs[selected[i]] for i,t in enumerate(TARGETS)},'minimum_joint_mse':float(means[best]),'selected_joint_mse':float(means[selected]),'selected_paired_se':float(se[selected]),'eligible_count':int(eligible.sum()),'def_alpha':5000,'criterion':'Equal-game mean squared error of summed actual shooting points, three grouped-game folds; largest product of six alphas within paired one-SE of joint minimum; lower MSE breaks equal-strength ties.'}
 # Exact paired game-error summary for chosen versus minimum, diagnostic only.
 checks=[]
 for f,(_,va) in enumerate(folds):
  P=np.load(OUT/f'cv_predictions_fold{f}.npy',mmap_mode='r');base=baseline(df,training_targets(df,folds[f][0]),folds[f][0]);actual=df.actual.to_numpy()[va]-base[va].sum(axis=1)
  ps=sum(P[t,selected[t]].astype(float) for t in range(3));pb=sum(P[t,best[t]].astype(float) for t in range(3))
  e=pd.DataFrame({'game_id':df.game_id.to_numpy()[va],'selected_sqerr':(actual-ps)**2,'minimum_sqerr':(actual-pb)**2}).groupby('game_id').mean().reset_index();e['fold']=f;checks.append(e)
 game_errors=pd.concat(checks);game_errors.to_csv(OUT/'selected_game_errors.csv',index=False)
 delta_game=game_errors.selected_sqerr-game_errors.minimum_sqerr
 report['selected_minus_minimum_game_mse']=float(delta_game.mean())
 report['selected_minus_minimum_game_se']=float(delta_game.std()/np.sqrt(len(delta_game)))
 report['games']=len(delta_game)
 (OUT/'selection.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
 return report

def board(df,selection):
 X,ids=design(df);n=len(ids);Y=df[TARGETS].to_numpy();base=baseline(df,Y,np.arange(len(df)));G=(X.T@X).toarray();B=X.T@(Y-base)
 names=build_player_name_map(set(map(int,ids)))
 own=np.bincount(np.searchsorted(ids,df.shooter.to_numpy()),minlength=n)
 exposure=np.bincount(np.searchsorted(ids,df[OFF].to_numpy(dtype=int).ravel()),minlength=n)
 u=np.divide(own,exposure,out=np.zeros(n,dtype=float),where=exposure>0)
 r=pd.DataFrame({'player_id':ids,'player_name':[names.get(int(i),f'ID_{i}') for i in ids],'shot_share':u,'self_shot_attempts':own,'teammate_shot_attempts':exposure-own,'offensive_shooting_opportunities':exposure})
 for t,target in enumerate(TARGETS):
  b=solve_gram(G,B[:,t],n,selection['selected_pairs'][target]);selfrate=b[:n];tmrate=b[n:2*n]
  center=np.average(u*selfrate+(1-u)*tmrate,weights=exposure)
  r['own_'+target+'_conditional']=100*(selfrate-center)
  r['teammate_'+target+'_conditional']=100*(tmrate-center)
  r['own_'+target]=u*r['own_'+target+'_conditional'];r['teammate_'+target]=(1-u)*r['teammate_'+target+'_conditional']
  r[target+'_parent']=r['own_'+target]+r['teammate_'+target]
 r['total_shot_impact']=r[[f'{role}_{t}' for t in TARGETS for role in ['own','teammate']]].sum(axis=1)
 return r.sort_values('total_shot_impact',ascending=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','cv','board','all'],default='all');p.add_argument('--own-grid',default='0.1,1,3,10,30,100,300,1000,3000,10000,30000,100000');p.add_argument('--grid',default='30,100,300,1000,3000,10000,30000,100000,300000,1000000,1000000000000');args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 df=prepare() if args.stage in ['prepare','all'] else pd.read_parquet(OUT/'opportunities.parquet')
 if args.stage=='prepare':return
 if args.stage in ['cv','all']:
  X,ids=design(df);selection=cv(df,X,ids,[float(x) for x in args.grid.split(',')],[float(x) for x in args.own_grid.split(',')])
 else:selection=json.loads((OUT/'selection.json').read_text())
 if args.stage=='cv':return
 boards=[]
 for y in [0,2022,2023,2024,2025,2026]:
  sub=df if not y else df[df.Season==y].reset_index(drop=True)
  r=board(sub,selection);r['window']='2022-2026' if not y else str(y);r['source']='russell_v2';boards.append(r);r.to_csv(OUT/f'v2_six_{r.window.iloc[0]}_results.csv',index=False)
 pd.concat(boards).to_csv(OUT/'all_results.csv',index=False)
 print('DONE',OUT,flush=True)
if __name__=='__main__':main()

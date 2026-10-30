"""Price resolved turnover effects using same-window canonical gross first-chance value."""
import pandas as pd
import numpy as np
import build_own_teammate_tov as model
import build_v2_six_shooting_split as base


def main():
 parts=[]
 for year in range(2022,2027):
  f=base.unique(base.norm(pd.read_parquet(base.ROOT/f'processed/FIRST_CHANCE{year%100}.parquet')))
  line=f[base.OFF+base.DEF].to_numpy(dtype=float)
  valid=np.isfinite(line).all(axis=1)&(line>0).all(axis=1)&(np.diff(np.sort(line,axis=1),axis=1)>0).all(axis=1)
  f=f[valid].copy();f['Season']=year
  if f.Net_Diff.isna().any():raise ValueError('Missing canonical first-chance value')
  parts.append(f)
 priced=pd.concat(parts);r=pd.read_csv(model.OUT/'all_results.csv',dtype={'window':str});prices=[]
 for window in r.window.unique():
  sub=priced if window=='2022-2026' else priced[priced.Season.eq(int(window))]
  non=sub[sub.Is_Turnover.eq(0)];price=float(non.Net_Diff.mean())
  if not np.isfinite(price) or price<=0:raise ValueError('Invalid turnover opportunity price')
  prices.append({'window':window,'tov_points_per_turnover':price,'price_non_tov_possessions':len(non),'method':'mean expected gross first-chance points on canonical non-turnover first-chance rows'})
 price_df=pd.DataFrame(prices);price_df.to_csv(model.OUT/'point_prices.csv',index=False)
 r=r.drop(columns=['tov_points_per_turnover','price_non_tov_possessions','method','own_tov_points','teammate_tov_points','total_otov_points'],errors='ignore').merge(price_df,on='window',validate='many_to_one')
 for source,target in [('own_tov_impact','own_tov_points'),('teammate_tov_suppression','teammate_tov_points'),('total_otov','total_otov_points')]:r[target]=r[source]*r.tov_points_per_turnover
 np.testing.assert_allclose(r.own_tov_points+r.teammate_tov_points,r.total_otov_points,atol=1e-12)
 r.to_csv(model.OUT/'all_results.csv',index=False);print(price_df.to_string(index=False))
if __name__=='__main__':main()

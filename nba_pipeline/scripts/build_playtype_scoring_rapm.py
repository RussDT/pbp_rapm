#!/usr/bin/env python3
"""Research FGA-only additive playtype RAPM. Not full ORAPM or causal role credit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from build_historical_playtype_proxy_backcast import unique_key_frame
from run_assist_points_hierarchy import build_player_name_map

ROOT = Path(__file__).resolve().parents[1]
LANES = ['creation', 'finishing', 'transition', 'spacing', 'other_twos']
MAPPING = {'SELF_CREATED_3':'creation', 'SELF_CREATED_2':'creation',
           'CUT_2':'finishing', 'OTHER_RIM_2':'finishing', 'TRANSITION':'transition',
           'OTHER_3':'spacing', 'OTHER_2':'other_twos'}
COLS = [f'{s}{i}' for s in 'OD' for i in range(1,6)]

def targets(categories, points):
    lane = categories.map(MAPPING)
    if lane.isna().any() or not np.isfinite(points).all():
        raise ValueError('Unknown categories or nonfinite scoring')
    y = np.column_stack([np.where(lane.eq(k), points, 0.) for k in LANES])
    np.testing.assert_allclose(y.sum(axis=1), points, atol=1e-12)
    return np.column_stack([y, points])

def fit(frame, y, alpha_o=2000., alpha_d=4000.):
    """Same linear ridge operator for all children and independently supplied parent."""
    ids = np.unique(frame[COLS].to_numpy(dtype=np.int64))
    n, p = len(frame), len(ids)
    idx = np.searchsorted(ids, frame[COLS].to_numpy(dtype=np.int64))
    idx[:,5:] += p
    x = sparse.coo_matrix((np.ones(n*10), (np.repeat(np.arange(n),10),idx.ravel())),shape=(n,2*p)).tocsr()
    # An unpenalized intercept; positive defense coefficients mean points allowed.
    x = sparse.hstack([x, np.ones((n,1))],format='csc')
    penalty = np.r_[np.full(p,alpha_o),np.full(p,alpha_d),0.]
    beta = splu((x.T@x + sparse.diags(penalty)).tocsc()).solve(np.asarray(x.T@y))
    residual = (x.T@x + sparse.diags(penalty))@beta - x.T@y
    if float(np.abs(residual).max()) > 1e-7:
        raise ValueError('Normal-equation residual exceeds tolerance')
    gap = float(np.max(np.abs(beta[:,:-1].sum(axis=1)-beta[:,-1])))
    if gap > 1e-9: raise ValueError(f'Coefficient closure failed: {gap}')
    result = pd.DataFrame({'nba_id':ids})
    for side,offset,sign in [('o',0,1),('d',p,-1)]:
        exposure=np.bincount(idx[:,:5].ravel() if side=='o' else idx[:,5:].ravel()-p,minlength=p)
        result[side+'_team_fga']=exposure
        block=beta[offset:offset+p].copy()
        block-=np.average(block,axis=0,weights=exposure)
        for j,k in enumerate(LANES+['total_fg']):result[side+'_'+k]=100*sign*block[:,j]
    return result, {'rows':n,'games':int(frame.game_id.nunique()),'coefficient_closure_max':gap,
                    'normal_equation_residual_max':float(np.abs(residual).max()),
                    'category_fga':frame.lane.value_counts().to_dict(),
                    'category_points':dict(zip(LANES,y[:,:-1].sum(axis=0).tolist()))}

def load(year):
    suffix=f'{year%100:02d}'
    paths=[ROOT/'processed'/f'PLAYTYPE_PROXY_PTS_BACKCAST{suffix}.parquet',ROOT/'raw_data'/f'NBA{suffix}.parquet']
    frame=unique_key_frame(pd.read_parquet(paths[0]),label='categories')
    raw=pd.read_parquet(paths[1],columns=['game_id','event_num','event_type','home_description','visitor_description'])
    raw=unique_key_frame(raw,label='raw outcomes',allow_conflicting_keep_last=True)
    frame=frame.merge(raw,on=['game_id','event_num'],how='left',validate='one_to_one',indicator=True)
    if not frame['_merge'].eq('both').all():raise ValueError('Missing raw outcome join')
    if not frame.event_type.isin([1,2]).all():raise ValueError('Non-FGA in source')
    desc=frame.home_description.fillna('')+' '+frame.visitor_description.fillna('')
    points=np.where(frame.event_type.eq(1),np.where(desc.str.contains('3PT',case=False),3.,2.),0.)
    category=frame.Playtype_Proxy_Category_Backcast
    frame['lane']=category.map(MAPPING)
    if frame[COLS].isna().any().any() or (frame[COLS]<=0).any().any():raise ValueError('Invalid lineup IDs')
    if any(len(set(row))!=10 for row in frame[COLS].to_numpy()):raise ValueError('Duplicate lineup player')
    y=targets(category,points)
    source=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    return frame,y,source

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start-year',type=int,default=2022);ap.add_argument('--end-year',type=int,default=2026)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'playtype_scoring_rapm')
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[];audits={};repeat=[]
    for year in range(args.start_year,args.end_year+1):
        print(f'Loading and fitting {year}',flush=True)
        frame,y,source=load(year);board,audit=fit(frame,y);board.insert(0,'season',year)
        records.append(board);audit['sources']=source;audits[str(year)]=audit
        # Deterministic interleaved game halves: descriptive repeatability, not forward validation.
        games=sorted(frame.game_id.unique());half=frame.game_id.isin(games[::2]).to_numpy()
        a,_=fit(frame.loc[half],y[half]);b,_=fit(frame.loc[~half],y[~half])
        common=a.merge(b,on='nba_id',suffixes=('_a','_b'))
        common=common[(common.o_team_fga_a>=1000)&(common.o_team_fga_b>=1000)]
        for k in LANES+['total_fg']:
            repeat.append({'season':year,'lane':k,'players':len(common),'split_game_r':common['o_'+k+'_a'].corr(common['o_'+k+'_b'])})
    result=pd.concat(records,ignore_index=True);names=build_player_name_map(set(result.nba_id))
    result.insert(2,'player_name',result.nba_id.map(names));result.to_csv(args.output_dir/'annual_rapm.csv',index=False)
    result.to_parquet(args.output_dir/'annual_rapm.parquet',index=False)
    result[result.nba_id.eq(203999)].to_csv(args.output_dir/'nikola_jokic.csv',index=False)
    pd.DataFrame(repeat).to_csv(args.output_dir/'split_game_repeatability.csv',index=False)
    manifest={'status':'research proxy; FGA-only, not full ORAPM','units':'points per 100 team FGA',
      'alpha_offense':2000,'alpha_defense':4000,'phase':'regular season','time_decay':False,'rubberband':False,
      'mapping':MAPPING,'interpretation':'Lineup scoring contributions by shot family, not personal role credit. Spacing is non-creation/non-transition threes, not gravity. Creation omits passing credit.',
      'validation':'Algebra and descriptive split-game repeatability only; no predictive admission.', 'seasons':audits}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(result[result.nba_id.eq(203999)][['season','player_name']+['o_'+k for k in LANES+['total_fg']]].to_string(index=False))
if __name__=='__main__':main()

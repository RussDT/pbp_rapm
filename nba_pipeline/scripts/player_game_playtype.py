"""PBP shot-role proxy with frozen prior-season prices and explicit game shrinkage."""
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .analyze_pbp_playtype_difficulty_ots import classify_pbp_shots, estimate_category_prices, PBP_CATEGORIES
except ImportError:
    from analyze_pbp_playtype_difficulty_ots import classify_pbp_shots, estimate_category_prices, PBP_CATEGORIES
ROOT=Path(__file__).resolve().parents[1]


def shot_rows(year,kind='rs'):
    path=ROOT/'raw_data'/f'NBA{year%100:02d}{"_PS" if kind=="ps" else ""}.parquet'
    columns=['game_id','event_num','event_type','player1_id','home_description','visitor_description']
    raw=pd.read_parquet(path,columns=columns)
    raw=raw.loc[pd.to_numeric(raw.event_type).isin([1,2])].drop_duplicates().copy()
    raw['game_id']=raw.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    if raw.duplicated(['game_id','event_num']).any():
        raise ValueError(f'Conflicting shot records: {path}')
    text=raw.home_description.fillna('')+' '+raw.visitor_description.fillna('')
    three=text.str.contains('3PT',case=False)
    raw['pbp_playtype_category']=classify_pbp_shots(raw,three)
    raw['actual_points']=np.where(raw.event_type.eq(1),np.where(three,3.,2.),0.)
    raw['player_id']=raw.player1_id.astype(int)
    return raw[['game_id','event_num','player_id','pbp_playtype_category','actual_points']],path


def price_and_prior(year):
    prior,path=shot_rows(year-1)
    prices=estimate_category_prices(prior,pricing_season=year-1,feature_season=year)
    league=float(prices.league_points_per_fga.iloc[0])
    ev=prices.set_index('pbp_playtype_category').expected_points_per_fga
    prior=prior.copy();prior['difficulty_pp']=50*(league-prior.pbp_playtype_category.map(ev))
    history=prior.groupby('player_id').difficulty_pp.agg(['mean','count'])
    return prices,history,path


def season_difficulty(year):
    prices,history,path=price_and_prior(year)
    ev=prices.set_index('pbp_playtype_category').expected_points_per_fga
    frames=[];paths=[path]
    for kind in ['rs','ps']:
        shots,p=shot_rows(year,kind);paths.append(p);frames.append(shots)
    shots=pd.concat(frames)
    shots['pbp_difficulty_pp']=50*(float(prices.league_points_per_fga.iloc[0])-shots.pbp_playtype_category.map(ev))
    return shots.groupby('player_id').pbp_difficulty_pp.mean().reset_index(),prices,paths


def shrink_difficulty(raw_difficulty,n,prior,k):
    if n<0 or k<0 or not all(np.isfinite(v) for v in [n,prior,k]):
        raise ValueError('Invalid playtype shrinkage inputs')
    if n==0:
        return prior,0.
    if not np.isfinite(raw_difficulty):
        raise ValueError('Missing observed playtype difficulty')
    weight=n/(n+k)
    return weight*raw_difficulty+(1-weight)*prior,weight


def game_difficulty(year,kind,game_ids,k):
    prices,history,path=price_and_prior(year)
    shots,gamepath=shot_rows(year,kind)
    shots=shots.loc[shots.game_id.isin(game_ids)].copy()
    ev=prices.set_index('pbp_playtype_category').expected_points_per_fga
    league=float(prices.league_points_per_fga.iloc[0])
    shots['category_ev']=shots.pbp_playtype_category.map(ev)
    output=[];buckets=[]
    for (gid,pid),group in shots.groupby(['game_id','player_id']):
        raw=50*(league-group.category_ev.mean());n=len(group)
        prior=float(history.loc[pid,'mean']) if pid in history.index else 0.
        value,weight=shrink_difficulty(raw,n,prior,k)
        output.append(dict(game_id=gid,player_id=int(pid),pbp_fga=n,pbp_raw_difficulty_pp=raw,pbp_prior_difficulty_pp=prior,pbp_prior_fga=int(history.loc[pid,'count']) if pid in history.index else 0,pbp_prior_season=year-1,pbp_prior_missing=pid not in history.index,pbp_prior_pseudo_fga=k,pbp_game_weight=weight,pbp_stabilized_difficulty_pp=value))
        for category in PBP_CATEGORIES:
            count=int(group.pbp_playtype_category.eq(category).sum())
            buckets.append(dict(game_id=gid,player_id=int(pid),category=category,FGA=count,share=count/n,prior_season_expected_points_per_fga=float(ev[category]),difficulty_contribution_pp=50*(count/n)*(league-float(ev[category]))))
    return pd.DataFrame(output),pd.DataFrame(buckets),prices,[path,gamepath]

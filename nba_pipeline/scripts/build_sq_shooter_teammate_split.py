#!/usr/bin/env python3
"""Descriptive RS FGA-only shooter/teammate SQ and residual RAPM export."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
try:
    from .run_joint_ts_shooter_interaction import (OFFENSE_COLS, DEFENSE_COLS,
        compute_player_season_shot_shares, collect_player_ids, build_joint_design_matrix, build_results)
except ImportError:
    from run_joint_ts_shooter_interaction import (OFFENSE_COLS, DEFENSE_COLS,
        compute_player_season_shot_shares, collect_player_ids, build_joint_design_matrix, build_results)

ROOT = Path(__file__).resolve().parents[1]
VPM = ROOT.parent.parent / 'vpm/vpm'
OUT = ROOT / 'results/sq_shooter_teammate_2024_2026_rs'
KEY = ['game_id', 'event_num']

def unique_events(frame):
    """Collapse exact duplicates; exclude every conflicting event key."""
    frame = frame.drop_duplicates().copy()
    return frame.loc[~frame.duplicated(KEY, keep=False)].copy()

def load_inputs():
    cols = ['season_end_year','game_id','event_num','shooter_id','actual_points','shot_value',
            *[f'O{i}' for i in range(1,6)],*DEFENSE_COLS]
    raw = pd.read_parquet(ROOT/'results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet',
        columns=cols,filters=[('season_end_year','>=',2024),('season_type','==','RS')])
    for c in KEY: raw[c] = pd.to_numeric(raw[c],errors='raise').astype('int64')
    raw = unique_events(raw)
    for c in ['shooter_id', 'actual_points', 'shot_value', *[f'O{i}' for i in range(1,6)], *DEFENSE_COLS]:
        raw[c] = pd.to_numeric(raw[c], errors='coerce').astype(float)
    sq = pd.read_parquet(VPM/'data/sq_shots_keyed.parquet')
    sq = sq.rename(columns={'GAME_EVENT_ID':'event_num'})
    sq = sq.dropna(subset=KEY)
    for c in KEY: sq[c] = pd.to_numeric(sq[c],errors='raise').astype('int64')
    sq = unique_events(sq)
    geom = 'feature_store.Distance to the Closest Defender (Feet)'
    sq['geometry_available'] = sq[geom].notna()
    sq = sq[KEY+['personId','ev.initial_ev.value','geometry_available']]
    df = raw.merge(sq,on=KEY,how='left',validate='one_to_one')
    df['valid_sq'] = (df.shooter_id.eq(df.personId) & df['ev.initial_ev.value'].notna()
                      & df['ev.initial_ev.value'].ge(0) & df['ev.initial_ev.value'].le(df.shot_value))
    off = df[[f'O{i}' for i in range(1,6)]].to_numpy()
    defense = df[DEFENSE_COLS].to_numpy()
    lineups = np.concatenate([off,defense],axis=1)
    valid = np.isfinite(lineups).all(axis=1) & (lineups>0).all(axis=1)
    valid &= (np.diff(np.sort(lineups,axis=1),axis=1)>0).all(axis=1)
    valid &= (off==df.shooter_id.to_numpy()[:,None]).sum(axis=1)==1
    df['retained'] = df.valid_sq & valid
    audit = df.groupby('season_end_year').agg(raw_fga=('game_id','size'),raw_games=('game_id','nunique'),
        sq_matched=('valid_sq','sum'),retained_fga=('retained','sum')).reset_index()
    df = df[df.retained].copy()
    off = df[[f'O{i}' for i in range(1,6)]].to_numpy()
    idx = (off==df.shooter_id.to_numpy()[:,None]).argmax(axis=1)
    context = np.empty((len(df),4),dtype=np.int64)
    for j in range(5): context[idx==j] = np.delete(off[idx==j],j,axis=1)
    df[OFFENSE_COLS[1:]] = context
    df['Focal_Player'] = df.shooter_id.astype('int64')
    df['Season'] = df.season_end_year
    pred = pd.read_parquet(ROOT/'results/sq_walk_forward_challenger/walk_forward_predictions.parquet')
    pred = pred.rename(columns={'game_id_normalized':'game_id','GAME_EVENT_ID':'event_num',
                                'personId':'pred_shooter','made':'pred_made'})
    pred = pred.dropna(subset=KEY)
    for c in KEY: pred[c] = pd.to_numeric(pred[c],errors='raise').astype('int64')
    pred = unique_events(pred)
    target = 'pps_context_epm_def_geometry_identity'
    df = df.merge(pred[KEY+['pred_shooter','pred_made',target]],on=KEY,how='left',validate='one_to_one')
    improved = df.pred_shooter.eq(df.shooter_id) & df.pred_made.eq(df.actual_points/df.shot_value) & df[target].notna()
    df['provider_sq'] = df['ev.initial_ev.value']
    df['improved_hybrid_sq'] = np.where(improved,df[target],df.provider_sq)
    df['improved_prediction'] = improved
    extra = df.groupby('Season').agg(covered_games=('game_id','nunique'),geometry_fga=('geometry_available','sum'),
                                    improved_fga=('improved_prediction','sum')).reset_index()
    audit = audit.merge(extra,left_on='season_end_year',right_on='Season').drop(columns='Season')
    return df.reset_index(drop=True),audit

def fit(X,y):
    return Ridge(alpha=1,fit_intercept=False,solver='lsqr',tol=1e-8).fit(X,y).coef_

def centered(df, values, baseline=None):
    if baseline is None: baseline = pd.Series(values).groupby(df.Season.to_numpy()).mean()
    return np.asarray(values)-df.Season.map(baseline).to_numpy()

def choose_role(df, values, ids, shares, tag):
    splits = list(GroupKFold(3).split(df,groups=df.game_id))
    records=[]
    for a in [300.,3000.,30000.,1e12]:
        X,_ = build_joint_design_matrix(df,ids,shares,3000,5000,a)
        losses=[]
        for tr,va in splits:
            baseline = pd.Series(values[tr]).groupby(df.Season.to_numpy()[tr]).mean()
            y = centered(df,values,baseline)
            b=fit(X[tr],y[tr]); losses.append(float(np.mean((y[va]-X[va]@b)**2)))
        records.append({'target':tag,'role_alpha':a,'mean_mse':np.mean(losses),**{f'fold_{i}':v for i,v in enumerate(losses)}})
        print(tag,a,losses,flush=True)
    cv=pd.DataFrame(records); best=cv.loc[cv.mean_mse.idxmin()]
    # Paired game-fold one-SE rule; strongest role shrinkage statistically tied to best.
    foldcols=[f'fold_{i}' for i in range(3)]
    delta=cv[foldcols].to_numpy()-best[foldcols].to_numpy(dtype=float)
    cv['paired_se']=delta.std(axis=1,ddof=1)/np.sqrt(3)
    cv['eligible']=cv.mean_mse-best.mean_mse<=cv.paired_se+1e-15
    selected=float(cv.loc[cv.eligible,'role_alpha'].max())
    cv['selected']=cv.role_alpha.eq(selected)
    return selected,cv

def solve(df, values, alpha):
    ids=collect_player_ids(df); shares=compute_player_season_shot_shares(df)
    X,_=build_joint_design_matrix(df,ids,shares,3000,5000,alpha)
    b=fit(X,centered(df,values))
    r,_=build_results(df,ids,b,3000,5000,alpha)
    return r

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df,audit=load_inputs(); audit.to_csv(OUT/'coverage.csv',index=False)
    print(audit.to_string(index=False),flush=True)
    ids=collect_player_ids(df); shares=compute_player_season_shot_shares(df)
    boards=[]; cvs=[]; penalties={}
    for source in ['provider','improved_hybrid']:
        values={'sq':df[source+'_sq'].to_numpy(), 'make':(df.actual_points-df[source+'_sq']).to_numpy()}
        penalties[source]={}
        for target,y in values.items():
            alpha,cv=choose_role(df,y,ids,shares,source+'_'+target)
            penalties[source][target]=alpha;cvs.append(cv)
        for season in [0,2024,2025,2026]:
            mask=np.ones(len(df),bool) if season==0 else df.Season.eq(season).to_numpy()
            sub=df.loc[mask].reset_index(drop=True); board=None
            for target,y in values.items():
                r=solve(sub,y[mask],penalties[source][target])
                keep=['player_id','player_name','shot_share','self_shot_attempts','teammate_shot_attempts']
                rename={'joint_ots_rapm':target+'_parent','conditional_self_shot_ots':'own_'+target+'_conditional',
                    'conditional_teammate_shot_ots':'teammate_'+target+'_conditional',
                    'self_shot_ots_contribution':'own_'+target,'teammate_shot_ots_contribution':'teammate_'+target}
                r=r[keep+list(rename)].rename(columns=rename)
                board=r if board is None else board.merge(r.drop(columns=keep[1:]),on='player_id',validate='one_to_one')
            board['total_shot_impact']=board.sq_parent+board.make_parent
            board['offensive_fga_exposures']=board.self_shot_attempts+board.teammate_shot_attempts
            board['source']=source;board['window']='2024-2026' if not season else str(season)
            for flag in ['geometry_available','improved_prediction']:
                counts={}
                for c in OFFENSE_COLS:
                    for pid,mean in sub.groupby(c)[flag].agg(['sum','count']).iterrows():
                        old=counts.get(int(pid),(0,0));counts[int(pid)]=(old[0]+mean['sum'],old[1]+mean['count'])
                board[flag+'_share']=board.player_id.map(lambda p:counts.get(p,(0,1))[0]/counts.get(p,(0,1))[1])
            if source == 'provider':
                board['improved_prediction_share'] = 0.0
            assert np.max(np.abs(board.own_sq+board.teammate_sq-board.sq_parent))<1e-10
            assert np.max(np.abs(board.own_make+board.teammate_make-board.make_parent))<1e-10
            boards.append(board)
            board.to_csv(OUT/f'{source}_{board.window.iloc[0]}_results.csv',index=False)
    result=pd.concat(boards,ignore_index=True)
    result.to_csv(OUT/'all_results.csv',index=False)
    pd.concat(cvs).to_csv(OUT/'role_alpha_cv.csv',index=False)
    manifest={'seasons':[2024,2025,2026],'season_type':'RS','rows':len(df),'games':int(df.game_id.nunique()),
        'off_alpha':3000,'def_alpha':5000,'role_alphas':penalties,'time_decay':False,'box_priors':False,
        'units':'points per 100 on-court team FGA; conditional columns per 100 own/teammate FGA',
        'scope':'FGA only; no free throws, turnovers or rebound continuation; descriptive research',
        'improved_hybrid':'saved context_epm_def_geometry_identity for matched 2025/2026 mids/threes; provider initial_ev elsewhere',
        'selection':'3 grouped-game folds, paired one-SE role-alpha selection independently per source/target; annual boards reuse pooled alphas',
        'limitations':'SQ includes shooter talent. Residual is actual minus talent-aware expected points. Teammate values do not isolate passing or gravity. Improved hybrid is not a uniform three-year SQ model. Missing provider EV excluded, never imputed. Geometry coverage is observed distance-field availability, not proof of provider internal feature availability.',
        'sources':[str(VPM/'data/sq_shots_keyed.parquet'),str(ROOT/'results/sq_walk_forward_challenger/walk_forward_predictions.parquet')]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    template = (ROOT/'scripts/templates/sq_shooter_teammate.html').read_text()
    report = template.replace('__DATA__', result.to_json(orient='records').replace('</', '<\\/')).replace('__AUDIT__', audit.to_json(orient='records'))
    (OUT/'shot_quality_split.html').write_text(report)
    print('DONE',OUT,flush=True)

if __name__=='__main__': main()

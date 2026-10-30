"""Nested season-held-out team workload calibration diagnostic; never changes ratings.

NBA: expanding chronology, outer 2018-2026; all feature/alpha selection uses only
prior seasons. WNBA: 2025 chronological primary and nested LOSO sensitivity.
Historical data are spent development evidence. Fixed incumbent prices merely
translate workload error into turnover units; price estimation is not held out.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from nba_pipeline.scripts import build_nba_rztov_box_backcast as nba
from nba_pipeline.scripts import audit_rztov_team_totals as audit
from wnba_test import build_wnba_rztov as wnba
OUT=ROOT/'nba_pipeline/results/rztov_team_calibration'


def season_split(panel, heldout, mode):
    if mode == 'chronological':
        train=panel[panel.season.lt(heldout)]
    elif mode == 'loso':
        train=panel[panel.season.ne(heldout)]
    else: raise ValueError(mode)
    test=panel[panel.season.eq(heldout)]
    if train.empty or test.empty: raise ValueError('Empty season fold')
    assert heldout not in set(train.season)
    return train.copy(),test.copy()


def wnba_select(train,target,features,mode):
    scores=[]
    seasons=sorted(train.season.unique())
    for alpha in wnba.ALPHAS:
        losses=[];weights=[]
        for heldout in seasons:
            inner=train[train.season.lt(heldout)] if mode=='chronological' else train[train.season.ne(heldout)]
            val=train[train.season.eq(heldout)]
            if inner.empty: continue
            model=wnba._model(features,alpha)
            model.fit(inner[features],inner[target],ridge__sample_weight=inner.off_chances)
            pred=np.maximum(model.predict(val[features]),0)
            if target=='assist_opp_share': pred=np.minimum(pred,1)
            losses.extend((pred-val[target])**2);weights.extend(val.off_chances)
        if not losses: raise ValueError('No inner validation fold')
        scores.append((float(np.average(losses,weights=weights)),alpha))
    _,alpha=min(scores)
    model=wnba._model(features,alpha)
    model.fit(train[features],train[target],ridge__sample_weight=train.off_chances)
    return model,alpha


def allowance_error(tsa, estimated_share, actual_share, estimated_potast, actual_potast, self_price, assisted_price, potast_price):
    # Positive = estimated workload grants more allowance than observed workload.
    return (self_price-assisted_price)*tsa*(actual_share-estimated_share)+potast_price*(estimated_potast-actual_potast)


def predict_fold(league,panel,heldout,mode):
    train,test=season_split(panel,heldout,mode)
    receipt={'league':league,'mode':mode,'heldout':int(heldout),'train_seasons':[int(x) for x in sorted(train.season.unique())]}
    for target,output in [('assist_opp_share','predicted_share'),('potential_assists100','predicted_potast')]:
        if league=='NBA':
            model,family,alpha,_,_=nba.select_model_family(train,target,nba.FEATURE_FAMILIES[target],0,1 if target=='assist_opp_share' else None)
            features=nba.FEATURE_FAMILIES[target][family]
        else:
            features=wnba.ASSIST_SHARE_FEATURES if target=='assist_opp_share' else wnba.POTENTIAL_ASSIST_FEATURES
            model,alpha=wnba_select(train,target,features,mode);family='incumbent'
        pred=np.maximum(model.predict(test[features]),0)
        if target=='assist_opp_share': pred=np.minimum(pred,1)
        test[output]=pred
        receipt[target]={'family':family,'alpha':float(alpha)}
    test['pid']=test.player_id.astype(int)
    print(receipt,flush=True)
    return test,receipt


def make_board(league,test,base):
    b=base.merge(test[['pid','season','predicted_share','predicted_potast','assist_opp_share','potential_assists100']],on=['pid','season'],validate='one_to_one')
    if league=='NBA':
        tsa=b.tsa100;ps=b.be_self;pa=b.be_ast;pp=b.be_potast
    else:
        tsa=b.tsa100_raw
        prices=json.loads((wnba.DEFAULT_OUTPUT/'manifest.json').read_text())['break_evens']
        ps=prices['self'];pa=prices['assisted'];pp=prices['potential_assist']
    b['role_error100']=allowance_error(tsa,b.predicted_share,b.assist_opp_share,b.predicted_potast,b.potential_assists100,ps,pa,pp)
    b['share_error']=b.predicted_share-b.assist_opp_share
    b['potast_error']=b.predicted_potast-b.potential_assists100
    # Keep the incumbent allowance only as a fixed scale/denominator. The error
    # comes entirely from held-out workload predictions at common canonical TSA.
    return b


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    features=pd.read_parquet(nba.DEFAULT_OUTPUT/'all_era_box_features_1997_2026.parquet')
    panels={'NBA':nba.build_training_panel(features,nba.DEFAULT_OBSERVED),'WNBA':wnba.build_training_panel(wnba.DEFAULT_RTSS_FEATURES)}
    bases={
      'NBA':pd.read_csv(nba.DEFAULT_OUTPUT/'nba_box_estimated_rztov_1997_2026_diagnostic.csv').rename(columns={'nba_id':'pid','year':'season','OffPoss':'off_poss','act':'actual','v_self':'self','v_ast':'assisted','v_potast':'potast','sc_100':'sc','bp_100':'bp'}),
      'WNBA':pd.read_csv(wnba.DEFAULT_OUTPUT/'wnba_box_estimated_rztov_2022_2026.csv').rename(columns={'player_id':'pid','tov100_raw':'actual','turnover_allowance100':'allowance','scoring_allowance100':'sc_allow','badpass_allowance100':'bp_allow','estimated_self_created_tsa100':'self','estimated_assisted_tsa100':'assisted','estimated_potential_assists100':'potast','sc_tov100_raw':'sc','bp_tov100_raw':'bp'})}
    all_teams=[];all_players=[];receipts=[]
    folds=[('NBA',year,'chronological') for year in range(2018,2027)]+[('WNBA',2025,'chronological')]+[('WNBA',year,'loso') for year in range(2023,2026)]
    for league,year,mode in folds:
        test,receipt=predict_fold(league,panels[league],year,mode)
        b=make_board(league,test,bases[league])
        teams,players,source=audit.audit(league,year,b)
        teams['mode']=mode;players['mode']=mode
        receipt['players_scored']=len(test);receipt['players_in_ledger']=len(b)
        receipt['unallocated_players']=source['board_players_without_slots']
        if receipt['unallocated_players']: raise ValueError('Missing player allocation')
        # Team+unknown allocation must preserve the original player's error sum.
        expected=float((b.role_error100*b.off_poss/100).sum())
        if not np.isclose(teams.role_error_count.sum(),expected,atol=1e-7): raise AssertionError('Team allocation loses role error')
        receipts.append(receipt);all_teams.append(teams);all_players.append(players)
        pd.concat(all_teams).to_csv(OUT/'team_folds.csv',index=False)
        pd.concat(all_players).to_csv(OUT/'player_folds.csv',index=False)
        (OUT/'fold_receipts.json').write_text(json.dumps(receipts,indent=2))
    result=pd.concat(all_teams)
    summary=[]
    for (league,mode),g in result[result.team.gt(0)&result.common_allowance.gt(0)].groupby(['league','mode']):
        pct=g.role_error_pct_allowance
        summary.append({'league':league,'mode':mode,'team_seasons':len(g),'median_abs_pct':float(pct.abs().median()),'p90_abs_pct':float(pct.abs().quantile(.9)),'min_pct':float(pct.min()),'max_pct':float(pct.max()),'min_count':float(g.role_error_count_centered.min()),'max_count':float(g.role_error_count_centered.max())})
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    manifest={'status':'completed_development_diagnostic_no_rating_change','workload_selection':'nested, outer test labels excluded from model and hyperparameter selection','prices':'fixed incumbent diagnostic conversion, not held-out pricing','team_allocation':'season rates by game-resolved offensive-possession shares; unknown team retained','centering':'test-season common-sample mean removed for diagnostic comparison only; raw error also exported','future_confirmation':'NBA 2027 untouched; WNBA needs fresh observed labels','summaries':summary}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__': main()

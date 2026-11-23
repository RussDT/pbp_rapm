"""Matched descriptive scoring of saved full-feature VPM and same-date EPM forecasts."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
HARNESS=ROOT.parents[1]/'vpm/vpm/results/harness'
OUT=ROOT/'results/estimated_skills/full_feature_history_epm_matched_daily_20261102_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    vp=HARNESS/'vpm_full_feature_history_v1/spent_forecasts_v3/predictions.parquet'
    ep=HARNESS/'vpm_daily_trained_epm_provenance_v2/comparison/game_predictions.parquet'
    va=ROOT/'results/estimated_skills/full_feature_history_spent_forecast_audit_v1/actual_v1/receipt.json'
    er=ep.parent/'receipt.json'
    audit=json.loads(va.read_bytes());receipt=json.loads(er.read_bytes())
    for path,record in [(vp,audit),(ep,receipt)]:
        pins={**record.get('input_hashes',{}),**record.get('output_hashes',{})}
        if pins.get(str(path.resolve()))!=sha(path):raise ValueError('Saved forecast differs from its existing result record')
    vr=HARNESS/'vpm_full_feature_history_v1/spent_replay_registration_v3/registration.json'
    registration=json.loads(vr.read_bytes());minutes=registration['actual_minutes_path']
    assert registration['input_hashes'][minutes]==receipt['input_hashes'][minutes]
    # Both forecasts use the same actual-minute source, not separately forecast minutes.
    assert sha(minutes)==registration['input_hashes'][minutes]
    v=pd.read_parquet(vp);e=pd.read_parquet(ep)
    for d in [v,e]:
        d.game_id=d.game_id.astype(str).str.zfill(10)
        assert not d.duplicated(['season','game_id']).any()
    assert set(v.model)=={'full_split_raw'}
    x=v.merge(e,on=['season','game_id'],suffixes=('_v','_e'),how='left',validate='one_to_one',indicator=True)
    absent=x.loc[x._merge.ne('both'),['season','game_id','game_date_v']].copy()
    x=x.loc[x._merge.eq('both')].copy()
    for c in ['home_score','away_score','margin','pace','duration','home_team','away_team','game_date']:
        np.testing.assert_array_equal(x[c+'_v'].to_numpy(),x[c+'_e'].to_numpy())
    for side in ['v','e']:assert pd.to_datetime(x['estimate_date_'+side]).eq(pd.to_datetime(x['game_date_'+side])).all()
    assert np.isfinite(x[['prediction_v','EPM','margin_v']].to_numpy()).all()
    x['VPM_squared_error']=(x.prediction_v-x.margin_v)**2
    x['EPM_squared_error']=(x.EPM-x.margin_v)**2
    x['EPM_minus_VPM_squared_error']=x.EPM_squared_error-x.VPM_squared_error
    groups=[g for _,g in x.groupby('season',sort=True)]
    means=np.array([g.EPM_minus_VPM_squared_error.mean()for g in groups])
    # Paired whole-game bootstrap within each season, with equal season weighting.
    settings=json.loads((HARNESS/'vpm_daily_epm_v3/registration.json').read_bytes())['settings']
    seed=int(settings['seed']);reps=int(settings['bootstrap_replicates']);rng=np.random.default_rng(seed)
    draws=np.empty(reps)
    errors=[g.EPM_minus_VPM_squared_error.to_numpy()for g in groups]
    for i in range(reps):draws[i]=np.mean([rng.choice(a,size=len(a),replace=True).mean()for a in errors])
    season_draws=np.array([means[list(ix)].mean()for ix in itertools.product(range(len(groups)),repeat=len(groups))])
    rows=[]
    for year,g in x.groupby('season',sort=True):
        rows.append(dict(season=int(year),games=len(g),VPM_RMSE=float(np.sqrt(g.VPM_squared_error.mean())),EPM_RMSE=float(np.sqrt(g.EPM_squared_error.mean()))))
    mse=x.groupby('season')[['VPM_squared_error','EPM_squared_error']].mean().mean()
    result=dict(comparison='full_split_raw_daily_vs_same_date_EPM_saved_forecasts',games=len(x),vpm_saved_games=len(v),
        vpm_games_without_saved_EPM_comparison=len(absent),seasons=[int(g.season.iloc[0])for g in groups],
        aggregation='equal_season_mean_game_squared_error_then_square_root',
        VPM_RMSE=float(np.sqrt(mse.VPM_squared_error)),EPM_RMSE=float(np.sqrt(mse.EPM_squared_error)),
        EPM_minus_VPM_MSE=float(means.mean()),paired_game_95_interval=np.quantile(draws,[.025,.975]).tolist(),
        paired_season_95_interval=np.quantile(season_draws,[.025,.975]).tolist(),bootstrap_seed=seed,
        game_replicates=reps,season_replicates=len(season_draws),by_season=rows,
        same_game_targets_dates_pace_duration_teams=True,same_actual_minutes_source=True,
        fits_and_selection_held_fixed=True,retrospective_spent_history=True,new_fits=0,
        measures_availability_forecasting=False,current_October30_board_prospectively_tested=False,
        analysis_scope='New descriptive matched calculation; not a new independent audit or complete seven-test report',
        source_files=[dict(path=str(p.resolve()),sha256=sha(p))for p in [vp,ep,va,er,vr,Path(__file__)]])
    OUT.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(OUT/'per_season.csv',index=False)
    x[['season','game_id','game_date_v','margin_v','prediction_v','EPM','VPM_squared_error','EPM_squared_error']].to_parquet(OUT/'matched_predictions.parquet',index=False)
    absent.to_csv(OUT/'games_without_saved_epm_comparison.csv',index=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    report=['# Full-feature VPM versus EPM: matched daily forecasts','',
        f'Historical ending seasons2023–2026; {len(x):,} common games. Equal weight per season. Lower game-margin RMSE is better.','',
        '| Season | Games | Full-feature VPM | EPM |','|---|---:|---:|---:|']
    report += [f"| {z['season']} | {z['games']} | {z['VPM_RMSE']:.6f} | {z['EPM_RMSE']:.6f} |"for z in rows]
    report += [f"| Equal-season overall | {len(x)} | {result['VPM_RMSE']:.6f} | {result['EPM_RMSE']:.6f} |",'',
        f"EPM-minus-VPM MSE: {result['EPM_minus_VPM_MSE']:+.6f}; paired game95% interval {result['paired_game_95_interval']}; exact season-block95% interval {result['paired_season_95_interval']}. Positive favors VPM.",'',
        'No demonstrated overall advantage. These are saved, date-matched daily forecasts using identical actual minutes, pace and game outcomes. This does not measure availability or minutes forecasting. Fits and selection are held fixed; historical outcomes are spent and source publication vintages are not independently established.',
        f'{len(absent)} of the4634 saved VPM games lack a corresponding row in the older saved daily EPM comparison and are explicitly excluded from this matched calculation. The EPM control retains its original missing-player fallback rules; a VPM-only entrant/universe change is not isolated here.',
        'The October30 current board itself has not received a prospective EPM performance test. This is a new descriptive calculation from existing forecast products, not a new independent audit or completed seven-test report.']
    (OUT/'comparison.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='source_files'},indent=2))

if __name__=='__main__':main()

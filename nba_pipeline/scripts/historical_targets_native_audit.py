"""Independent source/endpoint audit of quarterly native VPM results, without fits.

No quarterly fitted-module function is imported. Actual points and dated player
admission come from RAPM's frozen measurement receipts; zero-control snapshots
are checked against normal equations on declared representative cuts.
"""
from pathlib import Path
import argparse
import gc
import json
import numpy as np
import pandas as pd
from scipy import sparse
from historical_targets import OUT, ROOT, SLOTS, sha, load_before_cut, target_matrix
from historical_targets_identities import complete_rows


def read(p):
    return json.loads(Path(p).read_text())


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + str(path))


def raw_lineup_prediction(frame, ids, beta, intercept):
    ids = np.asarray(ids, np.int64); beta = np.asarray(beta, float)
    slots = frame[SLOTS].to_numpy(np.int64)
    ix = pd.Index(ids).get_indexer(slots.ravel()).reshape(-1, 10)
    known = (ix >= 0).all(axis=1)
    pred = np.full(len(frame), np.nan)
    # The second coefficient block is already negative for positive-good D.
    a = ix[known].copy(); a[:, 5:] += len(ids)
    pred[known] = beta[a].sum(axis=1) + float(intercept)
    return pred, known


def independent_bootstrap(by_game, replicates=4000, seed=20261021):
    rng = np.random.default_rng(seed); draw_means=[]; cut_means=[]
    for _, group in by_game.groupby('estimate_date', sort=True):
        values = group[['loss_gain_sum', 'rows']].to_numpy(float)
        cut_means.append(values[:, 0].sum() / values[:, 1].sum())
        index = rng.integers(len(values), size=(replicates, len(values)))
        totals = values[index].sum(axis=1)
        draw_means.append(totals[:, 0] / totals[:, 1])
    if not cut_means:
        raise ValueError('No nonempty native evaluation windows')
    mean = np.asarray(draw_means).mean(axis=0)
    return dict(mse_gain=float(np.mean(cut_means)),
                ci_low=float(np.quantile(mean, .025)), ci_high=float(np.quantile(mean, .975)))


def normal_equation_check(train, games, cut, ids, vectors, board):
    """Check saved fixed-control equations by sparse products; never solve/refit."""
    w = np.exp2(-(train.game_date.max() - train.game_date).dt.total_seconds().to_numpy() / (86400 * 550))
    target_matrix(train, games, cut, w)
    slots = train[SLOTS].to_numpy(np.int64)
    cols = pd.Index(ids).get_indexer(slots.ravel()).reshape(-1, 10)
    if (cols < 0).any():
        raise ValueError('Control snapshot omits historical training player')
    cols[:, 5:] += len(ids)
    x = sparse.csr_matrix((np.ones(cols.size), (np.repeat(np.arange(len(train)), 10), cols.ravel())),
                          shape=(len(train), 2 * len(ids)))
    xtw = x.T.multiply(w).tocsr()
    exposure = np.asarray(x.T @ w).ravel()
    np.testing.assert_allclose(exposure[:len(ids)], board.td_off_possessions, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(exposure[len(ids):], board.td_def_possessions, atol=1e-7, rtol=1e-10)
    cases = [('actual_beta', train.actual_points.to_numpy(), 4000., 8000.)]
    non = ~train.Is_Turnover.eq(1).to_numpy()
    for zone in ['mid', 'three']:
        value = train[zone + '_residual'].to_numpy().copy()
        value[~non] = np.average(value[non], weights=w[non])
        cases += [(zone + '_weak', value, 4000., 8000.), (zone + '_strong', value, 12000., 48000.)]
    gaps={}
    for key, value, oa, da in cases:
        beta = vectors[key]; penalty=np.r_[np.full(len(ids), oa), np.full(len(ids), da)]
        rhs = np.asarray(xtw @ (value - np.average(value, weights=w))).ravel()
        error = np.asarray(xtw @ (x @ beta)).ravel() + penalty * beta - rhs
        relative=float(abs(error).max() / max(1., abs(rhs).max()))
        if relative > 1e-9:
            raise ValueError(f'Saved control normal equation failed {cut} {key}: {relative}')
        gaps[key]=relative
    fields = {'raw_ovpm', 'raw_dvpm', 'combined_prior_off', 'combined_prior_def'}
    if fields.issubset(board):
        # Shared actual/SQ response cancels; remaining movement must equal
        # the fixed ridge response to exactly this saved combined prior.
        candidate = np.r_[board.raw_ovpm, -board.raw_dvpm] / 100
        prior = np.r_[board.combined_prior_off, -board.combined_prior_def] / 100
        baseline = vectors['actual_beta'] + sum(vectors[z + '_strong'] - vectors[z + '_weak'] for z in ['mid', 'three'])
        delta = candidate - baseline
        penalty = np.r_[np.full(len(ids), 4000.), np.full(len(ids), 8000.)]
        rhs = penalty * prior
        error = np.asarray(xtw @ (x @ delta)).ravel() + penalty * delta - rhs
        relative = float(abs(error).max() / max(1., abs(rhs).max()))
        if relative > 1e-9:
            raise ValueError(f'Candidate and control are not matched prior responses {cut}: {relative}')
        gaps['candidate_minus_zero_prior_response'] = relative
    return gaps


def load_admitted(root):
    packet = read(root / 'identity_admission.json')
    if packet['status'] != 'passed':
        raise ValueError('Identity packet has not passed')
    verify_hashes(packet['source_hashes'])
    d, g = load_before_cut(root, '2026-10-02', range(2000, 2027), sq=True)
    keep = complete_rows(d[SLOTS].to_numpy())
    for year, indices in d.groupby('source_season_end_year').indices.items():
        keep[indices] &= np.isin(d.iloc[indices][SLOTS].to_numpy(), packet['seasons'][str(year)]['admitted_ids']).all(axis=1)
    d = d.loc[keep].reset_index(drop=True)
    if len(d) != 6520215 or d.game_date.max() >= pd.Timestamp('2027-01-01'):
        raise ValueError('Independent admitted historical census changed')
    keys=d[['source_season_end_year', 'season_phase', 'game_id']].drop_duplicates()
    g=g.merge(keys, on=list(keys), validate='one_to_one')
    return d,g


def run(out, root=OUT, normal_cuts=('2014-01-01', '2016-01-01', '2026-04-01')):
    out, root = Path(out), Path(root)
    for rejection in out.glob('rejected*.json'):
        if str(read(rejection).get('status', '')).startswith('rejected_'):
            raise ValueError('Rejected experiment cannot receive a final native acceptance audit: ' + str(rejection))
    native=out/'native_forward'; summary=read(native/'summary.json')
    prepare=read(out/'preparation.json'); verify_hashes(prepare['source_hashes'])
    target_sources={}; target_artifacts={}
    for year in range(2000, 2027):
        for phase in ['RS', 'PS']:
            m=read(root/f'historical_{year}_{phase}.json')
            target_sources.update(m['source_files'])
            target_artifacts.update({str(root/name): h for name,h in m['artifacts'].items()})
    verify_hashes(target_sources); verify_hashes(target_artifacts)
    verify_hashes(summary['source_hashes'])
    verify_hashes({str(native/name): h for name,h in summary['audit_artifacts'].items()})
    native_code=ROOT.parent.parent/'vpm/vpm/harness/quarterly_native.py'
    if sha(native_code) != summary['code_sha256']:
        raise ValueError('Native result code changed')
    data,games=load_admitted(root)
    score=pd.read_csv(native/'quarterly_scores.csv'); paired=pd.read_parquet(native/'paired_games.parquet')
    selection=read(out/'benchmark/selection.json')
    dates=pd.date_range('2014-01-01','2026-10-01',freq='QS-JAN')
    if list(pd.to_datetime(score.estimate_date)) != list(dates):
        raise ValueError('Native result does not cover exactly the 52 declared quarters')
    checks=[]; reconstructed=[]; normal_checks={}; max_endpoint_gap=0.; max_scalar_gap=0.
    for cut in dates:
        stamp=str(cut.date()); end=cut+pd.DateOffset(months=3)
        future=data.loc[data.game_date.ge(cut)&data.game_date.lt(end)].reset_index(drop=True)
        row=score.loc[score.estimate_date.eq(stamp)].iloc[0]
        if future.empty:
            if row.status!='no_future_games_in_available_archive' or row.rows!=0 or row.games!=0:
                raise ValueError('Empty native quarter was treated as scored')
            if pd.notna(row.actual_mse):
                raise ValueError('Empty native quarter has a numerical score')
            checks.append(dict(estimate_date=stamp,status='verified_empty',rows=0));continue
        folder=out/'cuts'/stamp; meta=read(folder/'manifest.json')
        family=selection['selected_late'] if selection['selected_late'] in meta['families'] else selection['selected_early']
        if row.family!=family:
            raise ValueError('Native evaluation selected a different family')
        board=pd.read_parquet(folder/(family+'_ratings.parquet')); ids=board.player_id.to_numpy(np.int64)
        raw_beta=np.r_[board.raw_ovpm,-board.raw_dvpm]/100
        prediction,known=raw_lineup_prediction(future,ids,raw_beta,meta['families'][family]['raw_intercept'])
        d=future.loc[known].reset_index(drop=True); prediction=prediction[known]
        endpoint=pd.read_parquet(native/'endpoints'/(stamp+'.parquet'))
        for column in ['game_id','event_num']:
            np.testing.assert_array_equal(pd.to_numeric(endpoint[column]),pd.to_numeric(d[column]))
        np.testing.assert_array_equal(pd.to_datetime(endpoint.game_date),d.game_date)
        np.testing.assert_allclose(endpoint.actual,d.actual_points,atol=0,rtol=0)
        vectors=np.load(native/'controls'/(stamp+'.npz'))
        np.testing.assert_array_equal(vectors['ids'],ids)
        combined=(vectors['actual_beta']+vectors['mid_strong']-vectors['mid_weak']
                  +vectors['three_strong']-vectors['three_weak'])
        np.testing.assert_allclose(combined,vectors['final_beta'],atol=1e-12,rtol=0)
        control,known2=raw_lineup_prediction(d,ids,vectors['final_beta'],vectors['intercept'])
        if not known2.all():raise ValueError('Independent control has unknown future players')
        train=data.loc[data.game_date.lt(cut)]
        anchor=train.game_date.max()
        if str(anchor.date())!=str(vectors['weight_anchor'].item()) or float(vectors['half_life_days'])!=550:
            raise ValueError('Control uses a different time-decay anchor')
        w=np.exp2(-(anchor-train.game_date).dt.total_seconds().to_numpy()/(86400*550))
        mean=float(np.average(train.actual_points,weights=w))
        np.testing.assert_allclose(mean,vectors['intercept'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(mean,meta['families'][family]['raw_intercept'],atol=1e-12,rtol=0)
        prior_beta=np.r_[board.combined_prior_off,-board.combined_prior_def]/100
        exposure=np.r_[board.td_off_possessions,board.td_def_possessions]
        prior_intercept=mean-float(exposure@prior_beta/w.sum())
        np.testing.assert_allclose(prior_intercept,vectors['prior_intercept'],atol=1e-12,rtol=0)
        prior,_=raw_lineup_prediction(d,ids,prior_beta,prior_intercept)
        for name, value in [('pred',prediction),('control',control),('prior',prior)]:
            gap=float(np.max(abs(endpoint[name].to_numpy()-value)))
            if gap>1e-11:raise ValueError(f'Native endpoint reconstruction failed {stamp} {name}: {gap}')
            max_endpoint_gap=max(max_endpoint_gap,gap)
        actual=d.actual_points.to_numpy(); loss=(actual-prediction)**2; zero_loss=(actual-control)**2
        values=dict(actual_mse=float(loss.mean()),matched_zero_prior_mse=float(zero_loss.mean()),
                    prior_only_mse=float(np.mean((actual-prior)**2)),mse_gain_vs_zero_prior=float(np.mean(zero_loss-loss)))
        for key,value in values.items():
            gap=abs(float(row[key])-value)
            if gap>1e-11:raise ValueError(f'Native scalar score differs {stamp} {key}')
            max_scalar_gap=max(max_scalar_gap,gap)
        if int(row.rows)!=len(d) or int(row.games)!=d.game_id.nunique() or int(row.excluded_unknown_player_rows)!=int((~known).sum()):
            raise ValueError('Native common-player sample counts differ')
        frame=pd.DataFrame(dict(estimate_date=stamp,game_id=d.game_id,rows=1,loss_gain_sum=zero_loss-loss))
        bygame=frame.groupby(['estimate_date','game_id'],as_index=False).agg(rows=('rows','sum'),loss_gain_sum=('loss_gain_sum','sum'))
        reconstructed.append(bygame)
        if stamp in normal_cuts:
            keys=train[['source_season_end_year','season_phase','game_id']].drop_duplicates()
            g=games.merge(keys,on=list(keys),validate='one_to_one')
            normal_checks[stamp]=normal_equation_check(train.copy().reset_index(drop=True),g,cut,ids,vectors,board)
        checks.append(dict(estimate_date=stamp,status='verified_actual_endpoints',rows=len(d),games=int(d.game_id.nunique())))
        print('Independent native audit',stamp,len(d),'actual possessions',flush=True)
        del train,w,d,endpoint,vectors;gc.collect()
    independently_paired=pd.concat(reconstructed,ignore_index=True).sort_values(['estimate_date','game_id']).reset_index(drop=True)
    paired=paired.sort_values(['estimate_date','game_id']).reset_index(drop=True)
    for col in ['estimate_date','game_id','rows']:
        np.testing.assert_array_equal(independently_paired[col],paired[col])
    np.testing.assert_allclose(independently_paired.loss_gain_sum,paired.loss_gain_sum,atol=1e-9,rtol=1e-10)
    bootstrap=independent_bootstrap(independently_paired)
    for key,value in bootstrap.items():
        np.testing.assert_allclose(value,summary['paired_gain'][key],atol=1e-12,rtol=0)
    evaluated=sum(r['status']=='verified_actual_endpoints' for r in checks)
    if evaluated!=summary['evaluated_cuts'] or len(checks)-evaluated!=summary['empty_or_unavailable_cuts'] or sum(r['rows'] for r in checks)!=summary['evaluated_rows']:
        raise ValueError('Native summary counts differ')
    result=dict(status='passed',evaluated_cuts=evaluated,verified_empty_cuts=len(checks)-evaluated,
                actual_endpoint_rows=sum(r['rows'] for r in checks),maximum_endpoint_prediction_gap=max_endpoint_gap,
                maximum_scalar_score_gap=max_scalar_gap,paired_games=len(paired),paired_equal_quarter_gain=bootstrap,
                representative_control_normal_equations=normal_checks,normal_equations_are_not_model_refits=True,
                target_source_hashes_verified=len(target_sources),target_artifact_hashes_verified=len(target_artifacts),
                prepared_source_hashes_verified=len(prepare['source_hashes']),source_receipts_unchanged=True,
                protected_outcomes=0,checks=checks,code_sha256=sha(__file__))
    dest=out/'independent';dest.mkdir(exist_ok=True)
    (dest/'audit_quarterly_native_sources.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=Path,default=OUT);p.add_argument('--normal-cuts',default='2014-01-01,2016-01-01,2026-04-01')
    a=p.parse_args();run(a.out,a.root,tuple(a.normal_cuts.split(',')))

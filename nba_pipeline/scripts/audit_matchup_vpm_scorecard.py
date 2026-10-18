"""Independent arithmetic/forecast audit of the private v47 matchup experiment.

Does not import the VPM harness or reuse its score/selection functions. Read
saved player inputs, game predictions, scorecards and affine packets instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def weights(frame):
    n = frame.groupby('season').season.transform('size').to_numpy(float)
    return 1 / n / frame.season.nunique()


def moments(actual, predicted, w):
    y, p, w = np.asarray(actual, float), np.asarray(predicted, float), np.asarray(w, float)
    w = w / w.sum()
    ym, pm = np.sum(w*y), np.sum(w*p)
    vy, vp = np.sum(w*(y-ym)**2), np.sum(w*(p-pm)**2)
    cov = np.sum(w*(y-ym)*(p-pm))
    mse = np.sum(w*(p-y)**2)
    return dict(corr_r2=cov**2/(vy*vp) if vy > 0 and vp > 0 else np.nan,
                forecast_r2=1-mse/vy if vy > 0 else np.nan,
                rmse=np.sqrt(mse), bias=np.sum(w*(p-y)))


def sample(games, teams, years, test):
    veteran = test.endswith('_veterans')
    label = test.removesuffix('_veterans')
    frame = games if label.startswith('game_') else teams
    sub = frame.loc[frame.season.isin(years)].copy()
    if label.endswith('_low_continuity'):
        sub = sub.loc[sub.low_continuity]
    target = 'margin' if label.startswith('game_') else 'actual_win_pct' if label.startswith('team_win') else 'actual_net'
    return sub, target, '_veterans' if veteran else '', label.startswith('team_win')


def audit(root):
    root = Path(root)
    manifest = json.loads((root/'benchmark_manifest.json').read_text())
    settings = manifest['settings']
    games = pd.read_parquet(root/'game_predictions.parquet')
    teams = pd.read_csv(root/'team_predictions.csv')
    inputs = pd.read_parquet(root/'player_rating_inputs.parquet')
    card = pd.read_csv(root/'scorecard.csv')
    pairs = pd.read_csv(root/'paired_comparisons.csv')
    selection = json.loads((root/'selection.json').read_text())
    if games.season.gt(2026).any() or inputs.season.gt(2026).any() or teams.season.gt(2026).any():
        raise ValueError('Protected2027 outcomes present')
    periods = {'headline_2021_2026': settings['seasons'], 'development_2020_2022': [2020,2021,2022],
               'stress_2023_2026': [2023,2024,2025,2026], 'extended_2020_2026': list(range(2020,2027))}
    periods.update({str(y): [y] for y in range(2020,2027)})
    gaps = dict(scorecard=0., paired_point=0., game_from_player=0., team_quality=0., team_win=0.,
                coefficient_qr=0., board_from_affine=0.)

    def check(label, actual, expected, tolerance=1e-9):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise ValueError(label+' missingness differs')
        gap = float(np.max(np.abs(a-b), initial=0.)) if np.isfinite(a-b).all() else float(np.nanmax(np.abs(a-b)))
        if np.isnan(gap):
            gap = 0.
        gaps[label] = max(gaps.get(label,0.),gap)
        if gap > tolerance:
            raise ValueError(f'{label} gap {gap}')

    for row in card.itertuples():
        f, target, suffix, win = sample(games, teams, periods[row.period], row.test)
        result = moments(f[target], f[row.system+suffix], weights(f))
        if win:
            forecast = moments(f[target], f[row.system+suffix+'_win_pct'], weights(f))
            result.update({k:forecast[k] for k in ['forecast_r2','rmse','bias']})
        if row.n != len(f) or row.seasons != f.season.nunique():
            raise ValueError('Scorecard sample mismatch')
        for key, value in result.items():
            check('scorecard', value, getattr(row,key))
    for row in pairs.itertuples():
        f, target, suffix, win = sample(games, teams, [2023,2024,2025,2026], row.test)
        primary = row.primary+suffix
        reference = row.reference+suffix
        quality_a, quality_b = f[primary], f[reference]
        if win:
            primary += '_win_pct'; reference += '_win_pct'
        gain = np.average((f[target]-f[reference])**2-(f[target]-f[primary])**2,weights=weights(f))
        corr = moments(f[target],quality_a,weights(f))['corr_r2']-moments(f[target],quality_b,weights(f))['corr_r2']
        check('paired_point',[gain,corr],[row.mse_gain,row.corr_r2_gain])

    gi = games.set_index('game_id');ti = teams.set_index(['season','team'])
    hca = np.full(len(games), settings['home_advantage_points'], dtype=float)
    bubble = games.season.eq(2020)&pd.to_datetime(games.game_date).ge('2020-07-30')
    hca[bubble.to_numpy()] = settings['neutral_bubble_home_advantage_points']
    for system in manifest['systems']:
        for suffix in ['', '_veterans']:
            name = system+suffix
            if suffix:
                check('veteran_override', inputs.loc[~inputs.veteran,name], settings['replacement'])
            q = inputs.assign(weighted=inputs.minutes*inputs[name]).groupby(['game_id','side']).weighted.sum().unstack()
            q = q.reindex(games.game_id)
            prediction = (q.home.to_numpy()-q.away.to_numpy())/games.duration.to_numpy()*games.pace.to_numpy()/100+hca
            check('game_from_player', prediction, games[name])
            quality = inputs.assign(weighted=inputs.minutes*inputs[name]).groupby(['season','team']).agg(
                numerator=('weighted','sum'), minutes=('minutes','sum'))
            quality['q'] = quality.numerator/(quality.minutes/5)
            quality['q'] -= quality.groupby(level='season').q.transform('mean')
            check('team_quality', quality.q.reindex(ti.index), ti[name])
            prob = 1/(1+np.exp(-settings['win_logistic_k']*prediction))
            a = games[['season','home_team']].rename(columns={'home_team':'team'}).assign(p=prob)
            b = games[['season','away_team']].rename(columns={'away_team':'team'}).assign(p=1-prob)
            expected = pd.concat([a,b],ignore_index=True).groupby(['season','team']).p.mean().reindex(ti.index)
            check('team_win',expected,ti[name+'_win_pct'])

    contract = json.loads((root/'feature_contract.json').read_text())
    packet_rows = []
    for folder in sorted((root/'packets').iterdir()):
        receipt = json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'affine.npz',allow_pickle=False) as saved:
            packet_rows.append((receipt,{k:saved[k].copy() for k in saved.files}))
    for arm in ['selected','DE','simple']:
        for board_path in sorted((root/'boards'/arm/'monthly').glob('*/ratings.parquet')):
            receipt = json.loads(board_path.with_name('manifest.json').read_text())
            cut = pd.Timestamp(receipt['estimate_date'])
            current = next((r,p) for r,p in packet_rows if pd.Timestamp(r['cutoff']) == cut)
            history = [(r,p) for r,p in packet_rows if pd.Timestamp(r['cutoff']) < cut and pd.Timestamp(r['last_game']) < cut]
            if receipt['trained_outcome_seasons'] != [r['season'] for r,p in history]:
                raise ValueError('Coefficient outcome date/census mismatch')
            params = receipt['parameters']; ng = len(contract['columns'])
            wg = np.asarray(params['group_weights'])[contract['feature_groups']]
            if not history or params['strength'] == 0 or not wg.any():
                coef = np.zeros(ng)
            else:
                x = [p['design']*wg/np.sqrt(len(history)*len(p['actual'])) for r,p in history]
                y = [(p['actual']-p['base'])/np.sqrt(len(history)*len(p['actual'])) for r,p in history]
                x.append(np.eye(ng)*np.sqrt(params['ridge']));y.append(np.zeros(ng))
                coef = params['strength']*wg*np.linalg.lstsq(np.vstack(x),np.concatenate(y),rcond=None)[0]
            check('coefficient_qr',coef,receipt['coefficients'])
            r,p = current;beta = p['raw_base']+p['raw_response']@coef;n = len(p['ids'])
            board = pd.read_parquet(board_path)
            raw_o, raw_d = 100*beta[:n], -100*beta[n:]
            o = raw_o-np.average(raw_o,weights=p['off_weights'])
            d = raw_d-np.average(raw_d,weights=p['def_weights'])
            check('board_from_affine',np.c_[o,d,o+d],board[['ovpm','dvpm','vpm']])

    if selection['development_seasons'] != [2020,2021,2022] or selection['differential_evolution']['stress_outcomes_used'] != 0:
        raise ValueError('Selection scope changed')
    result = dict(status='passed',scorecard_rows=len(card),paired_point_rows=len(pairs),
                  games=len(games),team_seasons=len(teams),maximum_gaps=gaps,
                  coefficient_check='independent weighted augmented QR least squares',
                  protected_outcomes=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'independent_score_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]/'vpm/vpm/results/harness/vpm_matchup_defense_v1')
    audit(parser.parse_args().root)

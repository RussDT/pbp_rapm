"""Independent supplemental null-first QR and actual-possession score audit.

Reads only already-generated experiment outputs; never fits or reruns a model.
The actual-point audit independently rebuilds the diagnostic from pinned atoms,
then repeats its frozen paired game bootstrap in the exact registered RNG order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = [f'{side}{i}' for side in ['O','D'] for i in range(1,6)]
ARMS = {'selected':'VPM_matchup_selected','DE':'VPM_matchup_DE',
        'simple':'VPM_matchup_simple','null_first':'VPM_matchup_null_first'}
BASE = 'VPM_long_spm'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(values):
    return values.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)


def check(a,b,label,tolerance=1e-9):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=tolerance,equal_nan=True):
        raise ValueError('Supplemental mismatch: '+label)
    finite=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]),initial=0.))


def training_prefix(packets,current):
    cut=pd.Timestamp(current['cutoff'])
    earlier=[]
    for packet in packets:
        if packet['season']>=current['season'] or packet['season']==2019:
            continue
        if pd.Timestamp(packet['cutoff'])>=cut or pd.Timestamp(packet['last_game'])>=cut:
            raise ValueError('Unfinished null-first training outcome')
        earlier.append(packet)
    return sorted(earlier,key=lambda p:p['season'])


def qr_coefficients(history,params,groups):
    groups=np.asarray(groups,int)
    w=np.asarray(params['group_weights'],float)[groups]
    if params['ridge']<=0 or not np.isfinite(w).all():
        raise ValueError('Invalid frozen coefficient parameters')
    if not history or params['strength']==0 or not w.any():
        return np.zeros(len(groups))
    x,y=[],[]
    for p in history:
        n=len(p['actual']);scale=np.sqrt(len(history)*n)
        if n==0 or p['design'].shape!=(n,len(groups)):
            raise ValueError('Invalid QR training packet')
        x.append(p['design']*w/scale)
        y.append((p['actual']-p['base'])/scale)
    x.append(np.sqrt(params['ridge'])*np.eye(len(groups)))
    y.append(np.zeros(len(groups)))
    # Augmented QR/SVD least squares, independent of the production Gram solve.
    coefficient=np.linalg.lstsq(np.vstack(x),np.concatenate(y),rcond=None)[0]
    return params['strength']*w*coefficient


def aligned(frame,keys):
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate diagnostic key')
    return frame.set_index(keys).sort_index()


def compare_frame(a,b,keys,fields,label):
    a,b=aligned(a,keys),aligned(b,keys)
    if not a.index.equals(b.index):
        raise ValueError('Diagnostic row census mismatch: '+label)
    return max((check(a[f],b[f],label+'/'+f) for f in fields),default=0.)


def point_errors(atoms,boards,intercept):
    lineup=atoms[SLOTS].to_numpy(np.int64)
    if (lineup<=0).any() or (np.diff(np.sort(lineup,axis=1),axis=1)==0).any():
        raise ValueError('Invalid ten-player possession lineup')
    actual=atoms.actual_points.to_numpy(float)
    if not np.isfinite(actual).all() or not np.allclose(actual,np.round(actual),rtol=0,atol=1e-12):
        raise ValueError('Future target is not actual integer points')
    baseline=boards[BASE]
    common=np.isin(lineup,baseline.index.to_numpy()).all(axis=1)
    selected=atoms.loc[common,['game_id']].copy();selected['rows']=1
    for name,board in boards.items():
        indices=board.index.get_indexer(lineup[common].ravel()).reshape((-1,10))
        if (indices<0).any():
            raise ValueError('Diagnostic arm differs from common-native player universe')
        o,d=board.raw_ovpm.to_numpy()/100,board.raw_dvpm.to_numpy()/100
        prediction=intercept+o[indices[:,:5]].sum(axis=1)-d[indices[:,5:]].sum(axis=1)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite actual-point forecast')
        selected[name+'_squared_error']=(actual[common]-prediction)**2
    return selected.groupby('game_id').sum().reset_index(),int(common.sum())


def bootstrap_reference(aggregate,seed,reps):
    rng=np.random.default_rng(seed);rows=[]
    for period,years in [('development_2020_2022',[2020,2021,2022]),
                         ('stress_2023_2026',[2023,2024,2025,2026])]:
        selected=aggregate.loc[aggregate.season.isin(years)]
        if set(selected.season)!=set(years):
            raise ValueError('Missing diagnostic season')
        blocks=[g for _,g in selected.groupby('season')]
        base_mse=float(np.mean([g[BASE+'_squared_error'].sum()/g.rows.sum() for g in blocks]))
        for name in ARMS.values():
            mse=float(np.mean([g[name+'_squared_error'].sum()/g.rows.sum() for g in blocks]))
            gains=[(g[BASE+'_squared_error']-g[name+'_squared_error']).to_numpy() for g in blocks]
            counts=[g.rows.to_numpy() for g in blocks]
            distribution=np.empty(reps)
            # Keep draws/season/arm ordering identical to the frozen diagnostic.
            for draw in range(reps):
                ratios=[]
                for gain,count in zip(gains,counts):
                    sampled=rng.integers(0,len(count),size=len(count))
                    ratios.append(gain[sampled].sum()/count[sampled].sum())
                distribution[draw]=np.mean(ratios)
            low,high=np.quantile(distribution,[.025,.975])
            rows.append(dict(period=period,primary=name,reference=BASE,
                common_native_rows=int(selected.rows.sum()),games=len(selected),mse=mse,
                reference_mse=base_mse,mse_gain=base_mse-mse,
                mse_gain_lo=float(low),mse_gain_hi=float(high),replicates=reps))
    return pd.DataFrame(rows)


def audit(root):
    root=Path(root).resolve()
    required=['benchmark_manifest.json','game_predictions.parquet','selection.json','feature_contract.json',
        'null_first_development_predictions.parquet','future_possession_game_errors.parquet',
        'future_possession_coverage.csv','future_possession_comparisons.csv','input_source_lock.json']
    missing=[name for name in required if not (root/name).exists()]
    if missing:
        raise ValueError('Wait for pipeline-generated outputs; no new scoring permitted: '+str(missing))
    sources={}
    def track(path):
        path=Path(path);sources[str(path.resolve())]=sha(path);return path
    def read_json(path):
        return json.loads(track(path).read_text())
    benchmark=read_json(root/'benchmark_manifest.json');selection=read_json(root/'selection.json')
    contract=read_json(root/'feature_contract.json');locked=read_json(root/'input_source_lock.json')
    if selection['development_seasons']!=[2020,2021,2022] or selection['stress_seasons']!=[2023,2024,2025,2026]:
        raise ValueError('Frozen selection periods changed')
    packets=[]
    for folder in sorted((root/'packets').iterdir()):
        receipt=read_json(folder/'manifest.json')
        if receipt['season']>=2027:
            raise ValueError('Protected packet')
        for name,checksum in receipt['artifact_hashes'].items():
            if sha(track(folder/name))!=checksum:
                raise ValueError('Frozen packet changed')
        with np.load(folder/'affine.npz',allow_pickle=False) as saved:
            packet={k:saved[k].copy() for k in saved.files}
        packet.update({k:receipt[k] for k in ['season','cutoff','last_game','last_training_game']})
        if pd.Timestamp(packet['last_training_game'])>=pd.Timestamp(packet['cutoff']):
            raise ValueError('Baseline training reaches estimate date')
        packets.append(packet)
    gaps={'null_first_qr':0.,'affine_board':0.,'raw_center_identity':0.,'sensitivity_predictions':0.,
          'actual_game_errors':0.,'point_summaries_and_intervals':0.}
    boards_by_year={}
    count=0
    for p in packets:
        if p['season']<2020:
            continue
        cut=pd.Timestamp(p['cutoff']);history=training_prefix(packets,p)
        coefficient=qr_coefficients(history,selection['parameters'],contract['feature_groups'])
        folder=root/'boards/null_first/monthly'/str(cut.date())
        receipt=read_json(folder/'manifest.json');board=pd.read_parquet(track(folder/'ratings.parquet'))
        if sha(folder/'ratings.parquet')!=receipt['ratings_sha256']:
            raise ValueError('Null-first board changed')
        if (receipt['arm']!='null_first' or pd.Timestamp(receipt['estimate_date'])!=cut
                or receipt['source_packet_sha256']!=sha(root/'packets'/str(p['season'])/'manifest.json')
                or receipt['selection_sha256']!=sha(root/'selection.json') or receipt['protected_outcomes']!=0):
            raise ValueError('Null-first board source/date identity changed')
        if receipt['parameters']!=selection['parameters'] or receipt['trained_outcome_seasons']!=[h['season'] for h in history]:
            raise ValueError('Null-first parameters/training prefix changed')
        last=max((h['last_game'] for h in history),default=None)
        if receipt['latest_mapper_training_outcome']!=last:
            raise ValueError('Null-first latest mapper date differs')
        mutated=[{**h,'actual':np.full_like(h['actual'],1e12)} if h['season']>=p['season'] else h for h in packets]
        check(coefficient,qr_coefficients(training_prefix(mutated,p),selection['parameters'],contract['feature_groups']),
              'future outcome coefficient invariance',0.)
        gaps['null_first_qr']=max(gaps['null_first_qr'],check(coefficient,receipt['coefficients'],'null-first coefficient'))
        beta=p['raw_base']+p['raw_response']@coefficient;n=len(p['ids'])
        check(board.player_id,p['ids'],'null-first exact board IDs',0.)
        raw_o,raw_d=100*beta[:n],-100*beta[n:]
        o=raw_o-np.average(raw_o,weights=p['off_weights'])
        d=raw_d-np.average(raw_d,weights=p['def_weights'])
        gaps['affine_board']=max(gaps['affine_board'],check(board[['raw_ovpm','raw_dvpm','raw_vpm','ovpm','dvpm','vpm']],
            np.c_[raw_o,raw_d,raw_o+raw_d,o,d,o+d],'null-first affine board'))
        old=pd.read_parquet(root/'packets'/str(p['season'])/'baseline_board.parquet')
        delta=p['feature_basis']@coefficient
        check(board.matchup_def_prior_increment,delta,'own feature increment')
        check(board.own_prior_def,old.own_prior_def+delta,'own prior update')
        check(board.combined_prior_def,old.combined_prior_def+(1-old.background_weight)*delta,'combined prior update')
        check(receipt['raw_intercept'],p['intercept'],'raw training intercept')
        boards_by_year[p['season']]={BASE:old.set_index('player_id',verify_integrity=True)}
        for arm,name in ARMS.items():
            armfolder=root/'boards'/arm/'monthly'/str(cut.date())
            r=read_json(armfolder/'manifest.json');b=pd.read_parquet(track(armfolder/'ratings.parquet')).set_index('player_id',verify_integrity=True)
            if sha(armfolder/'ratings.parquet')!=r['ratings_sha256']:
                raise ValueError('Diagnostic board changed')
            if pd.Timestamp(r['estimate_date'])!=cut or pd.Timestamp(r['last_training_game'])>=cut:
                raise ValueError('Diagnostic board date fence failed')
            check(r['raw_intercept'],p['intercept'],'common raw intercept')
            offsets=r['reference_offsets']
            gaps['raw_center_identity']=max(gaps['raw_center_identity'],check(
                b[['raw_ovpm','raw_dvpm']],np.c_[b.ovpm+offsets['off'],b.dvpm+offsets['defense']], 'raw/centered O-D'))
            check(r['centered_intercept'],float(p['intercept'])+5*(offsets['off']-offsets['defense'])/100,'centered intercept')
            boards_by_year[p['season']][name]=b
        count+=1
    sensitivity=pd.read_parquet(track(root/'null_first_development_predictions.parquet'))
    if set(sensitivity.candidate)!=set(selection['candidates']) or set(sensitivity.season)!={2020,2021,2022}:
        raise ValueError('Null-first development candidate/outcome census differs')
    for name,params in selection['candidates'].items():
        expected=[]
        for p in packets:
            if p['season'] not in [2020,2021,2022]:
                continue
            coef=qr_coefficients(training_prefix(packets,p),params,contract['feature_groups'])
            expected.append(pd.DataFrame(dict(season=p['season'],game_id=p['game_ids'],actual=p['actual'],
                prediction=p['base']+p['design']@coef)))
        gaps['sensitivity_predictions']=max(gaps['sensitivity_predictions'],compare_frame(
            sensitivity.loc[sensitivity.candidate.eq(name)],pd.concat(expected),['season','game_id'],['actual','prediction'],'sensitivity'))
    games=pd.read_parquet(track(root/'game_predictions.parquet'));games.game_id=ids(games.game_id)
    if games.season.gt(2026).any() or games.duplicated(['season','game_id']).any() or not games.accepted.all():
        raise ValueError('Invalid/protected accepted-game census')
    per_game=[];coverage=[]
    for p in packets:
        season=p['season']
        if season<2020:
            continue
        matching=[path for path in locked if Path(path).name==f'atoms_{season}_RS.parquet']
        if len(matching)!=1:
            raise ValueError('Missing/ambiguous pinned actual-point atoms')
        path=track(matching[0])
        if sha(path)!=locked[matching[0]]:
            raise ValueError('Pinned target atoms changed')
        atoms=pd.read_parquet(path,columns=['game_id','game_date','event_num','actual_points',*SLOTS])
        atoms.game_id=ids(atoms.game_id)
        atoms=atoms.loc[atoms.game_id.isin(games.loc[games.season.eq(season),'game_id'])].copy()
        if atoms.duplicated(['game_id','event_num']).any() or not pd.to_datetime(atoms.game_date).gt(pd.Timestamp(p['cutoff'])).all():
            raise ValueError('Duplicate or unfinished future target row')
        aggregate,common=point_errors(atoms,boards_by_year[season],float(p['intercept']))
        aggregate['season']=season;per_game.append(aggregate)
        coverage.append(dict(season=season,full_rows=len(atoms),common_native_rows=common,common_native_fraction=common/len(atoms)))
    reconstructed=pd.concat(per_game,ignore_index=True)
    saved=pd.read_parquet(track(root/'future_possession_game_errors.parquet'));saved.game_id=ids(saved.game_id)
    gaps['actual_game_errors']=compare_frame(saved,reconstructed,['season','game_id'],
        ['rows',*[name+'_squared_error' for name in [BASE,*ARMS.values()]]],'actual point errors')
    compare_frame(pd.read_csv(track(root/'future_possession_coverage.csv')),pd.DataFrame(coverage),['season'],
                  ['full_rows','common_native_rows','common_native_fraction'],'common native coverage')
    settings=benchmark['settings'];summary=bootstrap_reference(reconstructed,settings['seed'],settings['bootstrap_replicates'])
    stored=pd.read_csv(track(root/'future_possession_comparisons.csv'))
    if stored.selection_used.any():
        raise ValueError('Possession diagnostic unexpectedly used in selection')
    fields=['common_native_rows','games','mse','reference_mse','mse_gain','mse_gain_lo','mse_gain_hi','replicates']
    gaps['point_summaries_and_intervals']=compare_frame(stored,summary,['period','primary','reference'],fields,'point summaries and paired intervals')
    for path,checksum in sources.items():
        if sha(path)!=checksum:
            raise ValueError('Source changed during supplemental audit')
    result=dict(status='passed',null_first_boards=count,sensitivity_rows=len(sensitivity),
        diagnostic_games=len(saved),diagnostic_rows=int(saved.rows.sum()),paired_interval_rows=len(stored),
        bootstrap_seed=settings['seed'],bootstrap_replicates=settings['bootstrap_replicates'],
        maximum_gaps=gaps,protected_outcomes=0,source_hashes=sources,auditor_sha256=sha(__file__),
        scope='supplements frozen root auditor: null_first independent QR/affine/prefix plus future actual-point errors and frozen paired intervals')
    (root/'supplemental_independent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)

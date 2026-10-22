"""Independent read-only audit of completed expanded annual prior fits.

The saved compressed geometry is sufficient to reconstruct all candidate
normal equations, group penalties and independent Net responses. This audit
does not claim to reconstruct possession targets or preprocessing from raw
observations; those source-level checks are explicitly separate. No fitting
job, model write, forecast, selected-pointer change or 2027 read is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def pins(values):
    for name, expected in values.items():
        if not Path(name).is_absolute() or sha(name) != expected:
            raise ValueError('Changed or nonabsolute evidence pin: ' + name)


def maximum(value):
    return float(np.max(np.abs(value), initial=0.))


def close(left, right, tolerance, message):
    gap = maximum(np.asarray(left) - np.asarray(right))
    if not np.isfinite(gap) or gap > tolerance:
        raise ValueError(message + ': ' + str(gap))
    return gap


def cross(left, right):
    """512-row compensated audit product, independent of production1024."""
    result = np.zeros((left.shape[1], right.shape[1])); correction = np.zeros_like(result)
    for start in range(0, len(left), 512):
        value = left[start:start+512].T @ right[start:start+512]
        adjusted = value - correction; updated = result + adjusted
        correction = (updated - result) - adjusted; result = updated
    return result


def expected_precisions(transform, spec, side, indices, conditional=False):
    result = []
    alpha = 100. if conditional and side == 'def' else 10.
    for i in indices:
        meta = transform['basis_metadata'][side][int(i)]
        if meta['kind'] == 'baseline':
            result.append(alpha); continue
        native = meta['native_side'] in [side, 'both']
        multiplier = 10. if meta['kind'] == 'overlap' and native else float(spec['m_'+side])
        if not native: multiplier *= 10.
        if meta['kind'] == 'engineered': multiplier *= 10.
        result.append(alpha * multiplier * max(1., meta['group_size']/10.))
    return np.asarray(result)


def audit_model(model, transform, information, rhs, baseline, old_profile, lane_count):
    if model['expanded_transform'] != transform or digest(model['preprocessors']) != transform['baseline_sha256']:
        raise ValueError('Changed common transform or baseline preprocessing')
    spec = model['candidate_spec']; columns = model['candidate_columns']; family = spec['family']
    widths = {side: len(transform['basis_names'][side]) for side in ['off', 'def']}
    for side in ['off', 'def']:
        ix = np.asarray(columns[side], int)
        if (len(ix) != len(set(ix)) or (ix < 0).any() or (ix >= widths[side]).any()
                or list(ix) != transform['candidate_family_indices'][family][side]):
            raise ValueError('Candidate support differs from common family support')
        meta = transform['basis_metadata'][side]
        if not all(i in ix for i, m in enumerate(meta) if m['kind'] == 'baseline'):
            raise ValueError('Candidate discarded old baseline coordinate')
        for i in ix:
            m = meta[i]; native = m['native_side'] in [side, 'both']
            if (family == 'baseline' and m['kind'] != 'baseline'
                    or family == 'overlap' and m['kind'] != 'baseline' and (m['kind'] != 'overlap' or not native)
                    or family == 'native' and m['kind'] != 'baseline' and (m['kind'] not in ['overlap', 'new'] or not native)
                    or family == 'both' and m['kind'] not in ['baseline', 'overlap', 'new']):
                raise ValueError('Family side/block eligibility mismatch')
    ix = np.r_[columns['off'], widths['off'] + np.asarray(columns['def'])].astype(int)
    p = len(columns['off']); info = information[np.ix_(ix, ix)]; target = rhs[ix]
    jp = np.r_[expected_precisions(transform, spec, 'off', columns['off']),
               expected_precisions(transform, spec, 'def', columns['def'])]
    dp = expected_precisions(transform, spec, 'def', columns['def'], True)
    close(jp, model['fit']['joint_precision'], 0., 'Joint group penalties')
    close(dp, model['fit']['conditional_def_precision'], 0., 'Conditional group penalties')
    normal = info + np.diag(jp); joint = np.linalg.solve(normal, target)
    conditional_rhs = target[p:] - info[p:, :p] @ joint[:p]
    coefficients = joint.copy()
    coefficients[p:] = np.linalg.solve(info[p:, p:] + np.diag(dp), conditional_rhs)
    saved = np.column_stack([model['net_coefficients'], model['coefficients']])
    coefficient_gap = close(coefficients, saved, 1e-9, 'Independent candidate solution')
    normal_gap = max(maximum(normal @ joint - target), maximum((info[p:, p:]+np.diag(dp)) @ saved[p:] - conditional_rhs))
    if normal_gap > 1e-8: raise ValueError('Original absolute normal-equation guard')
    net_gap = close(saved[:, :lane_count].sum(axis=1), saved[:, -1], 1e-9, 'Net versus independently profiled actual total')
    result = dict(dimensions={s:len(columns[s]) for s in columns}, maximum_coefficient_reconstruction_gap=coefficient_gap,
                  maximum_normal_residual=normal_gap, maximum_net_sum_gap=net_gap,
                  joint_offense_then_conditional_defense=True)
    if model['name'] == 'clean_base':
        result['baseline_parity'] = dict(information=close(info, old_profile['information'], 1e-8, 'C0 information'),
            rhs=close(target[:, -1], old_profile['rhs'], 1e-8, 'C0 RHS'),
            total_coefficients=close(saved[:, -1], baseline['coefficients'], 1e-8, 'C0 total coefficients'),
            net_coefficients=close(saved[:, :-1], baseline['net_coefficients'], 1e-8, 'C0 Net coefficients'))
    return result


def audit_cut(root, label, baseline_root):
    import pandas as pd
    root, baseline_root = Path(root).resolve(), Path(baseline_root).resolve()
    if str(label) not in [str(y) for y in range(2020, 2027)] + ['current']:
        raise ValueError('Completed2020–26/current cut required; no2027')
    folder = root/'calibrations'/str(label); receipt_path = folder/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'passed_expanded_annual_calibration' or receipt['label'] != str(label):
        raise ValueError('Completed exact-cut calibration required')
    pins(receipt['input_hashes']); pins(receipt['output_hashes'])
    registration_path=root/'registration.json'; registration=json.loads(registration_path.read_text())
    pins(registration['input_hashes'])
    transform=json.loads((folder/'transform.json').read_text()); profile=np.load(folder/'union_profile.npz')
    information=profile['information']; rhs=profile['rhs']; ty=profile['ty']
    directions=profile['directions']; response=profile['response_operator']
    n=len(information)
    if (information.shape != (n,n) or rhs.shape[0]!=n or directions.shape != response.shape
            or directions.shape[1]!=n or ty.shape[0]!=len(directions)):
        raise ValueError('Compressed geometry shape mismatch')
    symmetry=close(information,information.T,1e-10,'Profile symmetry')
    rhs_gap=close(cross(response,ty),rhs,1e-8,'Independent adjoint RHS reconstruction')
    oldpath=baseline_root/'calibrations'/str(label)/'static_season_home_h365.json'
    baseline=json.loads(oldpath.read_text()); old_profile=np.load(oldpath.with_name(oldpath.stem+'_profile.npz'))
    count=int(baseline['residual_player_cohort_pairs'])
    if directions.shape[0] < 2*count or np.any(directions[2*count:] != 0):
        raise ValueError('Free nuisance must have zero annual feature direction')
    penalized=directions.copy();penalized[:count]*=2000;penalized[count:2*count]*=4000;penalized[2*count:]=0
    reconstructed=cross(directions-response,penalized);reconstructed=(reconstructed+reconstructed.T)/2
    info_gap=close(reconstructed,information,1e-8,'Independent compressed information reconstruction')
    weights=profile['weights'];games=profile['game_ids'];cohorts=profile['cohort_ids']
    if (len(weights)!=len(games) or len(cohorts)!=len(games) or not np.isfinite(weights).all()
            or (weights<=0).any() or not np.isclose(weights.mean(),1,atol=1e-12)):
        raise ValueError('Common physical target weights/census mismatch')
    cut=pd.Timestamp(baseline['estimate_date'])
    if (pd.Timestamp(baseline['last_training_game'])>=cut or np.max(cohorts)>=cut.year
            or len(set(games))!=baseline['target_games']):
        raise ValueError('Matured training chronology/cohort mismatch')
    findings={}
    for name in sorted(receipt['models']):
        path=folder/(name+'.json');recordpath=folder/(name+'.registration.json')
        model=json.loads(path.read_text());record=json.loads(recordpath.read_text())
        if (record['model_sha256']!=sha(path) or record['candidate_spec_sha256']!=digest(model['candidate_spec'])
                or model['source_hashes']!=receipt['input_hashes'] or model['estimate_date']!=baseline['estimate_date']
                or model['last_training_game']!=baseline['last_training_game'] or model['target_rows']!=len(games)
                or model['residual_player_cohort_pairs']!=count or model['weights']!=baseline['weights']
                or model['nuisance']!=baseline['nuisance']):
            raise ValueError('Per-candidate common inputs/chronology/registration mismatch')
        expected_heads=['rim_freq','rim_conversion','mid_freq','mid_conversion','three_freq',
                        'three_conversion','ft','tov','second_chance','mid_reference_on_rim','mid_reference_on_three']
        if model['net_heads']!=expected_heads:raise ValueError('Physical Net/reference target identity changed')
        findings[name]=audit_model(model,transform,information,rhs,baseline,old_profile,9)
    pins(receipt['input_hashes']);pins(receipt['output_hashes'])
    return dict(status='passed_saved_expanded_annual_math_audit',label=str(label),models=findings,
        target_rows=len(games),target_games=len(set(games)),cohort_player_pairs=count,
        union_dimensions={s:len(transform['basis_names'][s]) for s in ['off','def']},
        common_information_symmetry_gap=symmetry,adjoint_rhs_reconstruction_gap=rhs_gap,
        information_reconstruction_gap=info_gap,training_last_game=baseline['last_training_game'],cut=baseline['estimate_date'],
        source_reconstruction_not_performed=True,forecast_or_model_fit_launched=False,
        caveat='Saved-profile math audit; raw target/preprocessing reconstruction and actual deployment are separate audits.',
        input_hashes={str(receipt_path):sha(receipt_path),str(registration_path):sha(registration_path),
                      str(Path(__file__).resolve()):sha(__file__),str(oldpath):sha(oldpath)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for label in args.labels:
            result=audit_cut(args.root,label,args.baseline_root);path=args.out/(label+'.json')
            payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
            if path.exists() and path.read_text()!=payload:raise ValueError('Immutable audit receipt already differs')
            path.write_text(payload)
            print(json.dumps(dict(label=label,status=result['status'],models=len(result['models']),
                max_net_gap=max(x['maximum_net_sum_gap'] for x in result['models'].values()),receipt_sha256=sha(path))),flush=True)


if __name__=='__main__':main()

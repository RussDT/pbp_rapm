"""Independent saved-fit audit of the new bounded-library experiment.

Reuse the independently implemented normal-equation/C0 audit, adding strict
identity and actual numerical support checks for the new unprojected bank.
Original audits, models and experiments remain immutable. No model is fitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_expanded_annual_prior import audit_cut, sha


def audit_bounds(transform, directions, player_cohort_pairs):
    recipe=transform['recipe']
    if (transform.get('schema')!='expanded_annual_vpm_bounded_features_v1'
            or recipe.get('legacy_projection') is not False
            or recipe.get('numeric_maximum_absolute_coordinate')!=10.
            or recipe.get('numeric_scale_floor')!=.1
            or recipe.get('missing_indicator_scale_floor')!=.1):
        raise ValueError('Exact bounded raw-feature recipe required')
    mathematical_numeric=[];mathematical_missing=[];features=0
    for block in transform['blocks']:
        if block.get('legacy_projection') is not False:
            raise ValueError('Added block contains legacy projection')
        for side in ['off','def']:
            if any(k in block['sides'][side] for k in ['projection','residual_scales']):
                raise ValueError('Legacy fitted projection/residual rescaling remains')
        for f in block['features']:
            clip=np.asarray(f['clip'],float);mean=float(f['mean']);scale=float(f['scale'])
            mm=float(f['missing_mean']);ms=float(f['missing_scale'])
            if (clip.shape!=(2,) or not np.isfinite(np.r_[clip,mean,scale,mm,ms]).all()
                    or clip[0]>clip[1] or not clip[0]-1e-12<=mean<=clip[1]+1e-12
                    or scale<.1 or ms<.1 or not 0<=mm<=1
                    or f['observed_rows']<transform['minimum_observed']):
                raise ValueError('Invalid training-only support/center/scale metadata')
            nb=float(max(abs(clip-mean))/scale);mb=float(max(abs(mm),abs(1-mm))/ms)
            if max(nb,mb)>10+1e-12:raise ValueError('Added coordinate support exceeds10')
            mathematical_numeric.append(nb);mathematical_missing.append(mb);features+=1
    n=player_cohort_pairs;ow=len(transform['basis_names']['off'])
    if directions.shape[1]!=ow+len(transform['basis_names']['def']) or len(directions)<2*n:
        raise ValueError('Training direction dimensions differ')
    observed={};baseline_width={}
    for side,raw in [('off',directions[:n,:ow]*100),('def',-directions[n:2*n,ow:]*100)]:
        meta=transform['basis_metadata'][side];p=sum(m['kind']=='baseline' for m in meta)
        if [m['kind']=='baseline' for m in meta]!=[True]*p+[False]*(len(meta)-p):
            raise ValueError('Original baseline is not an untouched prefix')
        baseline_width[side]=p;observed[side]=float(np.max(abs(raw[:,p:]),initial=0))
        if observed[side]>10+1e-12:raise ValueError('Actual training added coordinate exceeds10')
    return dict(recipe='bounded_raw_added_bank_no_legacy_projection_v1',fitted_feature_records=features,
        legacy_prefix_widths=baseline_width,maximum_mathematical_numeric_support=max(mathematical_numeric,default=0.),
        maximum_mathematical_missing_support=max(mathematical_missing,default=0.),
        actual_training_added_maxabs=observed,no_legacy_projection_or_residual_rescaling=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--labels',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for label in args.labels:
            result=audit_cut(args.root,label,args.baseline_root)
            folder=args.root/'calibrations'/label
            transform=json.loads((folder/'transform.json').read_text())
            with np.load(folder/'union_profile.npz') as profile:
                result['bounded_support']=audit_bounds(transform,profile['directions'],result['cohort_player_pairs'])
            result['status']='passed_saved_bounded_expanded_annual_math_and_support_audit'
            result['input_hashes'][str(Path(__file__).resolve())]=sha(__file__)
            path=args.out/(label+'.json');payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
            if path.exists() and path.read_text()!=payload:raise ValueError('Immutable bounded audit differs')
            path.write_text(payload)
            print(json.dumps(dict(label=label,status=result['status'],models=len(result['models']),
                maximum_added=max(result['bounded_support']['actual_training_added_maxabs'].values()),
                maximum_net=max(x['maximum_net_sum_gap'] for x in result['models'].values()),receipt_sha256=sha(path))),flush=True)


if __name__=='__main__':main()

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_expanded_annual_prior as m


def fixture():
    pre={'off':{'features':['a']},'def':{'features':['b']}}
    transform={'baseline_sha256':m.digest(pre),'basis_names':{'off':['a'],'def':['b']},
        'basis_metadata':{s:[dict(kind='baseline',native_side=s)] for s in ['off','def']},
        'candidate_family_indices':{'baseline':{'off':[0],'def':[0]}}}
    info=np.array([[8.,2.],[2.,5.]])
    rhs=np.array([[2.,3.,5.],[4.,1.,5.]])
    joint=np.linalg.solve(info+np.diag([10.,10.]),rhs)
    beta=joint.copy();beta[1]=(rhs[1]-info[1,0]*joint[0])/(info[1,1]+100.)
    model=dict(name='clean_base',expanded_transform=transform,preprocessors=pre,
        candidate_spec={'family':'baseline'},candidate_columns={'off':[0],'def':[0]},
        fit={'joint_precision':[10.,10.],'conditional_def_precision':[100.]},
        net_coefficients=beta[:,:2].tolist(),coefficients=beta[:,-1].tolist())
    baseline={'coefficients':model['coefficients'],'net_coefficients':model['net_coefficients']}
    return model,transform,info,rhs,baseline,{'information':info,'rhs':rhs[:,-1]}


def test_independent_joint_then_fixed_offense_conditional_normal_equations():
    args=fixture();result=m.audit_model(*args,lane_count=2)
    assert result['maximum_coefficient_reconstruction_gap']<1e-15
    assert result['maximum_net_sum_gap']<1e-15
    assert result['maximum_normal_residual']<1e-14


def test_audit_rejects_penalty_or_support_or_lane_tamper():
    args=fixture();args[0]['fit']['conditional_def_precision']=[10.]
    with pytest.raises(ValueError,match='Conditional group penalties'):m.audit_model(*args,lane_count=2)
    args=fixture();args[0]['candidate_columns']['def']=[0,0]
    with pytest.raises(ValueError,match='support'):m.audit_model(*args,lane_count=2)
    args=fixture();args[0]['net_coefficients'][0][0]+=.01
    with pytest.raises(ValueError,match='solution'):m.audit_model(*args,lane_count=2)


def test_nonbaseline_group_precision_side_size_and_engineered_penalties():
    transform={'basis_metadata':{'off':[
        dict(kind='baseline'),dict(kind='overlap',native_side='off',group_size=30),
        dict(kind='new',native_side='def',group_size=20),
        dict(kind='engineered',native_side='both',group_size=10)]}}
    spec={'m_off':2.}
    np.testing.assert_array_equal(m.expected_precisions(transform,spec,'off',[0,1,2,3]),[10.,300.,400.,200.])


def test_compensated_audit_product_preserves_cross_orientation():
    rng=np.random.default_rng(19);a=rng.normal(size=(1301,4));b=rng.normal(size=(1301,3))
    np.testing.assert_allclose(m.cross(a,b),a.T@b,atol=1e-11,rtol=0)

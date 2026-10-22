import copy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_bounded_expanded_annual_prior import audit_bounds


def fixture():
    f=dict(clip=[0.,2.],mean=1.,scale=.1,missing_mean=.3,missing_scale=.1,observed_rows=25)
    t=dict(schema='expanded_annual_vpm_bounded_features_v1',minimum_observed=25,
        recipe=dict(legacy_projection=False,numeric_maximum_absolute_coordinate=10.,numeric_scale_floor=.1,missing_indicator_scale_floor=.1),
        blocks=[dict(legacy_projection=False,features=[f],sides={'off':{},'def':{}})],
        basis_names={s:['base','added'] for s in ['off','def']},
        basis_metadata={s:[dict(kind='baseline'),dict(kind='overlap')] for s in ['off','def']})
    d=np.array([[100.,.1,0.,0.],[0.,0.,-20.,-.1]])
    return t,d


def test_bounded_audit_keeps_unbounded_baseline_prefix_and_checks_added_only():
    t,d=fixture();result=audit_bounds(t,d,1)
    assert result['actual_training_added_maxabs']=={'off':10.,'def':10.}
    assert result['maximum_mathematical_numeric_support']==10.
    assert result['legacy_prefix_widths']=={'off':1,'def':1}


@pytest.mark.parametrize('change',['schema','projection','scale','actual'])
def test_old_projection_or_unbounded_support_fails(change):
    t,d=fixture()
    if change=='schema':t['schema']='expanded_annual_vpm_features_v1'
    elif change=='projection':t['blocks'][0]['sides']['off']['projection']=[[1.]]
    elif change=='scale':t['blocks'][0]['features'][0]['scale']=.01
    else:d[0,1]=.11
    with pytest.raises(ValueError):audit_bounds(t,d,1)

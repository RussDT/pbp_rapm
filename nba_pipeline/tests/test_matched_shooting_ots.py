"""Guard the same-basis parent comparison and its additive target."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from test_v2_six_shooting_split import sample
from build_matched_shooting_ots import fit_matched


def test_matched_parent_ignores_component_allocation_and_shooter_identity():
 d=sample();d['sq']=[1.,0.,2.];d['make']=[.2,-.1,.3];d['ft']=[.1,.4,0.]
 a=fit_matched(d)
 d['sq']+=.8;d['ft']-=.8;d['shooter']=[5,4,2]
 b=fit_matched(d)
 np.testing.assert_allclose(a.normal_ots,b.normal_ots,atol=1e-12)
 np.testing.assert_array_equal(a.normal_ots_opportunities,[3]*5+[0]*5)
 assert abs(np.average(a.normal_ots,weights=a.normal_ots_opportunities))<1e-12


def test_component_fits_sum_to_original_parent():
 d=sample();d.loc[1,'O5']=11
 d['sq']=[.4,1.8,1.2];d['make']=[-.4,1.2,.8];d['ft']=[0,.3,-.1]
 r=fit_matched(d)
 np.testing.assert_allclose(r[['normal_ots_sq','normal_ots_make','normal_ots_ft']].sum(axis=1),r.normal_ots,atol=1e-12)
 assert r.normal_ots.abs().max()>0

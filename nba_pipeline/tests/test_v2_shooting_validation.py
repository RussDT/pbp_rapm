import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_v2_six_shooting_split import design
from validate_v2_six_shooting_split import usage,control_design
from test_v2_six_shooting_split import sample


def test_centered_role_has_zero_training_exposure_sum():
 X,ids=design(sample());n=len(ids);u=usage(X,n);M=control_design(X,n,u,True)
 np.testing.assert_allclose(np.asarray(M[:,2*n:].sum(axis=0)),0,atol=1e-12)


def test_lineup_only_prediction_does_not_use_observed_shooter():
 d=sample();X,ids=design(d);n=len(ids);u=usage(X,n)
 d2=d.copy();d2['shooter']=[5,4,1];X2,_=design(d2)
 b=np.arange(3*n,dtype=float)/10
 def predict(X):
  O=X[:,:n]+X[:,n:2*n];D=X[:,2*n:]
  return O@(u*b[:n]+(1-u)*b[n:2*n])+D@b[2*n:]
 np.testing.assert_allclose(predict(X),predict(X2))

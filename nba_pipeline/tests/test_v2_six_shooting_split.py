import numpy as np
import pandas as pd
from nba_pipeline.scripts.build_v2_six_shooting_split import design,solve_gram,baseline,TARGETS


def sample():
 d=pd.DataFrame({'shooter':[1,2,3], 'Season':[2024]*3})
 for i in range(1,6):d[f'O{i}']=i;d[f'D{i}']=i+5
 return d


def test_exact_role_partition_each_opportunity():
 X,ids=design(sample());n=len(ids)
 np.testing.assert_array_equal(np.asarray(X[:,:n].sum(axis=1)).ravel(),[1]*3)
 np.testing.assert_array_equal(np.asarray(X[:,n:2*n].sum(axis=1)).ravel(),[4]*3)
 np.testing.assert_array_equal(np.asarray(X[:,2*n:].sum(axis=1)).ravel(),[5]*3)
 for row,pid in enumerate([1,2,3]):
  i=np.where(ids==pid)[0][0]
  assert X[row,i]==1 and X[row,n+i]==0


def test_six_alpha_solver_matches_augmented_ridge():
 rng=np.random.default_rng(34);X=rng.normal(size=(80,12));Y=rng.normal(size=(80,3))
 penalties=np.repeat([30,300,5000],4)
 actual=solve_gram(X.T@X,X.T@Y,4,(30,300))
 expected=np.linalg.lstsq(np.vstack([X,np.diag(np.sqrt(penalties))]),np.vstack([Y,np.zeros((12,3))]),rcond=None)[0]
 np.testing.assert_allclose(actual,expected,atol=1e-11)


def test_summed_loss_quadratic_matches_direct_predictions():
 rng=np.random.default_rng(11);P=rng.normal(size=(3,4,20));y=rng.normal(size=20);weights=np.arange(1,21);weights=weights/weights.sum()
 Z=P*np.sqrt(weights);a=y*np.sqrt(weights)
 linear=np.sum(Z*Z,axis=2)-2*np.einsum('tcn,n->tc',Z,a)
 loss=a@a+linear[0][:,None,None]+linear[1][None,:,None]+linear[2][None,None,:]+2*((Z[0]@Z[1].T)[:,:,None]+(Z[0]@Z[2].T)[:,None,:]+(Z[1]@Z[2].T)[None,:,:])
 for i in range(4):
  for j in range(4):
   for k in range(4):
    assert np.isclose(loss[i,j,k],np.sum(weights*(y-P[0,i]-P[1,j]-P[2,k])**2))


def test_validation_baseline_excludes_test_outcome():
 d=pd.DataFrame({'Season':[2024,2024,2025,2025]});Y=np.array([[1,2,3],[999,999,999],[4,5,6],[888,888,888]])
 b=baseline(d,Y,[0,2]);np.testing.assert_array_equal(b,[[1,2,3],[1,2,3],[4,5,6],[4,5,6]])


def test_decomp_ft_replacement_trip_rules():
 from nba_pipeline.scripts.build_v2_six_shooting_split import ft_baseline_units
 text=pd.Series(['Free Throw 1 of 2','Free Throw 2 of 2','Free Throw 1 of 3','Free Throw 2 of 3','Free Throw 3 of 3','Free Throw 1 of 1','Free Throw Technical','Free Throw Technical 1 of 2','Free Throw'])
 np.testing.assert_allclose(ft_baseline_units(text),[.5,.5,1/3,1/3,1/3,0,0,0,0])


def test_training_ft_targets_ignore_heldout_outcomes_and_close():
 from nba_pipeline.scripts.build_v2_six_shooting_split import training_targets
 d=pd.DataFrame({'Season':[2024]*4,'shooter':[1]*4,'fta':[2,0,2,0],'fga':[0,1,0,1], 'ft_actual':[1,0,2,0], 'fg_points':[0,2,0,3], 'ft_baseline_units':[1,0,1,0], 'ft_baseline':[1.1,0,1.1,0], 'sq':[1.1,1.5,1.1,2], 'make':[0,.5,0,1], 'ft':[.5,0,.5,0]})
 y=training_targets(d,[0,1]);np.testing.assert_allclose(y[:2],[[2,0,-1],[1.5,.5,0]])
 np.testing.assert_allclose(y.sum(axis=1),[1,2,1,3])
 d.loc[2,['ft_actual','ft','sq','ft_baseline']]=[0,999,999,999]
 d.loc[3,'fg_points']=0
 np.testing.assert_allclose(training_targets(d,[0,1])[:2],y[:2])

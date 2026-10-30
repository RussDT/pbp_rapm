import numpy as np
from scipy.sparse import csr_matrix
from nba_pipeline.scripts.repair_ft_decomp_results import ridge_target_corrections


def test_target_delta_matches_full_refits_with_weights_and_unequal_penalties():
    rng=np.random.default_rng(92)
    X=csr_matrix(rng.normal(size=(120,7)))
    old=rng.normal(size=(120,3)); delta=rng.normal(size=(120,3))
    weights=rng.uniform(0.1,1,size=120); penalties=np.array([2,4,2,4,2,4,0.0])
    direct=ridge_target_corrections(X,old+delta,penalties,weights)
    patched=ridge_target_corrections(X,old,penalties,weights)+ridge_target_corrections(X,delta,penalties,weights)
    np.testing.assert_allclose(patched,direct,atol=1e-12,rtol=1e-12)

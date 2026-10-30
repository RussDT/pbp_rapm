import numpy as np
import pandas as pd
from nba_pipeline.scripts.build_sq_shooter_teammate_split import unique_events, centered, fit
from scipy.sparse import csr_matrix


def test_conflicting_provider_keys_never_receive_arbitrary_ev():
    d=pd.DataFrame({'game_id':[1,1,2,2,3], 'event_num':[10]*5,
                    'ev':[1.,1.,1.,1.2,0.8], 'shooter':[4,4,5,5,6]})
    out=unique_events(d)
    assert out.game_id.tolist()==[1,3]


def test_validation_target_uses_training_season_baselines():
    d=pd.DataFrame({'Season':[2024,2024,2025,2025]})
    y=np.array([1.,3.,2.,8.])
    baseline=pd.Series([1.,2.],index=[2024,2025])
    np.testing.assert_allclose(centered(d,y,baseline),[0,2,0,6])


def test_equal_penalty_sq_and_residual_solutions_close_to_actual():
    rng=np.random.default_rng(42)
    X=csr_matrix(rng.normal(size=(100,9)))
    sq=rng.normal(size=100); residual=rng.normal(size=100)
    np.testing.assert_allclose(fit(X,sq)+fit(X,residual),fit(X,sq+residual),atol=1e-7)

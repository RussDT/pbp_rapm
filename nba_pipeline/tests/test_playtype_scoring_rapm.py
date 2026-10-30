"""Scientific invariants for the isolated role-scoring prototype."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_playtype_scoring_rapm import targets, fit, COLS, LANES

def test_unknown_is_not_silently_spacing():
    with pytest.raises(ValueError):targets(pd.Series(['PICK_AND_ROLL_UNKNOWN']),np.array([2.]))

def test_misses_and_exclusive_mapping():
    y=targets(pd.Series(['SELF_CREATED_3','OTHER_3','TRANSITION','CUT_2','OTHER_2']),np.array([3.,0.,2.,2.,0.]))
    assert y[0,0]==3 and y[0,3]==0
    assert y[1].sum()==0
    np.testing.assert_array_equal(y[:,:-1].sum(axis=1),y[:,-1])

def test_independent_parent_and_positive_good_defense_close():
    rng=np.random.default_rng(8)
    frame=pd.DataFrame([rng.choice(np.arange(1,21),10,replace=False) for _ in range(300)],columns=COLS)
    frame['game_id']=np.repeat(np.arange(30),10)
    cats=pd.Series(rng.choice(['SELF_CREATED_2','CUT_2','TRANSITION','OTHER_3','OTHER_2'],300))
    frame['lane']=cats
    y=targets(cats,rng.choice([0.,2.,3.],300));board,audit=fit(frame,y)
    assert audit['normal_equation_residual_max']<1e-8
    for side in ['o','d']:
        np.testing.assert_allclose(board[[side+'_'+k for k in LANES]].sum(axis=1),board[side+'_total_fg'],atol=1e-10)
        np.testing.assert_allclose(np.average(board[side+'_total_fg'],weights=board[side+'_team_fga']),0,atol=1e-10)

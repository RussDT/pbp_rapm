import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from analyze_deflections_playoff_context import phase_pairs, offsets


def test_composition_control_and_prior_only():
    d = pd.DataFrame({'player_id':[1,1,2,2], 'season':[2019]*4, 'season_type':['rs','ps']*2, 'exposure':[100,100,1000,10], 'obs_rate':[.02,.01,.08,.07]})
    pairs = phase_pairs(d)
    future = pd.DataFrame({'season':[2019,2020,2020], 'season_type':['ps','ps','rs']})
    off, coef = offsets(future,pairs)
    np.testing.assert_allclose(off,[0,-.01,0])
    poisoned = pd.concat([pairs,pairs.assign(season=2020,delta=999)])
    np.testing.assert_allclose(offsets(future,poisoned)[0],off)

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
import numpy as np
from analyze_tracking_playoff_quartiles import aggregate_phase

def test_missing_and_unmatched_exposure_are_not_zero_observations():
    d=pd.DataFrame({'player_id':[1]*4,'MIN':[10]*4,'metric':[0,2,np.nan,5],'poss':[20,20,20,np.nan]})
    a=aggregate_phase(d,'metric','poss').loc[1]
    assert a['count']==2
    assert a.exposure==40
    assert a.games==2
    assert a.coverage==.5

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
import numpy as np
from research_rim_blocked_prior import prior_ratio

def test_blocked_history_is_strictly_prior_and_missing_pair_does_not_dilute():
    d=pd.DataFrame({'player_id':[1,1,1],'b':[2.,0.,9.],'a':[10.,0.,10.]})
    p=prior_ratio(d,'b','a',10,.1)
    np.testing.assert_allclose(p,[.1,.15,.15])
    d.loc[2,'b']=0
    np.testing.assert_allclose(prior_ratio(d,'b','a',10,.1),p)

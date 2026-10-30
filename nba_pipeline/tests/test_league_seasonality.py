import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
import numpy as np
from analyze_league_seasonality import fit_fe

def test_player_season_levels_do_not_create_seasonal_effect():
    rows=[]
    for p in range(10):
        for phase in range(7):rows.append(dict(player_id=p,season=2024,phase=phase,weight=1+p+phase,rate=20+p+phase*.5))
    result=fit_fe(pd.DataFrame(rows),'rate',draws=20)
    np.testing.assert_allclose([r[1] for r in result],np.arange(7)*.5,atol=1e-9)

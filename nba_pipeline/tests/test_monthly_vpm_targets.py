import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_monthly_vpm_targets import reprice_before_cut


def test_continuation_uses_only_past_counts_and_reprices_each_phase():
    d=pd.DataFrame(dict(game_date=pd.to_datetime(['2019-01-01','2019-01-03','2019-02-01','2019-01-02']),
        game_id=[1,2,3,4],
        source_season_end_year=[2019]*4,season_phase=['RS','RS','RS','PS'],
        FC_MISS_MID=[1,1,1,1],FC_MISS_THREE=[0,0,0,0],SC_AFTER_MISS_POINTS=[2.,0.,99.,0.],
        mid_gross_residual=[-1.,-1.,-1.,-1.],three_gross_residual=[0.]*4,actual_points=[2.,0.,99.,0.]))
    games=d[['game_id','game_date','source_season_end_year','season_phase']].copy()
    games['mid_misses']=1.;games['mid_points']=d.SC_AFTER_MISS_POINTS
    games['three_misses']=0.;games['three_points']=0.
    got,receipt=reprice_before_cut(d,'2019-02-01',games)
    np.testing.assert_allclose(got.mid_residual,[-.5,-.5,-1.])
    assert len(got)==3
    rs=next(r for r in receipt if r['phase']=='RS')
    assert rs['mid']==dict(misses=2.,continuation_points=2.,kappa=1.)
    changed=d.copy();changed.loc[2,['SC_AFTER_MISS_POINTS','actual_points']]=1e9
    games.loc[2,'mid_points']=1e9
    again,_=reprice_before_cut(changed,'2019-02-01',games)
    pd.testing.assert_frame_equal(got,again)


def test_continuation_retains_raw_misses_without_a_terminal_row():
    d=pd.DataFrame(dict(game_id=[1],game_date=pd.to_datetime(['2019-01-01']),source_season_end_year=[2019],
        season_phase=['RS'],actual_points=[2.],mid_gross_residual=[-1.],three_gross_residual=[0.]))
    games=d[['game_id','game_date','source_season_end_year','season_phase']].copy()
    # One retained terminal row, but two raw first misses: canonical kappa uses both.
    games['mid_misses']=2.;games['mid_points']=2.;games['three_misses']=0.;games['three_points']=0.
    got,_=reprice_before_cut(d,'2019-02-01',games)
    assert got.mid_residual.iloc[0]==-.5

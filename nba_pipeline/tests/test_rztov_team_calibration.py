import pandas as pd
import pytest
from nba_pipeline.scripts.check_rztov_team_calibration import season_split, allowance_error


def test_outer_chronology_excludes_target_and_future():
    train,test=season_split(pd.DataFrame({'season':[2022,2023,2024,2025]}),2024,'chronological')
    assert train.season.tolist()==[2022,2023]
    assert test.season.tolist()==[2024]


def test_loso_excludes_target():
    train,_=season_split(pd.DataFrame({'season':[2023,2024,2025]}),2024,'loso')
    assert train.season.tolist()==[2023,2025]


def test_allowance_difference_holds_tsa_fixed():
    # 20 TSA: estimated 12 self/8 assisted, observed 10/10; one extra potential assist.
    assert allowance_error(20,.4,.5,6,5,.15,.05,.2)==pytest.approx(.4)
    assert allowance_error(20,.5,.5,5,5,.15,.05,.2)==0

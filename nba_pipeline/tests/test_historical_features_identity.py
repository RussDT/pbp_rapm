"""Season-specific official NBA admission, including tiny/PS-only players."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from historical_features_identity import identity_filter,dated_name,filter_tracking


def packet():
    return dict(seasons={'2014':dict(admitted_ids=[203147,10],names={'203147':'Chris Smith','10':'Real Player'}),
        '2022':dict(admitted_ids=[1630492],names={'1630492':'Luca Vildoza'})})


def test_season_specific_identity_keeps_tiny_and_ps_only_players():
    d=pd.DataFrame(dict(season=[2014,2022,2014,2022],player_id=[203147,1630492,847,203147],
        minutes=[0.,.01,10.,10.],off_poss=[0,0,20,20]))
    np.testing.assert_array_equal(identity_filter(d,packet()),[True,True,False,False])


def test_dated_official_names_replace_missing_and_placeholder_names():
    d=pd.DataFrame(dict(season=[2014,2022],player_id=[203147,1630492],player_name=[None,'ID_1630492']))
    result=dated_name(d,packet())
    assert result.player_name.tolist()==['Chris Smith','Luca Vildoza']


def test_unknown_source_season_fails_closed():
    with pytest.raises(ValueError,match='season'):
        identity_filter(pd.DataFrame(dict(season=[2027],player_id=[10])),packet())


def test_tracking_uses_game_source_season_and_removes_quarantined_games_first():
    d=pd.DataFrame(dict(game_id=['1','2','3'],player_id=[1630492,203147,99],game_date=pd.to_datetime(['2022-04-30']*3)))
    game_seasons=pd.DataFrame(dict(game_id=['0000000001','0000000002'],season=[2022,2022]))
    got=filter_tracking(d,game_seasons,packet(),exclude_games={'0000000003'})
    assert got.player_id.tolist()==[1630492]

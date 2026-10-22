"""The isolated calendar adapter may fill only independently identified dates."""
from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import historical_targets_1997_dated as D


def test_calendar_recovers_row_nat_from_exact_same_game_source_date():
    raw=pd.DataFrame(dict(game_id=['1','1','2'],game_date=['1996-11-01',None,'1996-11-02']))
    canonical=pd.DataFrame(dict(game_id=[1,1,2],game_date=['1996-11-01','1996-11-01','1996-11-02']))
    rec=D.prove_calendar(raw,canonical,1997,'RS')
    assert rec['raw_row_dates_missing']==1 and rec['games']==2
    assert rec['calendar']==[dict(GAME_ID=1,season='1996-97',game_date='1996-11-01'),
        dict(GAME_ID=2,season='1996-97',game_date='1996-11-02')]
    assert pd.isna(raw.game_date.iloc[1])


@pytest.mark.parametrize('change',['all_missing','inconsistent','canonical_gap','different_games','canonical_missing'])
def test_calendar_rejects_any_missing_or_disputed_game_identity(change):
    raw=pd.DataFrame(dict(game_id=[1,1],game_date=['1996-11-01',None]))
    canonical=pd.DataFrame(dict(game_id=[1],game_date=['1996-11-01']))
    if change=='all_missing':raw['game_date']=None
    if change=='inconsistent':raw.loc[1,'game_date']='1996-11-02'
    if change=='canonical_gap':canonical.loc[0,'game_date']='1996-11-02'
    if change=='different_games':canonical.loc[0,'game_id']=2
    if change=='canonical_missing':canonical.loc[0,'game_date']=None
    with pytest.raises(ValueError):D.prove_calendar(raw,canonical,1997,'RS')

"""Reproduce the actual canonical finalizer's early-century/calendar failure."""
from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import historical_targets_1997_century as C


@pytest.mark.parametrize('year',[1997,1998,1999])
def test_exact_century_finalizer_fills_local_calendar_and_preserves_geometry(year,monkeypatch):
    original=C.common.normalize_season_end_year
    assert original(year%100)==year+100
    raw=pd.DataFrame([dict(game_id=1,Actual_Diff=2.,game_date=pd.NaT,
        **dict(zip(C.SLOTS,range(1001,1011))))])
    calendar=pd.DataFrame([dict(GAME_ID=1,season=f'{year-1}-{year%100:02d}',game_date=f'{year-1}-11-01')])
    monkeypatch.setattr(C.common,'load_game_dates',lambda:calendar.copy())
    with C.early_century(year):
        assert C.common.normalize_season_end_year(26)==2026
        out=C.common._finalize_df(raw.copy(),'Actual_Diff',year%100)
    assert C.common.normalize_season_end_year is original
    assert out.Season.tolist()==[year] and pd.to_datetime(out.game_date).iloc[0]==pd.Timestamp(f'{year-1}-11-01')
    pd.testing.assert_frame_equal(out[['game_id',*C.SLOTS,'Actual_Diff']],raw[['game_id',*C.SLOTS,'Actual_Diff']],check_dtype=False)


def test_century_binding_restores_even_on_error_and_rejects_other_years():
    original=C.common.normalize_season_end_year
    with pytest.raises(RuntimeError):
        with C.early_century(1997):raise RuntimeError('fixture')
    assert C.common.normalize_season_end_year is original
    with pytest.raises(ValueError):
        with C.early_century(2000):pass

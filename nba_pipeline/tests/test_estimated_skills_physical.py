import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
import pytest
import estimated_skills_physical as m

def test_no_silent_historical_measurement_assumption():
    with pytest.raises(ValueError,match='no historical'):m.attach_physical(pd.DataFrame({'player_id':[1]}))

def test_id_join_missing_and_span_difference(tmp_path,monkeypatch):
    p=tmp_path/'m.json';p.write_text('[{"nba_id":1,"player":"A","height_wo_shoes_in":78,"wingspan_in":84}]');monkeypatch.setattr(m,'SOURCE',p)
    d=m.attach_physical(pd.DataFrame({'player_id':[1,2]}),allow_undated=True)
    assert d.loc[0,'wingspan_minus_height_in']==6
    assert pd.isna(d.loc[1,'wingspan_in'])
    assert not d.physical_historical_availability_verified.any()

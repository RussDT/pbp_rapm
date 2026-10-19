"""Official source admission regressions, including NBA string season IDs."""
import importlib.util
from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('fresh_inputs',Path(__file__).parents[1]/'scripts/fetch_fresh_bam_inputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def frame():
    return pd.DataFrame([dict(SEASON_ID='22017',PLAYER_ID=1,TEAM_ID=10,PLAY_TYPE='Isolation',TYPE_GROUPING='Offensive',PPP=1.05,POSS=4.2,GP=70)])

def test_official_string_season_id_is_valid_but_wrong_year_or_playoffs_is_rejected():
    m.validate(frame(),2018,'Isolation')
    for value in ['22018','42017']:
        d=frame();d['SEASON_ID']=value
        with pytest.raises(ValueError,match='source year'):m.validate(d,2018,'Isolation')

def test_duplicate_player_team_type_is_ambiguous_but_trade_teams_are_distinct():
    d=pd.concat([frame(),frame()],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):m.validate(d,2018,'Isolation')
    d.loc[1,'TEAM_ID']=20;m.validate(d,2018,'Isolation')

def test_wrong_side_type_and_negative_counts_fail_but_measured_zero_is_preserved():
    for field,value in [('TYPE_GROUPING','Defensive'),('PLAY_TYPE','Postup'),('POSS',-1),('GP',0)]:
        d=frame();d[field]=value
        with pytest.raises(ValueError):m.validate(d,2018,'Isolation')
    d=frame();d[['POSS','PPP']]=0;m.validate(d,2018,'Isolation')

import importlib.util
from pathlib import Path
import sys

import pytest

scripts=Path(__file__).parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('fresh_bam_source_audit',scripts/'audit_fresh_bam_sources.py')
m=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(scripts));spec.loader.exec_module(m);sys.path.pop(0)


def payload():
    columns=['SEASON_ID','PLAYER_ID','TEAM_ID','PLAY_TYPE','TYPE_GROUPING','PPP','POSS','GP']
    return dict(parameters=dict(LeagueID='00',SeasonYear='2024-25',SeasonType='Regular Season',
        PerMode='PerGame',PlayerOrTeam='P',PlayType='Isolation',TypeGrouping='offensive'),
        resultSets=[dict(headers=columns,rowSet=[['22024',11,100,'Isolation','Offensive',1.1,3.2,70]])])


def test_actual_response_seasonid_string_and_same_completed_season_params_are_checked():
    assert m.official_table(payload(),2025,'Isolation').SEASON_ID.iloc[0]==22024
    for field,value in [('SeasonYear','2025-26'),('SeasonType','Playoffs'),('PerMode','Totals'),
                        ('PlayerOrTeam','T'),('TypeGrouping','defensive'),('PlayType','Postup')]:
        bad=payload();bad['parameters'][field]=value
        with pytest.raises(ValueError,match='parameter'):
            m.official_table(bad,2025,'Isolation')


@pytest.mark.parametrize('column,value',[(0,'22025'),(1,0),(2,0),(3,'Postup'),(4,'Defensive'),
                                       (5,float('nan')),(5,float('inf')),(5,-1.),(6,-1.),(7,0)])
def test_wrong_row_year_identity_side_or_production_is_rejected(column,value):
    bad=payload();bad['resultSets'][0]['rowSet'][0][column]=value
    with pytest.raises(ValueError):
        m.official_table(bad,2025,'Isolation')


def test_duplicate_player_team_playtype_rejected_but_multiple_historical_teams_explicit():
    bad=payload();bad['resultSets'][0]['rowSet']*=2
    with pytest.raises(ValueError,match='Duplicate'):
        m.official_table(bad,2025,'Isolation')
    changed=bad['resultSets'][0]['rowSet'][1].copy();changed[2]=200
    bad['resultSets'][0]['rowSet'][1]=changed
    assert len(m.official_table(bad,2025,'Isolation'))==2


def test_zero_positive_production_filter_is_deferred_to_literal_creation_cohort():
    source=payload();source['resultSets'][0]['rowSet'][0][5]=0.
    assert len(m.official_table(source,2025,'Isolation'))==1


@pytest.mark.parametrize('statuses',[[],['empty'],['ok','error'],['ok','pending'],['ok','ReadTimeout']])
def test_native_source_gate_requires_every_rs_game_resolved_and_nonempty_year(statuses):
    with pytest.raises(ValueError,match='Unresolved'):
        m.require_complete_statuses(statuses)
    m.require_complete_statuses(['ok','empty'])

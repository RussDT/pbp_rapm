import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

scripts=Path(__file__).parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('fresh_independent',scripts/'audit_fresh_bam_features.py')
m=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(scripts));spec.loader.exec_module(m);sys.path.pop(0)


def test_creation_preserves_literal_unweighted_stints_and_full_off_pair_population():
    rows=pd.DataFrame([
        dict(year=2025,PLAYER_ID=1,TEAM_ID=100,PPP=1.,POSS=2.,GP=10),
        dict(year=2025,PLAYER_ID=1,TEAM_ID=200,PPP=2.,POSS=3.,GP=30),
        dict(year=2025,PLAYER_ID=2,TEAM_ID=100,PPP=1.,POSS=100.,GP=70),
        dict(year=2025,PLAYER_ID=3,TEAM_ID=200,PPP=1.,POSS=3.,GP=70),
        dict(year=2025,PLAYER_ID=4,TEAM_ID=200,PPP=0.,POSS=100.,GP=70)])
    loads,meta=m.creation_reference(rows)
    assert loads.loc[(2025,1)]==8. # GP-weighted load would be5, not literal8.
    assert (2025,4) not in loads.index
    assert meta[2025]['median']==8.
    assert meta[2025]['fifth_percentile']==pytest.approx(3.5)
    assert meta[2025]['cohort_players']==3


def test_own_actual_ts_uses_completed_rs_counts_and_rounds_only_final_rate():
    lines=pd.DataFrame([
        dict(season=2025,season_type='rs',game_id='r1',player_id=1,pts=3.,fga=2.,fta=1.),
        dict(season=2025,season_type='rs',game_id='r2',player_id=1,pts=1.,fga=1.,fta=0.),
        dict(season=2025,season_type='ps',game_id='p1',player_id=1,pts=100.,fga=1.,fta=0.),
        dict(season=2026,season_type='rs',game_id='f1',player_id=1,pts=100.,fga=1.,fta=0.)])
    got=m.actual_ts_reference(lines).loc[(2025,1)]
    assert got.actual_pts==4 and got.actual_fga==3 and got.actual_fta==1
    assert got.source_ts==round(4/(2*3.44),3)
    with pytest.raises(ValueError,match='Ambiguous'):
        m.actual_ts_reference(pd.concat([lines,lines]))


def test_fresh_formula_uses_full_denominator_original_median_branch_and_positive_defense_sign():
    rows=pd.DataFrame([
        dict(year=2025,nba_id_off=1,nba_id_def=3,MATCHUP_FGA=1.,MATCHUP_FTA=0.,MATCHUP_TOV=1.,
            PLAYER_PTS=2.,PARTIAL_POSS=2.,source_ts=.5,median_fallback_load=4.,Offensive_Score=10.,
            ts_verified=True,position='PG',pair_source_counts_valid=True,pair_identity_verified=True),
        dict(year=2025,nba_id_off=2,nba_id_def=3,MATCHUP_FGA=3.,MATCHUP_FTA=0.,MATCHUP_TOV=0.,
            PLAYER_PTS=0.,PARTIAL_POSS=2.,source_ts=.5,median_fallback_load=4.,Offensive_Score=1.,
            ts_verified=True,position=None,pair_source_counts_valid=True,pair_identity_verified=True)])
    got=m.fresh_pair_formula(rows)
    # Effective TS falls below sourceTS after a forced turnover: positive-good.
    expected=-10*(2/(2*2.05)-.5)*4*(2.05/4)*1.3025
    assert got['dated_bam_pair'].iloc[0]==pytest.approx(expected)
    assert got['dated_bam_pair'].iloc[0]>0
    assert got['all_defender_fga'].iloc[0]==4
    assert np.isnan(got['dated_bam_pair'].iloc[1])
    rows.loc[0,'PLAYER_PTS']=0
    assert m.fresh_pair_formula(rows)['literal_adjusted_load'][0]==10.
    rows.loc[0,'pair_identity_verified']=False
    assert not m.fresh_pair_formula(rows)['pair_verified'].any()


def test_future_creation_population_cannot_change_earlier_source_year_prices():
    rows=pd.DataFrame([dict(year=2024,PLAYER_ID=1,PPP=1.,POSS=2.),dict(year=2024,PLAYER_ID=2,PPP=1.,POSS=4.),
                       dict(year=2025,PLAYER_ID=1,PPP=1.,POSS=3.)])
    before=m.creation_reference(rows)[1][2024]
    rows.loc[rows.year.eq(2025),'POSS']=1e12
    assert before==m.creation_reference(rows)[1][2024]

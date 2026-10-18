import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('bam_independent', Path(__file__).parents[1]/'scripts/audit_lagged_bam_features.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_literal_formula_keeps_all_defender_fga_even_when_other_pair_is_unverified():
    rows = pd.DataFrame([
        dict(year=2018,nba_id_off=1,nba_id_def=3,MATCHUP_FGA=1.,MATCHUP_FTA=0.,
             MATCHUP_TOV=1.,PLAYER_PTS=2.,TS_PCT=.5,Offensive_Score=9.,position='PG',
             PARTIAL_POSS=3.,observed_ts=.5,pair_source_counts_valid=True,pair_identity_verified=True),
        dict(year=2018,nba_id_off=2,nba_id_def=3,MATCHUP_FGA=3.,MATCHUP_FTA=0.,
             MATCHUP_TOV=0.,PLAYER_PTS=2.,TS_PCT=.5,Offensive_Score=9.,position='C',
             PARTIAL_POSS=3.,observed_ts=.5,pair_source_counts_valid=True,pair_identity_verified=False)])
    got = m.pair_formula(rows)
    # Raw TS=1>.5 uses the original full-PPP-population median, not load9.
    expected = -10*(2/(2*2.05)-.5)*1.9197*(2.05/4)*1.3025
    assert got['dated_bam_pair'][0] == pytest.approx(expected)
    assert got['all_defender_fga'][0] == 4
    assert np.isnan(got['dated_bam_pair'][1])
    rows.loc[0,'PLAYER_PTS']=0
    assert m.pair_formula(rows)['literal_adjusted_load'][0] == 9


def test_same_period_ts_disagreement_and_unknown_position_fail_closed():
    row = dict(year=2025,nba_id_off=1,nba_id_def=3,MATCHUP_FGA=2.,MATCHUP_FTA=0.,
               MATCHUP_TOV=0.,PLAYER_PTS=2.,TS_PCT=.5,Offensive_Score=1.,position='PF',
               PARTIAL_POSS=4.,observed_ts=.50052,pair_source_counts_valid=True,pair_identity_verified=True)
    assert not m.pair_formula(pd.DataFrame([row]))['pair_verified'][0]
    row.update(observed_ts=.5,position=None)
    assert not m.pair_formula(pd.DataFrame([row]))['pair_verified'][0]


def test_year_normalization_is_exposure_weighted_and_does_not_add_invalid_pairs():
    pairs=pd.DataFrame([
        dict(year=2024,nba_id_def=1,PARTIAL_POSS=1.,pair_verified=True,dated_bam_pair=2.),
        dict(year=2024,nba_id_def=1,PARTIAL_POSS=9.,pair_verified=False,dated_bam_pair=np.nan),
        dict(year=2024,nba_id_def=2,PARTIAL_POSS=3.,pair_verified=True,dated_bam_pair=4.)])
    pairs['source_rs_end']=pd.Timestamp('2024-04-14')
    pairs['position_available_at']=pd.Timestamp('2024-06-18')
    got=m.season_reference(pairs).set_index('player_id')
    assert got.loc[1,'season_raw']==2
    assert got.loc[1,'season_exposure']==1
    assert got.loc[1,'source_coverage']==.1
    assert got.loc[1,'season_z']==pytest.approx(-np.sqrt(3))
    assert got.loc[2,'season_z']==pytest.approx(1/np.sqrt(3))


def snapshot_fixture():
    return pd.DataFrame([dict(year=y,player_id=1,season_raw=2.,season_z=float(y-2020),
        season_exposure=100.,all_source_exposure=200.,source_rs_end=pd.Timestamp(f'{y}-04-14'),
        position_available_at=pd.Timestamp(f'{y}-06-18')) for y in range(2018,2026)])


def test_snapshot_uses_latest_four_calendar_seasons_exact_days_and_one_shrink():
    seasons=snapshot_fixture();ends={y:pd.Timestamp(f'{y}-04-14') for y in range(2018,2026)}
    got=m.snapshot_reference(seasons,['2025-10-01'],ends).iloc[0]
    years=range(2022,2026)
    weights=np.array([100*2**(-(pd.Timestamp('2025-10-01')-ends[y]).days/550) for y in years])
    numerator=np.dot(weights,np.array(list(years))-2020)
    assert got.matchup_dated_bam==pytest.approx(numerator/(weights.sum()+1000))
    assert got.source_seasons==4 and got.latest_source_year==2025
    assert got.source_coverage==.5
    # Do not substitute the latest four available player observations for the
    # latest four completed league seasons when a player's recent seasons lack.
    absent=seasons.loc[seasons.year.ne(2025)]
    assert m.snapshot_reference(absent,['2025-10-01'],ends).iloc[0].source_seasons==3


def test_future_source_changes_cannot_affect_prior_cut_and_publication_gate_is_strict():
    seasons=snapshot_fixture();ends={y:pd.Timestamp(f'{y}-04-14') for y in range(2018,2026)}
    first=m.snapshot_reference(seasons,['2024-10-01'],ends)
    seasons.loc[seasons.year.eq(2025),'season_z']=1e12
    second=m.snapshot_reference(seasons,['2024-10-01'],ends)
    pd.testing.assert_frame_equal(first,second)
    seasons.loc[seasons.year.eq(2024),'position_available_at']=pd.Timestamp('2024-10-01')
    with pytest.raises(ValueError,match='Position publication'):
        m.snapshot_reference(seasons,['2024-10-01'],ends)
    with pytest.raises(ValueError,match='Protected'):
        m.snapshot_reference(seasons,['2026-10-01'],ends)


def test_numeric_check_rejects_equal_shape_bad_values_and_row_census_mismatch():
    with pytest.raises(ValueError,match='mismatch'):
        m.assert_values([1.,np.nan],[1.,0.],'masked pair')
    with pytest.raises(ValueError,match='census'):
        m.compare_frames(pd.DataFrame({'id':[1],'v':[1]}),pd.DataFrame({'id':[2],'v':[1]}),['id'],['v'])

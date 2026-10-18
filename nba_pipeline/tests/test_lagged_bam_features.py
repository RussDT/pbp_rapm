import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/build_lagged_bam_features.py'
spec=importlib.util.spec_from_file_location('lagged_bam_features',PATH)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def fixture():
    annual=pd.DataFrame(dict(year=[2018]*3,playoffs=[0]*3,nba_id_off=[1,1,2],nba_id_def=[10,20,10],
        MATCHUP_FGA=[10.,10.,10.],MATCHUP_TOV=[2.,0.,0.],MATCHUP_FTA=[0.,0.,0.],
        PLAYER_PTS=[12.,8.,2.],PARTIAL_POSS=[100.,100.,50.],TS_PCT=[.5]*3,
        Offensive_Score=[2.,2.,1.],Adjusted_Offensive_Score=[4.,2.,1.],
        BAM_inverted=[.3,2.6,1.],**{'Matchup_TS%':[.6,.4,.1]}))
    native=annual[['year','nba_id_off','nba_id_def','MATCHUP_FGA','MATCHUP_TOV','MATCHUP_FTA','PLAYER_PTS','PARTIAL_POSS']].copy()
    native['pair_source_counts_valid']=True;native['pair_identity_verified']=True
    native.loc[0,'PLAYER_PTS']=11. # Fresh native observations, not archived PTS.
    pos=pd.DataFrame(dict(year=[2018],nba_id=[1],position=['PG'],
                          earliest_use_date=[pd.Timestamp('2018-04-12')]))
    lines=pd.DataFrame(dict(season=[2018,2018],season_type=['rs','rs'],player_id=[1,2],
                            pts=[20.,10.],fga=[20.,10.],fta=[0.,0.]))
    ends=pd.Series({2018:pd.Timestamp('2018-04-11')})
    return native,annual,pos,lines,ends


def test_literal_bam_uses_fresh_counts_tov_nonlinear_load_and_full_denominator():
    native,annual,pos,lines,ends=fixture()
    pairs,_,medians=M.dated_pairs(native,annual,pos,lines,ends)
    row=pairs.loc[pairs.nba_id_off.eq(1)&pairs.nba_id_def.eq(10)].iloc[0]
    expected=-10*(11/(2*(10+1.05*2))-.5)*4*((10+1.05*2)/20)*(1+.3025)
    np.testing.assert_allclose(row.dated_bam_pair,expected,atol=1e-12)
    assert row.PLAYER_PTS==11 and row.all_defender_fga==20
    assert row.literal_adjusted_load==4 and medians=={2018:4.}
    # Unknown attacker2 position contributes no inferred zero BAM, but its FGA
    # remains in the ALL-defender denominator used by verified attacker1.
    unknown=pairs.loc[pairs.nba_id_off.eq(2)].iloc[0]
    assert not unknown.pair_verified and np.isnan(unknown.dated_bam_pair)


def test_median_fallback_is_original_ppp_population_not_paired_player_median():
    native,annual,pos,lines,ends=fixture()
    pairs,_,medians=M.dated_pairs(native,annual,pos,lines,ends)
    assert medians[2018]==4.
    assert medians[2018]!=annual.groupby('nba_id_off').Offensive_Score.first().median()
    assert pairs.loc[pairs.nba_id_def.eq(20),'literal_adjusted_load'].iloc[0]==2.


def test_unverified_ts_or_native_identity_never_produces_finite_pair_bam():
    native,annual,pos,lines,ends=fixture()
    native.loc[0,'pair_identity_verified']=False
    lines.loc[lines.player_id.eq(1),'pts']=19.
    pairs,checks,_=M.dated_pairs(native,annual,pos,lines,ends)
    assert not checks.loc[checks.nba_id_off.eq(1),'ts_verified'].iloc[0]
    assert pairs.dated_bam_pair.isna().all()
    seasons,receipt=M.normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    assert seasons.season_z.isna().all() and seasons.season_raw.isna().all()
    assert receipt[0]['status']=='insufficient_verified_defenders'


def test_source_year_position_is_used_without_bpm_publication_floor():
    native,annual,pos,lines,ends=fixture()
    future=pos.copy();future.year=2025;future.position='C'
    pairs,_,_=M.dated_pairs(native,annual,pd.concat([pos,future]),lines,ends)
    assert pairs.loc[pairs.nba_id_off.eq(1),'dated_boost'].eq(.3025).all()
    assert pairs.loc[pairs.nba_id_off.eq(1),'pair_verified'].all()
    assert M.position_role('SF-PF')=='F' and M.position_role('SG-SF')=='G'
    assert M.position_role('unknown') is None


def test_source_year_normalization_preserves_weighted_zero_mean_unit_variance():
    pairs,_,_=M.dated_pairs(*fixture())
    seasons,_=M.normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    good=seasons.season_exposure.gt(0)
    weights=seasons.loc[good,'season_exposure']
    z=seasons.loc[good,'season_z']
    np.testing.assert_allclose(np.average(z,weights=weights),0.,atol=1e-12)
    np.testing.assert_allclose(np.average(z*z,weights=weights),1.,atol=1e-12)
    row=seasons.loc[seasons.player_id.eq(10)].iloc[0]
    np.testing.assert_allclose(row.source_coverage,100/150)


def test_exact_rs_end_decay_and_one_time_k1000_shrink():
    seasons=pd.DataFrame(dict(year=[2018,2019],player_id=[1,1],season_exposure=[100.,300.],
        all_source_exposure=[200.,300.],season_z=[-1.,2.],
        source_rs_end=pd.to_datetime(['2018-04-11','2019-04-10']),
        position_available_at=pd.to_datetime(['2018-04-12','2019-04-11'])))
    cut=pd.Timestamp('2019-10-01')
    features,provenance=M.lagged_snapshots(seasons,[str(cut.date())])
    decay=np.exp2(-(cut-seasons.source_rs_end).dt.days.to_numpy()/550.)
    denominator=np.sum(decay*seasons.season_exposure)
    numerator=np.sum(decay*seasons.season_exposure*seasons.season_z)
    np.testing.assert_allclose(features.matchup_dated_bam,numerator/(denominator+1000),atol=1e-12)
    np.testing.assert_allclose(features.matchup_exposure,denominator,atol=1e-12)
    assert provenance.last_observation_date.iloc[0]==pd.Timestamp('2019-04-10')
    assert provenance.parameter_fit_end.iloc[0]==pd.Timestamp('2019-04-11')


def test_future_and_fifth_old_season_are_excluded_by_calendar_window():
    years=np.arange(2018,2026)
    seasons=pd.DataFrame(dict(year=years,player_id=1,season_exposure=100.,all_source_exposure=100.,season_z=years-2020.,
        source_rs_end=pd.to_datetime([f'{y}-04-10' for y in years]),
        position_available_at=pd.to_datetime([f'{y}-04-11' for y in years])))
    calendar=seasons.set_index('year').source_rs_end
    before,_=M.lagged_snapshots(seasons,['2022-10-01'],completion_calendar=calendar)
    changed=seasons.copy();changed.loc[~changed.year.between(2019,2022),'season_z']=999999.
    after,_=M.lagged_snapshots(changed,['2022-10-01'],completion_calendar=calendar)
    pd.testing.assert_frame_equal(before,after)
    assert before.source_seasons.iloc[0]==4 and before.latest_source_year.iloc[0]==2022


def test_2026_incomplete_rows_are_removed_before_numeric_validation():
    native,annual,pos,lines,ends=fixture()
    future_native=native.iloc[[0]].copy();future_native.year=2026;future_native.MATCHUP_FGA='bad'
    future_annual=annual.iloc[[0]].copy();future_annual.year=2026;future_annual.TS_PCT='bad'
    before,_,_=M.dated_pairs(native,annual,pos,lines,ends)
    after,_,_=M.dated_pairs(pd.concat([native,future_native]),pd.concat([annual,future_annual]),pos,lines,ends)
    pd.testing.assert_frame_equal(before,after)


def test_late_position_availability_and_2027_emission_are_rejected():
    pairs,_,_=M.dated_pairs(*fixture())
    seasons,_=M.normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    seasons.position_available_at=pd.Timestamp('2018-10-01')
    with pytest.raises(ValueError,match='positions not available'):
        M.lagged_snapshots(seasons,['2018-10-01'])
    with pytest.raises(ValueError,match='2027'):
        M.lagged_snapshots(seasons,['2026-10-01'])


def test_unfinished_or_mismatched_native_source_gate_never_admits_later_years(tmp_path):
    path=tmp_path/'gate.json'
    path.write_text(json.dumps(dict(annual_sha256='expected',verified_years=[2018],
        per_year_details=[dict(year=2018,gate=dict(passed=True,all_games_resolved=True)),
                          dict(year=2019,gate=dict(passed=False,all_games_resolved=False))])))
    verified,pending,_=M.admission(path,'expected')
    assert verified==[2018] and pending==list(range(2019,2026))
    with pytest.raises(ValueError,match='another annual'):
        M.admission(path,'changed')
    receipt=json.loads(path.read_text());receipt['verified_years'].append(2019);path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='conflicts'):
        M.admission(path,'expected')

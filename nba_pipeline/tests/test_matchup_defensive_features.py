import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/build_matchup_defensive_features.py'
spec=importlib.util.spec_from_file_location('matchup_defensive_features',SCRIPT)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def lines_fixture():
    rows=[]
    for gid,day in [('0021700001','2017-10-01'),('0021700002','2017-10-02'),('0021700003','2017-10-03')]:
        for pid in [1,2]:
            rows.append(dict(game_id=gid,game_date=pd.Timestamp(day),player_id=pid,season=2018,season_type='rs',
                off_poss=100.,def_poss=100.,rim_fgm=2.,rim_fga=4.,mid_fgm=1.,mid_fga=2.,
                three_fgm=1.,three_fga=2.,ftm=0.,fta=0.,fga=8.,ast=1.,tov=1.,blk=1.,pf_committed=2.))
    return pd.DataFrame(rows)


def pairs_fixture():
    rows=[]
    for attacker,defender in [(1,2),(2,1)]:
        rows.append(dict(game_id='0021700002',game_date=pd.Timestamp('2017-10-02'),season=2018,season_type='rs',
            offense_player_id=attacker,defense_player_id=defender,source_counts_valid=True,
            partialPossessions=100.,matchupFieldGoalsMade=4.,matchupFieldGoalsAttempted=8.,
            matchupThreePointersMade=1.,matchupThreePointersAttempted=2.,matchupFreeThrowsMade=0.,
            matchupFreeThrowsAttempted=0.,matchupAssists=1.,matchupPotentialAssists=0.,
            matchupTurnovers=1.,matchupBlocks=1.,helpBlocks=0.,shootingFouls=1.,playerPoints=9.))
    return pd.DataFrame(rows)


def test_opponent_prices_exclude_entire_same_day_and_all_future_games():
    lines=lines_fixture();requests=pairs_fixture()
    before=M.previous_day_evidence(lines,requests)
    changed=lines.copy();future=changed.game_date.ge('2017-10-02')
    for column in M.PRICE_COUNTS:
        if column in changed:changed.loc[future,column]=999999.
    changed.loc[future,['rim_fgm','mid_fgm','rim_fga','mid_fga']]=999999.
    after=M.previous_day_evidence(changed,requests)
    pd.testing.assert_frame_equal(before,after)
    np.testing.assert_allclose(before.expected_fg_points,.09,atol=1e-12)
    np.testing.assert_allclose(before.expected_fg2pct,.5,atol=1e-12)
    assert before.player_evidence_day.lt(before.game_date).all()
    assert before.league_evidence_day.lt(before.game_date).all()


def test_same_day_second_game_cannot_update_attacker_or_league_prices():
    lines=lines_fixture();extra=lines.loc[lines.game_date.eq('2017-10-02')].copy()
    extra.game_id='0021700999';extra.rim_fgm=100000.
    before=M.previous_day_evidence(lines,pairs_fixture())
    after=M.previous_day_evidence(pd.concat([lines,extra],ignore_index=True),pairs_fixture())
    pd.testing.assert_frame_equal(before,after)


def test_stale_ft_and_allzero_potential_assists_are_unavailable():
    lines=lines_fixture();pairs=pairs_fixture()
    lines.loc[lines.game_id.eq('0021700002'),'fta']=10.
    audit=M.coverage_audit(pairs,lines).iloc[0]
    assert audit.ft_structurally_missing and not audit.ft_valid
    assert not audit.potential_assist_valid
    atoms,_,scales=M.defender_game_atoms(pairs,lines)
    features,provenance=M.snapshots(atoms,['2017-10-03'],scales)
    assert features.matchup_ft_points_saved100.isna().all()
    assert features.matchup_points_saved100.isna().all()
    assert features.matchup_potential_assists_allowed100.isna().all()
    assert features.matchup_fg_points_saved100.notna().all()
    p=provenance.loc[provenance.feature.eq('matchup_potential_assists_allowed100')]
    assert p.exposure.eq(0).all() and p.last_observation_date.isna().all()


def test_moderate_fga_overlap_retained_with_tight_sensitivity_flag():
    pairs=pairs_fixture();lines=lines_fixture()
    pairs.matchupFieldGoalsAttempted*=1.10
    audit=M.coverage_audit(pairs,lines).iloc[0]
    assert audit.fg_valid and not audit.fg_tight_valid and audit.fg_overcount
    pairs.matchupFieldGoalsAttempted*=1.30
    assert not M.coverage_audit(pairs,lines).iloc[0].fg_valid


def test_native_defender_aggregation_and_conditional_bam_identity():
    pairs=pairs_fixture();pairs.loc[pairs.defense_player_id.eq(2),'matchupFieldGoalsMade']=3.
    atoms,audit,scales=M.defender_game_atoms(pairs,lines_fixture())
    player=atoms.loc[atoms.game_id.eq('0021700002')].set_index('player_id').loc[2]
    np.testing.assert_allclose(player['matchup_bam__numerator'],2.)
    np.testing.assert_allclose(player['matchup_fg_points_saved100__numerator'],2.)
    np.testing.assert_allclose(player['matchup_bam__denominator'],100.)
    assert audit.assignment_valid.all() and scales['matchup_bam']==100.


def test_missing_metadata_stays_missing_and_current_year_is_never_used():
    pairs=pairs_fixture();pairs.season=2021;pairs.game_date=pd.Timestamp('2020-12-20')
    table=pd.DataFrame(dict(player_id=[1,1,2],season=[2020,2021,2021],
                            bpm=[2.,999.,999.],position=['PG','C','C']))
    result=M.lagged_metadata(pairs,table).set_index('offense_player_id')
    assert result.loc[1,'opponent_bpm']==2.
    assert result.loc[1,'opponent_position']=='PG'
    assert np.isnan(result.loc[2,'opponent_bpm'])
    assert pd.isna(result.loc[2,'opponent_position'])


def test_bpm_knowledge_floor_preserves_position_but_masks_modern_bpm_backcast():
    table=pd.DataFrame(dict(player_id=[1],season=[2017],bpm=[2.],obpm=[3.],position=['PG']))
    result=M.lagged_metadata(pairs_fixture(),table).set_index('offense_player_id')
    assert pd.isna(result.loc[1,'opponent_bpm']) and pd.isna(result.loc[1,'opponent_obpm'])
    assert result.loc[1,'opponent_position']=='PG'


def test_future_source_mutation_cannot_change_snapshot_or_population_shrinkage():
    atoms,_,scales=M.defender_game_atoms(pairs_fixture(),lines_fixture())
    first,prov=M.snapshots(atoms,['2017-10-03'],scales)
    changed=atoms.copy();changed.game_date=pd.Timestamp('2017-10-03');changed.game_id='0021700999'
    for column in changed:
        if column.endswith('__numerator'):changed[column]=9999999.
    later,p2=M.snapshots(pd.concat([atoms,changed],ignore_index=True),['2017-10-03'],scales)
    pd.testing.assert_frame_equal(first,later);pd.testing.assert_frame_equal(prov,p2)
    assert pd.to_datetime(prov.parameter_fit_end).lt(prov.estimate_date).all()
    finite=prov.exposure.gt(0)
    assert pd.to_datetime(prov.loc[finite,'last_observation_date']).lt(prov.loc[finite,'estimate_date']).all()


def test_no_prior_league_evidence_produces_missing_attacker_price():
    request=pairs_fixture();request.game_date=pd.Timestamp('2017-10-01')
    result=M.previous_day_evidence(lines_fixture(),request)
    assert result.expected_total_points.isna().all()
    assert result.league_evidence_day.isna().all()


def test_attacker_prices_match_direct_fixed_td550_count_formula():
    lines=lines_fixture();lines.loc[lines.game_date.eq('2017-10-01'),'game_date']=pd.Timestamp('2016-01-01')
    lines.loc[lines.game_date.eq('2017-10-02'),'game_date']=pd.Timestamp('2017-01-01')
    lines.loc[lines.game_date.eq('2017-10-03'),'game_date']=pd.Timestamp('2018-01-01')
    lines.loc[lines.player_id.eq(1)&lines.game_date.eq('2016-01-01'),'rim_fgm']=20.
    request=pairs_fixture().iloc[[0]].copy();request.game_date=pd.Timestamp('2017-01-02')
    result=M.previous_day_evidence(lines,request).iloc[0]
    d=lines.loc[lines.game_date.lt(request.game_date.iloc[0])].copy()
    w=np.exp2(-(request.game_date.iloc[0]-d.game_date).dt.days.to_numpy()/550.)
    points=2*(d.rim_fgm+d.mid_fgm)+3*d.three_fgm
    league=np.sum(w*points)/np.sum(w*d.off_poss)
    own=d.player_id.eq(1)
    expected=(np.sum(w[own]*points[own])+1000*league)/(np.sum(w[own]*d.loc[own,'off_poss'])+1000)
    np.testing.assert_allclose(result.expected_fg_points,expected,atol=1e-12)


def test_position_baseline_is_prior_season_minute_weighted():
    pairs=pairs_fixture();pairs.season=2021;pairs.game_date=pd.Timestamp('2020-12-20')
    table=pd.DataFrame(dict(player_id=[1,2],season=[2020,2020],bpm=[2.,6.],
                            obpm=[1.,3.],position=['PG','PG'],minutes=[1000.,100.]))
    result=M.lagged_metadata(pairs,table)
    np.testing.assert_allclose(result.opponent_bpm_position_mean,(2*1000+6*100)/1100)


def test_allzero_help_blocks_are_not_treated_as_measured_ability():
    atoms,_,scales=M.defender_game_atoms(pairs_fixture(),lines_fixture())
    features,_=M.snapshots(atoms,['2017-10-03'],scales)
    assert features.matchup_help_blocks100.isna().all()
    assert features.matchup_blocks100.notna().all()


def test_coverage_denominator_keeps_official_games_without_matchup_rows():
    atoms,_,scales=M.defender_game_atoms(pairs_fixture(),lines_fixture())
    features,_=M.snapshots(atoms,['2017-10-03'],scales)
    w1=np.exp2(-2/550.);w2=np.exp2(-1/550.)
    np.testing.assert_allclose(features.matchup_coverage,w2/(w1+w2),atol=1e-12)


def test_unknown_pair_identity_is_removed_without_rejecting_other_pairs():
    pairs=pairs_fixture();pairs['offense_team_id']=[10,20];pairs['defense_team_id']=[20,10]
    context=pd.DataFrame(dict(game_id=['0021700002']*2,player_id=[1,2],team_id=[10,20]))
    pairs.loc[pairs.offense_player_id.eq(1),'offense_team_id']=99
    atoms,audit,scales=M.defender_game_atoms(pairs,lines_fixture(),game_context=context)
    measured=atoms.loc[atoms.assignment_exposure.gt(0)]
    assert measured.player_id.tolist()==[1]
    assert audit.unverified_identity_partial_possessions.iloc[0]==100.


def test_creator_assignment_burden_and_raw_allowed_points_are_explicit():
    pairs=pairs_fixture();pairs.loc[pairs.defense_player_id.eq(2),'matchupFieldGoalsMade']=3.
    atoms,_,scales=M.defender_game_atoms(pairs,lines_fixture())
    row=atoms.loc[atoms.game_id.eq('0021700002')&atoms.player_id.eq(2)].iloc[0]
    expected={'matchup_assignment_assist_load100':1.,'matchup_assignment_usage_load100':9.,
              'matchup_fg_points_allowed100':7.,'matchup_points_allowed100':7.}
    for feature,value in expected.items():
        np.testing.assert_allclose(scales[feature]*row[feature+'__numerator']/row[feature+'__denominator'],value,atol=1e-12)
    np.testing.assert_allclose(row['matchup_fg_points_saved100__numerator']+row['matchup_fg_points_allowed100__numerator'],9.,atol=1e-12)
    lines=lines_fixture();lines.loc[lines.game_id.eq('0021700002'),'fta']=10.
    missing,_,_=M.defender_game_atoms(pairs,lines)
    assert missing.loc[missing.game_id.eq('0021700002'),'matchup_points_allowed100__denominator'].eq(0).all()
    assert missing.loc[missing.game_id.eq('0021700002'),'matchup_fg_points_allowed100__denominator'].gt(0).all()

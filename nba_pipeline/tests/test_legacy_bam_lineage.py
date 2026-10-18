import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("legacy_bam_audit", Path(__file__).parents[1] / "scripts/audit_legacy_bam_lineage.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_orientation_swap_exchanges_values_instead_of_just_named_index_levels():
    index = pd.MultiIndex.from_tuples([(11, 22), (22, 11)], names=["nba_id_off", "nba_id_def"])
    frame = pd.DataFrame({field: [2., 7.] for field in [*m.FIELDS, "GP"]}, index=index)
    assert m.compare(frame, frame)["fields"]["MATCHUP_FGA"]["l1_difference"] == 0
    assert m.compare(frame, frame, True)["fields"]["MATCHUP_FGA"]["l1_difference"] == 10


def test_absent_native_pairs_count_in_absolute_coverage_gap():
    index = pd.MultiIndex.from_tuples([(11, 22), (22, 11)], names=["nba_id_off", "nba_id_def"])
    frame = pd.DataFrame({field: [2., 7.] for field in [*m.FIELDS, "GP"]}, index=index)
    result = m.compare(frame, frame.iloc[[0]])
    assert result["common_pairs"] == 1
    assert result["fields"]["MATCHUP_FGA"]["l1_difference"] == 7


def direction(gap=0, fraction=0):
    return {"native_offense_defense": {"fields": {f: dict(l1_difference=gap, normalized_l1=fraction)
            for f in ["MATCHUP_FGA", "MATCHUP_TOV"]}},
        "reversed_native_ids": {"fields": {f: dict(l1_difference=100.) for f in ["MATCHUP_FGA", "MATCHUP_TOV"]}}}


@pytest.mark.parametrize("unresolved", ["pending", "error", "ReadTimeout"])
def test_unresolved_rs_games_cannot_be_verified(unresolved):
    gate = m.concordance_gate({"ok": 1229, unresolved: 1}, direction())
    assert not gate["passed"] and not gate["all_games_resolved"]


def test_resolved_valid_empty_games_are_reported_without_becoming_zero_actions():
    assert m.concordance_gate({"ok": 1222, "empty": 8}, direction())["passed"]
    assert not m.concordance_gate({"empty": 1230}, None)["passed"]


def test_source_version_tolerance_is_fixed_one_percent_and_needs_correct_direction():
    assert m.concordance_gate({"ok": 100}, direction(5, .009))["passed"]
    assert not m.concordance_gate({"ok": 100}, direction(5, .011))["passed"]
    assert not m.concordance_gate({"ok": 100}, direction(50, .005))["passed"]


def quality_fixture():
    raw = pd.DataFrame([{**{field: 0. for field in m.FIELDS.values()},
        'game_id': '0021700001', 'offense_player_id': 11, 'defense_player_id': 22,
        'offense_team_id': 100, 'defense_team_id': 200,
        'partialPossessions': 2., 'matchupFieldGoalsAttempted': 1.,
        'matchupThreePointersMade': 0., 'matchupFreeThrowsMade': 0.,
        'source_counts_valid': False, 'defender_in_opposing_native_roster': False}])
    lines = pd.DataFrame([dict(game_id='0021700001', player_id=11, off_poss=1, def_poss=0),
                          dict(game_id='0021700001', player_id=22, off_poss=0, def_poss=1)])
    context = pd.DataFrame([dict(game_id='0021700001', player_id=11, team_id=100, opp_team_id=200),
                            dict(game_id='0021700001', player_id=22, team_id=200, opp_team_id=100)])
    return raw, lines, context


def test_defense_only_identity_uses_exact_pbp_not_outer_offense_roster_or_game_flag():
    raw, lines, context = quality_fixture()
    got = m.pair_quality(raw, m.identity_evidence(lines, context))
    assert got.pair_source_counts_valid.all()
    assert got.pair_identity_verified.all()
    lines.loc[lines.player_id.eq(22), 'def_poss'] = 0
    assert not m.pair_quality(raw, m.identity_evidence(lines, context)).pair_identity_verified.any()


def test_pair_checks_made_attempts_three_subset_finite_and_exact_team_identity():
    raw, lines, context = quality_fixture()
    evidence = m.identity_evidence(lines, context)
    for field, value in [('matchupFieldGoalsMade', 2.), ('matchupThreePointersAttempted', 2.),
                         ('matchupFreeThrowsMade', 1.), ('matchupTurnovers', float('nan'))]:
        changed = raw.copy(); changed[field] = value
        assert not m.pair_quality(changed, evidence).pair_source_counts_valid.any()
    context.loc[context.player_id.eq(22), 'team_id'] = 999
    assert not m.pair_quality(raw, m.identity_evidence(lines, context)).pair_identity_verified.any()


def test_aggregate_quality_is_conservative_and_keeps_raw_counts_for_concordance():
    raw, lines, context = quality_fixture()
    raw = pd.concat([raw, raw], ignore_index=True)
    raw.loc[1, 'matchupFieldGoalsMade'] = 2.
    got = m.aggregate_native(m.pair_quality(raw, m.identity_evidence(lines, context)))
    assert got.MATCHUP_FGA.iloc[0] == 2.
    assert got.MATCHUP_FGM.iloc[0] == 2.
    assert got.pair_game_rows.iloc[0] == 2
    assert got.valid_count_rows.iloc[0] == 1
    assert not got.pair_source_counts_valid.iloc[0]
    assert got.pair_identity_verified.iloc[0]


def test_ambiguous_exact_pbp_identity_fails_closed():
    _, lines, context = quality_fixture()
    with pytest.raises(ValueError, match='Ambiguous'):
        m.identity_evidence(pd.concat([lines, lines]), context)

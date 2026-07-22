import pandas as pd

from nba_pipeline.scripts.repair_modern_raw_lineups import (
    AWAY_COLS,
    HOME_COLS,
    NameResolver,
    Substitution,
    actor_requires_on_court,
    collapse_substitutions,
    lineup_issue_mask,
    solve_period,
    sort_pbp_chronologically,
)


def _lineup_row(away, home, **overrides):
    row = {
        "game_id": "0021600712",
        "period": 1,
        "event_num": 1,
        "event_type": 1,
        "minute_game": 0.0,
        "minute_remaining_quarter": 12,
        "seconds_remaining_quarter": 0,
        "player1_id": home[0],
        "player1_team_id": 200,
    }
    row.update(dict(zip(AWAY_COLS, away)))
    row.update(dict(zip(HOME_COLS, home)))
    row.update(overrides)
    return row


def test_clock_first_sort_handles_late_corrected_event_number():
    frame = pd.DataFrame(
        [
            _lineup_row([11, 12, 13, 14, 15], [1, 2, 3, 4, 5], event_num=10, minute_game=1.0),
            _lineup_row([11, 12, 13, 14, 15], [1, 2, 3, 4, 5], event_num=999, minute_game=0.5),
            _lineup_row([11, 12, 13, 14, 15], [1, 2, 3, 4, 5], event_num=11, minute_game=1.0),
        ]
    )

    out = sort_pbp_chronologically(frame)

    assert out["event_num"].tolist() == [999, 10, 11]


def test_lineup_audit_detects_missing_duplicate_and_overlap():
    rows = [
        _lineup_row([11, 12, 13, 14, 15], [1, 2, 3, 4, 5]),
        _lineup_row([11, 12, 13, 14, 15], [1, 2, 3, 4, 4]),
        _lineup_row([11, 12, 13, 14, 15], [1, 2, 3, 4, None]),
        _lineup_row([5, 12, 13, 14, 15], [1, 2, 3, 4, 5]),
    ]

    assert lineup_issue_mask(pd.DataFrame(rows)).tolist() == [False, True, True, True]


def test_corrected_same_clock_substitutions_collapse_to_net_change():
    subs = [
        Substitution(4, 28000.0, 500, outgoing=1, incoming=6),
        Substitution(4, 28000.0, 501, outgoing=2, incoming=2),
        Substitution(4, 28000.0, 502, outgoing=3, incoming=3),
    ]

    assert collapse_substitutions(subs) == [((28000.0, 500), {1}, {6})]


def test_name_resolver_preserves_suffix_and_first_initial_disambiguation():
    resolver = NameResolver()
    resolver.add(1, "Sterling Brown")
    resolver.add(2, "Moses Brown")
    resolver.add(3, "Charlie Brown Jr.")
    resolver.add(4, "Jalen Williams")
    resolver.add(5, "Jaylin Williams")

    assert resolver.resolve("S. Brown", {1, 2, 3}) == 1
    assert resolver.resolve("M. Brown", {1, 2, 3}) == 2
    assert resolver.resolve("Brown Jr.", {1, 2, 3}) == 3
    assert resolver.resolve("Jal. Williams", {4, 5}) == 4
    assert resolver.resolve("Jay. Williams", {4, 5}) == 5


def test_bench_technical_actor_is_not_required_in_lineup():
    row = type(
        "Row",
        (),
        {
            "event_type": 6,
            "home_description": "Stewart T.FOUL (P2.T0)",
            "visitor_description": None,
            "neutral_description": None,
        },
    )()

    assert not actor_requires_on_court(row)


def test_period_solver_uses_actor_and_partial_lineup_evidence():
    rows = pd.DataFrame(
        [
            _lineup_row(
                [11, 12, 13, 14, 15],
                [1, 2, 3, 4, 5],
                event_num=10,
                event_elapsed_tenths=100.0,
                player1_id=5,
            ),
            _lineup_row(
                [11, 12, 13, 14, 15],
                [1, 2, 3, 4, 6],
                event_num=20,
                event_elapsed_tenths=4000.0,
                player1_id=6,
            ),
        ]
    )
    substitutions = [
        Substitution(1, 3000.0, 15, outgoing=5, incoming=6),
    ]
    sides = {player: "home" for player in range(1, 7)}

    solution = solve_period(
        rows,
        "home",
        set(range(1, 7)),
        substitutions,
        sides,
        pd.DataFrame(),
        unresolved_subs=0,
    )

    assert solution.states == [frozenset({1, 2, 3, 4, 5}), frozenset({1, 2, 3, 4, 6})]
    assert solution.penalty == 0

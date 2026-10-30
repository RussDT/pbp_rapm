#!/usr/bin/env python3
"""Build compact WNBA RTSS player-season features for component-prior research.

This is a research-only feature surface.  It uses the canonical 2D markings
tables, maps provider UUIDs through the audited exact-event crosswalk, and
shrinks all rates/averages toward the season league mean.  It does not alter
the frozen live WNBA rating.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTSS_ROOT = ROOT / "downloads" / "rtss-data" / "wnba"
DEFAULT_CROSSWALK = (
    ROOT / "wnba_test" / "Results" / "rtss_feature_audit" / "player_crosswalk.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "wnba_test" / "Results" / "rtss_prior_research" / "player_season_features.csv"
)

VALID_CROSSWALK_STATUSES = {"exact", "high_confidence_consensus"}
# Confirmation-season fence. 2026 is the untouched RTSS holdout, so the default
# feature surface stops at 2025 and no development fit can see 2026 by
# accident. Building 2026 rows requires the explicit
# --include-confirmation-season flag, and those rows are only legitimate as
# scoring input to run_wnba_rtss_confirmation.py, never as fit rows.
DEVELOPMENT_SEASONS = (2023, 2024, 2025)
CONFIRMATION_SEASON = 2026
RIM_REGIONS = {"ra", "restricted area"}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _probability(value: Any) -> float | None:
    """Normalize provider probabilities stored on either a 0-1 or 0-100 scale."""
    parsed = _number(value)
    if parsed is None:
        return None
    probability = parsed / 100.0 if parsed > 1.0 else parsed
    return probability if 0.0 <= probability <= 1.0 else None


def _player(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _add(
    rows: defaultdict[tuple[int, str], Counter[str]],
    season: int,
    player_id: Any,
    field: str,
    value: float = 1.0,
) -> None:
    player = _player(player_id)
    numeric = _number(value)
    if player is not None and numeric is not None:
        rows[(season, player)][field] += numeric


def _season(data: dict[str, Any]) -> int | None:
    for table in ("chances", "possessions", "shots", "touches"):
        for row in data.get(table, []):
            value = _number(row.get("season"))
            if value is not None:
                return int(value)
    return None


def _aggregate_game(
    path_string: str,
    seasons: tuple[int, ...] = DEVELOPMENT_SEASONS,
) -> list[dict[str, Any]]:
    path = Path(path_string)
    data = json.loads(path.read_text())
    season = _season(data)
    if season not in set(seasons):
        return []

    accum: defaultdict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    chance_seconds: dict[str, float] = {}
    for row in data.get("chances", []):
        chance_id = _player(row.get("id"))
        start_clock = _number(row.get("startGameClock"))
        end_clock = _number(row.get("endGameClock"))
        if chance_id is None or start_clock is None or end_clock is None:
            continue
        chance_seconds[chance_id] = max(start_clock - end_clock, 0.0)

    # Chance-player rows give stable opportunity denominators and rebound roles.
    for row in data.get("chance_players", []):
        player_id = row.get("playerId")
        side = "off" if bool(row.get("offense")) else "def"
        _add(accum, season, player_id, f"{side}_chances")
        if side == "off":
            _add(
                accum,
                season,
                player_id,
                "off_chance_seconds",
                chance_seconds.get(str(row.get("chanceId")), 0.0),
            )
        _add(accum, season, player_id, f"{side}_involved", bool(row.get("involved")))
        _add(
            accum,
            season,
            player_id,
            f"{side}_rb_opportunities",
            bool(row.get("rbOpportunity")),
        )
        if side == "off":
            _add(accum, season, player_id, "off_crashes", bool(row.get("crashed")))
        contest_level = str(row.get("contestLevel") or "").lower()
        if side == "def" and contest_level:
            _add(accum, season, player_id, "def_chance_contest_observations")
            _add(
                accum,
                season,
                player_id,
                "def_chance_heavy_contests",
                contest_level == "heavily_contested",
            )

    # qSQ is treated as context for a league-average shooter. qSP includes the
    # shooter, so qSP-qSQ is the provider shooter-adjustment candidate.
    for row in data.get("shots", []):
        shooter = row.get("shooterId")
        three = bool(row.get("three"))
        region = str(row.get("region") or "").lower()
        rim = (not three) and region in RIM_REGIONS
        mid = (not three) and not rim
        zone = "three" if three else ("rim" if rim else "mid")
        _add(accum, season, shooter, "shot_attempts")
        _add(accum, season, shooter, f"{zone}_attempts")
        assist_opportunity = bool(row.get("assistOpp"))
        _add(
            accum,
            season,
            shooter,
            "shot_assist_opp_attempts",
            assist_opportunity,
        )
        _add(
            accum,
            season,
            shooter,
            "self_created_attempts",
            not assist_opportunity,
        )
        _add(
            accum,
            season,
            shooter,
            f"{zone}_assist_opp_attempts",
            assist_opportunity,
        )
        _add(
            accum,
            season,
            shooter,
            f"{zone}_self_created_attempts",
            not assist_opportunity,
        )
        made = bool(row.get("outcome"))
        role = "assist_opp" if assist_opportunity else "self_created"
        _add(
            accum,
            season,
            shooter,
            f"{zone}_{role}_makes",
            made,
        )
        after_pass = row.get("passerId") is not None
        _add(accum, season, shooter, "shot_after_pass_attempts", after_pass)
        _add(
            accum,
            season,
            shooter,
            "shot_after_pass_self_created_attempts",
            after_pass and not assist_opportunity,
        )
        _add(accum, season, shooter, "catch_shoot_attempts", bool(row.get("catchAndShoot")))
        _add(accum, season, shooter, "shot_dribbles_sum", row.get("dribblesBefore"))
        if mid or three:
            value = 3.0 if three else 2.0
            qsq = _probability(row.get("qSQ"))
            qsp = _probability(row.get("qSP"))
            if qsq is not None:
                _add(accum, season, shooter, f"{zone}_qsq_sum", qsq * value)
                _add(accum, season, shooter, f"{zone}_qsq_n")
                _add(
                    accum,
                    season,
                    shooter,
                    f"{zone}_{role}_qsq_sum",
                    qsq * value,
                )
                _add(accum, season, shooter, f"{zone}_{role}_qsq_n")
            if qsp is not None:
                _add(accum, season, shooter, f"{zone}_qsp_sum", qsp * value)
                _add(accum, season, shooter, f"{zone}_qsp_n")
            if qsq is not None and qsp is not None:
                _add(
                    accum,
                    season,
                    shooter,
                    f"{zone}_shooter_adjustment_sum",
                    (qsp - qsq) * value,
                )
                _add(accum, season, shooter, f"{zone}_shooter_adjustment_n")
                _add(
                    accum,
                    season,
                    shooter,
                    f"{zone}_{role}_shooter_adjustment_sum",
                    (qsp - qsq) * value,
                )
                _add(
                    accum,
                    season,
                    shooter,
                    f"{zone}_{role}_shooter_adjustment_n",
                )

        # Closest-defender attribution. This is a provider-specific proximity
        # label, not causal defensive credit; every derived feature stays a
        # research prior candidate.
        defender = row.get("closestDefId")
        if defender is not None:
            defended_value = 3.0 if three else 2.0
            _add(accum, season, defender, "def_shots_all")
            _add(accum, season, defender, f"def_{zone}_shots")
            _add(accum, season, defender, f"def_{zone}_makes", made)
            _add(
                accum,
                season,
                defender,
                f"def_{zone}_points_allowed",
                defended_value * made,
            )
            defended_contest = str(row.get("contestLevel") or "").lower()
            if defended_contest:
                _add(accum, season, defender, f"def_{zone}_contest_observations")
                _add(
                    accum,
                    season,
                    defender,
                    f"def_{zone}_heavy_contests",
                    defended_contest == "heavily_contested",
                )
                _add(
                    accum,
                    season,
                    defender,
                    f"def_{zone}_uncontested",
                    defended_contest == "uncontested",
                )
            defended_distance = _number(row.get("closestDefDist"))
            if defended_distance is not None:
                _add(
                    accum,
                    season,
                    defender,
                    f"def_{zone}_closest_distance_sum",
                    defended_distance,
                )
                _add(accum, season, defender, f"def_{zone}_closest_distance_n")

        if defender is not None and (mid or three):
            value = 3.0 if three else 2.0
            _add(accum, season, defender, "def_mid3_shots")
            _add(
                accum,
                season,
                defender,
                "def_mid3_catch_shoot",
                bool(row.get("catchAndShoot")),
            )
            qsq = _probability(row.get("qSQ"))
            qsp = _probability(row.get("qSP"))
            if qsq is not None:
                _add(accum, season, defender, "def_mid3_qsq_sum", qsq * value)
                _add(accum, season, defender, "def_mid3_qsq_n")
                # Zone-split allowed shot quality: MID and THREE defense are
                # different skills and the pooled mid3 average hides both.
                _add(accum, season, defender, f"def_{zone}_qsq_sum", qsq * value)
                _add(accum, season, defender, f"def_{zone}_qsq_n")
            if qsp is not None:
                _add(accum, season, defender, "def_mid3_qsp_sum", qsp * value)
                _add(accum, season, defender, "def_mid3_qsp_n")
                _add(accum, season, defender, f"def_{zone}_qsp_sum", qsp * value)
                _add(accum, season, defender, f"def_{zone}_qsp_n")
            distance = _number(row.get("closestDefDist"))
            if distance is not None:
                _add(accum, season, defender, "def_closest_distance_sum", distance)
                _add(accum, season, defender, "def_closest_distance_n")

    # Foul markings. On an offensive foul the provider records the offensive
    # player as `foulerId` and the defender who absorbed it as `fouledId`, so
    # offensive fouls drawn are a defensive credit and offensive fouls
    # committed are an offensive debit.
    for row in data.get("fouls", []):
        if row.get("isTeam") or row.get("isTechnical"):
            continue
        fouler = row.get("foulerId")
        fouled = row.get("fouledId")
        offensive_foul = bool(row.get("isOffensive"))
        shooting_foul = bool(row.get("shooting"))
        if offensive_foul:
            _add(accum, season, fouled, "def_off_fouls_drawn")
            _add(accum, season, fouled, "def_charges_drawn", bool(row.get("charge")))
            _add(accum, season, fouler, "off_fouls_committed")
        else:
            _add(accum, season, fouler, "def_fouls_committed")
            _add(accum, season, fouler, "def_shooting_fouls_committed", shooting_foul)
            _add(
                accum,
                season,
                fouler,
                "def_non_shooting_fouls_committed",
                not shooting_foul,
            )
            _add(accum, season, fouled, "off_fouls_drawn")
            _add(accum, season, fouled, "off_shooting_fouls_drawn", shooting_foul)

    for row in data.get("touches", []):
        player_id = row.get("playerId")
        _add(accum, season, player_id, "touches")
        _add(accum, season, player_id, "touch_time_sum", row.get("touchTime"))
        _add(accum, season, player_id, "touch_dribbles_sum", row.get("numDribbles"))
        _add(accum, season, player_id, "touch_double_teams", bool(row.get("doubleTeam")))
        _add(accum, season, player_id, "touch_led_shot", bool(row.get("ledToShot")))

    for row in data.get("passes", []):
        passer = row.get("passerId")
        receiver = row.get("receiverId")
        _add(accum, season, passer, "passes")
        _add(accum, season, passer, "pass_led_shot", bool(row.get("ledToShot")))
        _add(accum, season, passer, "pass_assist_opps", bool(row.get("assistOpp")))
        _add(accum, season, passer, "pass_turnovers", bool(row.get("turnover")))
        _add(accum, season, passer, "pass_reversals", bool(row.get("reversal")))
        _add(accum, season, receiver, "passes_received")
        _add(
            accum,
            season,
            receiver,
            "potential_assists_received",
            bool(row.get("assistOpp")),
        )
        _add(accum, season, row.get("deflectorId"), "pass_deflections")
        for defender in row.get("gamblingDefenderIds") or []:
            _add(accum, season, defender, "pass_gambles")

    for row in data.get("drives", []):
        _add(accum, season, row.get("ballhandlerId"), "drives")
        _add(accum, season, row.get("ballhandlerId"), "drive_blowbys", bool(row.get("blowby")))
        _add(accum, season, row.get("helpDefId"), "drive_help_events")

    for table in ("isolations", "posts"):
        for row in data.get(table, []):
            _add(accum, season, row.get("ballhandlerId"), f"{table}_ballhandler")

    for row in data.get("picks", []):
        screener = row.get("screenerId")
        ballhandler = row.get("ballhandlerId")
        _add(accum, season, screener, "picks_set")
        _add(accum, season, screener, "pick_contact", str(row.get("contact") or "") != "none")
        _add(accum, season, screener, "pick_rolls", str(row.get("rollPop") or "") == "roll")
        _add(accum, season, screener, "pick_pops", str(row.get("rollPop") or "") == "pop")
        _add(accum, season, ballhandler, "picks_used")

    for row in data.get("off_ball_screens", []):
        _add(accum, season, row.get("screenerId"), "offball_screens_set")
        _add(
            accum,
            season,
            row.get("screenerId"),
            "offball_screen_led_shot",
            bool(row.get("ledToShot")),
        )
        _add(accum, season, row.get("cutterId"), "offball_screens_used")

    for row in data.get("handoffs", []):
        _add(accum, season, row.get("setterId"), "handoffs_set")
        _add(accum, season, row.get("receiverId"), "handoffs_received")

    for row in data.get("poss_matchups", []):
        _add(accum, season, row.get("defPlayerId"), "def_matchup_segments")

    for row in data.get("rotations", []):
        _add(accum, season, row.get("defPlayerId"), "def_rotations")

    for row in data.get("closeouts", []):
        defender = row.get("ballhandlerDefId")
        _add(accum, season, defender, "def_closeouts")
        start = _number(row.get("startDistance"))
        end = _number(row.get("endDistance"))
        if start is not None and end is not None:
            _add(accum, season, defender, "def_closeout_distance_closed_sum", start - end)
            _add(accum, season, defender, "def_closeout_distance_closed_n")
        _add(accum, season, defender, "def_closeout_flybys", row.get("flyBySide") is not None)

    for row in data.get("shot_contests", []):
        defender = row.get("contesterId")
        _add(accum, season, defender, "def_contests")
        level = str(row.get("contestLevel") or "").lower()
        _add(accum, season, defender, "def_heavy_contests", level == "heavily_contested")
        _add(accum, season, defender, "def_rearview_contests", bool(row.get("isRearview")))
        distance = _number(row.get("distanceToShooter"))
        if distance is not None:
            _add(accum, season, defender, "def_contest_distance_sum", distance)
            _add(accum, season, defender, "def_contest_distance_n")
        speed = _number(row.get("speedTowardShooter"))
        if speed is not None:
            _add(accum, season, defender, "def_contest_speed_sum", speed)
            _add(accum, season, defender, "def_contest_speed_n")

    return [
        {"season": key[0], "provider_player_id": key[1], **dict(values)}
        for key, values in accum.items()
    ]


def _league_smoothed_rate(
    numerator: pd.Series,
    denominator: pd.Series,
    prior_exposure: float,
    scale: float = 100.0,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    league = numerator.sum() / max(float(denominator.sum()), 1.0)
    return scale * (numerator + prior_exposure * league) / (denominator + prior_exposure)


def _feature_surface(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    for column in raw.columns:
        if column not in {"provider_player_id", "season"}:
            raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)

    for exposure in ("off_chances", "def_chances", "off_chance_seconds"):
        if exposure not in raw:
            raw[exposure] = 0.0
    result = raw[
        [
            "season",
            "provider_player_id",
            "off_chances",
            "def_chances",
            "off_chance_seconds",
        ]
    ].copy()
    rate_specs = {
        "rtss_on_ball_time_pct": (
            "touch_time_sum",
            "off_chance_seconds",
            1800.0,
        ),
        "rtss_touch_rate100": ("touches", "off_chances", 200.0),
        "rtss_pass_rate100": ("passes", "off_chances", 200.0),
        "rtss_pass_received_rate100": ("passes_received", "off_chances", 200.0),
        "rtss_potential_assist_rate100": ("pass_assist_opps", "off_chances", 200.0),
        "rtss_potential_assist_received_rate100": (
            "potential_assists_received",
            "off_chances",
            200.0,
        ),
        "rtss_self_created_shot_rate100": (
            "self_created_attempts",
            "off_chances",
            200.0,
        ),
        "rtss_rim_self_created_rate100": (
            "rim_self_created_attempts",
            "off_chances",
            200.0,
        ),
        "rtss_mid_self_created_rate100": (
            "mid_self_created_attempts",
            "off_chances",
            200.0,
        ),
        "rtss_three_self_created_rate100": (
            "three_self_created_attempts",
            "off_chances",
            200.0,
        ),
        "rtss_drive_rate100": ("drives", "off_chances", 200.0),
        "rtss_iso_post_rate100": ("isolations_ballhandler", "off_chances", 200.0),
        "rtss_picks_set_rate100": ("picks_set", "off_chances", 200.0),
        "rtss_picks_used_rate100": ("picks_used", "off_chances", 200.0),
        "rtss_offball_screen_rate100": ("offball_screens_set", "off_chances", 200.0),
        "rtss_handoff_set_rate100": ("handoffs_set", "off_chances", 200.0),
        "rtss_def_matchup_rate100": ("def_matchup_segments", "def_chances", 250.0),
        "rtss_def_rotation_rate100": ("def_rotations", "def_chances", 250.0),
        "rtss_def_closeout_rate100": ("def_closeouts", "def_chances", 250.0),
        "rtss_def_contest_rate100": ("def_contests", "def_chances", 250.0),
        "rtss_drive_help_rate100": ("drive_help_events", "def_chances", 250.0),
        "rtss_pass_deflection_rate100": ("pass_deflections", "def_chances", 250.0),
        "rtss_off_crash_rate100": ("off_crashes", "off_chances", 200.0),
        "rtss_off_rb_opp_rate100": ("off_rb_opportunities", "off_chances", 200.0),
        "rtss_def_rb_opp_rate100": ("def_rb_opportunities", "def_chances", 250.0),
        # Closest-defender dFGA volume, split by zone.
        "rtss_def_rim_dfga_rate100": ("def_rim_shots", "def_chances", 250.0),
        "rtss_def_mid_dfga_rate100": ("def_mid_shots", "def_chances", 250.0),
        "rtss_def_three_dfga_rate100": ("def_three_shots", "def_chances", 250.0),
        "rtss_def_dfga_rate100": ("def_shots_all", "def_chances", 250.0),
        # Points allowed as nearest defender, by zone.
        "rtss_def_rim_points_allowed_rate100": (
            "def_rim_points_allowed",
            "def_chances",
            250.0,
        ),
        "rtss_def_mid_points_allowed_rate100": (
            "def_mid_points_allowed",
            "def_chances",
            250.0,
        ),
        "rtss_def_three_points_allowed_rate100": (
            "def_three_points_allowed",
            "def_chances",
            250.0,
        ),
        # Foul markings.
        "rtss_def_off_fouls_drawn_rate100": (
            "def_off_fouls_drawn",
            "def_chances",
            250.0,
        ),
        "rtss_def_charges_drawn_rate100": ("def_charges_drawn", "def_chances", 250.0),
        "rtss_def_fouls_committed_rate100": (
            "def_fouls_committed",
            "def_chances",
            250.0,
        ),
        "rtss_def_shooting_fouls_rate100": (
            "def_shooting_fouls_committed",
            "def_chances",
            250.0,
        ),
        "rtss_def_non_shooting_fouls_rate100": (
            "def_non_shooting_fouls_committed",
            "def_chances",
            250.0,
        ),
        "rtss_off_fouls_drawn_rate100": ("off_fouls_drawn", "off_chances", 200.0),
        "rtss_off_shooting_fouls_drawn_rate100": (
            "off_shooting_fouls_drawn",
            "off_chances",
            200.0,
        ),
        "rtss_off_fouls_committed_rate100": (
            "off_fouls_committed",
            "off_chances",
            200.0,
        ),
    }
    share_specs = {
        "rtss_three_share": ("three_attempts", "shot_attempts", 75.0),
        "rtss_mid_share": ("mid_attempts", "shot_attempts", 75.0),
        "rtss_catch_shoot_share": ("catch_shoot_attempts", "shot_attempts", 75.0),
        "rtss_shot_assist_opp_share": (
            "shot_assist_opp_attempts",
            "shot_attempts",
            75.0,
        ),
        "rtss_rim_assist_opp_share": (
            "rim_assist_opp_attempts",
            "rim_attempts",
            50.0,
        ),
        "rtss_mid_assist_opp_share": (
            "mid_assist_opp_attempts",
            "mid_attempts",
            50.0,
        ),
        "rtss_three_assist_opp_share": (
            "three_assist_opp_attempts",
            "three_attempts",
            75.0,
        ),
        "rtss_after_pass_self_created_share": (
            "shot_after_pass_self_created_attempts",
            "shot_after_pass_attempts",
            75.0,
        ),
        "rtss_rim_assist_opp_fg_pct": (
            "rim_assist_opp_makes",
            "rim_assist_opp_attempts",
            50.0,
        ),
        "rtss_rim_self_created_fg_pct": (
            "rim_self_created_makes",
            "rim_self_created_attempts",
            50.0,
        ),
        "rtss_mid_assist_opp_fg_pct": (
            "mid_assist_opp_makes",
            "mid_assist_opp_attempts",
            50.0,
        ),
        "rtss_mid_self_created_fg_pct": (
            "mid_self_created_makes",
            "mid_self_created_attempts",
            50.0,
        ),
        "rtss_three_assist_opp_fg_pct": (
            "three_assist_opp_makes",
            "three_assist_opp_attempts",
            75.0,
        ),
        "rtss_three_self_created_fg_pct": (
            "three_self_created_makes",
            "three_self_created_attempts",
            75.0,
        ),
        # Allowed shooting as nearest defender, split by zone.
        "rtss_def_rim_fg_pct_allowed": ("def_rim_makes", "def_rim_shots", 50.0),
        "rtss_def_mid_fg_pct_allowed": ("def_mid_makes", "def_mid_shots", 50.0),
        "rtss_def_three_fg_pct_allowed": ("def_three_makes", "def_three_shots", 75.0),
        # Contest quality as nearest defender, split by zone.
        "rtss_def_rim_heavy_contest_share": (
            "def_rim_heavy_contests",
            "def_rim_contest_observations",
            50.0,
        ),
        "rtss_def_mid_heavy_contest_share": (
            "def_mid_heavy_contests",
            "def_mid_contest_observations",
            50.0,
        ),
        "rtss_def_three_heavy_contest_share": (
            "def_three_heavy_contests",
            "def_three_contest_observations",
            75.0,
        ),
        "rtss_def_mid_uncontested_share": (
            "def_mid_uncontested",
            "def_mid_contest_observations",
            50.0,
        ),
        "rtss_def_three_uncontested_share": (
            "def_three_uncontested",
            "def_three_contest_observations",
            75.0,
        ),
        "rtss_touch_double_team_share": ("touch_double_teams", "touches", 100.0),
        "rtss_touch_led_shot_share": ("touch_led_shot", "touches", 100.0),
        "rtss_pass_led_shot_share": ("pass_led_shot", "passes", 100.0),
        "rtss_pass_assist_opp_share": ("pass_assist_opps", "passes", 100.0),
        "rtss_pass_turnover_share": ("pass_turnovers", "passes", 100.0),
        "rtss_pass_reversal_share": ("pass_reversals", "passes", 100.0),
        "rtss_drive_blowby_share": ("drive_blowbys", "drives", 40.0),
        "rtss_pick_contact_share": ("pick_contact", "picks_set", 50.0),
        "rtss_pick_roll_share": ("pick_rolls", "picks_set", 50.0),
        "rtss_pick_pop_share": ("pick_pops", "picks_set", 50.0),
        "rtss_offball_screen_led_shot_share": (
            "offball_screen_led_shot",
            "offball_screens_set",
            50.0,
        ),
        "rtss_def_heavy_contest_share": ("def_heavy_contests", "def_contests", 75.0),
        "rtss_def_rearview_share": ("def_rearview_contests", "def_contests", 75.0),
        "rtss_def_mid3_catch_shoot_share": (
            "def_mid3_catch_shoot",
            "def_mid3_shots",
            75.0,
        ),
    }
    average_specs = {
        "rtss_touch_time": ("touch_time_sum", "touches", 100.0),
        "rtss_touch_dribbles": ("touch_dribbles_sum", "touches", 100.0),
        "rtss_mid_qsq_epv": ("mid_qsq_sum", "mid_qsq_n", 75.0),
        "rtss_mid_qsp_epv": ("mid_qsp_sum", "mid_qsp_n", 75.0),
        "rtss_mid_shooter_adjustment_epv": (
            "mid_shooter_adjustment_sum",
            "mid_shooter_adjustment_n",
            75.0,
        ),
        "rtss_three_qsq_epv": ("three_qsq_sum", "three_qsq_n", 100.0),
        "rtss_three_qsp_epv": ("three_qsp_sum", "three_qsp_n", 100.0),
        "rtss_three_shooter_adjustment_epv": (
            "three_shooter_adjustment_sum",
            "three_shooter_adjustment_n",
            100.0,
        ),
        "rtss_mid_assist_opp_qsq_epv": (
            "mid_assist_opp_qsq_sum",
            "mid_assist_opp_qsq_n",
            50.0,
        ),
        "rtss_mid_assist_opp_shooter_adjustment_epv": (
            "mid_assist_opp_shooter_adjustment_sum",
            "mid_assist_opp_shooter_adjustment_n",
            50.0,
        ),
        "rtss_mid_self_created_qsq_epv": (
            "mid_self_created_qsq_sum",
            "mid_self_created_qsq_n",
            50.0,
        ),
        "rtss_mid_self_created_shooter_adjustment_epv": (
            "mid_self_created_shooter_adjustment_sum",
            "mid_self_created_shooter_adjustment_n",
            50.0,
        ),
        "rtss_three_assist_opp_qsq_epv": (
            "three_assist_opp_qsq_sum",
            "three_assist_opp_qsq_n",
            75.0,
        ),
        "rtss_three_assist_opp_shooter_adjustment_epv": (
            "three_assist_opp_shooter_adjustment_sum",
            "three_assist_opp_shooter_adjustment_n",
            75.0,
        ),
        "rtss_three_self_created_qsq_epv": (
            "three_self_created_qsq_sum",
            "three_self_created_qsq_n",
            75.0,
        ),
        "rtss_three_self_created_shooter_adjustment_epv": (
            "three_self_created_shooter_adjustment_sum",
            "three_self_created_shooter_adjustment_n",
            75.0,
        ),
        "rtss_def_mid3_qsq_epv": ("def_mid3_qsq_sum", "def_mid3_qsq_n", 100.0),
        "rtss_def_mid3_qsp_epv": ("def_mid3_qsp_sum", "def_mid3_qsp_n", 100.0),
        # Allowed shot quality split by zone. qSQ is the league-average-shooter
        # context the defender conceded; qSP adds the actual shooter, so the
        # qSQ arm is the defender-attributable candidate and qSP is retained
        # only as a paired benchmark.
        "rtss_def_mid_qsq_epv": ("def_mid_qsq_sum", "def_mid_qsq_n", 75.0),
        "rtss_def_mid_qsp_epv": ("def_mid_qsp_sum", "def_mid_qsp_n", 75.0),
        "rtss_def_three_qsq_epv": ("def_three_qsq_sum", "def_three_qsq_n", 100.0),
        "rtss_def_three_qsp_epv": ("def_three_qsp_sum", "def_three_qsp_n", 100.0),
        "rtss_def_mid_closest_distance": (
            "def_mid_closest_distance_sum",
            "def_mid_closest_distance_n",
            75.0,
        ),
        "rtss_def_three_closest_distance": (
            "def_three_closest_distance_sum",
            "def_three_closest_distance_n",
            100.0,
        ),
        "rtss_def_rim_closest_distance": (
            "def_rim_closest_distance_sum",
            "def_rim_closest_distance_n",
            75.0,
        ),
        "rtss_def_closest_distance": (
            "def_closest_distance_sum",
            "def_closest_distance_n",
            75.0,
        ),
        "rtss_def_contest_distance": (
            "def_contest_distance_sum",
            "def_contest_distance_n",
            75.0,
        ),
        "rtss_def_contest_speed": (
            "def_contest_speed_sum",
            "def_contest_speed_n",
            75.0,
        ),
        "rtss_def_closeout_distance_closed": (
            "def_closeout_distance_closed_sum",
            "def_closeout_distance_closed_n",
            75.0,
        ),
    }

    for season, index in raw.groupby("season").groups.items():
        season_rows = raw.loc[index]
        def values(column: str) -> pd.Series:
            if column in season_rows:
                return season_rows[column]
            return pd.Series(0.0, index=season_rows.index)

        for name, (numerator, denominator, prior) in rate_specs.items():
            result.loc[index, name] = _league_smoothed_rate(
                values(numerator),
                values(denominator),
                prior,
                scale=100.0,
            ).to_numpy()
        for name, (numerator, denominator, prior) in share_specs.items():
            result.loc[index, name] = _league_smoothed_rate(
                values(numerator),
                values(denominator),
                prior,
                scale=1.0,
            ).to_numpy()
        for name, (numerator, denominator, prior) in average_specs.items():
            result.loc[index, name] = _league_smoothed_rate(
                values(numerator),
                values(denominator),
                prior,
                scale=1.0,
            ).to_numpy()

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(
    rtss_root: Path,
    crosswalk_path: Path,
    output_path: Path,
    workers: int,
    seasons: tuple[int, ...] = DEVELOPMENT_SEASONS,
) -> pd.DataFrame:
    paths = sorted(rtss_root.rglob("basketball-markings.json"))
    if not paths:
        raise FileNotFoundError(f"no basketball-markings.json files under {rtss_root}")

    combined: defaultdict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for game_number, records in enumerate(
            pool.map(
                partial(_aggregate_game, seasons=seasons),
                map(str, paths),
                chunksize=2,
            ),
            start=1,
        ):
            for record in records:
                key = (int(record.pop("season")), str(record.pop("provider_player_id")))
                combined[key].update(record)
            if game_number % 50 == 0 or game_number == len(paths):
                print(f"processed {game_number}/{len(paths)} games", flush=True)

    raw = pd.DataFrame(
        [
            {"season": season, "provider_player_id": player_id, **dict(values)}
            for (season, player_id), values in combined.items()
        ]
    ).fillna(0.0)
    surface = _feature_surface(raw)

    crosswalk = pd.read_csv(crosswalk_path)
    crosswalk = crosswalk[crosswalk["status"].isin(VALID_CROSSWALK_STATUSES)].copy()
    crosswalk["nba_id"] = pd.to_numeric(crosswalk["nba_id"], errors="coerce").astype("Int64")
    crosswalk = crosswalk.dropna(subset=["nba_id"]).drop_duplicates("provider_player_id")
    surface = surface.merge(
        crosswalk[["provider_player_id", "nba_id", "player_name", "status"]],
        on="provider_player_id",
        how="inner",
        validate="many_to_one",
    )
    surface["player_id"] = surface["nba_id"].astype(str)

    feature_columns = [column for column in surface if column.startswith("rtss_")]
    exposure_columns = ["off_chances", "def_chances", "off_chance_seconds"]
    # Provider UUIDs should be unique to players, but collapse defensively by
    # exposure-weighting if the audited map ever contains a duplicate identity.
    rows = []
    for (season, player_id), group in surface.groupby(["season", "player_id"], sort=True):
        row: dict[str, Any] = {
            "season": int(season),
            "player_id": str(player_id),
            "player_name": group["player_name"].iloc[0],
            "provider_player_ids": "|".join(sorted(group["provider_player_id"].astype(str))),
            "crosswalk_status": "|".join(sorted(set(group["status"].astype(str)))),
        }
        for exposure in exposure_columns:
            row[exposure] = float(group[exposure].sum())
        for column in feature_columns:
            side_exposure = "def_chances" if column.startswith("rtss_def_") or column.startswith("rtss_drive_help") or column.startswith("rtss_pass_deflection") else "off_chances"
            weights = group[side_exposure].clip(lower=1.0)
            row[column] = float(np.average(group[column], weights=weights))
        rows.append(row)
    final = pd.DataFrame(rows).sort_values(["season", "player_name", "player_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    manifest = {
        "source_root": str(rtss_root),
        "crosswalk": str(crosswalk_path),
        "games": len(paths),
        "rows": len(final),
        "seasons": {
            str(year): int((final["season"] == year).sum()) for year in sorted(final["season"].unique())
        },
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "semantic_contract": {
            "qSQ": "league-average-shooter shot context, per project definition",
            "qSP": "shot probability including shooter, per project definition",
            "q_value_scale": "normalize provider values from either 0-1 or 0-100",
            "shooter_adjustment": "qSP - qSQ, converted to expected points for MID/3 only",
            "on_ball_time_pct": (
                "100 * touch time / exact offensive chance floor time, "
                "league-smoothed with an 1800-second prior"
            ),
        },
        "status": "research_only_not_live",
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtss-root", type=Path, default=DEFAULT_RTSS_ROOT)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--include-confirmation-season",
        action="store_true",
        help=(
            f"also build {CONFIRMATION_SEASON} rows. These are scoring-only "
            "input for the confirmation harness and must never enter a "
            "development fit; write them to a separate --output path."
        ),
    )
    args = parser.parse_args()
    seasons = DEVELOPMENT_SEASONS
    if args.include_confirmation_season:
        seasons = DEVELOPMENT_SEASONS + (CONFIRMATION_SEASON,)
        if args.output.resolve() == DEFAULT_OUTPUT.resolve():
            parser.error(
                "--include-confirmation-season requires an explicit --output "
                "path so the fenced development surface is never overwritten"
            )
    frame = build_features(
        args.rtss_root.resolve(),
        args.crosswalk.resolve(),
        args.output.resolve(),
        max(1, args.workers),
        seasons,
    )
    print(
        f"wrote {len(frame)} player-seasons and "
        f"{sum(column.startswith('rtss_') for column in frame.columns)} features "
        f"to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

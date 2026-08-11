"""Phase 1 WNBA game-day minutes/allocation engine.

Contract
--------
``project_team_minutes(game_logs, team_tricode, as_of_date, exclude_person_ids=None)``
returns one row per available player with columns
``[person_id, player_name, proj_minutes, proj_share, flag]``; ``proj_minutes``
sums to exactly ``200.0`` (one regulation WNBA team-game: 5 slots x 40
minutes). Only games with ``game_date < as_of_date`` are visible — the game
being projected is never visible to its own projection.

Method (constants frozen; any tuning used 2025 only)
----------------------------------------------------
- Roster: players with ``played == 1`` for the franchise within its last
  ``ROSTER_WINDOW_GAMES`` team-games before ``as_of_date``.
- Availability: players whose ``status`` was INACTIVE in — or who were absent
  from the roster rows of — the franchise's most recent game are excluded from
  the default allocation, as are ``exclude_person_ids``. Excluded players
  appear (with 0.0 minutes) only when ``include_excluded=True`` is passed; the
  default frame is the 200-minute allocation over available players only.
- Weights: EWMA of minutes in games the player was ACTIVE for (DNP-CD zeros
  count as observed zeros; INACTIVE games are skipped, not zeros), with
  half-life ``EWMA_HALF_LIFE_GAMES`` measured in that player's active games
  (1.5, selected on the 2025 walk-forward only; spec default was ~3).
- Cap: ``PER_PLAYER_CAP_MINUTES`` per player; overflow is redistributed
  proportionally among uncapped players until feasible, then the allocation is
  corrected to sum to exactly 200.0.
- Season opener (franchise has zero current-season games before
  ``as_of_date``): fall back to the prior-season EWMA for returning players,
  suspend the latest-game availability exclusions (months-stale), and flag
  every row ``season_opener_fallback``. A brand-new franchise with zero
  history raises ``ValueError`` — a uniform allocation needs a roster source
  this panel cannot provide pregame.
- The PHO -> PHX rename (2025) is treated as one franchise.

Follow-up (documented, deliberately NOT built here): RTSS chance/stint
co-occurrence redistribution — reshaping the allocation using which players
actually share the floor — is a Phase 2 extension; the 2026 RTSS mirror is
currently stale behind an AWS reauth, so it must not be wired in yet.

Evaluation CLI
--------------
``.venv/bin/python wnba_test/project_wnba_minutes.py`` walks forward over
every 2025 and 2026 final team-game (skipping and counting each franchise's
first game of each season), projects pregame, and scores mean absolute error
per actual-player slot (players who actually played; missing projections
count as 0 projected) against two identically-fenced baselines:
(a) last-game minutes renormalized to 200, (b) season-to-date average minutes
renormalized to 200. Writes ``Results/gameday/minutes_eval/report.json`` and
``game_level.csv``.

Fences: market-blind (no odds inputs); 2026 is the confirmation season —
constants may be tuned on 2025 only and 2026 is scored with them frozen; no
proprietary ratings; generated eval artifacts stay out of git.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import wnba_gameday_common as common

# --------------------------------------------------------------------------
# Frozen constants (spec defaults; confirmed on 2025 only — see report.json)
# --------------------------------------------------------------------------

ROSTER_WINDOW_GAMES = 10
# Spec default was ~3; the 2025-only walk-forward sweep (0.5-step grid, 1.0-5.0)
# has a clean interior minimum at 1.5 (MAE 5.31 vs 5.40 at 3.0). 2026 was never
# consulted. Roster window and cap moves were within noise, so they stay at spec.
EWMA_HALF_LIFE_GAMES = 1.5
PER_PLAYER_CAP_MINUTES = 38.0
TEAM_MINUTES = 200.0
# Include ACTIVE-but-0-minute (DNP-CD) games as observed zeros in the EWMA.
EWMA_INCLUDE_ACTIVE_ZEROS = True
# Also exclude roster players absent from the most recent game's roster rows
# (they are no longer on the team's game-day roster: waived/traded/etc.).
EXCLUDE_ABSENT_FROM_LATEST = True

# PHO (2022-2024 CDN tricode) and PHX (2025+) are the same franchise.
FRANCHISE_ALIASES = {"PHO": ("PHO", "PHX"), "PHX": ("PHO", "PHX")}

MINUTES_EVAL_DIR = common.GAMEDAY_DIR / "minutes_eval"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

FLAG_OPENER = "season_opener_fallback"
FLAG_CAPPED = "capped"
FLAG_CAP_RELAXED = "cap_relaxed"
FLAG_UNIFORM = "uniform_fallback"


def _franchise_codes(team_tricode: str) -> tuple[str, ...]:
    code = str(team_tricode).strip().upper()
    return FRANCHISE_ALIASES.get(code, (code,))


def _validate_date(as_of_date: str) -> str:
    if not _DATE_RE.match(str(as_of_date)):
        raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")
    return str(as_of_date)


# --------------------------------------------------------------------------
# Roster context (shared by the engine and the eval baselines so all three
# methods see the identical roster, availability filter, and date fence)
# --------------------------------------------------------------------------


def _roster_context(
    game_logs: pd.DataFrame,
    team_tricode: str,
    as_of_date: str,
    exclude_person_ids: set | None,
) -> dict[str, Any]:
    as_of_date = _validate_date(as_of_date)
    codes = _franchise_codes(team_tricode)
    mask = game_logs["team_tricode"].isin(codes) & (game_logs["game_date"] < as_of_date)
    hist = game_logs.loc[mask]
    if hist.empty:
        raise ValueError(
            f"no franchise games before {as_of_date} for {team_tricode}; "
            "a brand-new team has no pregame roster source in this panel"
        )
    hist = hist.sort_values(["game_date", "game_id"], kind="mergesort")
    team_games = hist[["game_id", "game_date", "season"]].drop_duplicates("game_id")
    current_season = int(as_of_date[:4])
    season_opener = int(team_games["season"].iloc[-1]) != current_season

    window_ids = set(team_games["game_id"].tail(ROSTER_WINDOW_GAMES))
    window_rows = hist[hist["game_id"].isin(window_ids)]
    roster = sorted(set(window_rows.loc[window_rows["played"] == 1, "person_id"].astype(int)))
    if not roster:
        raise ValueError(
            f"no player with played=1 in the last {ROSTER_WINDOW_GAMES} "
            f"team-games for {team_tricode} before {as_of_date}"
        )

    latest_gid = team_games["game_id"].iloc[-1]
    latest = hist[hist["game_id"] == latest_gid]
    excluded: dict[int, str] = {}
    manual = {int(p) for p in (exclude_person_ids or set())}
    for pid in roster:
        if pid in manual:
            excluded[pid] = "excluded_manual"
    if not season_opener:
        inactive_latest = set(latest.loc[latest["status"] == "INACTIVE", "person_id"].astype(int))
        present_latest = set(latest["person_id"].astype(int))
        for pid in roster:
            if pid in excluded:
                continue
            if pid in inactive_latest:
                excluded[pid] = "excluded_inactive_latest"
            elif EXCLUDE_ABSENT_FROM_LATEST and pid not in present_latest:
                excluded[pid] = "excluded_absent_latest"
    available = [pid for pid in roster if pid not in excluded]
    if not available:
        raise ValueError(
            f"every roster player is excluded for {team_tricode} as of {as_of_date}; "
            "cannot allocate 200 minutes"
        )
    names = hist.groupby("person_id")["player_name"].last()
    return {
        "hist": hist,
        "team_games": team_games,
        "current_season": current_season,
        "season_opener": season_opener,
        "latest_gid": latest_gid,
        "roster": roster,
        "excluded": excluded,
        "available": available,
        "names": names,
    }


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def _ewma(minutes_oldest_first: np.ndarray, half_life: float) -> float:
    n = len(minutes_oldest_first)
    if n == 0:
        return 0.0
    ages = np.arange(n - 1, -1, -1, dtype=float)  # newest game has age 0
    weights = 0.5 ** (ages / half_life)
    return float(np.dot(weights, minutes_oldest_first) / weights.sum())


def _ewma_weights(ctx: dict[str, Any]) -> pd.Series:
    """EWMA of minutes-when-active per available player (hist is date-sorted)."""
    hist = ctx["hist"]
    if EWMA_INCLUDE_ACTIVE_ZEROS:
        rows = hist[hist["status"] == "ACTIVE"]
    else:
        rows = hist[hist["played"] == 1]
    grouped = rows.groupby("person_id")["minutes"]
    weights = {}
    for pid in ctx["available"]:
        try:
            series = grouped.get_group(pid)
        except KeyError:
            series = hist.loc[(hist["person_id"] == pid) & (hist["played"] == 1), "minutes"]
        weights[pid] = _ewma(series.to_numpy(dtype=float), EWMA_HALF_LIFE_GAMES)
    return pd.Series(weights, dtype=float)


def _reference_weights(ctx: dict[str, Any], method: str) -> pd.Series:
    """Baseline weightings for the eval, on the identical roster context."""
    hist = ctx["hist"]
    if method == "last_game":
        latest = hist[hist["game_id"] == ctx["latest_gid"]]
        by_pid = latest.set_index(latest["person_id"].astype(int))["minutes"]
        return pd.Series(
            {pid: float(by_pid.get(pid, 0.0)) for pid in ctx["available"]}, dtype=float
        )
    if method == "season_avg":
        rows = hist[(hist["season"] == ctx["current_season"]) & (hist["status"] == "ACTIVE")]
        means = rows.groupby("person_id")["minutes"].mean()
        return pd.Series(
            {pid: float(means.get(pid, 0.0)) for pid in ctx["available"]}, dtype=float
        )
    raise ValueError(f"unknown reference method: {method}")


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def _allocate(weights: pd.Series, cap: float | None) -> tuple[pd.Series, dict[int, str]]:
    """Scale nonnegative weights to TEAM_MINUTES with an optional per-player cap.

    Overflow above the cap is redistributed proportionally among uncapped
    players. Returns (allocation, per-player flag dict); the allocation sums
    to exactly TEAM_MINUTES after a final float correction.
    """
    flags: dict[int, str] = {}
    weights = weights.astype(float).clip(lower=0.0)
    n = len(weights)
    if n == 0:
        raise ValueError("cannot allocate minutes to an empty player set")
    if weights.sum() <= 0.0:
        weights = pd.Series(1.0, index=weights.index)
        flags = {int(pid): FLAG_UNIFORM for pid in weights.index}
    if cap is not None and cap * n <= TEAM_MINUTES:
        alloc = pd.Series(TEAM_MINUTES / n, index=weights.index)
        flags = {int(pid): FLAG_CAP_RELAXED for pid in weights.index}
        return alloc, flags
    alloc = weights / weights.sum() * TEAM_MINUTES
    if cap is not None:
        capped: set[int] = set()
        for _ in range(n):
            uncapped = [pid for pid in weights.index if pid not in capped]
            remaining = TEAM_MINUTES - cap * len(capped)
            sub = weights.loc[uncapped]
            if sub.sum() <= 0.0:
                sub = pd.Series(1.0, index=sub.index)
            alloc.loc[uncapped] = sub / sub.sum() * remaining
            newly = [pid for pid in uncapped if alloc.loc[pid] > cap + 1e-9]
            if not newly:
                break
            for pid in newly:
                capped.add(pid)
                alloc.loc[pid] = cap
        for pid in capped:
            flags[int(pid)] = FLAG_CAPPED
    # exact-sum correction: push the float residual onto the largest
    # below-cap allocation (residual is ~1e-13, never material)
    residual = TEAM_MINUTES - float(alloc.sum())
    if residual != 0.0:
        if cap is not None:
            below = alloc[alloc < cap - 1e-9]
            target = below.idxmax() if not below.empty else alloc.idxmax()
        else:
            target = alloc.idxmax()
        alloc.loc[target] += residual
    return alloc, flags


def _finalize_frame(
    ctx: dict[str, Any],
    alloc: pd.Series,
    flags: dict[int, str],
    include_excluded: bool,
) -> pd.DataFrame:
    names = ctx["names"]
    opener = ctx["season_opener"]
    rows = []
    for pid in alloc.index:
        pid = int(pid)
        flag_parts = []
        if opener:
            flag_parts.append(FLAG_OPENER)
        if pid in flags:
            flag_parts.append(flags[pid])
        rows.append(
            {
                "person_id": pid,
                "player_name": names.get(pid, ""),
                "proj_minutes": float(alloc.loc[pid]),
                "proj_share": float(alloc.loc[pid]) / TEAM_MINUTES,
                "flag": "+".join(flag_parts),
            }
        )
    if include_excluded:
        for pid, reason in sorted(ctx["excluded"].items()):
            rows.append(
                {
                    "person_id": int(pid),
                    "player_name": names.get(pid, ""),
                    "proj_minutes": 0.0,
                    "proj_share": 0.0,
                    "flag": reason,
                }
            )
    frame = pd.DataFrame(rows, columns=["person_id", "player_name", "proj_minutes", "proj_share", "flag"])
    frame = frame.sort_values(
        ["proj_minutes", "person_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def project_team_minutes(
    game_logs: pd.DataFrame,
    team_tricode: str,
    as_of_date: str,
    exclude_person_ids: set | None = None,
    *,
    include_excluded: bool = False,
) -> pd.DataFrame:
    """Project a 200-minute pregame allocation for one team.

    Uses only games with ``game_date < as_of_date`` (strict). Returns columns
    ``[person_id, player_name, proj_minutes, proj_share, flag]`` where
    ``proj_minutes`` sums to exactly 200.0 over the available players.
    ``include_excluded=True`` additionally returns availability-excluded
    players as 0.0-minute debug rows (the 200 total is unchanged).
    """
    ctx = _roster_context(game_logs, team_tricode, as_of_date, exclude_person_ids)
    weights = _ewma_weights(ctx)
    alloc, flags = _allocate(weights, cap=PER_PLAYER_CAP_MINUTES)
    return _finalize_frame(ctx, alloc, flags, include_excluded)


def project_team_minutes_reference(
    game_logs: pd.DataFrame,
    team_tricode: str,
    as_of_date: str,
    method: str,
    exclude_person_ids: set | None = None,
) -> pd.DataFrame:
    """Eval baselines on the identical roster/availability/date fence.

    ``method='last_game'``: most-recent-game minutes renormalized to 200.
    ``method='season_avg'``: season-to-date average minutes renormalized to 200.
    No per-player cap — these are the plain baselines the engine must beat.
    """
    ctx = _roster_context(game_logs, team_tricode, as_of_date, exclude_person_ids)
    weights = _reference_weights(ctx, method)
    alloc, flags = _allocate(weights, cap=None)
    return _finalize_frame(ctx, alloc, flags, include_excluded=False)


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------

EVAL_METHODS = ("engine", "last_game", "season_avg")


def _load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(common.GAMES_TABLE_PATH, dtype={"game_id": str})
    logs = pd.read_csv(common.GAME_LOG_PATH, dtype={"game_id": str})
    logs["person_id"] = logs["person_id"].astype(int)
    logs["minutes"] = logs["minutes"].astype(float)
    return games, logs


def _team_game_list(games: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """One row per final team-game with a franchise season-opener marker."""
    finals = games[games["status"] == 3]
    halves = []
    for side in ("home", "away"):
        half = finals[["game_id", "season", "season_type", "game_date", f"{side}_tricode"]].copy()
        half.columns = ["game_id", "season", "season_type", "game_date", "team_tricode"]
        halves.append(half)
    team_games = pd.concat(halves, ignore_index=True)
    team_games["franchise"] = team_games["team_tricode"].map(lambda t: _franchise_codes(t)[-1])
    team_games = team_games.sort_values(["game_date", "game_id"], kind="mergesort")
    team_games["is_season_opener"] = (
        team_games.groupby(["franchise", "season"]).cumcount() == 0
    )
    keep = team_games["season"].isin(list(seasons))
    return team_games.loc[keep].reset_index(drop=True)


def run_evaluation(seasons: tuple[int, ...], out_dir: Path) -> dict[str, Any]:
    games, logs = _load_panel()
    team_games = _team_game_list(games, seasons)
    actual_by_game = {
        key: grp
        for key, grp in logs.loc[logs["played"] == 1, ["game_id", "team_tricode", "person_id", "minutes"]]
        .groupby(["game_id", "team_tricode"])
    }

    totals: dict[int, dict[str, Any]] = {
        season: {
            "n_team_games": 0,
            "n_skipped": 0,
            "n_player_slots": 0,
            "abs_err": {method: 0.0 for method in EVAL_METHODS},
        }
        for season in seasons
    }
    game_rows = []
    for row in team_games.itertuples(index=False):
        season_totals = totals[int(row.season)]
        if row.is_season_opener:
            season_totals["n_skipped"] += 1
            continue
        actual = actual_by_game.get((row.game_id, row.team_tricode))
        if actual is None or actual.empty:
            raise AssertionError(
                f"final game {row.game_id} has no played rows for {row.team_tricode}"
            )
        projections = {
            "engine": project_team_minutes(logs, row.team_tricode, row.game_date),
            "last_game": project_team_minutes_reference(
                logs, row.team_tricode, row.game_date, "last_game"
            ),
            "season_avg": project_team_minutes_reference(
                logs, row.team_tricode, row.game_date, "season_avg"
            ),
        }
        game_record = {
            "game_id": row.game_id,
            "season": int(row.season),
            "season_type": row.season_type,
            "game_date": row.game_date,
            "team_tricode": row.team_tricode,
            "n_actual_players": int(len(actual)),
            "actual_team_minutes": round(float(actual["minutes"].sum()), 3),
        }
        for method, frame in projections.items():
            total = float(frame["proj_minutes"].sum())
            assert abs(total - TEAM_MINUTES) <= 0.01, (
                f"{method} projection sums to {total} for {row.team_tricode} "
                f"as of {row.game_date}"
            )
            proj_by_pid = frame.set_index("person_id")["proj_minutes"]
            # missing projections count as 0 projected, per the eval contract
            errors = (
                actual["person_id"].map(proj_by_pid).fillna(0.0) - actual["minutes"]
            ).abs()
            season_totals["abs_err"][method] += float(errors.sum())
            game_record[f"mae_{method}"] = round(float(errors.mean()), 4)
        season_totals["n_team_games"] += 1
        season_totals["n_player_slots"] += int(len(actual))
        game_rows.append(game_record)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "roster_window_games": ROSTER_WINDOW_GAMES,
            "ewma_half_life_games": EWMA_HALF_LIFE_GAMES,
            "per_player_cap_minutes": PER_PLAYER_CAP_MINUTES,
            "team_minutes": TEAM_MINUTES,
            "ewma_include_active_zeros": EWMA_INCLUDE_ACTIVE_ZEROS,
            "exclude_absent_from_latest": EXCLUDE_ABSENT_FROM_LATEST,
        },
        "selection_fence": (
            "constants selected on 2025 only; 2026 is reported with frozen "
            "constants and never used for selection"
        ),
        "seasons": {},
    }
    for season in seasons:
        season_totals = totals[season]
        slots = season_totals["n_player_slots"]
        report["seasons"][str(season)] = {
            "n_team_games": season_totals["n_team_games"],
            "n_skipped": season_totals["n_skipped"],
            "n_player_slots": slots,
            "mae": {
                method: (
                    round(season_totals["abs_err"][method] / slots, 4) if slots else None
                )
                for method in EVAL_METHODS
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(game_rows).to_csv(out_dir / "game_level.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seasons",
        default="2025,2026",
        help="comma-separated evaluation seasons (default: 2025,2026)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MINUTES_EVAL_DIR,
        help=f"output directory (default: {MINUTES_EVAL_DIR})",
    )
    args = parser.parse_args()
    seasons = tuple(int(s) for s in args.seasons.split(",") if s.strip())
    report = run_evaluation(seasons, args.out_dir)
    for season, block in report["seasons"].items():
        print(
            f"{season}: games={block['n_team_games']} skipped={block['n_skipped']} "
            f"slots={block['n_player_slots']} mae={block['mae']}"
        )


if __name__ == "__main__":
    main()

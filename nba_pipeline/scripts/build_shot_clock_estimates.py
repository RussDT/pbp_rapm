#!/usr/bin/env python3
"""Estimate shot-clock time for FGA rows from raw NBA play-by-play."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import pandas as pd


EVENT_TYPE_MAP = {
    1: "MAKE",
    2: "MISS",
    3: "FreeThrow",
    4: "Rebound",
    5: "Turnover",
    6: "Foul",
    7: "Violation",
    8: "Substitution",
    9: "Timeout",
    10: "JumpBall",
    11: "Ejection",
    12: "StartOfPeriod",
    13: "EndOfPeriod",
    14: "Empty",
}

OUTPUT_COLUMNS = [
    "game_id",
    "period",
    "event_num",
    "time_quarter",
    "clock_sec",
    "shooter_id",
    "shooter_name",
    "shot_side",
    "shot_clock_est",
    "shot_clock_raw",
    "elapsed_since_reset",
    "reset_len",
    "reset_reason",
    "reset_event_num",
    "reset_time_quarter",
    "confidence",
    "confidence_reason",
]


def _desc(row: pd.Series, side: str) -> str:
    col = "home_description" if side == "Home" else "visitor_description"
    value = row.get(col, "")
    return "" if pd.isna(value) else str(value)


def _all_desc(row: pd.Series) -> str:
    parts = []
    for col in ["home_description", "visitor_description", "neutral_description"]:
        value = row.get(col, "")
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    return " ".join(parts)


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text or "", flags=re.IGNORECASE))


def _clean_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _other_side(side: Optional[str]) -> Optional[str]:
    return {"Home": "Away", "Away": "Home"}.get(side or "")


def _event_type_name(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        num = int(value)
    except (TypeError, ValueError):
        return str(value)
    return EVENT_TYPE_MAP.get(num, str(value))


def _period_length_seconds(period: int) -> int:
    return 720 if int(period) <= 4 else 300


def _clock_seconds(row: pd.Series) -> int:
    if "clock_sec" in row and pd.notna(row["clock_sec"]):
        return int(row["clock_sec"])
    minute = int(row.get("minute_remaining_quarter", 0) or 0)
    second = int(row.get("seconds_remaining_quarter", 0) or 0)
    return minute * 60 + second


def _event_side(row: pd.Series, kind: Optional[str] = None) -> Optional[str]:
    event_type = row.get("_event_type_name", _event_type_name(row.get("event_type")))
    home = _desc(row, "Home")
    away = _desc(row, "Away")

    if event_type in {"MAKE", "MISS"} or kind == "shot":
        home_match = _has(home, r"\b(MISS|PTS)\b")
        away_match = _has(away, r"\b(MISS|PTS)\b")
    elif event_type == "Turnover" or kind == "turnover":
        home_match = _has(home, r"Turnover")
        away_match = _has(away, r"Turnover")
    elif event_type == "Rebound" or kind == "rebound":
        home_match = _has(home, r"REBOUND|Rebound")
        away_match = _has(away, r"REBOUND|Rebound")
    elif event_type == "Foul" or kind == "foul":
        home_match = _has(home, r"FOUL|Foul|Charge|Offensive")
        away_match = _has(away, r"FOUL|Foul|Charge|Offensive")
    elif event_type == "FreeThrow" or kind == "ft":
        home_match = _has(home, r"Free Throw")
        away_match = _has(away, r"Free Throw")
    elif event_type == "Violation" or kind == "violation":
        home_match = _has(home, r"Violation|Kicked Ball|Goaltending")
        away_match = _has(away, r"Violation|Kicked Ball|Goaltending")
    else:
        home_match = bool(home.strip())
        away_match = bool(away.strip())

    if home_match and not away_match:
        return "Home"
    if away_match and not home_match:
        return "Away"
    return None


def _build_game_name_side_map(game_df: pd.DataFrame) -> dict[str, str]:
    name_side: dict[str, str] = {}
    for _, row in game_df.iterrows():
        side = _event_side(row)
        name = _clean_name(row.get("player1_name"))
        if not side or not name:
            continue
        existing = name_side.get(name)
        if existing is None:
            name_side[name] = side
        elif existing != side:
            name_side[name] = "CONFLICT"
    return {name: side for name, side in name_side.items() if side in {"Home", "Away"}}


def _jump_tip_side(row: pd.Series, name_side: dict[str, str]) -> Optional[str]:
    match = re.search(r"Tip to\s+(.+?)(?:$|:|\s{2,})", _all_desc(row), flags=re.IGNORECASE)
    if not match:
        return None
    tip_name = _clean_name(match.group(1))
    if tip_name in name_side:
        return name_side[tip_name]

    matches = {
        side
        for name, side in name_side.items()
        if tip_name and (tip_name == name or tip_name in name or name in tip_name)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None


def _is_final_ft(text: str) -> bool:
    return _has(text, r"\b(1 of 1|2 of 2|3 of 3)\b")


def _is_missed(text: str) -> bool:
    return _has(text, r"\bMISS\b")


def _is_admin_ft(text: str) -> bool:
    return _has(text, r"Technical|Flagrant|Clear Path|Def\.?\s*3|Defensive 3|Transition Take")


def _is_offensive_foul(text: str) -> bool:
    return _has(text, r"offensive|charge")


def _is_shooting_foul(text: str) -> bool:
    return _has(text, r"\bS\.FOUL\b|shooting")


def _is_kicked_ball_violation(text: str) -> bool:
    return _has(text, r"Kicked Ball")


def _confidence(
    raw_estimate: int,
    shot_side: Optional[str],
    current_offense: Optional[str],
    inherited_reasons: list[str],
) -> tuple[str, str]:
    reasons = list(inherited_reasons)
    if current_offense and shot_side and current_offense != shot_side:
        reasons.append("side_mismatch")
    if raw_estimate < -2 or raw_estimate > 24:
        reasons.append("raw_out_of_range")

    if "side_mismatch" in reasons or "raw_out_of_range" in reasons:
        return "low", ";".join(reasons)
    if "unresolved_jumpball_tip" in reasons:
        return "medium", ";".join(reasons)
    return "high", ";".join(reasons) if reasons else "ok"


def estimate_shot_clock(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return one shot-clock estimate row for every raw FGA event."""
    if raw_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = raw_df.copy()
    df["_event_type_name"] = df["event_type"].map(_event_type_name)
    if "clock_sec" not in df.columns:
        df["clock_sec"] = (
            pd.to_numeric(df.get("minute_remaining_quarter", 0), errors="coerce").fillna(0).astype(int) * 60
            + pd.to_numeric(df.get("seconds_remaining_quarter", 0), errors="coerce").fillna(0).astype(int)
        )
    df["_orig_order"] = range(len(df))
    df["event_num"] = pd.to_numeric(df["event_num"], errors="coerce").fillna(0).astype(int)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype(int)

    sorted_df = df.sort_values(
        ["game_id", "period", "clock_sec", "event_num", "_orig_order"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )

    estimates: list[dict[str, object]] = []
    for game_id, game_df in sorted_df.groupby("game_id", sort=False):
        name_side = _build_game_name_side_map(game_df)
        for period, period_df in game_df.groupby("period", sort=False):
            period_length = _period_length_seconds(int(period))
            current_offense: Optional[str] = None
            reset_clock = period_length
            reset_len = 24
            reset_reason = "period_start"
            reset_event_num: Optional[int] = None
            reset_time_quarter = f"{period_length // 60:02d}:00"
            last_live_miss_side: Optional[str] = None
            inherited_confidence_reasons: list[str] = []

            def set_reset(
                *,
                offense: Optional[str],
                clock: int,
                length: int,
                reason: str,
                row: pd.Series,
                confidence_reason: Optional[str] = None,
            ) -> None:
                nonlocal current_offense, reset_clock, reset_len, reset_reason
                nonlocal reset_event_num, reset_time_quarter, inherited_confidence_reasons
                if offense:
                    current_offense = offense
                reset_clock = clock
                reset_len = length
                reset_reason = reason
                reset_event_num = int(row["event_num"])
                reset_time_quarter = str(row.get("time_quarter", ""))
                inherited_confidence_reasons = [confidence_reason] if confidence_reason else []

            for _, row in period_df.iterrows():
                event_type = row["_event_type_name"]
                clock = _clock_seconds(row)
                text = _all_desc(row)

                if event_type in {"MAKE", "MISS"}:
                    shot_side = _event_side(row, "shot")
                    if current_offense is None and shot_side:
                        current_offense = shot_side
                    elapsed_since_reset = reset_clock - clock
                    raw_estimate = reset_len - elapsed_since_reset
                    confidence, confidence_reason = _confidence(
                        raw_estimate,
                        shot_side,
                        current_offense,
                        inherited_confidence_reasons,
                    )
                    estimates.append(
                        {
                            "game_id": game_id,
                            "period": int(period),
                            "event_num": int(row["event_num"]),
                            "time_quarter": row.get("time_quarter", ""),
                            "clock_sec": clock,
                            "shooter_id": row.get("player1_id"),
                            "shooter_name": row.get("player1_name"),
                            "shot_side": shot_side or "",
                            "shot_clock_est": int(max(0, min(24, raw_estimate))),
                            "shot_clock_raw": int(raw_estimate),
                            "elapsed_since_reset": int(elapsed_since_reset),
                            "reset_len": int(reset_len),
                            "reset_reason": reset_reason,
                            "reset_event_num": reset_event_num,
                            "reset_time_quarter": reset_time_quarter,
                            "confidence": confidence,
                            "confidence_reason": confidence_reason,
                        }
                    )
                    last_live_miss_side = shot_side if event_type == "MISS" else None

                if event_type == "StartOfPeriod":
                    current_offense = None
                    reset_clock = period_length
                    reset_len = 24
                    reset_reason = "period_start"
                    reset_event_num = int(row["event_num"])
                    reset_time_quarter = str(row.get("time_quarter", ""))
                    last_live_miss_side = None
                    inherited_confidence_reasons = []
                elif event_type == "JumpBall":
                    jump_side = _jump_tip_side(row, name_side)
                    if jump_side:
                        if current_offense == jump_side:
                            set_reset(
                                offense=jump_side,
                                clock=clock,
                                length=reset_len,
                                reason="jumpball_retained",
                                row=row,
                                confidence_reason="jumpball_retained",
                            )
                        else:
                            set_reset(
                                offense=jump_side,
                                clock=clock,
                                length=24,
                                reason="jumpball_tip",
                                row=row,
                                confidence_reason=None if clock == period_length else "jumpball_change",
                            )
                    else:
                        inherited_confidence_reasons = ["unresolved_jumpball_tip"]
                elif event_type == "MAKE":
                    shot_side = _event_side(row, "shot")
                    if shot_side:
                        set_reset(
                            offense=_other_side(shot_side),
                            clock=clock,
                            length=24,
                            reason="made_fg",
                            row=row,
                        )
                    last_live_miss_side = None
                elif event_type == "Turnover":
                    turnover_side = _event_side(row, "turnover")
                    if turnover_side:
                        set_reset(
                            offense=_other_side(turnover_side),
                            clock=clock,
                            length=24,
                            reason="turnover",
                            row=row,
                        )
                    last_live_miss_side = None
                elif event_type == "Rebound":
                    rebound_side = _event_side(row, "rebound")
                    if rebound_side:
                        if last_live_miss_side and rebound_side == last_live_miss_side:
                            set_reset(
                                offense=rebound_side,
                                clock=clock,
                                length=14,
                                reason="off_rebound",
                                row=row,
                            )
                        else:
                            set_reset(
                                offense=rebound_side,
                                clock=clock,
                                length=24,
                                reason="def_rebound",
                                row=row,
                            )
                    last_live_miss_side = None
                elif event_type == "FreeThrow":
                    ft_side = _event_side(row, "ft")
                    if _is_admin_ft(text):
                        continue
                    if _is_final_ft(text):
                        if _is_missed(text):
                            last_live_miss_side = ft_side
                        elif ft_side:
                            set_reset(
                                offense=_other_side(ft_side),
                                clock=clock,
                                length=24,
                                reason="made_final_ft",
                                row=row,
                            )
                            last_live_miss_side = None
                elif event_type == "Foul":
                    foul_side = _event_side(row, "foul")
                    if foul_side and current_offense:
                        if foul_side == current_offense and _is_offensive_foul(text):
                            set_reset(
                                offense=_other_side(foul_side),
                                clock=clock,
                                length=24,
                                reason="offensive_foul",
                                row=row,
                            )
                            last_live_miss_side = None
                        elif foul_side != current_offense and not _is_shooting_foul(text):
                            implied_clock = reset_len - (reset_clock - clock)
                            if implied_clock < 14:
                                set_reset(
                                    offense=current_offense,
                                    clock=clock,
                                    length=14,
                                    reason="def_foul_reset14",
                                    row=row,
                                )
                            else:
                                inherited_confidence_reasons = ["def_foul_preserved_above_14"]
                elif event_type == "Violation":
                    violation_side = _event_side(row, "violation")
                    if (
                        violation_side
                        and current_offense
                        and violation_side != current_offense
                        and _is_kicked_ball_violation(text)
                    ):
                        implied_clock = reset_len - (reset_clock - clock)
                        if implied_clock < 14:
                            set_reset(
                                offense=current_offense,
                                clock=clock,
                                length=14,
                                reason="kicked_ball_reset14",
                                row=row,
                            )
                        else:
                            inherited_confidence_reasons = ["kicked_ball_preserved_above_14"]

    out = pd.DataFrame(estimates)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return out[OUTPUT_COLUMNS].sort_values(["game_id", "period", "clock_sec", "event_num"], ascending=[True, True, False, True])


def _season_input_path(root: Path, season: int, season_type: str) -> Path:
    suffix = "_PS" if season_type.upper() == "PS" else ""
    return root / "raw_data" / f"NBA{int(season):02d}{suffix}.parquet"


def _output_stem(season: int, season_type: str) -> str:
    return f"shot_clock_estimates_{int(season):02d}_{season_type.lower()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="Two-digit season end year, e.g. 26")
    parser.add_argument("--season-type", choices=["RS", "PS"], required=True)
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to nba_pipeline",
    )
    parser.add_argument("--input", type=Path, default=None, help="Optional raw parquet override")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to nba_pipeline/results/shot_clock",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_root = args.pipeline_root.resolve()
    input_path = args.input.resolve() if args.input else _season_input_path(pipeline_root, args.season, args.season_type)
    output_dir = args.output_dir.resolve() if args.output_dir else pipeline_root / "results" / "shot_clock"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_parquet(input_path)
    estimates = estimate_shot_clock(raw_df)
    diagnostics = estimates[estimates["confidence"] != "high"].copy()

    stem = _output_stem(args.season, args.season_type)
    parquet_path = output_dir / f"{stem}.parquet"
    diagnostics_path = output_dir / f"{stem}_diagnostics.csv"
    estimates.to_parquet(parquet_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    high_count = int((estimates["confidence"] == "high").sum())
    high_rate = high_count / len(estimates) if len(estimates) else 0.0
    print(f"Wrote {len(estimates):,} shot-clock estimates to {parquet_path}")
    print(f"Wrote {len(diagnostics):,} diagnostics to {diagnostics_path}")
    print(f"High confidence: {high_count:,}/{len(estimates):,} ({high_rate:.1%})")


if __name__ == "__main__":
    main()

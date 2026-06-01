#!/usr/bin/env python3
"""Build a standalone PopcornMachine-style game-flow HTML report."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DEFAULT = REPO_ROOT / "nba_pipeline/raw_data/NBA26_PS.parquet"
RAPM_DEFAULT = REPO_ROOT / "nba_pipeline/processed/RAPM26_PS.parquet"
PLAYER_MAP_DEFAULT = REPO_ROOT / "autocomplete_map.csv"
OUT_DEFAULT = REPO_ROOT / "nba_pipeline/results/gameflows/okc_sas_2026_playoffs_game3_popcorn.html"

TEAM_META = {
    "OKC": {
        "name": "Oklahoma City Thunder",
        "short": "Thunder",
        "primary": "#006bb6",
        "secondary": "#ef3b24",
        "off": "#edb489",
    },
    "SAS": {
        "name": "San Antonio Spurs",
        "short": "Spurs",
        "primary": "#171717",
        "secondary": "#c4ced4",
        "off": "#e4e4e4",
    },
}


def norm_game_id(value: Any) -> str:
    raw = str(value).strip()
    digits = re.sub(r"\D", "", raw)
    return digits[-8:] if len(digits) >= 8 else digits


def clock_to_elapsed(row: pd.Series) -> float:
    period = int(row["period"])
    if "minute_remaining_quarter" in row and "seconds_remaining_quarter" in row:
        rem = float(row["minute_remaining_quarter"]) * 60.0 + float(row["seconds_remaining_quarter"])
    else:
        clock = str(row["time_quarter"])
        minute, second = clock.split(":", 1)
        rem = float(minute) * 60.0 + float(second)
    elapsed = (period - 1) * 12.0 + (720.0 - rem) / 60.0
    return max(0.0, min(48.0, elapsed))


def esc(value: Any) -> str:
    return html.escape("" if pd.isna(value) else str(value), quote=True)


def fmt_pct(made: int, att: int) -> str:
    return ".---" if att == 0 else f"{made / att:.2f}".replace("0.", ".")


def name_key(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).strip()


def load_player_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(row.nba_id): str(row.player_name) for row in df.itertuples() if pd.notna(row.nba_id)}


def resolve_assist_player(raw_name: str, names: dict[int, str], rosters: dict[str, set[int]]) -> int | None:
    roster_ids = set().union(*rosters.values())
    target = name_key(raw_name)
    full = {name_key(name): pid for pid, name in names.items() if pid in roster_ids}
    if target in full:
        return full[target]

    parts = target.split()
    candidates = []
    for pid in roster_ids:
        full_name = name_key(names.get(pid, ""))
        full_parts = full_name.split()
        if not full_parts:
            continue
        if len(parts) == 1 and full_parts[-1] == parts[0]:
            candidates.append(pid)
        elif len(parts) >= 2 and full_parts[-1] == parts[-1] and full_parts[0].startswith(parts[0][0]):
            candidates.append(pid)
    return candidates[0] if len(candidates) == 1 else None


def detect_team_rosters(game: pd.DataFrame) -> tuple[dict[str, set[int]], dict[int, str]]:
    side_cols = {
        "OKC": [f"away_player{i}" for i in range(1, 6)],
        "SAS": [f"home_player{i}" for i in range(1, 6)],
    }
    rosters: dict[str, set[int]] = {"OKC": set(), "SAS": set()}
    player_team: dict[int, str] = {}
    for team, cols in side_cols.items():
        for col in cols:
            for value in game[col].dropna():
                pid = int(value)
                if pid > 0:
                    rosters[team].add(pid)
                    player_team[pid] = team
    return rosters, player_team


def event_description(row: pd.Series) -> str:
    for col in ("home_description", "visitor_description", "neutral_description"):
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def scoring_events(game: pd.DataFrame, player_team: dict[int, str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    prev_away = 0
    prev_home = 0
    for row in game.itertuples(index=False):
        away = getattr(row, "away_score")
        home = getattr(row, "home_score")
        if pd.isna(away) or pd.isna(home):
            continue
        away_i = int(away)
        home_i = int(home)
        delta_away = away_i - prev_away
        delta_home = home_i - prev_home
        prev_away, prev_home = away_i, home_i
        points = max(delta_away, delta_home)
        if points <= 0:
            continue
        pid = int(getattr(row, "player1_id") or 0)
        events.append(
            {
                "elapsed": clock_to_elapsed(pd.Series(row._asdict())),
                "event_num": int(getattr(row, "event_num")),
                "team": "OKC" if delta_away > delta_home else "SAS",
                "player_id": pid,
                "points": points,
                "away_delta": delta_away,
                "home_delta": delta_home,
                "desc": event_description(pd.Series(row._asdict())),
            }
        )
    return events


def lineup_from_row(row: pd.Series | dict[str, Any], side: str) -> list[int]:
    return [int(row[f"{side}_player{i}"]) for i in range(1, 6) if pd.notna(row[f"{side}_player{i}"])]


def build_intervals(game: pd.DataFrame) -> list[dict[str, Any]]:
    """Build clean rotation intervals from starters plus substitution rows only."""
    intervals: list[dict[str, Any]] = []
    for period, period_df in game.sort_values(["period", "elapsed", "event_num"]).groupby("period"):
        rows = period_df.to_dict("records")
        if not rows:
            continue
        quarter_start = (int(period) - 1) * 12.0
        quarter_end = int(period) * 12.0
        current = {
            "away": lineup_from_row(rows[0], "away"),
            "home": lineup_from_row(rows[0], "home"),
        }
        cursor = quarter_start
        for row in rows:
            if int(row["event_type"]) != 8:
                continue
            t = float(row["elapsed"])
            if t > cursor:
                intervals.append(
                    {
                        "start": cursor,
                        "end": t,
                        "away": list(current["away"]),
                        "home": list(current["home"]),
                    }
                )
                cursor = t
            if isinstance(row.get("visitor_description"), str) and "SUB:" in row["visitor_description"]:
                current["away"] = lineup_from_row(row, "away")
            if isinstance(row.get("home_description"), str) and "SUB:" in row["home_description"]:
                current["home"] = lineup_from_row(row, "home")
        if quarter_end > cursor:
            intervals.append(
                {
                    "start": cursor,
                    "end": quarter_end,
                    "away": list(current["away"]),
                    "home": list(current["home"]),
                }
            )
    return intervals


def build_stints(intervals: list[dict[str, Any]], rosters: dict[str, set[int]]) -> tuple[dict[int, list[dict[str, float]]], dict[int, float]]:
    stints: dict[int, list[dict[str, float]]] = defaultdict(list)
    minutes: dict[int, float] = defaultdict(float)
    side_for_team = {"OKC": "away", "SAS": "home"}
    for team, players in rosters.items():
        side = side_for_team[team]
        for pid in players:
            current: dict[str, float] | None = None
            for interval in intervals:
                on = pid in interval[side]
                if not on:
                    if current:
                        stints[pid].append(current)
                        current = None
                    continue
                minutes[pid] += interval["end"] - interval["start"]
                if current and abs(current["end"] - interval["start"]) < 1e-6:
                    current["end"] = interval["end"]
                else:
                    if current:
                        stints[pid].append(current)
                    current = {"start": interval["start"], "end": interval["end"]}
            if current:
                stints[pid].append(current)
    return stints, minutes


def lineups_at_event(raw_row: pd.Series) -> tuple[list[int], list[int]]:
    away = [int(raw_row[f"away_player{i}"]) for i in range(1, 6) if pd.notna(raw_row[f"away_player{i}"])]
    home = [int(raw_row[f"home_player{i}"]) for i in range(1, 6) if pd.notna(raw_row[f"home_player{i}"])]
    return away, home


def player_box(
    game: pd.DataFrame,
    scoring: list[dict[str, Any]],
    rosters: dict[str, set[int]],
    names: dict[int, str],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = defaultdict(lambda: {"pts": 0, "reb": 0, "ast": 0, "tov": 0, "pm": 0})
    name_to_pid: dict[str, int] = {}
    for pid in set().union(*rosters.values()):
        rows[pid]
    for score in scoring:
        pid = score["player_id"]
        if pid:
            rows[pid]["pts"] += int(score["points"])

    score_by_event = {item["event_num"]: item for item in scoring}
    for _, row in game.iterrows():
        desc = event_description(row)
        pid = int(row["player1_id"]) if pd.notna(row["player1_id"]) else 0
        if pid:
            name = row.get("player1_name")
            if isinstance(name, str) and name:
                name_to_pid[name] = pid
        if int(row["event_type"]) == 4 and pid:
            rows[pid]["reb"] += 1
        if int(row["event_type"]) == 5 and pid:
            rows[pid]["tov"] += 1
        ast_match = re.search(r"\(([^()]*)\s+(\d+)\s+AST\)", desc)
        if ast_match:
            name = ast_match.group(1).strip()
            ast_total = int(ast_match.group(2))
            ast_pid = name_to_pid.get(name) or resolve_assist_player(name, names, rosters)
            if ast_pid:
                rows[ast_pid]["ast"] = max(rows[ast_pid]["ast"], ast_total)
        scoring_row = score_by_event.get(int(row["event_num"]))
        if scoring_row:
            away, home = lineups_at_event(row)
            pm_delta_home = scoring_row["home_delta"] - scoring_row["away_delta"]
            for player_id in home:
                rows[player_id]["pm"] += pm_delta_home
            for player_id in away:
                rows[player_id]["pm"] -= pm_delta_home
    return rows


def team_period_stats(game: pd.DataFrame, scoring: list[dict[str, Any]]) -> dict[str, list[dict[str, int]]]:
    stats = {team: [defaultdict(int) for _ in range(4)] for team in ("OKC", "SAS")}
    score_by_event = {item["event_num"]: item for item in scoring}
    for _, row in game.iterrows():
        period = int(row["period"])
        if not 1 <= period <= 4:
            continue
        q = period - 1
        desc = event_description(row)
        side_team = "SAS" if isinstance(row.get("home_description"), str) and row["home_description"].strip() else "OKC"
        et = int(row["event_type"])
        if et in (1, 2):
            stats[side_team][q]["fga"] += 1
            if "3PT" in desc:
                stats[side_team][q]["3pa"] += 1
            if et == 1:
                stats[side_team][q]["fgm"] += 1
                if "3PT" in desc:
                    stats[side_team][q]["3pm"] += 1
        if et == 4:
            stats[side_team][q]["reb"] += 1
        if et == 5:
            stats[side_team][q]["tov"] += 1
        if re.search(r"\([^()]*\s+\d+\s+AST\)", desc):
            stats[side_team][q]["ast"] += 1
        score = score_by_event.get(int(row["event_num"]))
        if score:
            stats["OKC"][q]["pts"] += score["away_delta"]
            stats["SAS"][q]["pts"] += score["home_delta"]
    return stats


def score_flow(processed: pd.DataFrame) -> dict[str, Any]:
    points = [{"t": 0.0, "margin": 0.0, "away": 0, "home": 0}]
    for _, row in processed.sort_values(["period", "event_num"]).iterrows():
        t = clock_to_elapsed(row)
        away = int(row["away_score"])
        home = int(row["home_score"])
        points.append({"t": t, "margin": away - home, "away": away, "home": home})
    if points[-1]["t"] < 48:
        last = points[-1].copy()
        last["t"] = 48.0
        points.append(last)
    weighted = 0.0
    for prev, cur in zip(points, points[1:]):
        weighted += prev["margin"] * (cur["t"] - prev["t"])
    max_abs = max(6, int(max(abs(p["margin"]) for p in points) + 2))
    return {"points": points, "avg_margin": weighted / 48.0, "max_abs": max_abs}


def team_lineup_segments(team: str, intervals: list[dict[str, Any]], scoring: list[dict[str, Any]]) -> list[dict[str, Any]]:
    side = "away" if team == "OKC" else "home"
    score_events = sorted(scoring, key=lambda item: (item["elapsed"], item["event_num"]))
    segments: list[dict[str, Any]] = []
    for interval in intervals:
        lineup = list(interval[side])
        if not segments or segments[-1]["lineup"] != lineup:
            segments.append(
                {
                    "start": interval["start"],
                    "end": interval["end"],
                    "lineup": lineup,
                    "pm": 0,
                }
            )
        else:
            segments[-1]["end"] = interval["end"]
    for score in score_events:
        for seg in segments:
            if seg["start"] <= score["elapsed"] < seg["end"] + 1e-6:
                okc = score["away_delta"] - score["home_delta"]
                seg["pm"] += okc if team == "OKC" else -okc
                break
    return [seg for seg in segments if seg["end"] - seg["start"] >= 0.08]


def render_svg(flow: dict[str, Any]) -> str:
    width, height = 1000, 138
    top, bottom = 18, 116
    mid = (top + bottom) / 2
    scale = (bottom - top) / 2 / flow["max_abs"]
    pts = []
    for item in flow["points"]:
        x = item["t"] / 48 * width
        y = mid - item["margin"] * scale
        pts.append(f"{x:.1f},{y:.1f}")
    q_lines = "\n".join(
        f'<line x1="{q / 48 * width:.1f}" y1="{top}" x2="{q / 48 * width:.1f}" y2="{bottom}" class="qline"/>'
        for q in (12, 24, 36)
    )
    return f"""
    <svg class="flow-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" aria-label="Score margin flow">
      <line x1="0" y1="{mid:.1f}" x2="{width}" y2="{mid:.1f}" class="zero"/>
      {q_lines}
      <polyline points="{' '.join(pts)}" class="margin-line"/>
      <text x="6" y="15" class="svg-label">OKC +{flow['max_abs']}</text>
      <text x="6" y="134" class="svg-label">SAS +{flow['max_abs']}</text>
      <text x="916" y="17" class="svg-score">123</text>
      <text x="916" y="132" class="svg-score">108</text>
    </svg>
    """


def render_quarter_headers() -> str:
    return "".join(
        f'<div class="quarter-label" style="left:{i*25}%;width:25%">{label}</div>'
        for i, label in enumerate(("1st Quarter", "2nd Quarter", "3rd Quarter", "4th Quarter"))
    )


def render_player_row(
    pid: int,
    team: str,
    names: dict[int, str],
    stints: dict[int, list[dict[str, float]]],
    minutes: dict[int, float],
    box: dict[int, dict[str, Any]],
    scoring: list[dict[str, Any]],
) -> str:
    meta = TEAM_META[team]
    bars = []
    labels = []
    for stint in stints.get(pid, []):
        left = stint["start"] / 48 * 100
        width = (stint["end"] - stint["start"]) / 48 * 100
        bars.append(f'<span class="stint {team.lower()}" style="left:{left:.3f}%;width:{width:.3f}%"></span>')
        stint_points = 0
        stint_pm = 0
        for score in scoring:
            if not (stint["start"] <= score["elapsed"] < stint["end"] + 1e-6):
                continue
            if score["player_id"] == pid:
                stint_points += int(score["points"])
            okc_pm = int(score["away_delta"] - score["home_delta"])
            stint_pm += okc_pm if team == "OKC" else -okc_pm
        if stint_points or stint_pm:
            pm_text = str(stint_pm)
            label_left = min(left + 0.45, 98.0)
            label_width = max(width - 0.9, 0.8)
            labels.append(
                f'<span class="stint-label" title="{esc(names.get(pid, str(pid)))} stint: {stint_points} pts, {stint_pm:+d}" '
                f'style="left:{label_left:.3f}%;width:{label_width:.3f}%">{stint_points} {pm_text}</span>'
            )
    row_box = box[pid]
    pm = int(row_box["pm"])
    pm_text = f"+{pm}" if pm > 0 else str(pm)
    return f"""
      <div class="pm-row {team.lower()}-row">
        <div class="player-name">{esc(names.get(pid, str(pid)))}</div>
        <div class="timeline" style="--off:{meta['off']};--on:{meta['primary']}">
          {render_quarter_headers()}
          {''.join(bars)}
          {''.join(labels)}
        </div>
        <div class="total-cell">{minutes.get(pid, 0):.1f}</div>
        <div class="total-cell">{row_box['pts']}</div>
        <div class="total-cell">{row_box['reb']}</div>
        <div class="total-cell">{row_box['ast']}</div>
        <div class="total-cell">{pm_text}</div>
      </div>
    """


def render_lineup_row(team: str, segments: list[dict[str, Any]]) -> str:
    cells = []
    for seg in segments:
        left = seg["start"] / 48 * 100
        width = (seg["end"] - seg["start"]) / 48 * 100
        value = int(seg["pm"])
        text = f"+{value}" if value > 0 else str(value)
        if value == 0 and width < 2.2:
            text = ""
        cells.append(f'<span class="lineup-cell {team.lower()}" style="left:{left:.3f}%;width:{width:.3f}%">{text}</span>')
    return f"""
      <div class="pm-row lineup-row">
        <div class="player-name">lineup +/-</div>
        <div class="timeline lineup-timeline">{render_quarter_headers()}{''.join(cells)}</div>
        <div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div>
      </div>
    """


def render_period_table(period_stats: dict[str, list[dict[str, int]]]) -> str:
    rows = []
    for team in ("OKC", "SAS"):
        running = defaultdict(int)
        cells = []
        for q, stats in enumerate(period_stats[team], start=1):
            for key, value in stats.items():
                running[key] += value
            qline = (
                f"{stats['pts']}  FG {stats['fgm']}-{stats['fga']} {fmt_pct(stats['fgm'], stats['fga'])}  "
                f"3P {stats['3pm']}-{stats['3pa']} {fmt_pct(stats['3pm'], stats['3pa'])}  "
                f"R/A/T {stats['reb']}/{stats['ast']}/{stats['tov']}"
            )
            rline = (
                f"{running['pts']}  FG {running['fgm']}-{running['fga']} {fmt_pct(running['fgm'], running['fga'])}  "
                f"3P {running['3pm']}-{running['3pa']} {fmt_pct(running['3pm'], running['3pa'])}  "
                f"R/A/T {running['reb']}/{running['ast']}/{running['tov']}"
            )
            cells.append(f"<td><strong>Q{q}</strong> {esc(qline)}<br><span>{esc(rline)}</span></td>")
        rows.append(f"<tr><th>{team}</th>{''.join(cells)}</tr>")
    return f"""
    <table class="period-table">
      <caption>Period stats with running totals - points, FG%, 3P%, rebounds, assists, turnovers</caption>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def render_html(
    game: pd.DataFrame,
    processed: pd.DataFrame,
    names: dict[int, str],
    rosters: dict[str, set[int]],
    stints: dict[int, list[dict[str, float]]],
    minutes: dict[int, float],
    box: dict[int, dict[str, Any]],
    scoring: list[dict[str, Any]],
    lineup_segments: dict[str, list[dict[str, Any]]],
    period_stats: dict[str, list[dict[str, int]]],
) -> str:
    top_team = "OKC"
    bottom_team = "SAS"
    order = {
        team: sorted(rosters[team], key=lambda pid: (-minutes.get(pid, 0), names.get(pid, str(pid))))
        for team in ("OKC", "SAS")
    }
    flow = score_flow(processed)
    top_rows = "\n".join(render_player_row(pid, top_team, names, stints, minutes, box, scoring) for pid in order[top_team])
    bottom_rows = "\n".join(render_player_row(pid, bottom_team, names, stints, minutes, box, scoring) for pid in order[bottom_team])
    title = "PopcornMachine's GameFlows - Oklahoma City Thunder @ San Antonio Spurs - Game 3 - May 22, 2026"
    data_note = "Source: local nba_pipeline raw NBA26_PS.parquet and processed RAPM26_PS.parquet, game 0042500313."
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --yellow:#ffd918;
      --grid:#ffffff;
      --paper:#f8f8f8;
      --line:#cfcfcf;
      --okc:#006bb6;
      --sas:#171717;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0;
      background:#d7d7d7;
      color:#080808;
      font-family: Arial, Helvetica, sans-serif;
    }}
    .page {{
      width: min(1820px, calc(100vw - 16px));
      min-width: 1180px;
      margin: 8px auto 24px;
      border: 7px solid #0d1018;
      border-top-color:#26324d;
      background:white;
      box-shadow: 0 16px 45px rgba(0,0,0,.35);
    }}
    .hero {{
      height: 210px;
      position: relative;
      overflow:hidden;
      background: linear-gradient(#686868 0%, #252525 58%, #101010 100%);
      border-bottom: 8px solid #bfbfbf;
    }}
    h1 {{
      margin: 0;
      padding: 18px 0 0 30px;
      font-size: 58px;
      line-height: .95;
      color: var(--yellow);
      letter-spacing: 0;
      font-weight: 900;
    }}
    .quote {{
      position:absolute;
      top: 20px;
      right: 26px;
      width: 520px;
      text-align:right;
      color:var(--yellow);
      font-size: 25px;
      line-height: 1.02;
      font-weight:900;
    }}
    .pop {{
      position:absolute;
      width:44px;
      height:34px;
      filter: drop-shadow(3px 3px 0 #000);
    }}
    .pop::before, .pop::after {{
      content:"";
      position:absolute;
      width:21px;
      height:21px;
      border-radius:50%;
      background:#fff6a6;
      box-shadow: 11px -5px 0 #fff6a6, 22px 2px 0 #fff6a6, 7px 12px 0 #fff6a6;
    }}
    .nav {{
      height: 57px;
      display:flex;
      gap: 48px;
      align-items:center;
      padding-left: 28px;
      background:#2a2a2a;
      border-top: 2px solid #4d4d4d;
      border-bottom: 8px solid #cfcfcf;
      color:var(--yellow);
      font-size:22px;
      font-weight:900;
    }}
    .butterfly {{
      width:30px;
      height:30px;
      display:inline-block;
      background: conic-gradient(from 45deg,#3d8cff,#fff 25%,#3d8cff 50%,#fff 75%,#3d8cff);
      border-radius:7px 15px;
      transform: rotate(45deg);
    }}
    .main {{
      padding: 16px 16px 8px;
      overflow-x:auto;
    }}
    .game-title {{
      display:grid;
      grid-template-columns: 210px 1fr 260px;
      background:#d1d1d1;
      border: 1px solid var(--grid);
      font-weight:900;
      font-size:20px;
      text-align:center;
    }}
    .game-title div {{ padding: 4px 6px; border-right:1px solid var(--grid); }}
    .totals-head {{
      display:grid;
      grid-template-columns: repeat(5, 52px);
      gap:0;
    }}
    .pm-grid {{
      width: 100%;
      min-width: 1148px;
      border-left:1px solid var(--grid);
      border-right:1px solid var(--grid);
    }}
    .pm-row {{
      display:grid;
      grid-template-columns: 210px 1fr repeat(5, 52px);
      min-height: 28px;
      align-items:stretch;
      font-size:18px;
      font-weight:700;
    }}
    .player-name {{
      padding: 3px 8px 2px 10px;
      background:#fff;
      border-bottom:1px solid var(--grid);
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
    }}
    .timeline {{
      position:relative;
      overflow:hidden;
      background: var(--off);
      border-bottom:1px solid var(--grid);
      border-left:1px solid var(--grid);
      height:28px;
    }}
    .timeline::before {{
      content:"";
      position:absolute;
      inset:0;
      background:
        linear-gradient(90deg, transparent calc(25% - 1px), #fff calc(25% - 1px), #fff 25%, transparent 25%),
        linear-gradient(90deg, transparent calc(50% - 1px), #fff calc(50% - 1px), #fff 50%, transparent 50%),
        linear-gradient(90deg, transparent calc(75% - 1px), #fff calc(75% - 1px), #fff 75%, transparent 75%);
      z-index:1;
      pointer-events:none;
    }}
    .quarter-label {{
      position:absolute;
      top:0;
      bottom:0;
      display:flex;
      align-items:center;
      justify-content:center;
      color:#111;
      opacity:.16;
      font-size:16px;
      z-index:1;
      pointer-events:none;
    }}
    .stint {{
      position:absolute;
      top:0;
      bottom:0;
      background:var(--on);
      z-index:2;
    }}
    .stint.sas {{ background:#151515; }}
    .stint-label {{
      position:absolute;
      z-index:4;
      top:3px;
      transform:none;
      min-width:20px;
      height:22px;
      padding: 0 4px;
      color:#fff;
      text-shadow: 0 1px 1px #000;
      text-align:center;
      line-height:22px;
      font-size:18px;
      white-space:nowrap;
      overflow:hidden;
      pointer-events:none;
    }}
    .total-cell {{
      display:flex;
      align-items:center;
      justify-content:center;
      color:#fff;
      background:#087f48;
      border-bottom:1px solid var(--grid);
      border-left:1px solid var(--grid);
      font-size:18px;
    }}
    .sas-row .total-cell {{ background:#111; }}
    .lineup-row .player-name {{
      text-align:right;
      padding-right:8px;
      background:#fff;
    }}
    .lineup-row .timeline {{ background:#fff; height:26px; }}
    .lineup-cell {{
      position:absolute;
      top:0;
      bottom:0;
      display:flex;
      align-items:center;
      justify-content:center;
      color:#fff;
      font-weight:900;
      border-left:1px solid #fff;
      z-index:3;
      overflow:hidden;
      font-size:16px;
    }}
    .lineup-cell.okc {{ background:#006bb6; }}
    .lineup-cell.sas {{ background:#151515; }}
    .blank {{ background:#fff !important; }}
    .flow-band {{
      display:grid;
      grid-template-columns: 210px 1fr 210px;
      align-items:center;
      min-height:146px;
    }}
    .team-axis {{
      text-align:right;
      padding-right:8px;
      font-weight:900;
      font-size:19px;
      line-height:1.25;
    }}
    .flow-svg {{
      width:100%;
      height:146px;
      border:2px solid #222;
      background:#fff;
    }}
    .zero {{ stroke:#808080; stroke-width:2.5; }}
    .qline {{ stroke:#777; stroke-width:2; opacity:.75; }}
    .margin-line {{ fill:none; stroke:#151515; stroke-width:3.2; vector-effect:non-scaling-stroke; }}
    .svg-label, .svg-score {{ fill:#111; font: 700 18px Arial, sans-serif; }}
    .avg-lead {{
      padding-left:22px;
      font-size:20px;
      font-weight:900;
    }}
    .period-table {{
      margin: 26px 0 6px 210px;
      width: calc(100% - 210px);
      border-collapse:collapse;
      font-size:18px;
      font-weight:800;
      table-layout:fixed;
    }}
    .period-table caption {{
      background:#d2d2d2;
      padding:6px;
      caption-side:top;
      font-weight:900;
    }}
    .period-table th {{
      width:70px;
      text-align:right;
      padding-right:8px;
      background:#fff;
      border:1px solid #fff;
    }}
    .period-table td {{
      background:#050505;
      color:#fff;
      border:1px solid #777;
      padding:5px 9px;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
    }}
    .period-table span {{ color:#dedede; font-weight:700; }}
    .footer {{
      text-align:center;
      padding: 8px 0 16px;
      font-size:16px;
      font-weight:700;
    }}
    .source-note {{
      margin: 12px 0 0 210px;
      color:#555;
      font-size:13px;
      font-weight:700;
    }}
    @media print {{
      body {{ background:white; }}
      .page {{ box-shadow:none; margin:0; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>PopcornMachine.Net</h1>
      <div class="quote">"He put him into the popcorn machine...<br>he's got salt and butter all over him"<br>-Chick Hearn</div>
      <span class="pop" style="left:310px;top:76px"></span>
      <span class="pop" style="left:930px;top:58px"></span>
      <span class="pop" style="left:1330px;top:112px"></span>
      <span class="pop" style="left:1460px;top:152px"></span>
      <span class="pop" style="left:840px;top:160px"></span>
      <span class="pop" style="left:1120px;top:176px"></span>
      <span class="pop" style="left:1720px;top:160px"></span>
    </div>
    <div class="nav"><span>HOME</span><span>HELP</span><span class="butterfly"></span><span>BoxScore</span><span>Close Window</span></div>
    <main class="main">
      <div class="pm-grid">
        <div class="game-title">
          <div style="text-align:left">Player</div>
          <div>{esc(title)}</div>
          <div class="totals-head"><span>Min</span><span>Pts</span><span>Reb</span><span>Ast</span><span>+/-</span></div>
        </div>
        <div class="pm-row">
          <div class="player-name" style="text-align:right">Team</div>
          <div class="timeline">{render_quarter_headers()}</div>
          <div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div>
        </div>
        {top_rows}
        {render_lineup_row(top_team, lineup_segments[top_team])}
        <div class="flow-band">
          <div class="team-axis"><div>Thunder</div><div>(+15)</div></div>
          {render_svg(flow)}
          <div class="avg-lead">Avg Lead {flow['avg_margin']:.2f}</div>
        </div>
        {render_lineup_row(bottom_team, lineup_segments[bottom_team])}
        {bottom_rows}
        <div class="pm-row">
          <div class="player-name"></div>
          <div class="timeline">{render_quarter_headers()}</div>
          <div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div><div class="total-cell blank"></div>
        </div>
        {render_period_table(period_stats)}
        <div class="source-note">{esc(data_note)}</div>
      </div>
    </main>
    <div class="footer">(c) 2003-2026 <span style="color:#003dc6;text-decoration:underline">PopcornMachine.net</span> style recreation for local analysis.</div>
  </div>
</body>
</html>
"""


def build(raw_path: Path, processed_path: Path, player_map: Path, game_id: str, output: Path) -> None:
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["game_id"].map(norm_game_id) == norm_game_id(game_id)].copy()
    if raw.empty:
        raise SystemExit(f"No raw rows found for game {game_id} in {raw_path}")
    raw["elapsed"] = raw.apply(clock_to_elapsed, axis=1)
    raw = raw.sort_values(["period", "elapsed", "event_num"]).reset_index(drop=True)

    processed = pd.read_parquet(processed_path)
    processed = processed[processed["game_id"].map(norm_game_id) == norm_game_id(game_id)].copy()
    if processed.empty:
        raise SystemExit(f"No processed rows found for game {game_id} in {processed_path}")

    names = load_player_names(player_map)
    for _, row in raw.iterrows():
        for id_col, name_col in (
            ("player1_id", "player1_name"),
            ("player2_id", "player2_name"),
            ("player3_id", "player3_name"),
        ):
            if pd.notna(row.get(id_col)) and isinstance(row.get(name_col), str) and row.get(name_col):
                names.setdefault(int(row[id_col]), row[name_col])

    rosters, player_team = detect_team_rosters(raw)
    scoring = scoring_events(raw, player_team)
    intervals = build_intervals(raw)
    stints, minutes = build_stints(intervals, rosters)
    box = player_box(raw, scoring, rosters, names)
    period_stats = team_period_stats(raw, scoring)
    lineup_segments = {team: team_lineup_segments(team, intervals, scoring) for team in ("OKC", "SAS")}
    html_doc = render_html(raw, processed, names, rosters, stints, minutes, box, scoring, lineup_segments, period_stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--processed", type=Path, default=RAPM_DEFAULT)
    parser.add_argument("--player-map", type=Path, default=PLAYER_MAP_DEFAULT)
    parser.add_argument("--game-id", default="0042500313")
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    build(args.raw, args.processed, args.player_map, args.game_id, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

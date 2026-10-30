#!/usr/bin/env python3
"""Build a self-contained 3P difficulty table from SSI qSQ player seasons."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QSQ_CSV = Path("/Users/russ/Downloads/3p_shot_sq.csv")
DEFAULT_RUSSELL_V2_COUNTERFACTUAL = (
    PIPELINE_ROOT
    / "results"
    / "sq_three_point_difficulty"
    / "russell_v2_cf"
    / "russell_v2_counterfactual_player_season.csv"
)
DEFAULT_SQ_RICH_COUNTERFACTUAL = (
    PIPELINE_ROOT
    / "results"
    / "sq_three_point_difficulty"
    / "sq_rich_cf"
    / "sq_rich_counterfactual_player_season.csv"
)
DEFAULT_PLAYER_STATS_CSV = Path(
    "/Users/russ/Documents/Codex/basketball/data/csvs/player_stats_export.csv"
)
DEFAULT_OUTPUT = (
    PIPELINE_ROOT
    / "results"
    / "sq_three_point_difficulty"
    / "three_point_difficulty_table.html"
)


def _name_key(value: object) -> str:
    text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    text = re.sub(r"[^a-z0-9]", "", text)
    aliases = {
        "kjmartin": "kenyonmartin",
        "carltoncarrington": "bubcarrington",
        "gregoryggjackson": "ggjackson",
        "alexandresarr": "alexsarr",
    }
    return aliases.get(text, text)


def _nba_name_map(player_stats_csv: Path) -> pd.DataFrame:
    stats = pd.read_csv(
        player_stats_csv, usecols=["year", "nba_id", "ShortName"]
    )
    stats["name_key"] = stats["ShortName"].map(_name_key)
    stats = stats.rename(columns={"year": "season_end_year"})
    stats = stats.dropna(subset=["nba_id"])
    stats = stats.drop_duplicates(["season_end_year", "name_key"])

    # The local stats export begins in 2022, while frozen v2 begins in 2021.
    # Fill the earlier season from nba_api's static, network-free player list.
    try:
        from nba_api.stats.static import players as nba_players

        static = pd.DataFrame(nba_players.get_players())[["id", "full_name"]]
        static["name_key"] = static["full_name"].map(_name_key)
        unique = static.groupby("name_key")["id"].transform("nunique").eq(1)
        static = static[unique].rename(columns={"id": "nba_id"})
        static["season_end_year"] = 2021
        stats = pd.concat(
            [
                stats,
                static[["season_end_year", "name_key", "nba_id"]],
            ],
            ignore_index=True,
        ).drop_duplicates(["season_end_year", "name_key"], keep="first")
    except ImportError:
        pass
    return stats[["season_end_year", "name_key", "nba_id"]]


def _records(
    qsq_csv: Path,
    russell_v2_counterfactual: Path | None = None,
    player_stats_csv: Path | None = None,
    sq_rich_counterfactual: Path | None = None,
) -> list[dict[str, object]]:
    data = pd.read_csv(qsq_csv)
    required = {
        "Shooter",
        "Season",
        "Shots Attempted",
        "Shots Attempted Per 100 Possessions",
        "eFG%",
        "qSQ",
        "qSI",
        "qSP",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.duplicated(["SSI Player ID", "Season"]).any():
        raise ValueError("Duplicate SSI player-season rows found")

    identity_gap = (data["eFG%"] - data["qSQ"] - data["qSI"]).abs().max()
    if float(identity_gap) > 0.011:
        raise ValueError(f"eFG = qSQ + qSI identity failed: {identity_gap}")

    seasonal_qsq = (
        (data["qSQ"] * data["Shots Attempted"])
        .groupby(data["Season"])
        .sum()
        / data.groupby("Season")["Shots Attempted"].sum()
    )
    data = data.assign(
        seasonal_qsq=data["Season"].map(seasonal_qsq),
        season=data["Season"],
        player=data["Shooter"],
        player_id=data["SSI Player ID"],
        attempts=data["Shots Attempted"].astype(int),
        attempts_per_100=data["Shots Attempted Per 100 Possessions"],
        raw=data["eFG%"] / 150.0,
        expected=data["qSQ"] / 150.0,
        shotmaking=data["qSI"] / 150.0,
        qsp=data["qSP"] / 150.0,
        baseline=data["Season"].map(seasonal_qsq) / 150.0,
    )
    data["difficulty"] = data["baseline"] - data["expected"]
    data["effective"] = data["raw"] + data["difficulty"]
    data["v2_difficulty"] = np.nan
    data["v2_effective"] = np.nan
    data["v2_expected"] = np.nan
    data["v2_baseline"] = np.nan
    data["v2_attempts"] = np.nan
    data["sq_rich_difficulty"] = np.nan
    data["sq_rich_effective"] = np.nan
    data["sq_rich_expected"] = np.nan
    data["sq_rich_baseline"] = np.nan
    data["sq_rich_attempts"] = np.nan
    data["sq_rich_geometry_coverage"] = np.nan
    data["sq_rich_geometry_only_difficulty"] = np.nan

    season_suffix = data["Season"].astype(str).str.extract(r"(\d{2})$")[0]
    data["season_end_year"] = 2000 + pd.to_numeric(
        season_suffix, errors="coerce"
    )
    data["name_key"] = data["Shooter"].map(_name_key)

    if russell_v2_counterfactual is not None and player_stats_csv is not None:
        name_map = _nba_name_map(player_stats_csv)
        data = data.merge(
            name_map,
            on=["season_end_year", "name_key"],
            how="left",
            validate="many_to_one",
        )
        v2 = pd.read_csv(russell_v2_counterfactual)
        required_v2 = {
            "season_end_year",
            "shooter_id",
            "v2_attempts",
            "v2_neutral_expected",
            "v2_neutral_baseline",
            "v2_difficulty",
        }
        missing_v2 = sorted(required_v2 - set(v2.columns))
        if missing_v2:
            raise ValueError(
                f"Russell v2 counterfactual is missing columns: {missing_v2}"
            )
        v2 = v2.rename(
            columns={
                "v2_neutral_expected": "v2_expected",
                "v2_neutral_baseline": "v2_baseline",
            }
        )
        data = data.merge(
            v2,
            left_on=["season_end_year", "nba_id"],
            right_on=["season_end_year", "shooter_id"],
            how="left",
            suffixes=("", "_joined"),
            validate="many_to_one",
        )
        for column in [
            "v2_difficulty",
            "v2_expected",
            "v2_baseline",
            "v2_attempts",
        ]:
            joined = f"{column}_joined"
            if joined in data:
                data[column] = data[joined]
        data["v2_effective"] = data["raw"] + data["v2_difficulty"]

    if sq_rich_counterfactual is not None:
        rich = pd.read_csv(sq_rich_counterfactual)
        required_rich = {
            "season_end_year",
            "shooter_name",
            "sq_rich_attempts",
            "sq_rich_neutral_expected",
            "sq_rich_neutral_baseline",
            "sq_rich_difficulty",
            "sq_rich_geometry_coverage",
            "sq_rich_geometry_only_difficulty",
        }
        missing_rich = sorted(required_rich - set(rich.columns))
        if missing_rich:
            raise ValueError(
                f"SQ-rich counterfactual is missing columns: {missing_rich}"
            )
        rich["name_key"] = rich["shooter_name"].map(_name_key)
        rich = rich.rename(
            columns={
                "sq_rich_neutral_expected": "sq_rich_expected",
                "sq_rich_neutral_baseline": "sq_rich_baseline",
            }
        )
        rich = rich.drop_duplicates(["season_end_year", "name_key"])
        data = data.merge(
            rich[
                [
                    "season_end_year",
                    "name_key",
                    "sq_rich_attempts",
                    "sq_rich_expected",
                    "sq_rich_baseline",
                    "sq_rich_difficulty",
                    "sq_rich_geometry_coverage",
                    "sq_rich_geometry_only_difficulty",
                ]
            ],
            on=["season_end_year", "name_key"],
            how="left",
            suffixes=("", "_joined"),
            validate="many_to_one",
        )
        for column in [
            "sq_rich_attempts",
            "sq_rich_expected",
            "sq_rich_baseline",
            "sq_rich_difficulty",
            "sq_rich_geometry_coverage",
            "sq_rich_geometry_only_difficulty",
        ]:
            joined = f"{column}_joined"
            if joined in data:
                data[column] = data[joined]
        data["sq_rich_effective"] = data["raw"] + data["sq_rich_difficulty"]
    columns = [
        "season",
        "player",
        "player_id",
        "attempts",
        "attempts_per_100",
        "raw",
        "difficulty",
        "v2_difficulty",
        "effective",
        "v2_effective",
        "expected",
        "v2_expected",
        "shotmaking",
        "qsp",
        "baseline",
        "v2_baseline",
        "v2_attempts",
        "sq_rich_difficulty",
        "sq_rich_effective",
        "sq_rich_expected",
        "sq_rich_baseline",
        "sq_rich_attempts",
        "sq_rich_geometry_coverage",
        "sq_rich_geometry_only_difficulty",
    ]
    return data[columns].to_dict("records")


def render_html(records: list[dict[str, object]]) -> str:
    payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", payload)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NBA 3P Shot Difficulty</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f4f0e8;
      --card: #fffdf8;
      --ink: #151515;
      --muted: #6e6a62;
      --line: #d8d1c5;
      --hard: #176b45;
      --easy: #a43f32;
      --accent: #d7ff3f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-variant-numeric: tabular-nums;
    }
    main { width: min(1420px, calc(100% - 32px)); margin: 36px auto 64px; }
    header { display: grid; gap: 12px; margin-bottom: 22px; }
    .eyebrow {
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .13em;
      text-transform: uppercase;
    }
    h1 { margin: 0; font-size: clamp(34px, 6vw, 68px); line-height: .93; letter-spacing: -.055em; }
    .equation { margin: 0; font-size: clamp(18px, 2.5vw, 28px); font-weight: 750; }
    .equation mark { background: var(--accent); padding: 2px 7px; color: inherit; }
    .note { margin: 0; color: var(--muted); max-width: 850px; line-height: 1.5; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      padding: 14px;
      background: var(--card);
      border: 1px solid var(--line);
      border-bottom: 0;
    }
    button, input, select {
      min-height: 40px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      font: inherit;
    }
    button { padding: 8px 12px; cursor: pointer; font-weight: 700; }
    label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; font-weight: 700; }
    input, select { padding: 8px 10px; color: var(--ink); }
    #search { width: min(250px, 48vw); }
    #minimum { width: 90px; }
    .summary { margin-left: auto; padding: 0 4px 8px; font-size: 13px; color: var(--muted); }
    .table-wrap {
      max-height: calc(100vh - 300px);
      min-height: 380px;
      overflow: auto;
      background: var(--card);
      border: 1px solid var(--line);
    }
    table { width: 100%; min-width: 1480px; border-collapse: collapse; }
    thead { position: sticky; top: 0; z-index: 2; background: var(--ink); color: #fff; }
    th, td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: right; }
    th { padding: 0; border-color: #3e3e3e; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
    th button {
      width: 100%;
      padding: 12px;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: inherit;
      text-align: right;
      font-size: inherit;
      letter-spacing: inherit;
      text-transform: inherit;
    }
    th button::after { content: "  ↕"; color: #999; }
    th button.sorted-asc::after { content: "  ↑"; color: var(--accent); }
    th button.sorted-desc::after { content: "  ↓"; color: var(--accent); }
    th.player button, td.player { text-align: left; }
    td.rank { width: 44px; color: var(--muted); }
    td.player { min-width: 180px; font-weight: 750; }
    tbody tr:hover { background: #f5f8e9; }
    tbody tr { cursor: pointer; }
    tbody tr:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
    tbody tr.selected { background: #edf3d4; }
    td.difficulty { font-weight: 850; }
    td.hard { color: var(--hard); }
    td.easy { color: var(--easy); }
    td.effective { background: color-mix(in srgb, var(--accent) 16%, transparent); font-weight: 850; }
    .legend { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; font-size: 13px; color: var(--muted); }
    .legend b { color: var(--ink); }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
    .dot.hard { background: var(--hard); }
    .dot.easy { background: var(--easy); }
    .drawer-backdrop {
      position: fixed;
      inset: 0;
      z-index: 20;
      background: rgb(0 0 0 / .35);
      opacity: 0;
      pointer-events: none;
      transition: opacity 180ms ease;
    }
    .drawer-backdrop.open { opacity: 1; pointer-events: auto; }
    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      z-index: 21;
      width: min(820px, 94vw);
      padding: 24px;
      overflow: auto;
      background: var(--card);
      border-left: 1px solid var(--line);
      box-shadow: -24px 0 70px rgb(0 0 0 / .18);
      transform: translateX(102%);
      transition: transform 220ms cubic-bezier(.2,.8,.2,1);
    }
    .drawer.open { transform: translateX(0); }
    .drawer-head { display: flex; align-items: start; justify-content: space-between; gap: 18px; margin-bottom: 18px; }
    .drawer h2 { margin: 4px 0 0; font-size: clamp(30px, 5vw, 48px); line-height: 1; letter-spacing: -.045em; }
    .drawer-close { width: 42px; min-width: 42px; padding: 0; font-size: 25px; line-height: 1; }
    .drawer-copy { color: var(--muted); line-height: 1.45; margin: 0 0 18px; }
    .drawer-table-wrap { overflow-x: auto; border: 1px solid var(--line); }
    .drawer table { min-width: 1060px; }
    .drawer thead { position: static; }
    .drawer th { padding: 11px 10px; text-align: right; }
    .drawer th:first-child, .drawer td:first-child { text-align: left; }
    .drawer td { padding: 12px 10px; }
    body.drawer-open { overflow: hidden; }
    @media (max-width: 720px) {
      main { width: min(100% - 18px, 1420px); margin-top: 22px; }
      .summary { width: 100%; margin-left: 0; padding-bottom: 0; }
      .table-wrap { max-height: calc(100vh - 350px); }
      .drawer { width: 100%; padding: 18px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">SSI qSQ + RUSSELL_SHOTQUALITY v2 · NBA regular season</div>
    <h1>3P shot difficulty</h1>
    <p class="equation">Raw 3P% <mark>+ Difficulty</mark> = Effective 3P%</p>
    <p class="note">qSQ uses the provider's shooter-neutral context expectation. Russell v2 CF is the PBP-only architecture rebuild. SQ rich CF is the 2025-26 season-forward challenger: it adds release-time contest, closest/second-closest defender geometry, spacing, and ShotQuality distance when available, then falls back to a context-only model. Shooter history is neutralized in both arms. Geometry coverage is shown because tracking exists for only part of the season.</p>
  </header>

  <section class="controls" aria-label="Table filters">
    <label>Season<select id="season"></select></label>
    <label>Find player<input id="search" type="search" placeholder="Search name…"></label>
    <label>Minimum 3PA<input id="minimum" type="number" min="0" step="25" value="100"></label>
    <div class="summary" id="summary" aria-live="polite"></div>
  </section>

  <div class="table-wrap">
    <table>
      <thead><tr>
        <th><button type="button" data-key="rank">#</button></th>
        <th class="player"><button type="button" data-key="player">Player</button></th>
        <th><button type="button" data-key="attempts">3PA</button></th>
        <th><button type="button" data-key="attempts_per_100">3PA / 100</button></th>
        <th><button type="button" data-key="raw">Raw 3P%</button></th>
        <th><button type="button" data-key="difficulty">qSQ difficulty</button></th>
        <th><button type="button" data-key="v2_difficulty">Russell v2 CF difficulty</button></th>
        <th><button type="button" data-key="sq_rich_difficulty">SQ rich CF difficulty</button></th>
        <th><button type="button" data-key="sq_rich_geometry_coverage">Geometry</button></th>
        <th><button type="button" data-key="effective" class="sorted-desc">qSQ effective 3P%</button></th>
        <th><button type="button" data-key="v2_effective">V2 CF effective 3P%</button></th>
        <th><button type="button" data-key="sq_rich_effective">SQ rich effective 3P%</button></th>
        <th><button type="button" data-key="expected">qSQ expected 3P%</button></th>
        <th><button type="button" data-key="shotmaking">qSI shotmaking</button></th>
        <th><button type="button" data-key="qsp">qSP on 3P scale</button></th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <div class="legend">
    <span><i class="dot hard"></i><b>Positive difficulty:</b> harder shots</span>
    <span><i class="dot easy"></i><b>Negative difficulty:</b> easier shots</span>
    <span><b>V2 CF:</b> current-input reconstruction of the v2 counterfactual, not a mutation of the accepted frozen VPM target.</span>
    <span><b>SQ rich CF:</b> trained on 2024-25, scored on 2025-26; rich geometry where present and context-only fallback elsewhere.</span>
  </div>
</main>
<div class="drawer-backdrop" id="drawer-backdrop"></div>
<aside class="drawer" id="player-drawer" role="dialog" aria-modal="true" aria-hidden="true" aria-labelledby="drawer-player-name">
  <div class="drawer-head">
    <div>
      <div class="eyebrow">Year-by-year 3P profile</div>
      <h2 id="drawer-player-name"></h2>
    </div>
    <button class="drawer-close" id="drawer-close" type="button" aria-label="Close player details">×</button>
  </div>
  <p class="drawer-copy">Positive difficulty means the player's shooter-neutral expected 3P% was below that season's league-average expectation. SQ rich CF holds defender geometry fixed, neutralizes prior-game shooter history, and uses a context-only fallback on shots without tracking.</p>
  <div class="drawer-table-wrap">
    <table>
      <thead><tr>
        <th>Season</th>
        <th>3PA</th>
        <th>3PA / 100</th>
        <th>Raw 3P%</th>
        <th>qSQ difficulty</th>
        <th>Russell v2 CF difficulty</th>
        <th>SQ rich CF difficulty</th>
        <th>Geometry-only difficulty</th>
        <th>Geometry coverage</th>
        <th>qSQ effective</th>
        <th>V2 CF effective</th>
        <th>SQ rich effective</th>
      </tr></thead>
      <tbody id="drawer-rows"></tbody>
    </table>
  </div>
</aside>
<script>
const DATA = __DATA__;
const seasons = [...new Set(DATA.map(row => row.season))].sort((a, b) => b.localeCompare(a));
const state = { season: seasons[0], search: "", minimum: 100, sortKey: "effective", direction: -1 };
const tbody = document.getElementById("rows");
const summary = document.getElementById("summary");
const number = new Intl.NumberFormat("en-US");
const seasonSelect = document.getElementById("season");
const drawer = document.getElementById("player-drawer");
const drawerBackdrop = document.getElementById("drawer-backdrop");
const drawerRows = document.getElementById("drawer-rows");
const drawerName = document.getElementById("drawer-player-name");
const drawerClose = document.getElementById("drawer-close");
let selectedPlayerId = null;
seasonSelect.innerHTML = seasons.map(season => `<option value="${season}">${season}</option>`).join("");

function percent(value) { return `${(100 * value).toFixed(1)}%`; }
function isNumber(value) { return Number.isFinite(value); }
function optionalPercent(value) { return isNumber(value) ? percent(value) : "—"; }
function optionalCoverage(value) { return isNumber(value) ? `${Math.round(100 * value)}%` : "—"; }
function signedPoints(value) {
  if (!isNumber(value)) return "—";
  const points = 100 * value;
  if (Math.abs(points) < 0.05) return "0.0 pp";
  return `${points > 0 ? "+" : "−"}${Math.abs(points).toFixed(1)} pp`;
}
function filtered() {
  const query = state.search.trim().toLocaleLowerCase();
  return DATA.filter(row => row.season === state.season)
    .filter(row => row.attempts >= state.minimum)
    .filter(row => !query || row.player.toLocaleLowerCase().includes(query));
}
function sorted(rows) {
  return rows.sort((a, b) => {
    if (state.sortKey === "rank") return 0;
    const left = a[state.sortKey];
    const right = b[state.sortKey];
    const leftMissing = left === null || (typeof left === "number" && !Number.isFinite(left));
    const rightMissing = right === null || (typeof right === "number" && !Number.isFinite(right));
    if (leftMissing && rightMissing) return 0;
    if (leftMissing) return 1;
    if (rightMissing) return -1;
    if (typeof left === "string") return state.direction * left.localeCompare(right);
    return state.direction * (left - right);
  });
}
function render() {
  const rows = sorted(filtered());
  tbody.innerHTML = rows.map((row, index) => {
    const difficultyClass = row.difficulty > .0005 ? "hard" : row.difficulty < -.0005 ? "easy" : "";
    const v2DifficultyClass = row.v2_difficulty > .0005 ? "hard" : row.v2_difficulty < -.0005 ? "easy" : "";
    const richDifficultyClass = row.sq_rich_difficulty > .0005 ? "hard" : row.sq_rich_difficulty < -.0005 ? "easy" : "";
    const geometryOnlyClass = row.sq_rich_geometry_only_difficulty > .0005 ? "hard" : row.sq_rich_geometry_only_difficulty < -.0005 ? "easy" : "";
    return `<tr data-player-id="${escapeHtml(row.player_id)}" tabindex="0" role="button" aria-label="Open year-by-year details for ${escapeHtml(row.player)}" class="${row.player_id === selectedPlayerId ? "selected" : ""}">
      <td class="rank">${index + 1}</td>
      <td class="player">${escapeHtml(row.player)}</td>
      <td>${number.format(row.attempts)}</td>
      <td>${row.attempts_per_100.toFixed(1)}</td>
      <td>${percent(row.raw)}</td>
      <td class="difficulty ${difficultyClass}" title="Raw ${percent(row.raw)} ${row.difficulty >= 0 ? "+" : "−"} ${Math.abs(100 * row.difficulty).toFixed(3)} percentage points">${signedPoints(row.difficulty)}</td>
      <td class="difficulty ${v2DifficultyClass}" title="Two-score v2 architecture counterfactual with shooter talent set to league average">${signedPoints(row.v2_difficulty)}</td>
      <td class="difficulty ${richDifficultyClass}" title="Season-forward ShotQuality geometry counterfactual with context-only fallback">${signedPoints(row.sq_rich_difficulty)}</td>
      <td title="Share with closest-defender geometry; covered-shot difficulty ${signedPoints(row.sq_rich_geometry_only_difficulty)}">${optionalCoverage(row.sq_rich_geometry_coverage)}</td>
      <td class="effective">${percent(row.effective)}</td>
      <td class="effective">${optionalPercent(row.v2_effective)}</td>
      <td class="effective">${optionalPercent(row.sq_rich_effective)}</td>
      <td>${percent(row.expected)}</td>
      <td>${signedPoints(row.shotmaking)}</td>
      <td>${percent(row.qsp)}</td>
    </tr>`;
  }).join("");
  const baseline = DATA.find(row => row.season === state.season)?.baseline;
  const v2Covered = rows.filter(row => isNumber(row.v2_difficulty)).length;
  const richCovered = rows.filter(row => isNumber(row.sq_rich_difficulty)).length;
  summary.textContent = `${number.format(rows.length)} players · V2 ${number.format(v2Covered)} covered · SQ rich ${number.format(richCovered)} covered · season-average qSQ ${percent(baseline)} · sorted by ${labelFor(state.sortKey)}`;
}
function openPlayerDrawer(playerId) {
  const history = DATA.filter(row => row.player_id === playerId)
    .sort((a, b) => b.season.localeCompare(a.season));
  if (!history.length) return;
  selectedPlayerId = playerId;
  drawerName.textContent = history[0].player;
  drawerRows.innerHTML = history.map(row => {
    const difficultyClass = row.difficulty > .0005 ? "hard" : row.difficulty < -.0005 ? "easy" : "";
    const v2DifficultyClass = row.v2_difficulty > .0005 ? "hard" : row.v2_difficulty < -.0005 ? "easy" : "";
    const richDifficultyClass = row.sq_rich_difficulty > .0005 ? "hard" : row.sq_rich_difficulty < -.0005 ? "easy" : "";
    const geometryOnlyClass = row.sq_rich_geometry_only_difficulty > .0005 ? "hard" : row.sq_rich_geometry_only_difficulty < -.0005 ? "easy" : "";
    return `<tr>
      <td><b>${escapeHtml(row.season)}</b></td>
      <td>${number.format(row.attempts)}</td>
      <td>${row.attempts_per_100.toFixed(1)}</td>
      <td>${percent(row.raw)}</td>
      <td class="difficulty ${difficultyClass}">${signedPoints(row.difficulty)}</td>
      <td class="difficulty ${v2DifficultyClass}">${signedPoints(row.v2_difficulty)}</td>
      <td class="difficulty ${richDifficultyClass}">${signedPoints(row.sq_rich_difficulty)}</td>
      <td class="difficulty ${geometryOnlyClass}">${signedPoints(row.sq_rich_geometry_only_difficulty)}</td>
      <td>${optionalCoverage(row.sq_rich_geometry_coverage)}</td>
      <td class="effective">${percent(row.effective)}</td>
      <td class="effective">${optionalPercent(row.v2_effective)}</td>
      <td class="effective">${optionalPercent(row.sq_rich_effective)}</td>
    </tr>`;
  }).join("");
  drawer.classList.add("open");
  drawerBackdrop.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.body.classList.add("drawer-open");
  render();
  drawerClose.focus();
}
function closePlayerDrawer() {
  drawer.classList.remove("open");
  drawerBackdrop.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  document.body.classList.remove("drawer-open");
  selectedPlayerId = null;
  render();
}
function labelFor(key) {
  const labels = { rank: "rank", player: "player", attempts: "3PA", attempts_per_100: "3PA / 100", raw: "raw 3P%", difficulty: "qSQ difficulty", v2_difficulty: "Russell v2 CF difficulty", sq_rich_difficulty: "SQ rich CF difficulty", sq_rich_geometry_coverage: "geometry coverage", effective: "qSQ effective 3P%", v2_effective: "V2 CF effective 3P%", sq_rich_effective: "SQ rich effective 3P%", expected: "qSQ expected 3P%", shotmaking: "qSI shotmaking", qsp: "qSP" };
  return labels[key];
}
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
seasonSelect.addEventListener("change", event => { state.season = event.target.value; render(); });
document.getElementById("search").addEventListener("input", event => { state.search = event.target.value; render(); });
document.getElementById("minimum").addEventListener("input", event => { state.minimum = Math.max(0, Number(event.target.value) || 0); render(); });
tbody.addEventListener("click", event => {
  const row = event.target.closest("tr[data-player-id]");
  if (row) openPlayerDrawer(row.dataset.playerId);
});
tbody.addEventListener("keydown", event => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest("tr[data-player-id]");
  if (!row) return;
  event.preventDefault();
  openPlayerDrawer(row.dataset.playerId);
});
drawerClose.addEventListener("click", closePlayerDrawer);
drawerBackdrop.addEventListener("click", closePlayerDrawer);
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && drawer.classList.contains("open")) closePlayerDrawer();
});
document.querySelector("thead").addEventListener("click", event => {
  const button = event.target.closest("button[data-key]");
  if (!button || button.dataset.key === "rank") return;
  const key = button.dataset.key;
  state.direction = state.sortKey === key ? -state.direction : (key === "player" ? 1 : -1);
  state.sortKey = key;
  document.querySelectorAll("thead button").forEach(item => item.classList.remove("sorted-asc", "sorted-desc"));
  button.classList.add(state.direction === 1 ? "sorted-asc" : "sorted-desc");
  render();
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qsq-csv", type=Path, default=DEFAULT_QSQ_CSV)
    parser.add_argument(
        "--russell-v2-counterfactual",
        type=Path,
        default=DEFAULT_RUSSELL_V2_COUNTERFACTUAL,
    )
    parser.add_argument(
        "--player-stats-csv", type=Path, default=DEFAULT_PLAYER_STATS_CSV
    )
    parser.add_argument(
        "--sq-rich-counterfactual",
        type=Path,
        default=DEFAULT_SQ_RICH_COUNTERFACTUAL,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    html = render_html(
        _records(
            args.qsq_csv,
            russell_v2_counterfactual=args.russell_v2_counterfactual,
            player_stats_csv=args.player_stats_csv,
            sq_rich_counterfactual=args.sq_rich_counterfactual,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()

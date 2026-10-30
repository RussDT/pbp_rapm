#!/usr/bin/env python3
"""Build a standalone historical scoring-impact leaderboard.

The headline view deliberately excludes bad-pass turnover value:

    scoring impact (no bad-pass) = scoring value + scoring-TOV value

Bad-pass turnover value remains visible and adding it back closes exactly to
the full scoring security value. Every row expands into its own arithmetic so a
reader can reproduce each number from the published inputs.

``playtype_mix_rapm`` is one descriptive research column: the two-season,
550-day time-decay lineup shot-diet value in points per 100 possessions. It
never enters scoring value, turnover value, or scoring security.

The ``lineup_*_tov_*`` columns are the transparency view of the turnover lanes:
a two-season TD550 lineup ridge of on-court turnovers split into the player's
own, teammates', and unassigned team turnovers, priced in the same pts/100 as
the box ledger. They are shown beside the box values, never added to them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PIPELINE_DIR
    / "results"
    / "historical_scoring_security"
    / "historical_scoring_security_1997_2026.csv"
)
DEFAULT_OUTPUT = PIPELINE_DIR / "docs" / "historical-scoring-impact.html"
PASS_THROUGH_NAME = "lineup_tov_pass_through_1997_2026.csv"
PASS_THROUGH_FIELDS = [
    "scoring_total_slope",
    "scoring_raw_slope",
    "badpass_total_slope",
    "badpass_raw_slope",
    "turnover_total_slope",
    "turnover_total_corr",
]

SV_INTERCEPT = -1.211295
SV_SLOPE = 0.800618

REQUIRED = [
    "nba_id",
    "player_name",
    "year",
    "OffPoss",
    "role_source",
    "turnover_source",
    "tsa100",
    "rts",
    "role_difficulty",
    "zts",
    "scoring_value100",
    "rztov",
    "sc_excess",
    "bp_excess",
    "scoring_tov_point_price",
    "badpass_tov_point_price",
    "scoring_tov_value100",
    "scoring_impact_no_badpass100",
    "badpass_tov_value100",
    "scoring_security_value100",
]
LINEUP_TOV = [
    f"lineup_{lane}_tov_{part}_value100"
    for lane in ("scoring", "badpass")
    for part in ("own", "teammate", "team", "total", "raw")
] + ["lineup_tov_value100", "lineup_tov_point_price", "lineup_team_tov_coverage"]
OPTIONAL = ["playtype_mix_rapm", *LINEUP_TOV]
TEXT_FIELDS = {"player_name", "role_source", "turnover_source", "lineup_team_tov_coverage"}


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NBA Scoring Impact</title>
  <meta name="description" content="Historical NBA scoring impact 1997-2026: zTS scoring value plus scoring-turnover value, with bad-pass cost, a research playtype-mix RAPM, and a time-decay lineup turnover view beside it.">
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c0d0b;
      --surface: #131511;
      --surface-2: #1a1c17;
      --surface-3: #22251e;
      --ink: #f2f0e8;
      --muted: #a2a497;
      --faint: #6f7168;
      --rule: #272a23;
      --rule-strong: #3a3e34;
      --hl: #c9f13a;
      --hl-ink: #c9f13a;
      --hl-bg: rgba(201, 241, 58, .10);
      --hl-bg-2: rgba(201, 241, 58, .17);
      --good: #7fdcb0;
      --bad: #ff8f73;
      --blue: #8fc7ff;
      --gold: #f0c56c;
      --violet: #c2adff;
      --violet-bg: rgba(194, 173, 255, .08);
      --shadow: 0 1px 0 rgba(255, 255, 255, .03), 0 12px 32px rgba(0, 0, 0, .35);
    }
    :root[data-theme="light"] {
      color-scheme: light;
      --bg: #f6f4ee;
      --surface: #ffffff;
      --surface-2: #f0ede4;
      --surface-3: #e7e3d8;
      --ink: #17180f;
      --muted: #5c5e54;
      --faint: #8d8f84;
      --rule: #e2ded3;
      --rule-strong: #cdc8ba;
      --hl: #9bc41a;
      --hl-ink: #3f5a00;
      --hl-bg: rgba(155, 196, 26, .16);
      --hl-bg-2: rgba(155, 196, 26, .26);
      --good: #16734b;
      --bad: #b23c22;
      --blue: #1d5a97;
      --gold: #8a5d00;
      --violet: #5a3dc0;
      --violet-bg: rgba(90, 61, 192, .06);
      --shadow: 0 1px 2px rgba(23, 24, 15, .06), 0 10px 28px rgba(23, 24, 15, .07);
    }
    @media (prefers-color-scheme: light) {
      :root:not([data-theme="dark"]) {
        color-scheme: light;
        --bg: #f6f4ee;
        --surface: #ffffff;
        --surface-2: #f0ede4;
        --surface-3: #e7e3d8;
        --ink: #17180f;
        --muted: #5c5e54;
        --faint: #8d8f84;
        --rule: #e2ded3;
        --rule-strong: #cdc8ba;
        --hl: #9bc41a;
        --hl-ink: #3f5a00;
        --hl-bg: rgba(155, 196, 26, .16);
        --hl-bg-2: rgba(155, 196, 26, .26);
        --good: #16734b;
        --bad: #b23c22;
        --blue: #1d5a97;
        --gold: #8a5d00;
        --violet: #5a3dc0;
        --violet-bg: rgba(90, 61, 192, .06);
        --shadow: 0 1px 2px rgba(23, 24, 15, .06), 0 10px 28px rgba(23, 24, 15, .07);
      }
    }
    * { box-sizing: border-box; }
    html { -webkit-text-size-adjust: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
      -webkit-font-smoothing: antialiased;
    }
    button, input, select { font: inherit; color: inherit; }
    :focus-visible { outline: 2px solid var(--hl); outline-offset: 2px; }
    .page { width: min(1440px, 100%); margin: 0 auto; padding: 28px 28px 40px; }

    /* Header */
    .top { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; }
    .kicker { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; margin: 0 0 10px; color: var(--muted); font-size: 11.5px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; }
    .kicker .dot { width: 3px; height: 3px; border-radius: 50%; background: var(--faint); }
    .pill { padding: 2px 8px; border: 1px solid var(--rule-strong); border-radius: 999px; color: var(--muted); font-size: 10.5px; letter-spacing: .06em; }
    h1 { margin: 0; font-size: clamp(28px, 3.6vw, 42px); line-height: 1.02; letter-spacing: -.035em; font-weight: 780; }
    h1 .soft { color: var(--muted); font-weight: 560; }
    .lede { max-width: 760px; margin: 12px 0 0; color: var(--muted); font-size: 15px; line-height: 1.55; }
    .lede strong { color: var(--ink); font-weight: 650; }
    .icon-btn {
      flex: none; display: inline-flex; align-items: center; gap: 8px; height: 34px; padding: 0 12px;
      border: 1px solid var(--rule-strong); border-radius: 8px; background: var(--surface); cursor: pointer;
      color: var(--muted); font-size: 12px; font-weight: 600;
    }
    .icon-btn:hover { color: var(--ink); border-color: var(--faint); }
    .icon-btn svg { width: 15px; height: 15px; }

    /* Formula */
    .formula { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 24px 0 0; }
    .term {
      display: flex; flex-direction: column; justify-content: center; gap: 2px; min-width: 0;
      padding: 10px 14px; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface);
    }
    .term .t-label { color: var(--faint); font-size: 10.5px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
    .term .t-value { font-size: 14px; font-weight: 680; white-space: nowrap; }
    .term.own .t-value { color: var(--blue); }
    .term.head { border-color: color-mix(in srgb, var(--hl) 55%, transparent); background: var(--hl-bg); }
    .term.head .t-label, .term.head .t-value { color: var(--hl-ink); }
    .term.full .t-value { color: var(--gold); }
    .op { display: flex; align-items: center; padding: 0 2px; color: var(--faint); font-size: 18px; font-weight: 500; }
    .formula .spacer { flex: 1 1 24px; }
    .op.lead { display: none; }
    .term.mix { border-style: dashed; border-color: color-mix(in srgb, var(--violet) 45%, transparent); background: var(--violet-bg); }
    .term.mix .t-label, .term.mix .t-value { color: var(--violet); }

    /* Controls */
    .controls {
      position: sticky; top: 0; z-index: 6; display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px 14px;
      margin: 22px 0 0; padding: 12px 0; background: var(--bg); border-bottom: 1px solid var(--rule);
    }
    .field { display: grid; gap: 5px; min-width: 0; }
    .field > span { color: var(--faint); font-size: 10.5px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
    .search-wrap { position: relative; flex: 1 1 240px; max-width: 360px; }
    .search-wrap svg { position: absolute; left: 11px; bottom: 11px; width: 15px; height: 15px; color: var(--faint); pointer-events: none; }
    input[type="search"], select {
      width: 100%; height: 38px; padding: 0 12px; border: 1px solid var(--rule-strong); border-radius: 8px;
      background: var(--surface); font-size: 14px; outline: none;
    }
    input[type="search"] { padding-left: 34px; }
    input[type="search"]::placeholder { color: var(--faint); }
    input:focus, select:focus { border-color: var(--hl); box-shadow: 0 0 0 3px var(--hl-bg); }
    select { appearance: none; padding-right: 30px; cursor: pointer;
      background-image: linear-gradient(45deg, transparent 50%, var(--muted) 50%), linear-gradient(135deg, var(--muted) 50%, transparent 50%);
      background-position: calc(100% - 16px) 16px, calc(100% - 11px) 16px; background-size: 5px 5px; background-repeat: no-repeat; }
    .seg { display: inline-flex; height: 38px; padding: 3px; border: 1px solid var(--rule-strong); border-radius: 8px; background: var(--surface); }
    .seg button { border: 0; border-radius: 5px; padding: 0 11px; background: transparent; color: var(--muted); font-size: 12.5px; font-weight: 600; cursor: pointer; white-space: nowrap; }
    .seg button:hover { color: var(--ink); }
    .seg button[aria-pressed="true"] { background: var(--surface-3); color: var(--ink); }
    .count { margin-left: auto; align-self: center; color: var(--muted); font-size: 12.5px; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .count strong { color: var(--ink); font-weight: 650; }
    .reset { border: 0; background: none; padding: 0; color: var(--muted); font-size: 12.5px; text-decoration: underline; text-underline-offset: 3px; cursor: pointer; }
    .reset:hover { color: var(--ink); }

    /* Table */
    .table-shell { container-type: inline-size; position: relative; margin-top: 14px; overflow: auto; max-height: calc(100vh - 150px); border: 1px solid var(--rule); border-radius: 12px; background: var(--surface); box-shadow: var(--shadow); }
    table { width: 100%; min-width: 1180px; border-collapse: separate; border-spacing: 0; font-variant-numeric: tabular-nums; }
    thead th { position: sticky; z-index: 3; background: var(--surface); }
    thead tr.groups th { top: 0; height: 30px; padding: 10px 10px 0; font-size: 10px; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; text-align: center; vertical-align: bottom; }
    thead tr.groups th span { display: block; padding-bottom: 5px; border-bottom: 2px solid currentColor; }
    thead tr.cols th { top: 30px; padding: 8px 10px 9px; border-bottom: 1px solid var(--rule-strong); color: var(--muted); font-size: 11px; font-weight: 650; text-align: right; white-space: nowrap; vertical-align: bottom; }
    .g-own { color: var(--blue); } .g-sec { color: var(--gold); } .g-mix { color: var(--violet); }
    th button { all: unset; cursor: pointer; display: inline-flex; flex-direction: column; align-items: flex-end; gap: 1px; }
    th button .unit { color: var(--faint); font-size: 9.5px; font-weight: 550; letter-spacing: .02em; }
    th.left button { align-items: flex-start; }
    th button:hover { color: var(--ink); }
    th button[aria-sort] { color: var(--ink); }
    th button[aria-sort="descending"] .lbl::after { content: " ↓"; color: var(--hl-ink); }
    th button[aria-sort="ascending"] .lbl::after { content: " ↑"; color: var(--hl-ink); }
    td { padding: 9px 10px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; font-size: 13px; }
    tbody tr.row { cursor: pointer; }
    tbody tr.row:hover td, tbody tr.row.open td { background: var(--surface-2); }
    .sticky { position: sticky; left: 0; z-index: 2; background: var(--surface); text-align: left; }
    thead .sticky { z-index: 4; }
    .sticky::after { content: ""; position: absolute; top: 0; right: 0; bottom: -1px; width: 1px; background: var(--rule); }
    .who { display: grid; grid-template-columns: 30px minmax(0, 1fr); column-gap: 8px; min-width: 210px; }
    .who .rank { grid-row: 1 / 3; align-self: center; color: var(--faint); font-size: 12px; text-align: right; }
    .who .name { font-weight: 650; overflow: hidden; text-overflow: ellipsis; }
    .who .meta { color: var(--faint); font-size: 11px; }
    .who .meta .obs { color: var(--muted); }
    tr.open .who .rank { color: var(--hl-ink); }
    .num-dim { color: var(--muted); }
    .pos { color: var(--good); } .neg { color: var(--bad); } .nil { color: var(--faint); }
    .shot { color: var(--blue); font-weight: 600; }
    td.head-col { position: relative; background: var(--hl-bg); color: var(--hl-ink); font-weight: 760; }
    tbody tr.row:hover td.head-col, tbody tr.row.open td.head-col { background: var(--hl-bg-2); }
    td.head-col .bar { position: absolute; left: 8px; bottom: 5px; height: 2px; border-radius: 2px; background: var(--hl); opacity: .8; }
    td.head-col .bar.neg-bar { background: var(--bad); }
    th.head-col { background: color-mix(in srgb, var(--hl) 8%, var(--surface)); }
    th.head-col button { color: var(--hl-ink); }
    .sec { color: var(--gold); font-weight: 650; }
    .mixv { color: var(--violet); font-weight: 600; }
    td.mix-col { background: var(--violet-bg); border-left: 1px dashed color-mix(in srgb, var(--violet) 30%, transparent); }
    th.mix-col { border-left: 1px dashed color-mix(in srgb, var(--violet) 30%, transparent); }
    .empty { padding: 64px 20px; color: var(--muted); text-align: center; white-space: normal; }
    .more { display: flex; justify-content: center; padding: 14px; }
    .more button { height: 34px; padding: 0 16px; border: 1px solid var(--rule-strong); border-radius: 8px; background: var(--surface); color: var(--ink); font-size: 12.5px; font-weight: 600; cursor: pointer; }
    .more button:hover { border-color: var(--faint); }

    /* Ledger */
    tr.detail td { padding: 0; text-align: left; white-space: normal; background: var(--surface-2); }
    .ledger { position: sticky; left: 0; width: min(100cqw, 1180px); padding: 16px 18px 18px 48px; }
    .ledger-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; margin-bottom: 12px; }
    .ledger-head h3 { margin: 0; font-size: 15px; font-weight: 700; letter-spacing: -.01em; }
    .ledger-head .ctx { color: var(--muted); font-size: 12.5px; }
    .ledger-head .ctx strong { color: var(--ink); font-weight: 650; }
    .cards { display: grid; grid-template-columns: 1fr 1.25fr .95fr; gap: 10px; }
    .card { min-width: 0; padding: 12px 14px; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }
    .card h4 { margin: 0 0 8px; font-size: 10.5px; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; }
    .card.own h4 { color: var(--blue); } .card.sec-card h4 { color: var(--gold); } .card.mix-card h4 { color: var(--violet); }
    .card.lin-card { grid-column: 1 / -1; border-style: dashed; border-color: color-mix(in srgb, var(--violet) 40%, transparent); background: color-mix(in srgb, var(--violet) 4%, var(--surface)); }
    .card.lin-card h4 { color: var(--violet); }
    .lt-wrap { overflow-x: auto; }
    .lt { width: 100%; min-width: 620px; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    .card .lt th, .card .lt td { padding: 6px 10px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; font-size: 12.5px; background: transparent; }
    .lt thead th { position: static; color: var(--faint); font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; vertical-align: bottom; }
    .lt thead th small { display: block; font-size: 9.5px; font-weight: 550; letter-spacing: .02em; text-transform: none; }
    .lt th:first-child, .lt td:first-child { text-align: left; color: var(--muted); font-weight: 600; }
    .lt .box { border-right: 1px dashed var(--rule-strong); font-weight: 700; }
    .lt .tot { font-weight: 720; color: var(--violet); }
    .lt .raw { border-left: 1px dashed var(--rule-strong); color: var(--muted); }
    .lt tr.sum td { border-bottom: 0; border-top: 1px solid var(--rule-strong); }
    .card.mix-card { border-style: dashed; border-color: color-mix(in srgb, var(--violet) 40%, transparent); background: color-mix(in srgb, var(--violet) 4%, var(--surface)); }
    .eq { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 12px; padding: 7px 0; border-top: 1px solid var(--rule); }
    .eq:first-of-type { border-top: 0; padding-top: 0; }
    .eq .n { color: var(--muted); font-size: 12px; font-weight: 600; }
    .eq .o { font-size: 13.5px; font-weight: 720; text-align: right; font-variant-numeric: tabular-nums; }
    .eq .m { grid-column: 1 / -1; color: var(--faint); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
    .eq.hl .n, .eq.hl .o { color: var(--hl-ink); }
    .eq.gold .o { color: var(--gold); }
    .eq.violet .n, .eq.violet .o { color: var(--violet); }
    .note { margin: 8px 0 0; color: var(--muted); font-size: 11.5px; line-height: 1.5; }

    /* Notes */
    .notes { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 22px; margin-top: 26px; padding-top: 18px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 12.5px; line-height: 1.55; }
    .notes h2 { margin: 0 0 6px; color: var(--ink); font-size: 12px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
    .notes p { margin: 0 0 8px; }
    .notes p + h2 { margin-top: 16px; }
    .notes .v { color: var(--violet); font-weight: 650; }

    @media (max-width: 1020px) {
      .cards { grid-template-columns: 1fr 1fr; }
      .cards .mix-card { grid-column: 1 / -1; }
      .notes { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 720px) {
      .page { padding: 18px 16px 32px; }
      .top { flex-direction: column-reverse; gap: 12px; }
      .icon-btn { align-self: flex-end; }
      .lede { font-size: 14px; }
      .formula { display: grid; grid-template-columns: 18px minmax(0, 1fr); gap: 6px 6px; }
      .term { flex-direction: row; align-items: baseline; justify-content: space-between; gap: 10px; padding: 8px 12px; }
      .term .t-value { font-size: 13.5px; }
      .op, .op.lead { display: flex; justify-content: center; padding: 0; font-size: 15px; }
      .formula .spacer { display: none; }
      .term.mix { grid-column: 1 / -1; margin-top: 4px; }
      .controls { position: static; gap: 10px 8px; padding-top: 4px; }
      .search-wrap { flex-basis: 100%; max-width: none; }
      .field.season { flex: 1 1 88px; }
      .seg { height: 36px; }
      .seg button { padding: 0 7px; font-size: 12px; }
      .count { margin-left: 0; flex-basis: 100%; }
      .opt { display: none; }
      table { min-width: 0; }
      .who { min-width: 150px; grid-template-columns: 24px minmax(0, 1fr); }
      .who .name { max-width: 130px; }
      td, thead tr.cols th { padding-left: 7px; padding-right: 7px; }
      .table-shell { max-height: calc(100vh - 120px); border-radius: 10px; }
      .ledger { padding: 14px 12px; }
      .cards { grid-template-columns: 1fr; }
      .notes { grid-template-columns: 1fr; gap: 12px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header class="top">
      <div>
        <p class="kicker"><span>NBA</span><span class="dot"></span><span>1997–2026 regular seasons</span><span class="dot"></span><span class="pill">Research reader</span></p>
        <h1>Scoring impact <span class="soft">without bad&#8209;pass cost</span></h1>
        <p class="lede"><strong>How many points per 100 possessions a player adds through their own shots and the turnovers they commit while trying to score.</strong> Passing mistakes are kept out of the ranking but shown beside it, and adding them back gives full scoring security. Select a row to see every number rebuilt.</p>
      </div>
      <button class="icon-btn" id="theme" type="button" aria-label="Switch colour theme">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
        <span id="theme-label">Theme</span>
      </button>
    </header>

    <section class="formula" aria-label="How the columns add up">
      <span class="op lead" aria-hidden="true"></span>
      <div class="term own"><span class="t-label">Own shots</span><span class="t-value">Scoring value</span></div>
      <span class="op" aria-hidden="true">+</span>
      <div class="term"><span class="t-label">Handle while scoring</span><span class="t-value">Scoring TOV</span></div>
      <span class="op" aria-hidden="true">=</span>
      <div class="term head"><span class="t-label">Ranking</span><span class="t-value">Scoring impact</span></div>
      <span class="op" aria-hidden="true">+</span>
      <div class="term"><span class="t-label">Passing mistakes</span><span class="t-value">Bad-pass TOV</span></div>
      <span class="op" aria-hidden="true">=</span>
      <div class="term full"><span class="t-label">Everything</span><span class="t-value">Scoring security</span></div>
      <span class="spacer"></span>
      <div class="term mix" title="Shown beside the ledger, never added into it"><span class="t-label">Beside it · research</span><span class="t-value">Playtype mix · Lineup TOV</span></div>
    </section>

    <section class="controls" aria-label="Filters">
      <label class="field search-wrap"><span>Player</span>
        <input id="search" type="search" placeholder="Search a player" autocomplete="off" spellcheck="false">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
      </label>
      <label class="field season"><span>Season</span><select id="year"></select></label>
      <div class="field"><span>Min. offensive possessions</span>
        <div class="seg" id="poss" role="group" aria-label="Minimum offensive possessions">
          <button type="button" data-v="0">All</button><button type="button" data-v="500">500</button><button type="button" data-v="1000">1,000</button><button type="button" data-v="2000">2,000</button><button type="button" data-v="4000">4,000</button>
        </div>
      </div>
      <div class="field"><span>Role source</span>
        <div class="seg" id="role" role="group" aria-label="Role source">
          <button type="button" data-v="all">All</button><button type="button" data-v="observed">Observed</button><button type="button" data-v="estimated">PBP-estimated</button>
        </div>
      </div>
      <div class="count" id="count" aria-live="polite"></div>
    </section>

    <div class="table-shell" id="shell">
      <table>
        <thead>
          <tr class="groups">
            <th class="sticky" colspan="1"></th>
            <th class="opt" colspan="2"></th>
            <th colspan="4" class="g-own" data-full="4" data-narrow="2"><span>Own scoring</span></th>
            <th colspan="4" class="g-sec" data-full="4" data-narrow="2"><span>Ball security</span></th>
            <th colspan="2" class="g-mix mix-col" data-full="2" data-narrow="1"><span>Research</span></th>
          </tr>
          <tr class="cols">
            <th class="sticky left"><button data-key="player_name"><span class="lbl">Player</span><span class="unit">rank · season · role source</span></button></th>
            <th class="opt"><button data-key="year"><span class="lbl">Season</span><span class="unit">&nbsp;</span></button></th>
            <th class="opt"><button data-key="OffPoss"><span class="lbl">Off. poss</span><span class="unit">count</span></button></th>
            <th class="opt"><button data-key="tsa100" title="True-shooting attempts per 100 offensive possessions"><span class="lbl">TSA</span><span class="unit">per 100</span></button></th>
            <th class="opt"><button data-key="rts" title="True shooting above league, in TS points"><span class="lbl">rTS</span><span class="unit">TS pts</span></button></th>
            <th><button data-key="zts" title="rTS plus role difficulty"><span class="lbl">zTS</span><span class="unit">rTS + role</span></button></th>
            <th><button data-key="scoring_value100" title="Own-shot scoring value per 100 offensive possessions"><span class="lbl">Scoring value</span><span class="unit">pts / 100</span></button></th>
            <th class="opt"><button data-key="scoring_tov_value100" title="Value of scoring/handling turnovers versus the workload allowance"><span class="lbl">Scoring TOV</span><span class="unit">pts / 100</span></button></th>
            <th class="head-col"><button data-key="scoring_impact_no_badpass100" title="Scoring value + scoring-turnover value (the ranking)"><span class="lbl">Scoring impact</span><span class="unit">pts / 100</span></button></th>
            <th class="opt"><button data-key="badpass_tov_value100" title="Value of bad-pass turnovers versus the passing allowance"><span class="lbl">Bad-pass TOV</span><span class="unit">pts / 100</span></button></th>
            <th><button data-key="scoring_security_value100" title="Scoring impact + bad-pass TOV value"><span class="lbl">Scoring security</span><span class="unit">pts / 100</span></button></th>
            <th class="mix-col"><button data-key="playtype_mix_rapm" title="Time-decay lineup shot-diet value with the player on offense; negative = the lineup's shots are harder than an average diet"><span class="lbl">Playtype mix RAPM</span><span class="unit">pts / 100 · − harder</span></button></th>
            <th class="opt"><button data-key="lineup_tov_value100" title="Time-decay lineup view of the turnover lanes: points per 100 the lineup saves in turnovers (own + teammates + team) with the player on offense. Compare with Scoring TOV + Bad-pass TOV; never added to them"><span class="lbl">Lineup TOV</span><span class="unit">pts / 100 · TD550</span></button></th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="more" id="more" hidden><button type="button" id="more-btn">Show 100 more</button></div>
    </div>

    <section class="notes" aria-label="How to read">
      <div>
        <h2>Reading the numbers</h2>
        <p>Every value column is points per 100 offensive possessions, and positive is better. rTS, role and zTS are true-shooting percentage points. rZTOV in the row detail is turnovers per 100 plays above the workload allowance, so negative is better.</p>
        <p>Scoring value is the production zTS chain: 2 × TSA/100 × (zTS + 4) / 100, then the fitted line −1.211295 + 0.800618 × that.</p>
      </div>
      <div>
        <h2>Sources</h2>
        <p>Role is observed from 2006 to 2026 and a calibrated play-by-play backcast from 1997 to 2005. Missing observed-era roles stay blank. Turnover lanes are observed from 2014, identical to the rZTOV explorer, and box-estimated before that. Each lane is priced by a rolling five-season fit of the net-continuation turnover RAPMs.</p>
      </div>
      <div>
        <h2><span class="v">Playtype mix RAPM</span> · research</h2>
        <p>How easy or hard the whole lineup's shot diet is while the player is on offense, including the player's own shots and teammates' shots. It uses a lineup ridge with a 550-day half-life over the feature season and the one before it, with each playtype priced at the previous season's league value. A negative value means harder shots, which is the iso-heavy-creator pattern.</p>
        <p>It is descriptive only. It never enters the ranking or scoring security, and it is not zTS, VPM or a prior.</p>
        <h2><span class="v">Lineup TOV</span> · research</h2>
        <p>What lineup data says about the same turnovers. A two-season on-court ridge with a 550-day half-life measures how many fewer turnovers the lineup commits with the player on offense, split into the player's own turnovers, teammates' turnovers, and unassigned team turnovers such as shot-clock violations. Each piece is priced at one kept possession. It is shown beside the box turnover values, never added to them. Team turnovers are missing from 1997-2013 source data.</p>
      </div>
    </section>
  </main>

  <script>
    const PACKED = __DATA__;
    const DATA = PACKED.rows.map(values => Object.fromEntries(PACKED.fields.map((field, i) => [field, values[i]])));
    const SV_A = __SV_A__, SV_B = __SV_B__;
    const COLS = 13;
    const PASS = __PASS__;
    const PAGE = 100;
    const QUALIFIED = 1000;
    const DEFAULTS = { q: "", season: "all", min: 1000, role: "all", sort: "scoring_impact_no_badpass100", dir: -1 };
    const state = { ...DEFAULTS, open: null, limit: PAGE };

    const $ = id => document.getElementById(id);
    const search = $("search"), year = $("year"), poss = $("poss"), role = $("role");
    const tbody = $("rows"), count = $("count"), more = $("more"), shell = $("shell");
    const headers = Array.from(document.querySelectorAll("th button[data-key]"));

    const years = Array.from(new Set(DATA.map(d => d.year))).sort((a, b) => b - a);
    year.innerHTML = '<option value="all">All seasons</option>' + years.map(y => `<option value="${y}">${y}</option>`).join("");

    const has = v => v !== null && v !== undefined && Number.isFinite(v);
    const minus = t => t.replace("-", "−");
    const signed = (v, d = 2) => has(v) ? (v > 0 ? "+" : "") + minus(v.toFixed(d)) : "—";
    const fixed = (v, d = 2) => has(v) ? minus(v.toFixed(d)) : "—";
    const int = v => Math.round(v).toLocaleString();
    const tone = v => !has(v) ? "nil" : v > 0.005 ? "pos" : v < -0.005 ? "neg" : "";
    const esc = t => String(t).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
    const observedRole = d => d.role_source === "databallr_synergy_observed";
    const tovLabel = d => d.turnover_source === "observed" ? "observed TOV" : "box-estimated TOV";
    const rowKey = d => `${d.nba_id}-${d.year}`;
    const ordinal = n => n + (["th", "st", "nd", "rd"][(n % 100 - 20) % 10] || ["th", "st", "nd", "rd"][n % 100] || "th");

    // Season ranks among qualified player-seasons, for row context.
    const seasonRank = new Map();
    const seasonSize = new Map();
    for (const y of years) {
      const pool = DATA.filter(d => d.year === y && d.OffPoss >= QUALIFIED);
      seasonSize.set(y, pool.length);
      pool.sort((a, b) => b.scoring_impact_no_badpass100 - a.scoring_impact_no_badpass100)
        .forEach((d, i) => seasonRank.set(rowKey(d), i + 1));
    }

    // Theme toggle (per-viewer convenience; storage is optional).
    const root = document.documentElement;
    const themeLabel = $("theme-label");
    function currentTheme() {
      const set = root.getAttribute("data-theme");
      if (set) return set;
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }
    function applyTheme(theme) {
      root.setAttribute("data-theme", theme);
      themeLabel.textContent = theme === "light" ? "Dark" : "Light";
      try { localStorage.setItem("scoring-impact-theme", theme); } catch (e) {}
    }
    try { const saved = localStorage.getItem("scoring-impact-theme"); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}
    themeLabel.textContent = currentTheme() === "light" ? "Dark" : "Light";
    $("theme").addEventListener("click", () => applyTheme(currentTheme() === "light" ? "dark" : "light"));

    // URL state so a view can be shared.
    function readUrl() {
      try {
        const p = new URLSearchParams(location.hash.slice(1));
        Object.assign(state, DEFAULTS);
        state.q = "";
        if (p.has("q")) state.q = p.get("q");
        if (p.has("season") && (p.get("season") === "all" || years.includes(Number(p.get("season"))))) state.season = p.get("season");
        if (p.has("min") && Number.isFinite(Number(p.get("min")))) state.min = Number(p.get("min"));
        if (p.has("role") && ["all", "observed", "estimated"].includes(p.get("role"))) state.role = p.get("role");
        if (p.has("sort") && headers.some(b => b.dataset.key === p.get("sort"))) state.sort = p.get("sort");
        if (p.has("dir")) state.dir = p.get("dir") === "asc" ? 1 : -1;
      } catch (e) {}
    }
    function writeUrl() {
      try {
        const p = new URLSearchParams();
        if (state.q) p.set("q", state.q);
        for (const k of ["season", "min", "role", "sort"]) if (state[k] !== DEFAULTS[k]) p.set(k, state[k]);
        if (state.dir !== DEFAULTS.dir) p.set("dir", state.dir === 1 ? "asc" : "desc");
        const hash = p.toString();
        history.replaceState(null, "", hash ? `#${hash}` : location.pathname + location.search);
      } catch (e) {}
    }
    function syncControls() {
      search.value = state.q;
      year.value = state.season;
      poss.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.v) === state.min)));
      role.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === state.role)));
    }

    function filteredRows() {
      const q = state.q.trim().toLowerCase();
      return DATA.filter(d =>
        (state.season === "all" || d.year === Number(state.season)) &&
        d.OffPoss >= state.min &&
        (state.role === "all" || (state.role === "observed") === observedRole(d)) &&
        (!q || d.player_name.toLowerCase().includes(q))
      );
    }

    function eq(name, math, out, cls = "") {
      return `<div class="eq ${cls}"><span class="n">${name}</span><span class="o">${out}</span>${math ? `<span class="m">${math}</span>` : ""}</div>`;
    }

    function lineupTov(d) {
      if (!has(d.lineup_tov_value100)) {
        return `<p class="note">No lineup turnover solve for this player-season. The values stay blank, never zero.</p>`;
      }
      const cell = (v, cls = "") => `<td class="${cls} ${tone(v)}">${signed(v)}</td>`;
      const lane = (label, box, key) => `<tr><td>${label}</td>${cell(box, "box")}${cell(d[`lineup_${key}_tov_own_value100`])}${cell(d[`lineup_${key}_tov_teammate_value100`])}${cell(d[`lineup_${key}_tov_team_value100`])}<td class="tot">${signed(d[`lineup_${key}_tov_total_value100`])}</td>${cell(d[`lineup_${key}_tov_raw_value100`], "raw")}</tr>`;
      const sum = (...keys) => keys.every(k => has(d[k])) ? keys.reduce((a, k) => a + d[k], 0) : null;
      const boxBoth = sum("scoring_tov_value100", "badpass_tov_value100");
      const both = part => sum(`lineup_scoring_tov_${part}_value100`, `lineup_badpass_tov_${part}_value100`);
      const table = `<div class="lt-wrap"><table class="lt">
        <thead><tr><th></th><th class="box">Box ledger<small>vs workload allowance</small></th><th>Own<small>vs ⅕ lineup share</small></th><th>Teammates</th><th>Team<small>unassigned</small></th><th>Lineup total<small>adjusted</small></th><th class="raw">Raw on-court<small>vs league</small></th></tr></thead>
        <tbody>
          ${lane("Scoring TOV", d.scoring_tov_value100, "scoring")}
          ${lane("Bad-pass TOV", d.badpass_tov_value100, "badpass")}
          <tr class="sum"><td>Both lanes</td>${cell(boxBoth, "box")}${cell(both("own"))}${cell(both("teammate"))}${cell(both("team"))}<td class="tot">${signed(d.lineup_tov_value100)}</td>${cell(both("raw"), "raw")}</tr>
        </tbody></table></div>`;
      const notes = [];
      notes.push(`Points per 100 offensive possessions, positive = fewer turnovers. The lineup columns come from a ridge fit on every ${d.year - 1 >= 1997 ? `${d.year - 1}-${d.year}` : d.year} lineup with a 550-day half-life, and each turnover avoided is priced at one kept possession, ${fixed(d.lineup_tov_point_price, 3)} points. Own, teammates and team add up to the lineup total.`);
      notes.push("Own is measured against a flat one-fifth share of the lineup's turnovers, while the box ledger measures against a workload allowance, so heavy creators read lower on lineup own by construction. Teammates is how many fewer turnovers the other four commit with this player on the floor.");
      const pt = PASS[d.year];
      if (pt) notes.push(`Across qualified ${d.year} players, one point of box turnover value comes with ${fixed(pt.turnover_total_slope)} points of adjusted lineup value (correlation ${fixed(pt.turnover_total_corr)}). By lane: scoring ${fixed(pt.scoring_total_slope)} adjusted and ${fixed(pt.scoring_raw_slope)} raw; bad-pass ${fixed(pt.badpass_total_slope)} and ${fixed(pt.badpass_raw_slope)}. The adjusted view is ridge-shrunk toward zero; the raw view does not separate the player from teammates or opponents.`);
      if (d.lineup_team_tov_coverage === "absent") notes.push("Team turnovers are missing from this era's source possessions, so the team column is not comparable with 2015 and later.");
      else if (d.lineup_team_tov_coverage === "partial") notes.push("This window includes 2013, whose source possessions omit team turnovers, so the team column is only partly covered.");
      notes.push("This is a transparency view. It is not added to scoring impact or scoring security.");
      return table + notes.map(n => `<p class="note">${n}</p>`).join("");
    }

    function ledger(d) {
      const zAdd = 2 * d.tsa100 * (d.zts + 4) / 100;
      const rank = seasonRank.get(rowKey(d));
      const ctx = rank
        ? `<span class="ctx"><strong>${ordinal(rank)}</strong> of ${seasonSize.get(d.year)} qualified ${d.year} player-seasons by scoring impact</span>`
        : `<span class="ctx">Below the ${QUALIFIED.toLocaleString()}-possession qualifying line for ${d.year}</span>`;
      const own = [
        eq("zTS", `rTS ${signed(d.rts)} + role ${signed(d.role_difficulty)}`, signed(d.zts)),
        eq("zTS add−4", `2 × ${fixed(d.tsa100, 1)} TSA/100 × (${signed(d.zts)} + 4) / 100`, fixed(zAdd, 3)),
        eq("Scoring value", `${fixed(SV_A, 6)} + ${fixed(SV_B, 6)} × ${fixed(zAdd, 3)}`, signed(d.scoring_value100)),
        `<p class="note">${observedRole(d) ? "Observed role" : "Play-by-play-estimated role"} · ${fixed(d.tsa100, 1)} true-shooting attempts per 100 offensive possessions.</p>`,
      ];
      const sec = [
        eq("Scoring TOV", `−(${signed(d.sc_excess, 3)} excess TOV/100) × ${fixed(d.scoring_tov_point_price, 3)} pts per TOV`, signed(d.scoring_tov_value100)),
        eq("Scoring impact", `${signed(d.scoring_value100)} scoring value + ${signed(d.scoring_tov_value100)} scoring TOV`, signed(d.scoring_impact_no_badpass100), "hl"),
        eq("Bad-pass TOV", `−(${signed(d.bp_excess, 3)} excess TOV/100) × ${fixed(d.badpass_tov_point_price, 3)} pts per TOV`, signed(d.badpass_tov_value100)),
        eq("Scoring security", `${signed(d.scoring_impact_no_badpass100)} scoring impact + ${signed(d.badpass_tov_value100)} bad-pass TOV`, signed(d.scoring_security_value100), "gold"),
      ];
      if (has(d.rztov)) sec.push(`<p class="note">rZTOV ${signed(d.rztov)} per 100 plays (${tovLabel(d)}). Each lane's excess is priced at its own window's value per turnover.</p>`);
      let mix;
      if (has(d.playtype_mix_rapm)) {
        const caveats = [];
        if (d.year === 1997) caveats.push("1997 has no prior season, so its playtypes are priced on 1997 itself and the fit uses one season.");
        else if (d.year < 2009) caveats.push("Pre-2009 play-by-play rarely tags self-created threes, so older diets are read more coarsely.");
        mix = [
          eq("Playtype mix RAPM", "", signed(d.playtype_mix_rapm), "violet"),
          `<p class="note">${d.playtype_mix_rapm < 0 ? "With this player on offense, the lineup's shots were priced below an average diet: a harder mix." : "With this player on offense, the lineup's shots were priced above an average diet: an easier mix."} The value comes from a lineup ridge fit over two seasons with a 550-day half-life, and each playtype is priced at the previous season's league value. ${caveats.join(" ")}</p>`,
          `<p class="note">This is descriptive research. It is not added to scoring impact or scoring security.</p>`,
        ];
      } else {
        mix = [`<p class="note">No playtype-mix solve for this player-season. The value stays blank, never zero.</p>`];
      }
      return `<tr class="detail"><td colspan="${COLS}"><div class="ledger">
        <div class="ledger-head"><h3>${esc(d.player_name)} · ${d.year}</h3>${ctx}</div>
        <div class="cards">
          <div class="card own"><h4>Own scoring</h4>${own.join("")}</div>
          <div class="card sec-card"><h4>Ball security</h4>${sec.join("")}</div>
          <div class="card mix-card"><h4>Playtype mix · research</h4>${mix.join("")}</div>
          <div class="card lin-card"><h4>Turnovers in lineup data · research</h4>${lineupTov(d)}</div>
        </div></div></td></tr>`;
    }

    function render() {
      const rows = filteredRows();
      const k = state.sort, dir = state.dir;
      rows.sort((a, b) => {
        const l = a[k], r = b[k];
        if (typeof l === "string") return dir * l.localeCompare(r);
        const lh = has(l), rh = has(r);
        if (!lh || !rh) return lh === rh ? 0 : lh ? -1 : 1;  // blanks always last
        return l === r ? 0 : dir * (l - r);
      });
      const visible = rows.slice(0, state.limit);
      const scale = Math.max(1e-9, ...visible.map(d => Math.abs(d.scoring_impact_no_badpass100)));
      count.innerHTML = `<strong>${rows.length.toLocaleString()}</strong> player-season${rows.length === 1 ? "" : "s"}${rows.length > visible.length ? ` · showing ${visible.length.toLocaleString()}` : ""} · <button class="reset" id="reset" type="button">Reset</button>`;
      $("reset").addEventListener("click", () => { Object.assign(state, DEFAULTS, { open: null, limit: PAGE }); syncControls(); update(); });
      headers.forEach(b => {
        if (b.dataset.key === k) b.setAttribute("aria-sort", dir === -1 ? "descending" : "ascending");
        else b.removeAttribute("aria-sort");
      });
      more.hidden = rows.length <= visible.length;
      if (!visible.length) {
        tbody.innerHTML = `<tr><td class="empty" colspan="${COLS}">No player-seasons match. Try lowering the possession floor or clearing the search.</td></tr>`;
        return;
      }
      tbody.innerHTML = visible.map((d, i) => {
        const open = state.open === rowKey(d);
        const v = d.scoring_impact_no_badpass100;
        const w = Math.round(Math.min(1, Math.abs(v) / scale) * 100);
        return `<tr class="row${open ? " open" : ""}" data-key="${rowKey(d)}" tabindex="0" aria-expanded="${open}">
          <td class="sticky"><div class="who"><span class="rank">${i + 1}</span><span class="name">${esc(d.player_name)}</span><span class="meta">${d.year} · <span class="${observedRole(d) ? "obs" : ""}">${observedRole(d) ? "observed" : "estimated"} role</span></span></div></td>
          <td class="opt num-dim">${d.year}</td>
          <td class="opt num-dim">${int(d.OffPoss)}</td>
          <td class="opt">${fixed(d.tsa100, 1)}</td>
          <td class="opt ${tone(d.rts)}">${signed(d.rts)}</td>
          <td class="shot">${signed(d.zts)}</td>
          <td class="${tone(d.scoring_value100)}">${signed(d.scoring_value100)}</td>
          <td class="opt ${tone(d.scoring_tov_value100)}">${signed(d.scoring_tov_value100)}</td>
          <td class="head-col">${signed(v)}<span class="bar${v < 0 ? " neg-bar" : ""}" style="width:calc((100% - 16px) * ${w / 100})"></span></td>
          <td class="opt ${tone(d.badpass_tov_value100)}">${signed(d.badpass_tov_value100)}</td>
          <td class="sec">${signed(d.scoring_security_value100)}</td>
          <td class="mix-col ${has(d.playtype_mix_rapm) ? "mixv" : "nil"}">${signed(d.playtype_mix_rapm)}</td>
          <td class="opt ${tone(d.lineup_tov_value100)}">${signed(d.lineup_tov_value100)}</td>
        </tr>${open ? ledger(d) : ""}`;
      }).join("");
    }

    function update() { state.limit = PAGE; writeUrl(); render(); }

    tbody.addEventListener("click", e => {
      const row = e.target.closest("tr.row");
      if (!row) return;
      state.open = state.open === row.dataset.key ? null : row.dataset.key;
      render();
      const again = tbody.querySelector(`tr.row[data-key="${row.dataset.key}"]`);
      if (again) again.focus({ preventScroll: true });
    });
    tbody.addEventListener("keydown", e => {
      const row = e.target.closest("tr.row");
      if (!row) return;
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); row.click(); }
      else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const rows = Array.from(tbody.querySelectorAll("tr.row"));
        const next = rows[rows.indexOf(row) + (e.key === "ArrowDown" ? 1 : -1)];
        if (next) next.focus();
      }
    });
    headers.forEach(b => b.addEventListener("click", () => {
      const k = b.dataset.key;
      if (state.sort === k) state.dir *= -1;
      else { state.sort = k; state.dir = k === "player_name" ? 1 : -1; }
      update();
    }));
    let timer;
    search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { state.q = search.value; update(); }, 90); });
    year.addEventListener("input", () => { state.season = year.value; update(); });
    poss.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; state.min = Number(b.dataset.v); syncControls(); update(); });
    role.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; state.role = b.dataset.v; syncControls(); update(); });
    $("more-btn").addEventListener("click", () => { state.limit += PAGE; render(); });
    window.addEventListener("hashchange", () => { readUrl(); syncControls(); render(); });

    const narrow = window.matchMedia("(max-width: 720px)");
    function fitGroups() {
      document.querySelectorAll("th[data-full]").forEach(th => { th.colSpan = Number(narrow.matches ? th.dataset.narrow : th.dataset.full); });
    }
    if (narrow.addEventListener) narrow.addEventListener("change", fitGroups);

    fitGroups();
    readUrl();
    syncControls();
    render();
  </script>
</body>
</html>
'''


def build_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    missing = sorted(set(REQUIRED + OPTIONAL) - set(frame.columns))
    if missing:
        raise ValueError(f"Historical scoring board is missing columns: {missing}")
    core = [column for column in REQUIRED if column not in {"rztov"}]
    clean = frame.dropna(subset=core).copy()
    checks = {
        "security_gap": (
            clean["scoring_security_value100"]
            - clean["scoring_impact_no_badpass100"]
            - clean["badpass_tov_value100"]
        ),
        "sv_formula_gap": (
            SV_INTERCEPT
            + SV_SLOPE * 2 * clean["tsa100"] * (clean["zts"] + 4) / 100
            - clean["scoring_value100"]
        ),
        "scoring_tov_gap": (
            -clean["sc_excess"] * clean["scoring_tov_point_price"] - clean["scoring_tov_value100"]
        ),
        "badpass_tov_gap": (
            -clean["bp_excess"] * clean["badpass_tov_point_price"] - clean["badpass_tov_value100"]
        ),
    }
    gaps = {name: float(series.abs().max()) for name, series in checks.items()}
    if max(gaps.values()) > 5e-5:
        raise AssertionError(f"Scoring-impact viewer identity gaps: {gaps}")
    fields = REQUIRED + OPTIONAL
    numeric = [column for column in fields if column not in TEXT_FIELDS]
    clean[numeric] = clean[numeric].round(4)
    clean["year"] = clean["year"].astype(int)
    clean["nba_id"] = clean["nba_id"].astype(int)
    return clean[fields].replace({np.nan: None}).to_dict("records")


def load_pass_through(path: Path) -> dict[str, dict[str, float]]:
    table = pd.read_csv(path)
    missing = sorted({"year", *PASS_THROUGH_FIELDS} - set(table.columns))
    if missing:
        raise ValueError(f"Lineup pass-through table is missing columns: {missing}")
    return {
        str(int(row["year"])): {field: round(float(row[field]), 4) for field in PASS_THROUGH_FIELDS}
        for _, row in table.iterrows()
    }


def build_html(input_path: Path, output_path: Path, pass_through_path: Path | None = None) -> dict[str, object]:
    frame = pd.read_csv(input_path)
    payload = build_payload(frame)
    pass_through = load_pass_through(pass_through_path or input_path.parent / PASS_THROUGH_NAME)
    fields = REQUIRED + OPTIONAL
    packed = {"fields": fields, "rows": [[record[field] for field in fields] for record in payload]}
    data_json = json.dumps(packed, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    html = (
        HTML_TEMPLATE.replace("__DATA__", data_json)
        .replace("__SV_A__", repr(SV_INTERCEPT))
        .replace("__SV_B__", repr(SV_SLOPE))
        .replace("__PASS__", json.dumps(pass_through, separators=(",", ":"), allow_nan=False))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(payload),
        "rows_with_mix": sum(record["playtype_mix_rapm"] is not None for record in payload),
        "rows_with_lineup_tov": sum(record["lineup_tov_value100"] is not None for record in payload),
        "pass_through_seasons": len(pass_through),
        "bytes": output_path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pass-through", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_html(args.input, args.output, args.pass_through), indent=2))


if __name__ == "__main__":
    main()

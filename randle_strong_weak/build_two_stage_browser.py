#!/usr/bin/env python3
"""Build a static searchable HTML browser for two-stage residual outputs."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs_two_stage"
HTML_PATH = ROOT / "two_stage_results_browser.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static two-stage result browser.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--html-path", type=Path, default=HTML_PATH)
    parser.add_argument("--title", default="Two-Stage Strong/Weak RAPM")
    parser.add_argument("--strength-label", default="DARKO DPM strength")
    return parser.parse_args()


def player_files(outputs: Path) -> list[tuple[str, str, int, Path]]:
    return [
        ("Two-stage: frozen base + player continuous residual", "continuous", 8000, outputs / "player_continuous_slope_21_26_all_a8000.csv"),
        ("Two-stage: frozen base + global continuous + player residual", "continuous", 8000, outputs / "player_continuous_after_global_21_26_all_a8000.csv"),
        ("Two-stage: frozen base + player binary residual", "binary", 8000, outputs / "player_binary_delta_21_26_all_a8000.csv"),
        ("Two-stage: frozen base + global binary + player residual", "binary", 8000, outputs / "player_binary_after_global_21_26_all_a8000.csv"),
    ]

CONTINUOUS_COLUMNS = [
    "player_name",
    "base_net",
    "observed_net",
    "net_vs_weak_1sd",
    "net_vs_avg",
    "net_vs_strong_1sd",
    "net_strong_minus_weak_2sd",
    "off_vs_weak_1sd",
    "off_vs_avg",
    "off_vs_strong_1sd",
    "off_resid_strength_slope",
    "global_off_strength_slope",
    "player_off_strength_slope",
    "def_vs_weak_1sd",
    "def_vs_avg",
    "def_vs_strong_1sd",
    "def_resid_strength_slope",
    "global_def_strength_slope",
    "player_def_strength_slope",
    "observed_opp_def_strength_z",
    "observed_opp_off_strength_z",
    "possessions",
    "off_poss",
    "def_poss",
    "strong_perc_off",
    "strong_perc_def",
    "player_id",
]

BINARY_COLUMNS = [
    "player_name",
    "base_net",
    "stage2_overall_net",
    "off_vs_weak",
    "off_vs_strong",
    "off_resid_strong_delta",
    "global_off_strong_delta",
    "player_off_strong_delta",
    "def_vs_weak",
    "def_vs_strong",
    "def_resid_strong_delta",
    "global_def_strong_delta",
    "player_def_strong_delta",
    "net_vs_weak",
    "net_vs_strong",
    "net_strong_minus_weak",
    "possessions",
    "off_poss",
    "def_poss",
    "strong_perc_off",
    "strong_perc_def",
    "player_id",
]


def to_records(path: Path, columns: list[str]) -> list[dict]:
    df = pd.read_csv(path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[columns].copy()
    for col in df.columns:
        if col == "player_name":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return json.loads(df.round(4).to_json(orient="records"))


def build_payload(outputs: Path, strength_label: str) -> dict:
    validation = pd.read_csv(outputs / "validation_summary.csv")
    validation = validation.where(pd.notnull(validation), None)
    validation_records = json.loads(validation.round(9).to_json(orient="records"))

    global_continuous = json.loads((outputs / "global_continuous_21_26_all.json").read_text())
    global_binary = json.loads((outputs / "global_binary_21_26_all.json").read_text())

    datasets = {}
    for label, kind, alpha, path in player_files(outputs):
        columns = CONTINUOUS_COLUMNS if kind == "continuous" else BINARY_COLUMNS
        key = path.stem
        datasets[key] = {
            "label": f"{label} - alpha {alpha}",
            "kind": kind,
            "alpha": alpha,
            "columns": columns,
            "rows": to_records(path, columns),
        }

    return {
        "scope": {
            "seasons": "2020-21 through 2025-26 (season keys 2021-2026)",
            "season_type": "ALL (RS + PS)",
            "base": "Pure RAPM, off alpha 2000 / def alpha 4000",
            "strength": strength_label,
            "validation": "20% grouped holdout by game_id, seed 42",
        },
        "validation": validation_records,
        "global_continuous": global_continuous,
        "global_binary": global_binary,
        "datasets": datasets,
    }


def html_template(payload: dict, title: str) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f2ed;
      --ink: #171717;
      --muted: #66625a;
      --line: #d5d0c5;
      --panel: #fffefa;
      --head: #ece7dc;
      --accent: #1646d8;
      --good: #08753f;
      --bad: #a33030;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 18px 22px 12px;
      border-bottom: 1px solid var(--line);
      background: #faf8f2;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .summary div {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 8px 10px;
      min-height: 54px;
    }}
    .summary span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .summary strong {{
      display: block;
      margin-top: 3px;
      font-size: 13px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(300px, 1.3fr) minmax(220px, .9fr) minmax(150px, .45fr) minmax(150px, .45fr);
      gap: 10px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    input, select {{
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
    }}
    main {{ padding: 14px 22px 28px; }}
    .note {{
      max-width: 1180px;
      margin: 0 0 12px;
      color: var(--muted);
    }}
    .note strong {{ color: var(--ink); }}
    .cards {{
      display: grid;
      grid-template-columns: minmax(320px, .8fr) minmax(520px, 1.2fr);
      gap: 12px;
      margin-bottom: 14px;
      align-items: start;
    }}
    .panel {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      overflow: hidden;
    }}
    .panel h2 {{
      margin: 0;
      padding: 9px 11px;
      font-size: 14px;
      background: var(--head);
      border-bottom: 1px solid var(--line);
    }}
    .kv {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 14px;
      padding: 10px 11px;
      font-variant-numeric: tabular-nums;
    }}
    .kv span:nth-child(odd) {{ color: var(--muted); }}
    .mini {{
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
    }}
    .mini th, .mini td {{
      padding: 7px 9px;
      border-bottom: 1px solid #ece8de;
      text-align: right;
      white-space: nowrap;
    }}
    .mini th:first-child, .mini td:first-child {{ text-align: left; }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      max-height: calc(100vh - 270px);
    }}
    table.data {{
      width: 100%;
      min-width: 1700px;
      border-collapse: separate;
      border-spacing: 0;
      font-variant-numeric: tabular-nums;
    }}
    .data th, .data td {{
      padding: 7px 9px;
      border-bottom: 1px solid #ece8de;
      text-align: right;
      white-space: nowrap;
    }}
    .data th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--head);
      color: #3d3932;
      font-size: 12px;
      cursor: pointer;
      user-select: none;
    }}
    .data th:first-child, .data td:first-child {{
      position: sticky;
      left: 0;
      z-index: 1;
      text-align: left;
      min-width: 210px;
      max-width: 320px;
      overflow: hidden;
      text-overflow: ellipsis;
      background: var(--panel);
      background-clip: padding-box;
      box-shadow: 1px 0 0 var(--line);
    }}
    .data th:first-child {{
      z-index: 3;
      background: var(--head);
    }}
    tbody tr:nth-child(even) {{ background: #fbfaf4; }}
    tbody tr:nth-child(even) .data td:first-child,
    .data tbody tr:nth-child(even) td:first-child {{ background: #fbfaf4; }}
    tbody tr:hover {{ background: #edf2ff; }}
    tbody tr:hover .data td:first-child,
    .data tbody tr:hover td:first-child {{ background: #edf2ff; }}
    .name {{ font-weight: 700; }}
    .pos {{ color: var(--good); }}
    .neg {{ color: var(--bad); }}
    .muted {{ color: var(--muted); }}
    .empty {{ padding: 24px; color: var(--muted); }}
    @media (max-width: 900px) {{
      .summary, .controls, .cards {{ grid-template-columns: 1fr; }}
      .table-wrap {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="summary" id="summary"></div>
    <div class="controls">
      <label>Search player
        <input id="search" type="search" placeholder="Shai, Jokic, Wemby...">
      </label>
      <label>Player table
        <select id="dataset"></select>
      </label>
      <label>Min possessions
        <input id="minPoss" type="number" min="0" step="500" value="3000">
      </label>
      <label>Rows
        <select id="limit">
          <option value="30">Top 30</option>
          <option value="50">Top 50</option>
          <option value="100">Top 100</option>
          <option value="99999">All</option>
        </select>
      </label>
    </div>
  </header>
  <main>
    <p class="note"><strong>Read this as a validation-first table.</strong> Everything in the dropdown is a two-stage table: stage one is frozen 21-26 RAPM at off alpha 2000 / def alpha 4000, and stage two fits residual opponent-strength terms using the strength source shown above. “Global + player residual” means the second stage first applies the two-variable global adjustment, then fits heavily-shrunk player residuals. The best held-out model is the global continuous adjustment shown in the top panel, not a player table.</p>
    <section class="cards">
      <div class="panel">
        <h2>Global Residual Coefficients</h2>
        <div class="kv" id="globals"></div>
      </div>
      <div class="panel">
        <h2>Validation Ranking</h2>
        <table class="mini" id="validation"></table>
      </div>
    </section>
    <div class="table-wrap">
      <table class="data">
        <thead id="thead"></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </main>
  <script>
    const PAYLOAD = {payload_json};
    const datasets = PAYLOAD.datasets;
    let sortKey = null;
    let sortDir = -1;

    const fmt = (value, col) => {{
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      if (col === "player_name") return value;
      if (col === "player_id" || col === "possessions" || col === "off_poss" || col === "def_poss") return Math.round(Number(value)).toLocaleString();
      if (String(col).includes("perc")) return (Number(value) * 100).toFixed(1) + "%";
      return Number(value).toFixed(2);
    }};

    const cls = (value, col) => {{
      if (col === "player_name") return "name";
      if (typeof value !== "number") return "";
      if (col.includes("poss") || col.includes("perc") || col === "player_id") return "";
      if (value > 0.0001) return "pos";
      if (value < -0.0001) return "neg";
      return "";
    }};

    function initSummary() {{
      const s = PAYLOAD.scope;
      document.getElementById("summary").innerHTML = [
        ["Seasons", s.seasons],
        ["Scope", s.season_type],
        ["Stage 1", s.base],
        ["Strength", s.strength],
        ["Validation", s.validation],
      ].map(([k, v]) => `<div><span>${{k}}</span><strong>${{v}}</strong></div>`).join("");
    }}

    function initGlobals() {{
      const gc = PAYLOAD.global_continuous.coefficients_per_100;
      const gb = PAYLOAD.global_binary.coefficients_per_100;
      const pairs = [
        ["Continuous off slope vs +1SD opp defense", gc.global_off_opp_def_z_slope],
        ["Continuous def slope vs +1SD opp offense", gc.global_def_opp_off_z_slope],
        ["Binary off residual vs strong defense", gb.global_off_vs_strong_def_resid],
        ["Binary def residual vs strong offense", gb.global_def_vs_strong_off_resid],
      ];
      document.getElementById("globals").innerHTML = pairs.map(([k, v]) => `<span>${{k}}</span><strong class="${{v >= 0 ? "pos" : "neg"}}">${{Number(v).toFixed(3)}}</strong>`).join("");
    }}

    function initValidation() {{
      const rows = PAYLOAD.validation.slice(0, 12);
      const head = `<thead><tr><th>Model</th><th>Alpha</th><th>Val RMSE</th><th>Imp</th></tr></thead>`;
      const body = rows.map(r => `
        <tr>
          <td>${{r.model}}</td>
          <td>${{r.alpha === null ? "" : Number(r.alpha).toFixed(0)}}</td>
          <td>${{Number(r.validation_rmse).toFixed(6)}}</td>
          <td class="${{r.improvement_vs_base >= 0 ? "pos" : "neg"}}">${{Number(r.improvement_vs_base).toFixed(6)}}</td>
        </tr>`).join("");
      document.getElementById("validation").innerHTML = head + `<tbody>${{body}}</tbody>`;
    }}

    function initSelect() {{
      const select = document.getElementById("dataset");
      select.innerHTML = Object.entries(datasets)
        .map(([key, d]) => `<option value="${{key}}">${{d.label}}</option>`)
        .join("");
    }}

    function currentRows() {{
      const key = document.getElementById("dataset").value;
      const data = datasets[key];
      const q = document.getElementById("search").value.trim().toLowerCase();
      const minPoss = Number(document.getElementById("minPoss").value || 0);
      const limit = Number(document.getElementById("limit").value || 30);
      let rows = data.rows.filter(r => Number(r.possessions || 0) >= minPoss);
      if (q) rows = rows.filter(r => String(r.player_name).toLowerCase().includes(q) || String(r.player_id).includes(q));
      if (sortKey) {{
        rows = rows.slice().sort((a, b) => {{
          const av = a[sortKey], bv = b[sortKey];
          if (typeof av === "string") return av.localeCompare(bv) * sortDir;
          return ((av || 0) - (bv || 0)) * sortDir;
        }});
      }}
      return rows.slice(0, limit);
    }}

    function render() {{
      const key = document.getElementById("dataset").value;
      const data = datasets[key];
      const cols = data.columns;
      document.getElementById("thead").innerHTML = `<tr>${{cols.map(c => `<th data-col="${{c}}">${{c}}</th>`).join("")}}</tr>`;
      const rows = currentRows();
      document.getElementById("tbody").innerHTML = rows.length
        ? rows.map(r => `<tr>${{cols.map(c => `<td class="${{cls(r[c], c)}}">${{fmt(r[c], c)}}</td>`).join("")}}</tr>`).join("")
        : `<tr><td class="empty" colspan="${{cols.length}}">No players match the current filters.</td></tr>`;
      document.querySelectorAll("th[data-col]").forEach(th => {{
        th.onclick = () => {{
          const col = th.dataset.col;
          if (sortKey === col) sortDir *= -1;
          else {{
            sortKey = col;
            sortDir = col === "player_name" ? 1 : -1;
          }}
          render();
        }};
      }});
    }}

    initSummary();
    initGlobals();
    initValidation();
    initSelect();
    ["search", "dataset", "minPoss", "limit"].forEach(id => document.getElementById(id).addEventListener("input", render));
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    payload = build_payload(args.outputs_dir, args.strength_label)
    args.html_path.write_text(html_template(payload, args.title))
    print(f"Wrote {args.html_path}")


if __name__ == "__main__":
    main()

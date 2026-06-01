#!/usr/bin/env python3
"""Build a searchable browser for high-possession player-slope values."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "outputs_two_stage/player_continuous_highpos40000_after_global_a32000.csv"
SUMMARY_PATH = ROOT / "outputs_two_stage/player_continuous_highpos40000_after_global_a32000_summary.json"
HTML_PATH = ROOT / "high_possession_values_browser.html"

COLS = [
    "player_name",
    "qualified_for_player_slope",
    "base_net",
    "observed_net",
    "net_vs_weak_1sd",
    "net_vs_avg",
    "net_vs_strong_1sd",
    "net_strong_minus_weak_2sd",
    "off_vs_weak_1sd",
    "off_vs_avg",
    "off_vs_strong_1sd",
    "global_off_strength_slope",
    "player_off_strength_slope",
    "def_vs_weak_1sd",
    "def_vs_avg",
    "def_vs_strong_1sd",
    "global_def_strength_slope",
    "player_def_strength_slope",
    "possessions",
    "strong_perc_off",
    "strong_perc_def",
    "player_id",
]


def payload() -> dict:
    df = pd.read_csv(CSV_PATH)
    df = df[COLS].copy()
    summary = json.loads(SUMMARY_PATH.read_text())
    for col in df.columns:
        if col in {"player_name", "qualified_for_player_slope"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return {
        "summary": summary,
        "rows": json.loads(df.round(4).to_json(orient="records")),
    }


def html_template(data: dict) -> str:
    payload_json = json.dumps(data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>High-Possession DARKO Player Values</title>
  <style>
    :root {{
      --bg: #f4f2ed;
      --panel: #fffefa;
      --head: #ece7dc;
      --line: #d4cec1;
      --ink: #171717;
      --muted: #67615a;
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
      z-index: 10;
      padding: 18px 22px 12px;
      border-bottom: 1px solid var(--line);
      background: #faf8f2;
    }}
    h1 {{ margin: 0 0 12px; font-size: 22px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(170px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .summary div {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 8px 10px;
    }}
    .summary span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
    }}
    .summary strong {{
      display: block;
      margin-top: 3px;
      font-size: 14px;
      font-variant-numeric: tabular-nums;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(170px, .45fr) minmax(150px, .35fr);
      gap: 10px;
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }}
    input, select {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 0 10px;
      font: inherit;
    }}
    main {{ padding: 14px 22px 28px; }}
    .note {{
      max-width: 1120px;
      margin: 0 0 12px;
      color: var(--muted);
    }}
    .note strong {{ color: var(--ink); }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      max-height: calc(100vh - 215px);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1850px;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #ece8de;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--head);
      color: #3d3932;
      font-size: 12px;
      cursor: pointer;
      user-select: none;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      position: sticky;
      left: 0;
      background: var(--panel);
      box-shadow: 1px 0 0 var(--line);
      z-index: 1;
      max-width: 280px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    th:first-child {{ background: var(--head); z-index: 3; }}
    tbody tr:nth-child(even) {{ background: #fbfaf4; }}
    tbody tr:nth-child(even) td:first-child {{ background: #fbfaf4; }}
    tbody tr:hover {{ background: #edf2ff; }}
    tbody tr:hover td:first-child {{ background: #edf2ff; }}
    .pos {{ color: var(--good); }}
    .neg {{ color: var(--bad); }}
    .flag {{ text-align: center; }}
    @media (max-width: 850px) {{
      .summary, .controls {{ grid-template-columns: 1fr; }}
      .table-wrap {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>High-Possession DARKO Player Values</h1>
    <section class="summary" id="summary"></section>
    <section class="controls">
      <label>Search player
        <input id="search" type="search" placeholder="Shai, Jokic, Tatum...">
      </label>
      <label>Player slope eligibility
        <select id="qualified">
          <option value="all">All players</option>
          <option value="true">40k+ slope players</option>
          <option value="false">No player slope</option>
        </select>
      </label>
      <label>Rows
        <select id="limit">
          <option value="30">Top 30</option>
          <option value="75">Top 75</option>
          <option value="9999">All</option>
        </select>
      </label>
    </section>
  </header>
  <main>
    <p class="note"><strong>Model:</strong> frozen base RAPM + DARKO global continuous strength + player-specific continuous residual slopes only for players with at least 40,000 possessions. Non-qualified players keep the global adjustment but have zero player-specific slope.</p>
    <div class="table-wrap">
      <table>
        <thead id="thead"></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </main>
  <script>
    const PAYLOAD = {payload_json};
    const cols = {json.dumps(COLS)};
    let sortKey = "observed_net";
    let sortDir = -1;

    function fmt(value, col) {{
      if (value === null || value === undefined) return "";
      if (col === "player_name") return value;
      if (col === "qualified_for_player_slope") return value ? "yes" : "no";
      if (col === "player_id" || col === "possessions") return Math.round(Number(value)).toLocaleString();
      if (col.includes("perc")) return (Number(value) * 100).toFixed(1) + "%";
      return Number(value).toFixed(2);
    }}

    function cls(value, col) {{
      if (col === "qualified_for_player_slope") return "flag";
      if (["player_name", "player_id", "possessions"].includes(col) || col.includes("perc")) return "";
      const n = Number(value);
      if (n > 0.0001) return "pos";
      if (n < -0.0001) return "neg";
      return "";
    }}

    function init() {{
      const s = PAYLOAD.summary;
      document.getElementById("summary").innerHTML = [
        ["Model", "global + high-pos player"],
        ["Min poss", Number(s.min_possessions_for_player_slope).toLocaleString()],
        ["Slope players", Number(s.qualified_players).toLocaleString()],
        ["Validation RMSE", Number(s.validation_rmse).toFixed(6)],
      ].map(([k, v]) => `<div><span>${{k}}</span><strong>${{v}}</strong></div>`).join("");
      render();
    }}

    function rows() {{
      const q = document.getElementById("search").value.trim().toLowerCase();
      const qualified = document.getElementById("qualified").value;
      const limit = Number(document.getElementById("limit").value);
      let out = PAYLOAD.rows.slice();
      if (q) out = out.filter(r => String(r.player_name).toLowerCase().includes(q) || String(r.player_id).includes(q));
      if (qualified !== "all") out = out.filter(r => String(r.qualified_for_player_slope) === qualified);
      out.sort((a, b) => {{
        const av = a[sortKey], bv = b[sortKey];
        if (typeof av === "string") return av.localeCompare(bv) * sortDir;
        return (Number(av || 0) - Number(bv || 0)) * sortDir;
      }});
      return out.slice(0, limit);
    }}

    function render() {{
      document.getElementById("thead").innerHTML = `<tr>${{cols.map(c => `<th data-col="${{c}}">${{c}}</th>`).join("")}}</tr>`;
      document.getElementById("tbody").innerHTML = rows().map(r => (
        `<tr>${{cols.map(c => `<td class="${{cls(r[c], c)}}">${{fmt(r[c], c)}}</td>`).join("")}}</tr>`
      )).join("");
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

    ["search", "qualified", "limit"].forEach(id => document.getElementById(id).addEventListener("input", render));
    init();
  </script>
</body>
</html>
"""


def main() -> None:
    HTML_PATH.write_text(html_template(payload()))
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()

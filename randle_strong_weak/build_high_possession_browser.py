#!/usr/bin/env python3
"""Build a static browser for high-possession player-slope validation cuts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "outputs_two_stage/high_possession_validation_darko_continuous.csv"
DIAG_PATH = ROOT / "outputs_two_stage/diagnostics_21_26_all.json"
HTML_PATH = ROOT / "high_possession_validation_browser.html"


def payload() -> dict:
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("validation_rmse").copy()
    diagnostics = json.loads(DIAG_PATH.read_text())
    global_info = diagnostics["models"]["global_continuous"]
    return {
        "base_validation_rmse": diagnostics["base_only_validation_rmse"],
        "global_validation_rmse": global_info["validation_rmse"],
        "global_improvement_vs_base": global_info["improvement_vs_base"],
        "rows": json.loads(df.round(9).to_json(orient="records")),
    }


def html_template(data: dict) -> str:
    payload_json = json.dumps(data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>High-Possession DARKO Slope Validation</title>
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
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      letter-spacing: 0;
    }}
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
      grid-template-columns: minmax(180px, .6fr) minmax(220px, .8fr) minmax(180px, .6fr) minmax(180px, .6fr);
      gap: 10px;
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }}
    select {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 0 10px;
      font: inherit;
    }}
    main {{ padding: 14px 22px 28px; }}
    .note {{
      max-width: 1060px;
      margin: 0 0 12px;
      color: var(--muted);
    }}
    .note strong {{ color: var(--ink); }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      max-height: calc(100vh - 205px);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1040px;
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
    }}
    th:first-child {{
      background: var(--head);
      z-index: 3;
    }}
    tbody tr:nth-child(even) {{ background: #fbfaf4; }}
    tbody tr:nth-child(even) td:first-child {{ background: #fbfaf4; }}
    tbody tr:hover {{ background: #edf2ff; }}
    tbody tr:hover td:first-child {{ background: #edf2ff; }}
    .pos {{ color: var(--good); }}
    .neg {{ color: var(--bad); }}
    @media (max-width: 850px) {{
      .summary, .controls {{ grid-template-columns: 1fr; }}
      .table-wrap {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>High-Possession DARKO Slope Validation</h1>
    <section class="summary" id="summary"></section>
    <section class="controls">
      <label>Model
        <select id="model">
          <option value="all">All</option>
          <option value="global_plus_player">Global + player</option>
          <option value="player_only">Player only</option>
        </select>
      </label>
      <label>Min possession cutoff
        <select id="minPos"><option value="all">All cutoffs</option></select>
      </label>
      <label>Alpha
        <select id="alpha"><option value="all">All alphas</option></select>
      </label>
      <label>Rows
        <select id="limit">
          <option value="20">Top 20</option>
          <option value="50">Top 50</option>
          <option value="9999">All</option>
        </select>
      </label>
    </section>
  </header>
  <main>
    <p class="note"><strong>What this tests:</strong> only players above a possession threshold receive DARKO continuous player-slope coefficients. Low-possession players stay in the frozen base/global structure. Lower validation RMSE is better.</p>
    <div class="table-wrap">
      <table>
        <thead id="thead"></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </main>
  <script>
    const PAYLOAD = {payload_json};
    let sortKey = "validation_rmse";
    let sortDir = 1;
    const cols = ["model", "min_pos", "n_players", "alpha", "validation_rmse", "improvement_vs_base", "improvement_vs_global"];

    function fmt(value, col) {{
      if (col === "model") return value === "global_plus_player" ? "global + player" : "player only";
      if (col === "min_pos" || col === "n_players" || col === "alpha") return Number(value).toLocaleString();
      return Number(value).toFixed(6);
    }}

    function cls(value, col) {{
      if (!col.includes("improvement")) return "";
      return Number(value) >= 0 ? "pos" : "neg";
    }}

    function init() {{
      document.getElementById("summary").innerHTML = [
        ["Base RMSE", PAYLOAD.base_validation_rmse.toFixed(6)],
        ["Global RMSE", PAYLOAD.global_validation_rmse.toFixed(6)],
        ["Global improvement", PAYLOAD.global_improvement_vs_base.toFixed(6)],
        ["Rows tested", PAYLOAD.rows.length.toLocaleString()],
      ].map(([k, v]) => `<div><span>${{k}}</span><strong>${{v}}</strong></div>`).join("");

      for (const [id, key] of [["minPos", "min_pos"], ["alpha", "alpha"]]) {{
        const select = document.getElementById(id);
        const values = [...new Set(PAYLOAD.rows.map(r => r[key]))].sort((a, b) => a - b);
        select.innerHTML += values.map(v => `<option value="${{v}}">${{Number(v).toLocaleString()}}</option>`).join("");
      }}
      render();
    }}

    function filteredRows() {{
      const model = document.getElementById("model").value;
      const minPos = document.getElementById("minPos").value;
      const alpha = document.getElementById("alpha").value;
      const limit = Number(document.getElementById("limit").value);
      let rows = PAYLOAD.rows.slice();
      if (model !== "all") rows = rows.filter(r => r.model === model);
      if (minPos !== "all") rows = rows.filter(r => String(r.min_pos) === minPos);
      if (alpha !== "all") rows = rows.filter(r => String(r.alpha) === alpha);
      rows.sort((a, b) => {{
        const av = a[sortKey], bv = b[sortKey];
        if (typeof av === "string") return av.localeCompare(bv) * sortDir;
        return (Number(av) - Number(bv)) * sortDir;
      }});
      return rows.slice(0, limit);
    }}

    function render() {{
      document.getElementById("thead").innerHTML = `<tr>${{cols.map(c => `<th data-col="${{c}}">${{c}}</th>`).join("")}}</tr>`;
      const rows = filteredRows();
      document.getElementById("tbody").innerHTML = rows.map(r => (
        `<tr>${{cols.map(c => `<td class="${{cls(r[c], c)}}">${{fmt(r[c], c)}}</td>`).join("")}}</tr>`
      )).join("");
      document.querySelectorAll("th[data-col]").forEach(th => {{
        th.onclick = () => {{
          const col = th.dataset.col;
          if (sortKey === col) sortDir *= -1;
          else {{
            sortKey = col;
            sortDir = col === "validation_rmse" ? 1 : -1;
          }}
          render();
        }};
      }});
    }}

    ["model", "minPos", "alpha", "limit"].forEach(id => document.getElementById(id).addEventListener("input", render));
    init();
  </script>
</body>
</html>
"""


def main() -> None:
    data = payload()
    HTML_PATH.write_text(html_template(data))
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a static searchable HTML browser for randle_strong_weak outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
HTML_PATH = ROOT / "results_browser.html"


MODEL_FILES = [
    ("Binary reference", "binary", 2, OUTPUTS / "binary_reference_21_26_all_a1000_im2.csv"),
    ("Binary reference", "binary", 4, OUTPUTS / "binary_reference_21_26_all_a1000_im4.csv"),
    ("Binary reference", "binary", 8, OUTPUTS / "binary_reference_21_26_all_a1000_im8.csv"),
    ("Binary reference", "binary", 16, OUTPUTS / "binary_reference_21_26_all_a1000_im16.csv"),
    ("Continuous slope", "continuous", 2, OUTPUTS / "continuous_slope_21_26_all_a1000_im2.csv"),
    ("Continuous slope", "continuous", 4, OUTPUTS / "continuous_slope_21_26_all_a1000_im4.csv"),
    ("Continuous slope", "continuous", 8, OUTPUTS / "continuous_slope_21_26_all_a1000_im8.csv"),
    ("Continuous slope", "continuous", 16, OUTPUTS / "continuous_slope_21_26_all_a1000_im16.csv"),
]


BINARY_COLUMNS = [
    "player_name",
    "overall_net",
    "overall_off",
    "overall_def",
    "off_vs_strong",
    "off_vs_weak",
    "off_strong_delta",
    "def_vs_strong",
    "def_vs_weak",
    "def_strong_delta",
    "net_vs_strong",
    "net_vs_weak",
    "net_strong_minus_weak",
    "possessions",
    "strong_perc_off",
    "strong_perc_def",
    "player_id",
]

CONTINUOUS_COLUMNS = [
    "player_name",
    "observed_net",
    "net_vs_weak_1sd",
    "net_vs_avg",
    "net_vs_strong_1sd",
    "net_strong_minus_weak_2sd",
    "off_vs_weak_1sd",
    "off_vs_avg",
    "off_vs_strong_1sd",
    "off_strength_slope",
    "def_vs_weak_1sd",
    "def_vs_avg",
    "def_vs_strong_1sd",
    "def_strength_slope",
    "possessions",
    "strong_perc_off",
    "strong_perc_def",
    "observed_opp_def_strength_z",
    "observed_opp_off_strength_z",
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


def build_payload() -> dict:
    payload = {}
    for label, kind, multiplier, path in MODEL_FILES:
        columns = BINARY_COLUMNS if kind == "binary" else CONTINUOUS_COLUMNS
        key = f"{kind}_im{multiplier}"
        payload[key] = {
            "label": f"{label} - interaction x{multiplier}",
            "kind": kind,
            "multiplier": multiplier,
            "columns": columns,
            "rows": to_records(path, columns),
        }
    return payload


def html_template(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Randle Strong/Weak RAPM Results</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f6f2;
      --ink: #181817;
      --muted: #66645d;
      --line: #d8d4c8;
      --panel: #ffffff;
      --accent: #245bff;
      --good: #0b6b3a;
      --bad: #9a2c2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 20px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf7;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      font-weight: 750;
      letter-spacing: 0;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(260px, 1.4fr) minmax(180px, .8fr) minmax(150px, .5fr) minmax(150px, .5fr);
      gap: 10px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
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
    main {{ padding: 14px 24px 28px; }}
    .meta {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      margin-bottom: 10px;
      font-size: 13px;
    }}
    .meta strong {{ color: var(--ink); }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      max-height: calc(100vh - 156px);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1500px;
    }}
    th, td {{
      padding: 7px 9px;
      border-bottom: 1px solid #ebe8df;
      white-space: nowrap;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #eeeae0;
      color: #3e3c37;
      font-size: 12px;
      cursor: pointer;
      user-select: none;
    }}
    th:first-child, td:first-child {{
      position: sticky;
      left: 0;
      z-index: 1;
      text-align: left;
      background: inherit;
      min-width: 190px;
      box-shadow: 1px 0 0 var(--line);
    }}
    th:first-child {{ z-index: 3; background: #eeeae0; }}
    tbody tr:nth-child(even) {{ background: #fbfaf7; }}
    tbody tr:hover {{ background: #eef3ff; }}
    .pos {{ color: var(--good); }}
    .neg {{ color: var(--bad); }}
    .name {{ font-weight: 650; }}
    .empty {{
      padding: 28px;
      color: var(--muted);
      text-align: center;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    @media (max-width: 900px) {{
      .controls {{ grid-template-columns: 1fr 1fr; }}
      header, main {{ padding-left: 14px; padding-right: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Randle Strong/Weak RAPM Results</h1>
    <div class="controls">
      <label>Search
        <input id="search" type="search" placeholder="Player name or ID">
      </label>
      <label>Result Set
        <select id="dataset"></select>
      </label>
      <label>Min Possessions
        <input id="minPoss" type="number" min="0" step="100" value="0">
      </label>
      <label>Rows
        <select id="limit">
          <option value="50">50</option>
          <option value="100" selected>100</option>
          <option value="250">250</option>
          <option value="9999">All</option>
        </select>
      </label>
    </div>
  </header>
  <main>
    <div class="meta">
      <span><strong id="shown">0</strong> shown</span>
      <span><strong id="total">0</strong> total</span>
      <span id="sortLabel"></span>
    </div>
    <div id="mount"></div>
  </main>
  <script>
    const DATASETS = {payload_json};
    const state = {{ dataset: "continuous_im16", search: "", minPoss: 0, limit: 100, sortKey: null, sortDir: -1 }};

    const defaultSort = {{
      binary: "overall_net",
      continuous: "observed_net"
    }};

    const labels = {{
      player_name: "Player",
      player_id: "ID",
      overall_net: "Overall Net",
      overall_off: "Overall Off",
      overall_def: "Overall Def",
      off_vs_strong: "Off Strong",
      off_vs_weak: "Off Weak",
      off_strong_delta: "Off Delta",
      def_vs_strong: "Def Strong",
      def_vs_weak: "Def Weak",
      def_strong_delta: "Def Delta",
      net_vs_strong: "Net Strong",
      net_vs_weak: "Net Weak",
      net_strong_minus_weak: "Net Delta",
      observed_net: "Observed Net",
      off_vs_weak_1sd: "Off -1SD",
      off_vs_avg: "Off Avg",
      off_vs_strong_1sd: "Off +1SD",
      off_strength_slope: "Off Slope",
      def_vs_weak_1sd: "Def -1SD",
      def_vs_avg: "Def Avg",
      def_vs_strong_1sd: "Def +1SD",
      def_strength_slope: "Def Slope",
      net_vs_weak_1sd: "Net -1SD",
      net_vs_avg: "Net Avg",
      net_vs_strong_1sd: "Net +1SD",
      net_strong_minus_weak_2sd: "Net 2SD Delta",
      possessions: "Poss",
      off_poss: "Off Poss",
      def_poss: "Def Poss",
      faced_strong_off: "Strong Off Poss",
      faced_weak_off: "Weak Off Poss",
      faced_strong_def: "Strong Def Poss",
      faced_weak_def: "Weak Def Poss",
      strong_perc_off: "Strong Off%",
      strong_perc_def: "Strong Def%",
      observed_opp_def_strength_z: "Opp Def Z",
      observed_opp_off_strength_z: "Opp Off Z",
      report_off_center: "Off Center",
      report_def_center: "Def Center"
    }};

    const datasetSelect = document.getElementById("dataset");
    Object.entries(DATASETS).forEach(([key, value]) => {{
      const option = document.createElement("option");
      option.value = key;
      option.textContent = value.label;
      datasetSelect.appendChild(option);
    }});
    datasetSelect.value = state.dataset;

    function formatValue(key, value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      if (key === "player_name") return String(value);
      if (key === "player_id" || key.endsWith("_poss") || key === "possessions" || key.startsWith("faced_")) {{
        return Number(value).toLocaleString();
      }}
      if (key.includes("perc")) return (Number(value) * 100).toFixed(1) + "%";
      return Number(value).toFixed(2);
    }}

    function cellClass(key, value) {{
      if (key === "player_name") return "name";
      if (typeof value !== "number") return "";
      if (key.includes("poss") || key === "player_id" || key.includes("perc")) return "";
      return value > 0 ? "pos" : value < 0 ? "neg" : "";
    }}

    function filteredRows() {{
      const data = DATASETS[state.dataset];
      const q = state.search.trim().toLowerCase();
      const minPoss = Number(state.minPoss) || 0;
      let rows = data.rows.filter(row => {{
        if ((row.possessions || 0) < minPoss) return false;
        if (!q) return true;
        return String(row.player_name).toLowerCase().includes(q) || String(row.player_id).includes(q);
      }});
      const sortKey = state.sortKey || defaultSort[data.kind];
      rows.sort((a, b) => {{
        const av = a[sortKey];
        const bv = b[sortKey];
        if (typeof av === "string") return state.sortDir * av.localeCompare(bv);
        return state.sortDir * ((av || 0) - (bv || 0));
      }});
      return rows;
    }}

    function render() {{
      const data = DATASETS[state.dataset];
      const columns = data.columns;
      const rows = filteredRows();
      const visible = rows.slice(0, Number(state.limit));
      document.getElementById("shown").textContent = visible.length.toLocaleString();
      document.getElementById("total").textContent = rows.length.toLocaleString();
      const sortKey = state.sortKey || defaultSort[data.kind];
      document.getElementById("sortLabel").textContent = "Sorted by " + (labels[sortKey] || sortKey) + (state.sortDir < 0 ? " desc" : " asc");
      if (!visible.length) {{
        document.getElementById("mount").innerHTML = '<div class="empty">No matching rows.</div>';
        return;
      }}
      const head = columns.map(col => `<th data-key="${{col}}">${{labels[col] || col}}</th>`).join("");
      const body = visible.map(row => {{
        return "<tr>" + columns.map(col => {{
          const value = row[col];
          return `<td class="${{cellClass(col, value)}}">${{formatValue(col, value)}}</td>`;
        }}).join("") + "</tr>";
      }}).join("");
      document.getElementById("mount").innerHTML = `<div class="table-wrap"><table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table></div>`;
      document.querySelectorAll("th").forEach(th => {{
        th.addEventListener("click", () => {{
          const key = th.dataset.key;
          if (state.sortKey === key) state.sortDir *= -1;
          else {{
            state.sortKey = key;
            state.sortDir = key === "player_name" ? 1 : -1;
          }}
          render();
        }});
      }});
    }}

    document.getElementById("search").addEventListener("input", event => {{
      state.search = event.target.value;
      render();
    }});
    document.getElementById("dataset").addEventListener("change", event => {{
      state.dataset = event.target.value;
      state.sortKey = null;
      state.sortDir = -1;
      render();
    }});
    document.getElementById("minPoss").addEventListener("input", event => {{
      state.minPoss = event.target.value;
      render();
    }});
    document.getElementById("limit").addEventListener("change", event => {{
      state.limit = event.target.value;
      render();
    }});
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    HTML_PATH.write_text(html_template(payload), encoding="utf-8")
    print(HTML_PATH)


if __name__ == "__main__":
    main()

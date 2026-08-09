#!/usr/bin/env python3
"""Build a self-contained HTML review table for the current WNBA metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = ROOT / "wnba_test" / "Results" / "current"
DEFAULT_BOARD = CURRENT_DIR / "wnba_current_metrics.csv"
DEFAULT_PRIORS = (
    ROOT
    / "wnba_test"
    / "Results"
    / "box_prior_true_prior"
    / "box_priors_2026.csv"
)
DEFAULT_MANIFEST = CURRENT_DIR / "latest_manifest.json"
DEFAULT_OUTPUT = CURRENT_DIR / "wnba-results-review.html"


def load_review_rows(board_path: Path, priors_path: Path) -> list[dict[str, object]]:
    board = pd.read_csv(board_path, dtype={"player_id": str})
    priors = pd.read_csv(priors_path, dtype={"player_id": str})
    priors["player_id"] = (
        priors["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    columns = [
        "player_id",
        "box_prior_source",
        "box_off",
        "box_def",
        "box_net",
    ]
    merged = board.merge(
        priors[columns].drop_duplicates("player_id", keep="last"),
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    required = [
        "net_rapm",
        "off",
        "def",
        "possessions",
        "box_off",
        "box_def",
        "box_net",
        "prior_net_rapm",
        "delta_net_rapm",
    ]
    if merged[required].isna().any().any():
        missing = merged.loc[
            merged[required].isna().any(axis=1), ["player_id", "player_name"]
        ]
        raise RuntimeError(
            "review join has missing metrics for "
            + ", ".join(missing["player_name"].astype(str).tolist())
        )
    if not np.isfinite(merged[required].to_numpy(dtype=float)).all():
        raise RuntimeError("review metrics include nonfinite values")

    merged = merged.sort_values(
        ["net_rapm", "possessions", "player_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    merged["impact_rank"] = np.arange(1, len(merged) + 1)

    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        records.append(
            {
                "id": str(row["player_id"]),
                "player": str(row["player_name"]),
                "rank": int(row["impact_rank"]),
                "impact": round(float(row["net_rapm"]), 3),
                "offense": round(float(row["off"]), 3),
                "defense": round(float(row["def"]), 3),
                "boxPrior": round(float(row["box_net"]), 3),
                "boxOffense": round(float(row["box_off"]), 3),
                "boxDefense": round(float(row["box_def"]), 3),
                "effectivePrior": round(float(row["prior_net_rapm"]), 3),
                "lineupUpdate": round(float(row["delta_net_rapm"]), 3),
                "possessions": int(row["possessions"]),
                "priorSource": str(row["box_prior_source"]),
            }
        )
    return records


def build_html(
    rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> str:
    as_of = str(manifest.get("as_of", "current"))
    games = int(manifest.get("source", {}).get("after", {}).get("games", 0))
    qualified = sum(int(row["possessions"]) >= 1000 for row in rows)
    data_json = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).replace(
        "</", "<\\/"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WNBA Impact Review — {as_of}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1e8;
      --surface: #fffdf7;
      --ink: #171713;
      --muted: #68675f;
      --rule: #c9c3b6;
      --rule-strong: #171713;
      --accent: #d8ff3e;
      --accent-ink: #171713;
      --positive: #176b45;
      --negative: #a33a32;
      --focus: #3159c5;
      --stripe: rgba(23, 23, 19, 0.035);
      --sticky: #fffdf7;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        color-scheme: dark;
        --bg: #141511;
        --surface: #1d1f19;
        --ink: #f1efe6;
        --muted: #b0afa6;
        --rule: #4a4c43;
        --rule-strong: #f1efe6;
        --accent: #c8ef35;
        --accent-ink: #171713;
        --positive: #75d0a5;
        --negative: #f09a91;
        --focus: #8da8ff;
        --stripe: rgba(255, 255, 255, 0.035);
        --sticky: #1d1f19;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-variant-numeric: tabular-nums;
    }}
    main {{
      width: min(1500px, calc(100% - 32px));
      margin: 0 auto;
      padding: 38px 0 48px;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: end;
      padding-bottom: 22px;
      border-bottom: 3px solid var(--rule-strong);
    }}
    .eyebrow {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .11em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(30px, 5vw, 64px);
      line-height: .96;
      letter-spacing: -.055em;
    }}
    .subtitle {{
      max-width: 790px;
      margin: 16px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
    }}
    .run-meta {{
      display: grid;
      grid-template-columns: repeat(3, auto);
      gap: 18px;
      margin: 0;
    }}
    .run-meta div {{ min-width: 78px; }}
    .run-meta dt {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    .run-meta dd {{
      margin: 4px 0 0;
      font-size: 20px;
      font-weight: 750;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      align-items: end;
      padding: 18px 0;
      border-bottom: 1px solid var(--rule);
    }}
    .control {{
      display: grid;
      gap: 6px;
    }}
    label, .control-label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .07em;
      text-transform: uppercase;
    }}
    input[type="search"], select {{
      min-height: 40px;
      border: 1px solid var(--rule);
      border-radius: 0;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
      padding: 8px 11px;
    }}
    input[type="search"] {{ width: min(320px, 70vw); }}
    input:focus, select:focus, button:focus-visible {{
      outline: 3px solid color-mix(in srgb, var(--focus) 45%, transparent);
      outline-offset: 2px;
    }}
    .scope-buttons {{
      display: flex;
      min-height: 40px;
    }}
    .scope-buttons button {{
      border: 1px solid var(--rule);
      background: var(--surface);
      color: var(--ink);
      font: inherit;
      padding: 8px 12px;
      cursor: pointer;
    }}
    .scope-buttons button + button {{ border-left: 0; }}
    .scope-buttons button[aria-pressed="true"] {{
      background: var(--ink);
      color: var(--surface);
      border-color: var(--ink);
    }}
    .audit-toggle {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 40px;
      color: var(--ink);
      font-size: 14px;
    }}
    .audit-toggle input {{ width: 18px; height: 18px; accent-color: var(--ink); }}
    .result-count {{
      margin-left: auto;
      padding-bottom: 9px;
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--surface);
      border-bottom: 3px solid var(--rule-strong);
    }}
    table {{
      width: 100%;
      min-width: 1060px;
      border-collapse: collapse;
      font-size: 14px;
    }}
    caption {{
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
    }}
    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid var(--rule);
      text-align: right;
      white-space: nowrap;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--sticky);
      font-size: 11px;
      letter-spacing: .055em;
      text-transform: uppercase;
    }}
    thead .group-row th {{
      padding-top: 10px;
      padding-bottom: 7px;
      color: var(--muted);
      border-bottom: 1px solid var(--rule-strong);
      text-align: center;
    }}
    th.rank, td.rank {{
      position: sticky;
      left: 0;
      z-index: 1;
      width: 58px;
      background: var(--sticky);
      color: var(--muted);
      text-align: center;
    }}
    th.player, td.player {{
      position: sticky;
      left: 58px;
      z-index: 1;
      min-width: 188px;
      background: var(--sticky);
      text-align: left;
    }}
    thead th.rank, thead th.player {{ z-index: 4; }}
    tbody tr:nth-child(even) td:not(.rank):not(.player) {{ background: var(--stripe); }}
    tbody tr:hover td {{ background: color-mix(in srgb, var(--accent) 16%, var(--surface)) !important; }}
    tbody tr[data-player="Olivia Miles"] td {{
      box-shadow: inset 0 -2px 0 var(--accent);
    }}
    td.player {{ font-weight: 720; }}
    .player-id {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 500;
      letter-spacing: .04em;
    }}
    .metric-impact {{
      background: color-mix(in srgb, var(--accent) 32%, var(--surface));
      color: var(--accent-ink);
      font-weight: 800;
    }}
    @media (prefers-color-scheme: dark) {{
      .metric-impact {{ color: var(--ink); }}
    }}
    .positive {{ color: var(--positive); }}
    .negative {{ color: var(--negative); }}
    .source {{
      color: var(--muted);
      font-size: 11px;
      text-align: left;
    }}
    .sort-button {{
      border: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      font-weight: 750;
      letter-spacing: inherit;
      text-transform: inherit;
      padding: 0;
      cursor: pointer;
    }}
    .sort-button::after {{ content: " ↕"; color: var(--muted); }}
    .sort-button[data-active="true"][data-direction="desc"]::after {{ content: " ↓"; }}
    .sort-button[data-active="true"][data-direction="asc"]::after {{ content: " ↑"; }}
    .audit-col {{ display: none; }}
    body.show-audit .audit-col {{ display: table-cell; }}
    .notes {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 24px;
      padding-top: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }}
    .notes strong {{ color: var(--ink); }}
    .empty {{
      padding: 48px 16px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 860px) {{
      main {{ width: min(100% - 20px, 1500px); padding-top: 24px; }}
      header {{ grid-template-columns: 1fr; }}
      .run-meta {{ grid-template-columns: repeat(3, 1fr); }}
      .result-count {{ width: 100%; margin-left: 0; padding-bottom: 0; }}
      .notes {{ grid-template-columns: 1fr; gap: 8px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p class="eyebrow">2026 regular season · current single-season model</p>
        <h1>WNBA impact review</h1>
        <p class="subtitle">
          Impact equals offense plus positive-good defense. Box-prior columns are
          the raw public box predictions before the frozen 1.5 prior scale.
        </p>
      </div>
      <dl class="run-meta">
        <div><dt>As of</dt><dd>{as_of}</dd></div>
        <div><dt>Games</dt><dd>{games}</dd></div>
        <div><dt>Players</dt><dd>{len(rows)}</dd></div>
      </dl>
    </header>

    <section class="controls" aria-label="Table controls">
      <div class="control">
        <label for="player-search">Find player</label>
        <input id="player-search" type="search" placeholder="Search by name" autocomplete="off">
      </div>
      <div class="control">
        <span class="control-label">Player pool</span>
        <div class="scope-buttons" role="group" aria-label="Player pool">
          <button type="button" data-scope="qualified" aria-pressed="true">1,000+ possessions</button>
          <button type="button" data-scope="all" aria-pressed="false">All players</button>
        </div>
      </div>
      <div class="control">
        <label for="prior-source">Prior source</label>
        <select id="prior-source">
          <option value="all">All sources</option>
          <option value="current_single_season">Current season</option>
          <option value="previous_season_fallback">2025 fallback</option>
          <option value="imputed_zero">Zero fallback</option>
        </select>
      </div>
      <label class="audit-toggle">
        <input id="audit-toggle" type="checkbox">
        Show scaled prior + lineup update
      </label>
      <div id="result-count" class="result-count" aria-live="polite"></div>
    </section>

    <div class="table-wrap">
      <table>
        <caption>Current WNBA impact ratings and raw box priors</caption>
        <thead>
          <tr class="group-row">
            <th class="rank" rowspan="2">Rank</th>
            <th class="player" rowspan="2">Player</th>
            <th colspan="3">Impact results</th>
            <th colspan="3">Raw box priors</th>
            <th class="audit-col" colspan="2">Model audit</th>
            <th colspan="2">Sample</th>
          </tr>
          <tr>
            <th><button class="sort-button" data-sort="impact" data-active="true" data-direction="desc">Impact</button></th>
            <th><button class="sort-button" data-sort="offense">Offense</button></th>
            <th><button class="sort-button" data-sort="defense">Defense</button></th>
            <th><button class="sort-button" data-sort="boxPrior">Box prior</button></th>
            <th><button class="sort-button" data-sort="boxOffense">Box O</button></th>
            <th><button class="sort-button" data-sort="boxDefense">Box D</button></th>
            <th class="audit-col"><button class="sort-button" data-sort="effectivePrior">Scaled prior</button></th>
            <th class="audit-col"><button class="sort-button" data-sort="lineupUpdate">Lineup update</button></th>
            <th><button class="sort-button" data-sort="possessions">Possessions</button></th>
            <th>Prior source</th>
          </tr>
        </thead>
        <tbody id="results-body"></tbody>
      </table>
    </div>

    <section class="notes" aria-label="Metric notes">
      <div><strong>Impact results:</strong> single-season prior-centered RAPM; defense is already sign-flipped so positive is good.</div>
      <div><strong>Raw box priors:</strong> factor-specific public box predictions trained only on 2022–2025.</div>
      <div><strong>Model audit:</strong> scaled prior is the centered 1.5× solve prior; lineup update is the 2026 lineup evidence added to it.</div>
    </section>
  </main>

  <script>
    const players = {data_json};
    const state = {{
      scope: "qualified",
      priorSource: "all",
      search: "",
      sort: "impact",
      direction: "desc"
    }};
    const tbody = document.getElementById("results-body");
    const resultCount = document.getElementById("result-count");
    const sourceLabels = {{
      current_single_season: "Current season",
      previous_season_fallback: "2025 fallback",
      imputed_zero: "Zero fallback"
    }};

    function score(value) {{
      const numeric = Number(value);
      return `${{numeric > 0 ? "+" : ""}}${{numeric.toFixed(3)}}`;
    }}

    function scoreClass(value) {{
      if (Number(value) > 0) return "positive";
      if (Number(value) < 0) return "negative";
      return "";
    }}

    function filteredPool() {{
      const base = players.filter((player) => {{
        if (state.scope === "qualified" && player.possessions < 1000) return false;
        if (state.priorSource !== "all" && player.priorSource !== state.priorSource) return false;
        return true;
      }});
      const rankOrder = [...base].sort((a, b) => b.impact - a.impact || b.possessions - a.possessions);
      const rank = new Map(rankOrder.map((player, index) => [player.id, index + 1]));
      const query = state.search.trim().toLocaleLowerCase();
      const visible = query
        ? base.filter((player) => player.player.toLocaleLowerCase().includes(query))
        : base;
      visible.sort((a, b) => {{
        const left = a[state.sort];
        const right = b[state.sort];
        let comparison;
        if (typeof left === "string") comparison = left.localeCompare(right);
        else comparison = Number(left) - Number(right);
        if (comparison === 0) comparison = a.player.localeCompare(b.player);
        return state.direction === "asc" ? comparison : -comparison;
      }});
      return {{ visible, rank }};
    }}

    function render() {{
      const {{ visible, rank }} = filteredPool();
      resultCount.textContent = `${{visible.length}} shown · {qualified} qualified · {len(rows)} total`;
      if (!visible.length) {{
        tbody.innerHTML = `<tr><td class="empty" colspan="12">No players match these filters.</td></tr>`;
        return;
      }}
      tbody.innerHTML = visible.map((player) => `
        <tr data-player="${{player.player.replaceAll('"', '&quot;')}}">
          <td class="rank">${{rank.get(player.id)}}</td>
          <td class="player">${{player.player}}<span class="player-id">ID ${{player.id}}</span></td>
          <td class="metric-impact ${{scoreClass(player.impact)}}">${{score(player.impact)}}</td>
          <td class="${{scoreClass(player.offense)}}">${{score(player.offense)}}</td>
          <td class="${{scoreClass(player.defense)}}">${{score(player.defense)}}</td>
          <td class="${{scoreClass(player.boxPrior)}}">${{score(player.boxPrior)}}</td>
          <td class="${{scoreClass(player.boxOffense)}}">${{score(player.boxOffense)}}</td>
          <td class="${{scoreClass(player.boxDefense)}}">${{score(player.boxDefense)}}</td>
          <td class="audit-col ${{scoreClass(player.effectivePrior)}}">${{score(player.effectivePrior)}}</td>
          <td class="audit-col ${{scoreClass(player.lineupUpdate)}}">${{score(player.lineupUpdate)}}</td>
          <td>${{player.possessions.toLocaleString()}}</td>
          <td class="source">${{sourceLabels[player.priorSource] || player.priorSource}}</td>
        </tr>
      `).join("");
    }}

    document.getElementById("player-search").addEventListener("input", (event) => {{
      state.search = event.target.value;
      render();
    }});
    document.getElementById("prior-source").addEventListener("change", (event) => {{
      state.priorSource = event.target.value;
      render();
    }});
    document.querySelectorAll("[data-scope]").forEach((button) => {{
      button.addEventListener("click", () => {{
        state.scope = button.dataset.scope;
        document.querySelectorAll("[data-scope]").forEach((peer) => {{
          peer.setAttribute("aria-pressed", String(peer === button));
        }});
        render();
      }});
    }});
    document.getElementById("audit-toggle").addEventListener("change", (event) => {{
      document.body.classList.toggle("show-audit", event.target.checked);
    }});
    document.querySelectorAll("[data-sort]").forEach((button) => {{
      button.addEventListener("click", () => {{
        const key = button.dataset.sort;
        if (state.sort === key) {{
          state.direction = state.direction === "desc" ? "asc" : "desc";
        }} else {{
          state.sort = key;
          state.direction = key === "player" ? "asc" : "desc";
        }}
        document.querySelectorAll("[data-sort]").forEach((peer) => {{
          const active = peer === button;
          peer.dataset.active = String(active);
          if (active) peer.dataset.direction = state.direction;
          else delete peer.dataset.direction;
        }});
        render();
      }});
    }});
    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_review_rows(args.board, args.priors)
    manifest = json.loads(args.manifest.read_text())
    output = build_html(rows, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} players")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

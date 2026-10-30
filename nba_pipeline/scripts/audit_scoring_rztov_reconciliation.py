#!/usr/bin/env python3
"""Reconciliation receipt for the rZTOV explorer, scoring-impact explorer, and
time-decay playtype-mix and lineup-turnover research columns before a
Databallr integration.

Every check compares two independently written artifacts and records the
largest absolute gap. The audit fails (exit 1) when any gap exceeds its
tolerance, so it can gate a publish.

Checks
------
1. scoring board zTS / SV/100 vs the live Databallr ``player_zts`` rows
   (observed-role seasons only; the board must reproduce production SV/100).
2. scoring board turnover lanes vs the rZTOV season CSV (observed rows 2014+):
   rZTOV, both excess lanes, both lane point prices, and points saved.
3. rZTOV explorer page: the embedded payload is re-evaluated with the page's own
   arithmetic (``seasonRow``/``finish``) and compared with the rZTOV CSV.
4. scoring-impact page: the embedded columnar payload vs the board CSV.
5. every additive identity on the scoring board and the mix board.
6. playtype-mix board spec (two-season window, 550-day half-life, every season
   1997-2026, one-season history only for 1997) and the scoring board's
   ``playtype_mix_rapm`` column vs the mix board's ``mix_value100``.
7. lineup turnover board: its own exact identities (from the manifest and
   recomputed from the CSV), spec (two-season TD550, every season 1997-2026),
   the scoring board's ``lineup_*`` columns vs the lineup CSV times its price,
   split closure on the board, and the pass-through file's season coverage.

``--fetch-live`` refreshes the Databallr comparison snapshot read-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_historical_scoring_security import LINEUP_TOV_LANES, identity_gaps, lineup_tov_gaps


PIPELINE_DIR = Path(__file__).resolve().parents[1]
RESULTS = PIPELINE_DIR / "results"
DEFAULT_BOARD = RESULTS / "historical_scoring_security" / "historical_scoring_security_1997_2026.csv"
DEFAULT_RZTOV = RESULTS / "rztov_seasons_2014_2026.csv"
DEFAULT_MIX = RESULTS / "td_playtype_mix_board" / "td_playtype_mix_board_1997_2026.csv"
MIX_SPEC = {"half_life_days": 550.0, "window_seasons": 2, "first_season": 1997}
DEFAULT_LINEUP_TOV = RESULTS / "lineup_tov_board" / "lineup_tov_board_1997_2026.csv"
DEFAULT_PASS_THROUGH = RESULTS / "historical_scoring_security" / "lineup_tov_pass_through_1997_2026.csv"
DEFAULT_RZTOV_HTML = PIPELINE_DIR / "docs" / "rztov-explorer.html"
DEFAULT_SCORING_HTML = PIPELINE_DIR / "docs" / "historical-scoring-impact.html"
DEFAULT_OUTPUT = RESULTS / "scoring_rztov_integration"
DEFAULT_LIVE = DEFAULT_OUTPUT / "databallr_player_zts_live.csv"
DEFAULT_DATABALLR_ENV = PIPELINE_DIR.parents[2] / "pipelines" / "core" / ".env"  # WORKSPACE.yaml: nba/pipelines/core

# Published values are rounded to 4 dp (CSV) or 3 dp (Databallr), so gaps are
# judged against the coarser rounding of the two sides being compared.
TOL = {
    "databallr": 2e-3,
    "csv4": 2e-4,
    "html_rztov": 1.5e-3,
    # CSVs are written at 6 dp, so a three-term identity can drift by 1.5e-6.
    "identity": 2e-6,
}
OBSERVED_ROLE = "databallr_synergy_observed"


def fetch_live(env_path: Path, output: Path) -> None:
    """Read-only pull of regular-season Databallr ``player_zts`` rows."""
    from dotenv import dotenv_values
    from supabase import create_client

    values = dotenv_values(env_path)
    url = str(values.get("SUPABASE_URL") or "")
    key = str(values.get("SUPABASE_KEY") or values.get("SUPABASE_SECRET_KEY") or "")
    if not url or not key:
        raise RuntimeError(f"Databallr Supabase credentials are missing from {env_path}")
    client = create_client(url, key)
    rows: list[dict[str, object]] = []
    for start in range(0, 200_000, 1_000):
        batch = (
            client.table("player_zts")
            .select("nba_id,year,playoffs,player_name,minutes,tsa100,rts,role,zts,pred_ots_m4_tsa25fit")
            .eq("playoffs", 0)
            .range(start, start + 999)
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < 1_000:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["year", "nba_id"]).to_csv(output, index=False)


def gap(left: pd.Series, right: pd.Series) -> float:
    diff = (pd.to_numeric(left) - pd.to_numeric(right)).abs()
    return float(diff.max()) if len(diff) and diff.notna().any() else 0.0


def check(name: str, value: float, tol: float, **extra: object) -> dict[str, object]:
    return {"check": name, "max_abs_gap": value, "tolerance": tol, "pass": bool(value <= tol), **extra}


def scoring_vs_databallr(board: pd.DataFrame, live: pd.DataFrame) -> list[dict[str, object]]:
    live = live[live["playoffs"].eq(0)].drop_duplicates(["nba_id", "year"])
    observed = board[board["role_source"].eq(OBSERVED_ROLE)]
    merged = observed.merge(live, on=["nba_id", "year"], suffixes=("", "_live"))
    return [
        check("zts_vs_databallr", gap(merged["zts"], merged["zts_live"]), TOL["databallr"], rows=len(merged)),
        check("rts_vs_databallr", gap(merged["rts"], merged["rts_live"]), TOL["databallr"], rows=len(merged)),
        check("role_vs_databallr", gap(merged["role_difficulty"], merged["role"]), TOL["databallr"], rows=len(merged)),
        check(
            "sv100_vs_databallr_pred_ots_m4_tsa25fit",
            gap(merged["scoring_value100"], merged["pred_ots_m4_tsa25fit"]),
            TOL["databallr"],
            rows=len(merged),
            unmatched_observed_rows=int(len(observed) - len(merged)),
        ),
    ]


def board_vs_rztov(board: pd.DataFrame, rztov: pd.DataFrame) -> list[dict[str, object]]:
    observed = board[board["turnover_source"].eq("observed")]
    merged = observed.merge(rztov, on=["nba_id", "year"], suffixes=("", "_rz"))
    rows = {"rows": len(merged), "observed_board_rows": int(len(observed)), "rztov_rows": int(len(rztov))}
    return [
        check("rztov_board_vs_csv", gap(merged["rztov"], merged["rztov_rz"]), TOL["csv4"], **rows),
        check("sc_excess_board_vs_csv", gap(merged["sc_excess"], merged["sc_excess_rz"]), TOL["csv4"], **rows),
        check("bp_excess_board_vs_csv", gap(merged["bp_excess"], merged["bp_excess_rz"]), TOL["csv4"], **rows),
        check("sc_price_board_vs_csv", gap(merged["scoring_tov_point_price"], merged["sc_point_price"]), TOL["csv4"], **rows),
        check("bp_price_board_vs_csv", gap(merged["badpass_tov_point_price"], merged["bp_point_price"]), TOL["csv4"], **rows),
        check(
            "turnover_points_board_vs_csv",
            gap(merged["turnover_value100"], merged["tov_points_saved_100"]),
            TOL["csv4"],
            **rows,
        ),
        check("observed_rows_all_matched", float(len(observed) - len(merged)), 0.0),
    ]


def _page_constant(html: str, name: str) -> object:
    match = re.search(rf"^\s*const {name} = (.*);\s*$", html, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"page has no `const {name} = ...;` line")
    return json.loads(match.group(1))


def rztov_page(html_path: Path, rztov: pd.DataFrame) -> list[dict[str, object]]:
    """Re-evaluate the page's seasonRow/finish arithmetic on its own payload."""
    data = _page_constant(html_path.read_text(encoding="utf-8"), "DATA")
    be = {int(year): block for year, block in data["be"].items()}
    out = []
    for nba_id, year, poss, v_self, v_ast, v_potast, sc, bp, _qual in data["seasons"]:
        b = be[int(year)]
        allow_sc = b["self"] * v_self + b["ast"] * v_ast + b["centre_sc"]
        allow_bp = b["potast"] * v_potast + b["centre_bp"]
        act = sc + bp
        plays = v_self + v_ast + v_potast + act
        excess = act - allow_sc - allow_bp
        out.append(
            {
                "nba_id": int(nba_id),
                "year": int(year),
                "rztov_page": excess / max(plays, 1) * 100,
                "saved_page": -excess,
                "pts_page": (allow_sc - sc) * b["price_sc"] + (allow_bp - bp) * b["price_bp"],
            }
        )
    page = pd.DataFrame(out)
    merged = page.merge(rztov, on=["nba_id", "year"])
    rows = {"rows": len(merged), "page_rows": len(page), "csv_rows": len(rztov)}
    return [
        check("rztov_page_vs_csv", gap(merged["rztov_page"], merged["rztov"]), TOL["html_rztov"], **rows),
        check("saved_page_vs_csv", gap(merged["saved_page"], merged["tov_saved_100"]), TOL["html_rztov"], **rows),
        check("points_page_vs_csv", gap(merged["pts_page"], merged["tov_points_saved_100"]), TOL["html_rztov"], **rows),
        check("rztov_page_rows_match_csv", float(abs(len(page) - len(rztov)) + abs(len(merged) - len(rztov))), 0.0),
    ]


def scoring_page(html_path: Path, board: pd.DataFrame) -> list[dict[str, object]]:
    packed = _page_constant(html_path.read_text(encoding="utf-8"), "PACKED")
    page = pd.DataFrame(packed["rows"], columns=packed["fields"])
    merged = page.merge(board, on=["nba_id", "year"], suffixes=("_page", ""))
    numeric = [
        field
        for field in packed["fields"]
        if field not in {"nba_id", "year", "player_name", "role_source", "turnover_source", "lineup_team_tov_coverage"}
    ]
    worst = max(numeric, key=lambda field: gap(merged[f"{field}_page"], merged[field]))
    value = gap(merged[f"{worst}_page"], merged[worst])
    labels_equal = bool(
        (merged["turnover_source_page"] == merged["turnover_source"]).all()
        and (merged["role_source_page"] == merged["role_source"]).all()
    )
    return [
        check("scoring_page_vs_board", value, TOL["csv4"], rows=len(merged), page_rows=len(page), worst_field=worst),
        check("scoring_page_source_labels_match", 0.0 if labels_equal else 1.0, 0.0),
        check("scoring_page_rows_all_on_board", float(len(page) - len(merged)), 0.0),
    ]


def mix_checks(board: pd.DataFrame, mix: pd.DataFrame, manifest: dict[str, object]) -> list[dict[str, object]]:
    identity = max(
        gap(mix["mix_rapm"], mix["own_mix_contribution"] + mix["teammate_mix_contribution"]),
        gap(mix["mix_rapm"], mix["td_shot_share"] * mix["own_diet"] + (1 - mix["td_shot_share"]) * mix["teammate_diet"]),
        gap(mix["mix_value100"], mix["own_mix_value100"] + mix["teammate_mix_value100"]),
        gap(mix["mix_value100"], -mix["mix_rapm"] / 50 * mix["league_fga_per100"]),
    )
    first, last = MIX_SPEC["first_season"], int(board["year"].max())
    seasons = set(mix["year"].astype(int))
    expected_history = np.where(mix["year"].eq(first), 1, MIX_SPEC["window_seasons"])
    spec_misses = (
        float(abs(float(manifest.get("half_life_days", -1)) - MIX_SPEC["half_life_days"]) > 0)
        + float(int(manifest.get("window_seasons", -1)) != MIX_SPEC["window_seasons"])
        + float(len(set(range(first, last + 1)) - seasons))
        + float((mix["history_seasons"].to_numpy() != expected_history).sum())
    )
    sidecar = board.dropna(subset=["playtype_mix_rapm"]).merge(mix, on=["nba_id", "year"])
    orphans = board["playtype_mix_rapm"].notna().sum() - len(sidecar)
    return [
        check("mix_board_identities", identity, TOL["identity"], rows=len(mix)),
        check(
            "mix_board_spec_two_season_550_all_years",
            spec_misses,
            0.0,
            seasons=f"{min(seasons)}-{max(seasons)}",
            half_life_days=manifest.get("half_life_days"),
            window_seasons=manifest.get("window_seasons"),
        ),
        check(
            "scoring_board_playtype_mix_vs_mix_board",
            gap(sidecar["playtype_mix_rapm"], sidecar["mix_value100"]),
            TOL["csv4"],
            rows=len(sidecar),
        ),
        check("scoring_board_playtype_mix_rows_all_on_mix_board", float(orphans), 0.0),
    ]


def lineup_checks(
    board: pd.DataFrame, lineup: pd.DataFrame, manifest: dict[str, object], pass_through: pd.DataFrame
) -> list[dict[str, object]]:
    parts = {"own": "own", "teammate": "teammate", "team": "unassigned", "total": "total"}
    identity = 0.0
    for lane in LINEUP_TOV_LANES:
        for prefix in (lane, f"{lane}_raw"):
            pieces = sum(lineup[f"{prefix}_{part}_saved100"] for part in ("own", "teammate", "unassigned"))
            identity = max(identity, gap(lineup[f"{prefix}_total_saved100"], pieces))
    identity = max(
        identity,
        gap(lineup["lineup_tov_saved100"], lineup["scoring_total_saved100"] + lineup["badpass_total_saved100"]),
        gap(lineup["lineup_tov_value100"], lineup["scoring_total_value100"] + lineup["badpass_total_value100"]),
    )
    manifest_identity = max(float(value) for value in manifest["identity_checks"].values())
    window = manifest["window"]
    first, last = MIX_SPEC["first_season"], int(board["year"].max())
    seasons = set(lineup["year"].astype(int))
    expected_history = np.where(lineup["year"].eq(first), 1, MIX_SPEC["window_seasons"])
    spec_misses = (
        float(abs(float(window.get("half_life_days", -1)) - MIX_SPEC["half_life_days"]) > 0)
        + float(int(window.get("seasons", -1)) != MIX_SPEC["window_seasons"])
        + float(len(set(range(first, last + 1)) - seasons))
        + float((lineup["history_seasons"].to_numpy() != expected_history).sum())
    )
    joined = board.dropna(subset=["lineup_tov_value100"]).merge(lineup, on=["nba_id", "year"], suffixes=("", "_src"))
    price = joined["tov_point_price"]
    join_gap = gap(joined["lineup_tov_value100"], joined["lineup_tov_value100_src"])
    for lane in LINEUP_TOV_LANES:
        for part, source_part in parts.items():
            join_gap = max(
                join_gap, gap(joined[f"lineup_{lane}_tov_{part}_value100"], joined[f"{lane}_{source_part}_saved100"] * price)
            )
        join_gap = max(join_gap, gap(joined[f"lineup_{lane}_tov_raw_value100"], joined[f"{lane}_raw_total_saved100"] * price))
    orphans = board["lineup_tov_value100"].notna().sum() - len(joined)
    coverage = board.drop_duplicates("year").set_index("year")["lineup_team_tov_coverage"]
    coverage_misses = sum(
        1
        for year, label in coverage.dropna().items()
        if label != ("absent" if year <= 2013 else "partial" if year == 2014 else "full")
    )
    pass_missing = len(set(range(first, last + 1)) - set(pass_through["year"].astype(int)))
    return [
        check("lineup_board_identities", identity, TOL["identity"], rows=len(lineup)),
        check("lineup_board_manifest_identities", manifest_identity, 1e-9),
        check(
            "lineup_board_spec_two_season_550_all_years",
            spec_misses,
            0.0,
            seasons=f"{min(seasons)}-{max(seasons)}",
            half_life_days=window.get("half_life_days"),
            window_seasons=window.get("seasons"),
        ),
        # Board values are the 6 dp lineup CSV times its 6 dp price, then 6 dp again.
        check("scoring_board_lineup_tov_vs_lineup_board", join_gap, 1e-5, rows=len(joined)),
        check("scoring_board_lineup_tov_rows_all_on_lineup_board", float(orphans), 0.0),
        check("scoring_board_lineup_tov_split_closure", max(lineup_tov_gaps(board).values()), 1e-5),
        check("scoring_board_lineup_team_tov_coverage_labels", float(coverage_misses), 0.0),
        check("lineup_pass_through_all_seasons", float(pass_missing), 0.0, seasons=len(pass_through)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--rztov", type=Path, default=DEFAULT_RZTOV)
    parser.add_argument("--mix", type=Path, default=DEFAULT_MIX)
    parser.add_argument("--lineup-tov", type=Path, default=DEFAULT_LINEUP_TOV)
    parser.add_argument("--pass-through", type=Path, default=DEFAULT_PASS_THROUGH)
    parser.add_argument("--rztov-html", type=Path, default=DEFAULT_RZTOV_HTML)
    parser.add_argument("--scoring-html", type=Path, default=DEFAULT_SCORING_HTML)
    parser.add_argument("--live-zts", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--fetch-live", action="store_true", help="refresh the Databallr snapshot (read-only)")
    parser.add_argument("--databallr-env", type=Path, default=DEFAULT_DATABALLR_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fetch_live:
        fetch_live(args.databallr_env, args.live_zts)
    board = pd.read_csv(args.board)
    rztov = pd.read_csv(args.rztov)
    mix = pd.read_csv(args.mix)
    mix_manifest = json.loads((args.mix.parent / "manifest.json").read_text())
    lineup = pd.read_csv(args.lineup_tov)
    lineup_manifest = json.loads((args.lineup_tov.parent / "manifest.json").read_text())
    pass_through = pd.read_csv(args.pass_through)
    live = pd.read_csv(args.live_zts)

    results: list[dict[str, object]] = []
    results += scoring_vs_databallr(board, live)
    results += board_vs_rztov(board, rztov)
    results += rztov_page(args.rztov_html, rztov)
    results += scoring_page(args.scoring_html, board)
    identities = identity_gaps(board)
    results += [check(f"board_{name}", value, TOL["identity"]) for name, value in identities.items()]
    results += mix_checks(board, mix, mix_manifest)
    results += lineup_checks(board, lineup, lineup_manifest, pass_through)

    receipt = {
        "inputs": {
            "board": str(args.board),
            "rztov": str(args.rztov),
            "mix": str(args.mix),
            "lineup_tov": str(args.lineup_tov),
            "pass_through": str(args.pass_through),
            "rztov_html": str(args.rztov_html),
            "scoring_html": str(args.scoring_html),
            "databallr_snapshot": str(args.live_zts),
            "databallr_snapshot_rows": int(len(live)),
        },
        "tolerances": TOL,
        "all_pass": all(item["pass"] for item in results),
        "checks": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "reconciliation_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, default=float) + "\n")
    for item in results:
        flag = "ok  " if item["pass"] else "FAIL"
        print(f"{flag} {item['check']:<48} {item['max_abs_gap']:.3g}")
    print(f"wrote {path}; all_pass={receipt['all_pass']}")
    return 0 if receipt["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

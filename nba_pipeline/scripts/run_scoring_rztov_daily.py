#!/usr/bin/env python3
"""Daily Scoring Impact + rZTOV build for the season in progress.

One command runs the whole chain for the current season and promotes the result
only when the reconciliation audit and the frozen-rows check both pass:

    transition labels -> workload estimator (box backcast) -> rZTOV explorer
    -> playtype-mix board -> lineup turnover board -> scoring board
    -> Scoring Impact page -> reconciliation audit -> frozen-rows check

Every completed season is reused from the frozen ``_2026`` artifacts; only the
season after ``FROZEN_FIT_SEASON`` is built. Nothing is refit on it: break-evens,
lane point prices, estimator labels and the transition calibration stay frozen.

Layout under ``results/scoring_rztov_daily/<season>/``:

* ``work/``     the build in progress (overwritten by every run)
* ``current/``  the last build that passed; consumers read only this
* ``previous/`` the build ``current`` replaced
* ``last_failure.json`` written when a run fails; ``current`` is left untouched

``--solves run`` also produces the trailing five-season count solves and
low-alpha price targets that end at the current season. They are not consumed
while the fit is frozen, so the default is ``skip``.

The season comes from ``--season`` or ``NBA_RAPM_SEASON_YEAR`` (``2027`` is
2026-27), the same variable ``daily_rapm_update.sh`` uses. The run reads
Databallr ``player_zts`` (read-only) and writes nothing outside its output root
except, with ``--solves run``, solve files under ``results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from . import scoring_season as seasons
except ImportError:  # Direct script execution.
    import scoring_season as seasons


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
RESULTS = PIPELINE_DIR / "results"
DOCS = PIPELINE_DIR / "docs"
RAW_DIR = PIPELINE_DIR / "raw_data"
DEFAULT_ROOT = RESULTS / "scoring_rztov_daily"

FIRST_BOARD = 1997
FIRST_OBSERVED = 2014
FIRST_ANCHOR = 2006
WINDOW = 5
COUNT_METRICS = ("TOV", "SCORING_TOV", "BADPASS_TOV")
COUNT_ALPHA = 1500
ROLE_ALPHA = 300
FROZEN_TOLERANCE = 1e-9


def frozen_paths(fit: int) -> dict[str, Path]:
    board = seasons.span_tag(FIRST_BOARD, fit)
    return {
        "box_features": RESULTS / "nba_rztov_box_backcast" / f"all_era_box_features_{board}.parquet",
        "box_ledger": RESULTS / "nba_rztov_box_backcast" / f"nba_box_estimated_rztov_{board}_diagnostic.csv",
        "explorer_seasons": RESULTS / f"rztov_seasons_{seasons.span_tag(FIRST_OBSERVED, fit)}.csv",
        "explorer_page": DOCS / "rztov-explorer.html",
        "mix_dir": RESULTS / "td_playtype_mix_board",
        "mix_board": RESULTS / "td_playtype_mix_board" / f"td_playtype_mix_board_{board}.csv",
        "lineup_board": RESULTS / "lineup_tov_board" / f"lineup_tov_board_{board}.csv",
        "scoring_board": RESULTS / "historical_scoring_security" / f"historical_scoring_security_{board}.csv",
        "role_cache": RESULTS / "historical_scoring_security" / f"pbp_role_proxy_{board}.parquet",
        "role_prices": RESULTS / "historical_scoring_security" / f"pbp_role_category_prices_{board}.csv",
        "transition_calibration": RESULTS / "sq_transition_backfill" / "transition_calibration.json",
    }


def work_paths(work: Path, season: int) -> dict[str, Path]:
    board = seasons.span_tag(FIRST_BOARD, season)
    return {
        "transition": work / "transition",
        "box": work / "box_backcast",
        "box_features": work / "box_backcast" / f"all_era_box_features_{board}.parquet",
        "box_ledger": work / "box_backcast" / f"nba_box_estimated_rztov_{board}_diagnostic.csv",
        "explorer": work / "explorer",
        "explorer_page": work / "explorer" / "rztov-explorer.html",
        "explorer_seasons": work / "explorer" / f"rztov_seasons_{seasons.span_tag(FIRST_OBSERVED, season)}.csv",
        "mix": work / "mix",
        "mix_board": work / "mix" / f"td_playtype_mix_board_{board}.csv",
        "lineup": work / "lineup",
        "lineup_board": work / "lineup" / f"lineup_tov_board_{board}.csv",
        "scoring": work / "scoring",
        "scoring_board": work / "scoring" / f"historical_scoring_security_{board}.csv",
        "scoring_page": work / "scoring" / "historical-scoring-impact.html",
        "pass_through": work / "scoring" / f"lineup_tov_pass_through_{board}.csv",
        "anchor": work / "scoring" / f"databallr_synergy_zts_{seasons.span_tag(FIRST_ANCHOR, season)}.csv",
        "audit": work / "audit",
        "live_zts": work / "audit" / "databallr_player_zts_live.csv",
        "logs": work / "logs",
    }


def step_commands(
    season: int, fit: int, work: Path, offline: bool, solves: bool
) -> list[tuple[str, list[str]]]:
    """The chain as (name, argv) pairs; every child runs under ``sys.executable``."""
    f = frozen_paths(fit)
    w = work_paths(work, season)
    first_new = fit + 1
    py = sys.executable

    def script(name: str) -> str:
        return str(SCRIPT_DIR / name)

    steps: list[tuple[str, list[str]]] = []
    if solves:
        lo = season - WINDOW + 1
        for metric in COUNT_METRICS:
            steps.append(
                (
                    f"count_solve_{metric.lower()}",
                    [
                        py, script("rapm.py"), metric, f"{lo % 100:02d}", f"{season % 100:02d}", "RS",
                        "--season-effects", "--off-alpha", str(COUNT_ALPHA), "--def-alpha", str(COUNT_ALPHA),
                    ],
                )
            )
        steps.append(
            (
                "price_targets",
                [py, script("build_tov_price_targets.py"), "--first-end", str(season), "--last-end", str(season)],
            )
        )
    steps.append(
        (
            "transition",
            [
                py, script("score_historical_transition.py"),
                "--start-year", str(first_new), "--end-year", str(season),
                "--season-types", "rs", "--output-dir", str(w["transition"]), "--overwrite",
            ],
        )
    )
    steps.append(
        (
            "box_backcast",
            [
                py, script("build_nba_rztov_box_backcast.py"), "--last-season", str(season),
                "--output-dir", str(w["box"]), "--seed-feature-cache", str(f["box_features"]),
                "--bootstrap-draws", "0",
            ],
        )
    )
    steps.append(
        (
            "explorer",
            [
                py, script("build_rztov_explorer.py"), "--out-dir", str(w["explorer"]),
                "--csv-dir", str(w["explorer"]), "--html", str(w["explorer_page"]),
                "--frozen-payload", str(f["explorer_page"]), "--frozen-seasons", str(f["explorer_seasons"]),
                "--last-season", str(season), "--box-ledger", str(w["box_ledger"]),
            ],
        )
    )
    steps.append(
        (
            "playtype_mix",
            [
                py, script("build_td_playtype_mix_board.py"), "--seed-dir", str(f["mix_dir"]),
                "--last-feature-season", str(season), "--possessions", str(w["box_ledger"]),
                "--observed-rztov", str(w["explorer_seasons"]), "--transition-dir", str(w["transition"]),
                "--output-dir", str(w["mix"]),
            ],
        )
    )
    steps.append(
        (
            "lineup_tov",
            [
                py, script("build_lineup_tov_board.py"),
                "--first-season", str(first_new), "--last-season", str(season),
                "--role-alpha-scoring", str(ROLE_ALPHA), "--role-alpha-badpass", str(ROLE_ALPHA),
                "--seed-board", str(f["lineup_board"]), "--observed-rztov", str(w["explorer_seasons"]),
                "--output-dir", str(w["lineup"]),
            ],
        )
    )
    scoring = [
        py, script("build_historical_scoring_security.py"), "--last-season", str(season),
        "--box-features", str(w["box_features"]), "--rztov-ledger", str(w["box_ledger"]),
        "--observed-rztov", str(w["explorer_seasons"]), "--playtype-mix", str(w["mix_board"]),
        "--lineup-tov", str(w["lineup_board"]), "--seed-role-cache", str(f["role_cache"]),
        "--seed-role-prices", str(f["role_prices"]), "--synergy-anchor", str(w["anchor"]),
        "--output-dir", str(w["scoring"]),
    ]
    if not offline:
        scoring.append("--refresh-anchor-from-databallr")
    steps.append(("scoring_board", scoring))
    steps.append(
        (
            "scoring_page",
            [
                py, script("build_historical_scoring_impact_html.py"),
                "--input", str(w["scoring_board"]), "--output", str(w["scoring_page"]),
            ],
        )
    )
    audit = [
        py, script("audit_scoring_rztov_reconciliation.py"),
        "--board", str(w["scoring_board"]), "--rztov", str(w["explorer_seasons"]),
        "--mix", str(w["mix_board"]), "--lineup-tov", str(w["lineup_board"]),
        "--pass-through", str(w["pass_through"]), "--rztov-html", str(w["explorer_page"]),
        "--scoring-html", str(w["scoring_page"]), "--live-zts", str(w["live_zts"]),
        "--output-dir", str(w["audit"]),
    ]
    if not offline:
        audit.append("--fetch-live")
    steps.append(("audit", audit))
    return steps


def compare_frozen_rows(
    new: pd.DataFrame, frozen: pd.DataFrame, fit: int, keys: tuple[str, ...] = ("nba_id", "year")
) -> dict[str, object]:
    """Rows through ``fit`` in a daily board must equal the frozen board.

    Columns that exist only in the daily board (season status, for example) are
    ignored; every frozen column must be present and equal, including where it
    is blank.
    """
    year = keys[-1]
    a = new[new[year].le(fit)].sort_values(list(keys)).reset_index(drop=True)
    b = frozen[frozen[year].le(fit)].sort_values(list(keys)).reset_index(drop=True)
    missing = [c for c in b.columns if c not in a.columns]
    result: dict[str, object] = {
        "rows_daily": int(len(a)),
        "rows_frozen": int(len(b)),
        "missing_columns": missing,
        "differing_columns": [],
        "max_abs_gap": 0.0,
    }
    if missing or len(a) != len(b):
        result["ok"] = False
        return result
    differing: list[str] = []
    worst = 0.0
    for column in b.columns:
        if pd.api.types.is_numeric_dtype(b[column]):
            x = pd.to_numeric(a[column], errors="coerce")
            blank_mismatch = bool((x.isna() != b[column].isna()).any())
            gap = float((x - b[column]).abs().max()) if b[column].notna().any() else 0.0
            worst = max(worst, gap)
            if blank_mismatch or gap > FROZEN_TOLERANCE:
                differing.append(column)
        elif not a[column].fillna("").astype(str).equals(b[column].fillna("").astype(str)):
            differing.append(column)
    result["differing_columns"] = differing
    result["max_abs_gap"] = worst
    result["ok"] = not differing
    return result


def frozen_rows_check(season: int, fit: int, work: Path) -> dict[str, dict[str, object]]:
    f = frozen_paths(fit)
    w = work_paths(work, season)
    pairs = {
        "scoring_board": (w["scoring_board"], f["scoring_board"]),
        "rztov_seasons": (w["explorer_seasons"], f["explorer_seasons"]),
        "playtype_mix_board": (w["mix_board"], f["mix_board"]),
        "lineup_tov_board": (w["lineup_board"], f["lineup_board"]),
    }
    return {
        name: compare_frozen_rows(pd.read_csv(new, low_memory=False), pd.read_csv(old, low_memory=False), fit)
        for name, (new, old) in pairs.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    def run(*argv: str) -> str:
        done = subprocess.run(["git", *argv], cwd=PIPELINE_DIR, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def promote(root: Path, work: Path) -> None:
    """Copy the passed build to ``current``; the old ``current`` becomes ``previous``."""
    staged = root / "current.new"
    current = root / "current"
    previous = root / "previous"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(work, staged, ignore=shutil.ignore_patterns("logs"))
    if current.exists():
        if previous.exists():
            shutil.rmtree(previous)
        current.rename(previous)
    staged.rename(current)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, default=seasons.current_season(),
                        help=f"season end year; default ${seasons.SEASON_ENV} or {seasons.FALLBACK_SEASON}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT,
                        help="parent of the per-season output directory")
    parser.add_argument("--solves", choices=("skip", "run"), default="skip",
                        help="also produce the current trailing-window count solves and price targets "
                             "(not consumed while the fit is frozen)")
    parser.add_argument("--offline", action="store_true",
                        help="do not read Databallr; reuse the zTS snapshot of the last passed build")
    parser.add_argument("--no-promote", action="store_true", help="build and check but leave current/ alone")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    season = int(args.season)
    fit = seasons.FROZEN_FIT_SEASON
    if season <= fit:
        raise SystemExit(
            f"season {season} is inside the frozen fit (through {fit}); "
            "the frozen artifacts already cover it and this runner must not rebuild them"
        )
    missing = [str(path) for path in frozen_paths(fit).values() if not path.exists()]
    if missing:
        raise SystemExit("missing frozen artifacts:\n  " + "\n  ".join(missing))
    raw_file = RAW_DIR / f"NBA{season % 100:02d}.parquet"
    games = seasons.rs_games_played(RAW_DIR, season)
    if games == 0:
        raise SystemExit(f"no regular-season games in {raw_file}; nothing to build")

    root = args.output_root / str(season)
    work = root / "work"
    w = work_paths(work, season)
    started = datetime.now(timezone.utc)
    build_id = f"{started:%Y%m%dT%H%M%SZ}-g{games}"

    if work.exists():
        shutil.rmtree(work)
    for key in ("transition", "box", "explorer", "mix", "lineup", "scoring", "audit", "logs"):
        w[key].mkdir(parents=True, exist_ok=True)
    f = frozen_paths(fit)
    shutil.copy2(f["transition_calibration"], w["transition"] / "transition_calibration.json")
    shutil.copy2(f["explorer_page"], w["explorer_page"])
    if args.offline:
        last = work_paths(root / "current", season)
        for key in ("anchor", "live_zts"):
            if not last[key].exists():
                raise SystemExit(f"--offline needs {last[key]} from a passed build")
            shutil.copy2(last[key], w[key])

    manifest: dict[str, object] = {
        "build_id": build_id,
        "season": season,
        "season_label": seasons.season_label(season),
        "fit_through": fit,
        "season_state": seasons.season_state(season, games, fit),
        "started_utc": started.isoformat(timespec="seconds"),
        "git": git_state(),
        "raw_pbp": {"file": raw_file.name, "sha256": sha256(raw_file), "regular_season_games": games},
        "offline": bool(args.offline),
        "solves": (
            "run" if args.solves == "run"
            else "skipped: break-evens and lane prices are frozen through "
                 f"{fit}, so a window ending {season} is not consumed"
        ),
        "steps": [],
        "status": "failed",
    }

    def finish(status: str, code: int) -> int:
        manifest["status"] = status
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (work / "daily_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if code:
            (root / "last_failure.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"FAILED ({status}); {root / 'current'} is unchanged. See {w['logs']}")
        return code

    for name, command in step_commands(season, fit, work, args.offline, args.solves == "run"):
        t0 = time.time()
        done = subprocess.run(command, cwd=SCRIPT_DIR, capture_output=True, text=True)
        seconds = round(time.time() - t0, 1)
        log = w["logs"] / f"{name}.log"
        log.write_text(
            "$ " + " ".join(command) + "\n\nSTDOUT:\n" + done.stdout + "\n\nSTDERR:\n" + done.stderr
        )
        manifest["steps"].append({"step": name, "returncode": done.returncode, "seconds": seconds})
        print(f"{'ok  ' if done.returncode == 0 else 'FAIL'} {name:<22} {seconds:>7.1f}s")
        if done.returncode:
            tail = (done.stdout + done.stderr).strip().splitlines()[-15:]
            print("\n".join("     " + line for line in tail))
            return finish(f"step {name} failed", 1)

    receipt = json.loads((w["audit"] / "reconciliation_receipt.json").read_text())
    manifest["audit"] = {
        "all_pass": bool(receipt.get("all_pass")),
        "checks": len(receipt.get("checks", [])),
    }
    if not receipt.get("all_pass"):
        return finish("reconciliation audit failed", 1)

    frozen = frozen_rows_check(season, fit, work)
    manifest["frozen_rows"] = frozen
    for name, result in frozen.items():
        print(f"{'ok  ' if result['ok'] else 'FAIL'} frozen rows: {name:<20} "
              f"{result['rows_daily']} rows, max gap {result['max_abs_gap']:.2g}")
    if not all(result["ok"] for result in frozen.values()):
        return finish("rows through the frozen fit changed", 1)

    board = pd.read_csv(w["scoring_board"], usecols=["year", "qualified_scoring_security"])
    live = board[board["year"].eq(season)]
    manifest["current_season_rows"] = {
        "scoring_board": int(len(live)),
        "qualified": int(live["qualified_scoring_security"].sum()),
    }
    finish("passed", 0)
    if args.no_promote:
        print(f"passed {build_id}; not promoted (--no-promote). Build is in {work}")
        return 0
    promote(root, work)
    failure = root / "last_failure.json"
    if failure.exists():
        failure.unlink()
    print(f"passed {build_id}; promoted to {root / 'current'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

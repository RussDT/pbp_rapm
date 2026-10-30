#!/usr/bin/env python3
"""Build current-regular-season factor profiles for playoff translation research.

Each target season uses only that season's completed regular season.  Fixed
alphas come from existing factor practice and are not selected on playoff
outcomes.  The generated panel is research input; it does not publish or
replace any canonical six-factor artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAPM_SCRIPT = ROOT / "scripts" / "rapm.py"
RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "results" / "playoff_droppers" / "current_rs_factor_profiles.csv"

METRIC_SPECS = {
    "TS": {"off_alpha": 3000, "def_alpha": 5000, "prefix": "ts"},
    "TOV": {"off_alpha": 1500, "def_alpha": 1500, "prefix": "tov"},
    "REB": {"off_alpha": 3000, "def_alpha": 3000, "prefix": "reb"},
    "THREE_FREQ": {"off_alpha": 3000, "def_alpha": 3000, "prefix": "three_freq"},
}


def result_path(season: int, metric: str) -> Path:
    spec = METRIC_SPECS[metric]
    alpha_suffix = ""
    if (spec["off_alpha"], spec["def_alpha"]) != (3000, 3000):
        alpha_suffix = f"_a{spec['off_alpha']}_{spec['def_alpha']}"
    return RESULTS_DIR / f"{spec['prefix']}_{season % 100:02d}_rs_rb{alpha_suffix}_results.csv"


def run_factor_solve(season: int, metric: str, *, cores: int, force: bool) -> Path:
    output = result_path(season, metric)
    if output.exists() and not force:
        return output
    spec = METRIC_SPECS[metric]
    command = [
        sys.executable,
        str(RAPM_SCRIPT),
        metric,
        f"{season % 100:02d}",
        f"{season % 100:02d}",
        "RS",
        "--rubberband",
        "--off-alpha",
        str(spec["off_alpha"]),
        "--def-alpha",
        str(spec["def_alpha"]),
        "--cores",
        str(cores),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Factor solve failed for {season} {metric}:\n{completed.stdout}\n{completed.stderr}"
        )
    if not output.exists():
        raise FileNotFoundError(f"Factor solve did not create expected output: {output}")
    return output


def assemble_season_profiles(season: int, paths: dict[str, Path]) -> pd.DataFrame:
    """Merge solver-native outputs and export positive-good defense once."""
    merged = None
    for metric, path in paths.items():
        frame = pd.read_csv(path)
        required = {"player_id", "player_name", "off", "def", "off_poss", "def_poss"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        prefix = METRIC_SPECS[metric]["prefix"]
        part = frame[["player_id", "player_name", "off", "def", "off_poss", "def_poss"]].rename(
            columns={
                "player_name": f"player_name_{prefix}",
                "off": f"{prefix}_off",
                "def": f"{prefix}_def_solver",
                "off_poss": f"{prefix}_off_poss",
                "def_poss": f"{prefix}_def_poss",
            }
        )
        merged = part if merged is None else merged.merge(part, on="player_id", how="outer", validate="one_to_one")
    assert merged is not None
    name_columns = [column for column in merged if column.startswith("player_name_")]
    merged["player_name"] = merged[name_columns].bfill(axis=1).iloc[:, 0]
    merged.insert(0, "season", int(season))
    merged["oTS"] = merged["ts_off"]
    merged["dTS"] = -merged["ts_def_solver"]
    merged["oTOV"] = merged["tov_off"]
    merged["dTOV"] = -merged["tov_def_solver"]
    merged["oREB"] = merged["reb_off"]
    merged["dREB"] = -merged["reb_def_solver"]
    merged["oTHREE_FREQ"] = merged["three_freq_off"]
    merged["dTHREE_FREQ"] = -merged["three_freq_def_solver"]
    merged["off_poss"] = merged["ts_off_poss"]
    merged["def_poss"] = merged["ts_def_poss"]
    merged["source_kind"] = "current_regular_season_only"
    merged["source_max_season"] = int(season)
    keep = [
        "season",
        "player_id",
        "player_name",
        "oTS",
        "oTOV",
        "oREB",
        "oTHREE_FREQ",
        "dTS",
        "dTOV",
        "dREB",
        "dTHREE_FREQ",
        "off_poss",
        "def_poss",
        "source_kind",
        "source_max_season",
    ]
    return merged[keep].sort_values("player_id").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2007)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_season > args.end_season:
        raise ValueError("start-season must not exceed end-season")
    seasons = list(range(args.start_season, args.end_season + 1))
    task_paths: dict[tuple[int, str], Path] = {}
    tasks = [(season, metric) for season in seasons for metric in METRIC_SPECS]
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                run_factor_solve,
                season,
                metric,
                cores=args.cores,
                force=args.force,
            ): (season, metric)
            for season, metric in tasks
        }
        for future in as_completed(futures):
            season, metric = futures[future]
            task_paths[(season, metric)] = future.result()
            print(f"Solved {season} {metric}")

    frames = []
    source_rows = []
    for season in seasons:
        paths = {metric: task_paths[(season, metric)] for metric in METRIC_SPECS}
        frames.append(assemble_season_profiles(season, paths))
        source_rows.extend(
            {"season": season, "metric": metric, "path": str(path)}
            for metric, path in paths.items()
        )
        print(f"Prepared current-RS factor profile for {season}")
    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "status": "research_only",
        "timing": "target season completed regular season only",
        "seasons": [int(args.start_season), int(args.end_season)],
        "metrics": METRIC_SPECS,
        "rubberband": {"model": "smooth", "alpha": 100},
        "rows": int(len(output)),
        "players": int(output["player_id"].nunique()),
        "sources": source_rows,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(output):,} player-seasons to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build four within-season trailing-six-season TD RAPM snapshots per year.

Each checkpoint includes the previous five complete seasons plus the current
season through the date on which 25%, 50%, 75%, or 100% of that season's
RS+PS games have been completed. Solves are pure RAPM, ALL, RB, SE, TD550.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
OUTPUT_ROOT = PIPELINE_ROOT / "results" / "quarterly_td550"
TMP_ROOT = PIPELINE_ROOT / "tmp" / "quarterly_td550"
HALF_LIFE = 550
SUFFIX = "all_pure_rb_se_td550"


@dataclass(frozen=True)
class Snapshot:
    end_year: int
    quarter: int
    cutoff_date: str
    games_included: int
    season_games: int

    @property
    def start_year(self) -> int:
        return self.end_year - 5

    @property
    def year_range(self) -> str:
        return f"{self.start_year % 100:02d}_{self.end_year % 100:02d}"

    @property
    def checkpoint(self) -> str:
        return f"q{self.quarter}_{self.cutoff_date}"


def _season_file(year: int, playoffs: bool = False) -> Path:
    return PROCESSED_DIR / f"RAPM{year % 100:02d}{'_PS' if playoffs else ''}.parquet"


def season_game_table(end_year: int) -> pd.DataFrame:
    parts = []
    for playoffs in (False, True):
        path = _season_file(end_year, playoffs)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["game_id", "game_date"])
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        frame = frame.dropna(subset=["game_id", "game_date"])
        parts.append(frame[["game_id", "game_date"]].drop_duplicates("game_id"))
    games = pd.concat(parts, ignore_index=True).drop_duplicates("game_id")
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if games.empty:
        raise ValueError(f"No dated games found for season ending {end_year}")
    return games


def build_snapshots(end_year: int) -> list[Snapshot]:
    games = season_game_table(end_year)
    snapshots = []
    total = len(games)
    for quarter in range(1, 5):
        target = math.ceil(total * quarter / 4)
        cutoff = games.iloc[target - 1]["game_date"]
        included = int((games["game_date"] <= cutoff).sum())
        snapshots.append(
            Snapshot(
                end_year=end_year,
                quarter=quarter,
                cutoff_date=cutoff.date().isoformat(),
                games_included=included,
                season_games=total,
            )
        )
    return snapshots


def expected_result(snapshot: Snapshot, output_root: Path = OUTPUT_ROOT) -> Path:
    return (
        output_root / str(snapshot.end_year) / snapshot.checkpoint / "td"
        / f"rapm_{snapshot.year_range}_{SUFFIX}_results.csv"
    )


def expected_artifacts(snapshot: Snapshot, output_root: Path = OUTPUT_ROOT) -> list[Path]:
    result = expected_result(snapshot, output_root)
    stem = result.name.removesuffix("_results.csv")
    return [
        result,
        result.with_name(f"{stem}_rubberband_coefficients.csv"),
        result.with_name(f"{stem}_rubberband_effects.csv"),
        result.with_name(f"{stem}_season_effects.csv"),
    ]


def _link_full_seasons(snapshot: Snapshot, work_dir: Path) -> None:
    for year in range(snapshot.start_year, snapshot.end_year):
        for playoffs in (False, True):
            source = _season_file(year, playoffs).resolve()
            target = work_dir / source.name
            target.symlink_to(source)


def _write_current_partial(snapshot: Snapshot, work_dir: Path) -> None:
    cutoff = pd.Timestamp(snapshot.cutoff_date)
    for playoffs in (False, True):
        source = _season_file(snapshot.end_year, playoffs)
        frame = pd.read_parquet(source)
        dates = pd.to_datetime(frame["game_date"], errors="coerce")
        partial = frame.loc[dates.notna() & dates.le(cutoff)].copy()
        partial.to_parquet(work_dir / source.name, index=False)


def run_snapshot(snapshot: Snapshot, cores: int, force: bool) -> dict:
    output_file = expected_result(snapshot)
    if all(path.exists() for path in expected_artifacts(snapshot)) and not force:
        rows = len(pd.read_csv(output_file, usecols=["player_id"]))
        return {
            **asdict(snapshot),
            "status": "existing",
            "output": str(output_file),
            "player_rows": rows,
            "elapsed_seconds": 0.0,
        }

    checkpoint_root = output_file.parents[1]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"{snapshot.end_year}_q{snapshot.quarter}_", dir=TMP_ROOT) as temp:
        work_dir = Path(temp)
        _link_full_seasons(snapshot, work_dir)
        _write_current_partial(snapshot, work_dir)
        cmd = [
            sys.executable, str(SCRIPT_DIR / "rapm.py"), "RAPM",
            str(snapshot.start_year % 100), str(snapshot.end_year % 100), "ALL",
            "--pure", "--rubberband", "--season-effects",
            "--half-life", str(HALF_LIFE), "--cores", str(cores),
        ]
        env = os.environ.copy()
        env["NBA_PIPELINE_PROCESSED_DIR"] = str(work_dir)
        env["NBA_PIPELINE_RESULTS_DIR"] = str(checkpoint_root)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (checkpoint_root / "run.log").write_text(
            "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr
        )
        if result.returncode:
            tail = "\n".join((result.stderr or result.stdout).splitlines()[-40:])
            raise RuntimeError(f"{snapshot.end_year} Q{snapshot.quarter} failed; see {checkpoint_root / 'run.log'}\n{tail}")
    if not output_file.exists():
        raise FileNotFoundError(f"Snapshot solve completed without expected output: {output_file}")
    rows = len(pd.read_csv(output_file, usecols=["player_id"]))
    return {
        **asdict(snapshot),
        "status": "built",
        "output": str(output_file),
        "player_rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def parse_end_years(value: str) -> list[int]:
    if not value:
        return list(range(2002, 2027))
    years = []
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    years = sorted(set(years))
    invalid = [year for year in years if year < 2002 or year > 2026]
    if invalid:
        raise ValueError(f"End years outside the complete 1997-2026 six-season range: {invalid}")
    return years


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-years", default="", help="e.g. 2002-2026 or 2025,2026")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cores-per-rapm", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    end_years = parse_end_years(args.end_years)
    snapshots = [snapshot for year in end_years for snapshot in build_snapshots(year)]
    print(f"Quarterly TD550 snapshots={len(snapshots)} across {len(end_years)} trailing-six-season windows", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_snapshot, snapshot, args.cores_per_rapm, args.force): snapshot
            for snapshot in snapshots
        }
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(snapshots)}] {record['end_year']} Q{record['quarter']} "
                f"as-of {record['cutoff_date']}: {record['status']} ({record['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    records = sorted(records, key=lambda row: (row["end_year"], row["quarter"]))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT_ROOT / "quarterly_td550_manifest.csv", index=False)
    (OUTPUT_ROOT / "quarterly_td550_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {OUTPUT_ROOT / 'quarterly_td550_manifest.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

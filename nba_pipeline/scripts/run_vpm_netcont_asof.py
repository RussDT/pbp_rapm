#!/usr/bin/env python3
"""Build historical as-of no-prior net-continuation VPM snapshots.

Each season-end year receives four estimates: October 1 before the season,
January 1 during the season, May 1 late in the season, and August 30 after the
season.  Every component is
fit by the canonical RAPM solver with smooth rubberband, RS/PS season effects,
TD550, and the frozen VPM component-specific offense/defense alphas.

Partial seasons are rebuilt from raw games strictly before the cutoff before
the net-continuation and EV targets are attached.  This is necessary because
the continuation kappa values and shot-tree baselines are file-level values;
filtering a completed-season target parquet would leak later games.
"""

from __future__ import annotations

import argparse
import json
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
RAW_DIR = PIPELINE_ROOT / "raw_data"
CANONICAL_PROCESSED = PIPELINE_ROOT / "processed"
NETCONT_PROCESSED = PIPELINE_ROOT / "processed_netcont_lane"
NETCONT_TARGETS = PIPELINE_ROOT / "results" / "vpm_target_swap_netcont" / "processed"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "results" / "vpm_netcont_asof_td550"
RESEARCH_ROOT = PIPELINE_ROOT.parent.parent / "vpm"
EV_16_20 = RESEARCH_ROOT / "vpm" / "data" / "russell_sq_v2_ev_16_22.parquet"
EV_21_26 = RESEARCH_ROOT / "vpm" / "data" / "russell_sq_v2_ev_21_26.parquet"

HALF_LIFE = 550
START_YEAR = 2016
COMPONENTS = {
    "DECOMP_NET_RIM_FREQ": (750, 750),
    "DECOMP_NET_RIM_FG": (1000, 750),
    "DECOMP_NET_MID_FG": (750, 1500),
    "DECOMP_NET_THREE_FREQ": (1000, 2000),
    "DECOMP_NET_THREE_FG": (1000, 2000),
    "ALT_FT": (750, 750),
    "ALT_TOV_NET_VALUE": (2000, 6000),
    "DECOMP_NET_MID_VALUE": (8000, 48000),
    "DECOMP_NET_EFG": (6000, 48000),
    "SECOND_CHANCE_CLEAN": (2000, 3000),
}
SHOT_COMPONENTS = [
    component
    for component in COMPONENTS
    if component not in {"ALT_TOV_NET_VALUE", "SECOND_CHANCE_CLEAN"}
]


@dataclass(frozen=True)
class Snapshot:
    season_end_year: int
    label: str
    cutoff_date: str

    @property
    def load_end_year(self) -> int:
        return pd.Timestamp(self.cutoff_date).year

    @property
    def slug(self) -> str:
        return f"{self.label}_{self.cutoff_date}"


def season_snapshots(season_end_year: int) -> list[Snapshot]:
    return [
        Snapshot(season_end_year, "oct01", f"{season_end_year - 1}-10-01"),
        Snapshot(season_end_year, "jan01", f"{season_end_year}-01-01"),
        Snapshot(season_end_year, "may01", f"{season_end_year}-05-01"),
        Snapshot(season_end_year, "aug30", f"{season_end_year}-08-30"),
    ]


def _phase_suffix(playoffs: bool) -> str:
    return "_PS" if playoffs else ""


def _normalized_game_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    valid = numeric.notna()
    out.loc[valid] = numeric.loc[valid].astype("int64").astype(str).str.zfill(10)
    return out


def _eligible_game_ids(year: int, playoffs: bool, cutoff: pd.Timestamp) -> set[str]:
    suffix = _phase_suffix(playoffs)
    path = CANONICAL_PROCESSED / f"RAPM{year % 100:02d}{suffix}.parquet"
    if not path.exists():
        return set()
    games = pd.read_parquet(path, columns=["game_id", "game_date"])
    games["game_date"] = pd.to_datetime(games["game_date"], errors="raise")
    games = games.loc[games["game_date"] < cutoff, "game_id"]
    return set(_normalized_game_id(games))


def _all_game_ids(year: int, playoffs: bool) -> set[str]:
    suffix = _phase_suffix(playoffs)
    path = CANONICAL_PROCESSED / f"RAPM{year % 100:02d}{suffix}.parquet"
    if not path.exists():
        return set()
    return set(_normalized_game_id(pd.read_parquet(path, columns=["game_id"])["game_id"]))


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def _filter_parquet_by_games(source: Path, destination: Path, game_ids: set[str]) -> int:
    frame = pd.read_parquet(source)
    keep = _normalized_game_id(frame["game_id"]).isin(game_ids)
    filtered = frame.loc[keep].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(destination, index=False)
    return len(filtered)


def _run(cmd: list[str], *, log_path: Path, env: dict[str, str] | None = None) -> None:
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(cmd)
        + f"\n\nelapsed_seconds={time.time() - started:.3f}"
        + "\n\nSTDOUT:\n" + result.stdout
        + "\n\nSTDERR:\n" + result.stderr
    )
    if result.returncode:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-50:])
        raise RuntimeError(f"Command failed; see {log_path}\n{tail}")


def _ev_source(year: int) -> Path:
    return EV_16_20 if year <= 2020 else EV_21_26


def _prepare_partial_phase(
    year: int,
    playoffs: bool,
    game_ids: set[str],
    target_dir: Path,
    work_root: Path,
    log_dir: Path,
) -> dict[str, int | str]:
    suffix = _phase_suffix(playoffs)
    raw_name = f"NBA{year % 100:02d}{suffix}.parquet"
    raw_source = RAW_DIR / raw_name
    if not raw_source.exists():
        raise FileNotFoundError(raw_source)
    partial_raw_dir = work_root / "raw"
    partial_processed_dir = work_root / "processed"
    partial_raw_dir.mkdir(parents=True, exist_ok=True)
    partial_processed_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _filter_parquet_by_games(raw_source, partial_raw_dir / raw_name, game_ids)
    if raw_rows == 0:
        raise ValueError(f"No raw rows survived for {raw_name}")

    phase = "ps" if playoffs else "rs"
    for metric in ("FIRST_CHANCE", "SECOND_CHANCE_CLEAN"):
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "reprocess_metric.py"),
            metric,
            str(year),
            str(year),
            "--season-types",
            phase,
            "--workers",
            "1",
            "--raw-dir",
            str(partial_raw_dir),
            "--processed-dir",
            str(partial_processed_dir),
        ]
        _run(cmd, log_path=log_dir / f"prepare_{metric.lower()}_{year}_{phase}.log")
        expected = partial_processed_dir / f"{metric}{year % 100:02d}{suffix}.parquet"
        if not expected.exists():
            raise FileNotFoundError(f"Processor did not produce {expected}")

    target_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_vpm_target_swap.py"),
        "--ev-parquet",
        str(_ev_source(year)),
        "--years",
        str(year),
        "--season-types",
        phase,
        "--raw-dir",
        str(partial_raw_dir),
        "--processed-dir",
        str(partial_processed_dir),
        "--output-dir",
        str(target_dir),
        "--report",
        str(log_dir / f"target_attach_{year}_{phase}.csv"),
        "--net-lane",
        "--overwrite",
    ]
    _run(target_cmd, log_path=log_dir / f"target_attach_{year}_{phase}.log")

    rapm_source = CANONICAL_PROCESSED / f"RAPM{year % 100:02d}{suffix}.parquet"
    rapm_rows = _filter_parquet_by_games(
        rapm_source,
        target_dir / rapm_source.name,
        game_ids,
    )
    return {
        "year": year,
        "phase": phase,
        "mode": "partial_rebuild",
        "games": len(game_ids),
        "raw_rows": raw_rows,
        "rapm_rows": rapm_rows,
    }


def prepare_snapshot_inputs(snapshot: Snapshot, snapshot_root: Path) -> list[dict]:
    cutoff = pd.Timestamp(snapshot.cutoff_date)
    target_dir = snapshot_root / "processed"
    log_dir = snapshot_root / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="vpm_netcont_asof_") as temp:
        temp_root = Path(temp)
        for year in range(START_YEAR, snapshot.load_end_year + 1):
            for playoffs in (False, True):
                suffix = _phase_suffix(playoffs)
                eligible = _eligible_game_ids(year, playoffs, cutoff)
                all_games = _all_game_ids(year, playoffs)
                if not eligible:
                    continue
                phase = "ps" if playoffs else "rs"
                if eligible == all_games:
                    for family, source_dir in (
                        ("FIRST_CHANCE", NETCONT_TARGETS),
                        ("SECOND_CHANCE_CLEAN", NETCONT_TARGETS),
                        ("RAPM", CANONICAL_PROCESSED),
                    ):
                        name = f"{family}{year % 100:02d}{suffix}.parquet"
                        source = source_dir / name
                        if not source.exists():
                            raise FileNotFoundError(source)
                        _link(source, target_dir / name)
                    records.append(
                        {
                            "year": year,
                            "phase": phase,
                            "mode": "complete_link",
                            "games": len(eligible),
                        }
                    )
                else:
                    phase_root = temp_root / f"{year}_{phase}"
                    records.append(
                        _prepare_partial_phase(
                            year,
                            playoffs,
                            eligible,
                            target_dir,
                            phase_root,
                            log_dir,
                        )
                    )
    return records


def _result_file(component: str, components_dir: Path) -> Path:
    matches = sorted(components_dir.rglob(f"{component.lower()}_*_results.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one result for {component} in {components_dir}; found {matches}"
        )
    return matches[0]


def solve_component(
    component: str,
    alphas: tuple[int, int],
    snapshot: Snapshot,
    snapshot_root: Path,
    cores: int,
) -> dict:
    components_dir = snapshot_root / "components"
    components_dir.mkdir(parents=True, exist_ok=True)
    off_alpha, def_alpha = alphas
    existing = sorted(components_dir.rglob(f"{component.lower()}_*_results.csv"))
    if len(existing) == 1:
        return {
            "component": component,
            "off_alpha": off_alpha,
            "def_alpha": def_alpha,
            "elapsed_seconds": 0.0,
            "status": "existing",
            "result": str(existing[0]),
        }
    if len(existing) > 1:
        raise ValueError(f"Multiple existing results for {component}: {existing}")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "rapm.py"),
        component,
        str(START_YEAR % 100),
        str(snapshot.load_end_year % 100),
        "ALL",
        "--rubberband",
        "--season-effects",
        "--half-life",
        str(HALF_LIFE),
        "--off-alpha",
        str(off_alpha),
        "--def-alpha",
        str(def_alpha),
        "--cores",
        str(cores),
    ]
    env = os.environ.copy()
    env["NBA_PIPELINE_PROCESSED_DIR"] = str(snapshot_root / "processed")
    env["NBA_PIPELINE_RESULTS_DIR"] = str(components_dir)
    log_path = snapshot_root / "logs" / f"solve_{component.lower()}.log"
    started = time.time()
    _run(cmd, log_path=log_path, env=env)
    result_path = _result_file(component, components_dir)
    return {
        "component": component,
        "off_alpha": off_alpha,
        "def_alpha": def_alpha,
        "elapsed_seconds": round(time.time() - started, 3),
        "status": "built",
        "result": str(result_path),
    }


def assemble_snapshot(snapshot: Snapshot, snapshot_root: Path) -> tuple[pd.DataFrame, dict]:
    merged: pd.DataFrame | None = None
    for component in COMPONENTS:
        frame = pd.read_csv(_result_file(component, snapshot_root / "components"))
        required = {"player_id", "player_name", "off", "def", "possessions", "off_poss", "def_poss"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{component} result missing {sorted(missing)}")
        values = frame[["player_id", "player_name", "off", "def", "possessions", "off_poss", "def_poss"]].copy()
        values = values.rename(
            columns={
                "player_name": f"name_{component}",
                "off": f"o_{component}",
                "def": f"d_solver_{component}",
                "possessions": f"poss_{component}",
                "off_poss": f"off_poss_{component}",
                "def_poss": f"def_poss_{component}",
            }
        )
        merged = values if merged is None else merged.merge(values, on="player_id", how="outer", validate="one_to_one")

    assert merged is not None
    component_presence = [merged[f"o_{component}"].notna() for component in COMPONENTS]
    if not pd.concat(component_presence, axis=1).all(axis=None):
        missing_rows = int((~pd.concat(component_presence, axis=1).all(axis=1)).sum())
        raise ValueError(f"Component player universes differ for {missing_rows} players")

    first = next(iter(COMPONENTS))
    out = pd.DataFrame({"player_id": merged["player_id"]})
    name_columns = [f"name_{component}" for component in COMPONENTS]
    out["player_name"] = merged[name_columns].bfill(axis=1).iloc[:, 0]
    out["possessions"] = merged[f"poss_{first}"]
    out["off_poss"] = merged[f"off_poss_{first}"]
    out["def_poss"] = merged[f"def_poss_{first}"]
    for component in COMPONENTS:
        out[f"o_{component}"] = merged[f"o_{component}"]
        out[f"d_{component}"] = -merged[f"d_solver_{component}"]

    out["vpm_off"] = out[[f"o_{component}" for component in COMPONENTS]].sum(axis=1)
    out["vpm_def"] = out[[f"d_{component}" for component in COMPONENTS]].sum(axis=1)
    out["vpm_net"] = out["vpm_off"] + out["vpm_def"]
    out["oTS"] = out[[f"o_{component}" for component in SHOT_COMPONENTS]].sum(axis=1)
    out["oTOV"] = out["o_ALT_TOV_NET_VALUE"]
    out["oSC"] = out["o_SECOND_CHANCE_CLEAN"]
    out["dTS"] = out[[f"d_{component}" for component in SHOT_COMPONENTS]].sum(axis=1)
    out["dTOV"] = out["d_ALT_TOV_NET_VALUE"]
    out["dSC"] = out["d_SECOND_CHANCE_CLEAN"]
    out.insert(2, "season_end_year", snapshot.season_end_year)
    out.insert(3, "as_of_date", snapshot.cutoff_date)
    out.insert(4, "pricing", "net_continuation_kappa")
    out.insert(5, "prior_scale", 0.0)

    identity = {
        "max_abs_off_component_gap": float(
            (out["vpm_off"] - out[[f"o_{c}" for c in COMPONENTS]].sum(axis=1)).abs().max()
        ),
        "max_abs_def_component_gap": float(
            (out["vpm_def"] - out[[f"d_{c}" for c in COMPONENTS]].sum(axis=1)).abs().max()
        ),
        "max_abs_net_gap": float((out["vpm_net"] - out["vpm_off"] - out["vpm_def"]).abs().max()),
        "max_abs_off_three_factor_gap": float((out["vpm_off"] - out[["oTS", "oTOV", "oSC"]].sum(axis=1)).abs().max()),
        "max_abs_def_three_factor_gap": float((out["vpm_def"] - out[["dTS", "dTOV", "dSC"]].sum(axis=1)).abs().max()),
    }
    if max(identity.values()) > 1e-9:
        raise ValueError(f"Assembled VPM identity failed: {identity}")

    numeric = out.select_dtypes(include="number").columns.difference(["player_id", "season_end_year"])
    out[numeric] = out[numeric].round(3)
    out = out.sort_values(["vpm_net", "player_id"], ascending=[False, True]).reset_index(drop=True)
    manifest = {
        "schema_version": 1,
        **asdict(snapshot),
        "training_season_end_years": list(range(START_YEAR, snapshot.load_end_year + 1)),
        "half_life_days": HALF_LIFE,
        "rubberband": "smooth_clock_aware_alpha100",
        "season_effects": "rs_ps_aware",
        "prior_scale": 0.0,
        "defense_sign": "positive_is_good",
        "units": "league-centered points per 100 possessions",
        "components": {
            component: {"off_alpha": alphas[0], "def_alpha": alphas[1]}
            for component, alphas in COMPONENTS.items()
        },
        "player_count": len(out),
        "identity": identity,
    }
    return out, manifest


def run_snapshot(
    snapshot: Snapshot,
    output_root: Path,
    workers: int,
    cores_per_rapm: int,
    force: bool,
) -> dict:
    snapshot_root = output_root / str(snapshot.season_end_year) / snapshot.slug
    output_csv = snapshot_root / "player_estimates.csv"
    output_parquet = snapshot_root / "player_estimates.parquet"
    manifest_path = snapshot_root / "manifest.json"
    if output_csv.exists() and output_parquet.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        return {**asdict(snapshot), "status": "existing", "output": str(output_csv), "player_count": manifest["player_count"]}

    snapshot_root.mkdir(parents=True, exist_ok=True)
    prep = prepare_snapshot_inputs(snapshot, snapshot_root)
    solve_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                solve_component,
                component,
                alphas,
                snapshot,
                snapshot_root,
                cores_per_rapm,
            ): component
            for component, alphas in COMPONENTS.items()
        }
        for future in as_completed(futures):
            solve_records.append(future.result())

    board, manifest = assemble_snapshot(snapshot, snapshot_root)
    manifest["input_preparation"] = prep
    manifest["solves"] = sorted(solve_records, key=lambda row: row["component"])
    board.to_csv(output_csv, index=False)
    board.to_parquet(output_parquet, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**asdict(snapshot), "status": "built", "output": str(output_csv), "player_count": len(board)}


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            years.extend(range(start, end + 1))
        elif token:
            years.append(int(token))
    years = sorted(set(years))
    if not years or min(years) < 2020 or max(years) > 2026:
        raise ValueError("Season-end years must lie within 2020-2026")
    return years


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season-end-years", default="2020-2026")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    years = parse_years(args.season_end_years)
    snapshots = [snapshot for year in years for snapshot in season_snapshots(year)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index, snapshot in enumerate(snapshots, 1):
        started = time.time()
        print(f"[{index}/{len(snapshots)}] {snapshot.season_end_year} {snapshot.label} as of {snapshot.cutoff_date}", flush=True)
        record = run_snapshot(snapshot, args.output_root, args.workers, args.cores_per_rapm, args.force)
        record["elapsed_seconds"] = round(time.time() - started, 3)
        records.append(record)
        print(f"  {record['status']}: players={record['player_count']} elapsed={record['elapsed_seconds']:.1f}s", flush=True)

    manifest_csv = args.output_root / "snapshot_manifest.csv"
    manifest_json = args.output_root / "snapshot_manifest.json"
    pd.DataFrame(records).to_csv(manifest_csv, index=False)
    manifest_json.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {manifest_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

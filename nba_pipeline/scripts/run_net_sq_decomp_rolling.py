#!/usr/bin/env python3
"""Build rolling Net DECOMP tables with ShotQuality EV plus realized residuals.

Two alpha profiles are intentionally distinct:

``explanatory``
    Every component uses 2000/4000 so the ten-component SQ/residual split can
    be compared directly with the parent Net DECOMP accounting.

``vpm``
    Uses the frozen component-specific no-prior VPM penalties. This is the
    predictive board profile and must not be described as the parent RAPM.

The input directory is the isolated output of ``build_vpm_target_swap.py
--net-lane``. It must contain FIRST_CHANCE, SECOND_CHANCE_CLEAN, and RAPM
parquets for every requested RS/PS season.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_PROCESSED_DIR = (
    PIPELINE_ROOT / "results" / "vpm_target_swap_netcont" / "processed"
)
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "results" / "net_sq_decomp_rolling"
DEFAULT_MASTER_ROOT = PIPELINE_ROOT / "master_results" / "net_sq_decomp_rolling"

PARENT_ALPHAS = (2000, 4000)
MAX_SEPARATE_PARENT_GAP = 0.30
VPM_ALPHAS = {
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
COMPONENTS = tuple(VPM_ALPHAS)
SHOT_COMPONENTS = tuple(
    component
    for component in COMPONENTS
    if component not in {"ALT_TOV_NET_VALUE", "SECOND_CHANCE_CLEAN"}
)


@dataclass(frozen=True)
class Window:
    start_year: int
    end_year: int

    @property
    def compact(self) -> str:
        if self.start_year == self.end_year:
            return f"{self.end_year % 100:02d}"
        return f"{self.start_year % 100:02d}_{self.end_year % 100:02d}"

    @property
    def seasons(self) -> int:
        return self.end_year - self.start_year + 1


def normalize_year(value: int) -> int:
    value = int(value)
    if value < 100:
        return 2000 + value if value <= 50 else 1900 + value
    return value


def parse_window(value: str) -> Window:
    normalized = value.strip().replace("_", "-")
    if "-" in normalized:
        start_text, end_text = normalized.split("-", 1)
    else:
        start_text = end_text = normalized
    window = Window(normalize_year(int(start_text)), normalize_year(int(end_text)))
    if window.seasons < 1:
        raise ValueError(f"Window ends before it starts: {value}")
    return window


def current_release_windows(end_year: int = 2026) -> list[Window]:
    return [Window(end_year - seasons + 1, end_year) for seasons in range(1, 6)]


def all_five_season_windows() -> list[Window]:
    return [Window(start, start + 4) for start in range(1997, 2023)]


def all_one_to_five_season_windows(
    end_year: int = 2026, first_year: int = 1997
) -> list[Window]:
    if end_year < first_year:
        raise ValueError(f"release end year {end_year} precedes first year {first_year}")
    return [
        Window(window_end - seasons + 1, window_end)
        for window_end in range(first_year, end_year + 1)
        for seasons in range(1, min(5, window_end - first_year + 1) + 1)
    ]


def alpha_map(profile: str) -> dict[str, tuple[int, int]]:
    if profile == "explanatory":
        return {component: PARENT_ALPHAS for component in COMPONENTS}
    if profile == "vpm":
        return dict(VPM_ALPHAS)
    raise ValueError(f"Unknown alpha profile: {profile}")


def spec_id(profile: str, timedecay: bool, half_life: int) -> str:
    base = "net_sq_resid_a2000_4000" if profile == "explanatory" else "net_sq_resid_vpmalpha"
    return f"{base}_td{half_life}" if timedecay else base


def expected_input_paths(processed_dir: Path, window: Window) -> list[Path]:
    paths: list[Path] = []
    for year in range(window.start_year, window.end_year + 1):
        yy = year % 100
        for suffix in ("", "_PS"):
            for family in ("FIRST_CHANCE", "SECOND_CHANCE_CLEAN", "RAPM"):
                paths.append(processed_dir / f"{family}{yy:02d}{suffix}.parquet")
    return paths


def _result_file(component: str, components_dir: Path) -> Path:
    matches = sorted(components_dir.rglob(f"{component.lower()}_*_results.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {component} result in {components_dir}; found {matches}"
        )
    return matches[0]


def _run_solve(
    component: str,
    alphas: tuple[int, int],
    window: Window,
    processed_dir: Path,
    components_dir: Path,
    log_dir: Path,
    cores: int,
    timedecay: bool,
    half_life: int,
    force: bool,
) -> dict[str, object]:
    existing = sorted(components_dir.rglob(f"{component.lower()}_*_results.csv"))
    if existing and not force:
        if len(existing) != 1:
            raise ValueError(f"Multiple existing results for {component}: {existing}")
        return {"component": component, "status": "existing", "result": str(existing[0])}

    off_alpha, def_alpha = alphas
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "rapm.py"),
        component,
        str(window.start_year % 100),
        str(window.end_year % 100),
        "ALL",
        "--rubberband",
        "--season-effects",
        "--off-alpha",
        str(off_alpha),
        "--def-alpha",
        str(def_alpha),
        "--cores",
        str(cores),
    ]
    if component == "RAPM":
        cmd.append("--pure")
    if timedecay:
        cmd.extend(["--half-life", str(half_life)])

    components_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NBA_PIPELINE_PROCESSED_DIR"] = str(processed_dir)
    env["NBA_PIPELINE_RESULTS_DIR"] = str(components_dir)
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - started
    log_path = log_dir / f"solve_{component.lower()}.log"
    log_path.write_text(
        "$ "
        + " ".join(cmd)
        + f"\n\nelapsed_seconds={elapsed:.3f}"
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr
    )
    if result.returncode:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-50:])
        raise RuntimeError(f"{component} {window.compact} failed; see {log_path}\n{tail}")
    output = _result_file(component, components_dir)
    return {
        "component": component,
        "status": "built",
        "result": str(output),
        "elapsed_seconds": round(elapsed, 3),
        "off_alpha": off_alpha,
        "def_alpha": def_alpha,
    }


def _component_frame(component: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "player_id",
        "player_name",
        "off",
        "def",
        "net_rapm",
        "possessions",
        "off_poss",
        "def_poss",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def assemble_window(
    window: Window,
    components_dir: Path,
    profile: str,
    timedecay: bool,
    half_life: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    parent = _component_frame("RAPM", _result_file("RAPM", components_dir))
    out = parent[
        ["player_id", "player_name", "possessions", "off_poss", "def_poss"]
    ].copy()
    out["parent_off"] = parent["off"]
    out["parent_def"] = -parent["def"]
    out["parent_net"] = parent["net_rapm"]

    for component in COMPONENTS:
        frame = _component_frame(component, _result_file(component, components_dir))
        values = frame[["player_id", "off", "def"]].rename(
            columns={"off": f"o_{component}", "def": f"d_solver_{component}"}
        )
        out = out.merge(values, on="player_id", how="left", validate="one_to_one")
        if out[f"o_{component}"].isna().any():
            missing_ids = out.loc[out[f"o_{component}"].isna(), "player_id"].head(10).tolist()
            raise ValueError(f"{component} player universe misses parent players: {missing_ids}")
        out[f"d_{component}"] = -out.pop(f"d_solver_{component}")
        out[f"n_{component}"] = out[f"o_{component}"] + out[f"d_{component}"]

    out["estimate_off"] = out[[f"o_{component}" for component in COMPONENTS]].sum(axis=1)
    out["estimate_def"] = out[[f"d_{component}" for component in COMPONENTS]].sum(axis=1)
    out["estimate_net"] = out["estimate_off"] + out["estimate_def"]
    out["oTS"] = out[[f"o_{component}" for component in SHOT_COMPONENTS]].sum(axis=1)
    out["dTS"] = out[[f"d_{component}" for component in SHOT_COMPONENTS]].sum(axis=1)
    out["oTOV"] = out["o_ALT_TOV_NET_VALUE"]
    out["dTOV"] = out["d_ALT_TOV_NET_VALUE"]
    out["oSC"] = out["o_SECOND_CHANCE_CLEAN"]
    out["dSC"] = out["d_SECOND_CHANCE_CLEAN"]
    out["parent_gap_off"] = out["estimate_off"] - out["parent_off"]
    out["parent_gap_def"] = out["estimate_def"] - out["parent_def"]
    out["parent_gap_net"] = out["estimate_net"] - out["parent_net"]

    out.insert(2, "start_year", window.start_year)
    out.insert(3, "end_year", window.end_year)
    out.insert(4, "pricing", "net_continuation_kappa")
    out.insert(5, "shot_target", "russell_sq_v2_ev_plus_realized_residuals")
    out.insert(6, "alpha_profile", profile)
    out.insert(7, "half_life_days", half_life if timedecay else pd.NA)

    identity = {
        "max_abs_net_sum_gap": float(
            (out["estimate_net"] - out["estimate_off"] - out["estimate_def"]).abs().max()
        ),
        "max_abs_off_three_factor_gap": float(
            (out["estimate_off"] - out[["oTS", "oTOV", "oSC"]].sum(axis=1)).abs().max()
        ),
        "max_abs_def_three_factor_gap": float(
            (out["estimate_def"] - out[["dTS", "dTOV", "dSC"]].sum(axis=1)).abs().max()
        ),
        "max_abs_parent_gap_off": float(out["parent_gap_off"].abs().max()),
        "max_abs_parent_gap_def": float(out["parent_gap_def"].abs().max()),
        "max_abs_parent_gap_net": float(out["parent_gap_net"].abs().max()),
    }
    if max(
        identity["max_abs_net_sum_gap"],
        identity["max_abs_off_three_factor_gap"],
        identity["max_abs_def_three_factor_gap"],
    ) > 1e-9:
        raise ValueError(f"Assembled SQ/residual identity failed: {identity}")
    if profile == "explanatory" and max(
        identity["max_abs_parent_gap_off"],
        identity["max_abs_parent_gap_def"],
        identity["max_abs_parent_gap_net"],
    ) > MAX_SEPARATE_PARENT_GAP:
        raise ValueError(f"Uniform-alpha SQ/residual sum is too far from parent: {identity}")

    numeric = out.select_dtypes(include="number").columns.difference(
        ["player_id", "start_year", "end_year"]
    )
    out[numeric] = out[numeric].round(3)
    out = out.sort_values(["estimate_net", "player_id"], ascending=[False, True]).reset_index(drop=True)
    manifest = {
        "schema_version": 1,
        "series_id": spec_id(profile, timedecay, half_life),
        "start_year": window.start_year,
        "end_year": window.end_year,
        "season_type": "ALL",
        "pricing": "net_continuation_kappa",
        "shot_target": "russell_sq_v2_ev_plus_realized_residuals",
        "alpha_profile": profile,
        "component_alphas": {
            component: {"off": values[0], "def": values[1]}
            for component, values in alpha_map(profile).items()
        },
        "parent_alphas": {"off": PARENT_ALPHAS[0], "def": PARENT_ALPHAS[1]},
        "timedecay": timedecay,
        "half_life_days": half_life if timedecay else None,
        "rubberband": "smooth_clock_aware_alpha100",
        "season_effects": "rs_ps_aware",
        "defense_sign": "positive_is_good",
        "units": "league-centered points per 100 possessions",
        "player_count": len(out),
        "identity": identity,
    }
    return out, manifest


def output_stem(window: Window, profile: str, timedecay: bool, half_life: int) -> str:
    if profile == "explanatory":
        base = f"weighted_factors_decomp_net_sq_resid_{window.compact}_all_rb_se_a2000_4000"
    else:
        base = f"weighted_factors_vpm_net_sq_resid_{window.compact}_all_rb_se"
    return f"{base}_td{half_life}" if timedecay else base


def run_window(
    window: Window,
    processed_dir: Path,
    output_root: Path,
    master_root: Path,
    profile: str,
    timedecay: bool,
    half_life: int,
    workers: int,
    cores_per_rapm: int,
    force: bool,
) -> dict[str, object]:
    series = spec_id(profile, timedecay, half_life)
    window_root = output_root / series / window.compact
    components_dir = window_root / "components"
    log_dir = window_root / "logs"
    stem = output_stem(window, profile, timedecay, half_life)
    output_csv = window_root / f"{stem}.csv"
    output_parquet = window_root / f"{stem}.parquet"
    manifest_path = window_root / "manifest.json"
    if output_csv.exists() and output_parquet.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        return {
            "window": window.compact,
            "status": "existing",
            "output": str(output_csv),
            "player_count": manifest["player_count"],
        }

    missing_inputs = [path for path in expected_input_paths(processed_dir, window) if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            f"{window.compact} is missing {len(missing_inputs)} target inputs; first: {missing_inputs[:6]}"
        )

    jobs = {"RAPM": PARENT_ALPHAS, **alpha_map(profile)}
    solve_records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_solve,
                component,
                alphas,
                window,
                processed_dir,
                components_dir,
                log_dir,
                cores_per_rapm,
                timedecay,
                half_life,
                force,
            ): component
            for component, alphas in jobs.items()
        }
        for future in as_completed(futures):
            solve_records.append(future.result())

    board, manifest = assemble_window(window, components_dir, profile, timedecay, half_life)
    manifest["solves"] = sorted(solve_records, key=lambda row: str(row["component"]))
    window_root.mkdir(parents=True, exist_ok=True)
    board.to_csv(output_csv, index=False)
    board.to_parquet(output_parquet, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    master_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_csv, master_root / output_csv.name)
    shutil.copy2(output_parquet, master_root / output_parquet.name)
    shutil.copy2(manifest_path, master_root / f"{stem}.manifest.json")
    return {
        "window": window.compact,
        "status": "built",
        "output": str(output_csv),
        "player_count": len(board),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--windows", default="", help="Comma-separated season windows.")
    group.add_argument("--current-release", action="store_true", help="Current 1Y-5Y family.")
    group.add_argument(
        "--all-five-season-windows",
        action="store_true",
        help="All 26 rolling five-season windows from 1997-2001 through 2022-2026.",
    )
    group.add_argument(
        "--all-one-to-five-season-windows",
        action="store_true",
        help="Every rolling 1Y-5Y window from 1997 through --release-end-year.",
    )
    parser.add_argument("--release-end-year", type=int, default=2026)
    parser.add_argument("--profile", choices=["explanatory", "vpm"], default="explanatory")
    parser.add_argument("--timedecay", action="store_true")
    parser.add_argument("--half-life", type=int, default=550)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--master-root", type=Path, default=DEFAULT_MASTER_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.windows:
        windows = [parse_window(value) for value in args.windows.split(",") if value.strip()]
    elif args.all_one_to_five_season_windows:
        windows = all_one_to_five_season_windows(args.release_end_year)
    elif args.all_five_season_windows:
        windows = all_five_season_windows()
    else:
        windows = current_release_windows(args.release_end_year)

    series = spec_id(args.profile, args.timedecay, args.half_life)
    print(f"series={series} windows={len(windows)}")
    if args.dry_run:
        for window in windows:
            missing = len(
                [path for path in expected_input_paths(args.processed_dir, window) if not path.exists()]
            )
            print(f"PLAN {window.compact} components={len(COMPONENTS) + 1} missing_inputs={missing}")
        return 0

    records: list[dict[str, object]] = []
    for index, window in enumerate(windows, 1):
        print(f"[{index}/{len(windows)}] {series} {window.compact}", flush=True)
        records.append(
            run_window(
                window,
                args.processed_dir,
                args.output_root,
                args.master_root,
                args.profile,
                args.timedecay,
                args.half_life,
                args.workers,
                args.cores_per_rapm,
                args.force,
            )
        )
    summary = args.output_root / series / "run_manifest.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"schema_version": 1, "series_id": series, "windows": records}, indent=2) + "\n")
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

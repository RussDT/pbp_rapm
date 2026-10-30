#!/usr/bin/env python3
"""Build and optionally publish the 2020-2026 no-prior VPM snapshots."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "nba_pipeline" / "results" / "vpm_netcont_asof_td550"
DEFAULT_OUTPUT = DEFAULT_SOURCE_ROOT / "supabase_rows.csv"
TABLE = "nba_no_prior_vpm_td550_snapshots"
EXPECTED_SNAPSHOTS = 28
SOURCE_COLUMN_MAP = {
    "as_of_date": "snapshot_date",
    "o_DECOMP_NET_RIM_FREQ": "o_decomp_net_rim_freq",
    "d_DECOMP_NET_RIM_FREQ": "d_decomp_net_rim_freq",
    "o_DECOMP_NET_RIM_FG": "o_decomp_net_rim_fg",
    "d_DECOMP_NET_RIM_FG": "d_decomp_net_rim_fg",
    "o_DECOMP_NET_MID_FG": "o_decomp_net_mid_fg",
    "d_DECOMP_NET_MID_FG": "d_decomp_net_mid_fg",
    "o_DECOMP_NET_THREE_FREQ": "o_decomp_net_three_freq",
    "d_DECOMP_NET_THREE_FREQ": "d_decomp_net_three_freq",
    "o_DECOMP_NET_THREE_FG": "o_decomp_net_three_fg",
    "d_DECOMP_NET_THREE_FG": "d_decomp_net_three_fg",
    "o_ALT_FT": "o_alt_ft",
    "d_ALT_FT": "d_alt_ft",
    "o_ALT_TOV_NET_VALUE": "o_alt_tov_net_value",
    "d_ALT_TOV_NET_VALUE": "d_alt_tov_net_value",
    "o_DECOMP_NET_MID_VALUE": "o_decomp_net_mid_value",
    "d_DECOMP_NET_MID_VALUE": "d_decomp_net_mid_value",
    "o_DECOMP_NET_EFG": "o_decomp_net_efg",
    "d_DECOMP_NET_EFG": "d_decomp_net_efg",
    "o_SECOND_CHANCE_CLEAN": "o_second_chance_clean",
    "d_SECOND_CHANCE_CLEAN": "d_second_chance_clean",
    "oTS": "ots",
    "oTOV": "otov",
    "oSC": "osc",
    "dTS": "dts",
    "dTOV": "dtov",
    "dSC": "dsc",
}


def _chunks(records: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _with_transport_retry(operation: Any, label: str, attempts: int = 5) -> Any:
    import httpx

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except httpx.TransportError:
            if attempt == attempts:
                raise
            delay = 0.5 * (2 ** (attempt - 1))
            print(f"retrying {label} after transport failure ({attempt}/{attempts})")
            time.sleep(delay)


def build_publish_frame(source_root: Path = DEFAULT_SOURCE_ROOT) -> pd.DataFrame:
    paths = sorted(source_root.glob("*/*/player_estimates.csv"))
    if len(paths) != EXPECTED_SNAPSHOTS:
        raise ValueError(f"expected {EXPECTED_SNAPSHOTS} snapshot files, found {len(paths)}")

    frames: list[pd.DataFrame] = []
    alpha_contract: str | None = None
    for path in paths:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        alphas = json.dumps(manifest["components"], sort_keys=True, separators=(",", ":"))
        if alpha_contract is None:
            alpha_contract = alphas
        elif alphas != alpha_contract:
            raise ValueError(f"component alpha drift in {path.parent}")

        frame = pd.read_csv(path, low_memory=False).rename(columns=SOURCE_COLUMN_MAP)
        frame["checkpoint"] = path.parent.name.split("_", 1)[0]
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="raise")
        frame["snapshot_year"] = frame["snapshot_date"].dt.year
        frame["snapshot_month"] = frame["snapshot_date"].dt.month
        frame["time_decay_half_life_days"] = 550
        frame["rubberband_model"] = "smooth_clock_aware"
        frame["season_effects"] = "rs_ps"
        frame["component_alphas"] = alphas
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["snapshot_date"] = combined["snapshot_date"].dt.strftime("%Y-%m-%d")
    if combined.duplicated(["player_id", "snapshot_date"]).any():
        raise ValueError("duplicate (player_id, snapshot_date) rows")
    if set(combined["prior_scale"]) != {0.0}:
        raise ValueError("publication contains nonzero priors")
    if combined["snapshot_date"].nunique() != EXPECTED_SNAPSHOTS:
        raise ValueError("publication does not contain 28 distinct snapshot dates")

    lead = [
        "player_id", "player_name", "season_end_year", "snapshot_date",
        "snapshot_year", "snapshot_month", "checkpoint", "pricing", "prior_scale",
        "time_decay_half_life_days", "rubberband_model", "season_effects",
        "component_alphas", "possessions", "off_poss", "def_poss",
    ]
    return combined[lead + [column for column in combined.columns if column not in lead]]


def publish(frame: pd.DataFrame, batch_size: int) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    from supabase import create_client
    client = create_client(url, key)
    records = json.loads(frame.to_json(orient="records"))
    for number, batch in enumerate(_chunks(records, batch_size), start=1):
        _with_transport_retry(
            lambda batch=batch: client.table(TABLE)
            .upsert(batch, on_conflict="player_id,snapshot_date")
            .execute(),
            f"batch {number}",
        )
        print(f"uploaded batch {number}: {len(batch)} rows")

    count = _with_transport_retry(
        lambda: client.table(TABLE).select("player_id", count="exact", head=True).execute(),
        "verification",
    ).count
    if count != len(frame):
        raise RuntimeError(f"remote row count {count} does not match local {len(frame)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000:
        raise ValueError("--batch-size must be between 1 and 1000")
    frame = build_publish_frame(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(frame), "snapshots": frame.snapshot_date.nunique(), "output": str(args.output)}))
    if args.publish:
        publish(frame, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

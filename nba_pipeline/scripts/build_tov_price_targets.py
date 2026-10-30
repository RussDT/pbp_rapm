#!/usr/bin/env python3
"""Low-alpha lane turnover RAPM used only as the dependent variable of the
rZTOV lane point-price fit.

The lane prices regress point-valued lane turnover RAPM on box excess. A ridge
coefficient that is shrunk hard toward zero pulls that slope down by the same
factor (noise in the dependent variable does not bias a slope; shrinkage does).
The alpha-2000/4000 Net DECOMP components therefore understate the price by
roughly a third to a half. This runner solves the identical net-continuation
child targets (`ALT_{SCORING,BADPASS}_TOV_NET_VALUE`, rubberband + season
effects, RS+PS) at a near-unpenalized alpha for every trailing five-season
window. The files feed the price fit only; they are not a published rating.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
NET_PROCESSED_DIR = PIPELINE_ROOT / "processed_netcont_lane"
RESULTS_DIR = PIPELINE_ROOT / "results" / "tov_price_targets"
LOG_DIR = PIPELINE_ROOT / "logs" / "tov_price_targets"
PRICE_OFF_ALPHA = 20
PRICE_DEF_ALPHA = 40
WINDOW = 5
LANES = ("ALT_SCORING_TOV_NET_VALUE", "ALT_BADPASS_TOV_NET_VALUE")


def target_name(lane: str, lo: int, hi: int) -> str:
    return (
        f"{lane.lower()}_{lo % 100:02d}_{hi % 100:02d}_all_rb_se_"
        f"a{PRICE_OFF_ALPHA}_{PRICE_DEF_ALPHA}_results.csv"
    )


def solve(lane: str, lo: int, hi: int, cores: int) -> tuple[str, float]:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "rapm.py"), lane,
        str(lo % 100), str(hi % 100), "ALL",
        "--rubberband", "--season-effects",
        "--off-alpha", str(PRICE_OFF_ALPHA), "--def-alpha", str(PRICE_DEF_ALPHA),
        "--cores", str(cores),
    ]
    env = os.environ.copy()
    env["NBA_PIPELINE_PROCESSED_DIR"] = str(NET_PROCESSED_DIR)
    env["NBA_PIPELINE_RESULTS_DIR"] = str(RESULTS_DIR)
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = LOG_DIR / target_name(lane, lo, hi).replace("_results.csv", ".log")
    log.write_text("$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{lane} {lo}-{hi} failed; see {log}")
    return f"{lane} {lo}-{hi}", time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-end", type=int, default=2001)
    parser.add_argument("--last-end", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cores-per-rapm", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (lane, hi - WINDOW + 1, hi)
        for hi in range(args.first_end, args.last_end + 1)
        for lane in LANES
        if args.force or not (RESULTS_DIR / target_name(lane, hi - WINDOW + 1, hi)).exists()
    ]
    print(f"{len(jobs)} solves at alpha {PRICE_OFF_ALPHA}/{PRICE_DEF_ALPHA}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve, lane, lo, hi, args.cores_per_rapm) for lane, lo, hi in jobs]
        for future in as_completed(futures):
            name, elapsed = future.result()
            print(f"done {name} in {elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

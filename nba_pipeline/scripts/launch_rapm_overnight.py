#!/usr/bin/env python3
"""Launch the RAPM overnight queue as a detached, logged process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
STATE_DIR = PIPELINE_ROOT / "results" / "rapm_overnight"
LAUNCH_PATH = STATE_DIR / "launch.json"
LOG_DIR = PIPELINE_ROOT / "logs" / "rapm_overnight"


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-end-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if LAUNCH_PATH.exists():
        previous = json.loads(LAUNCH_PATH.read_text())
        previous_pid = int(previous.get("pid", -1))
        if process_is_running(previous_pid):
            raise RuntimeError(
                f"RAPM overnight queue already appears active as PID {previous_pid}; "
                f"see {previous.get('log')}"
            )

    command = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "run_rapm_overnight.py"),
        "--release-end-year",
        str(args.release_end_year),
        "--workers",
        str(args.workers),
        "--cores-per-rapm",
        str(args.cores_per_rapm),
    ]
    if args.force:
        command.append("--force")
    if args.skip_history:
        command.append("--skip-history")
    if args.dry_run:
        print(" ".join(command))
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    launched_at = datetime.now(timezone.utc)
    log_path = LOG_DIR / f"rapm_overnight_{launched_at.strftime('%Y%m%dT%H%M%SZ')}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "schema_version": 1,
        "launched_at": launched_at.isoformat(),
        "pid": process.pid,
        "log": str(log_path.relative_to(REPO_ROOT)),
        "command": command,
    }
    LAUNCH_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

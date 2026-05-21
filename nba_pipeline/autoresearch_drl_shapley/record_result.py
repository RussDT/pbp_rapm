#!/usr/bin/env python3
"""Append the latest run summary to results.tsv."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_TSV = HERE / "results.tsv"
LATEST_SUMMARY = HERE / "latest_summary.json"


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE.parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the latest autoresearch run.")
    parser.add_argument("--description", type=str, default="manual run")
    parser.add_argument("--status", type=str, default="keep", choices=["keep", "discard", "crash"])
    parser.add_argument("--commit", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not LATEST_SUMMARY.exists() and args.status != "crash":
        raise FileNotFoundError(f"Missing summary file: {LATEST_SUMMARY}")
    if args.status == "crash":
        summary = {
            "research_score": 0.0,
            "audit_score": 0.0,
            "research_rmse": 0.0,
            "research_brier": 0.0,
            "research_logloss": 0.0,
        }
    elif LATEST_SUMMARY.exists():
        summary = json.loads(LATEST_SUMMARY.read_text())
    else:
        summary = {
            "research_score": 0.0,
            "audit_score": 0.0,
            "research_rmse": 0.0,
            "research_brier": 0.0,
            "research_logloss": 0.0,
        }
    commit = args.commit or git_commit()
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    with RESULTS_TSV.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                timestamp,
                commit,
                f"{summary['research_score']:.6f}",
                f"{summary['audit_score']:.6f}",
                f"{summary['research_rmse']:.6f}",
                f"{summary['research_brier']:.6f}",
                f"{summary['research_logloss']:.6f}",
                args.status,
                args.description,
            ]
        )
    print(f"Recorded run to {RESULTS_TSV}")


if __name__ == "__main__":
    main()

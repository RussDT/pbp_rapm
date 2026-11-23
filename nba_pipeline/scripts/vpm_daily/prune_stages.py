#!/usr/bin/env python3
"""Delete bulky intermediate folders from old nightly VPM builds.

Only `current_refresh_{YYYYMMDD}_v1` and `_tracking_v1` are removed, only for
dates after the last hand-run build, and only once that date's publication is
recorded. The small `_sq_v1` and `_sq_board_v1` receipts are always kept.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results/estimated_skills"
LAST_HAND_RUN = "20261103"
OK = {"published_and_read_back", "skipped"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dated = sorted(
        m.group(1)
        for p in ROOT.iterdir()
        if (m := re.fullmatch(r"current_refresh_(\d{8})_v1", p.name)) and m.group(1) > LAST_HAND_RUN
    )
    for tag in dated[: -args.keep] if args.keep else dated:
        receipt = ROOT / f"current_refresh_{tag}_sq_v1/preview_publication.json"
        if not receipt.exists() or not (json.loads(receipt.read_text()).get("status") in OK or json.loads(receipt.read_text()).get("skip")):
            print(f"keep {tag}: no publication receipt")
            continue
        for suffix in ("v1", "tracking_v1"):
            target = ROOT / f"current_refresh_{tag}_{suffix}"
            if target.is_dir():
                print(("would remove " if args.dry_run else "remove ") + target.name)
                if not args.dry_run:
                    shutil.rmtree(target)


if __name__ == "__main__":
    main()

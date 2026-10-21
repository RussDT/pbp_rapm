"""Complete the authorized historical build after its annual-fit process ends."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_full_skill_history import write_json
from full_skill_history_specs import SKILLS
from prepare_full_skill_history import ROOT, sha


def finish(out, baseline_pid):
    out = Path(out)
    started = time.monotonic()
    path = out / "run_status.json"
    write_json(path, dict(status="running", stage="annual_fits_and_replay", baseline_pid=baseline_pid, supervisor_pid=os.getpid()))
    try:
        while len(list((out / "baseline").glob("*.parquet"))) < 30:
            try:
                os.kill(baseline_pid, 0)
            except ProcessLookupError:
                raise RuntimeError("Annual-fit process ended before all 30 complete replay files were written") from None
            time.sleep(20)
        lock_sha = sha(out / "run_lock.json")
        for year in range(1997, 2027):
            for spec in SKILLS:
                model = json.loads((out / "models" / str(year) / (spec.name + ".json")).read_text())
                if model["run_lock_sha256"] != lock_sha:
                    raise ValueError("Annual model source lock differs")
        stages = [
            ("chronology_and_source_audit", "audit_full_skill_history.py", []),
            ("cross_stat_testing", "research_skills_nextgame_lab.py", ["--full-history", "--workers", "6"]),
            ("coherent_exports_and_audits", "export_full_skill_history.py", []),
        ]
        for stage, script, extra in stages:
            write_json(path, dict(status="running", stage=stage, supervisor_pid=os.getpid()))
            with (out / (stage + ".log")).open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).with_name(script)), "--out", str(out), *extra], stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(path, dict(status="completed_research_build", stage="finished", supervisor_seconds=round(time.monotonic() - started, 2),
                             library_receipt_sha256=sha(out / "library_receipt.json"), selected_vpm_changed=False, protected_2027_outcomes=0))
    except Exception as error:
        write_json(path, dict(status="failed", error=str(error), supervisor_pid=os.getpid()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--baseline-pid", type=int, required=True)
    args = parser.parse_args()
    finish(args.out, args.baseline_pid)

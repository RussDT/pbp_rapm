#!/usr/bin/env python3
"""Independent readback and Parent-normal audit of coherent later entrant boards."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAPM = Path(__file__).resolve().parents[2]
VPM = RAPM.parent / "vpm"
sys.path.insert(0, str(VPM))

from vpm.harness.artifacts import sha256_file  # noqa: E402
from vpm.harness.daily_history_1997_numerics import LANES, PARENT  # noqa: E402
from vpm.harness.full_bank_history_numerics import FullBankHistoryGeometry  # noqa: E402
from vpm.harness.full_bank_history_storage import read_compact_board  # noqa: E402

BASE = VPM / "vpm/results/harness/vpm_full_feature_history_v1/compact_history_fast_v1"
LOCK = VPM / "vpm/results/harness/vpm_net_lane_clean_v1/entrant_source_lock.json"
PRESEASON = RAPM / "nba_pipeline/results/hoop_impact_bench_v1/prod_preseason_reconstruction/ratings.parquet"


def audit(root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("Preserve independent audit output")
    if shutil.disk_usage(out.parent).free < 15 * 2**30:
        raise OSError("Preserve 15 GiB reserve")
    locked = json.loads(LOCK.read_text())
    for name, digest in locked.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("Original entrant source snapshot changed: " + name)
    source_receipt = json.loads((root / "receipt.json").read_text())
    if source_receipt["status"] != "produced_bounded_annual_entrant_boards_pending_independent_history_audit":
        raise ValueError("Wrong annual source status")
    if len(source_receipt["cuts"]) != 9:
        raise ValueError("Incomplete annual census")
    records = []
    for row in source_receipt["cuts"]:
        day = row["date"]
        folder = root / day
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, digest in {**manifest["input_hashes"], **manifest["output_hashes"]}.items():
            if sha256_file(Path(name)) != digest:
                raise ValueError("Annual board input/output pin changed: " + name)
        frame = pd.read_parquet(folder / "ratings.parquet")
        with np.load(folder / "arrays.npz") as saved:
            arrays = {name: saved[name] for name in saved.files}
        old = read_compact_board(BASE / "cuts" / day)
        engine = FullBankHistoryGeometry.resume(BASE / "checkpoints" / day)
        ids = old.arrays["ids"]
        if not np.array_equal(ids, frame.player_id.to_numpy()) or not np.array_equal(ids, arrays["ids"]):
            raise ValueError("Annual player identity or ordering changed")
        n = len(ids)
        w = frame.effective_background_weight.to_numpy(float)
        expected = arrays["own_lanes"] + w[:, None, None] * (arrays["background_lanes"] - arrays["own_lanes"])
        prior_gap = float(np.max(abs(arrays["combined_lanes"] - expected)))
        if prior_gap > 1e-12 or not np.array_equal(arrays["own_lanes"], old.arrays["own_lanes"]):
            raise ValueError("Own or combined 11-head prior differs")
        if (not np.array_equal(arrays["combined_lanes"][w == 0], arrays["own_lanes"][w == 0])
                or not np.array_equal(arrays["mid_sq_delta"], old.arrays["mid_sq_delta"])
                or not np.array_equal(arrays["three_sq_delta"], old.arrays["three_sq_delta"])):
            raise ValueError("Zero-weight or frozen SQ control changed")
        combined = arrays["combined_lanes"]
        prior = np.vstack([combined[:, 0] / 100, -combined[:, 1] / 100])
        order = np.r_[[2 * engine.index[int(p)] for p in ids], [2 * engine.index[int(p)] + 1 for p in ids]]
        internal = np.empty_like(prior)
        internal[order] = prior
        g = engine.response_geometry(pd.Timestamp(day))
        penalty = np.tile(PARENT, n)
        direct, normal_error = engine._solve(g["actual_rhs"] + penalty * internal[:, :len(LANES)].sum(axis=1), PARENT)
        direct = direct[order] + arrays["mid_sq_delta"] + arrays["three_sq_delta"]
        native_gap = float(np.max(abs(direct - arrays["raw_parent_native"])))
        net_gap = float(np.max(abs(arrays["raw_native_lanes"][:, :len(LANES)].sum(axis=1) - direct)))
        total_gap = float(np.max(abs(frame.ovpm.to_numpy() + frame.dvpm.to_numpy() - frame.vpm.to_numpy())))
        if max(native_gap, net_gap, total_gap, normal_error) > 1e-9:
            raise ValueError("Annual Parent/Net/normal equation readback failed")
        if int((w > 0).sum()) != row["positive_effective_weights"]:
            raise ValueError("Annual entrant census changed")
        records.append(dict(date=day, rows=n, positive_weights=int((w > 0).sum()),
            max_prior_gap=prior_gap, max_native_gap=native_gap, max_net_gap=net_gap,
            max_total_gap=total_gap, normal_equation_error=float(normal_error),
            board_manifest_sha256=sha256_file(folder / "manifest.json")))
        print(json.dumps(records[-1]), flush=True)
    current = pd.read_parquet(root / "2026-10-01/ratings.parquet").set_index("player_id")
    original = pd.read_parquet(PRESEASON).set_index("player_id")
    if not current.index.equals(original.index):
        raise ValueError("October production reconstruction identities differ")
    production_gaps = {name: float(np.max(abs(current[name].to_numpy(float) - original[name].to_numpy(float))))
                       for name in ["ovpm", "dvpm", "vpm"]}
    # The lone retained 1996-drafted, low-exposure inactive player is now
    # identified throughout history. The original current reconstruction left
    # him unidentified, so this is a diagnostic, not an exact parity gate.
    out.mkdir(parents=True)
    result = dict(status="passed_independent_coherent_late_annual_entrant_audit", cuts=records,
        production_reconstruction_max_gaps=production_gaps,
        frozen_source_lock_sha256=sha256_file(LOCK), annual_receipt_sha256=sha256_file(root / "receipt.json"),
        preseason_reconstruction_sha256=sha256_file(PRESEASON),
        independent_auditor_sha256=sha256_file(Path(__file__)),
        daily_history_complete=False, historical_reader_changed=False)
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    audit(args.root.resolve(), args.out.resolve())

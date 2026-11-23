#!/usr/bin/env python3
"""Independent readback and Parent-normal audit of 1997–2017 entrant boards."""
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
EARLY_AUDIT = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_models_audit_v1/receipt.json"


def audit(root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("Preserve independent audit output")
    if shutil.disk_usage(out.parent).free < 15 * 2**30:
        raise OSError("Preserve 15 GiB reserve")
    model_audit = json.loads(EARLY_AUDIT.read_text())
    if model_audit["status"] != "passed_independent_18_early_entrant_model_normal_audit":
        raise ValueError("Early entrant model audit absent")
    source_receipt = json.loads((root / "receipt.json").read_text())
    if source_receipt["status"] != "produced_early_annual_entrant_boards_pending_independent_history_audit":
        raise ValueError("Wrong annual source status")
    if len(source_receipt["cuts"]) != 21:
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
        model = json.loads(Path(row["model"]).read_text())
        if sha256_file(Path(row["model"])) != row["model_sha256"]:
            raise ValueError("Early model path/sha changed")
        if row["entrant_model_backcast"] != (pd.Timestamp(day) < pd.Timestamp("2000-10-01")):
            raise ValueError("Early entrant backcast flag differs")
        if row["entrant_model_backcast"] and model["fit_cut"] != "2000-10-01":
            raise ValueError("Wrong first release for early backcast")
        if not row["entrant_model_backcast"] and pd.Timestamp(model["fit_end"]) >= pd.Timestamp(day):
            raise ValueError("Early model not trained before cut")
        records.append(dict(date=day, rows=n, positive_weights=int((w > 0).sum()),
            max_prior_gap=prior_gap, max_native_gap=native_gap, max_net_gap=net_gap,
            max_total_gap=total_gap, normal_equation_error=float(normal_error),
            board_manifest_sha256=sha256_file(folder / "manifest.json")))
        print(json.dumps(records[-1]), flush=True)
    out.mkdir(parents=True)
    result = dict(status="passed_independent_21_early_annual_entrant_audit", cuts=records,
        early_model_audit_sha256=sha256_file(EARLY_AUDIT), annual_receipt_sha256=sha256_file(root / "receipt.json"),
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

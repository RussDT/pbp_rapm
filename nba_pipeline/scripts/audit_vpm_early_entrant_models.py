#!/usr/bin/env python3
"""Independently audit saved chronological early entrant normal equations."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hashlib import sha256


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("Preserve prior independent early audit")
    if shutil.disk_usage(out.parent).free < 15 * 2**30:
        raise OSError("Preserve 15 GiB reserve")
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt["status"] != "early_entrant_models_pending_independent_audit" or len(receipt["releases"]) != 18:
        raise ValueError("Incomplete early entrant release census")
    for name, value in receipt["source_hashes"].items():
        if digest(Path(name)) != value:
            raise ValueError("Early fit source changed: " + name)
    records = []
    for year, record in zip(range(2000, 2018), receipt["releases"]):
        model_path = root / f"{year}.json"
        normal_path = root / f"{year}_normal.npz"
        if record["vintage"] != year or digest(model_path) != record["model_sha256"] or digest(normal_path) != record["normal_sha256"]:
            raise ValueError("Frozen early model artifacts differ")
        model = json.loads(model_path.read_text())
        if model["vintage"] != year or model["fit_cut"] != f"{year}-10-01" or model["training_seasons"] != list(range(1997, year + 1)):
            raise ValueError("Early model release/training chronology differs")
        if pd.Timestamp(model["fit_end"]) >= pd.Timestamp(model["fit_cut"]) or model["protected_2027_inputs"] != 0:
            raise ValueError("Early model fit looks beyond release")
        if model["components"] != ["rim_freq", "rim_conversion", "mid_freq", "mid_conversion", "three_freq", "three_conversion", "ft", "tov", "second_chance", "mid_reference_on_rim", "mid_reference_on_three"]:
            raise ValueError("Early model 11-head target differs")
        with np.load(normal_path) as saved:
            info, rhs, penalty = (saved[k] for k in ["information", "rhs", "penalty"])
        co = np.asarray(model["coefficients"], float)
        parent = np.asarray(model["parent_coefficients"], float)
        p = len(model["scaler"]["columns"])
        if co.shape != (2 * p, 11) or parent.shape != (2 * p,) or info.shape != (2 * p, 2 * p) or rhs.shape != (2 * p, 11):
            raise ValueError("Early model geometry differs")
        if not np.allclose(info, info.T, atol=1e-9, rtol=0) or np.linalg.eigvalsh(info + np.diag(penalty)).min() <= 1e-9:
            raise ValueError("Early model normal matrix not symmetric/identified")
        if not np.array_equal(penalty, np.array([0 if i in [0, p] else 10 for i in range(2 * p)], float)):
            raise ValueError("Early model penalty changed")
        normal = info + np.diag(penalty)
        error = float(np.max(abs(normal @ co - rhs)))
        parent_error = float(np.max(abs(normal @ parent - rhs[:, :9].sum(axis=1))))
        sum_gap = float(np.max(abs(co[:, :9].sum(axis=1) - parent)))
        if max(error, parent_error, sum_gap) > 1e-9:
            raise ValueError("Early model independent normal/Parent closure failed")
        records.append(dict(vintage=year, features=p, training_rows=model["training_rows"],
            training_games=model["training_games"], normal_error=error,
            parent_error=parent_error, coefficient_sum_gap=sum_gap,
            model_sha256=digest(model_path)))
    out.mkdir(parents=True)
    result = dict(status="passed_independent_18_early_entrant_model_normal_audit", releases=records,
        source_receipt_sha256=digest(root / "receipt.json"), auditor_sha256=digest(Path(__file__)),
        source_publication_vintages_certified=False, daily_history_complete=False)
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.out.resolve())
    print(json.dumps(dict(status=result["status"], releases=len(result["releases"]),
        max_normal_error=max(r["normal_error"] for r in result["releases"]))))

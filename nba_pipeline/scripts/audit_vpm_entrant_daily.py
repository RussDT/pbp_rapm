#!/usr/bin/env python3
"""Independent full-census and 11 fresh-normal audit of entrant daily history."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAPM = Path(__file__).resolve().parents[2]
VPM = RAPM.parent / "vpm"
sys.path.insert(0, str(VPM))

from vpm.harness.daily_history_1997_numerics import LANES, PARENT  # noqa: E402
from vpm.harness.full_bank_history_numerics import FullBankHistoryGeometry  # noqa: E402
from vpm.harness.full_bank_history_storage import read_compact_board  # noqa: E402

BASE = VPM / "vpm/results/harness/vpm_full_feature_history_v1/compact_history_fast_v1"
CALENDAR = VPM / "vpm/results/harness/vpm_daily_1997_preflight_v3/calendar.parquet"
V74_REGISTRATION = VPM / "vpm/results/harness/vpm_daily_1997_v1/registration.json"
POINTER = VPM / "vpm/configs/current_model.json"
POINTER_SHA = "08e5882c463d2ac10eccc8161afd92f14650afd8585634b4008227fde75d05d5"
KNOWN_HISTORICAL_CODE_DRIFT = {
    str(RAPM / "nba_pipeline/scripts/process_rapm_blocks/process_russell_shotquality.py"): (
        "176d192987c3806c4ade7676c6c0db3ac0e63bb46a1a288e179d1f113ab6be6f",
        "88a746f28abec7769d9258c2585745dca1a460cd0359a9b1940622af189cb0ec",
    ),
    str(VPM / "vpm/harness/cli.py"): (
        "19348d0b5398c3a2e8fab91b8e1a04fedb0da33d6a022a770451d73ef2bb7668",
        "0068994d4c0f68e57338847e2e0ba9892e4bf9b29fbbdb71e53be8a911491440",
    ),
}
EARLY_MODELS = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_models_2000_2017_v2"
LATE_MODELS = VPM / "vpm/results/harness/vpm_net_lane_clean_v1/entrant_models"
EARLY_EXPOSURE = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_career_exposure_v2/player_game_exposure.parquet"
LATE_EXPOSURE = RAPM / "nba_pipeline/results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet"
SAMPLES = ["1997-10-01", "2000-10-01", "2003-10-01", "2006-10-01", "2009-10-01",
    "2012-10-01", "2015-10-01", "2018-10-01", "2020-11-01", "2023-10-01", "2026-10-01"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample(root: Path, day: str) -> dict:
    folder = root / "cuts" / day
    frame = pd.read_parquet(folder / "ratings.parquet")
    with np.load(folder / "arrays.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    old = read_compact_board(BASE / "cuts" / day)
    engine = FullBankHistoryGeometry.resume(root / "checkpoints" / day)
    ids = arrays["ids"]
    n = len(ids)
    if not np.array_equal(ids, old.arrays["ids"]) or not np.array_equal(ids, frame.player_id.to_numpy()):
        raise ValueError("Sampled daily player order differs")
    if not np.array_equal(arrays["own_lanes"], old.arrays["own_lanes"]):
        raise ValueError("Sampled frozen own prior changed")
    for zone in ["mid", "three"]:
        if not np.array_equal(arrays[zone + "_sq_delta"], old.arrays[zone + "_sq_delta"]):
            raise ValueError("Sampled frozen SQ correction changed")
    w = frame.effective_background_weight.to_numpy(float)
    combined = arrays["own_lanes"] + w[:, None, None] * (arrays["background_lanes"] - arrays["own_lanes"])
    prior_gap = float(np.max(abs(combined - arrays["combined_lanes"])))
    if prior_gap > 1e-12:
        raise ValueError("Sampled combined prior algebra failed")
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
    if max(prior_gap, native_gap, net_gap, normal_error) > 1e-9:
        raise ValueError("Fresh sampled daily normal or Net closure failed")
    return dict(date=day, players=n, prior_gap=prior_gap, native_parent_gap=native_gap,
        native_net_gap=net_gap, normal_equation_error=float(normal_error),
        positive_weights=int((w > 0).sum()), unidentified_weights=int(frame.background_weight_identified.eq(False).sum()))


def audit(root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("Preserve prior independent daily audit")
    if shutil.disk_usage(out.parent).free < 15 * 2**30:
        raise OSError("Preserve 15 GiB disk reserve")
    final = json.loads((root / "receipt.json").read_text())
    if final["status"] != "produced_full_entrant_daily_history_pending_independent_audit" or final["completed_dates"] != 6108:
        raise ValueError("Full completed producer required")
    registration = json.loads((root / "registration.json").read_text())
    if final["registration_sha256"] != sha(root / "registration.json") or registration["schema"] != "full_feature_entrant_daily_research_replay_v1":
        raise ValueError("Daily registration differs")
    for name, digest in registration["source_pins"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Full daily registered source changed: " + name)
    native_registration = json.loads(V74_REGISTRATION.read_text())
    if native_registration["dates"] != 6108:
        raise ValueError("Native daily source registration census changed")
    historical_code_drift = []
    archived_shotquality = subprocess.check_output([
        "git", "-C", str(RAPM), "show",
        "10e6d615fd44b176ca2ab918695944e3d0ed17d9:nba_pipeline/scripts/process_rapm_blocks/process_russell_shotquality.py",
    ])
    if hashlib.sha256(archived_shotquality).hexdigest() != KNOWN_HISTORICAL_CODE_DRIFT[str(RAPM / "nba_pipeline/scripts/process_rapm_blocks/process_russell_shotquality.py")][0]:
        raise ValueError("Archived historical shot quality code differs")
    for name, digest in native_registration["input_hashes"].items():
        observed = sha(Path(name))
        if observed != digest:
            if KNOWN_HISTORICAL_CODE_DRIFT.get(name) != (digest, observed):
                raise ValueError("Native source-transitive input changed: " + name)
            archived = name.endswith("process_russell_shotquality.py")
            historical_code_drift.append(dict(path=name, registered_sha256=digest,
                present_sha256=observed, historical_bytes_verified=archived,
                historical_evidence=("git:10e6d615fd44b176ca2ab918695944e3d0ed17d9" if archived else None)))
    if {item["path"] for item in historical_code_drift} != set(KNOWN_HISTORICAL_CODE_DRIFT):
        raise ValueError("Native historical code drift census changed")
    if sha(POINTER) != POINTER_SHA:
        raise ValueError("Selected VPM pointer changed")
    calendar = pd.read_parquet(CALENDAR)
    models = []
    for model_root in [EARLY_MODELS, LATE_MODELS]:
        for path in model_root.glob("20??.json"):
            item = json.loads(path.read_text())
            core = item.get("parent", item)
            models.append((core["fit_cut"], core["fit_end"], sha(path)))
    if len(models) != 26:
        raise ValueError("Early/later entrant model census differs")
    models.sort()
    early_exposure = pd.read_parquet(EARLY_EXPOSURE, columns=["player_id", "game_date", "possessions"])
    late_exposure = pd.read_parquet(LATE_EXPOSURE, columns=["player_id", "game_date", "off_poss", "def_poss"])
    late_exposure["possessions"] = (late_exposure.off_poss + late_exposure.def_poss) / 2
    ledger = pd.concat([early_exposure[["player_id", "game_date", "possessions"]],
        late_exposure[["player_id", "game_date", "possessions"]]], ignore_index=True)
    ledger.game_date = pd.to_datetime(ledger.game_date, utc=True).dt.tz_localize(None)
    ledger = ledger.sort_values("game_date", kind="stable").reset_index(drop=True)
    if ledger.game_date.isna().any() or (ledger.possessions < 0).any():
        raise ValueError("Independent dated career ledger invalid")
    exposure_dates = ledger.game_date.to_numpy("datetime64[ns]")
    exposure_ids = ledger.player_id.to_numpy(np.int64)
    exposure_values = ledger.possessions.to_numpy(float)
    cumulative_exposure = np.zeros(int(exposure_ids.max()) + 1)
    exposure_position = 0
    dirs = sorted((root / "cuts").iterdir())
    expected = set(calendar.estimate_date.dt.strftime("%Y-%m-%d"))
    if len(dirs) != 6108 or {p.name for p in dirs} != expected:
        raise ValueError("Full daily entrant date census differs")
    all_rows = 0
    all_positive = 0
    max_normal = 0.
    max_net = 0.
    max_total = 0.
    max_prior = 0.
    max_exposure = 0.
    max_weight = 0.
    backcast_dates = 0
    annual_checks = 0
    max_annual_gap = 0.
    for index, row in enumerate(calendar.itertuples(index=False)):
        day = str(pd.Timestamp(row.estimate_date).date())
        folder = root / "cuts" / day
        new_position = int(np.searchsorted(exposure_dates, np.datetime64(day), side="left"))
        if new_position > exposure_position:
            np.add.at(cumulative_exposure,
                exposure_ids[exposure_position:new_position], exposure_values[exposure_position:new_position])
            exposure_position = new_position
        if index == 0:
            cold = json.loads((folder / "cold.json").read_text())
            if cold != dict(status="unestimated_no_earlier_admitted_lineups", date=day):
                raise ValueError("Cold first date differs")
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        metadata = manifest["source_metadata"]
        daily = json.loads((folder / "daily_replay.json").read_text())
        if (metadata["estimate_date"] != day or metadata["query_season"] != int(row.query_season)
                or daily["date"] != day or daily["query_season"] != int(row.query_season)
                or metadata["prior_regime"] != "full_feature_own_with_frozen_entrant_background"
                or metadata["recipe"] != "full_feature_historical_entrant_daily_replay_v1"
                or manifest["rows"] != daily["players"] or manifest["columns"] != 171
                or manifest["arrays"] != 18
                or metadata["normal_equation_error"] != daily["normal_equation_error"]
                or metadata["maximum_native_lane_sum_gap"] != daily["native_net_gap"]):
            raise ValueError("Daily date/shape/season/prior identity differs")
        if daily["entrant_model_backcast"] != (day < "2000-10-01"):
            raise ValueError("Daily early entrant backcast flag differs")
        if row.annual_checkpoint:
            if daily["annual_gap"] is None or not np.isfinite(daily["annual_gap"]) or daily["annual_gap"] > 1e-8:
                raise ValueError("Daily/annual entrant checkpoint parity failed")
            annual_checks += 1
            max_annual_gap = max(max_annual_gap, daily["annual_gap"])
        elif daily["annual_gap"] is not None:
            raise ValueError("Unregistered daily/annual parity assertion")
        model_path = next(Path(name) for name, digest in manifest["input_hashes"].items() if digest == daily["model_sha256"])
        model = json.loads(model_path.read_text())
        core = model.get("parent", model)
        if core["fit_cut"] != daily["model_fit_cut"] or metadata["entrant_model_sha256"] != daily["model_sha256"]:
            raise ValueError("Daily selected entrant model differs")
        if day >= "2000-10-01" and not (core["fit_cut"] <= day and core["fit_end"] < day):
            raise ValueError("Daily entrant model is not past-fitted")
        if day < "2000-10-01" and core["fit_cut"] != "2000-10-01":
            raise ValueError("Daily entrant backcast is not first model")
        eligible = [item for item in models if item[0] <= day and item[1] < day]
        expected_model = eligible[-1][2] if eligible else models[0][2]
        if daily["model_sha256"] != expected_model:
            raise ValueError("Daily entrant model is not the newest eligible release")
        for name, digest in {**manifest["input_hashes"], **manifest["output_hashes"]}.items():
            if sha(Path(name)) != digest:
                raise ValueError("Daily source/output pin changed: " + name)
        if pq.read_metadata(folder / "ratings.parquet").num_columns != 171:
            raise ValueError("Daily persisted rating column census differs")
        with zipfile.ZipFile(folder / "arrays.npz") as archive:
            if len(archive.namelist()) != 18 or len(set(archive.namelist())) != 18:
                raise ValueError("Daily persisted array census differs")
        old_manifest = json.loads((BASE / "cuts" / day / "manifest.json").read_text())
        if old_manifest["rows"] != manifest["rows"]:
            raise ValueError("Daily own/entrant native player count differs")
        frame = pd.read_parquet(folder / "ratings.parquet", columns=["ovpm", "dvpm", "vpm",
            "player_id", "career_nba_possessions", "entrant_career_exposure_complete", "background_weight",
            "own_prior_off", "own_prior_def", "combined_prior_off", "combined_prior_def",
            "background_prior_off", "background_prior_def", "effective_background_weight",
            "background_weight_identified"])
        if len(frame) != manifest["rows"]:
            raise ValueError("Daily persisted rating row count differs")
        total_gap = float(np.max(abs(frame.ovpm.to_numpy(float) + frame.dvpm.to_numpy(float) - frame.vpm.to_numpy(float))))
        w = frame.effective_background_weight.to_numpy(float)
        ids = frame.player_id.to_numpy(np.int64)
        if ids.min() < 0:
            raise ValueError("Negative daily player identity")
        if ids.max() >= len(cumulative_exposure):
            cumulative_exposure = np.pad(cumulative_exposure,
                (0, int(ids.max()) + 1 - len(cumulative_exposure)))
        expected_exposure = cumulative_exposure[ids]
        saved_exposure = frame.career_nba_possessions.to_numpy(float)
        if not np.isfinite(saved_exposure).all():
            raise ValueError("Daily career exposure is nonfinite")
        exposure_gap = float(np.max(abs(expected_exposure - saved_exposure)))
        complete = frame.entrant_career_exposure_complete.to_numpy(bool)
        identified = complete | (expected_exposure >= 2000)
        expected_weight = np.where(identified, np.maximum(1 - expected_exposure / 2000, 0), 0)
        weight_gap = float(np.max(abs(expected_weight - w)))
        reported = frame.background_weight.to_numpy(float)
        if (not np.array_equal(identified, frame.background_weight_identified.to_numpy(bool))
                or not np.array_equal(np.isnan(reported), ~identified)
                or np.max(abs(reported[identified] - expected_weight[identified]), initial=0) > 1e-10
                or exposure_gap > 1e-8 or weight_gap > 1e-10):
            raise ValueError("Daily predate career exposure or entrant weight differs")
        if (w < 0).any() or (w > 1).any() or not np.isfinite(w).all():
            raise ValueError("Daily experience weight invalid")
        prior_off_gap = frame.combined_prior_off - (frame.own_prior_off + w * (frame.background_prior_off - frame.own_prior_off))
        prior_def_gap = frame.combined_prior_def - (frame.own_prior_def + w * (frame.background_prior_def - frame.own_prior_def))
        prior_gap = float(max(abs(prior_off_gap).max(), abs(prior_def_gap).max()))
        if max(total_gap, prior_gap) > 1e-9 or int((w > 0).sum()) != daily["positive_weights"]:
            raise ValueError("Daily persisted Total/prior/weight closure failed")
        if int(frame.background_weight_identified.eq(False).sum()) != daily["unidentified_weights"]:
            raise ValueError("Daily unidentified weight count differs")
        max_total = max(max_total, total_gap)
        max_prior = max(max_prior, prior_gap)
        max_exposure = max(max_exposure, exposure_gap)
        max_weight = max(max_weight, weight_gap)
        max_normal = max(max_normal, daily["normal_equation_error"])
        max_net = max(max_net, daily["native_net_gap"])
        all_rows += len(frame)
        all_positive += daily["positive_weights"]
        backcast_dates += int(daily["entrant_model_backcast"])
        if index % 250 == 0:
            print(json.dumps(dict(stage="full_daily_audit_progress", dates=index + 1, total=6108, rows=all_rows)), flush=True)
    if max(max_total, max_prior, max_normal, max_net) > 1e-9:
        raise ValueError("Full daily numerical summary exceeds guard")
    base_manifest = json.loads((BASE / "manifest.json").read_text())
    if base_manifest["dates"] != 6108 or base_manifest["rows"] != all_rows or annual_checks != 30:
        raise ValueError("Full entrant history row census differs from own source")
    sampled = [sample(root, day) for day in SAMPLES]
    out.mkdir(parents=True)
    result = dict(status="passed_independent_6108_date_entrant_numerical_audit_with_qualified_base_provenance",
        dates=6108, nonempty_dates=6107, rows=all_rows, positive_player_date_weights=all_positive,
        entrant_backcast_dates=backcast_dates, maximum_total_gap=max_total,
        maximum_prior_gap=max_prior, annual_endpoint_checks=annual_checks,
        maximum_independent_career_exposure_gap=max_exposure,
        maximum_independent_entrant_weight_gap=max_weight,
        maximum_annual_endpoint_gap=max_annual_gap, maximum_producer_normal_error=max_normal,
        maximum_producer_net_gap=max_net, fresh_normal_samples=sampled,
        registration_sha256=sha(root / "registration.json"), producer_receipt_sha256=sha(root / "receipt.json"),
        auditor_sha256=sha(Path(__file__)), native_v74_registered_inputs=len(native_registration["input_hashes"]),
        native_v74_exact_present_inputs=len(native_registration["input_hashes"]) - len(historical_code_drift),
        native_v74_historical_code_drift=historical_code_drift, full_feature_base_audit_pending=True,
        publication_vintage_certified=False, reader_changed=False)
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.out.resolve())
    print(json.dumps({k: result[k] for k in ["status", "dates", "rows", "maximum_producer_normal_error"]}), flush=True)

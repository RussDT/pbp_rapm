#!/usr/bin/env python3
"""Re-solve 1997–2017 full-feature annual boards with audited early entrant maps.

This bounded research stage never refits a mapper or rewrites an own-only board.
It uses the original Parent/native geometry and the production entrant operator.
"""
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
from vpm.harness.current_entrant_prior import apply_entrant_background, save_board  # noqa: E402
from vpm.harness.entrant_priors import feature_packet, load_sources, transformed_features  # noqa: E402
from vpm.harness.full_bank_history_numerics import FullBankHistoryGeometry  # noqa: E402
from vpm.harness.full_bank_history_storage import read_compact_board  # noqa: E402
from vpm.harness.monthly_vpm import entrant_weight  # noqa: E402
from vpm.harness.quarterly_history import LANES  # noqa: E402
from fit_vpm_early_entrant_models_v2 import career_packet  # noqa: E402

BASE = VPM / "vpm/results/harness/vpm_full_feature_history_v1/compact_history_fast_v1"
MODELS = VPM / "vpm/results/harness/vpm_net_lane_clean_v1/entrant_models"
EARLY_MODELS = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_models_2000_2017_v2"
EARLY_AUDIT = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_models_audit_v1/receipt.json"
SOURCES = VPM / "vpm/data/entrants"
EARLY = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_career_exposure_v2"
LATER = RAPM / "nba_pipeline/results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet"
NAMES = RAPM / "nba_pipeline/results/hoop_impact_bench_v1/prod_preseason_reconstruction/entrant_background_inputs/entrant_canonical_names.csv"


def choose_model(cut: pd.Timestamp) -> tuple[Path, dict, bool]:
    eligible = []
    for path in sorted(EARLY_MODELS.glob("20??.json")):
        model = json.loads(path.read_text())
        if pd.Timestamp(model["fit_cut"]) <= cut and pd.Timestamp(model["fit_end"]) < cut:
            eligible.append((pd.Timestamp(model["fit_cut"]), path, model))
    if not eligible:
        if cut >= pd.Timestamp("2000-10-01"):
            raise ValueError(f"No past-fitted early entrant model at {cut.date()}")
        path = EARLY_MODELS / "2000.json"
        return path, json.loads(path.read_text()), True
    _, path, model = max(eligible, key=lambda x: (x[0], x[1].name))
    return path, model, False


def legacy_career_packet(ids: np.ndarray, cut: pd.Timestamp, sources, early, later, first) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    past_early = early.loc[early.game_date.lt(cut)]
    past_later = later.loc[later.game_date.lt(cut)]
    if past_later.duplicated(["player_id", "game_id"]).any():
        raise ValueError("Duplicate later player-game career exposure")
    a = past_early[["player_id", "possessions", "game_date"]].rename(columns={"game_date": "last_game"})
    b = past_later[["player_id", "game_date", "off_poss", "def_poss"]].copy()
    b["possessions"] = (b.off_poss + b.def_poss) / 2
    b = b[["player_id", "possessions", "game_date"]].rename(columns={"game_date": "last_game"})
    history = pd.concat([a, b], ignore_index=True)
    packet = feature_packet(sources, history, ids, cut, first)
    exposure = packet.nba_possessions.to_numpy(float)
    if (exposure < 0).any() or not np.isfinite(exposure).all():
        raise ValueError("Invalid dated career exposure")
    first_year = first.reindex(ids).to_numpy(float)
    identified = (first_year >= 1998) | (exposure >= 2000)
    # The first archive year contains undistinguished veterans and rookies.
    # Suppress a possible false positive instead of treating missing pre-1997
    # games as zero career possessions.
    raw_weight = entrant_weight(exposure, 2000.)
    effective = np.where(identified, raw_weight, 0.)
    reported = np.where(identified, raw_weight, np.nan)
    return packet, effective, reported, identified


def build(cuts: list[str], out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Preserve previous annual entrant root: {out}")
    if shutil.disk_usage(out.parent).free < 15 * 2**30:
        raise OSError("Preserve 15 GiB disk reserve")
    early_path = EARLY / "player_game_exposure.parquet"
    early_receipt = json.loads((EARLY / "receipt.json").read_text())
    if early_receipt["status"] != "passed_exact_annual_reconciliation" or early_receipt["ledger_sha256"] != sha256_file(early_path):
        raise ValueError("Unverified exact early career source")
    early = pd.read_parquet(early_path)
    later = pd.read_parquet(LATER)
    early.game_date = pd.to_datetime(early.game_date, utc=True).dt.tz_localize(None)
    later.game_date = pd.to_datetime(later.game_date, utc=True).dt.tz_localize(None)
    if early.game_date.max() >= later.game_date.min() or later.season.max() > 2026:
        raise ValueError("Career sources overlap or include protected season")
    first = pd.read_parquet(SOURCES / "nba_history.parquet").groupby("player_id").season.min()
    sources = load_sources(SOURCES, NAMES)
    audit = json.loads(EARLY_AUDIT.read_text())
    if audit["status"] != "passed_independent_18_early_entrant_model_normal_audit":
        raise ValueError("Early entrant models lack independent normal audit")
    common = {str(p.resolve()): sha256_file(p) for p in [Path(__file__), Path(__file__).with_name("fit_vpm_early_entrant_models_v2.py"),
        early_path, EARLY / "receipt.json", LATER, NAMES, SOURCES / "nba_history.parquet", EARLY_AUDIT, EARLY_MODELS / "receipt.json"]}
    out.mkdir(parents=True)
    records = []
    for day in cuts:
        cut = pd.Timestamp(day)
        annual = f"{cut.year}-10-01"
        if cut.year < 1997 or cut.year > 2017 or day != annual:
            raise ValueError("This stage permits annual report cuts, 1997–2017")
        source = BASE / "cuts" / day
        checkpoint = BASE / "checkpoints" / day
        own = read_compact_board(source)
        engine = FullBankHistoryGeometry.resume(checkpoint)
        if own.metadata["estimate_date"] != day or engine.anchor >= cut:
            raise ValueError("Stored board/checkpoint chronology differs")
        ids = np.asarray(own.arrays["ids"], dtype=np.int64)
        path, model, backcast = choose_model(cut)
        packet, weights, reported, identified = career_packet(ids, cut, sources, early, later, first)
        draft_year = sources.draft.reindex(ids).draft_year.to_numpy(float)
        first_year = first.reindex(ids).to_numpy(float)
        career_complete = (first_year >= 1998) | ((first_year == 1997) & (draft_year == 1996))
        f = transformed_features(packet, model["scaler"])
        p = f.shape[1]
        co = np.asarray(model["coefficients"], float)
        if model["components"] != own.intercepts["components"] or co.shape != (2*p, 11):
            raise ValueError("Frozen 11-head entrant model differs from native board")
        background = np.stack([f @ co[:p], f @ co[p:]], axis=1)
        parent_coef = np.asarray(model["parent_coefficients"], float)
        parent_background = np.column_stack([f @ parent_coef[:p], f @ parent_coef[p:]])
        if np.max(np.abs(background[:, :, :len(LANES)].sum(axis=2) - parent_background)) > 1e-7:
            raise ValueError("Frozen entrant Parent and 9 native Net coefficients differ")
        board = apply_entrant_background(engine, own, background, weights,
            reported_weights=reported, identified=identified,
            career_possessions=packet.nba_possessions.to_numpy(float),
            career_complete=career_complete,
            metadata={"entrant_model_path": str(path.resolve()), "entrant_model_sha256": sha256_file(path),
                "entrant_model_fit_cut": model["fit_cut"], "entrant_model_fit_end": model["fit_end"],
                "historical_entrant_model_backcast": backcast,
                "historical_entrant_stage": "bounded_annual_1997_2017_research"})
        board.metadata["recipe"] = "full_feature_historical_early_entrant_annual_v1"
        pins = dict(common)
        pins[str(path.resolve())] = sha256_file(path)
        for folder in [source, checkpoint]:
            for file in folder.iterdir():
                if file.is_file():
                    pins[str(file.resolve())] = sha256_file(file)
        target = out / day
        saved = save_board(board, target, input_hashes=pins)
        own_v = own.ratings.vpm.to_numpy(float)
        records.append({"date": day, "players": len(ids), "positive_effective_weights": int((weights > 0).sum()),
            "unidentified_weights": int((~identified).sum()), "model": str(path.resolve()),
            "model_sha256": sha256_file(path), "entrant_model_backcast": backcast,
            "max_abs_vpm_change": float(np.max(abs(board.ratings.vpm.to_numpy(float) - own_v))),
            "normal_equation_error": board.metadata["normal_equation_error"], "net_gap": board.metadata["maximum_native_lane_sum_gap"],
            "board_manifest_sha256": sha256_file(target / "manifest.json"), "rows": saved["rows"]})
        print(json.dumps(records[-1]), flush=True)
    result = {"status": "produced_early_annual_entrant_boards_pending_independent_history_audit", "cuts": records,
        "daily_history_completed": False, "full_feature_history_admitted": False, "reader_changed": False,
        "prior_regime": "chronological_early_entrant_background_with_explicit_1997_1999_backcast", "input_hashes": common}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cut", action="append", required=True)
    args = parser.parse_args()
    build(args.cut, args.out.resolve())

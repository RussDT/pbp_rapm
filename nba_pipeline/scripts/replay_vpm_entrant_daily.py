#!/usr/bin/env python3
"""Checkpointed full-feature daily VPM replay with chronological entrant priors.

This writes a separate research history. It never changes an own-only source,
selected mapper, public reader, or production board. The first 2000 entrant map
is an explicit 1997–99 backcast, as the existing own mapper already is.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAPM = Path(__file__).resolve().parents[2]
VPM = RAPM.parent / "vpm"
sys.path.insert(0, str(VPM))
sys.path.insert(0, str(RAPM / "nba_pipeline/scripts"))

from vpm.harness.annual_history_1997 import adapter  # noqa: E402
from vpm.harness.artifacts import sha256_file  # noqa: E402
from vpm.harness.current_entrant_prior import apply_entrant_background, save_board  # noqa: E402
from vpm.harness.daily_history_1997 import exact_native_calendar  # noqa: E402
from vpm.harness.entrant_priors import feature_packet, load_sources, transformed_features  # noqa: E402
from vpm.harness.full_bank_history_numerics import FullBankHistoryGeometry  # noqa: E402
from vpm.harness.full_bank_history_storage import read_compact_board  # noqa: E402
from vpm.harness.monthly_vpm import entrant_weight  # noqa: E402
from vpm.harness.paths import discover_paths  # noqa: E402
from vpm.harness.quarterly_history import LANES  # noqa: E402
from build_vpm_entrant_annual_cuts import BASE, EARLY, LATER, NAMES, SOURCES  # noqa: E402

TARGET = RAPM / "nba_pipeline/results/vpm_historical_targets_1997_dated_v2"
CLEAN_SQ = RAPM / "nba_pipeline/results/monthly_vpm_targets_clean_sq_v2"
CALENDAR = VPM / "vpm/results/harness/vpm_daily_1997_preflight_v3/calendar.parquet"
TARGET_GAMES = CALENDAR.parent / "target_games.parquet"
V74_REGISTRATION = VPM / "vpm/results/harness/vpm_daily_1997_v1/registration.json"
V74_COMPLETION = VPM / "vpm/results/harness/vpm_daily_1997_v1/independent_completion.json"
EARLY_MODELS = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_models_2000_2017_v2"
EARLY_MODEL_AUDIT = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_models_audit_v1/receipt.json"
LATE_MODELS = VPM / "vpm/results/harness/vpm_net_lane_clean_v1/entrant_models"
EARLY_ANNUAL = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_annual_cuts_v1"
EARLY_ANNUAL_AUDIT = RAPM / "nba_pipeline/results/estimated_skills/vpm_early_entrant_annual_audit_v1/receipt.json"
LATE_ANNUAL = RAPM / "nba_pipeline/results/estimated_skills/vpm_late_entrant_annual_cuts_v2"
LATE_ANNUAL_AUDIT = RAPM / "nba_pipeline/results/estimated_skills/vpm_late_entrant_annual_audit_v2/receipt.json"


def free_space(folder: Path) -> None:
    ancestor = folder
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 15 * 2**30:
        raise OSError("Preserve 15 GiB free")


class CareerAccumulator:
    def __init__(self):
        early = pd.read_parquet(EARLY / "player_game_exposure.parquet")
        later = pd.read_parquet(LATER)
        a = early[["player_id", "game_date", "possessions"]].copy()
        b = later[["player_id", "game_date", "off_poss", "def_poss"]].copy()
        b["possessions"] = (b.off_poss + b.def_poss) / 2
        self.lines = pd.concat([a, b[["player_id", "game_date", "possessions"]]], ignore_index=True)
        self.lines.game_date = pd.to_datetime(self.lines.game_date, utc=True).dt.tz_localize(None)
        self.lines = self.lines.sort_values("game_date", kind="stable").reset_index(drop=True)
        if self.lines.game_date.isna().any() or (self.lines.possessions < 0).any() or not np.isfinite(self.lines.possessions).all():
            raise ValueError("Invalid dated career ledger")
        self.dates = self.lines.game_date.to_numpy("datetime64[ns]")
        self.position = 0
        self.exposure = {}
        self.last_game = {}

    def before(self, cut: pd.Timestamp) -> pd.DataFrame:
        end = int(np.searchsorted(self.dates, np.datetime64(cut), side="left"))
        if end < self.position:
            raise ValueError("Career ledger rewound")
        if end > self.position:
            new = self.lines.iloc[self.position:end]
            grouped = new.groupby("player_id").agg(possessions=("possessions", "sum"), last_game=("game_date", "max"))
            for pid, row in grouped.iterrows():
                pid = int(pid)
                self.exposure[pid] = self.exposure.get(pid, 0.) + float(row.possessions)
                self.last_game[pid] = row.last_game
            self.position = end
        ids = list(self.exposure)
        return pd.DataFrame({"player_id": ids, "possessions": [self.exposure[i] for i in ids],
            "last_game": [self.last_game[i] for i in ids]})


def models() -> list[dict]:
    audit = json.loads(EARLY_MODEL_AUDIT.read_text())
    if audit["status"] != "passed_independent_18_early_entrant_model_normal_audit":
        raise ValueError("Early entrant model audit not admitted")
    result = []
    for root, early in [(EARLY_MODELS, True), (LATE_MODELS, False)]:
        for path in sorted(root.glob("20??.json")):
            saved = json.loads(path.read_text())
            core = saved if early else saved["parent"]
            if saved["components"] != ["rim_freq", "rim_conversion", "mid_freq", "mid_conversion",
                    "three_freq", "three_conversion", "ft", "tov", "second_chance",
                    "mid_reference_on_rim", "mid_reference_on_three"]:
                raise ValueError("Entrant head census differs")
            result.append(dict(path=path, sha256=sha256_file(path), fit_cut=pd.Timestamp(core["fit_cut"]),
                fit_end=pd.Timestamp(core["fit_end"]), scaler=core["scaler"],
                coefficients=np.asarray(saved["coefficients"], float),
                parent=np.asarray(saved["parent_coefficients"] if early else core["coef"], float),
                early=early))
    result.sort(key=lambda item: item["fit_cut"])
    if len(result) != 26 or result[0]["fit_cut"] != pd.Timestamp("2000-10-01"):
        raise ValueError("Exact 18 early plus 8 frozen later vintages required")
    return result


def choose(models: list[dict], cut: pd.Timestamp) -> tuple[dict, bool]:
    eligible = [m for m in models if m["fit_cut"] <= cut and m["fit_end"] < cut]
    if eligible:
        return eligible[-1], False
    if cut < pd.Timestamp("2000-10-01"):
        return models[0], True
    raise ValueError("No chronological entrant model")


def background(ids, cut, history, sources, first, models):
    model, backcast = choose(models, cut)
    packet = feature_packet(sources, history, ids, cut, first)
    exposure = packet.nba_possessions.to_numpy(float)
    first_year = first.reindex(ids).to_numpy(float)
    draft_year = sources.draft.reindex(ids).draft_year.to_numpy(float)
    career_complete = (first_year >= 1998) | ((first_year == 1997) & (draft_year == 1996))
    identified = career_complete | (exposure >= 2000)
    raw_weight = entrant_weight(exposure, 2000.)
    effective = np.where(identified, raw_weight, 0.)
    reported = np.where(identified, raw_weight, np.nan)
    f = transformed_features(packet, model["scaler"])
    p = f.shape[1]
    co = model["coefficients"]
    if co.shape != (2 * p, 11) or model["parent"].shape != (2 * p,):
        raise ValueError("Entrant model feature geometry differs")
    bg = np.stack([f @ co[:p], f @ co[p:]], axis=1)
    parent = np.column_stack([f @ model["parent"][:p], f @ model["parent"][p:]])
    if np.max(abs(bg[:, :, :len(LANES)].sum(axis=2) - parent), initial=0) > 1e-7:
        raise ValueError("Entrant 11-head/Parent closure failed")
    return model, bg, effective, reported, identified, exposure, career_complete, backcast


def annual_comparison(board, day):
    annual = EARLY_ANNUAL if day.year <= 2017 else LATE_ANNUAL
    saved = pd.read_parquet(annual / str(day.date()) / "ratings.parquet")
    if not np.array_equal(board.ratings.player_id.to_numpy(), saved.player_id.to_numpy()):
        raise ValueError("Entrant daily/annual player identity differs")
    fields = ["ovpm", "dvpm", "vpm", "own_prior_off", "own_prior_def", "combined_prior_off", "combined_prior_def"]
    gap = float(np.max(abs(board.ratings[fields].to_numpy(float) - saved[fields].to_numpy(float))))
    if gap > 1e-8:
        raise ValueError(f"Entrant daily/annual source or solve gap {day.date()}: {gap}")
    return gap


def run(out: Path, limit: int, resume: bool) -> dict:
    if out.exists() and not resume:
        raise FileExistsError("Preserve prior daily entrant replay root")
    if not out.exists() and resume:
        raise FileNotFoundError("Cannot resume absent root")
    free_space(out)
    start = time.monotonic()
    calendar = pd.read_parquet(CALENDAR)
    if len(calendar) != 6108 or str(calendar.estimate_date.min().date()) != "1996-11-01" or str(calendar.estimate_date.max().date()) != "2026-10-01":
        raise ValueError("Full frozen 6108-date calendar required")
    model_list = models()
    source_files = [Path(__file__), CALENDAR, TARGET_GAMES, V74_REGISTRATION, V74_COMPLETION,
        BASE / "manifest.json", EARLY / "receipt.json", EARLY / "player_game_exposure.parquet", LATER,
        NAMES, SOURCES / "nba_history.parquet", EARLY_MODEL_AUDIT, EARLY_MODELS / "receipt.json",
        EARLY_ANNUAL / "receipt.json", EARLY_ANNUAL_AUDIT,
        LATE_ANNUAL / "receipt.json", LATE_ANNUAL_AUDIT]
    source_files += [RAPM / "nba_pipeline/scripts" / name for name in ["build_vpm_entrant_annual_cuts.py",
        "fit_vpm_early_entrant_models_v2.py", "build_vpm_early_entrant_annual_cuts.py",
        "build_vpm_late_entrant_annual_cuts_v2.py"]]
    source_files += [VPM / "vpm/harness" / name for name in ["current_entrant_prior.py",
        "full_bank_history_storage.py", "full_bank_history_numerics.py", "entrant_priors.py",
        "monthly_vpm.py", "daily_history_1997.py"]]
    source_files += [m["path"] for m in model_list]
    source_files += [SOURCES / name for name in ["draft_identity_snapshot_20260926.json", "college_snapshot_20260926.json",
        "college_crosswalk_20260926.json", "gleague_player_games.parquet", "verified_draft_metadata.json", "verified_identity_overrides.json"]]
    source_pins = {str(p.resolve()): sha256_file(p) for p in source_files}
    out.mkdir(parents=True, exist_ok=True)
    (out / "cuts").mkdir(exist_ok=True)
    registration = out / "registration.json"
    proposal = dict(schema="full_feature_entrant_daily_research_replay_v1", calendar_dates=6108,
        source_pins=source_pins, exact_early_backcast="2000-10-01 before release; explicitly retrospective",
        selected_pointer_changed=False, full_feature_base_audit_pending=True,
        old_history_immutable=True, minimum_free_gib=15)
    if registration.exists():
        if json.loads(registration.read_text()) != proposal:
            raise ValueError("Daily entrant replay registration differs")
    else:
        registration.write_text(json.dumps(proposal, indent=2, allow_nan=False) + "\n")
    small = {str(registration.resolve()): sha256_file(registration)}
    paths = discover_paths()
    data, games = adapter(paths).load_before_cut(TARGET, "2026-10-02", range(1997, 2027), clean_sq_root=CLEAN_SQ)
    exact_native_calendar(games, pd.read_parquet(TARGET_GAMES))
    data = data.sort_values("game_date", kind="stable").reset_index(drop=True)
    games = games.sort_values("game_date", kind="stable").reset_index(drop=True)
    native_dates = data.game_date.to_numpy("datetime64[ns]")
    game_dates = games.game_date.to_numpy("datetime64[ns]")
    career = CareerAccumulator()
    first = pd.read_parquet(SOURCES / "nba_history.parquet").groupby("player_id").season.min()
    sources = load_sources(SOURCES, NAMES)
    checkpoints = sorted((out / "checkpoints").glob("*/controller.json"))
    if checkpoints:
        control = json.loads(checkpoints[-1].read_text())
        engine = FullBankHistoryGeometry.resume(checkpoints[-1].parent)
        begin = control["next_date_index"]
        if control["state_fingerprint"] != engine.fingerprint or begin > len(calendar):
            raise ValueError("Daily entrant checkpoint differs")
    else:
        engine = FullBankHistoryGeometry()
        begin = 0
    atompos = 0 if engine.anchor is None else int(np.searchsorted(native_dates, np.datetime64(engine.anchor), side="right"))
    gamepos = 0 if engine.anchor is None else int(np.searchsorted(game_dates, np.datetime64(engine.anchor), side="right"))
    completed = begin
    for i, row in enumerate(calendar.itertuples(index=False)):
        if i < begin:
            continue
        cut = pd.Timestamp(row.estimate_date)
        day = str(cut.date())
        free_space(out)
        end = int(np.searchsorted(native_dates, np.datetime64(cut), side="left"))
        gend = int(np.searchsorted(game_dates, np.datetime64(cut), side="left"))
        if end > atompos:
            engine.append(data.iloc[atompos:end], games.iloc[gamepos:gend], as_of=cut, input_hashes=small)
            atompos, gamepos = end, gend
        elif gend != gamepos:
            raise ValueError("Pricing games without admitted target")
        history = career.before(cut)
        target = out / "cuts" / day
        if target.exists():
            manifest = json.loads((target / "manifest.json").read_text())
            if manifest.get("source_metadata", {}).get("estimate_date") != day:
                raise ValueError("Existing daily entrant cut differs")
            for name, digest in manifest["output_hashes"].items():
                if sha256_file(Path(name)) != digest:
                    raise ValueError("Existing daily entrant output changed")
        elif engine.anchor is None:
            target.mkdir(parents=True)
            (target / "cold.json").write_text(json.dumps(dict(status="unestimated_no_earlier_admitted_lineups", date=day)) + "\n")
        else:
            own_path = BASE / "cuts" / day
            own = read_compact_board(own_path)
            ids = np.asarray(own.arrays["ids"], np.int64)
            if own.metadata["estimate_date"] != day or len(ids) != len(engine.players):
                raise ValueError("Daily own board/native geometry identity differs")
            model, bg, weights, reported, identified, exposure, complete, backcast = background(ids, cut, history, sources, first, model_list)
            board = apply_entrant_background(engine, own, bg, weights,
                reported_weights=reported, identified=identified, career_possessions=exposure,
                career_complete=complete, metadata=dict(entrant_model_fit_cut=str(model["fit_cut"].date()),
                    entrant_model_fit_end=str(model["fit_end"].date()), entrant_model_sha256=model["sha256"],
                    historical_entrant_model_backcast=backcast, historical_entrant_daily_research=True))
            board.metadata["recipe"] = "full_feature_historical_entrant_daily_replay_v1"
            annual_gap = annual_comparison(board, cut) if row.annual_checkpoint else None
            pins = dict(small)
            pins[str((own_path / "manifest.json").resolve())] = sha256_file(own_path / "manifest.json")
            pins[str(model["path"].resolve())] = model["sha256"]
            save_board(board, target, input_hashes=pins)
            (target / "daily_replay.json").write_text(json.dumps(dict(date=day, query_season=int(row.query_season),
                players=len(ids), model_sha256=model["sha256"], model_fit_cut=str(model["fit_cut"].date()),
                entrant_model_backcast=backcast, positive_weights=int((weights > 0).sum()),
                unidentified_weights=int((~identified).sum()), annual_gap=annual_gap,
                normal_equation_error=board.metadata["normal_equation_error"],
                native_net_gap=board.metadata["maximum_native_lane_sum_gap"]), indent=2, allow_nan=False) + "\n")
        completed = i + 1
        if completed % 100 == 0 or row.annual_checkpoint or (limit and completed >= limit):
            checkpoint = out / "checkpoints" / day
            if not (checkpoint / "checkpoint.json").exists():
                engine.checkpoint(checkpoint, input_hashes=small)
                (checkpoint / "controller.json").write_text(json.dumps(dict(next_date_index=completed,
                    state_fingerprint=engine.fingerprint, date=day), indent=2) + "\n")
        if completed % 25 == 0 or row.annual_checkpoint:
            (out / "progress.json").write_text(json.dumps(dict(completed_dates=completed, total_dates=len(calendar),
                latest_date=day, elapsed_seconds=time.monotonic() - start), indent=2) + "\n")
            print(json.dumps(dict(stage="daily_entrant_progress", completed=completed, total=len(calendar), day=day,
                elapsed_seconds=round(time.monotonic() - start, 1))), flush=True)
        if limit and completed >= limit:
            break
    result = dict(status="produced_full_entrant_daily_history_pending_independent_audit" if completed == len(calendar)
        else "partial_entrant_daily_research_replay", completed_dates=completed, total_dates=len(calendar),
        last_date=str(calendar.iloc[completed - 1].estimate_date.date()),
        registration_sha256=sha256_file(registration), full_feature_base_audit_pending=True,
        reader_changed=False, selected_pointer_changed=False, elapsed_seconds=time.monotonic() - start)
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve(), args.limit, args.resume), indent=2), flush=True)

#!/usr/bin/env python3
"""Fit chronological 2000–17 entrant Net models on admitted historical targets.

Only observed pre-release NBA games enter each fit. The 1997 left-censored
population is excluded from entrant design; the first releasable model is 2000.
This is a new early-era model family, separate from frozen 2018–25 models.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve

RAPM = Path(__file__).resolve().parents[2]
VPM = RAPM.parent / "vpm"
sys.path.insert(0, str(VPM))
sys.path.insert(0, str(RAPM / "nba_pipeline/scripts"))

from vpm.harness.annual_history_1997 import adapter  # noqa: E402
from vpm.harness.artifacts import sha256_file  # noqa: E402
from vpm.harness.entrant_priors import SeasonLikelihood, fit_scaler, load_sources, transformed_features, SLOTS  # noqa: E402
from vpm.harness.monthly_vpm import entrant_weight  # noqa: E402
from vpm.harness.paths import discover_paths  # noqa: E402
from vpm_net_lane_targets import LANES, REFERENCES, targets  # noqa: E402
from build_vpm_entrant_annual_cuts import career_packet as base_career_packet, EARLY, LATER, NAMES, SOURCES  # noqa: E402

TARGET = RAPM / "nba_pipeline/results/vpm_historical_targets_1997_dated_v2"
CLEAN_SQ = RAPM / "nba_pipeline/results/monthly_vpm_targets_clean_sq_v2"
EXCLUDE = {21300316, 21400843, 21401185, 41300305}


def career_packet(ids, cut, sources, early, later, first):
    """Identify the 1996 drafted class as true first-observed-1997 entrants."""
    packet, weights, reported, identified = base_career_packet(ids, cut, sources, early, later, first)
    draft = sources.draft.reindex(ids)
    class_of_1996 = (first.reindex(ids).to_numpy(float) == 1997) & draft.draft_year.eq(1996).to_numpy() & draft.pick.notna().to_numpy()
    identified = identified | class_of_1996
    raw = entrant_weight(packet.nba_possessions.to_numpy(float), 2000.)
    weights = np.where(identified, raw, 0.)
    reported = np.where(identified, raw, np.nan)
    return packet, weights, reported, identified


def fit(out: Path, through: int) -> dict:
    if out.exists():
        raise FileExistsError("Preserve previous early entrant fit root")
    if not 2000 <= through <= 2017:
        raise ValueError("Early entrant releases are 2000–2017 only")
    if shutil.disk_usage(out.parent).free < 15 * 2**30:
        raise OSError("Preserve 15 GiB disk reserve")
    paths = discover_paths()
    data, games = adapter(paths).load_before_cut(TARGET, f"{through}-10-01", range(1997, through + 1), clean_sq_root=CLEAN_SQ)
    data = data.loc[~pd.to_numeric(data.game_id).isin(EXCLUDE)].copy()
    games = games.loc[~pd.to_numeric(games.game_id).isin(EXCLUDE)].copy()
    # Assert exact season/source boundary before fitting any model.
    if data.source_season_end_year.max() > through or games.source_season_end_year.max() > through:
        raise ValueError("Future season reached entrant training data")
    early = pd.read_parquet(EARLY / "player_game_exposure.parquet")
    later = pd.read_parquet(LATER)
    early.game_date = pd.to_datetime(early.game_date, utc=True).dt.tz_localize(None)
    later.game_date = pd.to_datetime(later.game_date, utc=True).dt.tz_localize(None)
    first = pd.read_parquet(SOURCES / "nba_history.parquet").groupby("player_id").season.min()
    sources = load_sources(SOURCES, NAMES)
    source_paths = [Path(__file__), Path(__file__).with_name("build_vpm_entrant_annual_cuts.py"),
        Path(__file__).with_name("vpm_net_lane_targets.py"), EARLY / "player_game_exposure.parquet",
        EARLY / "receipt.json", LATER, NAMES, SOURCES / "nba_history.parquet"]
    source_paths += [SOURCES / name for name in ["draft_identity_snapshot_20260926.json", "college_snapshot_20260926.json",
        "college_crosswalk_20260926.json", "gleague_player_games.parquet", "verified_draft_metadata.json",
        "verified_identity_overrides.json"]]
    for year in range(1997, through + 1):
        for phase in ["RS", "PS"]:
            for prefix in ["atoms", "games"]:
                p = TARGET / f"{prefix}_{year}_{phase}.parquet"
                if p.exists():
                    source_paths.append(p)
    pins = {str(p.resolve()): sha256_file(p) for p in source_paths}
    out.mkdir(parents=True)
    cache = []
    records = []
    for year in range(1997, through + 1):
        d = data.loc[data.source_season_end_year.eq(year) & data.season_phase.eq("RS")].copy().reset_index(drop=True)
        g = games.loc[games.source_season_end_year.eq(year) & games.season_phase.eq("RS")].copy()
        if d.empty or g.empty:
            raise ValueError(f"Empty identified RS cohort: {year}")
        y, pricing = targets(d, g)
        ids = np.unique(d[SLOTS].to_numpy(np.int64))
        d["Off_Diff"] = d.actual_points
        likelihood = SeasonLikelihood(d, ids)
        zero = cho_solve(likelihood.factor, likelihood.x.T @ (y - y.mean(axis=0)), check_finite=False)
        if np.max(abs(zero[:, :len(LANES)].sum(axis=1) - likelihood.zero)) > 1e-9:
            raise ValueError("11-head target and Parent actual-points fit differ")
        cohort_cut = pd.Timestamp(f"{year - 1}-10-01")
        packet, weights, _, identified = career_packet(ids, cohort_cut, sources, early, later, first)
        if (weights[~identified] != 0).any():
            raise ValueError("Left-censored entrant prior is active")
        cache.append(dict(year=year, packet=packet, weight=weights, likelihood=likelihood,
            zero=zero, last_game=str(pd.to_datetime(d.game_date).max().date()), rows=len(d),
            games=int(g.game_id.nunique()), target_gap=pricing["maximum_row_gap"]))
        print(json.dumps(dict(stage="cohort", year=year, rows=len(d), active_entrants=int((weights > 0).sum()))), flush=True)
        if year < 2000:
            continue
        cut = pd.Timestamp(f"{year}-10-01")
        if any(pd.Timestamp(c["last_game"]) >= cut for c in cache):
            raise ValueError("Entrant fit reaches release date")
        active_packets = [c["packet"].iloc[np.flatnonzero(c["weight"] > 0)] for c in cache if (c["weight"] > 0).any()]
        scaler = fit_scaler(active_packets, "combined", 2000.)
        p = len(scaler["columns"])
        information = np.zeros((2 * p, 2 * p))
        rhs = np.zeros((2 * p, len(LANES) + len(REFERENCES)))
        for cohort in cache:
            likelihood = cohort["likelihood"]
            f = transformed_features(cohort["packet"], scaler) * cohort["weight"][:, None] / 100
            n = len(f)
            z = np.zeros((2 * n, 2 * p))
            z[:n, :p] = f
            z[n:, p:] = -f
            response = cho_solve(likelihood.factor, likelihood.penalty[:, None] * z, check_finite=False)
            info = z.T @ (likelihood.penalty[:, None] * (z - response))
            mass_weight = 250000 / likelihood.mass
            information += mass_weight * (info + info.T) / 2
            rhs += mass_weight * (z.T @ (likelihood.penalty[:, None] * cohort["zero"]))
        penalty = np.full(2 * p, 10.)
        penalty[[0, p]] = 0.
        normal = information + np.diag(penalty)
        if np.linalg.eigvalsh(normal).min() <= 1e-9:
            raise ValueError("Early entrant profile is not identified")
        coef = np.linalg.solve(normal, rhs)
        parent = np.linalg.solve(normal, rhs[:, :len(LANES)].sum(axis=1))
        coefficient_gap = float(np.max(abs(coef[:, :len(LANES)].sum(axis=1) - parent)))
        normal_error = float(np.max(abs(normal @ coef - rhs)))
        if coefficient_gap > 1e-9 or normal_error > 1e-9:
            raise ValueError("Early entrant coefficient/normal closure failed")
        model = dict(vintage=year, fit_cut=str(cut.date()), fit_end=max(c["last_game"] for c in cache),
            training_seasons=[c["year"] for c in cache], components=LANES + REFERENCES,
            scaler=scaler, coefficients=coef.tolist(), parent_coefficients=parent.tolist(),
            full_own_possessions=2000, ridge=10, coefficient_sum_gap=coefficient_gap,
            normal_equation_error=normal_error, condition_number=float(np.linalg.cond(normal)),
            training_rows=sum(c["rows"] for c in cache), training_games=sum(c["games"] for c in cache),
            early_archive_left_censor_policy="first-observed-1997 1996 drafted class identified; other under2000 players get zero effective entrant weight",
            source_publication_vintages_certified=False, protected_2027_inputs=0,
            status="early_entrant_model_pending_independent_audit")
        path = out / f"{year}.json"
        path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(out / f"{year}_normal.npz", information=information, rhs=rhs, penalty=penalty)
        records.append(dict(vintage=year, model_sha256=sha256_file(path), normal_sha256=sha256_file(out / f"{year}_normal.npz"),
            training_rows=model["training_rows"], training_games=model["training_games"],
            coefficient_sum_gap=coefficient_gap, normal_equation_error=normal_error))
        print(json.dumps(dict(stage="model", **records[-1])), flush=True)
    result = dict(status="early_entrant_models_pending_independent_audit", releases=records,
        source_hashes=pins, first_backcast_release="2000-10-01", backcast_dates_before_release=True,
        protected_2027_inputs=0, full_daily_history_complete=False)
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--through", required=True, type=int)
    args = parser.parse_args()
    fit(args.out.resolve(), args.through)

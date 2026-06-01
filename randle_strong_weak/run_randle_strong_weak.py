#!/usr/bin/env python3
"""Standalone opponent-strength RAPM experiments.

This is a research harness, not a production publisher. It uses processed RAPM
parquets as the row surface and fits identifiable split/slope models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "nba_pipeline"
DEFAULT_DARKO = Path("/Users/russellthomas/Docs/2026_NBA_PIPELINE/databallr/darko/dpm_history.csv")

O_COLS = [f"O{i}" for i in range(1, 6)]
D_COLS = [f"D{i}" for i in range(1, 6)]


@dataclass(frozen=True)
class FitConfig:
    start_year: int
    end_year: int
    season_type: str
    base_alpha: float
    interaction_alpha_mult: float
    darko_history: Path
    validation_fraction: float
    random_seed: int


def parse_args() -> FitConfig:
    parser = argparse.ArgumentParser(description="Fit standalone strong/weak RAPM research models.")
    parser.add_argument("--start-year", type=int, default=21)
    parser.add_argument("--end-year", type=int, default=26)
    parser.add_argument("--season-type", choices=["RS", "PS", "ALL"], default="ALL")
    parser.add_argument("--base-alpha", type=float, default=1000.0)
    parser.add_argument(
        "--interaction-alpha-mult",
        type=float,
        default=4.0,
        help="Penalty multiplier for strong-delta/slope features relative to base player effects.",
    )
    parser.add_argument("--darko-history", type=Path, default=DEFAULT_DARKO)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    return FitConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        season_type=args.season_type,
        base_alpha=args.base_alpha,
        interaction_alpha_mult=args.interaction_alpha_mult,
        darko_history=args.darko_history,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
    )


def two_digit_years(start_year: int, end_year: int) -> list[int]:
    return list(range(int(start_year), int(end_year) + 1))


def processed_paths(config: FitConfig) -> list[Path]:
    paths: list[Path] = []
    for yy in two_digit_years(config.start_year, config.end_year):
        if config.season_type in ("RS", "ALL"):
            paths.append(PIPELINE_ROOT / "processed" / f"RAPM{yy}.parquet")
        if config.season_type in ("PS", "ALL"):
            paths.append(PIPELINE_ROOT / "processed" / f"RAPM{yy}_PS.parquet")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing processed RAPM files:\n" + "\n".join(missing))
    return paths


def load_rows(config: FitConfig) -> pd.DataFrame:
    frames = []
    for path in processed_paths(config):
        df = pd.read_parquet(path)
        if not set(O_COLS + D_COLS + ["Off_Diff", "game_id"]).issubset(df.columns):
            raise ValueError(f"{path} is missing required processed RAPM columns")
        yy = int(path.stem.replace("RAPM", "").replace("_PS", ""))
        phase = "PS" if path.stem.endswith("_PS") else "RS"
        keep = ["game_id", "Off_Diff", "Season"] + O_COLS + D_COLS
        if "game_date" in df.columns:
            keep.append("game_date")
        part = df[keep].copy()
        part["source_year"] = 2000 + yy if yy < 90 else 1900 + yy
        part["season_phase"] = phase
        frames.append(part)
    rows = pd.concat(frames, ignore_index=True)
    rows["season"] = pd.to_numeric(rows["Season"], errors="coerce").fillna(rows["source_year"]).astype(int)
    rows["target"] = pd.to_numeric(rows["Off_Diff"], errors="coerce").fillna(0.0)
    for col in O_COLS + D_COLS:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0).astype(int)
    return rows


def load_player_names() -> dict[int, str]:
    names: dict[int, str] = {}
    map_path = REPO_ROOT / "autocomplete_map.csv"
    if map_path.exists():
        df = pd.read_csv(map_path)
        for row in df.itertuples(index=False):
            try:
                names[int(row.nba_id)] = str(row.player_name)
            except Exception:
                continue
    return names


def load_darko(config: FitConfig) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    df = pd.read_csv(config.darko_history)
    off: dict[tuple[int, int], float] = {}
    defense: dict[tuple[int, int], float] = {}
    for row in df.itertuples(index=False):
        try:
            key = (int(row.nba_id), int(row.season))
            off[key] = float(row.o_dpm)
            defense[key] = float(row.d_dpm)
        except Exception:
            continue
    return off, defense


def lineup_sum(players: np.ndarray, seasons: np.ndarray, lookup: dict[tuple[int, int], float]) -> np.ndarray:
    out = np.zeros(players.shape[0], dtype=np.float64)
    for slot in range(players.shape[1]):
        pids = players[:, slot]
        vals = np.fromiter(
            (lookup.get((int(pid), int(season)), 0.0) if pid > 0 else 0.0 for pid, season in zip(pids, seasons)),
            dtype=np.float64,
            count=len(pids),
        )
        out += vals
    return out


def add_strength_columns(rows: pd.DataFrame, config: FitConfig) -> pd.DataFrame:
    off_lookup, def_lookup = load_darko(config)
    offense = rows[O_COLS].to_numpy(dtype=np.int64)
    defense = rows[D_COLS].to_numpy(dtype=np.int64)
    seasons = rows["season"].to_numpy(dtype=np.int64)

    opp_def_strength = lineup_sum(defense, seasons, def_lookup)
    opp_off_strength = lineup_sum(offense, seasons, off_lookup)
    rows = rows.copy()
    rows["opp_def_strength"] = opp_def_strength
    rows["opp_off_strength"] = opp_off_strength
    rows["opp_def_strong"] = opp_def_strength > 2.0
    rows["opp_off_strong"] = opp_off_strength > 2.5

    rows["opp_def_strength_z"] = 0.0
    rows["opp_off_strength_z"] = 0.0
    for season, idx in rows.groupby("season").groups.items():
        idx = np.array(list(idx), dtype=np.int64)
        for source, target in [
            ("opp_def_strength", "opp_def_strength_z"),
            ("opp_off_strength", "opp_off_strength_z"),
        ]:
            vals = rows.loc[idx, source].to_numpy(dtype=np.float64)
            std = float(vals.std())
            rows.loc[idx, target] = 0.0 if std == 0.0 else (vals - float(vals.mean())) / std
    return rows


def player_universe(rows: pd.DataFrame) -> list[int]:
    players = np.unique(rows[O_COLS + D_COLS].to_numpy(dtype=np.int64).ravel())
    return [int(pid) for pid in players if int(pid) > 0]


def add_entries(
    row_idx: list[int],
    col_idx: list[int],
    values: list[float],
    rows: np.ndarray,
    players: np.ndarray,
    col_lookup: dict[tuple[str, int], int],
    kind: str,
    value_by_row: np.ndarray | None = None,
    scale: float = 1.0,
) -> None:
    for slot in range(players.shape[1]):
        pids = players[:, slot]
        for i, pid in enumerate(pids):
            if pid <= 0:
                continue
            col = col_lookup.get((kind, int(pid)))
            if col is None:
                continue
            row_idx.append(int(rows[i]))
            col_idx.append(col)
            value = 1.0 if value_by_row is None else float(value_by_row[i])
            values.append(value * scale)


def build_matrix(
    rows: pd.DataFrame,
    players: list[int],
    model: str,
    interaction_alpha_mult: float,
) -> tuple[sparse.csr_matrix, dict[tuple[str, int], int], dict[str, float]]:
    col_lookup: dict[tuple[str, int], int] = {}
    col = 0
    for pid in players:
        col_lookup[("off_base", pid)] = col
        col += 1
        col_lookup[("def_base", pid)] = col
        col += 1
    for pid in players:
        if model == "baseline":
            continue
        if model == "binary_reference":
            col_lookup[("off_strong_delta", pid)] = col
            col += 1
            col_lookup[("def_strong_delta", pid)] = col
            col += 1
        elif model == "continuous_slope":
            col_lookup[("off_strength_slope", pid)] = col
            col += 1
            col_lookup[("def_strength_slope", pid)] = col
            col += 1
        else:
            raise ValueError(f"unknown model {model}")

    offense = rows[O_COLS].to_numpy(dtype=np.int64)
    defense = rows[D_COLS].to_numpy(dtype=np.int64)
    row_numbers = np.arange(len(rows), dtype=np.int64)
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []

    add_entries(row_idx, col_idx, values, row_numbers, offense, col_lookup, "off_base")
    add_entries(row_idx, col_idx, values, row_numbers, defense, col_lookup, "def_base")

    interaction_scale = 1.0 / math.sqrt(interaction_alpha_mult)
    if model == "baseline":
        recover = {"interaction_scale": interaction_scale}
    elif model == "binary_reference":
        off_strong = rows["opp_def_strong"].to_numpy(dtype=np.float64)
        def_strong = rows["opp_off_strong"].to_numpy(dtype=np.float64)
        add_entries(row_idx, col_idx, values, row_numbers, offense, col_lookup, "off_strong_delta", off_strong, interaction_scale)
        add_entries(row_idx, col_idx, values, row_numbers, defense, col_lookup, "def_strong_delta", def_strong, interaction_scale)
        recover = {"interaction_scale": interaction_scale}
    elif model == "continuous_slope":
        off_z = rows["opp_def_strength_z"].to_numpy(dtype=np.float64)
        def_z = rows["opp_off_strength_z"].to_numpy(dtype=np.float64)
        add_entries(row_idx, col_idx, values, row_numbers, offense, col_lookup, "off_strength_slope", off_z, interaction_scale)
        add_entries(row_idx, col_idx, values, row_numbers, defense, col_lookup, "def_strength_slope", def_z, interaction_scale)
        recover = {"interaction_scale": interaction_scale}
    else:
        raise ValueError(f"unknown model {model}")

    x = sparse.coo_matrix((values, (row_idx, col_idx)), shape=(len(rows), col), dtype=np.float64).tocsr()
    return x, col_lookup, recover


def fit_ridge(x: sparse.csr_matrix, y: np.ndarray, alpha: float) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", max_iter=5000, tol=1e-6)
    model.fit(x, y)
    return model.coef_.astype(np.float64)


def predict(x: sparse.csr_matrix, beta: np.ndarray, y_mean: float) -> np.ndarray:
    return x @ beta + y_mean


def grouped_validation_mask(game_ids: Iterable[object], fraction: float, seed: int) -> np.ndarray:
    threshold = int(fraction * 10_000)
    mask = []
    for gid in game_ids:
        key = f"{seed}:{gid}".encode("utf-8")
        bucket = int(hashlib.md5(key).hexdigest()[:8], 16) % 10_000
        mask.append(bucket < threshold)
    return np.array(mask, dtype=bool)


def weighted_center(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    valid = weights > 0
    center = float(np.average(values[valid], weights=weights[valid])) if valid.any() else 0.0
    return values - center, center


def player_counts(rows: pd.DataFrame, players: list[int]) -> pd.DataFrame:
    records = {pid: {
        "player_id": pid,
        "off_poss": 0,
        "def_poss": 0,
        "faced_strong_off": 0,
        "faced_weak_off": 0,
        "faced_strong_def": 0,
        "faced_weak_def": 0,
        "off_strength_z_sum": 0.0,
        "def_strength_z_sum": 0.0,
    } for pid in players}
    offense = rows[O_COLS].to_numpy(dtype=np.int64)
    defense = rows[D_COLS].to_numpy(dtype=np.int64)
    off_strong = rows["opp_def_strong"].to_numpy(dtype=bool)
    def_strong = rows["opp_off_strong"].to_numpy(dtype=bool)
    off_z = rows["opp_def_strength_z"].to_numpy(dtype=np.float64)
    def_z = rows["opp_off_strength_z"].to_numpy(dtype=np.float64)
    for i in range(len(rows)):
        for pid in offense[i]:
            if pid <= 0:
                continue
            rec = records[int(pid)]
            rec["off_poss"] += 1
            rec["off_strength_z_sum"] += float(off_z[i])
            rec["faced_strong_off" if off_strong[i] else "faced_weak_off"] += 1
        for pid in defense[i]:
            if pid <= 0:
                continue
            rec = records[int(pid)]
            rec["def_poss"] += 1
            rec["def_strength_z_sum"] += float(def_z[i])
            rec["faced_strong_def" if def_strong[i] else "faced_weak_def"] += 1
    out = pd.DataFrame.from_records(list(records.values()))
    out["possessions"] = out["off_poss"] + out["def_poss"]
    out["strong_perc_off"] = np.where(out["off_poss"] > 0, out["faced_strong_off"] / out["off_poss"], 0.0)
    out["strong_perc_def"] = np.where(out["def_poss"] > 0, out["faced_strong_def"] / out["def_poss"], 0.0)
    out["observed_opp_def_strength_z"] = np.where(out["off_poss"] > 0, out["off_strength_z_sum"] / out["off_poss"], 0.0)
    out["observed_opp_off_strength_z"] = np.where(out["def_poss"] > 0, out["def_strength_z_sum"] / out["def_poss"], 0.0)
    return out


def export_binary(
    beta: np.ndarray,
    col_lookup: dict[tuple[str, int], int],
    recover: dict[str, float],
    counts: pd.DataFrame,
    names: dict[int, str],
) -> pd.DataFrame:
    scale = recover["interaction_scale"]
    off_base_values = np.array([beta[col_lookup[("off_base", int(rec.player_id))]] * 100.0 for rec in counts.itertuples(index=False)])
    def_base_values = np.array([beta[col_lookup[("def_base", int(rec.player_id))]] * 100.0 for rec in counts.itertuples(index=False)])
    off_centered, off_center = weighted_center(off_base_values, counts["off_poss"].to_numpy(dtype=np.float64))
    def_centered, def_center = weighted_center(def_base_values, counts["def_poss"].to_numpy(dtype=np.float64))
    centered_by_pid = {
        int(rec.player_id): (float(off_centered[i]), float(def_centered[i]))
        for i, rec in enumerate(counts.itertuples(index=False))
    }
    rows = []
    for rec in counts.itertuples(index=False):
        pid = int(rec.player_id)
        off_base, def_base = centered_by_pid[pid]
        off_delta = beta[col_lookup[("off_strong_delta", pid)]] * scale * 100.0
        def_delta = beta[col_lookup[("def_strong_delta", pid)]] * scale * 100.0
        off_vs_weak = off_base
        off_vs_strong = off_base + off_delta
        def_vs_weak = def_base
        def_vs_strong = def_base + def_delta
        overall_off = (1.0 - rec.strong_perc_off) * off_vs_weak + rec.strong_perc_off * off_vs_strong
        overall_def = (1.0 - rec.strong_perc_def) * def_vs_weak + rec.strong_perc_def * def_vs_strong
        rows.append({
            "player_id": pid,
            "player_name": names.get(pid, f"ID_{pid}"),
            "overall_net": overall_off - overall_def,
            "overall_off": overall_off,
            "overall_def": overall_def,
            "off_vs_strong": off_vs_strong,
            "off_vs_weak": off_vs_weak,
            "off_strong_delta": off_delta,
            "def_vs_strong": def_vs_strong,
            "def_vs_weak": def_vs_weak,
            "def_strong_delta": def_delta,
            "net_vs_strong": off_vs_strong - def_vs_strong,
            "net_vs_weak": off_vs_weak - def_vs_weak,
            "net_strong_minus_weak": (off_vs_strong - def_vs_strong) - (off_vs_weak - def_vs_weak),
            "possessions": int(rec.possessions),
            "off_poss": int(rec.off_poss),
            "def_poss": int(rec.def_poss),
            "faced_strong_off": int(rec.faced_strong_off),
            "faced_weak_off": int(rec.faced_weak_off),
            "faced_strong_def": int(rec.faced_strong_def),
            "faced_weak_def": int(rec.faced_weak_def),
            "strong_perc_off": rec.strong_perc_off,
            "strong_perc_def": rec.strong_perc_def,
            "report_off_center": off_center,
            "report_def_center": def_center,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("overall_net", ascending=False)


def export_continuous(
    beta: np.ndarray,
    col_lookup: dict[tuple[str, int], int],
    recover: dict[str, float],
    counts: pd.DataFrame,
    names: dict[int, str],
) -> pd.DataFrame:
    scale = recover["interaction_scale"]
    off_base_values = np.array([beta[col_lookup[("off_base", int(rec.player_id))]] * 100.0 for rec in counts.itertuples(index=False)])
    def_base_values = np.array([beta[col_lookup[("def_base", int(rec.player_id))]] * 100.0 for rec in counts.itertuples(index=False)])
    off_centered, off_center = weighted_center(off_base_values, counts["off_poss"].to_numpy(dtype=np.float64))
    def_centered, def_center = weighted_center(def_base_values, counts["def_poss"].to_numpy(dtype=np.float64))
    centered_by_pid = {
        int(rec.player_id): (float(off_centered[i]), float(def_centered[i]))
        for i, rec in enumerate(counts.itertuples(index=False))
    }
    rows = []
    for rec in counts.itertuples(index=False):
        pid = int(rec.player_id)
        off_avg, def_avg = centered_by_pid[pid]
        off_slope = beta[col_lookup[("off_strength_slope", pid)]] * scale * 100.0
        def_slope = beta[col_lookup[("def_strength_slope", pid)]] * scale * 100.0
        observed_off = off_avg + off_slope * rec.observed_opp_def_strength_z
        observed_def = def_avg + def_slope * rec.observed_opp_off_strength_z
        rows.append({
            "player_id": pid,
            "player_name": names.get(pid, f"ID_{pid}"),
            "observed_net": observed_off - observed_def,
            "off_vs_weak_1sd": off_avg - off_slope,
            "off_vs_avg": off_avg,
            "off_vs_strong_1sd": off_avg + off_slope,
            "off_strength_slope": off_slope,
            "def_vs_weak_1sd": def_avg - def_slope,
            "def_vs_avg": def_avg,
            "def_vs_strong_1sd": def_avg + def_slope,
            "def_strength_slope": def_slope,
            "net_vs_weak_1sd": (off_avg - off_slope) - (def_avg - def_slope),
            "net_vs_avg": off_avg - def_avg,
            "net_vs_strong_1sd": (off_avg + off_slope) - (def_avg + def_slope),
            "net_strong_minus_weak_2sd": 2.0 * (off_slope - def_slope),
            "observed_opp_def_strength_z": rec.observed_opp_def_strength_z,
            "observed_opp_off_strength_z": rec.observed_opp_off_strength_z,
            "possessions": int(rec.possessions),
            "off_poss": int(rec.off_poss),
            "def_poss": int(rec.def_poss),
            "strong_perc_off": rec.strong_perc_off,
            "strong_perc_def": rec.strong_perc_def,
            "report_off_center": off_center,
            "report_def_center": def_center,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("observed_net", ascending=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    config = parse_args()
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    rows = add_strength_columns(load_rows(config), config)
    players = player_universe(rows)
    names = load_player_names()
    counts = player_counts(rows, players)

    y_raw = rows["target"].to_numpy(dtype=np.float64)
    y_mean = float(y_raw.mean())
    y = y_raw - y_mean
    val_mask = grouped_validation_mask(rows["game_id"], config.validation_fraction, config.random_seed)
    train_mask = ~val_mask

    diagnostics = {
        "config": {
            "start_year": config.start_year,
            "end_year": config.end_year,
            "season_type": config.season_type,
            "base_alpha": config.base_alpha,
            "interaction_alpha_mult": config.interaction_alpha_mult,
            "darko_history": str(config.darko_history),
            "validation_fraction": config.validation_fraction,
            "random_seed": config.random_seed,
        },
        "rows": int(len(rows)),
        "players": int(len(players)),
        "target_mean": y_mean,
        "lineup_distribution": {
            "offense_vs_strong_defense_rows": int(rows["opp_def_strong"].sum()),
            "offense_vs_weak_defense_rows": int((~rows["opp_def_strong"]).sum()),
            "offense_vs_strong_defense_pct": float(rows["opp_def_strong"].mean()),
            "defense_vs_strong_offense_rows": int(rows["opp_off_strong"].sum()),
            "defense_vs_weak_offense_rows": int((~rows["opp_off_strong"]).sum()),
            "defense_vs_strong_offense_pct": float(rows["opp_off_strong"].mean()),
        },
        "validation_rows": int(val_mask.sum()),
        "train_rows": int(train_mask.sum()),
        "models": {},
    }

    for model_name in ["baseline", "binary_reference", "continuous_slope"]:
        x, lookup, recover = build_matrix(rows, players, model_name, config.interaction_alpha_mult)
        beta = fit_ridge(x[train_mask], y[train_mask], config.base_alpha)
        yhat_train = predict(x[train_mask], beta, y_mean)
        yhat_val = predict(x[val_mask], beta, y_mean)
        beta_full = fit_ridge(x, y, config.base_alpha)
        diagnostics["models"][model_name] = {
            "features": int(x.shape[1]),
            "train_rmse": rmse(y_raw[train_mask], yhat_train),
            "validation_rmse": rmse(y_raw[val_mask], yhat_val),
        }
        if model_name == "baseline":
            continue
        if model_name == "binary_reference":
            out = export_binary(beta_full, lookup, recover, counts, names)
            path = output_dir / f"binary_reference_{config.start_year}_{config.end_year}_{config.season_type.lower()}_a{int(config.base_alpha)}_im{config.interaction_alpha_mult:g}.csv"
        else:
            out = export_continuous(beta_full, lookup, recover, counts, names)
            path = output_dir / f"continuous_slope_{config.start_year}_{config.end_year}_{config.season_type.lower()}_a{int(config.base_alpha)}_im{config.interaction_alpha_mult:g}.csv"
        out = out.round(4)
        out.to_csv(path, index=False)
        diagnostics["models"][model_name]["output"] = str(path)

    diag_path = output_dir / f"diagnostics_{config.start_year}_{config.end_year}_{config.season_type.lower()}_a{int(config.base_alpha)}_im{config.interaction_alpha_mult:g}.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

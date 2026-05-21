#!/usr/bin/env python3
"""
Run ridge regression on SHOOTER_OREB missed-FGA rebound opportunities.

Design:
    y = Offensive_Rebound
    X = 4 non-shooter offensive players + shooter + 5 defensive players

Outputs per-100 percentage-point coefficients:
    non_shooter_oreb
    shooter_miss_recoverability
    def_oreb_suppression
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
RESULTS_DIR = PIPELINE_ROOT / "results"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_end_year(year: int) -> int:
    year = int(year)
    if year < 100:
        return 1900 + year if year >= 97 else 2000 + year
    return year


def season_suffix(year: int) -> str:
    return f"{normalize_end_year(year) % 100:02d}"


def expand_year_window(start_year: int, end_year: int) -> list[int]:
    start_full = normalize_end_year(start_year)
    end_full = normalize_end_year(end_year)
    if end_full < start_full:
        end_full += 100
    return list(range(start_full, end_full + 1))


def season_type_label(value: str) -> str:
    value = value.upper()
    if value not in {"RS", "PS", "ALL"}:
        raise ValueError("season_type must be RS, PS, or ALL")
    return value


def input_files(start_year: int, end_year: int, season_type: str) -> list[Path]:
    paths = []
    for full_year in expand_year_window(start_year, end_year):
        suffix = season_suffix(full_year)
        if season_type in {"RS", "ALL"}:
            paths.append(PROCESSED_DIR / f"SHOOTER_OREB{suffix}.parquet")
        if season_type in {"PS", "ALL"}:
            paths.append(PROCESSED_DIR / f"SHOOTER_OREB{suffix}_PS.parquet")
    return paths


def read_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            logging.warning("Skipping missing input %s", path)
            continue
        frame = pd.read_parquet(path)
        frame["source_file"] = path.name
        frames.append(frame)
        logging.info("Loaded %s (%d rows)", path.name, len(frame))

    if not frames:
        raise FileNotFoundError("No SHOOTER_OREB input parquets found for the requested window.")

    df = pd.concat(frames, ignore_index=True)
    required = ["Offensive_Rebound", "Shooter", *[f"O{i}" for i in range(1, 5)], *[f"D{i}" for i in range(1, 6)]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["Offensive_Rebound", "Self_Offensive_Rebound", "Shooter", "Rebounder", *[f"O{i}" for i in range(1, 5)], *[f"D{i}" for i in range(1, 6)]]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=required).copy()
    logging.info("Rows after required-column filter: %d (removed %d)", len(df), before - len(df))
    return df


def build_player_name_map(player_ids: set[str]) -> dict[str, str]:
    name_map = {}

    def store(pid, name):
        if pd.isna(pid) or pd.isna(name):
            return
        try:
            key = str(int(float(pid)))
        except (TypeError, ValueError):
            return
        if key not in player_ids:
            return
        cleaned = str(name).strip()
        if not cleaned:
            return
        current = name_map.get(key)
        if current is None or (len(cleaned.split()), len(cleaned)) > (len(current.split()), len(current)):
            name_map[key] = cleaned

    autocomplete = PROJECT_ROOT / "autocomplete_map.csv"
    if autocomplete.exists():
        try:
            amap = pd.read_csv(autocomplete)
            if {"nba_id", "player_name"}.issubset(amap.columns):
                for pid, name in amap[["nba_id", "player_name"]].itertuples(index=False, name=None):
                    store(pid, name)
        except Exception as exc:
            logging.warning("Could not read %s: %s", autocomplete, exc)

    unresolved = player_ids - set(name_map)
    if unresolved:
        for raw_path in sorted(RAW_DATA_DIR.glob("NBA*.parquet")):
            try:
                raw = pd.read_parquet(
                    raw_path,
                    columns=["player1_id", "player1_name", "player2_id", "player2_name", "player3_id", "player3_name"],
                )
            except Exception:
                continue
            for id_col, name_col in [("player1_id", "player1_name"), ("player2_id", "player2_name"), ("player3_id", "player3_name")]:
                for pid, name in raw[[id_col, name_col]].dropna().itertuples(index=False, name=None):
                    store(pid, name)
            unresolved = player_ids - set(name_map)
            if not unresolved:
                break

    return name_map


def build_design_matrix(df: pd.DataFrame, player_to_col: dict[str, int]) -> coo_matrix:
    rows = []
    cols = []
    data = []

    def add(row_idx: int, feature: str, value: float = 1.0):
        col_idx = player_to_col.get(feature)
        if col_idx is None:
            return
        rows.append(row_idx)
        cols.append(col_idx)
        data.append(value)

    for row_idx, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        for i in range(1, 5):
            pid = int(row_dict[f"O{i}"])
            if pid > 0:
                add(row_idx, f"{pid}_non_shooter_off")
        shooter = int(row_dict["Shooter"])
        if shooter > 0:
            add(row_idx, f"{shooter}_shooter")
        for i in range(1, 6):
            pid = int(row_dict[f"D{i}"])
            if pid > 0:
                add(row_idx, f"{pid}_def")

    return coo_matrix(
        (np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(len(df), len(player_to_col)),
        dtype=np.float64,
    ).tocsr()


def weighted_center_coefficients(beta: np.ndarray, player_to_col: dict[str, int], counts: dict[str, dict[str, int]]) -> None:
    for suffix, count_key in [
        ("_non_shooter_off", "non_shooter_poss"),
        ("_shooter", "shooter_misses"),
        ("_def", "def_poss"),
    ]:
        numer = 0.0
        denom = 0.0
        for feature, col_idx in player_to_col.items():
            if not feature.endswith(suffix):
                continue
            player_id = feature[: -len(suffix)]
            weight = counts[player_id][count_key]
            numer += beta[col_idx] * weight
            denom += weight
        if denom > 0:
            offset = numer / denom
            for feature, col_idx in player_to_col.items():
                if feature.endswith(suffix):
                    beta[col_idx] -= offset
            logging.info("Centered %s block by %.6f", suffix, offset)


def value_counts_for_columns(df: pd.DataFrame, columns: list[str]) -> defaultdict[str, int]:
    vals = pd.to_numeric(pd.Series(df[columns].to_numpy().ravel()), errors="coerce").dropna().astype(int)
    vals = vals[vals > 0].astype(str)
    return defaultdict(int, vals.value_counts().to_dict())


def run(start_year: int, end_year: int, season_type: str, alpha: float, min_shooter_misses: int) -> Path:
    season_type = season_type_label(season_type)
    df = read_inputs(input_files(start_year, end_year, season_type))

    player_ids = set()
    for col in ["Shooter", *[f"O{i}" for i in range(1, 5)], *[f"D{i}" for i in range(1, 6)]]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        player_ids.update(str(pid) for pid in vals.unique() if pid > 0)
    player_ids = set(sorted(player_ids))
    logging.info("Found %d players", len(player_ids))

    player_to_col = {}
    idx = 0
    for player_id in sorted(player_ids):
        for suffix in ["non_shooter_off", "shooter", "def"]:
            player_to_col[f"{player_id}_{suffix}"] = idx
            idx += 1

    X = build_design_matrix(df, player_to_col)
    y = pd.to_numeric(df["Offensive_Rebound"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    y_mean = float(y.mean())
    y_centered = y - y_mean
    logging.info("Design matrix shape=%s, y_mean=%.6f", X.shape, y_mean)

    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr")
    model.fit(X, y_centered)
    beta = model.coef_.astype(np.float64)

    counts = defaultdict(lambda: defaultdict(int))
    non_shooter_counts = value_counts_for_columns(df, [f"O{i}" for i in range(1, 5)])
    def_counts = value_counts_for_columns(df, [f"D{i}" for i in range(1, 6)])
    shooter_counts = defaultdict(int, df["Shooter"].astype(int).astype(str).value_counts().to_dict())
    team_oreb_by_shooter = defaultdict(int, df.groupby(df["Shooter"].astype(int).astype(str))["Offensive_Rebound"].sum().astype(int).to_dict())
    self_oreb_by_shooter = defaultdict(int, df.groupby(df["Shooter"].astype(int).astype(str))["Self_Offensive_Rebound"].sum().astype(int).to_dict())

    for player_id in player_ids:
        counts[player_id]["non_shooter_poss"] = int(non_shooter_counts[player_id])
        counts[player_id]["def_poss"] = int(def_counts[player_id])
        counts[player_id]["shooter_misses"] = int(shooter_counts[player_id])

    weighted_center_coefficients(beta, player_to_col, counts)
    name_map = build_player_name_map(player_ids)

    rows = []
    for player_id in sorted(player_ids, key=lambda x: int(x)):
        non_shooter = beta[player_to_col[f"{player_id}_non_shooter_off"]] * 100.0
        shooter = beta[player_to_col[f"{player_id}_shooter"]] * 100.0
        def_allowed = beta[player_to_col[f"{player_id}_def"]] * 100.0
        def_suppression = -def_allowed
        shooter_misses = counts[player_id]["shooter_misses"]
        team_oreb = int(team_oreb_by_shooter[player_id])
        self_oreb = int(self_oreb_by_shooter[player_id])
        rows.append({
            "player_id": player_id,
            "player_name": name_map.get(player_id, f"ID_{player_id}"),
            "non_shooter_oreb": round(non_shooter, 3),
            "shooter_miss_recoverability": round(shooter, 3),
            "def_oreb_suppression": round(def_suppression, 3),
            "shooter_misses": shooter_misses,
            "raw_team_oreb_after_miss": team_oreb,
            "raw_team_oreb_pct_after_miss": round(team_oreb / shooter_misses, 4) if shooter_misses else np.nan,
            "self_oreb_after_miss": self_oreb,
            "self_oreb_pct_after_miss": round(self_oreb / shooter_misses, 4) if shooter_misses else np.nan,
            "non_shooter_poss": counts[player_id]["non_shooter_poss"],
            "def_poss": counts[player_id]["def_poss"],
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["shooter_miss_recoverability", "shooter_misses"],
        ascending=[False, False],
    )
    if min_shooter_misses > 0:
        display_result = result[result["shooter_misses"] >= min_shooter_misses].copy()
    else:
        display_result = result

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{season_suffix(start_year)}_{season_suffix(end_year)}_{season_type.lower()}_a{int(alpha)}"
    output_path = RESULTS_DIR / f"shooter_oreb_{suffix}_results.csv"
    result.to_csv(output_path, index=False)
    logging.info("Wrote %s (%d rows)", output_path, len(result))

    print("\nTop shooter miss recoverability")
    print(display_result.head(20)[[
        "player_name", "shooter_miss_recoverability", "shooter_misses",
        "raw_team_oreb_pct_after_miss", "self_oreb_pct_after_miss",
    ]].to_string(index=False))

    print("\nBottom shooter miss recoverability")
    print(display_result.tail(20)[[
        "player_name", "shooter_miss_recoverability", "shooter_misses",
        "raw_team_oreb_pct_after_miss", "self_oreb_pct_after_miss",
    ]].to_string(index=False))

    print("\nTop non-shooter OREB")
    print(result[result["non_shooter_poss"] >= min_shooter_misses].sort_values("non_shooter_oreb", ascending=False).head(20)[[
        "player_name", "non_shooter_oreb", "non_shooter_poss",
    ]].to_string(index=False))

    print("\nTop defensive OREB suppression")
    print(result[result["def_poss"] >= min_shooter_misses].sort_values("def_oreb_suppression", ascending=False).head(20)[[
        "player_name", "def_oreb_suppression", "def_poss",
    ]].to_string(index=False))

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHOOTER_OREB ridge regression.")
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("season_type", choices=["RS", "PS", "ALL", "rs", "ps", "all"])
    parser.add_argument("--alpha", type=float, default=3000.0)
    parser.add_argument("--min-shooter-misses", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.start_year,
        args.end_year,
        args.season_type,
        alpha=args.alpha,
        min_shooter_misses=args.min_shooter_misses,
    )


if __name__ == "__main__":
    main()

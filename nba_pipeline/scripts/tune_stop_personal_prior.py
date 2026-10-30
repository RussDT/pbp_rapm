#!/usr/bin/env python3
"""Tune a defensive personal-STOP prior on rolling future-season folds.

The final prior uses a player's pooled credited STOP rate:

    (steals + offensive fouls drawn + recovered blocks) / defensive possessions

Alpha selection is date-correct.  Each fold builds both the lineup solve and
the personal prior from the trailing three seasons, then scores the next
season's STOP possessions.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

import rapm as rapm_solver


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"
DEFAULT_GRID = "300,1000,2000,3000,5000,10000,20000"


def fetch_personal_stop_history(start_year=2021, end_year=2026):
    """Fetch the canonical regular-season personal STOP fields."""
    load_dotenv(PIPELINE_ROOT.parent / ".env")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and a Supabase key are required")

    client = create_client(url, key)
    rows = []
    page_size = 1000
    start = 0
    while True:
        response = (
            client.table("player_stats_with_metrics")
            .select("nba_id,Name,year,playoffs,DefPoss,STOPS_100")
            .gte("year", int(start_year))
            .lte("year", int(end_year))
            .eq("playoffs", 0)
            .order("year")
            .order("nba_id")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    history = pd.DataFrame(rows)
    if history.empty:
        raise ValueError("No personal STOP history returned from Supabase")
    for col in ["nba_id", "year", "DefPoss", "STOPS_100"]:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.dropna(subset=["nba_id", "year", "DefPoss", "STOPS_100"])
    history["nba_id"] = history["nba_id"].astype(int)
    history["year"] = history["year"].astype(int)
    history["credited_stops"] = history["STOPS_100"] * history["DefPoss"] / 100.0
    return history


def aggregate_personal_stop_prior(history, start_year, end_year):
    """Build a pooled three-year relative STOP prior in percentage points."""
    selected = history[
        history["year"].between(int(start_year), int(end_year), inclusive="both")
    ].copy()
    if selected.empty:
        raise ValueError(f"No personal STOP rows for {start_year}-{end_year}")

    league_poss = float(selected["DefPoss"].sum())
    league_stops = float(selected["credited_stops"].sum())
    league_stop_pct = 100.0 * league_stops / league_poss

    grouped = (
        selected.groupby("nba_id", as_index=False)
        .agg(
            name=("Name", "last"),
            def_poss=("DefPoss", "sum"),
            credited_stops=("credited_stops", "sum"),
            seasons=("year", "nunique"),
        )
    )
    grouped["personal_stop_pct"] = (
        100.0 * grouped["credited_stops"] / grouped["def_poss"]
    )
    grouped["league_stop_pct"] = league_stop_pct
    grouped["relative_stop_pct"] = grouped["personal_stop_pct"] - league_stop_pct
    # rapm.py's raw defensive coefficient is negative-good.
    grouped["prior_value"] = -grouped["relative_stop_pct"]
    grouped["start_year"] = int(start_year)
    grouped["end_year"] = int(end_year)
    return grouped


def load_stop_window(start_year, end_year):
    frames = []
    for year in range(int(start_year), int(end_year) + 1):
        suffix = year % 100
        for phase in ["RS", "PS"]:
            phase_suffix = "" if phase == "RS" else "_PS"
            path = PROCESSED_DIR / f"STOP{suffix:02d}{phase_suffix}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path)
            frame["season_phase"] = phase
            frame["source_season_end_year"] = year
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _all_players(*frames):
    players = set()
    for frame in frames:
        for col in [f"O{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)]:
            values = pd.to_numeric(frame[col], errors="coerce").dropna().astype(int)
            players.update(str(value) for value in values if value > 0)
    return sorted(players)


def center_prior_on_training_defense(prior_frame, train_frame, all_players):
    """Center prior values over actual training defensive player exposures."""
    raw = dict(
        zip(
            prior_frame["nba_id"].astype(str),
            prior_frame["prior_value"].astype(float) / 100.0,
        )
    )
    counts = pd.to_numeric(
        pd.Series(train_frame[[f"D{i}" for i in range(1, 6)]].to_numpy().ravel()),
        errors="coerce",
    ).dropna().astype(int)
    counts = counts[counts > 0].astype(str).value_counts()
    total = float(counts.sum())
    offset = (
        sum(float(count) * raw.get(player, 0.0) for player, count in counts.items()) / total
        if total
        else 0.0
    )
    centered = {player: raw.get(player, 0.0) - offset for player in all_players}
    coverage = float(sum(counts.get(player, 0) for player in raw) / total) if total else 0.0
    return centered, offset, coverage


def prepare_fold(history, train_start, train_end, validation_year):
    train = load_stop_window(train_start, train_end)
    validation = load_stop_window(validation_year, validation_year)
    players = _all_players(train, validation)
    player_to_col = {}
    for player in players:
        player_to_col[f"{player}_off"] = len(player_to_col)
        player_to_col[f"{player}_def"] = len(player_to_col)

    train_ha = rapm_solver.transform_to_home_away_format(train)
    validation_ha = rapm_solver.transform_to_home_away_format(validation)
    X_train, y_train, _ = rapm_solver.build_simple_design_matrix(train_ha, player_to_col)
    X_validation, y_validation, _ = rapm_solver.build_simple_design_matrix(
        validation_ha, player_to_col
    )
    personal = aggregate_personal_stop_prior(history, train_start, train_end)
    def_prior, prior_offset, coverage = center_prior_on_training_defense(
        personal, train, players
    )
    return {
        "train_start": train_start,
        "train_end": train_end,
        "validation_year": validation_year,
        "X_train": X_train,
        "y_train": y_train - np.mean(y_train),
        "X_validation": X_validation,
        "y_validation": y_validation - np.mean(y_validation),
        "player_to_col": player_to_col,
        "players": players,
        "def_prior": def_prior,
        "prior_offset": prior_offset,
        "coverage": coverage,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }


def evaluate_pair(folds, off_alpha, def_alpha, use_prior=True):
    rows = []
    for fold in folds:
        beta = rapm_solver.simplified_alternating_rapm(
            fold["X_train"],
            fold["y_train"],
            fold["player_to_col"],
            fold["players"],
            alpha_offense=float(off_alpha),
            alpha_defense=float(def_alpha),
            max_iter=100,
            tol=1e-4,
            off_prior=None,
            def_prior=fold["def_prior"] if use_prior else None,
        )
        prediction = fold["X_validation"] @ beta
        mse = float(np.mean((fold["y_validation"] - prediction) ** 2))
        rows.append(
            {
                "validation_year": fold["validation_year"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "off_alpha": float(off_alpha),
                "def_alpha": float(def_alpha),
                "use_prior": bool(use_prior),
                "mse": mse,
                "train_rows": fold["train_rows"],
                "validation_rows": fold["validation_rows"],
                "prior_coverage": fold["coverage"],
                "prior_center_offset_per_100": fold["prior_offset"] * 100.0,
            }
        )
    return rows


def tune_alphas(folds, grid, passes=2, seed_off=2000.0, seed_def=2000.0):
    current_off = float(seed_off)
    current_def = float(seed_def)
    receipt = []
    for pass_index in range(1, int(passes) + 1):
        off_scores = {}
        for candidate in grid:
            rows = evaluate_pair(folds, candidate, current_def, use_prior=True)
            for row in rows:
                row.update({"pass": pass_index, "axis": "off", "candidate_alpha": candidate})
            receipt.extend(rows)
            off_scores[candidate] = float(np.mean([row["mse"] for row in rows]))
        best_off = min(off_scores, key=off_scores.get)
        print(f"pass {pass_index}: best offense alpha={best_off:g}, MSE={off_scores[best_off]:.8f}")

        def_scores = {}
        for candidate in grid:
            rows = evaluate_pair(folds, best_off, candidate, use_prior=True)
            for row in rows:
                row.update({"pass": pass_index, "axis": "def", "candidate_alpha": candidate})
            receipt.extend(rows)
            def_scores[candidate] = float(np.mean([row["mse"] for row in rows]))
        best_def = min(def_scores, key=def_scores.get)
        print(f"pass {pass_index}: best defense alpha={best_def:g}, MSE={def_scores[best_def]:.8f}")

        if best_off == current_off and best_def == current_def:
            current_off, current_def = best_off, best_def
            break
        current_off, current_def = best_off, best_def
    return current_off, current_def, pd.DataFrame(receipt)


def parse_grid(value):
    grid = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not grid or any(item <= 0 for item in grid):
        raise ValueError("alpha grid must contain positive values")
    return grid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-grid", default=DEFAULT_GRID)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed-off", type=float, default=2000.0)
    parser.add_argument("--seed-def", type=float, default=2000.0)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=RESULTS_DIR / "stop_personal_prior_forward_alpha_cv.csv",
    )
    parser.add_argument(
        "--prior-output",
        type=Path,
        default=RESULTS_DIR / "stop_personal_prior_24_26.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    history = fetch_personal_stop_history(2021, 2026)
    fold_specs = [(2021, 2023, 2024), (2022, 2024, 2025), (2023, 2025, 2026)]
    folds = []
    for spec in fold_specs:
        fold = prepare_fold(history, *spec)
        folds.append(fold)
        print(
            f"prepared {spec[0]}-{spec[1]} -> {spec[2]}: "
            f"train={fold['train_rows']:,}, validation={fold['validation_rows']:,}, "
            f"prior coverage={fold['coverage']:.3%}"
        )

    best_off, best_def, receipt = tune_alphas(
        folds,
        parse_grid(args.alpha_grid),
        passes=args.passes,
        seed_off=args.seed_off,
        seed_def=args.seed_def,
    )
    no_prior_rows = evaluate_pair(folds, best_off, best_def, use_prior=False)
    for row in no_prior_rows:
        row.update({"pass": 0, "axis": "benchmark", "candidate_alpha": np.nan})
    receipt = pd.concat([receipt, pd.DataFrame(no_prior_rows)], ignore_index=True)
    receipt["selected_off_alpha"] = best_off
    receipt["selected_def_alpha"] = best_def
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.to_csv(args.receipt, index=False)

    final_frame = load_stop_window(2024, 2026)
    final_players = _all_players(final_frame)
    final_prior = aggregate_personal_stop_prior(history, 2024, 2026)
    centered, offset, coverage = center_prior_on_training_defense(
        final_prior, final_frame, final_players
    )
    prior_lookup = final_prior.set_index(final_prior["nba_id"].astype(str))
    output_rows = []
    for player in final_players:
        source = prior_lookup.loc[player] if player in prior_lookup.index else None
        output_rows.append(
            {
                "player_id": player,
                "player_name": source["name"] if source is not None else f"ID_{player}",
                "prior_value": centered[player] * 100.0,
                "personal_stop_pct": source["personal_stop_pct"] if source is not None else np.nan,
                "relative_stop_pct": source["relative_stop_pct"] if source is not None else np.nan,
                "personal_def_poss": source["def_poss"] if source is not None else 0.0,
                "personal_seasons": source["seasons"] if source is not None else 0,
                "prior_center_offset_per_100": offset * 100.0,
                "selected_off_alpha": best_off,
                "selected_def_alpha": best_def,
                "source_start_year": 2024,
                "source_end_year": 2026,
            }
        )
    pd.DataFrame(output_rows).to_csv(args.prior_output, index=False)

    prior_selected = receipt[
        (receipt["use_prior"])
        & (receipt["off_alpha"] == best_off)
        & (receipt["def_alpha"] == best_def)
    ].drop_duplicates("validation_year", keep="last")
    benchmark = receipt[~receipt["use_prior"]]
    print(f"selected alphas: offense={best_off:g}, defense={best_def:g}")
    print(f"selected prior mean future MSE: {prior_selected['mse'].mean():.8f}")
    print(f"same-alpha no-prior mean future MSE: {benchmark['mse'].mean():.8f}")
    print(f"final prior coverage={coverage:.3%}; wrote {args.prior_output}")
    print(f"wrote alpha receipt {args.receipt}")


if __name__ == "__main__":
    main()

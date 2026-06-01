#!/usr/bin/env python3
"""Two-stage residual opponent-strength models.

Stage 1 is an ordinary RAPM result file with fixed player values. Stage 2
subtracts those fixed values from each possession and fits only opponent-strength
residual terms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge

from run_randle_strong_weak import (
    DEFAULT_DARKO,
    D_COLS,
    O_COLS,
    add_strength_columns,
    grouped_validation_mask,
    load_player_names,
    load_rows,
    player_counts,
    player_universe,
    rmse,
    FitConfig,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_BASE = REPO_ROOT / "nba_pipeline/results/rapm_21_26_all_pure_a2000_4000_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage residual strong/weak RAPM models.")
    parser.add_argument("--start-year", type=int, default=21)
    parser.add_argument("--end-year", type=int, default=26)
    parser.add_argument("--season-type", choices=["RS", "PS", "ALL"], default="ALL")
    parser.add_argument("--base-results", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--darko-history", type=Path, default=DEFAULT_DARKO)
    parser.add_argument("--strength-source", choices=["darko", "base_rapm"], default="darko")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_two_stage")
    parser.add_argument("--residual-alpha-grid", default="250,500,1000,2000,4000,8000")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def fit_config(args: argparse.Namespace) -> FitConfig:
    return FitConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        season_type=args.season_type,
        base_alpha=0.0,
        interaction_alpha_mult=1.0,
        darko_history=args.darko_history,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
    )


def load_base_values(path: Path) -> tuple[dict[int, float], dict[int, float], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"player_id", "off", "def"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    off = dict(zip(df["player_id"].astype(int), pd.to_numeric(df["off"], errors="coerce").fillna(0.0) / 100.0))
    defense = dict(zip(df["player_id"].astype(int), pd.to_numeric(df["def"], errors="coerce").fillna(0.0) / 100.0))
    return off, defense, df


def row_sum(players: np.ndarray, values: dict[int, float]) -> np.ndarray:
    out = np.zeros(players.shape[0], dtype=np.float64)
    for slot in range(players.shape[1]):
        pids = players[:, slot]
        out += np.fromiter((values.get(int(pid), 0.0) if pid > 0 else 0.0 for pid in pids), dtype=np.float64, count=len(pids))
    return out


def add_fixed_base_prediction(rows: pd.DataFrame, base_off: dict[int, float], base_def: dict[int, float]) -> pd.DataFrame:
    rows = rows.copy()
    offense = rows[O_COLS].to_numpy(dtype=np.int64)
    defense = rows[D_COLS].to_numpy(dtype=np.int64)
    rows["base_player_prediction"] = row_sum(offense, base_off) + row_sum(defense, base_def)
    rows["target_mean"] = float(rows["target"].mean())
    rows["stage2_residual"] = rows["target"] - rows["target_mean"] - rows["base_player_prediction"]
    return rows


def add_season_z_scores(rows: pd.DataFrame) -> pd.DataFrame:
    rows["opp_def_strength_z"] = 0.0
    rows["opp_off_strength_z"] = 0.0
    for _, idx in rows.groupby("season").groups.items():
        idx = np.array(list(idx), dtype=np.int64)
        for source, target in [
            ("opp_def_strength", "opp_def_strength_z"),
            ("opp_off_strength", "opp_off_strength_z"),
        ]:
            vals = rows.loc[idx, source].to_numpy(dtype=np.float64)
            std = float(vals.std())
            rows.loc[idx, target] = 0.0 if std == 0.0 else (vals - float(vals.mean())) / std
    return rows


def add_base_rapm_strength_columns(
    rows: pd.DataFrame,
    base_off: dict[int, float],
    base_def: dict[int, float],
) -> pd.DataFrame:
    rows = rows.copy()
    offense = rows[O_COLS].to_numpy(dtype=np.int64)
    defense = rows[D_COLS].to_numpy(dtype=np.int64)

    # Base defensive RAPM is an allowed-points coefficient: lower is better.
    # Flip it so larger strength means a stronger opposing defense.
    rows["opp_def_strength"] = -row_sum(defense, base_def) * 100.0
    rows["opp_off_strength"] = row_sum(offense, base_off) * 100.0
    rows["opp_def_strong"] = rows["opp_def_strength"] > 0.0
    rows["opp_off_strong"] = rows["opp_off_strength"] > 0.0
    return add_season_z_scores(rows)


def make_global_matrix(rows: pd.DataFrame, model: str) -> tuple[sparse.csr_matrix, list[str]]:
    if model == "global_binary":
        data = np.column_stack([
            rows["opp_def_strong"].to_numpy(dtype=np.float64),
            rows["opp_off_strong"].to_numpy(dtype=np.float64),
        ])
        names = ["global_off_vs_strong_def_resid", "global_def_vs_strong_off_resid"]
    elif model == "global_continuous":
        data = np.column_stack([
            rows["opp_def_strength_z"].to_numpy(dtype=np.float64),
            rows["opp_off_strength_z"].to_numpy(dtype=np.float64),
        ])
        names = ["global_off_opp_def_z_slope", "global_def_opp_off_z_slope"]
    else:
        raise ValueError(model)
    return sparse.csr_matrix(data), names


def make_player_matrix(rows: pd.DataFrame, players: list[int], model: str) -> tuple[sparse.csr_matrix, dict[tuple[str, int], int]]:
    lookup: dict[tuple[str, int], int] = {}
    col = 0
    for pid in players:
        if model == "player_binary_delta":
            lookup[("off_strong_delta", pid)] = col
            col += 1
            lookup[("def_strong_delta", pid)] = col
            col += 1
        elif model == "player_continuous_slope":
            lookup[("off_strength_slope", pid)] = col
            col += 1
            lookup[("def_strength_slope", pid)] = col
            col += 1
        else:
            raise ValueError(model)

    offense = rows[O_COLS].to_numpy(dtype=np.int64)
    defense = rows[D_COLS].to_numpy(dtype=np.int64)
    row_idx: list[int] = []
    col_idx: list[int] = []
    vals: list[float] = []

    if model == "player_binary_delta":
        off_values = rows["opp_def_strong"].to_numpy(dtype=np.float64)
        def_values = rows["opp_off_strong"].to_numpy(dtype=np.float64)
        off_key = "off_strong_delta"
        def_key = "def_strong_delta"
    else:
        off_values = rows["opp_def_strength_z"].to_numpy(dtype=np.float64)
        def_values = rows["opp_off_strength_z"].to_numpy(dtype=np.float64)
        off_key = "off_strength_slope"
        def_key = "def_strength_slope"

    for i in range(len(rows)):
        off_val = float(off_values[i])
        def_val = float(def_values[i])
        if off_val:
            for pid in offense[i]:
                key = (off_key, int(pid))
                if pid > 0 and key in lookup:
                    row_idx.append(i)
                    col_idx.append(lookup[key])
                    vals.append(off_val)
        if def_val:
            for pid in defense[i]:
                key = (def_key, int(pid))
                if pid > 0 and key in lookup:
                    row_idx.append(i)
                    col_idx.append(lookup[key])
                    vals.append(def_val)

    return sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(len(rows), col), dtype=np.float64).tocsr(), lookup


def fit_stage2(x: sparse.csr_matrix, y: np.ndarray, train_mask: np.ndarray, alpha: float | None) -> np.ndarray:
    if alpha is None:
        model = LinearRegression(fit_intercept=False)
    else:
        model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", max_iter=5000, tol=1e-6)
    model.fit(x[train_mask], y[train_mask])
    return model.coef_.astype(np.float64)


def export_player_binary(
    beta: np.ndarray,
    lookup: dict[tuple[str, int], int],
    rows: pd.DataFrame,
    counts: pd.DataFrame,
    base_df: pd.DataFrame,
    names: dict[int, str],
    global_beta: np.ndarray | None = None,
) -> pd.DataFrame:
    base_lookup = base_df.set_index("player_id")[["off", "def", "net_rapm"]].to_dict("index")
    global_off_delta = float(global_beta[0] * 100.0) if global_beta is not None else 0.0
    global_def_delta = float(global_beta[1] * 100.0) if global_beta is not None else 0.0
    records = []
    for rec in counts.itertuples(index=False):
        pid = int(rec.player_id)
        base = base_lookup.get(pid, {"off": 0.0, "def": 0.0, "net_rapm": 0.0})
        off_idx = lookup.get(("off_strong_delta", pid))
        def_idx = lookup.get(("def_strong_delta", pid))
        player_off_delta = beta[off_idx] * 100.0 if off_idx is not None else 0.0
        player_def_delta = beta[def_idx] * 100.0 if def_idx is not None else 0.0
        off_delta = global_off_delta + player_off_delta
        def_delta = global_def_delta + player_def_delta
        off_weak = float(base["off"])
        def_weak = float(base["def"])
        off_strong = off_weak + off_delta
        def_strong = def_weak + def_delta
        overall_off = (1.0 - rec.strong_perc_off) * off_weak + rec.strong_perc_off * off_strong
        overall_def = (1.0 - rec.strong_perc_def) * def_weak + rec.strong_perc_def * def_strong
        records.append({
            "player_id": pid,
            "player_name": names.get(pid, f"ID_{pid}"),
            "base_net": float(base["net_rapm"]),
            "stage2_overall_net": overall_off - overall_def,
            "off_vs_weak": off_weak,
            "off_vs_strong": off_strong,
            "off_resid_strong_delta": off_delta,
            "global_off_strong_delta": global_off_delta,
            "player_off_strong_delta": player_off_delta,
            "def_vs_weak": def_weak,
            "def_vs_strong": def_strong,
            "def_resid_strong_delta": def_delta,
            "global_def_strong_delta": global_def_delta,
            "player_def_strong_delta": player_def_delta,
            "net_vs_weak": off_weak - def_weak,
            "net_vs_strong": off_strong - def_strong,
            "net_strong_minus_weak": (off_strong - def_strong) - (off_weak - def_weak),
            "possessions": int(rec.possessions),
            "off_poss": int(rec.off_poss),
            "def_poss": int(rec.def_poss),
            "strong_perc_off": rec.strong_perc_off,
            "strong_perc_def": rec.strong_perc_def,
        })
    return pd.DataFrame(records).sort_values("stage2_overall_net", ascending=False)


def export_player_continuous(
    beta: np.ndarray,
    lookup: dict[tuple[str, int], int],
    counts: pd.DataFrame,
    base_df: pd.DataFrame,
    names: dict[int, str],
    global_beta: np.ndarray | None = None,
) -> pd.DataFrame:
    base_lookup = base_df.set_index("player_id")[["off", "def", "net_rapm"]].to_dict("index")
    global_off_slope = float(global_beta[0] * 100.0) if global_beta is not None else 0.0
    global_def_slope = float(global_beta[1] * 100.0) if global_beta is not None else 0.0
    records = []
    for rec in counts.itertuples(index=False):
        pid = int(rec.player_id)
        base = base_lookup.get(pid, {"off": 0.0, "def": 0.0, "net_rapm": 0.0})
        off_avg = float(base["off"])
        def_avg = float(base["def"])
        off_idx = lookup.get(("off_strength_slope", pid))
        def_idx = lookup.get(("def_strength_slope", pid))
        player_off_slope = beta[off_idx] * 100.0 if off_idx is not None else 0.0
        player_def_slope = beta[def_idx] * 100.0 if def_idx is not None else 0.0
        off_slope = global_off_slope + player_off_slope
        def_slope = global_def_slope + player_def_slope
        observed_off = off_avg + off_slope * rec.observed_opp_def_strength_z
        observed_def = def_avg + def_slope * rec.observed_opp_off_strength_z
        records.append({
            "player_id": pid,
            "player_name": names.get(pid, f"ID_{pid}"),
            "base_net": float(base["net_rapm"]),
            "observed_net": observed_off - observed_def,
            "net_vs_weak_1sd": (off_avg - off_slope) - (def_avg - def_slope),
            "net_vs_avg": off_avg - def_avg,
            "net_vs_strong_1sd": (off_avg + off_slope) - (def_avg + def_slope),
            "net_strong_minus_weak_2sd": 2.0 * (off_slope - def_slope),
            "off_vs_weak_1sd": off_avg - off_slope,
            "off_vs_avg": off_avg,
            "off_vs_strong_1sd": off_avg + off_slope,
            "off_resid_strength_slope": off_slope,
            "global_off_strength_slope": global_off_slope,
            "player_off_strength_slope": player_off_slope,
            "def_vs_weak_1sd": def_avg - def_slope,
            "def_vs_avg": def_avg,
            "def_vs_strong_1sd": def_avg + def_slope,
            "def_resid_strength_slope": def_slope,
            "global_def_strength_slope": global_def_slope,
            "player_def_strength_slope": player_def_slope,
            "observed_opp_def_strength_z": rec.observed_opp_def_strength_z,
            "observed_opp_off_strength_z": rec.observed_opp_off_strength_z,
            "possessions": int(rec.possessions),
            "off_poss": int(rec.off_poss),
            "def_poss": int(rec.def_poss),
            "strong_perc_off": rec.strong_perc_off,
            "strong_perc_def": rec.strong_perc_def,
        })
    return pd.DataFrame(records).sort_values("observed_net", ascending=False)


def write_global_output(path: Path, names: list[str], beta: np.ndarray, train_rmse: float, val_rmse: float) -> None:
    data = {
        "train_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "coefficients_per_100": {name: float(value * 100.0) for name, value in zip(names, beta)},
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def scope_label(args: argparse.Namespace) -> str:
    return f"{args.start_year}_{args.end_year}_{args.season_type.lower()}"


def main() -> None:
    args = parse_args()
    config = fit_config(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    label = scope_label(args)

    rows = load_rows(config)
    base_off, base_def, base_df = load_base_values(args.base_results)
    if args.strength_source == "darko":
        rows = add_strength_columns(rows, config)
    else:
        rows = add_base_rapm_strength_columns(rows, base_off, base_def)
    rows = add_fixed_base_prediction(rows, base_off, base_def)
    players = player_universe(rows)
    counts = player_counts(rows, players)
    names = load_player_names()

    y = rows["stage2_residual"].to_numpy(dtype=np.float64)
    val_mask = grouped_validation_mask(rows["game_id"], args.validation_fraction, args.random_seed)
    train_mask = ~val_mask
    base_only_pred = rows["target_mean"].to_numpy(dtype=np.float64) + rows["base_player_prediction"].to_numpy(dtype=np.float64)
    target = rows["target"].to_numpy(dtype=np.float64)
    base_train_rmse = rmse(target[train_mask], base_only_pred[train_mask])
    base_val_rmse = rmse(target[val_mask], base_only_pred[val_mask])

    diagnostics = {
        "config": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "season_type": args.season_type,
            "base_results": str(args.base_results),
            "darko_history": str(args.darko_history),
            "strength_source": args.strength_source,
            "output_dir": str(output_dir),
            "residual_alpha_grid": args.residual_alpha_grid,
            "validation_fraction": args.validation_fraction,
            "random_seed": args.random_seed,
        },
        "rows": int(len(rows)),
        "players": int(len(players)),
        "base_only_train_rmse": base_train_rmse,
        "base_only_validation_rmse": base_val_rmse,
        "residual_mean": float(y.mean()),
        "residual_std": float(y.std()),
        "models": {},
    }

    global_fits: dict[str, tuple[sparse.csr_matrix, np.ndarray]] = {}
    for model_name in ["global_binary", "global_continuous"]:
        x, feature_names = make_global_matrix(rows, model_name)
        beta = fit_stage2(x, y, train_mask, None)
        global_fits[model_name] = (x, beta)
        resid_pred = x @ beta
        train_rmse = rmse(target[train_mask], base_only_pred[train_mask] + resid_pred[train_mask])
        val_rmse = rmse(target[val_mask], base_only_pred[val_mask] + resid_pred[val_mask])
        path = output_dir / f"{model_name}_{label}.json"
        write_global_output(path, feature_names, beta, train_rmse, val_rmse)
        diagnostics["models"][model_name] = {
            "features": int(x.shape[1]),
            "train_rmse": train_rmse,
            "validation_rmse": val_rmse,
            "improvement_vs_base": base_val_rmse - val_rmse,
            "output": str(path),
        }

    alpha_grid = [float(x.strip()) for x in args.residual_alpha_grid.split(",") if x.strip()]
    player_model_specs = [
        ("player_binary_delta", None),
        ("player_binary_after_global", "global_binary"),
        ("player_continuous_slope", None),
        ("player_continuous_after_global", "global_continuous"),
    ]
    for model_name, global_model_name in player_model_specs:
        matrix_model_name = "player_binary_delta" if "binary" in model_name else "player_continuous_slope"
        global_x = None
        global_beta = None
        global_pred = np.zeros(len(rows), dtype=np.float64)
        y_for_player = y
        if global_model_name is not None:
            global_x, global_beta = global_fits[global_model_name]
            global_pred = global_x @ global_beta
            y_for_player = y - global_pred
        x, lookup = make_player_matrix(rows, players, matrix_model_name)
        diagnostics["models"][model_name] = {"features": int(x.shape[1]), "alpha_runs": {}}
        for alpha in alpha_grid:
            beta = fit_stage2(x, y_for_player, train_mask, alpha)
            resid_pred = x @ beta
            combined_pred = global_pred + resid_pred
            train_rmse = rmse(target[train_mask], base_only_pred[train_mask] + combined_pred[train_mask])
            val_rmse = rmse(target[val_mask], base_only_pred[val_mask] + combined_pred[val_mask])
            beta_full = fit_stage2(x, y_for_player, np.ones(len(rows), dtype=bool), alpha)
            alpha_label = f"a{int(alpha)}" if alpha.is_integer() else f"a{alpha:g}"
            if "binary" in model_name:
                out = export_player_binary(beta_full, lookup, rows, counts, base_df, names, global_beta=global_beta)
            else:
                out = export_player_continuous(beta_full, lookup, counts, base_df, names, global_beta=global_beta)
            out_path = output_dir / f"{model_name}_{label}_{alpha_label}.csv"
            out.round(4).to_csv(out_path, index=False)
            diagnostics["models"][model_name]["alpha_runs"][alpha_label] = {
                "alpha": alpha,
                "train_rmse": train_rmse,
                "validation_rmse": val_rmse,
                "improvement_vs_base": base_val_rmse - val_rmse,
                "output": str(out_path),
            }

    diag_path = output_dir / f"diagnostics_{label}.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n")

    rows_out = []
    for model_name, info in diagnostics["models"].items():
        if "alpha_runs" in info:
            for alpha_label, run in info["alpha_runs"].items():
                rows_out.append({
                    "model": model_name,
                    "alpha": run["alpha"],
                    "validation_rmse": run["validation_rmse"],
                    "improvement_vs_base": run["improvement_vs_base"],
                    "output": run["output"],
                })
        else:
            rows_out.append({
                "model": model_name,
                "alpha": None,
                "validation_rmse": info["validation_rmse"],
                "improvement_vs_base": info["improvement_vs_base"],
                "output": info["output"],
            })
    validation = pd.DataFrame(rows_out).sort_values(["validation_rmse", "model"])
    validation.to_csv(output_dir / f"validation_summary_{label}.csv", index=False)
    validation.to_csv(output_dir / "validation_summary.csv", index=False)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

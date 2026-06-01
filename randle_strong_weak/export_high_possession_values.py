#!/usr/bin/env python3
"""Export player values for a high-possession DARKO continuous residual cut."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from run_randle_strong_weak import (  # noqa: E402
    DEFAULT_DARKO,
    FitConfig,
    add_strength_columns,
    grouped_validation_mask,
    load_player_names,
    load_rows,
    player_counts,
    player_universe,
    rmse,
)
from run_two_stage_residual import (  # noqa: E402
    add_fixed_base_prediction,
    export_player_continuous,
    fit_stage2,
    load_base_values,
    make_global_matrix,
    make_player_matrix,
)


ROOT = Path(__file__).resolve().parent
BASE_RESULTS = ROOT.parent / "nba_pipeline/results/rapm_21_26_all_pure_a2000_4000_results.csv"
OUTPUT_DIR = ROOT / "outputs_two_stage"
OUT_CSV = OUTPUT_DIR / "player_continuous_highpos40000_after_global_a32000.csv"
OUT_SUMMARY = OUTPUT_DIR / "player_continuous_highpos40000_after_global_a32000_summary.json"


def main() -> None:
    config = FitConfig(
        start_year=21,
        end_year=26,
        season_type="ALL",
        base_alpha=0.0,
        interaction_alpha_mult=1.0,
        darko_history=DEFAULT_DARKO,
        validation_fraction=0.2,
        random_seed=42,
    )
    base_off, base_def, base_df = load_base_values(BASE_RESULTS)
    rows = add_strength_columns(load_rows(config), config)
    rows = add_fixed_base_prediction(rows, base_off, base_def)
    players = player_universe(rows)
    counts = player_counts(rows, players)
    count_by_pid = dict(zip(counts["player_id"].astype(int), counts["possessions"].astype(int)))
    qualified = [pid for pid in players if count_by_pid.get(pid, 0) >= 40000]

    y = rows["stage2_residual"].to_numpy(dtype=np.float64)
    target = rows["target"].to_numpy(dtype=np.float64)
    base_pred = rows["target_mean"].to_numpy(dtype=np.float64) + rows["base_player_prediction"].to_numpy(dtype=np.float64)
    val_mask = grouped_validation_mask(rows["game_id"], 0.2, 42)
    train_mask = ~val_mask

    global_x, _ = make_global_matrix(rows, "global_continuous")
    global_beta = fit_stage2(global_x, y, train_mask, None)
    global_pred = global_x @ global_beta

    player_x, lookup = make_player_matrix(rows, qualified, "player_continuous_slope")
    beta = fit_stage2(player_x, y - global_pred, np.ones(len(rows), dtype=bool), 32000.0)
    out = export_player_continuous(
        beta,
        lookup,
        counts,
        base_df,
        load_player_names(),
        global_beta=global_beta,
    )
    out["qualified_for_player_slope"] = out["player_id"].isin(qualified)
    out.round(4).to_csv(OUT_CSV, index=False)

    beta_val = fit_stage2(player_x, y - global_pred, train_mask, 32000.0)
    val_pred = base_pred + global_pred + player_x @ beta_val
    summary = {
        "model": "DARKO continuous global + high-pos player residual",
        "min_possessions_for_player_slope": 40000,
        "qualified_players": len(qualified),
        "alpha": 32000,
        "validation_rmse": rmse(target[val_mask], val_pred[val_mask]),
        "output": str(OUT_CSV),
    }
    OUT_SUMMARY.write_text(pd.Series(summary).to_json(indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()

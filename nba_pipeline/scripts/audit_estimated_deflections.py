"""Post-freeze exposure sensitivity; no model fitting or selection."""
import json
import pandas as pd
from build_estimated_deflections import OUT, BASELINES, paired


def main():
    predictions = pd.read_parquet(OUT/"predictions.parquet")
    predictions["prior_exposure"] = predictions.groupby("player_id").exposure.cumsum()
    cells = pd.read_parquet(OUT/"stress_cells.parquet")
    parts = []
    for date, g in cells.groupby("asof"):
        past = predictions[predictions.game_date.lt(date)].groupby("player_id").tail(1)
        parts.append(g.merge(past[["player_id", "prior_exposure"]], on="player_id", validate="one_to_one"))
    joined = pd.concat(parts, ignore_index=True)
    selected = json.loads((OUT/"frozen.json").read_text())["selected"]
    other = "random_walk" if selected == "ou" else "ou"
    sensitivity = joined[joined.prior_exposure.ge(200)].copy()
    results = [paired(sensitivity, selected, b) for b in [*BASELINES, other]]
    weights = joined.e/joined.groupby("asof").e.transform("sum")
    joined["career_error_contribution"] = weights*(joined.career-joined.target)**2
    worst = joined.nlargest(10, "career_error_contribution")
    worst = worst.merge(predictions[["player_id", "player_name"]].drop_duplicates("player_id"), on="player_id")
    report = {"label": "post-freeze diagnostic; no retuning or winner switching",
        "minimum_prior_defensive_possessions": 200,
        "original_cells": len(joined), "retained_cells": len(sensitivity), "metrics": results,
        "largest_naive_baseline_errors": worst[["player_id", "player_name", "asof", "prior_exposure", "career", "target"]].astype({"asof": str}).to_dict("records")}
    (OUT/"exposure_sensitivity.json").write_text(json.dumps(report, indent=2))
    sensitivity.to_parquet(OUT/"exposure_sensitivity_cells.parquet", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

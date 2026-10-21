"""Whole-panel chronology, identity, source-era and bounds gates before sharing."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_full_skill_history import write_json
from full_skill_history_specs import SKILLS
from prepare_full_skill_history import ROOT, sha


def audit(out=ROOT):
    out = Path(out)
    source = pd.read_parquet(out / "observations.parquet", columns=["player_id", "game_id", "season", "day"])
    lock = sha(out / "run_lock.json")
    rows = 0
    for year in range(1997, 2027):
        values = pd.read_parquet(out / "baseline" / f"{year}.parquet")
        expected = source.loc[source.season.eq(year)].reset_index(drop=True)
        if not np.array_equal(values[["player_id", "game_id"]].to_numpy(), expected[["player_id", "game_id"]].to_numpy()):
            raise ValueError("Historical output identity/census changed")
        for spec in SKILLS:
            model = json.loads((out / "models" / str(year) / (spec.name + ".json")).read_text())
            if model["run_lock_sha256"] != lock or model["training_end_season"] >= year:
                raise ValueError("Annual model source/date fence failed")
            if model.get("last_training_observation", "0000") >= model["target_first_date"]:
                raise ValueError("Annual model observation date reaches a target game")
            last = values[spec.name + "__last_day"].to_numpy()
            if not np.all((last == -1) | (last < expected.day.to_numpy())):
                raise ValueError("Pregame state includes a current/future measurement")
            p = values[spec.name].dropna()
            if not np.isfinite(p).all() or not p.ge(0).all() or (spec.kind == "pct" and not p.le(1).all()):
                raise ValueError("Forecast bounds failed")
            if year < spec.first_season or year > spec.last_season:
                if len(p) or values[spec.name + "__observed"].any():
                    raise ValueError("Unsupported source epoch entered a forecast or update")
        rows += len(values)
    if rows != len(source):
        raise ValueError("Historical outputs lost or repeated source rows")
    write_json(out / "chronology_audit.json", dict(status="passed", rows=rows, years=30, model_heads=len(SKILLS),
                                                  source_lock_sha256=lock, current_or_future_measurement_violations=0,
                                                  source_epoch_violations=0, model_date_violations=0))
    print(json.dumps(dict(status="passed", rows=rows, model_heads=len(SKILLS))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    audit(parser.parse_args().out)

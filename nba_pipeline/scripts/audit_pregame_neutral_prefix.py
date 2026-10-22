"""Real-data perturbation audit of the complete neutral core adapter.

Counterfactual copies are diagnostic only and never replace source files or
production exports. Frozen annual parameter vintages are held fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_pregame_neutral_skills as neutral
from build_full_skill_history import write_json


def audit(source, out, year=2019, cutoff="2020-01-22"):
    source, out, cutoff = Path(source).resolve(), Path(out), pd.Timestamp(cutoff)
    original_read = pd.read_parquet
    records = []
    for skill in ["ft_pct", "three_pct", "ast_rate"]:
        values = []
        for mutate in [False, True]:
            folder = out / ("counterfactual" if mutate else "reference")
            neutral.prepare(folder, source)
            def read(path, *args, **kwargs):
                frame = original_read(path, *args, **kwargs)
                if mutate and Path(path).resolve() == source / "player_game_lines.parquet":
                    frame = frame.copy()
                    later = pd.to_datetime(frame.game_date).ge(cutoff)
                    frame.loc[later, ["ftm", "three_fgm", "ast"]] = 0.
                return frame
            pd.read_parquet = read
            try:
                neutral.one((str(folder), year, skill))
            finally:
                pd.read_parquet = original_read
            values.append(original_read(folder / "families" / str(year) / (skill + ".parquet")))
        a, b = values
        assert np.array_equal(a[["player_id", "game_id"]].values, b[["player_id", "game_id"]].values)
        earlier = a.game_date.le(cutoff)
        gap = float((a.loc[earlier, "value"] - b.loc[earlier, "value"]).abs().max())
        if gap != 0:
            raise ValueError(f"Target/future outcomes changed {skill} pregame states: {gap}")
        future = a.game_date.gt(cutoff)
        future_gap = float((a.loc[future, "value"] - b.loc[future, "value"]).abs().max())
        if future_gap == 0:
            raise ValueError("Perturbation did not exercise future state updates")
        records.append(dict(skill=skill, unchanged_rows=int(earlier.sum()), max_prefix_gap=gap,
                            later_state_max_change=future_gap))
    write_json(out / "receipt.json", dict(status="passed_complete_core_adapter_prefix_audit", cutoff=str(cutoff.date()),
                annual_vintage=year, records=records, original_sources_modified=False, counterfactual_exports_admitted=False))
    print(json.dumps(dict(status="passed_complete_core_adapter_prefix_audit", records=records)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=neutral.ROOT / "results/estimated_skills/pregame_source_validated_v1")
    parser.add_argument("--out", type=Path, default=neutral.ROOT / "results/estimated_skills/pregame_neutral_prefix_v1")
    args = parser.parse_args()
    audit(args.source, args.out)

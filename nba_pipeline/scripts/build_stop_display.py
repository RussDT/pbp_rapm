#!/usr/bin/env python3
"""Build a reader-ready positive-good STOP RAPM CSV."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PIPELINE_ROOT / "results" / "stop_24_26_all_se_cv_a2000_2000_results.csv"
)
DEFAULT_OUTPUT = (
    PIPELINE_ROOT / "results" / "stop_24_26_all_se_cv_a2000_2000_display.csv"
)


def build_stop_display(results, identity=None):
    """Rename sides, flip defense once, and preserve exact net closure."""
    required = {
        "player_id",
        "player_name",
        "off",
        "def",
        "net_rapm",
        "possessions",
        "off_poss",
        "def_poss",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"STOP results missing columns: {missing}")

    output = results.copy()
    output["nba_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    output["name"] = output["player_name"].astype(str)
    if identity is not None and not identity.empty:
        names = identity.copy()
        names["nba_id"] = pd.to_numeric(names["nba_id"], errors="coerce")
        names = names.dropna(subset=["nba_id", "name"]).copy()
        names["nba_id"] = names["nba_id"].astype(int)
        names = names.drop_duplicates("nba_id", keep="last")
        name_map = names.set_index("nba_id")["name"]
        output["name"] = output["nba_id"].map(name_map).fillna(output["name"])

    output["oSTOP"] = pd.to_numeric(output["off"], errors="raise")
    output["dSTOP"] = -pd.to_numeric(output["def"], errors="raise")
    output["STOP"] = pd.to_numeric(output["net_rapm"], errors="raise")
    gap = (output["oSTOP"] + output["dSTOP"] - output["STOP"]).abs().max()
    if not np.isfinite(gap) or gap > 0.0011:
        raise ValueError(f"STOP display identity failed: max gap={gap}")

    return output[
        [
            "nba_id",
            "name",
            "oSTOP",
            "dSTOP",
            "STOP",
            "possessions",
            "off_poss",
            "def_poss",
        ]
    ].sort_values(["STOP", "possessions"], ascending=[False, False])


def load_identity():
    """Load the newest available full-name map from repo-owned EPM artifacts."""
    frames = []
    player_values = PIPELINE_ROOT / "results" / "epm_game_value" / "player_values_26.csv"
    if player_values.exists():
        frame = pd.read_csv(player_values, usecols=["nba_id", "player_name"])
        frames.append(frame.rename(columns={"player_name": "name"}))

    epm_files = sorted((PIPELINE_ROOT / "cache" / "epm").glob("epm_all_*.csv"))
    for path in epm_files:
        frame = pd.read_csv(path, usecols=["player_id", "player_name"])
        frames.append(frame.rename(columns={"player_id": "nba_id", "player_name": "name"}))
    if not frames:
        return pd.DataFrame(columns=["nba_id", "name"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("nba_id", keep="last")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    display = build_stop_display(pd.read_csv(args.input), load_identity())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(args.output, index=False)
    print(f"Wrote {len(display):,} players to {args.output}")


if __name__ == "__main__":
    main()

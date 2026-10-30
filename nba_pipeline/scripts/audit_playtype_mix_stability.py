#!/usr/bin/env python3
"""Disjoint-window stability receipt for the two-season TD550 playtype-mix RAPM.

The DeRozan question is whether a player's teammates take a harder shot diet
because of *him*, or because of the roster around him. Each pair of windows
shares no seasons, so any agreement between them is not shared noise. The
receipt reports correlations for every player with enough exposure in both
windows, and again for "movers": players whose frequent-teammate sets barely
overlap across the two windows (Jaccard < 0.15). If the teammate-diet signal
held only for players who kept their roster, it would be a roster effect; if it
holds for movers too, it travels with the player.

Uses the attempt panel written by ``build_td_playtype_mix_board.py``; the solve
is the same two-season, 550-day half-life lineup model at the board's alphas.
Each window spans two seasons, matching the reader board's history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import analyze_td_overall_playtype_mix_ots as mix


PIPELINE_DIR = Path(__file__).resolve().parents[1]
BOARD_DIR = PIPELINE_DIR / "results" / "td_playtype_mix_board"
WINDOW_PAIRS = [
    ((2003, 2004), (2006, 2007)),
    ((2010, 2011), (2013, 2014)),
    ((2017, 2018), (2020, 2021)),
    ((2022, 2023), (2025, 2026)),
]
FIELDS = {
    "td_overall_playtype_difficulty_rapm": "mix_rapm",
    "conditional_own_playtype_difficulty": "own_diet",
    "conditional_teammate_playtype_difficulty": "teammate_diet",
    "teammate_mix_contribution": "teammate_mix_contribution",
}
MIN_EXPOSURE = 1500.0
MIN_SHARED_FGA = 300
MOVER_JACCARD = 0.15


def solve(attempts: pd.DataFrame, names: pd.DataFrame, lo: int, hi: int, args: argparse.Namespace) -> pd.DataFrame:
    result, _, _ = mix.build_td_results(
        attempts[attempts["Season"].between(lo, hi)].copy(),
        feature_season=hi,
        half_life_days=args.half_life,
        off_alpha=args.off_alpha,
        def_alpha=args.def_alpha,
        role_alpha=args.role_alpha,
        names=names,
    )
    return result


def frequent_teammates(attempts: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    """Set of teammates sharing at least MIN_SHARED_FGA lineup FGAs with each player."""
    frame = attempts[attempts["Season"].between(lo, hi)]
    long = frame.melt(id_vars=["Focal_Player"], value_vars=["O1", "O2", "O3", "O4"], value_name="tm")
    pairs = pd.concat(
        [
            long[["Focal_Player", "tm"]].rename(columns={"Focal_Player": "p"}),
            long[["tm", "Focal_Player"]].rename(columns={"tm": "p", "Focal_Player": "tm"}),
        ]
    )
    counts = pairs.groupby(["p", "tm"]).size().rename("n").reset_index()
    return counts[counts["n"] >= MIN_SHARED_FGA].groupby("p")["tm"].apply(set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--board-dir", type=Path, default=BOARD_DIR)
    parser.add_argument("--half-life", type=float, default=550.0)
    parser.add_argument("--off-alpha", type=float, default=3000.0)
    parser.add_argument("--def-alpha", type=float, default=5000.0)
    parser.add_argument("--role-alpha", type=float, default=10000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attempts = pd.read_parquet(args.board_dir / "attempt_panel_1997_2026.parquet")
    names = (
        pd.read_csv(args.board_dir / "td_playtype_mix_board_1997_2026.csv")[["nba_id", "player_name"]]
        .drop_duplicates("nba_id")
        .rename(columns={"nba_id": "player_id"})
    )
    rows = []
    for (a, b), (c, d) in WINDOW_PAIRS:
        first, second = solve(attempts, names, a, b, args), solve(attempts, names, c, d, args)
        mates_first, mates_second = frequent_teammates(attempts, a, b), frequent_teammates(attempts, c, d)
        both = first.merge(second, on="player_id", suffixes=("_a", "_b"))
        both = both[
            both["weighted_offensive_fga_exposure_a"].gt(MIN_EXPOSURE)
            & both["weighted_offensive_fga_exposure_b"].gt(MIN_EXPOSURE)
        ]
        jaccard = both["player_id"].map(
            lambda p: len(mates_first.get(p, set()) & mates_second.get(p, set()))
            / max(1, len(mates_first.get(p, set()) | mates_second.get(p, set())))
        )
        for group, mask in [("all", jaccard.ge(0)), ("movers", jaccard.lt(MOVER_JACCARD))]:
            subset = both[mask]
            row = {"window_a": f"{a}-{b}", "window_b": f"{c}-{d}", "group": group, "players": int(len(subset))}
            for source, label in FIELDS.items():
                row[f"corr_{label}"] = round(float(subset[f"{source}_a"].corr(subset[f"{source}_b"])), 3)
            rows.append(row)
            print(row)
    receipt = pd.DataFrame(rows)
    receipt.to_csv(args.board_dir / "disjoint_window_stability.csv", index=False)
    (args.board_dir / "disjoint_window_stability.json").write_text(
        json.dumps(
            {
                "min_exposure_each_window": MIN_EXPOSURE,
                "frequent_teammate_min_shared_fga": MIN_SHARED_FGA,
                "mover_rule": f"frequent-teammate Jaccard < {MOVER_JACCARD}",
                "alphas": {"off": args.off_alpha, "def": args.def_alpha, "role": args.role_alpha},
                "half_life_days": args.half_life,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

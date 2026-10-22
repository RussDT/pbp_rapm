"""Adjudicate native actual-point totals against independent terminal scoreboards.

Player-point sums can omit unattributed scoring events. They are a useful
cross-check, not a substitute for the final scoreboard. No model is fitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_full_skill_history import write_json
from prepare_full_skill_history import sha

ROOT = Path(__file__).resolve().parents[1]


def normal(values):
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def terminal_scores(frame):
    d = frame.copy()
    d["game_id"] = normal(d.game_id)
    split = d.score.astype(str).str.extract(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$")
    d["score_total"] = pd.to_numeric(split[0], errors="coerce") + pd.to_numeric(split[1], errors="coerce")
    end = d.sort_values(["game_id", "period", "minute_game", "event_num"], kind="stable").groupby("game_id", sort=False).tail(1)
    expected_clock = 48 + (end.period - 4) * 5
    good = end.period.ge(4) & (end.minute_game - expected_clock).abs().le(.001) & end.score_total.notna()
    end["terminal_score"] = end.score_total.where(good)
    return end[["game_id", "terminal_score"]]


def run(out):
    out = Path(out)
    if (out / "receipt.json").exists():
        raise ValueError("Scoreboard audit exists; preserve its receipt")
    atoms = ROOT / "results/monthly_vpm_targets_clean_sq_v2"
    source = ROOT / "results/estimated_skills/pregame_source_validated_v1/player_game_lines.parquet"
    ledger = pd.read_parquet(source, columns=["game_id", "pts"]).groupby("game_id").pts.sum()
    records, hashes = [], {str(source): sha(source), str(Path(__file__)): sha(__file__)}
    for year in range(2016, 2027):
        for phase, suffix in [("RS", ""), ("PS", "_PS")]:
            raw_file = ROOT / "raw_data" / f"NBA{year % 100:02d}{suffix}.parquet"
            target_file = atoms / f"atoms_{year}_{phase}.parquet"
            raw = pd.read_parquet(raw_file, columns=["game_id", "event_num", "period", "minute_game", "score"])
            final = terminal_scores(raw).set_index("game_id")
            target = pd.read_parquet(target_file, columns=["game_id", "actual_points"])
            target.game_id = normal(target.game_id)
            actual = target.groupby("game_id").actual_points.sum().rename("native_points")
            result = actual.to_frame().join(final, how="left")
            result["player_points"] = ledger.reindex(result.index)
            result["season"] = year
            result["phase"] = phase
            result["native_scoreboard_gap"] = result.native_points - result.terminal_score
            result["player_scoreboard_gap"] = result.player_points - result.terminal_score
            result["native_matches_scoreboard"] = result.native_scoreboard_gap.abs().le(1e-8)
            records.append(result.reset_index())
            hashes.update({str(file): sha(file) for file in [raw_file, target_file]})
    result = pd.concat(records, ignore_index=True)
    if result.game_id.duplicated().any():
        raise ValueError("A native game was counted twice")
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "games.csv", index=False)
    summary = dict(status="completed_independent_scoreboard_audit", games=len(result),
        native_matches_scoreboard=int(result.native_matches_scoreboard.sum()),
        terminal_score_unresolved=int(result.terminal_score.isna().sum()),
        native_scoreboard_mismatch=int((result.terminal_score.notna() & ~result.native_matches_scoreboard).sum()),
        player_points_disagree_but_native_matches=int((result.player_scoreboard_gap.abs().gt(1e-8) & result.native_matches_scoreboard).sum()),
        protected_2027_rows=0, model_fitted=False, output_sha256=sha(out / "games.csv"), source_hashes=hashes)
    write_json(out / "receipt.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/estimated_skills/pregame_scoreboard_audit_v1")
    run(parser.parse_args().out)

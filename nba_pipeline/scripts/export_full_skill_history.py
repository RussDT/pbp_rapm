"""Coherent pregame-only views of the separate full-history skills library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_full_skill_history import write_json
from full_skill_history_specs import SKILLS, canonical_aliases
from prepare_full_skill_history import ROOT, sha


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)


def derived(frame):
    result = canonical_aliases(frame)
    zones = ["at_rim", "short_midrange", "long_midrange", "corner3", "arc3"]
    for zone in zones:
        result[zone + "_fgm100"] = result[zone + "_fga100"] * result[zone + "_pct"]
        result[zone + "_assisted_fgm100"] = result[zone + "_fgm100"] * result[zone + "_assisted_make_share"]
        result[zone + "_unassisted_fgm100"] = result[zone + "_fgm100"] - result[zone + "_assisted_fgm100"]
    result["rim_fga100"] = result.at_rim_fga100
    result["rim_pct"] = result.at_rim_pct
    for tag, children in [("mid", ["short_midrange", "long_midrange"]), ("three", ["corner3", "arc3"]), ("two", zones[:3])]:
        for stat in ["fga100", "fgm100", "assisted_fgm100", "unassisted_fgm100"]:
            result[tag + "_" + stat] = sum(result[z + "_" + stat].astype(float) for z in children)
        result[tag + "_pct"] = ratio(result[tag + "_fgm100"].to_numpy(), result[tag + "_fga100"].to_numpy())
    result["fga100"] = sum(result[z + "_fga100"].astype(float) for z in zones)
    result["fgm100"] = sum(result[z + "_fgm100"].astype(float) for z in zones)
    result["ftm100"] = result.fta100.astype(float) * result.ft_pct.astype(float)
    result["points100"] = 2 * result.two_fgm100 + 3 * result.three_fgm100 + result.ftm100
    result["efg_pct"] = ratio((result.fgm100 + .5 * result.three_fgm100).to_numpy(), result.fga100.to_numpy())
    result["approx_tsa100"] = result.fga100 + .44 * result.fta100
    result["approx_ts_pct"] = ratio(result.points100.to_numpy(), (2 * result.approx_tsa100).to_numpy())
    result["ast100"] = sum(result["assists_" + z + "100"].astype(float) for z in zones)
    turnover_shares = ["tov_" + k + "_share" for k in ["bad_pass_any", "lost_ball_any", "traveling", "remainder"]]
    turnover_total = result[turnover_shares].sum(axis=1, min_count=4).to_numpy()
    for column in turnover_shares:
        result[column + "__before_composition"] = result[column]
        result[column] = ratio(result[column].to_numpy(), turnover_total)
        result[column.replace("_share", "100")] = result.tov100 * result[column]
    result["tov_bad_pass_oob100"] = result.tov_bad_pass_any100 * result.tov_bad_pass_oob_share
    result["tov_lost_ball_oob100"] = result.tov_lost_ball_any100 * result.tov_lost_ball_oob_share
    result["usage_events100"] = result.fga100 + .44 * result.fta100 + result.tov100
    fouls = ["shooting", "offensive", "technical", "flagrant", "loose_ball", "double", "personal_group"]
    result["all_fouls100_total_possessions"] = sum(result["foul_" + k + "100"].astype(float) for k in fouls)
    result = result.copy()
    for reb in ["oreb", "dreb"]:
        children = [reb + "_after_" + k + "_share" for k in ["2pt", "3pt", "ft", "other"]]
        weights = result[children].sum(axis=1, min_count=len(children)).to_numpy()
        for child in children:
            share = ratio(result[child].to_numpy(), weights)
            result[child + "__before_composition"] = result[child]
            result[child] = share
            result[child.replace("_share", "100")] = result[reb + "100"] * share
    for stem in ["catch_shoot", "pull_up"]:
        for side in ["two", "three"]:
            result[stem + "_" + side + "_fgm100"] = result[stem + "_" + side + "_fga100"] * result[stem + "_" + side + "_pct"]
        result[stem + "_fga100"] = result[stem + "_two_fga100"] + result[stem + "_three_fga100"]
        result[stem + "_fgm100"] = result[stem + "_two_fgm100"] + result[stem + "_three_fgm100"]
        result[stem + "_pct"] = ratio(result[stem + "_fgm100"].to_numpy(), result[stem + "_fga100"].to_numpy())
    for stem in ["overall_def", "two_pt_def", "three_pt_def", "less_6ft_def", "less_10ft_def", "more_15ft_def"]:
        result[stem + "_fgm100"] = result[stem + "_fga100"] * result[stem + "_fg_pct"]
    result["drive_fgm100"] = result.drive_fga100 * result.drive_pct
    return result


def audit(frame):
    for zone in ["at_rim", "short_midrange", "long_midrange", "corner3", "arc3"]:
        finite = frame[zone + "_fgm100"].notna()
        if not frame.loc[finite, zone + "_fgm100"].le(frame.loc[finite, zone + "_fga100"] + 1e-6).all():
            raise ValueError("Predicted makes exceed predicted attempts")
        gap = frame[zone + "_fgm100"] - frame[zone + "_assisted_fgm100"] - frame[zone + "_unassisted_fgm100"]
        if gap.abs().max() > 1e-5:
            raise ValueError("Assisted/unassisted expected makes do not close")
    gap = frame.points100 - (2 * frame.two_fgm100 + 3 * frame.three_fgm100 + frame.ftm100)
    if gap.abs().max() > 1e-8:
        raise ValueError("Point accounting does not close")
    return float(gap.abs().max())


def run(out=ROOT):
    out = Path(out)
    records = []
    expected = json.loads((out / "prepare_receipt.json").read_text())["rows"]
    for profile, folder in [("independent", out / "baseline"), ("cross_stat_research", out / "cross_stat/predictions")]:
        total = 0
        for year in range(1997, 2027):
            source = folder / f"{year}.parquet"
            frame = pd.read_parquet(source, columns=["player_id", "game_id", "game_date", "season", *[s.name for s in SKILLS]])
            frame = derived(frame)
            gap = audit(frame)
            target = out / "library" / profile / f"{year}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target, index=False)
            total += len(frame)
            records.append(dict(profile=profile, year=year, rows=len(frame), columns=len(frame.columns), point_gap=gap, sha256=sha(target)))
        if total != expected:
            raise ValueError("Final library does not cover the complete observation census")
    # Inspectability: current research values at each player's latest actual
    # game cutoff, rather than silently relabeling stale states as today's.
    recent = pd.read_parquet(out / "library/cross_stat_research/2026.parquet")
    recent = recent.sort_values(["game_date", "game_id"]).groupby("player_id", sort=False).tail(1)
    recent.to_csv(out / "latest_2026_pregame_values.csv", index=False)
    write_json(out / "library_receipt.json", dict(status="completed_research_library", profiles=["independent", "cross_stat_research"],
                                                 code_sha256=sha(__file__), baseline_lock_sha256=sha(out / "run_lock.json"),
                                                 cross_stat_lock_sha256=sha(out / "cross_stat/run_lock.json"),
                                                 model_targets=len(SKILLS), rows_per_profile=expected, records=records,
                                                 data_through="2026", protected_2027_rows=0, selected_vpm_changed=False,
                                                 context="context_free_game_rates", neutral_causal_talent_identified=False,
                                                 tsa="approximate FGA + .44 FTA", usage="usage events, not NBA USG%",
                                                 head_scores="before deterministic derived products and rebound composition normalization",
                                                 cross_stat_joint_makes_gate_passed=False))
    print(json.dumps(dict(status="completed_research_library", rows_per_profile=expected, columns=len(recent.columns))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    run(parser.parse_args().out)

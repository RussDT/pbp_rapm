"""Core-only neutral replay source with pinned identity/possession validation.

Preserve the old PF-counter definition on exact valid identities. Do not replace
it with a different sum of foul event types. The new source is a separate
estimator change; legacy annual OU parameters are transported, not refitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_full_skill_history import ROOT as HISTORY, read_table, sha
from build_full_skill_history import write_json
from build_monthly_own_skills import SOURCE as LEGACY

ROOT = HISTORY.parent / "pregame_source_validated_v1"


def build(out=ROOT):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise ValueError("Validated source exists; verify it rather than overwrite")
    core, source_manifest = read_table(HISTORY / "source", "player_game")
    core = core.loc[core.season.ge(2014)].copy()
    old = pd.read_parquet(LEGACY / "player_game_lines.parquet")
    old["game_id"] = pd.to_numeric(old.game_id).astype(np.int64)
    if old.duplicated(["player_id", "game_id"]).any():
        raise ValueError("Legacy source repeats identities")
    keys = core[["game_id", "person_id"]].rename(columns={"person_id": "player_id"})
    census = old.merge(keys.assign(validated_identity=True), on=["game_id", "player_id"], how="left", validate="one_to_one")
    positive = census.off_poss.gt(0) | census.def_poss.gt(0) | census.fga.gt(0) | census.fta.gt(0)
    foreign = census.loc[positive & census.validated_identity.isna()].copy()
    result = keys.copy()
    for name in ["season", "team_id", "opponent_id", "is_home", "fga", "fgm", "fta", "ftm", "pts", "tov", "oreb", "dreb", "ast", "stl", "blk"]:
        result[name] = core[name].to_numpy()
    for zone, children in [("rim", ["at_rim"]), ("mid", ["short_midrange", "long_midrange"]), ("three", ["corner3", "arc3"])]:
        for stat in ["fga", "fgm"]:
            result[zone + "_" + stat] = sum(core[z + "_" + stat].to_numpy() for z in children)
    good = core.lineups_clean.eq(True) & core.provably_wrong_rows.eq(0) & core.rows_without_10_players.eq(0)
    result["off_poss"] = core.off_poss.where(good & core.off_poss.gt(0)).to_numpy()
    result["def_poss"] = core.def_poss.where(good & core.def_poss.gt(0)).to_numpy()
    result["minutes"] = core.seconds_pbp.to_numpy() / 60.
    result["game_date"] = pd.to_datetime(core.date.astype(str), format="%Y%m%d").to_numpy()
    result["season_type"] = np.where(core.playoffs, "PS", "RS")
    # Only the previously validated personal-foul counter is carried. Its
    # missingness is explicit; event totals are not a substitute measurement.
    result = result.merge(old[["player_id", "game_id", "pf_committed"]], on=["player_id", "game_id"],
                          how="left", validate="one_to_one")
    result["player_name"] = result.player_id.astype(str)
    result.game_id = result.game_id.astype(str).str.zfill(10)
    context = result.loc[result.minutes.gt(0), ["game_id", "player_id", "is_home", "team_id", "opponent_id"]].rename(columns={"opponent_id": "opp_team_id"})
    result = result.drop(columns=["is_home", "team_id", "opponent_id"])
    result = result.sort_values(["player_id", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out / "player_game_lines.parquet", index=False)
    context.to_parquet(out / "game_context.parquet", index=False)
    foreign[["player_id", "game_id", "game_date", "off_poss", "def_poss", "fga", "fta"]].to_csv(out / "foreign_legacy_player_games.csv", index=False)
    manifest = dict(status="prepared_validated_core_source", rows=len(result), first_season=2014, last_season=2026,
        foreign_legacy_positive_rows=len(foreign), foreign_legacy_games=int(foreign.game_id.nunique()),
        unsafe_rate_denominator_rows=int((~good).sum()), personal_foul_counter_missing=int(result.pf_committed.isna().sum()),
        source_snapshot=source_manifest["snapshot_id"], legacy_counter_source_sha256=sha(LEGACY / "player_game_lines.parquet"),
        code_sha256=sha(__file__), pf_definition="legacy_personal_foul_counter_on_exact_validated_player_game_identity",
        estimator_change="new_validated_counts_with_transported_legacy_annual_parameters_not_refitted",
        selected_vpm_changed=False, protected_2027_rows=0,
        artifacts={name: sha(out / name) for name in ["player_game_lines.parquet", "game_context.parquet", "foreign_legacy_player_games.csv"]})
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, dict)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    build(parser.parse_args().out)

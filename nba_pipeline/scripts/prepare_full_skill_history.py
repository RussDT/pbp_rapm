"""Validate/normalize the pinned Databallr historical player tables.

Tracking vintage choice is row-specific, never cross-version missing-value fill.
All files are a separate research library; selected VPM sources are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from full_skill_history_specs import SKILLS, GENERIC_SKILLS

ROOT = Path(__file__).resolve().parents[1] / "results/estimated_skills/full_history_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(source, name):
    manifest = json.loads((source / name / "manifest.json").read_text())
    frames = []
    for item in manifest["objects"]:
        p = Path(item["local_file"])
        if sha(p) != item["sha256"]:
            raise ValueError("Historical source checksum mismatch")
        frames.append(pd.read_parquet(p))
    result = pd.concat(frames, ignore_index=True)
    if not result.season.between(1997, 2026).all():
        raise ValueError("Historical source contains an out-of-scope season")
    if result.duplicated(["game_id", "person_id"]).any():
        raise ValueError("Duplicate historical player/game identity")
    return result, manifest


def selected_report_column(report, name):
    # Disjoint two/three shot counts, derived only when both observed source
    # counts exist. This is exact accounting, not an imputed measurement.
    if name in {"catch_shoot_fg2a", "catch_shoot_fg2m", "pull_up_fg2a", "pull_up_fg2m"}:
        total = selected_report_column(report, name.replace("fg2", "fg"))
        three = selected_report_column(report, name.replace("fg2", "fg3"))
        return (total - three).where(total.ge(three))
    vintage = report.tracking_vintage
    if name + "_v1" in report or name + "_v2" in report:
        result = pd.Series(np.nan, index=report.index)
        for version in ["v1", "v2"]:
            column = name + "_" + version
            if column in report:
                take = vintage.eq(version)
                result.loc[take] = pd.to_numeric(report.loc[take, column], errors="coerce")
    elif name in report:
        result = pd.to_numeric(report[name], errors="coerce")
    else:
        return pd.Series(np.nan, index=report.index)
    # Explicit source reliability flags outrank the presence of another vintage.
    if name.startswith("ft_ast"):
        result = result.where(report.ft_ast_status.eq("ok"))
    if name.startswith(("drive_ft", "drive_pf", "drive_pts", "post_touch_ft", "post_touch_foul", "post_touch_pts")):
        result = result.where(report.ft_tracking_status.eq("ok"))
    good_join = report.ours_match.eq(True) & report.report_label_wrong.eq(False)
    return result.where(good_join & result.ge(0))


def prepare(out=ROOT):
    out = Path(out)
    source = out / "source"
    core, cm = read_table(source, "player_game")
    report, rm = read_table(source, "player_game_report")
    # Stable coarse partitions across source coding changes. Do not infer
    # offensive-foul turnovers from foul counts or call unsupported zeros real.
    core["tov_bad_pass_any"] = core.tov_bad_pass + core.tov_bad_pass_oob
    core["tov_lost_ball_any"] = core.tov_lost_ball + core.tov_lost_ball_oob
    core["tov_remainder"] = core.tov - core.tov_bad_pass_any - core.tov_lost_ball_any - core.tov_traveling
    core["foul_personal_group"] = core.foul_personal + core.foul_away_from_play + core.foul_inbound
    if not core.tov_remainder.ge(0).all():
        raise ValueError("Coarse turnover accounting is inconsistent")
    report_keys = ["game_id", "person_id", "date", "season", "playoffs"]
    joined = report[report_keys].merge(core[report_keys], on=["game_id", "person_id"], suffixes=("_report", "_core"), how="left", validate="one_to_one")
    matched = joined.date_core.notna()
    for key in ["date", "season", "playoffs"]:
        if not joined.loc[matched, key + "_report"].eq(joined.loc[matched, key + "_core"]).all():
            raise ValueError("Report joins to a different game date/season/type")
    numeric_keys = ["game_id", "person_id", "date", "season", "team_id", "opponent_id", "off_poss", "def_poss", "seconds_pbp"]
    if core[numeric_keys].isna().any().any():
        raise ValueError("Missing core identity or exposure")
    for zone in ["at_rim", "short_midrange", "long_midrange", "corner3", "arc3"]:
        if not (core[zone + "_fgm"].le(core[zone + "_fga"]) & core[zone + "_fga"].ge(0)).all():
            raise ValueError("Invalid core shooting counts")
    zone_sum = sum(core[z + "_fga"] for z in ["at_rim", "short_midrange", "long_midrange", "corner3", "arc3"])
    if not zone_sum.eq(core.fga).all():
        raise ValueError("PBP shot zones do not close to FGA")
    used = sorted({x for s in SKILLS for x in [s.numerator, s.denominator] if "." in x})
    identities = ["game_id", "person_id", "season", "playoffs", "date", "team_id", "opponent_id", "is_home", "seconds_pbp", "lineups_clean", "known_issue"]
    panel = core[identities].copy()
    panel = panel.rename(columns={"person_id": "player_id"})
    # Possession rates require admissible lineups. Box attempt percentages can
    # still use observed shooting counts when the lineup denominator is unsafe.
    good_exposure = core.lineups_clean.eq(True) & core.provably_wrong_rows.eq(0) & core.rows_without_10_players.eq(0)
    panel["off_poss"] = core.off_poss.where(good_exposure & core.off_poss.gt(0))
    panel["def_poss"] = core.def_poss.where(good_exposure & core.def_poss.gt(0))
    panel["total_poss"] = (core.off_poss + core.def_poss).where(good_exposure & (core.off_poss + core.def_poss).gt(0))
    for column in used:
        if column.startswith("core."):
            panel[column] = pd.to_numeric(core[column[5:]], errors="coerce").where(lambda x: x.ge(0))
    report_values = {column: selected_report_column(report, column[7:]) for column in used if column.startswith("report.")}
    report_fields = pd.DataFrame({"game_id": report.game_id, "player_id": report.person_id, **report_values})
    panel = panel.merge(report_fields, on=["game_id", "player_id"], how="left", validate="one_to_one")
    # A documented individual passing outage is not an observed activity zero.
    passing_bad = pd.Series(False, index=panel.index)
    for version in ["v1", "v2"]:
        potential = "report.potential_ast_" + version
        passes = "report.passes_made_" + version if "report.passes_made_" + version in panel else "report.passes_made"
        touches = "report.touches_" + version if "report.touches_" + version in panel else "report.touches"
        if potential not in panel or passes not in panel:
            continue
        bad = (panel[potential].eq(0) & panel[passes].eq(0) &
               ((core.ast.to_numpy() > 0) | (panel.get(touches, pd.Series(0, index=panel.index)).ge(10) & panel.seconds_pbp.ge(300))))
        passing_bad |= bad
        panel.loc[bad, [potential, passes]] = np.nan
    panel["game_date"] = pd.to_datetime(panel.date.astype(str), format="%Y%m%d")
    panel = panel.sort_values(["player_id", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    same_day_rows = int(panel.duplicated(["player_id", "game_date"], keep=False).sum())
    # Date-only sources do not establish which same-date game came first. The
    # kernel predicts the whole player/date batch, then updates from its totals.
    # Stable scalar ordinals are the only dates needed by the compiled replay.
    panel["day"] = (panel.game_date - pd.Timestamp("1970-01-01")).dt.days.astype(np.int32)
    panel["player_idx"] = pd.factorize(panel.player_id)[0].astype(np.int32)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "observations.parquet"
    panel.to_parquet(path, index=False)
    (out / "registry.json").write_text(json.dumps([s.record() for s in SKILLS], indent=2) + "\n")
    coverage = []
    invalid = {}
    for spec in SKILLS:
        num, den = panel[spec.numerator], panel[spec.denominator]
        valid = num.notna() & den.gt(0) & num.ge(0)
        valid &= panel.season.between(spec.first_season, spec.last_season)
        if spec.kind == "pct":
            valid &= num.le(den)
            invalid[spec.name] = int((num.notna() & den.notna() & num.gt(den)).sum())
        for year, sub in panel.groupby("season"):
            take = valid.loc[sub.index]
            coverage.append(dict(skill=spec.name, season=int(year), family=spec.family, player_games=len(sub), observed_rows=int(take.sum()),
                                 observed_players=int(sub.loc[take, "player_id"].nunique()), exposure=float(den.loc[sub.index][take].sum())))
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    receipt = dict(status="prepared_research_observations", rows=len(panel), players=int(panel.player_id.nunique()),
                   seasons=sorted(map(int, panel.season.unique())), min_date=str(panel.game_date.min().date()), max_date=str(panel.game_date.max().date()),
                   model_heads=len(SKILLS), distinct_concepts=len(GENERIC_SKILLS), unsafe_lineup_player_games=int((~good_exposure).sum()), passing_outage_rows=int(passing_bad.sum()),
                   unmatched_report_rows=int((~matched).sum()), invalid_proportion_rows=invalid,
                   same_player_date_batch_rows=same_day_rows,
                   sources={"core": cm["snapshot_id"], "report": rm["snapshot_id"]},
                   observation_sha256=sha(path), code_sha256=sha(__file__), registry_sha256=sha(out / "registry.json"),
                   historical_publication_vintages_certified=False, selected_vpm_changed=False, protected_2027_rows=0)
    (out / "prepare_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    prepare(parser.parse_args().out)

#!/usr/bin/env python3
"""Score a live season's mid/three shots with the frozen clean-SQ models.

The sibling VPM harness (`vpm.harness.chronological_sq`) refuses any season
>= 2027 because 2027 is its protected confirmation season. The owner allowed
the *live* luck-adjusted SQ time-decay to score 2027 (November 3, 2026) on one
condition: no refitting or selection on the new season. This script therefore
fits nothing. It

1. builds the causal feature panel exactly as the harness `prepare` stage does,
   from the frozen pre-season shot table plus a separately built shot table for
   the live season (never rebuilt over the frozen artifact);
2. checks every pre-season feature row against the frozen harness panel;
3. re-scores the frozen model season with its saved models and requires the
   frozen EV to reproduce;
4. scores the live season with the same saved models and vocabularies;
5. writes the frozen EV plus the live season to a new parquet outside the
   immutable harness output.

Usage:
    python scripts/score_live_clean_sq.py --season 2027
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
VPM_REPO = PIPELINE_ROOT.parent.parent / "vpm"
FROZEN_SQ = VPM_REPO / "vpm" / "results" / "harness" / "vpm_clean_sq_preseason_v2" / "sq"
FROZEN_SHOTS = PIPELINE_ROOT / "results" / "pbp_shot_quality" / "pbp_shot_quality_1997_2026.parquet"
PROCESSED = PIPELINE_ROOT / "processed"
EPM_CACHE = PIPELINE_ROOT / "cache" / "epm"
OUTPUT_ROOT = PIPELINE_ROOT / "results" / "live_clean_sq"
FIRST_MODERN_SEASON = 2015
EV_COLUMNS = ["season_end_year", "game_id", "event_num", "shooter_id", "zone", "made",
              "shot_value", "p_make", "ev_pts", "game_date"]

sys.path.insert(0, str(VPM_REPO))
from vpm.harness.artifacts import sha256_file, write_json  # noqa: E402
from vpm.harness.chronological_sq import (  # noqa: E402
    FEATURES, RAW_FEATURES, causal_talent, normal_game_id,
)


def load_shots(season: int, frozen_season: int) -> pd.DataFrame:
    frozen = pd.read_parquet(FROZEN_SHOTS, columns=RAW_FEATURES)
    if frozen.season_end_year.max() != frozen_season:
        raise ValueError(f"frozen shot table does not end at {frozen_season}")
    live_path = FROZEN_SHOTS.with_name(f"pbp_shot_quality_{season}.parquet")
    live = pd.read_parquet(live_path, columns=RAW_FEATURES)
    if set(live.season_end_year.unique()) != {season}:
        raise ValueError(f"{live_path.name} must contain only season {season}")
    return pd.concat([frozen, live], ignore_index=True)


def prepare_panel(shots: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict]:
    """Mirror of chronological_sq.prepare; only the live season's PS half is optional."""
    d = shots.loc[shots.shot_zone.isin(["short_mid", "long_mid", "deep_two", "three"])].copy()
    d["zone"] = np.where(d.shot_zone.eq("three"), "three", "mid")
    d["made"] = d.is_make.astype(int)
    d["shooter_id"] = pd.to_numeric(d.shooter_id, errors="raise").astype(np.int64)
    d["game_id"] = normal_game_id(d.game_id)
    d["event_num"] = d.event_num.astype(np.int64)
    seed = d.loc[d.season_end_year.lt(FIRST_MODERN_SEASON)].copy()
    d = d.loc[d.season_end_year.ge(FIRST_MODERN_SEASON) & d.season_end_year.le(season)].copy()
    dates, hashes = [], {}
    for year in range(FIRST_MODERN_SEASON, season + 1):
        for suffix in ["", "_PS"]:
            path = PROCESSED / f"RAPM{year % 100:02d}{suffix}.parquet"
            if not path.exists() and year == season and suffix == "_PS":
                continue
            table = pd.read_parquet(path, columns=["game_id", "game_date"]).drop_duplicates()
            table.game_id = normal_game_id(table.game_id)
            table.game_date = pd.to_datetime(table.game_date)
            dates.append(table)
            hashes[str(path)] = sha256_file(path)
    dates = pd.concat(dates).drop_duplicates()
    if dates.game_id.duplicated().any():
        raise ValueError("Conflicting game dates")
    d = d.merge(dates, on="game_id", how="left", validate="many_to_one")
    if d.game_date.isna().any():
        raise ValueError("Undated shot")
    d["playoffs"] = d.season_type.str.lower().eq("ps").astype(int)
    clock = d.time_quarter.astype("string").str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
    d["seconds_remaining_quarter"] = pd.to_numeric(clock[0]) * 60 + pd.to_numeric(clock[1])
    d = causal_talent(d, seed)
    snapshots = []
    for path in sorted(EPM_CACHE.glob("epm_all_*_go0.csv")):
        cut = pd.Timestamp(path.name.split("_")[2])
        if cut >= d.game_date.max():
            continue
        meta = json.loads(path.with_suffix(".manifest.json").read_text())
        if meta["csv_sha256"] != sha256_file(path) or meta["requested_date"] != str(cut.date()):
            raise ValueError("EPM snapshot hash/date mismatch")
        if any(pd.Timestamp(date) > cut for date in meta["provider_effective_dates"]):
            raise ValueError("Future EPM effective date")
        e = pd.read_csv(path, usecols=["player_id", "p_fg3pct", "p_fgpct_mid", "p_ftpct", "p_fg3a_100"])
        e["epm_requested_date"] = cut
        e.player_id = e.player_id.astype(np.int64)
        if e.player_id.duplicated().any():
            raise ValueError("Duplicate EPM identity")
        e.loc[e.p_fg3a_100.fillna(0) < .5, "p_fg3pct"] = np.nan
        snapshots.append(e)
        hashes[str(path)] = meta["csv_sha256"]
    e = pd.concat(snapshots, ignore_index=True)
    e["epm_requested_date"] = pd.to_datetime(e.epm_requested_date).astype("datetime64[ns]")
    d["game_date"] = pd.to_datetime(d.game_date).astype("datetime64[ns]")
    e = e.rename(columns={"player_id": "shooter_id", "p_fg3pct": "epm_fg3pct",
                          "p_fgpct_mid": "epm_midpct", "p_ftpct": "epm_ftpct", "p_fg3a_100": "epm_fg3a"})
    d = pd.merge_asof(d.sort_values("game_date"), e.sort_values("epm_requested_date"),
                      left_on="game_date", right_on="epm_requested_date", by="shooter_id",
                      direction="backward", allow_exact_matches=False)
    if (d.epm_requested_date.dropna() >= d.loc[d.epm_requested_date.notna(), "game_date"]).any():
        raise ValueError("Future EPM skill")
    d = d.sort_values(["game_date", "game_id", "period", "event_num"], kind="stable").reset_index(drop=True)
    return d, hashes


def check_frozen_panel(panel: pd.DataFrame, frozen_season: int) -> int:
    frozen = pd.read_parquet(FROZEN_SQ / "features.parquet")
    ours = panel.loc[panel.season_end_year.le(frozen_season), frozen.columns]
    key = ["game_id", "event_num"]
    merged = frozen.merge(ours, on=key, how="outer", suffixes=("_f", "_o"), indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError(f"pre-season panel keys differ: {merged._merge.value_counts().to_dict()}")
    for column in frozen.columns.difference(key):
        a, b = merged[column + "_f"], merged[column + "_o"]
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            same = np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-12, equal_nan=True)
        else:
            same = (a == b) | (a.isna() & b.isna())
        if not np.all(same):
            raise ValueError(f"pre-season panel column {column} differs on {int((~same).sum())} rows")
    return len(merged)


def score_with_frozen(panel: pd.DataFrame, season: int, model_season: int, zone: str) -> tuple[pd.DataFrame, dict]:
    receipt_path = FROZEN_SQ / "models" / f"{model_season}_{zone}.json"
    receipt = json.loads(receipt_path.read_text())
    model_path = receipt_path.with_suffix(".joblib")
    if sha256_file(model_path) != receipt["model_sha256"]:
        raise ValueError(f"frozen model {model_path.name} changed")
    if receipt["features"] != FEATURES:
        raise ValueError("frozen model feature list differs from the harness")
    test = panel.loc[panel.zone.eq(zone) & panel.season_end_year.eq(season)].sort_values(
        ["game_date", "game_id", "period", "event_num"], kind="stable").copy()
    if pd.Timestamp(receipt["training_end"]) >= test.game_date.min():
        raise ValueError("frozen model training overlaps the scored season")
    for raw, column in [("event_action_type", "action_code"), ("shot_family", "family_code")]:
        mapping = {value: index for index, value in enumerate(receipt["vocabulary"][raw])}
        test[column] = test[raw].map(mapping).fillna(-1).astype(float)
    model = joblib.load(model_path)
    test["p_make"] = model.predict_proba(test[FEATURES].astype(float))[:, 1]
    test["ev_pts"] = test.p_make * test.shot_value
    return test[EV_COLUMNS], dict(model=model_path.name, model_sha256=receipt["model_sha256"],
                                  training_end=receipt["training_end"], scored_rows=len(test),
                                  first_scored_game=str(test.game_date.min().date()),
                                  last_scored_game=str(test.game_date.max().date()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=2027)
    parser.add_argument("--frozen-season", type=int, default=2026)
    args = parser.parse_args()
    season, frozen_season = args.season, args.frozen_season
    if season != frozen_season + 1:
        raise SystemExit("score only the season immediately after the frozen model season")

    panel, hashes = prepare_panel(load_shots(season, frozen_season), season)
    checked = check_frozen_panel(panel, frozen_season)
    print(f"pre-season panel matches frozen harness features on {checked:,} shots")

    frozen_ev = pd.read_parquet(FROZEN_SQ / "clean_ev.parquet")
    receipts = {}
    live = []
    for zone in ["mid", "three"]:
        rescored, _ = score_with_frozen(panel, frozen_season, frozen_season, zone)
        reference = frozen_ev.loc[frozen_ev.season_end_year.eq(frozen_season) & frozen_ev.zone.eq(zone)]
        joined = reference.merge(rescored, on=["game_id", "event_num"], suffixes=("_f", "_r"), validate="one_to_one")
        gap = float(np.abs(joined.ev_pts_f - joined.ev_pts_r).max())
        if len(joined) != len(reference) or gap > 1e-12:
            raise ValueError(f"{frozen_season} {zone} re-score does not reproduce frozen EV (max gap {gap})")
        print(f"{frozen_season} {zone}: re-score reproduces frozen EV on {len(joined):,} shots")
        scored, receipt = score_with_frozen(panel, season, frozen_season, zone)
        receipts[zone] = receipt
        live.append(scored)
        print(f"{season} {zone}: {len(scored):,} shots scored with {receipt['model']}")

    live = pd.concat(live, ignore_index=True)
    out = pd.concat([frozen_ev[EV_COLUMNS], live], ignore_index=True)
    if out.duplicated(["game_id", "event_num"]).any():
        raise ValueError("Duplicate EV keys")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    first = int(frozen_ev.season_end_year.min())
    out_path = OUTPUT_ROOT / f"clean_ev_live_{first}_{season}.parquet"
    out.to_parquet(out_path, index=False)
    write_json(out_path.with_suffix(".manifest.json"), dict(
        season=season, frozen_model_season=frozen_season, refit=False,
        owner_decision="2026-11-03: live luck-adjusted SQ time-decay may score 2027; no refit or selection on 2027",
        frozen_ev=str(FROZEN_SQ / "clean_ev.parquet"), frozen_ev_sha256=sha256_file(FROZEN_SQ / "clean_ev.parquet"),
        live_rows=len(live), rows=len(out), rows_by_season=out.groupby("season_end_year").size().to_dict(),
        live_mean_ev_by_zone=live.groupby("zone").ev_pts.mean().round(6).to_dict(),
        live_mean_actual_by_zone=(live.made * live.shot_value).groupby(live.zone).mean().round(6).to_dict(),
        epm_coverage_live=float(panel.loc[panel.season_end_year.eq(season), "epm_requested_date"].notna().mean()),
        frozen_panel_rows_checked=checked, models=receipts, source_hashes=hashes,
        output_sha256=sha256_file(out_path)))
    print(f"wrote {out_path} ({len(out):,} rows; {len(live):,} live)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Conservative, date-ordered PA/DFGA repair of the local player_sheets archive.

Never edits the source archive. Missingness detection and imputation are distinct.
Fitted estimates are NOT measured observations for an estimated-skills filter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "player_sheets/game_report"
OUT = ROOT / "nba_pipeline/results/estimated_skills/tracking_backfill"
TARGETS = {"potential_ast": "POTENTIAL_AST", "dfga": "overall_def_D_FGA"}
BOX = ["AST", "FGA", "FTA", "TOV", "OREB", "DREB", "STL", "BLK", "PF"]
KEEP = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "date", "MIN", *BOX,
        "POTENTIAL_AST", "PASSES_MADE", "PASSES_RECEIVED", "TOUCHES",
        "DRIVES", "TIME_OF_POSS", "DREB_CONTEST", "hustle_DEFLECTIONS",
        "hustle_CONTESTED_SHOTS", "overall_def_D_FGA", "overall_def_D_FGM",
        "two_pt_def_FG2A", "three_pt_def_FG3A"]


def load_archive(source: Path):
    rows, receipts = [], []
    for year in range(2014, 2027):
        for suffix in ("", "ps"):
            path = source / f"year_files/{year}{suffix}_games.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            d = pd.read_csv(path, usecols=lambda c: c in KEEP, low_memory=False)
            for c in KEEP:
                if c not in d:
                    d[c] = np.nan
                if c != "PLAYER_NAME":
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d[d.MIN.gt(0)].copy()
            if d[["PLAYER_ID", "date"]].isna().any().any():
                raise ValueError(f"Missing identity: {path}")
            if d.duplicated(["PLAYER_ID", "date"]).any():
                raise ValueError(f"Duplicate player/date: {path}")
            d["season"], d["playoffs"] = year, bool(suffix)
            for target in TARGETS.values():
                d[f"{target}_alternate"] = np.nan
            altpath = source / f"all_games/all_{year}{suffix}.parquet"
            if altpath.exists():
                a = pd.read_parquet(altpath)
                a = a.drop_duplicates()
                if a.duplicated(["PLAYER_ID", "date"]).any():
                    raise ValueError(f"Ambiguous alternate source: {altpath}")
                cols = ["PLAYER_ID", "date", "MIN", "AST", *TARGETS.values()]
                a = a.reindex(columns=cols).rename(columns={c: c+"_alt" for c in cols[2:]})
                d = d.merge(a, on=["PLAYER_ID", "date"], how="left", validate="one_to_one")
                agrees = (d.MIN-d.MIN_alt).abs().lt(.05) & d.AST.eq(d.AST_alt)
                for target in TARGETS.values():
                    d[f"{target}_alternate"] = d[target+"_alt"].where(agrees)
                d = d.drop(columns=[c+"_alt" for c in cols[2:]])
                receipts.append({"path": str(altpath), "sha256": hashlib.sha256(altpath.read_bytes()).hexdigest()})
            receipts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            rows.append(d)
    d = pd.concat(rows, ignore_index=True).sort_values(["date", "PLAYER_ID"]).reset_index(drop=True)
    if d.duplicated(["PLAYER_ID", "date"]).any():
        raise ValueError("Player/date collision across season files")
    d["day"] = pd.to_datetime(d.date.astype(int).astype(str), format="%Y%m%d")
    return d, receipts


def classify(d):
    """PA<AST alone is diagnostic only; require independent outage evidence."""
    d = d.copy()
    keys = ["season", "playoffs", "date", "TEAM_ID"]
    totals = d.groupby(keys)[["MIN", "AST", "POTENTIAL_AST", "PASSES_MADE", "TOUCHES"]].transform("sum")
    # Ignore small/stale-team fragments in the source's team assignments.
    team_bad = (totals.MIN.ge(180) & totals.TOUCHES.ge(100)
                & totals.PASSES_MADE.lt(.35*totals.TOUCHES)
                & totals.POTENTIAL_AST.lt(totals.AST))
    row_bad = (d.POTENTIAL_AST.eq(0) & d.PASSES_MADE.eq(0)
               & (d.AST.gt(0) | (d.TOUCHES.ge(10) & d.MIN.ge(5))))
    d["pa_below_ast_flag"] = d.POTENTIAL_AST.lt(d.AST)
    d["passing_team_outage"] = team_bad
    team_row_bad = team_bad & d.POTENTIAL_AST.eq(0) & d.PASSES_MADE.fillna(0).eq(0)
    d["potential_ast_reason"] = np.select(
        [d.POTENTIAL_AST.isna(), d.POTENTIAL_AST.lt(0), team_row_bad, row_bad],
        ["missing", "negative", "team_passing_outage", "zero_passing_with_activity"], default="observed")
    d["dfga_reason"] = np.select(
        [d.overall_def_D_FGA.isna(), d.overall_def_D_FGA.lt(0),
         d.overall_def_D_FGA.lt(d.overall_def_D_FGM)],
        ["missing_or_omitted_zero", "negative", "attempts_below_makes"], default="observed")
    for name, raw in TARGETS.items():
        valid = d[f"{name}_reason"].eq("observed")
        d[f"{name}_observed"] = d[raw].where(valid)
        alt = d[f"{raw}_alternate"]
        recover = ~valid & alt.ge(0) & (alt.ne(d[raw]) | d[raw].isna())
        if name == "potential_ast":
            recover &= alt.gt(0)  # an alternate zero cannot establish coverage
        else:
            recover &= alt.ge(d.overall_def_D_FGM.fillna(0))
        d.loc[recover, f"{name}_observed"] = alt[recover]
        d[f"{name}_source"] = np.where(recover, "recovered_archive", np.where(valid, "observed", "unresolved"))
    # Missing overall defense can be reconstructed only when BOTH partitions exist.
    split = d.two_pt_def_FG2A + d.three_pt_def_FG3A
    recover = d.dfga_observed.isna() & split.ge(0) & split.ge(d.overall_def_D_FGM.fillna(0))
    d.loc[recover, "dfga_observed"] = split[recover]
    d.loc[recover, "dfga_source"] = "recovered_two_plus_three"
    defense = d.assign(covered_minutes=d.MIN.where(d.dfga_observed.notna(), 0))
    grouped = defense.groupby(keys)
    covered = grouped.covered_minutes.transform("sum")
    total_minutes = grouped.MIN.transform("sum")
    total_attempts = grouped.dfga_observed.transform("sum")
    defenders = grouped.dfga_observed.transform("count")
    missing = d.dfga_observed.isna()
    outage = total_minutes.ge(180) & covered.eq(0)
    # Endpoint has no explicit DFGA zeros anywhere in the archive. Infer omitted
    # zeros only with strong team coverage AND a short stint; otherwise abstain.
    probable_zero = (missing & total_minutes.ge(180) & covered.ge(.85*total_minutes)
                     & defenders.ge(5) & total_attempts.ge(20) & d.MIN.le(10)
                     & d.two_pt_def_FG2A.isna() & d.three_pt_def_FG3A.isna()
                     & d.overall_def_D_FGM.fillna(0).eq(0))
    d.loc[missing & outage, "dfga_reason"] = "team_defense_outage"
    d.loc[probable_zero, "dfga_reason"] = "likely_omitted_zero"
    d.loc[probable_zero, "dfga_source"] = "inferred_endpoint_zero"
    return d


def features(d, name):
    # Only box counts and strictly earlier measured history: independent of the
    # currently missing tracking family. Never feed an imputation back into history.
    exposure = d.MIN.clip(lower=1)
    x = pd.DataFrame({"minutes": d.MIN})
    for c in BOX:
        x[c+"_36"] = 36*d[c]/exposure
    valid = d[f"{name}_observed"].notna()
    numerator = d[f"{name}_observed"].fillna(0)
    denominator = d.MIN.where(valid, 0)
    # Rolling observed-history sums (30 preceding appearances), not current row.
    ids = d.PLAYER_ID
    n = numerator.groupby(ids).transform(lambda s: s.shift().rolling(30, min_periods=1).sum()).fillna(0)
    e = denominator.groupby(ids).transform(lambda s: s.shift().rolling(30, min_periods=1).sum()).fillna(0)
    # Prior-date league rate; same-day outcomes excluded for every player.
    daily = pd.DataFrame({"n": numerator, "e": denominator, "day": d.day}).groupby("day")[["n", "e"]].sum()
    prior = daily.cumsum().shift()
    rate = (prior.n/prior.e).reindex(d.day).to_numpy()
    historical = (n.to_numpy()+200*rate)/(e.to_numpy()+200)
    x["history_36"] = 36*historical
    x["history_minutes"] = e
    return x, historical


def recover_daily(d, source):
    """Try preserved endpoint caches for flagged outages before estimating."""
    receipts = []
    flagged = d.potential_ast_observed.isna() | d.dfga_reason.eq("team_defense_outage")
    for (season, playoffs, date), group in d[flagged].groupby(["season", "playoffs", "date"]):
        suffix = "ps" if playoffs else ""
        paths = [source/f"year_files/cache_{season}{suffix}/{int(date)}.csv",
                 source/f"additional_data/{season}{suffix}/{int(date)}_additional.csv"]
        for path in paths:
            if not path.exists():
                continue
            a = pd.read_csv(path, usecols=lambda c: c in ["PLAYER_ID", "date", *TARGETS.values()])
            if "date" not in a or "PLAYER_ID" not in a:
                continue
            a = a[a.date.eq(date)].drop_duplicates()
            if a.duplicated("PLAYER_ID").any():
                continue
            a = a.set_index("PLAYER_ID")
            recovered = 0
            for name, raw in TARGETS.items():
                if raw not in a:
                    continue
                values = group.PLAYER_ID.map(pd.to_numeric(a[raw], errors="coerce"))
                eligible = d.loc[group.index, f"{name}_observed"].isna() & values.gt(0)
                if name == "dfga":
                    eligible &= values.ge(d.loc[group.index, "overall_def_D_FGM"].fillna(0))
                ii = group.index[eligible]
                d.loc[ii, f"{name}_observed"] = values[eligible]
                d.loc[ii, f"{name}_source"] = "recovered_daily_cache"
                recovered += len(ii)
            receipts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "recovered": recovered})
    return d, receipts


def fit_backfill(d):
    reports = []
    # Annual models prevent target-year tuning; first year is warm-up only.
    for name in TARGETS:
        x, history = features(d, name)
        value = d[f"{name}_observed"]
        d[f"{name}_estimate"] = np.nan
        d[f"{name}_estimate_sd"] = np.nan
        d[f"{name}_fit_before"] = ""
        # Repair the archive's first season using only earlier dates. These are
        # warm-up reconstructions, not part of the model-selection evidence.
        initial = d.season.eq(2014) & value.isna()
        if name == "dfga":
            initial &= d.dfga_reason.eq("team_defense_outage")
        for day in sorted(d.loc[initial, "day"].unique()):
            test = initial & d.day.eq(day)
            train = d.day.lt(day) & value.notna() & d.MIN.ge(5)
            if train.sum() < 500:
                continue
            model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                min_samples_leaf=80, l2_regularization=10, learning_rate=.06,
                early_stopping=False, random_state=20260912)
            model.fit(x[train], 36*value[train]/d.MIN[train], sample_weight=d.MIN[train])
            d.loc[test, f"{name}_estimate"] = np.maximum(0, model.predict(x[test]))*d.loc[test, "MIN"]/36
            d.loc[test, f"{name}_fit_before"] = pd.Timestamp(day).strftime("%Y-%m-%d")
        for season in range(2015, 2027):
            test = d.season.eq(season)
            cutoff = d.loc[test, "day"].min()
            train = d.day.lt(cutoff) & value.notna() & d.MIN.ge(5)
            if train.sum() < 500:
                continue
            model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                min_samples_leaf=80, l2_regularization=10, learning_rate=.06,
                early_stopping=False, random_state=20260912)
            model.fit(x[train], 36*value[train]/d.MIN[train], sample_weight=d.MIN[train])
            pred = np.maximum(0, model.predict(x[test])) * d.loc[test, "MIN"]/36
            d.loc[test, f"{name}_estimate"] = pred
            d.loc[test, f"{name}_fit_before"] = cutoff.strftime("%Y-%m-%d")
            # Honest temporal validation, dev only; no 2026 target scoring.
            score = test & value.notna() & d.MIN.ge(5)
            if season <= 2025:
                yy = value[score].to_numpy()
                pp = d.loc[score, f"{name}_estimate"].to_numpy()
                hh = history[score]*d.loc[score, "MIN"].to_numpy()
                base = 2*d.loc[score, "AST"].to_numpy() if name == "potential_ast" else hh
                rows = pd.DataFrame({"date": d.loc[score, "date"], "model_se": (pp-yy)**2,
                                     "history_se": (hh-yy)**2, "baseline_se": (base-yy)**2})
                reports.append({"skill": name, "season": season, "n": len(yy),
                    "rmse": float(np.sqrt(rows.model_se.mean())),
                    "history_rmse": float(np.sqrt(rows.history_se.mean())),
                    "baseline_rmse": float(np.sqrt(rows.baseline_se.mean())),
                    "bias": float(np.mean(pp-yy)),
                    "residual_variance_per_minute": float(np.mean((pp-yy)**2/d.loc[score, "MIN"])),
                    "fit_before": cutoff.strftime("%Y-%m-%d"),
                    "date_errors": rows.groupby("date")[["model_se", "history_se", "baseline_se"]].mean().reset_index().to_dict("records")})
            earlier = [r for r in reports if r["skill"] == name and r["season"] < season]
            if earlier:
                variance = np.average([r["residual_variance_per_minute"] for r in earlier], weights=[r["n"] for r in earlier])
                d.loc[test, f"{name}_estimate_sd"] = np.sqrt(variance*d.loc[test, "MIN"])
            print(f"{name} {season}: train={train.sum()} gaps={(test & value.isna()).sum()}", flush=True)
        missing = value.isna() & d[f"{name}_estimate"].notna()
        if name == "dfga":
            # Only confirmed whole-team outages receive the model estimate.
            # Ambiguous individual omissions remain missing, never inflated to
            # a positive count by a model trained on recorded positive counts.
            missing &= d.dfga_reason.eq("team_defense_outage")
        d[f"{name}_filled"] = value.copy()
        d.loc[missing, f"{name}_filled"] = d.loc[missing, f"{name}_estimate"]
        d.loc[missing, f"{name}_source"] = "estimated"
        if name == "dfga":
            d.loc[d.dfga_source.eq("inferred_endpoint_zero"), "dfga_filled"] = 0.
        # Observation-only output is authoritative for later independent filters.
        d[f"{name}_is_measured"] = value.notna()
    return d, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d, receipts = load_archive(args.source)
    d = classify(d)
    d, cache_receipts = recover_daily(d, args.source)
    receipts.extend(cache_receipts)
    d, reports = fit_backfill(d)
    summary = []
    for (season, playoffs), group in d.groupby(["season", "playoffs"]):
        for name in TARGETS:
            summary.append({"season": int(season), "playoffs": bool(playoffs), "skill": name,
                "rows": len(group), "source_counts": group[f"{name}_source"].value_counts().to_dict(),
                "reason_counts": group[f"{name}_reason"].value_counts().to_dict()})
    for name, raw in TARGETS.items():
        untouched = d[f"{name}_source"].eq("observed")
        assert np.array_equal(d.loc[untouched, raw], d.loc[untouched, f"{name}_filled"])
        assert d[f"{name}_filled"].dropna().ge(0).all()
    d.drop(columns="day").to_parquet(args.out/"tracking_backfill.parquet", index=False)
    changed = ~d.potential_ast_source.eq("observed") | ~d.dfga_source.eq("observed")
    d.loc[changed].drop(columns="day").to_csv(args.out/"repair_ledger.csv", index=False)
    flat = [{k: v for k, v in r.items() if k != "date_errors"} for r in reports]
    pd.DataFrame(flat).to_csv(args.out/"validation.csv", index=False)
    report = {"version": 1, "rows": len(d), "inputs": receipts, "coverage": summary,
        "validation": reports, "fences": [
            "2014 uses expanding earlier-date fits after at least 500 observed rows; earlier unsupported gaps remain unresolved. No future-fitted historical imputation.",
            "2026 not scored; all parameters fixed without 2026 target validation.",
            "DFGA modeled only for whole-team outages. Healthy-team short-stint omissions receive labeled inferred zeros; ambiguous individual gaps remain unresolved.",
            "DFGA estimates/validation condition on recorded attempts (zero-truncated); do not treat them as a calibrated all-player count model.",
            "Validation is sequential missing-row simulation on observed records, not proof for missing-not-at-random outages.",
            "No estimated row enters history or constitutes independent evidence for a Kalman observation.",
            "Same-game box inputs mean postgame reconstruction, never pregame prediction.",
            "Snapshot recovery is retrospective: provider acquisition timestamps are unavailable."]}
    (args.out/"audit.json").write_text(json.dumps(report, indent=2))
    notebook_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Tracking backfill review\n",
            "Postgame reconstruction; original observations and inferred values remain separate.\n",
            "Run from this output directory. 2026 target errors are deliberately not evaluated."]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "from pathlib import Path\nimport pandas as pd\nimport json\n",
            "root = Path('.')\n",
            "panel = pd.read_parquet(root / 'tracking_backfill.parquet')\n",
            "validation = pd.read_csv(root / 'validation.csv')\n",
            "audit = json.loads((root / 'audit.json').read_text())\n",
            "validation"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "for skill in ['potential_ast', 'dfga']:\n",
            "    display(panel.groupby(['season', skill + '_source']).size().unstack(fill_value=0))"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "panel.loc[(panel.season == 2026) & (panel.potential_ast_source == 'estimated'),\n",
            "    ['PLAYER_NAME', 'date', 'MIN', 'AST', 'POTENTIAL_AST',\n",
            "     'potential_ast_filled', 'potential_ast_estimate_sd', 'potential_ast_reason']]"]},
    ]
    (args.out/"review.ipynb").write_text(json.dumps({"nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": notebook_cells}, indent=2))
    print(json.dumps({"rows": len(d), "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()

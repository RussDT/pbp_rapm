#!/usr/bin/env python3
"""Matched annual repeatability audit of production zTS; no fitted metric changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = next(p for p in ROOT.parents if (p / "WORKSPACE.yaml").exists())
METRICS = ("conventional_ts", "native_ts", "rts", "zts", "role")


def load_panel(zts_path, box_path, role_path, end_year=2025):
    z = pd.read_csv(zts_path)
    z = z.loc[z.playoffs.eq(0)].copy()
    cols = ["nba_id", "year", "Minutes", "TSA", "OffPoss", "TS_pct", "rTSPct",
            "Points", "FGA", "FTA", "TeamAbbreviation", "Technical Free Throw Trips"]
    box = pd.read_csv(box_path, usecols=cols, low_memory=False)
    d = z.merge(box, on=["nba_id", "year"], how="left", validate="one_to_one", indicator=True)
    d["rts_source_gap"] = abs(d.rts - d.rTSPct)
    d["minutes_source_gap"] = abs(d.minutes - d.Minutes)
    d["published_closure_gap"] = abs(d.zts - d.rts - d.role)
    d["native_ts_formula_gap"] = abs(
        d.TS_pct - (d.Points - .8*d["Technical Free Throw Trips"])/(2*d.TSA))
    d["league_ts"] = 100*d.TS_pct - d.rTSPct
    audit = d.groupby("year").agg(
        source_rows=("nba_id", "size"), finite_zts=("zts", "count"),
        max_rts_source_gap=("rts_source_gap", "max"),
        max_minutes_source_gap=("minutes_source_gap", "max"),
        max_published_closure_gap=("published_closure_gap", "max"),
        max_native_ts_formula_gap=("native_ts_formula_gap", "max"),
        league_ts_min=("league_ts", "min"), league_ts_max=("league_ts", "max"))
    d = d.loc[d.year.between(2006, end_year)].copy()
    if (d.rts_source_gap.dropna() > .000501).any() or (d.minutes_source_gap.dropna() > 1).any():
        raise ValueError("Scoring and zTS snapshots disagree in requested seasons")
    if d.native_ts_formula_gap.max() > 1e-9 or d.published_closure_gap.max() > .001001:
        raise ValueError("Production scoring identity failed")
    # Preserve exact production algebra while removing only export rounding.
    d["rts"] = d.rTSPct
    d["native_ts"] = 100*d.TS_pct
    d["zts"] = d.rts + d.role
    d["conventional_ts"] = 100*d.Points/(2*(d.FGA + .44*d.FTA))
    d["team"] = d.TeamAbbreviation.fillna("").astype(str)
    d["single_team"] = d.team.str.fullmatch(r"[A-Z]{2,3}") & ~d.team.isin(["TOT", "FA"])

    roles = pd.read_csv(role_path)
    if roles.estimated.dtype != bool:
        raise ValueError("Expected explicit Boolean estimated flags")
    roles["estimated_tsa"] = roles.TSA * roles.estimated
    roles["expected_mass"] = roles.TSA * roles.playtype_avg_ts
    roles["league_mass"] = roles.TSA * roles.league_playtype_avg_ts
    a = roles.groupby(["nba_id", "year"]).agg(
        role_tsa=("TSA", "sum"), estimated_tsa=("estimated_tsa", "sum"),
        expected_mass=("expected_mass", "sum"), league_mass=("league_mass", "sum"))
    a["estimated_share"] = a.estimated_tsa/a.role_tsa
    a["reconstructed_role"] = 100*(a.league_mass-a.expected_mass)/a.role_tsa
    d = d.merge(a[["estimated_share", "reconstructed_role"]], on=["nba_id", "year"],
                how="left", validate="one_to_one")
    d["role_source_gap"] = abs(d.role-d.reconstructed_role)
    if d.role_source_gap.max() > .000501:
        raise ValueError("Modern role rows do not match the published zTS snapshot")
    eligible = np.isfinite(d[list(METRICS)+["TSA"]]).all(axis=1) & d.TSA.gt(0)
    coverage = d.assign(eligible=eligible).groupby("year").agg(
        rows=("nba_id", "size"), usable=("eligible", "sum"))
    return d.loc[eligible].copy(), audit.join(coverage)


def make_pairs(panel):
    prior = panel.copy()
    prior["target_year"] = prior.year+1
    pairs = prior.merge(panel, left_on=["nba_id", "target_year"], right_on=["nba_id", "year"],
                        suffixes=("_a", "_b"), validate="one_to_one")
    pairs["role_change"] = pairs.role_b-pairs.role_a
    pairs["abs_role_change"] = abs(pairs.role_change)
    pairs["both_single_team"] = pairs.single_team_a & pairs.single_team_b
    pairs["changed_team"] = pairs.both_single_team & pairs.team_a.ne(pairs.team_b)
    pairs["harmonic_tsa"] = 2*pairs.TSA_a*pairs.TSA_b/(pairs.TSA_a+pairs.TSA_b)
    return pairs


def summarize_sums(s):
    """Columns: weight, x, y, x², y², xy, |y-x|, (y-x)²; vectorized."""
    w = s[..., 0]
    mx, my = s[..., 1]/w, s[..., 2]/w
    vx = np.maximum(s[..., 3]/w-mx*mx, 0)
    vy = np.maximum(s[..., 4]/w-my*my, 0)
    denom = np.sqrt(vx*vy)
    corr = np.divide(s[..., 5]/w-mx*my, denom,
                     out=np.full_like(w, np.nan, dtype=float), where=denom > 1e-12)
    return np.stack([corr, s[..., 6]/w, np.sqrt(s[..., 7]/w), my-mx,
                     np.sqrt(vx), np.sqrt(vy)], axis=-1)


def evaluate(pairs, label, weighting="equal_pair", reps=2000, seed=90317, cluster_by="nba_id"):
    if len(pairs) < 20:
        return [], [], []
    w = np.ones(len(pairs))
    if weighting == "harmonic_tsa":
        w = pairs.harmonic_tsa.to_numpy()
    elif weighting == "equal_season":
        w = (1/pairs.groupby("target_year").target_year.transform("size")).to_numpy()
    ids, inverse = np.unique(pairs[cluster_by], return_inverse=True)
    moments = []
    for metric in METRICS:
        x, y = pairs[f"{metric}_a"].to_numpy(), pairs[f"{metric}_b"].to_numpy()
        moments.append(np.column_stack([w, w*x, w*y, w*x*x, w*y*y, w*x*y,
                                        w*abs(y-x), w*(y-x)**2]))
    moments = np.stack(moments, axis=1)
    cluster = np.zeros((len(ids), len(METRICS), 8))
    np.add.at(cluster, inverse, moments)
    point = summarize_sums(cluster.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(ids), np.ones(len(ids))/len(ids), size=reps)
    boot = summarize_sums((draws @ cluster.reshape(len(ids), -1)).reshape(reps, len(METRICS), 8))
    keys = ["pearson", "mae_pp", "rmse_pp", "mean_change_pp", "sd_a_pp", "sd_b_pp"]
    common = dict(sample=label, weighting=weighting, pairs=len(pairs),
                  players=pairs.nba_id.nunique(), bootstrap_cluster=cluster_by, clusters=len(ids))
    rows = []
    for i, metric in enumerate(METRICS):
        row = dict(common, metric=metric)
        for j, key in enumerate(keys):
            row[key] = point[i, j]
            row[key+"_lo"], row[key+"_hi"] = np.nanquantile(boot[:, i, j], [.025, .975])
        row["spearman_unweighted"] = pairs[f"{metric}_a"].corr(pairs[f"{metric}_b"], method="spearman")
        rows.append(row)
    contrasts = []
    for baseline in ("conventional_ts", "native_ts", "rts"):
        b = METRICS.index(baseline)
        for j, key in enumerate(keys[:3]):
            delta = boot[:, 3, j]-boot[:, b, j]
            lo, hi = np.nanquantile(delta, [.025, .975])
            contrasts.append(dict(common, baseline=baseline, statistic=key,
                                  zts_minus_baseline=point[3, j]-point[b, j], lo=lo, hi=hi))
    # Explain mechanical stability: Δz² = Δr² + 2ΔrΔdifficulty + Δdifficulty².
    dr = (pairs.rts_b-pairs.rts_a).to_numpy()
    dd = pairs.role_change.to_numpy()
    dz = (pairs.zts_b-pairs.zts_a).to_numpy()
    decomposition = dict(common, rts_msd=np.average(dr**2, weights=w),
                         role_msd=np.average(dd**2, weights=w),
                         cross_term=np.average(2*dr*dd, weights=w),
                         zts_msd=np.average(dz**2, weights=w),
                         closure_max=float(np.max(abs(dz**2-dr**2-2*dr*dd-dd**2))))
    return rows, contrasts, [decomposition]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zts", type=Path, default=ROOT/"nba_pipeline/results/historical_scoring_security/databallr_synergy_zts_2006_2026.csv")
    parser.add_argument("--box", type=Path, default=WORKSPACE/"nba/pipelines/core/csvs/merged_rs_full.csv.gz")
    parser.add_argument("--roles", type=Path, default=WORKSPACE/"nba/pipelines/core/nba_rapm/data/outputs/playtype_rows_with_estimates.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"nba_pipeline/results/zts_stability/2006_2025")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="retrospective repeatability, not prospective validation",
                    seasons=[2006, 2025], primary_min_tsa_each=250, bootstrap=args.bootstrap,
                    bootstrap_unit="player, preserving all their transitions", seed=90317,
                    role_change_cutoffs_pp=[.5, 1.5],
                    within_season="unavailable: production playtype inputs are annual aggregates",
                    excluded_2026="scoring and published zTS snapshots disagree",
                    estimated_rows_caveat="surrounding-season priors can mechanically smooth history",
                    post_inspection_diagnostics=["modern large-role changes", "season-cluster uncertainty"],
                    inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (args.zts, args.box, args.roles)})
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    panel, audit = load_panel(args.zts, args.box, args.roles)
    pairs = make_pairs(panel)
    primary = pairs.loc[pairs.TSA_a.ge(250) & pairs.TSA_b.ge(250)].copy()
    samples = [("primary_250", primary)]
    for threshold in (100, 500, 1000):
        samples.append((f"min_tsa_{threshold}", pairs.loc[pairs.TSA_a.ge(threshold)&pairs.TSA_b.ge(threshold)]))
    samples += [("role_stable_le_0.5", primary.loc[primary.abs_role_change.le(.5)]),
                ("role_change_0.5_to_1.5", primary.loc[primary.abs_role_change.gt(.5)&primary.abs_role_change.le(1.5)]),
                ("role_change_gt_1.5", primary.loc[primary.abs_role_change.gt(1.5)]),
                ("same_team", primary.loc[primary.both_single_team&~primary.changed_team]),
                ("changed_team", primary.loc[primary.changed_team]),
                ("historical_2006_2013", primary.loc[primary.year_b.le(2013)]),
                ("modern_2014_2025", primary.loc[primary.year_a.ge(2014)]),
                ("modern_no_estimated_rows", primary.loc[primary.estimated_share_a.eq(0)&primary.estimated_share_b.eq(0)]),
                ("modern_le_5pct_estimated", primary.loc[primary.estimated_share_a.le(.05)&primary.estimated_share_b.le(.05)])]
    samples += [("modern_role_change_gt_1.5", primary.loc[primary.year_a.ge(2014)&primary.abs_role_change.gt(1.5)]),
                ("modern_low_estimated_role_change_gt_1.5", primary.loc[
                    primary.estimated_share_a.le(.05)&primary.estimated_share_b.le(.05)&primary.abs_role_change.gt(1.5)])]
    samples += [(f"season_{year}", group) for year, group in primary.groupby("target_year")]
    metrics, comparisons, decomposition = [], [], []
    for label, sample in samples:
        weights = ["equal_pair", "harmonic_tsa", "equal_season"] if label == "primary_250" else ["equal_pair"]
        for weighting in weights:
            a, b, c = evaluate(sample, label, weighting, args.bootstrap)
            metrics.extend(a); comparisons.extend(b); decomposition.extend(c)
    a, b, c = evaluate(primary, "primary_250_season_clusters", reps=args.bootstrap, cluster_by="target_year")
    metrics.extend(a); comparisons.extend(b); decomposition.extend(c)
    pd.DataFrame(metrics).to_csv(args.output/"metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output/"paired_comparisons.csv", index=False)
    pd.DataFrame(decomposition).to_csv(args.output/"change_decomposition.csv", index=False)
    audit.to_csv(args.output/"source_audit.csv")
    panel.to_csv(args.output/"player_seasons.csv", index=False)
    primary.to_csv(args.output/"matched_transitions.csv", index=False)
    display = pd.DataFrame(metrics)
    print(display.loc[display['sample'].eq('primary_250')&display.weighting.eq('equal_pair'),
                      ['metric','pairs','players','pearson','mae_pp','rmse_pp']].to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()

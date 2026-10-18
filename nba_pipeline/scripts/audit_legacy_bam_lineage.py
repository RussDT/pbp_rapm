"""Read-only annual BAM lineage and 2018–2025 native-game direction audit.

No model targets, forecasts, outcomes or tuning are read. Existing frozen
research modules, source files and configurations remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANNUAL = ROOT.parents[1] / "vpm/vpm/data/matchup/matchups_two_rs_2018_2026.parquet"
ARCHIVE = ROOT / "cache/matchup_games_v1"
LINES = ROOT / "results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet"
CONTEXT = ROOT / "results/estimated_skills/monthly_vintage_v1/game_context.parquet"
FIELDS = {"PARTIAL_POSS": "partialPossessions", "MATCHUP_FGA": "matchupFieldGoalsAttempted",
          "MATCHUP_TOV": "matchupTurnovers", "PLAYER_PTS": "playerPoints",
          "MATCHUP_FGM": "matchupFieldGoalsMade", "MATCHUP_FG3A": "matchupThreePointersAttempted",
          "MATCHUP_FTA": "matchupFreeThrowsAttempted"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    value = float(value)
    return value if np.isfinite(value) else None


def compare(annual, native, reverse=False):
    index = ["nba_id_off", "nba_id_def"]
    if reverse:
        native = native.copy()
        native.index = native.index.swaplevel(0, 1)
        # Restore semantic column names after swapping values. Otherwise a
        # named-level join aligns by the old names and silently undoes reversal.
        native.index.names = index
        native = native.sort_index()
    fields = [*FIELDS, "GP"]
    joined = annual[fields].join(native[fields], how="outer", lsuffix="_annual", rsuffix="_native")
    common = joined.filter(like="_annual").notna().all(axis=1) & joined.filter(like="_native").notna().all(axis=1)
    result = {}
    for field in fields:
        a, b = joined[field + "_annual"].fillna(0), joined[field + "_native"].fillna(0)
        difference = a - b
        result[field] = dict(annual_total=float(a.sum()), native_total=float(b.sum()),
            l1_difference=float(difference.abs().sum()),
            normalized_l1=float(difference.abs().sum() / max(1., a.sum())),
            common_pair_correlation=clean(a[common].corr(b[common])),
            common_pair_exact_fraction=float(np.mean(np.isclose(a[common], b[common], atol=1e-8, rtol=0))))
    return dict(common_pairs=int(common.sum()), annual_pairs=len(annual), native_pairs=len(native), fields=result)


def concordance_gate(status_counts, direction, tolerance=.01):
    complete = sum(status_counts.get(status, 0) for status in ("ok", "empty")) == sum(status_counts.values())
    checks = {}
    if direction is not None:
        correct = direction["native_offense_defense"]["fields"]
        reversed_ = direction["reversed_native_ids"]["fields"]
        for field in ("MATCHUP_FGA", "MATCHUP_TOV"):
            checks[field] = bool(correct[field]["normalized_l1"] <= tolerance
                and correct[field]["l1_difference"] <= .2 * reversed_[field]["l1_difference"] + 1e-8)
    passed = bool(complete and checks and all(checks.values()) and status_counts.get("ok", 0) > 0)
    return dict(passed=passed, all_games_resolved=complete, field_checks=checks,
        integer_normalized_l1_tolerance=tolerance,
        direction_rule="correct-orientation L1 <= 20% reversed L1; raw archive-version tolerance only, not model tuning")


def identity_evidence(lines, context):
    """One exact PBP identity per game/player; zero-exposure actors stay ineligible."""
    lines, context = lines.copy(), context.copy()
    for frame in (lines, context):
        frame["game_id"] = frame.game_id.astype(str).str.zfill(10)
        if frame.duplicated(["game_id", "player_id"]).any():
            raise ValueError("Ambiguous PBP game/player identity")
    return lines[["game_id", "player_id", "off_poss", "def_poss"]].merge(
        context[["game_id", "player_id", "team_id", "opp_team_id"]],
        on=["game_id", "player_id"], how="left", validate="one_to_one")


def pair_quality(native, evidence):
    """Row admission is independent of the normalizer's whole-game count flag."""
    native = native.copy()
    numeric = [*FIELDS.values(), "matchupThreePointersMade", "matchupFreeThrowsMade"]
    values = native.reindex(columns=numeric).apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(float)).all(axis=1) & values.ge(0).all(axis=1)
    for made, attempted in [("matchupFieldGoalsMade", "matchupFieldGoalsAttempted"),
                            ("matchupThreePointersMade", "matchupThreePointersAttempted"),
                            ("matchupFreeThrowsMade", "matchupFreeThrowsAttempted"),
                            ("matchupThreePointersAttempted", "matchupFieldGoalsAttempted"),
                            ("matchupThreePointersMade", "matchupFieldGoalsMade")]:
        valid &= values[made].le(values[attempted])
    native["pair_source_counts_valid"] = valid
    native["game_id"] = native.game_id.astype(str).str.zfill(10)
    joined = native
    for side in ("offense", "defense"):
        renamed = evidence.rename(columns={column: (f"{side}_player_id" if column == "player_id" else f"{side}_pbp_{column}")
            for column in evidence if column != "game_id"})
        joined = joined.merge(renamed, on=["game_id", f"{side}_player_id"], how="left", validate="many_to_one")
    native["pair_identity_verified"] = (
        joined.offense_pbp_off_poss.gt(0) & joined.defense_pbp_def_poss.gt(0)
        & joined.offense_team_id.eq(joined.offense_pbp_team_id)
        & joined.defense_team_id.eq(joined.defense_pbp_team_id)
        & joined.offense_team_id.ne(joined.defense_team_id)
        & joined.offense_pbp_opp_team_id.eq(joined.defense_team_id)
        & joined.defense_pbp_opp_team_id.eq(joined.offense_team_id)).to_numpy()
    return native


def aggregate_native(native):
    grouped = native.groupby(["offense_player_id", "defense_player_id"])
    aggregated = grouped[list(FIELDS.values())].sum().rename(columns={v: k for k, v in FIELDS.items()})
    aggregated["GP"] = grouped.game_id.nunique()
    aggregated["pair_game_rows"] = grouped.size()
    for flag, count in [("pair_source_counts_valid", "valid_count_rows"),
                        ("pair_identity_verified", "verified_identity_rows")]:
        aggregated[flag] = grouped[flag].all()
        aggregated[count] = grouped[flag].sum()
    aggregated.index.names = ["nba_id_off", "nba_id_def"]
    return aggregated.loc[aggregated.MATCHUP_FGA.gt(0)]


def native_season(archive, census, annual, year, sources, output, evidence):
    selected = census.loc[census.season.eq(year) & census.season_type.eq("rs")]
    parts, statuses = [], []
    pending, errored = [], []
    for row in selected.itertuples():
        meta = archive / "metadata" / f"{row.game_id}.json"
        if not meta.exists():
            statuses.append("pending"); pending.append(row.game_id); continue
        record = json.loads(meta.read_text())
        status = record.get("status", "pending")
        statuses.append(status)
        sources[str(meta.resolve())] = sha(meta)
        if status == "empty":
            raw = archive / "raw" / f"{row.game_id}.json"
            if sha(raw) != record["sha256"]:
                raise ValueError(f"Empty native response changed: {row.game_id}")
            sources[str(raw.resolve())] = sha(raw)
            continue
        if status != "ok":
            errored.append(row.game_id); continue
        path = archive / "pairs" / f"{row.game_id}.parquet"
        raw = archive / "raw" / f"{row.game_id}.json"
        if sha(raw) != record["sha256"]:
            raise ValueError(f"Native response changed: {row.game_id}")
        d = pd.read_parquet(path)
        if (not d.season.eq(year).all() or not d.season_type.eq("rs").all()
                or not pd.to_datetime(d.game_date).eq(pd.Timestamp(row.game_date)).all()
                or not d.game_id.astype(str).eq(row.game_id).all()):
            raise ValueError("Native RS history contains mixed identity/phase/year")
        parts.append(d)
        for source in (path, raw):
            sources[str(source.resolve())] = sha(source)
    status_counts = pd.Series(statuses).value_counts().to_dict()
    if not parts:
        return dict(year=year, census_games=len(selected), status_counts=status_counts,
            pending_games=pending, error_games=errored, direction=None,
            gate=concordance_gate(status_counts, None)), None
    native = pair_quality(pd.concat(parts, ignore_index=True), evidence)
    aggregated = aggregate_native(native)
    old = annual.loc[annual.year.eq(year)].set_index(["nba_id_off", "nba_id_def"])
    direction = {"native_offense_defense": compare(old, aggregated),
                 "reversed_native_ids": compare(old, aggregated, True)}
    artifact = output / f"native_{year}_rs_pairs_verified.parquet"
    aggregated.reset_index().to_parquet(artifact, index=False)
    result = dict(year=year, census_games=len(selected), status_counts=status_counts,
        pending_games=pending, error_games=errored, pair_game_rows=len(native),
        raw_count_invalid_rows=int((~native.pair_source_counts_valid).sum()),
        identity_unverified_rows=int((~native.pair_identity_verified).sum()),
        aggregate_pairs_count_valid=int(aggregated.pair_source_counts_valid.sum()),
        aggregate_pairs_identity_verified=int(aggregated.pair_identity_verified.sum()),
        gamewide_source_invalid_flag_rows=int((~native.source_counts_valid).sum()) if "source_counts_valid" in native else None,
        source_validity_note="source_counts_valid is a whole-game flag, not a per-pair mask; invalid raw rows retained for lineage comparison",
        direction=direction, gate=concordance_gate(status_counts, direction))
    return result, artifact


def audit(annual_path=ANNUAL, archive=ARCHIVE, output=None, years=range(2018, 2026)):
    annual_path, archive = Path(annual_path), Path(archive)
    output = Path(output) if output else ROOT / "results/legacy_bam_lineage_audit_v1"
    output.mkdir(parents=True, exist_ok=True)
    years = list(years)
    if not years or any(year not in range(2018, 2026) for year in years):
        raise ValueError("Only complete annual source years2018–2025;2026 excluded")
    annual = pd.read_parquet(annual_path)
    if annual.duplicated(["year", "nba_id_off", "nba_id_def"]).any() or annual.year.max() > 2026:
        raise ValueError("Ambiguous/protected annual identity")
    if not annual.playoffs.eq(0).all():
        raise ValueError("Annual source contains mixed phases")
    source_manifest = annual_path.with_suffix(".manifest.json")
    if json.loads(source_manifest.read_text())["sha256"] != sha(annual_path):
        raise ValueError("Pinned annual archive hash changed")
    census_path = archive / "game_census.parquet"
    census = pd.read_parquet(census_path)
    sources = {str(p.resolve()): sha(p) for p in [annual_path, source_manifest, census_path, LINES, CONTEXT, Path(__file__)]}
    lines = pd.read_parquet(LINES)
    evidence = identity_evidence(lines, pd.read_parquet(CONTEXT))
    details, artifacts = [], {}
    for year in years:
        detail, artifact = native_season(archive, census, annual, year, sources, output, evidence)
        details.append(detail)
        if artifact is not None:
            artifacts[artifact.name] = sha(artifact)
    formula = []
    for year, d in annual.groupby("year"):
        expected_bam = d["Effective_Matchup_aTS%"] * d.Weighted_Offensive_Score * (1 + d.boost) * 10
        expected_total = d.groupby("nba_id_def").MATCHUP_FGA.transform("sum")
        formula.append(dict(year=int(year), rows=len(d), max_gp=int(d.GP.max()),
            stored_numeric_precision="unrounded_float" if year < 2025 else "three_decimal_archive",
            formula_reconstruction_note="exact_algebra_expected" if year < 2025 else "reported_gap_from_multiplying_rounded_fields; rebuilt_BAM_must_use_unrounded_native_counts",
            position_missing=int(d.POSITION.isna().sum()), defender_position_missing=int(d.DEF_POSITION.isna().sum()),
            max_effective_fga_gap=float((d.MATCHUP_FGA + 1.05 * d.MATCHUP_TOV - d.Effective_MATCHUP_FGA).abs().max()),
            max_tsa_gap=float((d.MATCHUP_FGA + .44 * d.MATCHUP_FTA - d.denom).abs().max()),
            max_bam_algebra_gap=float((expected_bam - d.BAM).abs().max()),
            max_defender_fga_denominator_gap=float((expected_total - d.TOTAL_FGA_DEFENDER).abs().max()),
            max_bam_sign_gap=float((d.BAM + d.BAM_inverted).abs().max()),
            season_ids=sorted(map(int, d.SEASON_ID.unique()))))
    lines_path = LINES
    counts = lines.loc[lines.season_type.eq("rs")].groupby(["season", "player_id"])[["pts", "fga", "fta"]].sum()
    counts["observed_ts"] = counts.pts / (2 * (counts.fga + .44 * counts.fta))
    counts.index.names = ["year", "nba_id_off"]
    ts = annual.groupby(["year", "nba_id_off"]).TS_PCT.first().to_frame().join(counts.observed_ts)
    ts["gap"] = ts.TS_PCT - ts.observed_ts
    ts_checks = [dict(year=int(y), matched=len(g.dropna()),
        median_abs_gap=clean(g.gap.abs().median()), max_abs_gap=clean(g.gap.abs().max()))
        for y, g in ts.groupby(level=0)]
    sources[str(lines_path.resolve())] = sha(lines_path)
    # Independent archived Synergy totals corroborate the calendar meaning of
    # Offensive_Score; per-game endpoint POSS is rounded to one decimal. A few
    # rows differ, so this is corroboration, not exact source reconstruction.
    synergy_path = Path("/Users/russ/Documents/Codex/basketball/data/csvs/playtype (1).csv")
    synergy_checks = []
    if synergy_path.exists():
        synergy = pd.read_csv(synergy_path)
        synergy = synergy.loc[synergy.PLAY_TYPE.isin(["Isolation", "Postup", "PRBallHandler"])
            & synergy.PPP.gt(0) & synergy.POSS.gt(0)].copy()
        synergy["candidate_score"] = synergy.PPP * (synergy.POSS / synergy.GP).round(1)
        scores = synergy.groupby(["year", "nba_id"]).candidate_score.sum()
        scores.index.names = ["year", "nba_id_off"]
        observed = annual.groupby(["year", "nba_id_off"]).Offensive_Score.first().to_frame().join(scores)
        for year, group in observed.groupby(level=0):
            group = group.dropna()
            if len(group):
                gap = (group.Offensive_Score - group.candidate_score).abs()
                synergy_checks.append(dict(year=int(year), matched=len(group),
                    exact_reproductions=int(gap.le(1e-7).sum()), median_abs_gap=float(gap.median()), max_abs_gap=float(gap.max())))
        sources[str(synergy_path)] = sha(synergy_path)
    positions = annual.loc[annual.year.lt(2026)].drop_duplicates(["year", "nba_id_off"])
    stability = positions.groupby("nba_id_off").agg(years=("year", "size"), positions=("POSITION", "nunique"))
    result = dict(status="source_audit_only_no_model_runs", annual_rows=len(annual),
        requested_native_years=years, native_year_details=details,
        annual_formula_audits=formula, annual_same_year_ts_against_observed_rs=ts_checks,
        same_year_synergy_offensive_score_corroboration=synergy_checks,
        position_history=dict(multi_year_players=int(stability.years.gt(1).sum()),
            multi_year_players_with_position_change=int((stability.years.gt(1) & stability.positions.gt(1)).sum()),
            source_code="PlayerIndex(historical=1) current profile; not season-specific dated position history"),
        admission=dict(conditional_research_control=True, certified_historical_publication_vintage=False,
            required_rules=["source year strictly below forecast ending season", "2026 annual source excluded", 
                "separately name lagged archival legacy BAM; do not conflate conditional-FG matchup_bam",
                "certify later annual-year direction against native aggregates before opening their forecasts"],
            limitations=["No within-season dates exist; only completed-season forecast cuts are admissible.",
                "Static profile position boosts are retrospective; strict as-of position vintage is not established.",
                "Provider FGA/FTA assignment coverage is nonexclusive/era-specific, inherited by the legacy formula.",
                "2025 quantities are rounded, preventing high-precision algebra reconstruction.",
                "2026 has no positions/boosts and incomplete outcome/TS evidence; never emit2027 from it."]),
        sources=sources, artifact_hashes=artifacts,
        protected_outcomes=0)
    audit_path = output / "all_year_lineage_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    verified = [detail["year"] for detail in details if detail["gate"]["passed"]]
    gate = dict(status="passed_source_concordance" if len(verified) == len(years) else "blocked_incomplete_or_mismatched_source",
        annual_sha256=sha(annual_path), requested_years=years, verified_years=verified,
        fully_resolved_years=[detail["year"] for detail in details if detail["gate"]["all_games_resolved"]],
        per_year_details=details, audit_path=str(audit_path.resolve()), audit_sha256=sha(audit_path),
        source_hashes=sources, artifacts=artifacts, protected_outcomes=0)
    (output / "annual_native_concordance.json").write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: gate[key] for key in ["status", "requested_years", "verified_years", "fully_resolved_years"]}))
    result["gate"] = gate
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annual", type=Path, default=ANNUAL)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--out", type=Path)
    p.add_argument("--years", default="2018,2019,2020,2021,2022,2023,2024,2025")
    p.add_argument("--allow-incomplete", action="store_true", help="Write a blocked partial audit without exit2; never marks missing/error years verified")
    args = p.parse_args()
    result = audit(args.annual, args.archive, args.out, map(int, args.years.split(",")))
    if result["gate"]["status"] != "passed_source_concordance" and not args.allow_incomplete:
        raise SystemExit(2)

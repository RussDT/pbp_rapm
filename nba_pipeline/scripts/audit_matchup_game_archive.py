"""Independent native-JSON/normalized-pair and field-coverage matchup audit.

This auditor never assumes historical matchup FT/shot coverage is complete or
exclusive. Identity/direction fidelity is a hard gate; field coverage is
reported explicitly against exact observed PBP player-game totals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["game_id", "offense_player_id", "defense_player_id"]
COUNT_PAIRS = [("matchupFieldGoalsMade", "matchupFieldGoalsAttempted"),
               ("matchupThreePointersMade", "matchupThreePointersAttempted"),
               ("matchupFreeThrowsMade", "matchupFreeThrowsAttempted"),
               ("helpFieldGoalsMade", "helpFieldGoalsAttempted")]
BOX_FIELDS = {"playerPoints": "pts", "matchupFieldGoalsAttempted": "fga",
              "matchupFieldGoalsMade": "fgm", "matchupFreeThrowsAttempted": "fta",
              "matchupFreeThrowsMade": "ftm"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(values):
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def native_rows(payload, game):
    box = payload.get("boxScoreMatchups", {})
    if str(box.get("gameId", "")).zfill(10) != game["game_id"]:
        raise ValueError("Native game ID differs from census")
    teams = [box.get("homeTeam", {}), box.get("awayTeam", {})]
    rosters = [{int(p["personId"]) for p in t.get("players", [])} for t in teams]
    if rosters[0] & rosters[1]:
        raise ValueError("A native player appears on both opposing rosters")
    rows, unrostered = [], []
    for side in (0, 1):
        team, opponent = teams[side], teams[1 - side]
        if not team.get("players", []):
            continue
        tid, oid = int(team.get("teamId", 0)), int(opponent.get("teamId", 0))
        if tid <= 0 or oid <= 0 or tid == oid:
            raise ValueError("Invalid native opposing team identity")
        official = game.get("home_team_id" if side == 0 else "away_team_id", 0)
        if pd.notna(official) and int(official) > 0 and int(official) != tid:
            raise ValueError("Official/native team identity differs")
        for outer in team.get("players", []):
            attacker = int(outer["personId"])
            for inner in outer.get("matchups", []):
                defender = int(inner["personId"])
                if defender in rosters[side]:
                    raise ValueError("Nested defender belongs to attacker's native team")
                if defender not in rosters[1 - side] or attacker == defender:
                    unrostered.append((game["game_id"], attacker, defender))
                rows.append(dict(game_id=game["game_id"], offense_player_id=attacker,
                    defense_player_id=defender, offense_team_id=tid, defense_team_id=oid,
                    **inner.get("statistics", {})))
    frame = pd.DataFrame(rows)
    if len(frame) and frame.duplicated(KEYS).any():
        raise ValueError("Duplicate native attacker/defender pair")
    frame.attrs["unrostered_keys"] = unrostered
    return frame


def audit_pair_fidelity(payload, normalized, game):
    native = native_rows(payload, game)
    if len(native) != len(normalized):
        raise ValueError("Native/normalized pair count differs")
    if not len(native):
        return dict(rows=0, source_count_defects=[], time_max_difference=0.)
    normalized = normalized.copy()
    normalized["game_id"] = ids(normalized.game_id)
    if normalized.duplicated(KEYS).any():
        raise ValueError("Duplicate normalized pair")
    if not pd.to_datetime(normalized.game_date).eq(pd.Timestamp(game["game_date"])).all():
        raise ValueError("Normalized game date differs from exact census")
    if not normalized.season.eq(game["season"]).all() or not normalized.season_type.eq(game["season_type"]).all():
        raise ValueError("Normalized ending season/phase differs")
    if int(game["season"]) >= 2027:
        raise ValueError("Protected2027 outcome")
    a, b = (d.set_index(KEYS, verify_integrity=True).sort_index() for d in (native, normalized))
    if not a.index.equals(b.index):
        raise ValueError("Native offense/defense orientation or player IDs changed")
    if "defender_in_opposing_native_roster" in b:
        expected = ~b.index.isin(native.attrs["unrostered_keys"])
        np.testing.assert_array_equal(expected, b.defender_in_opposing_native_roster)
    for column in a:
        if column not in b:
            raise ValueError(f"Native statistic disappeared: {column}")
        left, right = a[column], b[column]
        if column == "matchupMinutes":
            if not left.fillna("").astype(str).equals(right.fillna("").astype(str)):
                raise ValueError("Native matchup time text changed")
        else:
            np.testing.assert_allclose(pd.to_numeric(left), pd.to_numeric(right), atol=0, rtol=0, equal_nan=True)
    defects = []
    for made, attempts in COUNT_PAIRS:
        if made in b and attempts in b and (b[made] > b[attempts]).any():
            defects.append(made + ">" + attempts)
    for column in a:
        if column != "matchupMinutes" and pd.to_numeric(a[column]).lt(0).any():
            defects.append(column + "<0")
    for smaller, larger in [("matchupThreePointersAttempted", "matchupFieldGoalsAttempted"),
                            ("matchupThreePointersMade", "matchupFieldGoalsMade")]:
        if smaller in b and larger in b and b[smaller].gt(b[larger]).any():
            defects.append(smaller + ">" + larger)
    gap = 0.
    if "matchupMinutes" in b and "matchupMinutesSort" in b:
        clock = b.matchupMinutes.astype(str).str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
        seconds = pd.to_numeric(clock[0]) * 60 + pd.to_numeric(clock[1])
        gap = float((seconds - b.matchupMinutesSort).abs().max())
        if seconds.isna().any() or gap > 1.000001:
            raise ValueError("Display clock and numeric matchup seconds disagree")
    return dict(rows=len(b), source_count_defects=sorted(defects), time_max_difference=gap,
                unrostered_defender_rows=len(native.attrs["unrostered_keys"]))


def field_coverage(pairs, box):
    if box.player_id.duplicated().any():
        raise ValueError("PBP player-game reference is ambiguous")
    reference = box.set_index("player_id")
    records = []
    for field, target in BOX_FIELDS.items():
        if field not in pairs or not pairs[field].notna().any():
            records.append(dict(field=field, status="absent", tracked=None,
                observed=float(reference[target].sum()), offense_absolute_gap=None, reversed_absolute_gap=None))
            continue
        correct = pairs.groupby("offense_player_id")[field].sum(min_count=1)
        reversed_ = pairs.groupby("defense_player_id")[field].sum(min_count=1)
        universe = sorted(set(reference.index) | set(correct.index) | set(reversed_.index))
        observed = reference[target].reindex(universe).fillna(0)
        records.append(dict(field=field, status="measured", tracked=float(correct.sum()),
            observed=float(observed.sum()), source_nonmissing_rows=int(pairs[field].notna().sum()),
            source_missing_rows=int(pairs[field].isna().sum()),
            offense_absolute_gap=float((correct.reindex(universe).fillna(0) - observed).abs().sum()),
            reversed_absolute_gap=float((reversed_.reindex(universe).fillna(0) - observed).abs().sum())))
    return records


def audit(root, output=None):
    root = Path(root)
    output = Path(output) if output else ROOT / "results/matchup_game_archive_audit_v1"
    census_path = root / "game_census.parquet"
    lines_path = ROOT / "results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet"
    census = pd.read_parquet(census_path)
    census["game_id"] = ids(census.game_id)
    if census.game_id.duplicated().any() or census.season.gt(2026).any():
        raise ValueError("Invalid/protected census")
    games = census.set_index("game_id")
    lines = pd.read_parquet(lines_path)
    lines["game_id"] = ids(lines.game_id)
    lines = lines.loc[(lines.off_poss + lines.def_poss).gt(0)]
    lookup = {game: group for game, group in lines.groupby("game_id", sort=False)}
    sources = {str(p.resolve()): sha(p) for p in [census_path, lines_path, Path(__file__)]}
    records, coverage = [], []
    for receipt_path in sorted((root / "metadata").glob("*.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") not in ("ok", "empty"):
            continue
        gid = receipt_path.stem
        if gid not in games.index:
            raise ValueError("Cached game is outside frozen census")
        game = games.loc[gid].to_dict()
        game["game_id"] = gid
        raw = root / "raw" / f"{gid}.json"
        if sha(raw) != receipt["sha256"]:
            raise ValueError("Cached native response changed")
        pair_path = root / "pairs" / f"{gid}.parquet"
        pairs = pd.read_parquet(pair_path) if pair_path.exists() else pd.DataFrame()
        result = audit_pair_fidelity(json.loads(raw.read_text()), pairs, game)
        result.update(game_id=gid, season=int(game["season"]), season_type=game["season_type"],
            game_date=str(pd.Timestamp(game["game_date"]).date()), in_pbp=gid in lookup)
        records.append(result)
        for path in [receipt_path, raw, *([pair_path] if pair_path.exists() else [])]:
            sources[str(path.resolve())] = sha(path)
        if len(pairs) and gid in lookup:
            for record in field_coverage(pairs, lookup[gid]):
                coverage.append(dict(game_id=gid, season=int(game["season"]), **record))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    fields = pd.DataFrame(coverage)
    frame.to_parquet(output / "game_audits.parquet", index=False)
    fields.to_parquet(output / "field_coverage.parquet", index=False)
    summary = dict(status="passed_native_identity_fidelity; field_coverage_retained_as_measurement",
        audited_games=len(frame), cached_pair_rows=int(frame.rows.sum()) if len(frame) else 0,
        pending_games=len(census) - len(frame),
        source_defect_games=int(frame.source_count_defects.map(bool).sum()) if len(frame) else 0,
        unrostered_defender_rows=int(frame.get("unrostered_defender_rows", pd.Series(dtype=float)).sum()),
        sources=sources, audited_at=datetime.now(timezone.utc).isoformat(),
        limitations=["Current download snapshot may be incomplete; pending games are not zero observations.",
            "Native JSON defines attacker/defender direction; box gaps are independent corroboration.",
            "Tracked FGA/FTA/points need not equal full/exclusive box totals; keep field coverage.",
            "Source publication and retrospective measurement vintages remain uncertified."],
        artifacts={name: sha(output / name) for name in ["game_audits.parquet", "field_coverage.parquet"]},
        protected_outcomes=0)
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ["audited_games", "cached_pair_rows", "pending_games", "source_defect_games"]}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "cache/matchup_games_v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    audit(args.root, args.out)

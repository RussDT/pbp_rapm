"""Pin a query-preserving, conservatively masked pregame observation source.

Independently recount typed raw events and terminal scoreboards for every
2014--2026 source game, including the upstream 2014--2015 warm-up. Counts are
never repaired here: uncertain observations become missing, while identities,
query dates, validated possession exposures, minutes and legacy PF survive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from audit_pregame_target_scoreboards import normal, terminal_scores
from build_full_skill_history import write_json
from prepare_full_skill_history import sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/estimated_skills/pregame_source_validated_v1"
OUT = ROOT / "results/estimated_skills/pregame_source_validated_v2"
FG = ["fga", "fgm", "rim_fga", "rim_fgm", "mid_fga", "mid_fgm", "three_fga", "three_fgm"]
FT = ["fta", "ftm"]
SCORING = FG + FT + ["pts"]
TAG_PATTERNS = {"ast": r"\bAST\)", "stl": r"\(\d+ STL\)", "blk": r"\(\d+ BLK\)"}
RAW_COLUMNS = ["game_id", "event_num", "event_type", "period", "minute_game", "score",
               "player1_id", "home_description", "visitor_description", "neutral_description"]
KEYS = ["game_id", "player_id"]


def recount_raw(raw):
    """Independent typed-event counts, not an alternative player-stat export.

    Explicit MISS is a word. A FT with neither MISS nor a PTS tag has unknown
    outcome even when the provider's event type is 3. AST/STL/BLK tags lack
    reliable player IDs in these archives, so only game totals are compared.
    Team rebounds/turnovers are not personal-stat observations.
    """
    d = raw.copy()
    d["game_id"] = normal(d.game_id)
    text = pd.Series("", index=d.index)
    for field in ["home_description", "visitor_description", "neutral_description"]:
        text = text + " " + d[field].fillna("").astype(str)
    text = text.str.upper()
    event = pd.to_numeric(d.event_type, errors="coerce")
    pid = pd.to_numeric(d.player1_id, errors="coerce")
    personal = pid.between(1, 99999999)
    missed = text.str.contains(r"\bMISS\b", regex=True)
    made_tag = text.str.contains(r"\(\d+ PTS\)", regex=True)
    d["player_id"] = pid
    d["fga"] = event.isin([1, 2]).astype(int)
    d["fgm"] = event.eq(1).astype(int)
    d["three_fga"] = (event.isin([1, 2]) & text.str.contains(r"\b3PT\b")).astype(int)
    d["three_fgm"] = (event.eq(1) & text.str.contains(r"\b3PT\b")).astype(int)
    d["fta"] = event.eq(3).astype(int)
    d["ftm"] = (event.eq(3) & ~missed & made_tag).astype(int)
    d["pts"] = 2*d.fgm + d.three_fgm + d.ftm
    d["tov"] = (event.eq(5) & personal).astype(int)
    d["reb"] = (event.eq(4) & personal).astype(int)
    d["uncertain_ft"] = (event.eq(3) & ~missed & ~made_tag).astype(int)
    d["unowned_scoring"] = (event.isin([1, 2, 3]) & ~personal).astype(int)
    for field, pattern in TAG_PATTERNS.items():
        d[field] = text.str.contains(pattern, regex=True).astype(int)
    fields = ["fga", "fgm", "three_fga", "three_fgm", "fta", "ftm", "pts", "tov", "reb"]
    players = d.loc[personal].groupby(KEYS, sort=False)[fields].sum()
    games = d.groupby("game_id", sort=False)[fields + list(TAG_PATTERNS) +
                  ["uncertain_ft", "unowned_scoring"]].sum()
    games = games.rename(columns={c: "raw_" + c for c in games})
    uncertain_players = d.loc[d.uncertain_ft.gt(0) & personal, KEYS].drop_duplicates()
    observed_personal = d.loc[personal & event.isin([1, 2, 3, 4, 5]), KEYS].drop_duplicates()
    availability = {name: bool(d[name].sum()) for name in TAG_PATTERNS}
    return players, games, uncertain_players, observed_personal, availability


def audit_phase(source, raw, season, phase):
    """Return game evidence and localized masks without mutating the source."""
    s = source.copy()
    s["game_id"] = normal(s.game_id)
    if s.duplicated(KEYS).any():
        raise ValueError("Duplicate source player/game")
    raw_players, raw_games, uncertain, observed, available = recount_raw(raw)
    source_players = s.set_index(KEYS)
    sums = s.groupby("game_id")[SCORING + ["ast", "tov", "oreb", "dreb", "stl", "blk"]].sum(min_count=1)
    final = terminal_scores(raw).set_index("game_id")
    games = sums.rename(columns={c: "source_" + c for c in sums}).join(raw_games).join(final)
    # A terminal 0--0 placeholder is unresolved, never a basketball final.
    games["terminal_score_original"] = games.terminal_score
    games["terminal_score"] = games.terminal_score.where(games.terminal_score.gt(0))
    games["source_scoreboard_gap"] = games.source_pts-games.terminal_score
    games["raw_scoreboard_gap"] = games.raw_pts-games.terminal_score
    games["scoreboard_status"] = np.select(
        [games.terminal_score.isna(), games.raw_scoreboard_gap.abs().gt(1e-8)],
        ["terminal_unresolved", "raw_scoring_disagrees"], default="raw_scoring_reconciled")
    games["core_scoring_query_only"] = ~games.scoreboard_status.eq("raw_scoring_reconciled")
    games["season"] = int(season)
    games["phase"] = phase
    masks = []

    def mark(indices, fields, reason):
        for gid, pid in indices:
            for field in fields:
                if (gid, pid) in source_players.index:
                    value = source_players.loc[(gid, pid), field]
                    if pd.notna(value):
                        masks.append(dict(game_id=gid, player_id=int(pid), field=field,
                                          reason=reason, source_value=float(value)))

    for reason in ["terminal_unresolved", "raw_scoring_disagrees"]:
        selected = set(games.index[games.scoreboard_status.eq(reason)])
        mark(source_players.index[source_players.index.get_level_values("game_id").isin(selected)],
             SCORING, reason)
    # An unowned scoring event or an unknown FT outcome cannot be localized to
    # a complete personal ledger, even if two errors happen to offset in sum.
    for flag, reason in [("raw_unowned_scoring", "raw_scoring_ownership_unknown"),
                         ("raw_uncertain_ft", "raw_free_throw_outcome_unknown")]:
        selected = set(games.index[games[flag].gt(0)])
        mark(source_players.index[source_players.index.get_level_values("game_id").isin(selected)],
             SCORING if "ownership" in reason else FT + ["pts"], reason)
    # Raw event actors absent from the fixed census cannot safely be assigned
    # to a different player. The relevant game families remain query-only.
    foreign = observed.merge(s[KEYS].assign(in_source=True), on=KEYS, how="left", validate="one_to_one")
    foreign = foreign.loc[foreign.in_source.isna()]
    foreign_games = set(foreign.game_id)
    if foreign_games:
        mark(source_players.index[source_players.index.get_level_values("game_id").isin(foreign_games)],
             SCORING + ["tov", "oreb", "dreb"], "personal_raw_actor_absent_from_source_census")
    # Full outer identity comparison catches missing personal events and false
    # zero observations. A player with no typed events has real zero counts.
    compare = source_players.join(raw_players.rename(columns={c:c+"_raw" for c in raw_players}), how="left")
    for field in raw_players:
        compare[field + "_raw"] = compare[field + "_raw"].fillna(0)
    for fields, affected, reason in [
        (["fga", "fgm", "three_fga", "three_fgm"], FG + ["pts"], "raw_field_goal_identity_count_mismatch"),
        (["fta", "ftm"], FT + ["pts"], "raw_free_throw_identity_count_mismatch"),
        (["tov"], ["tov"], "raw_turnover_identity_count_mismatch"),
    ]:
        mismatch = pd.Series(False, index=compare.index)
        for field in fields:
            mismatch |= compare[field].sub(compare[field+"_raw"]).abs().gt(1e-8)
        mark(compare.index[mismatch], affected, reason)
    rebound = compare.oreb.add(compare.dreb).sub(compare.reb_raw).abs().gt(1e-8)
    mark(compare.index[rebound], ["oreb", "dreb"], "raw_rebound_identity_count_mismatch")
    # These tags do not expose exact assisting/defending IDs. A mismatch masks
    # the affected family for the game; no invented reassignment or zeros.
    for field in TAG_PATTERNS:
        games[field + "_raw_tag_available"] = available[field]
        if not available[field]:
            selected = set(games.index)
            reason = "raw_" + field + "_tags_unavailable"
        else:
            selected = set(games.index[games["source_"+field].sub(games["raw_"+field]).abs().gt(1e-8)])
            reason = "raw_" + field + "_game_tag_count_mismatch"
        mark(source_players.index[source_players.index.get_level_values("game_id").isin(selected)],
             [field], reason)
    columns = ["game_id", "player_id", "field", "reason", "source_value"]
    mask = pd.DataFrame(masks, columns=columns).drop_duplicates(KEYS + ["field", "reason"])
    games["source_player_rows"] = s.groupby("game_id").size()
    games["source_personal_actor_absent"] = games.index.isin(foreign_games)
    return games.reset_index(), mask


def apply_masks(source, masks):
    """Mask only explicitly named observations; leave every query intact."""
    if not set(masks.field).issubset(set(SCORING + ["ast", "tov", "oreb", "dreb", "stl", "blk"])):
        raise ValueError("Mask attempts to change protected identity/exposure/PF fields")
    d = source.copy()
    indexed = d.set_index(KEYS)
    for field, group in masks.groupby("field"):
        keys = pd.MultiIndex.from_frame(group[KEYS].drop_duplicates())
        if not keys.isin(indexed.index).all():
            raise ValueError("Mask refers to a nonexistent query")
        indexed.loc[keys, field] = np.nan
    d = indexed.reset_index()[source.columns]
    if len(d) != len(source) or not d[KEYS].equals(source[KEYS]):
        raise ValueError("Observation masking changed query identities/order")
    protected = [c for c in source if c not in set(masks.field)]
    pd.testing.assert_frame_equal(d[protected], source[protected], check_exact=True)
    # Explicitly retain legacy PF, even if future callers add more maskable fields.
    pd.testing.assert_series_equal(d.pf_committed, source.pf_committed, check_exact=True)
    return d


def build(source=SOURCE, out=OUT):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new immutable source output directory")
    manifest_path = source / "manifest.json"
    original = json.loads(manifest_path.read_text())
    hashes = {str(manifest_path): sha(manifest_path), str(Path(__file__).resolve()): sha(__file__),
              str(Path(__file__).with_name("audit_pregame_target_scoreboards.py")): sha(Path(__file__).with_name("audit_pregame_target_scoreboards.py"))}
    for name, expected in original["artifacts"].items():
        path = source / name
        if sha(path) != expected:
            raise ValueError("Pinned input artifact changed: " + name)
        hashes[str(path)] = expected
    frame = pd.read_parquet(source / "player_game_lines.parquet")
    frame.game_id = normal(frame.game_id)
    if not frame.season.between(2014, 2026).all() or frame.duplicated(KEYS).any():
        raise ValueError("Protected season or duplicate identity in source")
    game_tables, mask_tables = [], []
    for year in range(2014, 2027):
        for phase, suffix in [("RS", ""), ("PS", "_PS")]:
            path = ROOT / "raw_data" / f"NBA{year%100:02d}{suffix}.parquet"
            raw = pd.read_parquet(path, columns=RAW_COLUMNS)
            selected = frame.loc[frame.season.eq(year) & frame.season_type.str.upper().eq(phase)]
            games, masks = audit_phase(selected, raw, year, phase)
            game_tables.append(games)
            mask_tables.append(masks)
            hashes[str(path)] = sha(path)
            print(json.dumps(dict(season=year, phase=phase, games=len(games),
                                  unresolved=int(games.terminal_score.isna().sum()), masks=len(masks))), flush=True)
    games = pd.concat(game_tables, ignore_index=True)
    masks = pd.concat(mask_tables, ignore_index=True)
    if games.game_id.duplicated().any() or len(games) != frame.game_id.nunique():
        raise ValueError("Source games are not audited exactly once")
    result = apply_masks(frame, masks)
    # Recheck all parents immediately before writing the immutable package.
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("A pinned parent changed during the audit")
    out.mkdir(parents=True)
    result.to_parquet(out / "player_game_lines.parquet", index=False)
    shutil.copyfile(source / "game_context.parquet", out / "game_context.parquet")
    games.to_csv(out / "completeness_games.csv", index=False)
    masks.to_csv(out / "observation_masks.csv", index=False)
    unique = masks.drop_duplicates(KEYS + ["field"])
    summary = dict(status="prepared_query_preserving_observation_completeness_source",
        rows=len(result), first_season=2014, last_season=2026, source_games=len(games),
        warmup_games=int(games.season.le(2015).sum()),
        source_point_ledger_incomplete_or_unresolved_games=int((games.terminal_score.isna() | games.source_scoreboard_gap.abs().gt(1e-8)).sum()),
        full_scoring_query_only_games=int(games.core_scoring_query_only.sum()),
        masked_player_game_fields=len(unique), masked_player_game_rows=len(unique[KEYS].drop_duplicates()),
        masked_fields={str(k): int(v) for k,v in unique.groupby("field").size().items()},
        reasons={str(k): int(v) for k,v in masks.groupby("reason").size().items()},
        preserved_query_rows=True, preserved_possession_exposure=True, preserved_minutes=True,
        personal_foul_counter_missing=int(result.pf_committed.isna().sum()),
        pf_definition=original["pf_definition"],
        estimator_change="v1_validated_counts_with_conservative_observation_masks_and_transported_annual_parameters_not_refitted",
        selected_vpm_changed=False, model_fitted=False, protected_2027_rows=0,
        source_directory=str(source), source_hashes=hashes,
        mask_policy="typed raw actor counts and exact terminal scoreboards; no count repair or query removal",
        limitations=["Terminal score strings can be missing/placeholders; no numeric maxima or partial snapshot promoted to a final",
          "Raw event recount is an independent extraction of the same archive, not an independently collected official box score",
          "AST/STL/BLK tag totals validate aggregate counts, not every personal attribution; absent field tags remain unavailable",
          "Rebound audit checks personal total only, not offensive/defensive classification",
          "Matching totals do not prove all underlying events, coordinate zones or player attributions correct",
          "Source publication vintages remain uncertified; legacy PF semantics retained without a new subtype-based recount",
          "Scoring disagreements do not invalidate independently measured tracking/minutes; existing unsafe exposures remain missing"],
        artifacts={name: sha(out/name) for name in ["player_game_lines.parquet", "game_context.parquet", "completeness_games.csv", "observation_masks.csv"]})
    write_json(out / "manifest.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,(dict,list))}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    build(args.source, args.out)

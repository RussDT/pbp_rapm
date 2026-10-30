"""Per-player-game fouls drawn (offensive fouls, charges, all personal fouls) from the CDN foul ledger.

Reads ``foul_ledger.parquet`` and ``game_status.csv`` written by
``fetch_fouls_drawn_cdn.py``, aggregates the ``foulDrawnPersonId`` side per
player-game, and reconciles every game before any zero is written:

1. internal identity: every non-technical foul names exactly one drawer, so
   the game's drawn total equals its committed total, and no drawer is outside
   the game's PBP player universe;
2. committed side versus the repo PBP ``pf_committed`` in
   ``player_game_lines.parquet`` (same official feed, independent parse). The
   repo counter reads the ``(Pn)`` ledger tag, which double fouls do not carry
   in the modern ``Foul : Double Personal - A (n PF), B (m PF)`` row, so a
   player's CDN count may exceed the PBP count by at most their CDN double
   fouls; any other disagreement fails this check;
3. official box: the local game-report archive
   (``player_sheets/game_report/{season}/{game_id}.csv``) must agree with the
   CDN on both ``PF`` and ``PFD`` for every player it shares with the game's
   PBP universe. The archive files traded players under a later team, so a
   file can carry a player from another game that day and omit one who played;
   only shared players are compared.

A game is measured when identity holds and either the PBP committed check
passes with the archive absent or agreeing, or the PBP check fails but the
archive agrees on every player with any CDN foul activity (the official box
arbitrates repo-PBP parse gaps such as a missing foul event). A game that was
not fetched, or that fails, is unmeasured for every player in it (values NaN,
never zero). Players in a measured game who drew no foul are genuine zeros.
Offensive fouls drawn are ``subType == offensive`` (charges plus non-charge
offensive fouls such as illegal screens and push-offs; the feed cannot split
the non-charge kinds further); charges are the ``charge`` descriptor subset;
``pfd`` is every non-technical foul drawn.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"nba_pipeline/results/estimated_skills"
FOULS = RESULTS/"fouls_drawn"
LEDGER = FOULS/"foul_ledger.parquet"
STATUS = FOULS/"game_status.csv"
LINES = RESULTS/"player_game_lines.parquet"
ARCHIVE = ROOT/"player_sheets/game_report"
OUT = FOULS/"fouls_drawn_game_lines.parquet"
FIRST_SEASON = 2020
DRAWN = ["pfd", "off_fouls_drawn", "charges_drawn", "off_fouls_drawn_noncharge", "shooting_fouls_drawn",
         "looseball_fouls_drawn", "offball_off_fouls_drawn"]
COMMITTED = ["pf_committed_cdn", "off_fouls_committed_cdn", "charges_committed_cdn", "double_fouls_committed_cdn"]
RULE = ("measured only if fetched, every non-technical foul has one drawer inside the game's PBP universe, and either "
        "(a) the CDN committed side equals repo PBP pf_committed per player up to the player's CDN double fouls and the "
        "archive box (PF and PFD on shared players) is absent or agrees, or (b) the PBP check fails but the archive box "
        "agrees on every player with CDN foul activity")


def normalize(value):
    """Descriptor vocabulary drifts across seasons ('loose ball', 'loose-ball', 'looseball')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return re.sub(r"[^a-z0-9]", "", str(value).lower()) or None


def foul_flags(ledger):
    """Non-technical fouls with typed flags; technical/defensive-three-second rows have no drawer and are dropped."""
    f = ledger.copy()
    f["sub_type"] = f.sub_type.map(normalize)
    f["descriptor"] = f.descriptor.map(normalize)
    f = f[f.sub_type.ne("technical")].copy()
    offensive = f.sub_type.eq("offensive")
    charge = offensive & f.descriptor.eq("charge")
    f["pfd"] = 1
    f["off_fouls_drawn"] = offensive.astype(int)
    f["charges_drawn"] = charge.astype(int)
    f["off_fouls_drawn_noncharge"] = (offensive & ~charge).astype(int)
    f["shooting_fouls_drawn"] = f.descriptor.eq("shooting").astype(int)
    f["looseball_fouls_drawn"] = f.descriptor.eq("looseball").astype(int)
    f["offball_off_fouls_drawn"] = (offensive & f.descriptor.eq("offtheball")).astype(int)
    f["double_foul"] = f.descriptor.eq("double").astype(int)
    return f


def aggregate(ledger):
    """Drawn-side and committed-side counts per (game_id, player_id) plus per-game identity checks."""
    f = foul_flags(ledger)
    drawn = (f[f.drawn_id.notna()].assign(player_id=lambda d: d.drawn_id.astype(int))
             .groupby(["game_id", "player_id"], as_index=False)[DRAWN].sum())
    committed = (f[f.fouler_id.notna()].assign(player_id=lambda d: d.fouler_id.astype(int))
                 .groupby(["game_id", "player_id"], as_index=False)
                 .agg(pf_committed_cdn=("pfd", "sum"), off_fouls_committed_cdn=("off_fouls_drawn", "sum"),
                      charges_committed_cdn=("charges_drawn", "sum"), double_fouls_committed_cdn=("double_foul", "sum")))
    games = f.groupby("game_id").agg(nontechnical_fouls=("pfd", "size"), fouls_without_drawer=("drawn_id", lambda s: int(s.isna().sum())),
                                     fouls_without_fouler=("fouler_id", lambda s: int(s.isna().sum()))).reset_index()
    return drawn, committed, games


def archive_box(game_id, season, archive=ARCHIVE):
    """Official box PF and PFD per player from the local game-report archive (DataFrame indexed by player id),
    or None when the file is absent or unusable."""
    path = Path(archive)/str(int(season))/f"{str(game_id).lstrip('0')}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["PLAYER_ID", "PF", "PFD"]).drop_duplicates()
    except (ValueError, KeyError):
        return None
    if df.PLAYER_ID.duplicated().any() or df[["PF", "PFD"]].isna().any().any():
        return None
    return pd.DataFrame({"PF": df.PF.astype(int).to_numpy(), "PFD": df.PFD.astype(int).to_numpy()},
                        index=df.PLAYER_ID.astype(int).to_numpy())


def _same(a, b):
    idx = a.index.union(b.index)
    return bool(a.reindex(idx).fillna(0).eq(b.reindex(idx).fillna(0)).all())


def _pbp_committed_ok(cdn, pbp):
    """Per player: 0 <= CDN personal fouls - PBP counter <= CDN double fouls (the PBP counter cannot see doubles)."""
    idx = cdn.index.union(pbp.index)
    gap = cdn.pf_committed_cdn.reindex(idx).fillna(0) - pbp.reindex(idx).fillna(0)
    slack = cdn.double_fouls_committed_cdn.reindex(idx).fillna(0)
    return bool(gap.ge(0).all() and gap.le(slack).all())


def build_lines(ledger, status, lines, archive_lookup=archive_box):
    """One row per (player_id, game_id) of the lines panel for seasons >= FIRST_SEASON."""
    base = lines[lines.season.ge(FIRST_SEASON)].copy()
    base["game_id"] = base.game_id.astype(str)
    dup = base.duplicated(["game_id", "player_id"], keep=False)
    base = base.loc[~dup, ["player_id", "player_name", "game_id", "season", "season_type", "game_date", "minutes", "pf_committed",
                          "off_fouls_committed", "charges_committed"]]
    drawn, committed, games = aggregate(ledger)
    st = status.assign(game_id=lambda d: d.game_id.astype(str)).set_index("game_id").status
    drawn_by = {gid: g.set_index("player_id") for gid, g in drawn.groupby("game_id")}
    committed_by = {gid: g.set_index("player_id") for gid, g in committed.groupby("game_id")}
    games_by = games.set_index("game_id")
    empty_drawn = drawn.iloc[:0].set_index("player_id")
    empty_committed = committed.iloc[:0].set_index("player_id")
    checks = []
    for gid, g in base.groupby("game_id", sort=False):
        season = int(g.season.iloc[0])
        row = {"game_id": gid, "season": season, "status": st.get(gid, "unfetched")}
        if row["status"] != "ok":
            row.update(reason="unfetched" if row["status"] == "unfetched" else row["status"])
            checks.append(row)
            continue
        dg = drawn_by.get(gid, empty_drawn)
        cg = committed_by.get(gid, empty_committed)
        players = pd.Index(g.player_id)
        row["nontechnical_fouls"] = int(games_by.nontechnical_fouls.get(gid, 0))
        row["fouls_without_drawer"] = int(games_by.fouls_without_drawer.get(gid, 0))
        row["drawn_total"], row["committed_total"] = int(dg.pfd.sum()), int(cg.pf_committed_cdn.sum())
        row["double_fouls"] = int(cg.double_fouls_committed_cdn.sum())
        row["identity_match"] = row["drawn_total"] == row["committed_total"] and row["fouls_without_drawer"] == 0
        row["drawn_players_outside_lines"] = int((~dg.index.isin(players)).sum())
        row["pbp_committed_gap"] = row["committed_total"] - float(g.pf_committed.sum())
        row["pbp_committed_match"] = _pbp_committed_ok(cg, g.set_index("player_id").pf_committed.astype(float))
        arch = archive_lookup(gid, season)
        common = pd.Index([]) if arch is None else arch.index.intersection(players)
        active = cg.index[cg.pf_committed_cdn.gt(0)].union(dg.index[dg.pfd.gt(0)]).intersection(players)
        row["archive_players_checked"] = int(len(common))
        if len(common) == 0:
            row["archive_match"], row["archive_covers_activity"] = None, None
        else:
            row["archive_match"] = (_same(cg.pf_committed_cdn.reindex(common).fillna(0), arch.PF.reindex(common))
                                    and _same(dg.pfd.reindex(common).fillna(0), arch.PFD.reindex(common)))
            row["archive_covers_activity"] = bool(active.isin(common).all())
        if not row["identity_match"]:
            row["reason"] = "drawer_identity_mismatch"
        elif row["drawn_players_outside_lines"]:
            row["reason"] = "drawer_outside_lines"
        elif row["archive_match"] is False:
            row["reason"] = "archive_box_mismatch"
        elif row["pbp_committed_match"] or (row["archive_match"] and row["archive_covers_activity"]):
            row["reason"] = "observed"
        else:
            row["reason"] = "pbp_committed_mismatch"
        checks.append(row)
    checks = pd.DataFrame(checks)
    measured = set(checks.loc[checks.reason.eq("observed"), "game_id"])
    out = base.merge(drawn, on=["game_id", "player_id"], how="left").merge(committed, on=["game_id", "player_id"], how="left")
    out["fouls_drawn_is_measured"] = out.game_id.isin(measured)
    out["fouls_drawn_reason"] = out.game_id.map(checks.set_index("game_id").reason)
    for c in DRAWN+COMMITTED:
        out[c] = out[c].where(~out.fouls_drawn_is_measured, out[c].fillna(0)).astype(float)
        out.loc[~out.fouls_drawn_is_measured, c] = np.nan
    out = out.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    return out, checks


def season_summary(out, checks):
    rows = []
    for season, g in checks.groupby("season"):
        meas = out[out.season.eq(season) & out.fouls_drawn_is_measured]
        team_games = 2*int(g.reason.eq("observed").sum())
        obs = g[g.reason.eq("observed")]
        rows.append({"season": int(season), "games": int(len(g)), "measured_games": int(len(obs)),
            "reasons": g.reason.value_counts().to_dict(),
            "archive_checked": int(g.archive_match.notna().sum()) if "archive_match" in g else 0,
            "archive_mismatch": int(g.archive_match.eq(False).sum()) if "archive_match" in g else 0,
            "observed_with_archive_agreement": int(obs.archive_match.eq(True).sum()) if "archive_match" in g else 0,
            "observed_pbp_gap_arbitrated_by_archive": int(obs.pbp_committed_match.eq(False).sum()) if "pbp_committed_match" in g else 0,
            "pfd_per_team_game": float(meas.pfd.sum()/team_games) if team_games else None,
            "off_fouls_drawn_per_team_game": float(meas.off_fouls_drawn.sum()/team_games) if team_games else None,
            "charges_drawn_per_team_game": float(meas.charges_drawn.sum()/team_games) if team_games else None,
            "noncharge_off_fouls_drawn_per_team_game": float(meas.off_fouls_drawn_noncharge.sum()/team_games) if team_games else None,
            "committed_vs_pbp_off_fouls_agree_share": float(meas.off_fouls_committed_cdn.eq(meas.off_fouls_committed).mean()) if len(meas) else None,
            "committed_vs_pbp_charges_agree_share": float(meas.charges_committed_cdn.eq(meas.charges_committed).mean()) if len(meas) else None})
    return rows


def run(out_path=OUT):
    ledger = pd.read_parquet(LEDGER)
    status = pd.read_csv(STATUS, dtype={"game_id": str})
    lines = pd.read_parquet(LINES)
    out, checks = build_lines(ledger, status, lines)
    out.to_parquet(out_path, index=False)
    checks.to_csv(FOULS/"game_reconciliation.csv", index=False)
    summary = {"rows": int(len(out)), "measured_rows": int(out.fouls_drawn_is_measured.sum()),
               "games": int(len(checks)), "measured_games": int(checks.reason.eq("observed").sum()),
               "reasons": checks.reason.value_counts().to_dict(), "seasons": season_summary(out, checks), "rule": RULE}
    (FOULS/"reconciliation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["rows", "measured_rows", "games", "measured_games", "reasons"]}, indent=2))
    print(pd.DataFrame(summary["seasons"]).drop(columns=["reasons"]).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out)

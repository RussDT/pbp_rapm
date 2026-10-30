"""Per-player-game NBA observation lines for the estimated-skills panel.

Extracts, from the repo's own raw modern PBP (``raw_data/NBA{YY}[_PS].parquet``)
plus the processed possession surface (``processed/RAPM{YY}[_PS].parquet``):

- zone shot lines (rim / mid / three attempts+makes) using the exact repo zone
  conventions (rim = ``RIM_ACTION_TYPES`` union ``layup|dunk| tip `` description
  patterns; three = ``3PT`` text; mid = all non-rim twos),
- FT lines (FTA/FTM incl. technical/flagrant trips, matching box FTA),
- box counting events: PTS, AST, STL, BLK, TOV (player turnovers), OREB/DREB,
  personal fouls committed (official ``(Pn.Tn)`` ledger), offensive fouls and
  charges committed,
- on-court offensive/defensive possession counts from the possession surface,
- on-court minutes from event-clock deltas (audit column, career-teammate
  convention), game_date, season, season_type.

Attribution notes (validated in the reconciliation report):
- ``player2_id`` / ``player3_id`` are unpopulated in the fetched raw files, so
  AST / STL / BLK are parsed from cumulative description tags
  (``(Name N AST)``, ``Name STEAL (N STL)``, ``Name BLOCK (N BLK)``) and the
  tag display name is resolved to a player id through per-game anchor rows
  (rebound / FT / substitution / foul rows whose ``player1_id`` is known) plus
  the season surname map. Same-surname teammates are disambiguated by the
  PBP's own initial prefixes (``Ja. Green`` vs ``Je. Green``).
- Offensive fouls DRAWN cannot be attributed: the drawer appears in neither
  the player2 field (0% populated) nor the description (the trailing
  parenthetical is the referee). The report carries the evidence; the drawn
  side of the requested skill is a documented blocker.

Outputs under ``results/estimated_skills/``: ``player_game_lines.parquet``
and ``lines_reconciliation.json``. Generated artifacts stay out of git.

Run: ``.venv/bin/python nba_pipeline/scripts/build_nba_player_game_lines.py [--start 2019] [--end 2026]``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PIPELINE_ROOT / "raw_data"
PROCESSED_DIR = PIPELINE_ROOT / "processed"
OUT_DIR = PIPELINE_ROOT / "results" / "estimated_skills"

TEAM_ID_MIN = 1610612737  # NBA team ids start here; below = person ids

RAW_COLS = [
    "game_id", "event_num", "event_type", "event_action_type", "period",
    "minute_remaining_quarter", "seconds_remaining_quarter",
    "home_description", "visitor_description", "neutral_description",
    "player1_id", "player1_name",
    "away_score", "home_score",
    "away_player1", "away_player2", "away_player3", "away_player4", "away_player5",
    "home_player1", "home_player2", "home_player3", "home_player4", "home_player5",
]
HOME_LINEUP = [f"home_player{i}" for i in range(1, 6)]
AWAY_LINEUP = [f"away_player{i}" for i in range(1, 6)]

AST_TAG_RE = re.compile(r"\(([^()]+?) +(\d+) AST\)")
STL_RE = re.compile(r"^(.*?) STEAL \((\d+) STL\)\s*$")
BLK_RE = re.compile(r"^(.*?) BLOCK \((\d+) BLK\)\s*$")
REB_ANCHOR_RE = re.compile(r"^(.*?) REBOUND \(Off:(\d+) Def:(\d+)\)\s*$")
FT_ANCHOR_RE = re.compile(r"^(?:MISS )?(.*?) Free Throw ")
SUB_ANCHOR_RE = re.compile(r"^SUB: .*? FOR (.*)\s*$")
FOUL_ANCHOR_RE = re.compile(
    r"^(.*?) (?:P\.FOUL|S\.FOUL|L\.B\.FOUL|OFF\.Foul|T\.FOUL|C\.P\.FOUL|"
    r"IN\.FOUL|AWAY\.FROM\.PLAY\.FOUL|Offensive Charge Foul|Personal Take Foul|"
    r"Personal Block Foul|Shooting Block Foul|FLAGRANT)"
)
P_COUNTER_RE = re.compile(r"\(P(\d+)[.)（]?")
PTS_TAG_RE = re.compile(r"\((\d+) PTS\)")
INITIAL_PREFIX_RE = re.compile(r"^([a-z]+)\. (.+)$")
CHARGE_RE = re.compile(r"Offensive Charge Foul", re.IGNORECASE)
OFF_FOUL_RE = re.compile(r"OFF\.Foul|Offensive Charge Foul", re.IGNORECASE)


def load_common():
    """Load process_rapm_blocks.common standalone (skip heavy package init)."""
    path = PIPELINE_ROOT / "scripts" / "process_rapm_blocks" / "common.py"
    spec = importlib.util.spec_from_file_location("rapm_common_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Per-season extraction
# ---------------------------------------------------------------------------


def extract_season(raw_path: Path, season: int, season_type: str, common) -> dict:
    """Return dict with per-player-game lines plus reconciliation counters."""
    df = pd.read_parquet(raw_path, columns=RAW_COLS)
    df["game_id"] = common.normalize_game_id_series(df["game_id"])
    n_before = len(df)
    df = common.dedupe_exact_raw_pbp_rows(df, raw_path.name)
    n_dupes = n_before - len(df)
    df = common.sort_raw_pbp_chronologically(df)

    home = df["home_description"].fillna("").astype(str)
    away = df["visitor_description"].fillna("").astype(str)
    etype = df["event_type"].to_numpy()
    p1 = pd.to_numeric(df["player1_id"], errors="coerce").fillna(0).astype(np.int64)
    is_person = (p1 > 0) & (p1 < TEAM_ID_MIN)
    gid = df["game_id"].to_numpy()

    # ------------------------------------------------------------------
    # Shot rows (event_type 1 make / 2 miss), repo zone conventions
    # ------------------------------------------------------------------
    shot_text = (home + " " + away).str.strip()
    is_fga = pd.Series(np.isin(etype, [1, 2]), index=df.index) & is_person
    is_make = pd.Series(etype == 1, index=df.index) & is_person
    is_3pt = shot_text.str.contains("3PT", case=False, regex=False, na=False)
    action = pd.to_numeric(df["event_action_type"], errors="coerce")
    is_rim = action.isin(common.RIM_ACTION_TYPES) | shot_text.str.contains(
        common.RIM_DESC_PATTERN, case=False, regex=True, na=False
    )
    rim_and_3pt = int((is_fga & is_3pt & is_rim).sum())
    zone = np.where(is_3pt, "three", np.where(is_rim, "rim", "mid"))

    # ------------------------------------------------------------------
    # Event side: which description column carries the actor's text
    # ------------------------------------------------------------------
    side = np.full(len(df), "", dtype=object)
    miss_mask = etype == 2
    side[miss_mask & home.str.contains("MISS", regex=False).to_numpy()] = "H"
    side[miss_mask & away.str.contains("MISS", regex=False).to_numpy()] = "A"
    make_mask = etype == 1
    side[make_mask & (home != "").to_numpy() & (away == "").to_numpy()] = "H"
    side[make_mask & (away != "").to_numpy() & (home == "").to_numpy()] = "A"
    ft_mask = etype == 3
    side[ft_mask & home.str.contains("Free Throw", regex=False).to_numpy()] = "H"
    side[ft_mask & away.str.contains("Free Throw", regex=False).to_numpy()] = "A"
    reb_mask = etype == 4
    side[reb_mask & home.str.contains("REBOUND", regex=False).to_numpy()] = "H"
    side[reb_mask & away.str.contains("REBOUND", regex=False).to_numpy()] = "A"
    tov_mask = etype == 5
    side[tov_mask & home.str.contains("Turnover", regex=False).to_numpy()] = "H"
    side[tov_mask & away.str.contains("Turnover", regex=False).to_numpy()] = "A"
    foul_mask = etype == 6
    side[foul_mask & (home != "").to_numpy() & (away == "").to_numpy()] = "H"
    side[foul_mask & (away != "").to_numpy() & (home == "").to_numpy()] = "A"
    sub_mask = etype == 8
    side[sub_mask & home.str.startswith("SUB:").to_numpy()] = "H"
    side[sub_mask & away.str.startswith("SUB:").to_numpy()] = "A"
    n_side_missing_core = int(
        ((side == "") & np.isin(etype, [1, 2, 3]) & is_person.to_numpy()).sum()
    )

    own_text = np.where(side == "H", home, np.where(side == "A", away, ""))
    opp_text = np.where(side == "H", away, np.where(side == "A", home, ""))

    # ------------------------------------------------------------------
    # Direct player1-attributed stats
    # ------------------------------------------------------------------
    ftm = ft_mask & is_person.to_numpy() & (
        ~pd.Series(own_text).str.contains("MISS", regex=False).to_numpy()
    )
    fta = ft_mask & is_person.to_numpy()
    shot_pts = np.where(is_make, np.where(is_3pt, 3, 2), 0)

    player_tov = tov_mask & is_person.to_numpy()

    # Rebounds: player rebounds carry (Off:X Def:Y) cumulative counters.
    reb_person = reb_mask & is_person.to_numpy()
    reb_off = np.full(len(df), np.nan)
    reb_def = np.full(len(df), np.nan)
    reb_rows = np.where(reb_person)[0]
    for i in reb_rows:
        m = REB_ANCHOR_RE.match(own_text[i])
        if m:
            reb_off[i] = int(m.group(2))
            reb_def[i] = int(m.group(3))
    n_reb_no_counter = int(reb_person.sum() - np.isfinite(reb_off[reb_rows]).sum())

    # Fouls: personal-foul ledger counter (Pn...), offensive fouls / charges.
    foul_person = foul_mask & is_person.to_numpy()
    foul_p = np.full(len(df), np.nan)
    is_off_foul = np.zeros(len(df), dtype=bool)
    is_charge = np.zeros(len(df), dtype=bool)
    for i in np.where(foul_person)[0]:
        text = own_text[i]
        m = P_COUNTER_RE.search(text)
        if m:
            foul_p[i] = int(m.group(1))
        if OFF_FOUL_RE.search(text):
            is_off_foul[i] = True
            if CHARGE_RE.search(text):
                is_charge[i] = True

    # Cumulative points tag audit (max "(N PTS)" per player-game).
    pts_tag = np.full(len(df), np.nan)
    scoring = (make_mask | (ftm & ft_mask)) & is_person.to_numpy()
    for i in np.where(scoring)[0]:
        m = PTS_TAG_RE.search(own_text[i])
        if m:
            pts_tag[i] = int(m.group(1))

    # ------------------------------------------------------------------
    # Per-game display-name anchors: (game, side, display) -> player1_id
    # ------------------------------------------------------------------
    anchors: dict[tuple, set] = defaultdict(set)

    def add_anchor(i, name):
        if name and is_person.iloc[i]:
            anchors[(gid[i], side[i], norm_name(name))].add(int(p1.iloc[i]))

    for i in reb_rows:
        m = REB_ANCHOR_RE.match(own_text[i])
        if m:
            add_anchor(i, m.group(1))
    for i in np.where(fta)[0]:
        m = FT_ANCHOR_RE.match(own_text[i])
        if m:
            add_anchor(i, m.group(1))
    for i in np.where(sub_mask & is_person.to_numpy())[0]:
        m = SUB_ANCHOR_RE.match(own_text[i])
        if m:
            add_anchor(i, m.group(1))
    for i in np.where(foul_person)[0]:
        m = FOUL_ANCHOR_RE.match(own_text[i])
        if m:
            add_anchor(i, m.group(1))
    # Keep only unique anchors; drop corrupt one-name-two-id entries.
    anchor_map = {k: next(iter(v)) for k, v in anchors.items() if len(v) == 1}
    n_anchor_conflicts = sum(1 for v in anchors.values() if len(v) > 1)

    # Season bare-surname display map and per-game side rosters.
    nm = (
        df.loc[is_person, ["player1_id", "player1_name"]]
        .dropna()
        .assign(n=lambda x: x["player1_name"].map(norm_name))
    )
    season_name = nm.groupby("player1_id")["n"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else ""
    )
    season_name = {int(k): v for k, v in season_name.items() if v}

    rosters: dict[tuple, set] = defaultdict(set)
    for cols, side_key in ((HOME_LINEUP, "H"), (AWAY_LINEUP, "A")):
        sub = df[["game_id", *cols]].melt(id_vars="game_id", value_name="pid")
        sub["pid"] = pd.to_numeric(sub["pid"], errors="coerce").fillna(0).astype(np.int64)
        sub = sub[(sub["pid"] > 0) & (sub["pid"] < TEAM_ID_MIN)]
        for g, pids in sub.groupby("game_id")["pid"]:
            rosters[(g, side_key)].update(int(x) for x in pids.unique())
    # player1 side attribution also feeds rosters (guards lineup gaps)
    p1_np = p1.to_numpy()
    for i in np.where((side != "") & is_person.to_numpy())[0]:
        rosters[(gid[i], side[i])].add(int(p1_np[i]))

    # ------------------------------------------------------------------
    # Tag resolution (AST on make rows; STL/BLK in opponent column)
    # ------------------------------------------------------------------
    unresolved = {"ast": 0, "stl": 0, "blk": 0}
    resolved_events: dict[str, list] = {"ast": [], "stl": [], "blk": []}

    def resolve(g, s, raw_tag_name):
        name = norm_name(raw_tag_name)
        pid = anchor_map.get((g, s, name))
        if pid is not None:
            return pid
        roster = rosters.get((g, s), set())
        m = INITIAL_PREFIX_RE.match(name)
        surname = m.group(2) if m else name
        cands = [r for r in roster if season_name.get(r) == surname]
        if len(cands) == 1:
            return cands[0]
        if m and len(cands) > 1:
            # elimination: remove roster ids anchored to a different initialed
            # display of the same surname in this game
            taken = {
                v
                for (gg, ss, nn), v in anchor_map.items()
                if gg == g and ss == s and nn != name and nn.endswith(f" {surname}")
            }
            remaining = [c for c in cands if c not in taken]
            if len(remaining) == 1:
                return remaining[0]
        return None

    flip = {"H": "A", "A": "H", "": ""}
    for i in np.where(make_mask & is_person.to_numpy())[0]:
        m = AST_TAG_RE.search(own_text[i])
        if m and side[i]:
            pid = resolve(gid[i], side[i], m.group(1))
            if pid is None:
                unresolved["ast"] += 1
            else:
                resolved_events["ast"].append((gid[i], pid, int(m.group(2))))
    for i in np.where(player_tov)[0]:
        m = STL_RE.match(str(opp_text[i]))
        if m and side[i]:
            pid = resolve(gid[i], flip[side[i]], m.group(1))
            if pid is None:
                unresolved["stl"] += 1
            else:
                resolved_events["stl"].append((gid[i], pid, int(m.group(2))))
    # team turnovers can also carry steals (e.g. stolen inbound credited to team)
    for i in np.where(tov_mask & ~is_person.to_numpy())[0]:
        text_h = STL_RE.match(home.iloc[i])
        text_a = STL_RE.match(away.iloc[i])
        for m, s in ((text_h, "H"), (text_a, "A")):
            if m:
                pid = resolve(gid[i], s, m.group(1))
                if pid is None:
                    unresolved["stl"] += 1
                else:
                    resolved_events["stl"].append((gid[i], pid, int(m.group(2))))
    for i in np.where(miss_mask & is_person.to_numpy())[0]:
        m = BLK_RE.match(str(opp_text[i]))
        if m and side[i]:
            pid = resolve(gid[i], flip[side[i]], m.group(1))
            if pid is None:
                unresolved["blk"] += 1
            else:
                resolved_events["blk"].append((gid[i], pid, int(m.group(2))))

    # ------------------------------------------------------------------
    # Aggregate per (game, player)
    # ------------------------------------------------------------------
    base = pd.DataFrame(
        {
            "game_id": gid,
            "player_id": p1_np,
            "fga": is_fga.to_numpy().astype(int),
            "fgm": is_make.to_numpy().astype(int),
            "rim_fga": (is_fga.to_numpy() & (zone == "rim")).astype(int),
            "rim_fgm": (is_make.to_numpy() & (zone == "rim")).astype(int),
            "mid_fga": (is_fga.to_numpy() & (zone == "mid")).astype(int),
            "mid_fgm": (is_make.to_numpy() & (zone == "mid")).astype(int),
            "three_fga": (is_fga.to_numpy() & (zone == "three")).astype(int),
            "three_fgm": (is_make.to_numpy() & (zone == "three")).astype(int),
            "fta": fta.astype(int),
            "ftm": ftm.astype(int),
            "pts": shot_pts + ftm.astype(int),
            "tov": player_tov.astype(int),
            "reb_events": reb_person.astype(int),
            "oreb_ctr": reb_off,
            "dreb_ctr": reb_def,
            "pf_ctr": foul_p,
            "foul_rows": foul_person.astype(int),
            "off_fouls_committed": is_off_foul.astype(int),
            "charges_committed": is_charge.astype(int),
            "pts_tag": pts_tag,
        }
    )
    base = base[(base["player_id"] > 0) & (base["player_id"] < TEAM_ID_MIN)]
    agg = base.groupby(["game_id", "player_id"]).agg(
        fga=("fga", "sum"), fgm=("fgm", "sum"),
        rim_fga=("rim_fga", "sum"), rim_fgm=("rim_fgm", "sum"),
        mid_fga=("mid_fga", "sum"), mid_fgm=("mid_fgm", "sum"),
        three_fga=("three_fga", "sum"), three_fgm=("three_fgm", "sum"),
        fta=("fta", "sum"), ftm=("ftm", "sum"), pts=("pts", "sum"),
        tov=("tov", "sum"), reb_events=("reb_events", "sum"),
        oreb=("oreb_ctr", "max"), dreb=("dreb_ctr", "max"),
        pf_committed=("pf_ctr", "max"), foul_rows=("foul_rows", "sum"),
        off_fouls_committed=("off_fouls_committed", "sum"),
        charges_committed=("charges_committed", "sum"),
        pts_tag_max=("pts_tag", "max"),
    ).reset_index()
    for col in ("oreb", "dreb", "pf_committed"):
        agg[col] = agg[col].fillna(0).astype(int)

    for stat in ("ast", "stl", "blk"):
        ev = pd.DataFrame(resolved_events[stat], columns=["game_id", "player_id", "ctr"])
        if len(ev):
            tot = ev.groupby(["game_id", "player_id"]).agg(
                **{stat: ("ctr", "max"), f"{stat}_events": ("ctr", "size")}
            ).reset_index()
        else:
            tot = pd.DataFrame(columns=["game_id", "player_id", stat, f"{stat}_events"])
        agg = agg.merge(tot, on=["game_id", "player_id"], how="left")
        agg[stat] = agg[stat].fillna(0).astype(int)
        agg[f"{stat}_events"] = agg[f"{stat}_events"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Minutes: event-clock deltas to the earlier row's lineup
    # ------------------------------------------------------------------
    clock = (
        pd.to_numeric(df["minute_remaining_quarter"], errors="coerce").fillna(0) * 60.0
        + pd.to_numeric(df["seconds_remaining_quarter"], errors="coerce").fillna(0)
    ).to_numpy(float)
    same_seg = (gid == np.roll(gid, -1)) & (
        df["period"].to_numpy() == np.roll(df["period"].to_numpy(), -1)
    )
    elapsed = np.where(same_seg, clock - np.roll(clock, -1), 0.0)
    elapsed = np.where((elapsed > 0) & (elapsed <= 720), elapsed, 0.0)
    minute_frames = []
    for cols in (HOME_LINEUP, AWAY_LINEUP):
        lineup = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        for j in range(5):
            minute_frames.append(
                pd.DataFrame({"game_id": gid, "player_id": lineup[:, j], "sec": elapsed})
            )
    mins = pd.concat(minute_frames, ignore_index=True)
    mins["player_id"] = mins["player_id"].fillna(0).astype(np.int64)
    mins = mins[(mins["player_id"] > 0) & (mins["player_id"] < TEAM_ID_MIN)]
    mins = mins.groupby(["game_id", "player_id"])["sec"].sum().reset_index()
    mins["minutes"] = mins["sec"] / 60.0

    lines = agg.merge(
        mins[["game_id", "player_id", "minutes"]], on=["game_id", "player_id"], how="outer"
    )
    count_cols = [
        c for c in lines.columns
        if c not in ("game_id", "player_id", "minutes", "pts_tag_max", "oreb", "dreb", "pf_committed")
    ]
    lines[count_cols] = lines[count_cols].fillna(0).astype(int)
    for col in ("oreb", "dreb", "pf_committed"):
        lines[col] = lines[col].fillna(0).astype(int)
    lines["minutes"] = lines["minutes"].fillna(0.0)
    lines["season"] = season
    lines["season_type"] = season_type

    # ------------------------------------------------------------------
    # Reconciliation counters
    # ------------------------------------------------------------------
    # team score identity: attributed player points vs final scoreboard
    pts_side = pd.DataFrame(
        {"game_id": gid, "side": side, "pts": shot_pts + ftm.astype(int)}
    )
    pts_side = pts_side[pts_side["side"].isin(["H", "A"]) & (pts_side["pts"] > 0)]
    team_pts = pts_side.groupby(["game_id", "side"])["pts"].sum().unstack(fill_value=0)
    finals = df.groupby("game_id").agg(
        home_final=("home_score", "max"), away_final=("away_score", "max")
    )
    score_check = team_pts.join(finals, how="outer").fillna(0)
    home_exact = (score_check.get("H", 0) == score_check["home_final"])
    away_exact = (score_check.get("A", 0) == score_check["away_final"])
    n_games = len(score_check)

    name_display = (
        df.loc[is_person, ["player1_id", "player1_name"]]
        .dropna()
        .groupby("player1_id")["player1_name"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    )

    recon = {
        "raw_rows": int(n_before),
        "exact_duplicate_rows_removed": int(n_dupes),
        "games": int(pd.Series(gid).nunique()),
        "rim_and_3pt_rows": rim_and_3pt,
        "core_rows_missing_side": n_side_missing_core,
        "reb_rows_missing_counter": n_reb_no_counter,
        "anchor_conflicts": int(n_anchor_conflicts),
        "unresolved_tags": dict(unresolved),
        "total_tags": {
            k: int(len(resolved_events[k]) + unresolved[k]) for k in unresolved
        },
        "team_score_exact_share": float((home_exact.sum() + away_exact.sum()) / (2 * n_games)),
        "team_games_checked": int(2 * n_games),
    }
    return {"lines": lines, "recon": recon, "names": name_display}


# ---------------------------------------------------------------------------
# Possession surface join
# ---------------------------------------------------------------------------


def load_possessions(season: int, season_type: str, common) -> pd.DataFrame:
    yy = f"{season % 100:02d}"
    suffix = "" if season_type == "rs" else "_PS"
    path = PROCESSED_DIR / f"RAPM{yy}{suffix}.parquet"
    cols = ["game_id"] + [f"O{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)] + ["game_date"]
    df = pd.read_parquet(path, columns=cols)
    df["game_id"] = common.normalize_game_id_series(df["game_id"])
    date_map = (
        pd.to_datetime(df.groupby("game_id")["game_date"].first(), errors="coerce")
    )
    frames = []
    for role, cols_role in (("off_poss", [f"O{i}" for i in range(1, 6)]),
                            ("def_poss", [f"D{i}" for i in range(1, 6)])):
        sub = df[["game_id", *cols_role]].melt(id_vars="game_id", value_name="pid")
        sub["pid"] = pd.to_numeric(sub["pid"], errors="coerce").fillna(0).astype(np.int64)
        sub = sub[(sub["pid"] > 0) & (sub["pid"] < TEAM_ID_MIN)]
        counts = sub.groupby(["game_id", "pid"]).size().rename(role).reset_index()
        frames.append(counts)
    poss = frames[0].merge(frames[1], on=["game_id", "pid"], how="outer")
    poss[["off_poss", "def_poss"]] = poss[["off_poss", "def_poss"]].fillna(0).astype(int)
    poss = poss.rename(columns={"pid": "player_id"})
    poss["game_date"] = poss["game_id"].map(date_map)
    return poss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2019)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--season-types", type=str, default="rs,ps")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Isolated output package; default preserves the current estimated-skills location")
    args = parser.parse_args()
    season_types = [s.strip() for s in args.season_types.split(",")]

    common = load_common()
    all_lines = []
    recon: dict = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "per_season": [],
    }
    names: dict[int, str] = {}
    t0 = time.time()
    for season in range(args.start, args.end + 1):
        for st in season_types:
            yy = f"{season % 100:02d}"
            suffix = "" if st == "rs" else "_PS"
            raw_path = RAW_DIR / f"NBA{yy}{suffix}.parquet"
            if not raw_path.exists():
                raise SystemExit(f"missing raw file {raw_path}")
            t = time.time()
            out = extract_season(raw_path, season, st, common)
            lines = out["lines"]
            for k, v in out["names"].items():
                names.setdefault(int(k), str(v))

            poss = load_possessions(season, st, common)
            merged = lines.merge(poss, on=["game_id", "player_id"], how="outer", indicator=True)
            pbp_only = int((merged["_merge"] == "left_only").sum())
            poss_only = int((merged["_merge"] == "right_only").sum())
            both = int((merged["_merge"] == "both").sum())
            merged = merged.drop(columns=["_merge"])
            merged[["off_poss", "def_poss"]] = (
                merged[["off_poss", "def_poss"]].fillna(0).astype(int)
            )
            count_cols = [
                c for c in lines.columns
                if c not in ("game_id", "player_id", "season", "season_type", "minutes")
            ]
            merged[count_cols] = merged[count_cols].fillna(0)
            merged["minutes"] = merged["minutes"].fillna(0.0)
            merged["season"] = season
            merged["season_type"] = st
            # dates for PBP-only rows come from the game-level map
            game_dates = poss.groupby("game_id")["game_date"].first()
            merged["game_date"] = merged["game_date"].fillna(
                merged["game_id"].map(game_dates)
            )
            undated = int(merged["game_date"].isna().sum())

            season_rec = out["recon"]
            zone_ok = bool(
                (
                    merged["rim_fga"] + merged["mid_fga"] + merged["three_fga"]
                    == merged["fga"]
                ).all()
                and (
                    merged["rim_fgm"] + merged["mid_fgm"] + merged["three_fgm"]
                    == merged["fgm"]
                ).all()
            )
            pts_tag_rows = merged["pts_tag_max"].notna()
            season_rec.update(
                {
                    "season": season,
                    "season_type": st,
                    "player_game_rows": int(len(merged)),
                    "zone_partition_exact": zone_ok,
                    "pts_vs_pts_tag_exact_share": float(
                        (merged.loc[pts_tag_rows, "pts"] == merged.loc[pts_tag_rows, "pts_tag_max"]).mean()
                    ),
                    "reb_counter_vs_events_exact_share": float(
                        (merged["oreb"] + merged["dreb"] == merged["reb_events"]).mean()
                    ),
                    "ast_counter_vs_events_exact_share": float(
                        (merged["ast"] == merged["ast_events"]).mean()
                    ),
                    "stl_counter_vs_events_exact_share": float(
                        (merged["stl"] == merged["stl_events"]).mean()
                    ),
                    "blk_counter_vs_events_exact_share": float(
                        (merged["blk"] == merged["blk_events"]).mean()
                    ),
                    "pf_counter_vs_foulrows_exact_share": float(
                        (merged["pf_committed"] == merged["foul_rows"]).mean()
                    ),
                    "poss_join": {
                        "both": both,
                        "pbp_only": pbp_only,
                        "poss_only": poss_only,
                        "pbp_row_share_with_poss": float(both / max(both + pbp_only, 1)),
                        # phantom groups from admin rows (subs/ejections/jump
                        # balls) have ~0 minutes and no counting stats; the
                        # meaningful coverage metric is active rows only
                        "active_row_share_with_poss": float(
                            (
                                ((merged["off_poss"] + merged["def_poss"]) > 0)
                                & (
                                    (merged["minutes"] >= 0.5)
                                    | (
                                        merged[
                                            ["fga", "fta", "tov", "oreb", "dreb",
                                             "ast", "stl", "blk", "pf_committed"]
                                        ].sum(axis=1)
                                        > 0
                                    )
                                )
                            ).sum()
                            / max(
                                (
                                    (merged["minutes"] >= 0.5)
                                    | (
                                        merged[
                                            ["fga", "fta", "tov", "oreb", "dreb",
                                             "ast", "stl", "blk", "pf_committed"]
                                        ].sum(axis=1)
                                        > 0
                                    )
                                ).sum(),
                                1,
                            )
                        ),
                    },
                    "undated_rows": undated,
                    "minutes_per_game_team_mean": float(
                        merged.groupby("game_id")["minutes"].sum().mean() / 10.0
                    ),
                    "league": {
                        "ft_pct": float(merged["ftm"].sum() / max(merged["fta"].sum(), 1)),
                        "three_pct": float(
                            merged["three_fgm"].sum() / max(merged["three_fga"].sum(), 1)
                        ),
                        "rim_share_of_fga": float(
                            merged["rim_fga"].sum() / max(merged["fga"].sum(), 1)
                        ),
                        "mean_off_poss_per_game_team": float(
                            merged.groupby("game_id")["off_poss"].sum().mean() / 10.0
                        ),
                    },
                }
            )
            recon["per_season"].append(season_rec)
            all_lines.append(merged)
            print(
                f"{season} {st}: rows={len(merged)} games={season_rec['games']} "
                f"score_exact={season_rec['team_score_exact_share']:.4f} "
                f"ast_unres={season_rec['unresolved_tags']['ast']}/{season_rec['total_tags']['ast']} "
                f"({time.time()-t:.1f}s)"
            )

    panel = pd.concat(all_lines, ignore_index=True)
    panel["player_name"] = panel["player_id"].map(names)
    panel = panel.sort_values(
        ["player_id", "game_date", "game_id"], kind="mergesort"
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Offensive-fouls-drawn attribution evidence (documented blocker)
    # ------------------------------------------------------------------
    recon["off_fouls_drawn_attribution"] = {
        "status": "BLOCKED",
        "player2_id_populated_share_on_fouls": 0.0,
        "evidence": [
            "player2_id/player3_id are null on 100% of foul rows in every raw "
            "season file 2019-2026 (checked NBA19-NBA26).",
            "The trailing parenthetical in foul descriptions resolves to exactly "
            "3 distinct names per game (mean 3.00, max 3 in NBA24) matching NBA "
            "referee crews (S.Foster, Z.Zarba, J.Goble...), not defenders.",
            "Offensive Foul Turnover rows name only the committing player.",
            "The STOP lane's individual foul-drawn credit comes from Supabase "
            "season aggregates, not repo PBP.",
        ],
        "committed_side_available": True,
    }
    recon["totals"] = {
        "rows": int(len(panel)),
        "players": int(panel["player_id"].nunique()),
        "seasons": sorted(panel["season"].unique().tolist()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "player_game_lines.parquet"
    panel.to_parquet(out_path, index=False)
    (args.out_dir / "lines_reconciliation.json").write_text(json.dumps(recon, indent=2))
    print(
        f"\nwrote {out_path} ({len(panel)} rows, {panel['player_id'].nunique()} players) "
        f"+ lines_reconciliation.json in {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()

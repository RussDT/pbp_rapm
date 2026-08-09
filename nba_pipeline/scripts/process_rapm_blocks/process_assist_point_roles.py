"""
Role-attributed ASSIST_POINTS processors.

Each standard offensive possession is expanded to five focal-player rows:

* ASSIST_POINTS_RECEIVER assigns made assisted field-goal points to the shooter.
* ASSIST_POINTS_PASSER assigns the same points to the passer parsed from the
  ``(... N AST)`` description tag.

These are role-identification inputs, not final child RAPMs. The hierarchy
solver fits the passer-minus-receiver contrast on the common resolved sample,
then anchors that contrast to parent ``ASSIST_POINTS`` RAPM so the final passer
and receiver components sum exactly to the parent for every player.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    prepare_standard_possession_df,
)


ASSIST_NAME_PATTERN = re.compile(r"\(([^()]+?)\s+\d+\s+AST\)", re.IGNORECASE)
OFFENSE_COLS = [f"O{i}" for i in range(1, 6)]
CONTEXT_COLS = [f"O{i}" for i in range(1, 5)]
DEFENSE_COLS = [f"D{i}" for i in range(1, 6)]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def normalize_player_name(value) -> str:
    """Normalize a PBP name fragment for lineup-constrained matching."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def parse_assist_name(value) -> str:
    """Extract the passer name fragment from an NBA made-shot description."""
    match = ASSIST_NAME_PATTERN.search(str(value or ""))
    return match.group(1).strip() if match else ""


def build_player_aliases(nba_df: pd.DataFrame) -> dict[int, set[str]]:
    """Build player-id aliases from the raw PBP actor name columns."""
    aliases: dict[int, set[str]] = defaultdict(set)
    for id_col, name_col in [
        ("player1_id", "player1_name"),
        ("player2_id", "player2_name"),
        ("player3_id", "player3_name"),
    ]:
        if id_col not in nba_df.columns or name_col not in nba_df.columns:
            continue
        pairs = nba_df[[id_col, name_col]].dropna().drop_duplicates()
        for player_id, player_name in pairs.itertuples(index=False, name=None):
            try:
                player_id = int(float(player_id))
            except (TypeError, ValueError):
                continue
            if player_id <= 0:
                continue
            alias = normalize_player_name(player_name)
            if alias:
                aliases[player_id].add(alias)
    autocomplete = PROJECT_ROOT / "autocomplete_map.csv"
    if autocomplete.exists():
        try:
            names = pd.read_csv(autocomplete, usecols=["nba_id", "player_name"])
        except (OSError, ValueError):
            names = pd.DataFrame()
        for player_id, player_name in names.itertuples(index=False, name=None):
            try:
                player_id = int(float(player_id))
            except (TypeError, ValueError):
                continue
            alias = normalize_player_name(player_name)
            if player_id > 0 and alias:
                aliases[player_id].add(alias)
    return aliases


def learn_assist_tag_aliases(
    nba_df: pd.DataFrame,
    aliases: dict[int, set[str]],
) -> dict[int, set[str]]:
    """Learn stable abbreviated assist tags from their repeated lineup support.

    Modern PBP often stores only a surname in ``player1_name`` while the assist
    tag distinguishes same-surname teammates (for example ``Jal. Williams`` and
    ``Jay. Williams``). A tag is attached to a player only when one candidate is
    present on at least 90% of that tag's events and strictly more often than
    every other candidate.
    """
    home_cols = (
        [f"home_player{i}" for i in range(1, 6)]
        if "home_player1" in nba_df.columns
        else [f"h{i}" for i in range(1, 6)]
    )
    away_cols = (
        [f"away_player{i}" for i in range(1, 6)]
        if "away_player1" in nba_df.columns
        else [f"a{i}" for i in range(1, 6)]
    )
    required = {
        "event_type",
        "home_description",
        "visitor_description",
        *home_cols,
        *away_cols,
    }
    if not required.issubset(nba_df.columns):
        return aliases

    support: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    event_type = nba_df["event_type"]
    made_event = (
        pd.to_numeric(event_type, errors="coerce").eq(1)
        | event_type.astype(str).str.upper().eq("MAKE")
    )
    descriptions = (
        nba_df["home_description"].fillna("").astype(str)
        + " "
        + nba_df["visitor_description"].fillna("").astype(str)
    )
    assisted = nba_df[
        made_event
        & descriptions.str.contains(ASSIST_NAME_PATTERN, regex=True, na=False)
    ]
    for row in assisted.itertuples(index=False):
        home_description = str(getattr(row, "home_description") or "")
        visitor_description = str(getattr(row, "visitor_description") or "")
        description = f"{home_description} {visitor_description}"
        tag = normalize_player_name(parse_assist_name(description))
        if not tag:
            continue
        candidate_cols = home_cols if home_description.strip() else away_cols
        candidates = set()
        for column in candidate_cols:
            try:
                player_id = int(float(getattr(row, column)))
            except (TypeError, ValueError):
                continue
            if player_id > 0:
                candidates.add(player_id)
        totals[tag] += 1
        for player_id in candidates:
            support[tag][player_id] += 1

    for tag, candidate_counts in support.items():
        ranked = sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        top_player, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if (
            top_count >= 2
            and top_count / max(totals[tag], 1) >= 0.90
            and top_count > second_count
        ):
            aliases[top_player].add(tag)
    return aliases


def _surname_tokens(tokens: list[str]) -> tuple[str, str]:
    """Return first-name token and surname token, preserving common suffixes."""
    if not tokens:
        return "", ""
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    surname_idx = -2 if len(tokens) >= 2 and tokens[-1] in suffixes else -1
    return tokens[0], tokens[surname_idx]


def resolve_assister_id(
    assist_name: str,
    offense_ids,
    aliases: dict[int, set[str]],
    player2_id=None,
) -> int | None:
    """Resolve a passer against the five offensive players on the event row."""
    candidates = []
    for value in offense_ids:
        try:
            player_id = int(float(value))
        except (TypeError, ValueError):
            continue
        if player_id > 0 and player_id not in candidates:
            candidates.append(player_id)

    try:
        secondary_id = int(float(player2_id))
    except (TypeError, ValueError):
        secondary_id = 0
    if secondary_id > 0 and secondary_id in candidates:
        return secondary_id

    target = normalize_player_name(assist_name)
    if not target:
        return None

    exact = [
        player_id
        for player_id in candidates
        if target in aliases.get(player_id, set())
    ]
    if len(exact) == 1:
        return exact[0]

    scored = []
    target_tokens = target.split()
    target_first, target_surname = _surname_tokens(target_tokens)
    for player_id in candidates:
        best_score = 0
        for alias in aliases.get(player_id, set()):
            if alias.endswith(target) or target.endswith(alias):
                best_score = max(best_score, 4)
                continue
            alias_tokens = alias.split()
            alias_first, alias_surname = _surname_tokens(alias_tokens)
            if target_surname and target_surname == alias_surname:
                if len(target_tokens) == 1:
                    best_score = max(best_score, 1)
                elif (
                    alias_first.startswith(target_first)
                    or target_first.startswith(alias_first)
                ):
                    best_score = max(best_score, 3)
        if best_score:
            scored.append((best_score, player_id))
    if not scored:
        return None
    top_score = max(score for score, _ in scored)
    top_ids = list(dict.fromkeys(player_id for score, player_id in scored if score == top_score))
    return top_ids[0] if len(top_ids) == 1 else None


def _role_points_by_possession(
    nba_df: pd.DataFrame,
    role_col: str,
) -> pd.DataFrame:
    assisted = nba_df[
        nba_df["Assist_Points_Event"].gt(0) & nba_df[role_col].notna()
    ].copy()
    if assisted.empty:
        return pd.DataFrame(columns=["game_id", "poss_group", "Focal_Player", "Role_Assist_Points"])
    assisted["Focal_Player"] = pd.to_numeric(assisted[role_col], errors="coerce")
    assisted = assisted[assisted["Focal_Player"].gt(0)].copy()
    return (
        assisted.groupby(
            ["game_id", "poss_group", "Focal_Player"],
            as_index=False,
            sort=False,
        )["Assist_Points_Event"]
        .sum()
        .rename(columns={"Assist_Points_Event": "Role_Assist_Points"})
    )


def expand_role_possessions(
    terminal_rows: pd.DataFrame,
    role_points: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Expand each terminal possession into five focal-player opportunities."""
    base = terminal_rows.reset_index(drop=True).copy()
    repeated = base.loc[base.index.repeat(5)].reset_index(drop=True)
    offense = base[OFFENSE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
    repeated["Focal_Player"] = offense.reshape(-1)

    context_blocks = []
    for focal_idx in range(5):
        context_blocks.append(np.delete(offense, focal_idx, axis=1))
    repeated[CONTEXT_COLS] = np.stack(context_blocks, axis=1).reshape(-1, 4)

    lookup = role_points.rename(columns={"Role_Assist_Points": target_col})
    repeated = repeated.merge(
        lookup,
        on=["game_id", "poss_group", "Focal_Player"],
        how="left",
        validate="many_to_one",
    )
    repeated[target_col] = (
        pd.to_numeric(repeated[target_col], errors="coerce").fillna(0).astype(int)
    )
    resolved_by_possession = repeated.groupby(
        ["game_id", "poss_group"], sort=False
    )[target_col].transform("sum")
    repeated["Unresolved_Assist_Points"] = (
        pd.to_numeric(repeated["Assist_Points"], errors="coerce").fillna(0).astype(int)
        - resolved_by_possession
    )
    repeated["Role_Resolution_Complete"] = repeated[
        "Unresolved_Assist_Points"
    ].eq(0)
    return repeated


def process_assist_point_roles_py(file_path, year, season_str):
    """Build receiver and passer player-opportunity parquets for one season."""
    print(f"  Starting ASSIST_POINTS role processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    aliases = learn_assist_tag_aliases(nba_df, build_player_aliases(nba_df))
    nba_df, group_cols = prepare_standard_possession_df(nba_df, "ASSIST_POINTS_ROLES")
    shot_flags = build_shot_flags_df(nba_df)
    nba_df["Assist_Points_Event"] = shot_flags["shot_points"].where(
        shot_flags["is_assisted_make"], 0
    ).astype(int)

    nba_df["Assist_Receiver"] = pd.to_numeric(
        nba_df.get("player1_id"), errors="coerce"
    ).where(nba_df["Assist_Points_Event"].gt(0))
    nba_df["Assist_Name"] = shot_flags["shot_text"].map(parse_assist_name)

    assisted_mask = nba_df["Assist_Points_Event"].gt(0)
    nba_df["Assist_Passer"] = np.nan
    assisted_rows = nba_df.loc[
        assisted_mask,
        [*OFFENSE_COLS, "Assist_Name", "player2_id"],
    ]
    resolved_passers = []
    for row in assisted_rows.itertuples(index=False):
        row_dict = row._asdict()
        resolved_passers.append(
            resolve_assister_id(
                row_dict["Assist_Name"],
                [row_dict[col] for col in OFFENSE_COLS],
                aliases,
                player2_id=row_dict.get("player2_id"),
            )
        )
    nba_df.loc[assisted_mask, "Assist_Passer"] = pd.Series(
        resolved_passers,
        index=nba_df.index[assisted_mask],
        dtype="float64",
    )

    receiver_in_lineup = nba_df[OFFENSE_COLS].eq(
        nba_df["Assist_Receiver"], axis=0
    ).any(axis=1)
    nba_df.loc[assisted_mask & ~receiver_in_lineup, "Assist_Receiver"] = np.nan

    if group_cols:
        nba_df["Assist_Points_Total"] = nba_df.groupby(group_cols)[
            "Assist_Points_Event"
        ].cumsum()
    else:
        nba_df["Assist_Points_Total"] = nba_df["Assist_Points_Event"].cumsum()

    terminal = nba_df[nba_df["End_of_Possession"]].copy()
    if group_cols:
        terminal["Assist_Points"] = terminal["Assist_Points_Total"] - terminal.groupby(
            group_cols
        )["Assist_Points_Total"].shift(fill_value=0)
    else:
        terminal["Assist_Points"] = (
            terminal["Assist_Points_Total"]
            - terminal["Assist_Points_Total"].shift(fill_value=0)
        )
    terminal["Assist_Points"] = terminal["Assist_Points"].astype(int)

    receiver_points = _role_points_by_possession(nba_df, "Assist_Receiver")
    passer_points = _role_points_by_possession(nba_df, "Assist_Passer")
    receiver_long = expand_role_possessions(
        terminal, receiver_points, "Receiver_Assist_Points"
    )
    passer_long = expand_role_possessions(
        terminal, passer_points, "Passer_Assist_Points"
    )

    receiver_output = _finalize_df(
        receiver_long,
        [
            "Receiver_Assist_Points",
            "Assist_Points",
            "Unresolved_Assist_Points",
            "Role_Resolution_Complete",
        ],
        year,
        id_cols=["game_id", "poss_group", "Focal_Player"],
        o_cols=CONTEXT_COLS,
        d_cols=DEFENSE_COLS,
    )
    passer_output = _finalize_df(
        passer_long,
        [
            "Passer_Assist_Points",
            "Assist_Points",
            "Unresolved_Assist_Points",
            "Role_Resolution_Complete",
        ],
        year,
        id_cols=["game_id", "poss_group", "Focal_Player"],
        o_cols=CONTEXT_COLS,
        d_cols=DEFENSE_COLS,
    )

    total_points = int(nba_df["Assist_Points_Event"].sum())
    resolved_receiver_points = int(
        nba_df.loc[nba_df["Assist_Receiver"].notna(), "Assist_Points_Event"].sum()
    )
    resolved_passer_points = int(
        nba_df.loc[nba_df["Assist_Passer"].notna(), "Assist_Points_Event"].sum()
    )
    print(
        "      Assisted-point role resolution: "
        f"receiver={resolved_receiver_points:,}/{total_points:,}, "
        f"passer={resolved_passer_points:,}/{total_points:,}"
    )
    print(
        f"  Finished ASSIST_POINTS role processing in {time.time() - start_time:.2f}s."
    )
    return receiver_output, passer_output

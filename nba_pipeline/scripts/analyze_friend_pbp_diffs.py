#!/usr/bin/env python3
"""
Write concrete diff artifacts for the BKN 2020 friend-PBP validation fixture.

Run after validate_friend_pbp_ingest.py.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_ROOT = PIPELINE_ROOT / "validation" / "friend_ingest"
PROCESSED_ROOT = VALIDATION_ROOT / "processed"
DIFF_ROOT = VALIDATION_ROOT / "diffs"

OFFICIAL_RAW = PIPELINE_ROOT / "raw_data" / "BKN_2020_rs_ours.parquet"
FRIEND_ORIGINAL_RAW = PIPELINE_ROOT / "raw_data" / "BKN_2020_rs.parquet"
FRIEND_CONVERTED_RAW = VALIDATION_ROOT / "raw" / "BKN_2020_rs_friend_converted.parquet"

METRICS = {
    "RAPM": ["Net_Diff", "Off_Diff", "Def_Diff"],
    "TS": ["Net_Diff"],
    "TOV": ["Is_Turnover", "Is_BadPass_TOV"],
    "REB": ["Offensive_Rebound"],
    "RIM_FREQ": ["Is_Rim_Attempt"],
    "RIM_FG_PCT": ["Is_Rim_Make"],
    "THREE_FREQ": ["Is_Three_Attempt"],
    "THREE_FG_PCT": ["Is_Three_Make"],
    "MIDRANGE_FREQ": ["Is_Midrange_Attempt"],
    "MIDRANGE_FG_PCT": ["Is_Midrange_Make"],
    "ASSIST_POINTS": ["Assist_Points"],
}

EVENT_TYPE_NAMES = {
    1: "MAKE",
    2: "MISS",
    3: "FT",
    4: "REB",
    5: "TOV",
    6: "FOUL",
    7: "VIOLATION",
    8: "SUB",
    9: "TIMEOUT",
    10: "JUMP",
    11: "EJECT",
    12: "START",
    13: "END",
    14: "EMPTY",
}


def desc(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    return (
        df.get(f"{prefix}home_description", pd.Series("", index=df.index)).fillna("").astype(str)
        + " | "
        + df.get(f"{prefix}visitor_description", pd.Series("", index=df.index)).fillna("").astype(str)
        + " | "
        + df.get(f"{prefix}neutral_description", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.strip()


def enrich_friend_raw(friend_raw: pd.DataFrame) -> pd.DataFrame:
    friend_raw = friend_raw.sort_values(["game_id", "period", "event_num"], kind="stable").reset_index(drop=True)
    friend_raw["friend_event_type_name"] = friend_raw["event_type"].map(EVENT_TYPE_NAMES).fillna(
        friend_raw["event_type"].astype(str)
    )
    friend_raw["friend_desc"] = desc(friend_raw)
    grouped = friend_raw.groupby("game_id", sort=False)
    friend_raw["friend_prev_event_num"] = grouped["event_num"].shift(1)
    friend_raw["friend_prev_event_type"] = grouped["event_type"].shift(1)
    friend_raw["friend_prev_event_type_name"] = friend_raw["friend_prev_event_type"].map(EVENT_TYPE_NAMES).fillna(
        friend_raw["friend_prev_event_type"].astype(str)
    )
    friend_raw["friend_prev_desc"] = (
        grouped["home_description"].shift(1).fillna("").astype(str)
        + " | "
        + grouped["visitor_description"].shift(1).fillna("").astype(str)
        + " | "
        + grouped["neutral_description"].shift(1).fillna("").astype(str)
    ).str.strip()
    return friend_raw


def write_raw_schema_diff(summary: dict) -> None:
    official_raw = pd.read_parquet(OFFICIAL_RAW)
    friend_original = pd.read_parquet(FRIEND_ORIGINAL_RAW)
    friend_converted = pd.read_parquet(FRIEND_CONVERTED_RAW)

    official_keys = set(zip(official_raw["game_id"].astype(int), official_raw["event_num"].astype(int)))
    friend_keys = set(zip(friend_converted["game_id"].astype(int), friend_converted["event_num"].astype(int)))

    official_only = pd.DataFrame(sorted(official_keys - friend_keys), columns=["game_id", "event_num"])
    friend_only = pd.DataFrame(sorted(friend_keys - official_keys), columns=["game_id", "event_num"])
    official_only.to_csv(DIFF_ROOT / "raw_official_only_events.csv", index=False)
    friend_only.to_csv(DIFF_ROOT / "raw_friend_only_events.csv", index=False)

    dup_sets = friend_original.groupby(["game_id", "actionNumber"])["actionType"].agg(lambda values: "|".join(sorted(set(values))))
    dup_sets = dup_sets.reset_index(name="action_types")
    block_dups = dup_sets["action_types"].str.contains("block") & dup_sets["action_types"].str.contains("2pt|3pt", regex=True)
    steal_dups = dup_sets["action_types"].str.contains("steal") & dup_sets["action_types"].str.contains("turnover")

    summary["raw_event_coverage"] = {
        "official_rows": int(len(official_raw)),
        "friend_converted_rows": int(len(friend_converted)),
        "official_unique_game_events": int(len(official_keys)),
        "friend_unique_game_events": int(len(friend_keys)),
        "official_only_unique_events": int(len(official_only)),
        "friend_only_unique_events": int(len(friend_only)),
        "friend_duplicate_game_event_groups": int((friend_original.groupby(["game_id", "actionNumber"]).size() > 1).sum()),
        "friend_block_rows": int((friend_original["actionType"] == "block").sum()),
        "friend_steal_rows": int((friend_original["actionType"] == "steal").sum()),
        "friend_block_duplicate_action_numbers": int(block_dups.sum()),
        "friend_steal_duplicate_action_numbers": int(steal_dups.sum()),
    }


def write_processed_diffs(summary: dict) -> None:
    official_raw = pd.read_parquet(OFFICIAL_RAW)
    official_raw["official_event_type_name"] = official_raw["event_type"].map(EVENT_TYPE_NAMES).fillna(
        official_raw["event_type"].astype(str)
    )
    official_raw["official_desc"] = desc(official_raw)
    official_context = official_raw[
        ["game_id", "event_num", "official_event_type_name", "official_desc"]
    ].copy()
    friend_raw = enrich_friend_raw(pd.read_parquet(FRIEND_CONVERTED_RAW))
    friend_context = (
        friend_raw.groupby(["game_id", "event_num"], as_index=False)
        .agg(
            friend_event_type_name=("friend_event_type_name", lambda values: "|".join(pd.Series(values).dropna().astype(str).unique())),
            friend_desc=("friend_desc", lambda values: " || ".join(pd.Series(values).dropna().astype(str).unique())),
            friend_prev_event_num=("friend_prev_event_num", "first"),
            friend_prev_event_type_name=("friend_prev_event_type_name", lambda values: "|".join(pd.Series(values).dropna().astype(str).unique())),
            friend_prev_desc=("friend_prev_desc", lambda values: " || ".join(pd.Series(values).dropna().astype(str).unique())),
        )
    )

    metric_summary = {}
    raw_cols = [
        "game_id",
        "event_num",
        "period",
        "time_quarter",
        "official_event_type_name",
        "official_desc",
        "friend_event_type_name",
        "friend_desc",
        "friend_prev_event_num",
        "friend_prev_event_type_name",
        "friend_prev_desc",
    ]
    for metric, metric_cols in METRICS.items():
        official_path = PROCESSED_ROOT / "official" / f"{metric}20.parquet"
        friend_path = PROCESSED_ROOT / "friend_converted" / f"{metric}20.parquet"
        if not official_path.exists() or not friend_path.exists():
            continue

        official = pd.read_parquet(official_path)
        friend = pd.read_parquet(friend_path)
        keys = ["game_id", "event_num"]
        missing = official.merge(friend[keys], on=keys, how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
        missing = missing.merge(
            official_raw[["game_id", "event_num", "official_event_type_name", "official_desc"]],
            on=keys,
            how="left",
        ).merge(
            friend_context[
                [
                    "game_id",
                    "event_num",
                    "friend_event_type_name",
                    "friend_desc",
                    "friend_prev_event_num",
                    "friend_prev_event_type_name",
                    "friend_prev_desc",
                ]
            ],
            on=keys,
            how="left",
        )
        missing["friend_prev_is_aux_block"] = missing["friend_prev_desc"].str.contains("BLOCK", case=False, na=False)
        missing["friend_prev_is_aux_steal"] = missing["friend_prev_desc"].str.contains("STEAL", case=False, na=False)
        missing["friend_same_event_exists"] = missing["friend_event_type_name"].notna()

        ordered_cols = [col for col in raw_cols + metric_cols + ["friend_prev_is_aux_block", "friend_prev_is_aux_steal", "friend_same_event_exists"] if col in missing.columns]
        missing[ordered_cols].to_csv(DIFF_ROOT / f"{metric.lower()}_official_unmatched_rows.csv", index=False)

        friend_missing = friend.merge(official[keys], on=keys, how="left", indicator=True)
        friend_missing = friend_missing[friend_missing["_merge"] == "left_only"].drop(columns=["_merge"])
        friend_missing = friend_missing.merge(
            friend_context[
                [
                    "game_id",
                    "event_num",
                    "friend_event_type_name",
                    "friend_desc",
                    "friend_prev_event_num",
                    "friend_prev_event_type_name",
                    "friend_prev_desc",
                ]
            ],
            on=keys,
            how="left",
        ).merge(
            official_context,
            on=keys,
            how="left",
        )
        friend_missing["official_same_event_exists"] = friend_missing["official_event_type_name"].notna()
        friend_ordered_cols = [
            col
            for col in [
                "game_id",
                "event_num",
                "friend_event_type_name",
                "friend_desc",
                "friend_prev_event_num",
                "friend_prev_event_type_name",
                "friend_prev_desc",
                "official_event_type_name",
                "official_desc",
            ]
            + metric_cols
            + ["official_same_event_exists"]
            if col in friend_missing.columns
        ]
        friend_missing[friend_ordered_cols].to_csv(
            DIFF_ROOT / f"{metric.lower()}_friend_unmatched_rows.csv", index=False
        )

        matched = official.merge(friend, on=keys, suffixes=("_official", "_friend"))
        matched_out = {col: {} for col in metric_cols if f"{col}_official" in matched and f"{col}_friend" in matched}
        for col in list(matched_out):
            left = pd.to_numeric(matched[f"{col}_official"], errors="coerce")
            right = pd.to_numeric(matched[f"{col}_friend"], errors="coerce")
            diff = right - left
            matched_out[col] = {
                "matched_rows": int(len(matched)),
                "mae": float(diff.abs().mean()) if len(diff) else None,
                "exact_rate": float((diff == 0).mean()) if len(diff) else None,
                "corr": float(left.corr(right)) if left.nunique() > 1 and right.nunique() > 1 else None,
            }

        metric_summary[metric] = {
            "official_rows": int(len(official)),
            "friend_rows": int(len(friend)),
            "official_unmatched_rows": int(len(missing)),
            "friend_unmatched_rows": int(len(friend_missing)),
            "friend_same_event_exists_for_official_unmatched": int(missing["friend_same_event_exists"].sum()),
            "official_same_event_exists_for_friend_unmatched": int(friend_missing["official_same_event_exists"].sum()),
            "official_unmatched_with_friend_prev_aux_block": int(missing["friend_prev_is_aux_block"].sum()),
            "official_unmatched_with_friend_prev_aux_steal": int(missing["friend_prev_is_aux_steal"].sum()),
            "official_unmatched_event_type_counts": missing["official_event_type_name"].value_counts(dropna=False).to_dict(),
            "friend_unmatched_event_type_counts": friend_missing["friend_event_type_name"].value_counts(dropna=False).to_dict(),
            "matched_metric_comparison": matched_out,
        }

    summary["processed_metric_diffs"] = metric_summary


def write_lineup_diffs(summary: dict) -> None:
    official_path = PROCESSED_ROOT / "official" / "RAPM20.parquet"
    friend_path = PROCESSED_ROOT / "friend_converted" / "RAPM20.parquet"
    if not official_path.exists() or not friend_path.exists():
        return

    official = pd.read_parquet(official_path)
    friend = pd.read_parquet(friend_path)
    merged = official.merge(friend, on=["game_id", "event_num"], suffixes=("_official", "_friend"))
    rows = []
    for _, row in merged.iterrows():
        official_off = tuple(sorted(int(row[f"O{i}_official"]) for i in range(1, 6)))
        official_def = tuple(sorted(int(row[f"D{i}_official"]) for i in range(1, 6)))
        friend_off = tuple(sorted(int(row[f"O{i}_friend"]) for i in range(1, 6)))
        friend_def = tuple(sorted(int(row[f"D{i}_friend"]) for i in range(1, 6)))
        if official_off == friend_off and official_def == friend_def:
            continue
        official_players = set(official_off) | set(official_def)
        friend_players = set(friend_off) | set(friend_def)
        rows.append(
            {
                "game_id": int(row["game_id"]),
                "event_num": int(row["event_num"]),
                "official_offense": "|".join(map(str, official_off)),
                "friend_offense": "|".join(map(str, friend_off)),
                "official_defense": "|".join(map(str, official_def)),
                "friend_defense": "|".join(map(str, friend_def)),
                "official_only_players": "|".join(map(str, sorted(official_players - friend_players))),
                "friend_only_players": "|".join(map(str, sorted(friend_players - official_players))),
                "net_diff_official": row.get("Net_Diff_official"),
                "net_diff_friend": row.get("Net_Diff_friend"),
            }
        )

    lineup_diffs = pd.DataFrame(rows)
    lineup_diffs.to_csv(DIFF_ROOT / "rapm_lineup_mismatches.csv", index=False)
    summary["rapm_lineup_diffs"] = {
        "matched_rapm_rows": int(len(merged)),
        "lineup_mismatch_rows": int(len(lineup_diffs)),
        "lineup_mismatch_rate": float(len(lineup_diffs) / len(merged)) if len(merged) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write friend-PBP validation diff artifacts.")
    parser.add_argument("--validation-root", type=Path, default=VALIDATION_ROOT)
    return parser.parse_args()


def configure_paths(validation_root: Path) -> None:
    global VALIDATION_ROOT, PROCESSED_ROOT, DIFF_ROOT, FRIEND_CONVERTED_RAW
    VALIDATION_ROOT = validation_root
    PROCESSED_ROOT = VALIDATION_ROOT / "processed"
    DIFF_ROOT = VALIDATION_ROOT / "diffs"
    FRIEND_CONVERTED_RAW = VALIDATION_ROOT / "raw" / "BKN_2020_rs_friend_converted.parquet"


def main() -> int:
    args = parse_args()
    configure_paths(args.validation_root)
    DIFF_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    write_raw_schema_diff(summary)
    write_processed_diffs(summary)
    write_lineup_diffs(summary)
    summary_path = DIFF_ROOT / "difference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

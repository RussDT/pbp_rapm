#!/usr/bin/env python3
"""Build negative-only LAKER priors for the 1997-2026 age-27 RAPM parent.

LAKER is stored as positive-good offense and defense.  The RAPM solver stores
defense as points allowed, so the defense prior CSV emitted here flips that
sign exactly once. Eligibility follows the requested research rule: a player
must have a negative minutes-weighted age-27 LAKER total and no more than the
configured regular-season career minutes. Because raw LAKER is extremely
noisy for tiny samples, its center is reliability-weighted by
``minutes / (minutes + stabilization_minutes)`` before entering RAPM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


LAKER_COLUMNS = [
    "nba_id",
    "player_id",
    "player_name",
    "year",
    "age",
    "team_abbreviation",
    "playoffs",
    "minutes",
    "orapm",
    "drapm",
    "rapm",
]


def _normalize_nba_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return None
    return str(int(numeric))


def fetch_laker_history(
    *,
    supabase_url: str,
    supabase_key: str,
    start_year: int,
    end_year: int,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Fetch regular-season LAKER rows from the public Supabase table."""

    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/LAKER_history"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = requests.get(
            endpoint,
            headers=headers,
            params={
                "select": ",".join(LAKER_COLUMNS),
                "year": f"gte.{int(start_year)}",
                "and": f"(year.lte.{int(end_year)},playoffs.eq.0)",
                "order": "year.asc,nba_id.asc,team_abbreviation.asc",
                "limit": page_size,
                "offset": offset,
            },
            timeout=60,
        )
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError("Supabase LAKER response was not a row list.")
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return pd.DataFrame(rows, columns=LAKER_COLUMNS)


def load_age_curve(path: Path) -> pd.DataFrame:
    curve = pd.read_csv(path)
    required = {"age", "orapm_age_effect", "drapm_age_effect"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Age curve missing columns: {sorted(missing)}")
    curve = curve.copy()
    for column in required:
        curve[column] = pd.to_numeric(curve[column], errors="coerce")
    curve = curve.dropna(subset=list(required)).sort_values("age")
    if curve.empty or curve["age"].duplicated().any():
        raise ValueError("Age curve must contain unique numeric ages.")
    return curve


def build_negative_laker_priors(
    laker_rows: pd.DataFrame,
    age_curve: pd.DataFrame,
    *,
    start_year: int = 1997,
    end_year: int = 2026,
    minute_threshold: float = 1000.0,
    prior_scale: float = 1.0,
    stabilization_minutes: float = 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return offense CSV, raw-defense CSV, player audit, and receipt."""

    if minute_threshold <= 0:
        raise ValueError("minute_threshold must be positive.")
    if prior_scale < 0:
        raise ValueError("prior_scale must be nonnegative.")
    if stabilization_minutes < 0:
        raise ValueError("stabilization_minutes must be nonnegative.")

    required = set(LAKER_COLUMNS)
    missing = required - set(laker_rows.columns)
    if missing:
        raise ValueError(f"LAKER history missing columns: {sorted(missing)}")

    frame = laker_rows.copy()
    frame["player_id"] = frame["nba_id"].map(_normalize_nba_id)
    numeric = ["year", "age", "playoffs", "minutes", "orapm", "drapm", "rapm"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    initial_rows = len(frame)
    tot_rows = int(frame["team_abbreviation"].astype(str).str.upper().eq("TOT").sum())
    frame = frame[
        frame["year"].between(int(start_year), int(end_year), inclusive="both")
        & frame["playoffs"].eq(0)
        & frame["player_id"].notna()
        & frame["minutes"].gt(0)
        & ~frame["team_abbreviation"].astype(str).str.upper().eq("TOT")
    ].copy()
    impact_complete = frame[["orapm", "drapm", "rapm"]].notna().all(axis=1)
    age_complete = frame["age"].notna()
    incomplete_impact_rows = int((~impact_complete).sum())
    incomplete_age_rows = int((impact_complete & ~age_complete).sum())
    frame = frame[impact_complete & age_complete].copy()
    if frame.empty:
        raise ValueError("No complete regular-season LAKER rows survived filtering.")

    ages = age_curve["age"].to_numpy(dtype=float)
    clipped_age = frame["age"].clip(float(ages.min()), float(ages.max()))
    frame["orapm_age_effect"] = np.interp(
        clipped_age,
        ages,
        age_curve["orapm_age_effect"].to_numpy(dtype=float),
    )
    frame["drapm_age_effect"] = np.interp(
        clipped_age,
        ages,
        age_curve["drapm_age_effect"].to_numpy(dtype=float),
    )
    frame["orapm_age27"] = frame["orapm"] - frame["orapm_age_effect"]
    frame["drapm_age27"] = frame["drapm"] - frame["drapm_age_effect"]
    frame["net_age27"] = frame["orapm_age27"] + frame["drapm_age27"]
    frame["identity_gap"] = frame["rapm"] - frame["orapm"] - frame["drapm"]

    weighted_columns = [
        "orapm",
        "drapm",
        "rapm",
        "orapm_age27",
        "drapm_age27",
        "net_age27",
        "age",
        "identity_gap",
    ]
    for column in weighted_columns:
        frame[f"weighted_{column}"] = frame["minutes"] * frame[column]

    name_rows = (
        frame.sort_values(["minutes", "year"], ascending=[False, False])
        .drop_duplicates("player_id")[["player_id", "player_name"]]
    )
    sums = frame.groupby("player_id", as_index=False).agg(
        career_minutes=("minutes", "sum"),
        seasons=("year", "nunique"),
        team_stints=("year", "size"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        **{f"sum_{column}": (f"weighted_{column}", "sum") for column in weighted_columns},
    )
    audit = sums.merge(name_rows, on="player_id", how="left", validate="one_to_one")
    for column in weighted_columns:
        audit[f"career_{column}"] = audit[f"sum_{column}"] / audit["career_minutes"]

    audit["negative_laker"] = audit["career_net_age27"] < 0
    audit["under_minute_threshold"] = audit["career_minutes"] <= float(minute_threshold)
    audit["eligible"] = audit["negative_laker"] & audit["under_minute_threshold"]
    audit["prior_scale"] = float(prior_scale)
    audit["stabilization_minutes"] = float(stabilization_minutes)
    audit["prior_reliability"] = audit["career_minutes"] / (
        audit["career_minutes"] + float(stabilization_minutes)
    )
    audit["off_prior_age27"] = np.where(
        audit["eligible"],
        prior_scale * audit["prior_reliability"] * audit["career_orapm_age27"],
        0.0,
    )
    audit["def_prior_age27_positive_good"] = np.where(
        audit["eligible"],
        prior_scale * audit["prior_reliability"] * audit["career_drapm_age27"],
        0.0,
    )
    audit["net_prior_age27"] = (
        audit["off_prior_age27"] + audit["def_prior_age27_positive_good"]
    )
    audit["def_prior_solver_raw"] = -audit["def_prior_age27_positive_good"]
    audit["minute_threshold"] = float(minute_threshold)

    display_columns = [
        "player_id",
        "player_name",
        "career_minutes",
        "seasons",
        "team_stints",
        "first_year",
        "last_year",
        "career_age",
        "career_orapm",
        "career_drapm",
        "career_rapm",
        "career_orapm_age27",
        "career_drapm_age27",
        "career_net_age27",
        "negative_laker",
        "under_minute_threshold",
        "eligible",
        "prior_scale",
        "stabilization_minutes",
        "prior_reliability",
        "off_prior_age27",
        "def_prior_age27_positive_good",
        "net_prior_age27",
        "def_prior_solver_raw",
        "career_identity_gap",
        "minute_threshold",
    ]
    audit = audit[display_columns].sort_values(
        ["eligible", "career_minutes", "player_name"],
        ascending=[False, True, True],
    )

    eligible = audit[audit["eligible"]].copy()
    off_prior = eligible[["player_id", "off_prior_age27"]].rename(
        columns={"off_prior_age27": "prior_value"}
    )
    def_prior = eligible[["player_id", "def_prior_solver_raw"]].rename(
        columns={"def_prior_solver_raw": "prior_value"}
    )

    row_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            frame.sort_values(["year", "player_id", "team_abbreviation"]), index=False
        ).values.tobytes()
    ).hexdigest()
    receipt = {
        "model": "negative_laker_age27_prior",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "minute_threshold": float(minute_threshold),
        "prior_scale": float(prior_scale),
        "stabilization_minutes": float(stabilization_minutes),
        "eligibility": "minutes-weighted age-27 career LAKER total < 0 and RS career minutes <= threshold",
        "prior_estimand": "minutes-weighted LAKER O/D translated to age 27, then reliability-weighted by minutes/(minutes+stabilization_minutes)",
        "defense_prior_sign": "raw solver points-allowed; negative is positive-good",
        "input_rows": int(initial_rows),
        "complete_rows": int(len(frame)),
        "excluded_tot_rows": int(tot_rows),
        "excluded_incomplete_impact_rows": int(incomplete_impact_rows),
        "excluded_incomplete_age_rows": int(incomplete_age_rows),
        "players": int(len(audit)),
        "eligible_players": int(audit["eligible"].sum()),
        "eligible_minutes": float(audit.loc[audit["eligible"], "career_minutes"].sum()),
        "max_abs_laker_identity_gap": float(frame["identity_gap"].abs().max()),
        "source_row_hash_sha256": row_hash,
    }
    return off_prior, def_prior, audit, receipt


def _scale_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-csv", type=Path)
    source.add_argument("--supabase-env", type=Path)
    parser.add_argument("--age-curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--minute-threshold", type=float, default=1000.0)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--stabilization-minutes", type=float, default=500.0)
    parser.add_argument("--cache-source", type=Path)
    args = parser.parse_args()

    if args.source_csv:
        laker = pd.read_csv(args.source_csv)
        source_description = str(args.source_csv)
    else:
        load_dotenv(args.supabase_env, override=False)
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase env file must provide SUPABASE_URL and SUPABASE_KEY/ANON_KEY.")
        laker = fetch_laker_history(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        source_description = "Supabase public.LAKER_history"

    if args.cache_source:
        args.cache_source.parent.mkdir(parents=True, exist_ok=True)
        laker.to_csv(args.cache_source, index=False)

    curve = load_age_curve(args.age_curve)
    off_prior, def_prior, audit, receipt = build_negative_laker_priors(
        laker,
        curve,
        start_year=args.start_year,
        end_year=args.end_year,
        minute_threshold=args.minute_threshold,
        prior_scale=args.prior_scale,
        stabilization_minutes=args.stabilization_minutes,
    )
    receipt["source"] = source_description.replace("\u200b", "")
    receipt["age_curve"] = str(args.age_curve)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = (
        f"m{int(args.minute_threshold)}_k{int(args.stabilization_minutes)}_"
        f"s{_scale_label(args.prior_scale)}"
    )
    off_path = args.output_dir / f"negative_laker_{label}_off_prior.csv"
    def_path = args.output_dir / f"negative_laker_{label}_def_prior.csv"
    audit_path = args.output_dir / f"negative_laker_{label}_audit.csv"
    receipt_path = args.output_dir / f"negative_laker_{label}_receipt.json"
    off_prior.to_csv(off_path, index=False)
    def_prior.to_csv(def_path, index=False)
    audit.to_csv(audit_path, index=False)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    print(json.dumps({
        "off_prior": str(off_path),
        "def_prior": str(def_path),
        "audit": str(audit_path),
        "receipt": str(receipt_path),
        "players": receipt["players"],
        "eligible_players": receipt["eligible_players"],
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a PBP-only WNBA shot-role (playtype) difficulty proxy.

The WNBA has no Synergy playtype feed, so the NBA six-family PBP taxonomy in
``nba_pipeline/scripts/analyze_pbp_playtype_difficulty_ots.py`` is ported here
verbatim: shot families are assigned from outcome-invariant description tokens,
priced with the immediately preceding completed regular season, and the feature
season's player shot mix is valued at those fixed prices.

Two level conventions are emitted because WNBA description vocabulary drifts
much harder than the NBA's:

``prior_league`` reproduces the NBA definition exactly (prior-season league
points/FGA minus the prior-priced player mix).

``season_centered`` replaces the prior-season league reference with the
feature season's own FGA-weighted mean of the prior-priced mix, which removes
the league-wide descriptor-drift level shift and leaves a pure within-season
role contrast.

Nothing here uses make/miss, realized points, score state, or assist tags when
assigning a family, so the feature can adjust observed shooting without
reusing the shooter's own result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PBP_DIR = ROOT / "nba_pipeline" / "external" / "gabriel_wnba" / "pbp_data"
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "playtype_difficulty"

PRICE_SHRINK_FGA = 100.0

# Display-only scaling for the reader-facing role-adjusted efficiency column.
# ``playtype_diff`` itself stays RAW: forecasting must carry it as its own
# regressor, the box-prior lane fits its own weight against a component target,
# and the NBA columns are unscaled, so pre-scaling the feature would break both
# the other two consumers and cross-league comparability.  This coefficient is
# the fitted next-season role slope from evaluate_wnba_playtype_difficulty.py
# (-0.814, 95% CI [-0.995, -0.638]) on development seasons 2019-2024.  It is
# provisional and pre-registered for re-confirmation once 2026 completes; it
# must not be re-tuned on 2026.  Any artifact records it in manifest.json so a
# stored file can be traced to the coefficient it was built under.
ROLE_DISPLAY_COEFFICIENT = 0.81

PBP_CATEGORIES = [
    "CUT_2",
    "SELF_CREATED_3",
    "OTHER_3",
    "SELF_CREATED_2",
    "OTHER_RIM_2",
    "OTHER_2",
]
# Frozen NBA patterns.  Ported unchanged so the two leagues share one taxonomy.
SELF_CREATED_PATTERN = (
    r"pull.?up|step.?back|turnaround|fadeaway|driving|floating|floater|hook"
)
CUT_PATTERN = r"cutting|alley oop"
RIM_PATTERN = r"dunk|layup|tip"

# WNBA regular-season game ids are 102YYNNNNN; playoffs are 104YYNNNNN.
REGULAR_SEASON_PREFIX = "102"
PLAYOFF_PREFIX = "104"

# Description richness collapses before 2016 (``cutting`` is literally absent),
# so seasons below this are loadable but never a default pricing or feature
# season.
MODERN_DESCRIPTION_FIRST_SEASON = 2018


def classify_wnba_shots(
    description: pd.Series, is_three: pd.Series
) -> pd.Series:
    """Assign outcome-invariant broad shot families from WNBA descriptions."""
    text = description.fillna("").astype(str).str.lower()
    cut = text.str.contains(CUT_PATTERN, regex=True)
    self_created = text.str.contains(SELF_CREATED_PATTERN, regex=True)
    rim = text.str.contains(RIM_PATTERN, regex=True)
    conditions = [
        cut & ~is_three,
        self_created & is_three,
        is_three,
        self_created & ~is_three,
        rim & ~is_three,
    ]
    choices = [
        "CUT_2",
        "SELF_CREATED_3",
        "OTHER_3",
        "SELF_CREATED_2",
        "OTHER_RIM_2",
    ]
    return pd.Series(
        np.select(conditions, choices, default="OTHER_2"),
        index=description.index,
        name="pbp_playtype_category",
    )


def _season_from_game_id(game_id: str) -> int:
    return 2000 + int(str(game_id)[3:5])


def load_season_shots(
    pbp_dir: Path, season: int, *, season_type: str = "RS"
) -> pd.DataFrame:
    """Load one WNBA season's FGA with shooter, family, and realized points."""
    prefixes = []
    if season_type in {"RS", "ALL"}:
        prefixes.append(REGULAR_SEASON_PREFIX)
    if season_type in {"PS", "ALL"}:
        prefixes.append(PLAYOFF_PREFIX)
    paths = sorted(
        path
        for prefix in prefixes
        for path in pbp_dir.glob(f"{prefix}{season % 100:02d}*.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No {season_type} play-by-play for {season}")
    frames = []
    for path in paths:
        frame = pd.read_csv(
            path,
            low_memory=False,
            usecols=[
                "GAME_ID",
                "EVENTNUM",
                "EVENTMSGTYPE",
                "HOMEDESCRIPTION",
                "VISITORDESCRIPTION",
                "PLAYER1_ID",
                "PLAYER1_NAME",
            ],
        )
        frames.append(frame[frame["EVENTMSGTYPE"].isin([1, 2])])
    shots = pd.concat(frames, ignore_index=True)
    description = (
        shots["HOMEDESCRIPTION"]
        .fillna("")
        .astype(str)
        .str.cat(shots["VISITORDESCRIPTION"].fillna("").astype(str), sep=" ")
    )
    is_three = description.str.contains("3pt", case=False, regex=False)
    made = shots["EVENTMSGTYPE"].eq(1)
    out = pd.DataFrame(
        {
            "season": int(season),
            "game_id": shots["GAME_ID"].astype(str),
            "event_num": pd.to_numeric(shots["EVENTNUM"], errors="coerce"),
            "player_id": pd.to_numeric(shots["PLAYER1_ID"], errors="coerce"),
            "player_name": shots["PLAYER1_NAME"].fillna("").astype(str),
            "pbp_playtype_category": classify_wnba_shots(description, is_three),
            "actual_points": np.where(made, np.where(is_three, 3.0, 2.0), 0.0),
        }
    )
    return out[out["player_id"].gt(0)].reset_index(drop=True)


def estimate_category_prices(
    pricing_shots: pd.DataFrame,
    *,
    pricing_season: int,
    feature_season: int,
    shrink_fga: float = PRICE_SHRINK_FGA,
) -> pd.DataFrame:
    """Estimate category points/FGA from one completed prior season."""
    global_mean = float(pricing_shots["actual_points"].mean())
    prices = (
        pricing_shots.groupby("pbp_playtype_category", sort=False)["actual_points"]
        .agg(["mean", "count"])
        .reindex(PBP_CATEGORIES)
    )
    prices["mean"] = prices["mean"].fillna(global_mean)
    prices["count"] = prices["count"].fillna(0)
    prices["expected_points_per_fga"] = (
        prices["count"] * prices["mean"] + shrink_fga * global_mean
    ) / (prices["count"] + shrink_fga)
    prices["league_points_per_fga"] = global_mean
    prices["category_value_vs_league"] = (
        prices["expected_points_per_fga"] - global_mean
    )
    prices["pricing_season"] = int(pricing_season)
    prices["feature_season"] = int(feature_season)
    return prices.reset_index()


def aggregate_player_difficulty(
    feature_shots: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    feature_season: int,
) -> pd.DataFrame:
    """Price the feature-season shot mix with prior-season category values."""
    scored = feature_shots.merge(
        prices[
            [
                "pbp_playtype_category",
                "expected_points_per_fga",
                "league_points_per_fga",
            ]
        ],
        on="pbp_playtype_category",
        how="left",
        validate="many_to_one",
    )
    if scored["expected_points_per_fga"].isna().any():
        raise ValueError("Every feature-season category must have a prior price")
    prior_league = float(scored["league_points_per_fga"].iloc[0])
    season_league = float(scored["expected_points_per_fga"].mean())
    rows = []
    for player_id, group in scored.groupby("player_id", sort=False):
        expected = float(group["expected_points_per_fga"].mean())
        counts = group["pbp_playtype_category"].value_counts()
        row = {
            "feature_season": int(feature_season),
            "player_id": int(player_id),
            "player_name": group["player_name"].iloc[-1],
            "pbp_fga": int(len(group)),
            "pbp_expected_points_per_fga": expected,
            "prior_league_points_per_fga": prior_league,
            "season_league_points_per_fga": season_league,
            # TS% = points / (2 * TSA), so the FGA-only equivalent in TS
            # percentage points is 50 * (reference - expected).
            "playtype_diff_prior_league": 50.0 * (prior_league - expected),
            "playtype_diff": 50.0 * (season_league - expected),
        }
        # Display only.  rTS + playtype_diff_scaled = rTS_role, so the reader
        # needs no multiplication at display time.  Never feed this to a
        # forecast or a prior; use raw ``playtype_diff`` there.
        row["playtype_diff_scaled"] = (
            ROLE_DISPLAY_COEFFICIENT * row["playtype_diff"]
        )
        row["descriptor_regime"] = (
            "modern"
            if feature_season >= MODERN_DESCRIPTION_FIRST_SEASON
            else "sparse_descriptions"
        )
        for category in PBP_CATEGORIES:
            row[f"share_{category}"] = float(counts.get(category, 0)) / len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def build(
    *,
    pbp_dir: Path,
    first_feature_season: int,
    last_feature_season: int,
    season_type: str = "RS",
) -> dict[str, pd.DataFrame]:
    """Build player-season difficulty for every requested feature season."""
    needed = range(first_feature_season - 1, last_feature_season + 1)
    shots = {season: load_season_shots(pbp_dir, season, season_type=season_type) for season in needed}
    price_frames = []
    player_frames = []
    for feature_season in range(first_feature_season, last_feature_season + 1):
        prices = estimate_category_prices(
            shots[feature_season - 1],
            pricing_season=feature_season - 1,
            feature_season=feature_season,
        )
        price_frames.append(prices)
        player_frames.append(
            aggregate_player_difficulty(
                shots[feature_season], prices, feature_season=feature_season
            )
        )
    league = (
        pd.concat(shots.values(), ignore_index=True)
        .groupby(["season", "pbp_playtype_category"], sort=True)
        .agg(fga=("actual_points", "size"), points_per_fga=("actual_points", "mean"))
        .reset_index()
    )
    league["share"] = league["fga"] / league.groupby("season")["fga"].transform("sum")
    league["descriptor_regime"] = np.where(
        league["season"] >= MODERN_DESCRIPTION_FIRST_SEASON,
        "modern",
        "sparse_descriptions",
    )
    return {
        "player_season": pd.concat(player_frames, ignore_index=True),
        "category_prices": pd.concat(price_frames, ignore_index=True),
        "league_category_mix": league,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp-dir", type=Path, default=DEFAULT_PBP_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # 2011 is buildable and is what the drift diagnostic needs, but every row
    # before MODERN_DESCRIPTION_FIRST_SEASON is labelled sparse_descriptions
    # and must not be used as a feature or a price.
    parser.add_argument("--first-feature-season", type=int, default=2011)
    parser.add_argument("--last-feature-season", type=int, default=2026)
    parser.add_argument("--season-type", choices=["RS", "PS", "ALL"], default="RS")
    args = parser.parse_args()

    artifacts = build(
        pbp_dir=args.pbp_dir,
        first_feature_season=args.first_feature_season,
        last_feature_season=args.last_feature_season,
        season_type=args.season_type,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in artifacts.items():
        frame.to_csv(args.output / f"wnba_playtype_{name}.csv", index=False)
    manifest = {
        "pbp_dir": str(args.pbp_dir),
        "season_type": args.season_type,
        "feature_seasons": [args.first_feature_season, args.last_feature_season],
        "categories": PBP_CATEGORIES,
        "self_created_pattern": SELF_CREATED_PATTERN,
        "cut_pattern": CUT_PATTERN,
        "rim_pattern": RIM_PATTERN,
        "price_shrink_fga": PRICE_SHRINK_FGA,
        "role_display_coefficient": ROLE_DISPLAY_COEFFICIENT,
        "role_display_coefficient_note": (
            "Display scaling for playtype_diff_scaled only; playtype_diff "
            "ships raw. Fitted next-season role slope -0.814 "
            "[-0.995, -0.638] on development seasons 2019-2024. "
            "Provisional pending 2026 confirmation; do not re-tune on 2026."
        ),
        "player_seasons": int(len(artifacts["player_season"])),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

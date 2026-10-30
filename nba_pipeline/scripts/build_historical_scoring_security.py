#!/usr/bin/env python3
"""Build a provenance-safe 1997-present own-scoring and ball-security board.

The scoring side preserves the public zTS contract::

    zTS = rTS + role difficulty
    zTS add -4 = 2 * TSA100 * (zTS + 4) / 100
    scoring value / 100 = -1.211295 + 0.800618 * (zTS add -4)

Observed Synergy role difficulty is used whenever it exists.  A broad,
outcome-invariant PBP shot-family proxy is calibrated on the curated Databallr
2006-2013 Synergy archive and is used only before that archive begins.

The turnover side uses the observed rZTOV ledger (the same rows the rZTOV
explorer shows) from 2014 onward and the annual box-estimated rZTOV backcast
only before 2014 or for the few modern player-seasons without observed creation
labels; ``turnover_source`` records which one each row uses.  Separate
scoring-turnover and bad-pass excess rates are converted to points with rolling
five-season through-origin fits against the existing point-valued RAPM lanes,
fitted on the same ledger that supplies the row.  Observed 2014-2017 rows
inherit the first complete observed window, 2014-2018, exactly like their
break-evens.

This is an own-scoring plus ball-security model.  It is not total offense,
playmaking value, shot quality, ORAPM, or production rZTOV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .analyze_pbp_playtype_difficulty_ots import (
        PBP_CATEGORIES,
        aggregate_player_difficulty,
        classify_pbp_shots,
        estimate_category_prices,
    )
except ImportError:  # Direct script execution.
    from analyze_pbp_playtype_difficulty_ots import (
        PBP_CATEGORIES,
        aggregate_player_difficulty,
        classify_pbp_shots,
        estimate_category_prices,
    )


PIPELINE_DIR = Path(__file__).resolve().parents[1]
NBA_ROOT = PIPELINE_DIR.parents[2]
DEFAULT_RAW_DIR = PIPELINE_DIR / "raw_data"
DEFAULT_BOX_FEATURES = (
    PIPELINE_DIR
    / "results"
    / "nba_rztov_box_backcast"
    / "all_era_box_features_1997_2026.parquet"
)
DEFAULT_RZTOV_LEDGER = (
    PIPELINE_DIR
    / "results"
    / "nba_rztov_box_backcast"
    / "nba_box_estimated_rztov_1997_2026_diagnostic.csv"
)
DEFAULT_OBSERVED_RZTOV = PIPELINE_DIR / "results" / "rztov_seasons_2014_2026.csv"
DEFAULT_MERGED = NBA_ROOT / "pipelines" / "core" / "csvs" / "merged_rs_full.csv.gz"
DEFAULT_HISTORICAL_NAMES = PIPELINE_DIR / "external" / "bref_ft_pct_1997_2000.csv"
DEFAULT_DATABALLR_ENV = NBA_ROOT / "pipelines" / "core" / ".env"
# Lane point prices regress near-unpenalized lane TOV RAPM on box excess. The
# alpha-2000/4000 Net DECOMP components shrink the dependent variable toward zero
# and pulled the old prices down by a third to a half; see RZTOV.md.
DEFAULT_COMPONENT_DIR = PIPELINE_DIR / "results" / "tov_price_targets"
PRICE_TARGET_ALPHA_TAG = "a20_40"
DEFAULT_OUTPUT = PIPELINE_DIR / "results" / "historical_scoring_security"
DEFAULT_PLAYTYPE_MIX = (
    PIPELINE_DIR / "results" / "td_playtype_mix_board" / "td_playtype_mix_board_1997_2026.csv"
)
DEFAULT_LINEUP_TOV = (
    PIPELINE_DIR / "results" / "lineup_tov_board" / "lineup_tov_board_1997_2026.csv"
)
DEFAULT_OBSERVED = DEFAULT_OUTPUT / "databallr_synergy_zts_2006_2026.csv"

FIRST_SEASON = 1997
LAST_SEASON = 2026
ANCHOR_FIRST = 2006
ANCHOR_LAST = 2013
MODERN_FIRST = 2014
OBSERVED_TOV_FIRST = 2014
OBSERVED_TOV_FIRST_WINDOW_END = 2018
WINDOW = 5
VALUE_BASELINE = -4.0
VALUE_FIT_INTERCEPT = -1.211295
VALUE_FIT_SLOPE = 0.800618
MIN_CALIBRATION_FGA = 50
QUAL_MIN_POSS = 1000.0
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SHARE_FEATURES = [f"share_{category.lower()}" for category in PBP_CATEGORIES[:-1]]
CALIBRATION_FEATURES = ["pbp_difficulty_pp", *SHARE_FEATURES]
MONOTONIC_ROLE_SIGNS = {
    "share_cut_2": -1.0,
    "share_self_created_3": 1.0,
    "share_other_3": 1.0,
    "share_self_created_2": 1.0,
    "share_other_rim_2": -1.0,
}


def refresh_databallr_observed(env_path: Path, output_path: Path) -> pd.DataFrame:
    """Read the observed 2006-26 Synergy rows from Databallr Supabase.

    This function is deliberately read-only.  Credentials are loaded without
    printing them, and the result is stored as a local generated snapshot.
    """

    from dotenv import dotenv_values
    from supabase import create_client

    values = dotenv_values(env_path)
    url = str(values.get("SUPABASE_URL") or "")
    key = str(values.get("SUPABASE_KEY") or values.get("SUPABASE_SECRET_KEY") or "")
    if not url or not key:
        raise RuntimeError(f"Databallr Supabase credentials are missing from {env_path}")
    client = create_client(url, key)
    rows: list[dict[str, object]] = []
    for start in range(0, 100_000, 1_000):
        batch = (
            client.table("player_zts")
            .select("nba_id,year,playoffs,player_name,minutes,tsa100,rts,role,zts")
            .gte("year", ANCHOR_FIRST)
            .lte("year", LAST_SEASON)
            .eq("playoffs", 0)
            .range(start, start + 999)
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < 1_000:
            break
    anchor = pd.DataFrame(rows).sort_values(["year", "nba_id"])
    if anchor.empty:
        raise RuntimeError("Databallr returned no 2006-2026 regular-season zTS rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchor.to_csv(output_path, index=False)
    print(
        f"refreshed Databallr observed roles: {len(anchor)} rows, "
        f"{anchor['role'].notna().sum()} observed roles",
        flush=True,
    )
    return anchor


def weighted_metrics(actual: Iterable[float], predicted: Iterable[float], weight: Iterable[float]) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    w = np.asarray(weight, dtype=float)
    mean = float(np.average(y, weights=w))
    sse = float(np.sum(w * np.square(y - p)))
    sst = float(np.sum(w * np.square(y - mean)))
    return {
        "rmse": float(np.sqrt(sse / np.sum(w))),
        "mae": float(np.average(np.abs(y - p), weights=w)),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "correlation": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else np.nan,
        "rows": int(len(y)),
        "weight": float(np.sum(w)),
    }


def load_pbp_shots(raw_path: Path, season: int) -> pd.DataFrame:
    """Load FGA and assign the existing outcome-invariant six-family taxonomy."""

    columns = [
        "game_id",
        "event_num",
        "event_type",
        "player1_id",
        "player1_name",
        "home_description",
        "visitor_description",
    ]
    raw = pd.read_parquet(raw_path, columns=columns).drop_duplicates().reset_index(drop=True)
    event_type = pd.to_numeric(raw["event_type"], errors="coerce")
    player_id = pd.to_numeric(raw["player1_id"], errors="coerce")
    shots = raw[event_type.isin([1, 2]) & player_id.gt(0) & player_id.lt(1_000_000_000)].copy()
    shots["player_id"] = pd.to_numeric(shots["player1_id"], errors="coerce").astype("int64")
    description = (
        shots["home_description"].fillna("").astype(str)
        + " "
        + shots["visitor_description"].fillna("").astype(str)
    )
    is_three = description.str.contains(r"\b3PT\b", case=False, regex=True)
    shots["pbp_playtype_category"] = classify_pbp_shots(shots, is_three)
    made = pd.to_numeric(shots["event_type"], errors="coerce").eq(1)
    shots["actual_points"] = made.astype(float) * (2.0 + is_three.astype(float))
    shots["season"] = int(season)
    shots["player_name"] = shots["player1_name"].fillna("").astype(str)
    return shots[
        [
            "season",
            "game_id",
            "event_num",
            "player_id",
            "player_name",
            "pbp_playtype_category",
            "actual_points",
        ]
    ]


def build_pbp_role_panel(raw_dir: Path, first_season: int, last_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price each season's shot mix with the immediately prior season.

    NBA96 is absent from the repo.  The first 1997 row therefore uses an
    explicitly flagged same-season retrospective price; every later season is
    lag priced.
    """

    first_shots = load_pbp_shots(raw_dir / "NBA97.parquet", first_season)
    previous = first_shots
    feature_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []
    for season in range(first_season, last_season + 1):
        current = first_shots if season == first_season else load_pbp_shots(
            raw_dir / f"NBA{season % 100:02d}.parquet", season
        )
        pricing = current if season == first_season else previous
        pricing_season = season if season == first_season else season - 1
        prices = estimate_category_prices(
            pricing,
            pricing_season=pricing_season,
            feature_season=season,
        )
        prices["pricing_status"] = (
            "same_season_first_year_fallback" if season == first_season else "prior_season"
        )
        feature = aggregate_player_difficulty(current, prices, feature_season=season)
        feature["pricing_status"] = prices["pricing_status"].iloc[0]
        feature_parts.append(feature)
        price_parts.append(prices)
        previous = current
        print(f"role proxy {season}: {len(feature)} players", flush=True)
    return pd.concat(feature_parts, ignore_index=True), pd.concat(price_parts, ignore_index=True)


def _fit_role_model(frame: pd.DataFrame, features: list[str], alpha: float) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(frame[features], frame["role"], ridge__sample_weight=frame["pbp_fga"])
    return model


def _monotonic_role_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Orient compositional shares so positive coefficients mean harder role."""

    return pd.DataFrame(
        {
            feature: pd.to_numeric(frame[feature], errors="coerce") * sign
            for feature, sign in MONOTONIC_ROLE_SIGNS.items()
        },
        index=frame.index,
    )


def _fit_monotonic_role_model(frame: pd.DataFrame, alpha: float) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=float(alpha),
                    positive=True,
                    max_iter=100_000,
                    tol=1e-10,
                ),
            ),
        ]
    )
    model.fit(
        _monotonic_role_features(frame),
        frame["role"],
        ridge__sample_weight=frame["pbp_fga"],
    )
    return model


def _center_by_season(frame: pd.DataFrame, column: str, weight: str = "pbp_fga") -> pd.Series:
    centered = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("feature_season", sort=False):
        mean = float(np.average(group[column], weights=group[weight]))
        centered.loc[group.index] = group[column] - mean
    return centered


def calibrate_role_proxy(
    proxy: pd.DataFrame,
    anchor: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select a LOSO calibration on 2006-13 and fit the retrospective mapper."""

    anchor_columns = ["nba_id", "year", "role", "zts", "rts", "tsa100", "minutes"]
    observed = anchor[anchor_columns].copy()
    observed["nba_id"] = pd.to_numeric(observed["nba_id"], errors="coerce").astype("Int64")
    development = proxy.merge(
        observed,
        left_on=["player_id", "feature_season"],
        right_on=["nba_id", "year"],
        how="inner",
    )
    development = development[
        development["role"].notna() & development["pbp_fga"].ge(MIN_CALIBRATION_FGA)
    ].copy()
    candidates = [
        ("raw_identity", ["pbp_difficulty_pp"], np.nan),
        *[("affine", ["pbp_difficulty_pp"], alpha) for alpha in ALPHAS],
        *[("share_ridge", CALIBRATION_FEATURES, alpha) for alpha in ALPHAS],
        *[("share_monotonic", SHARE_FEATURES, alpha) for alpha in ALPHAS],
    ]
    validation_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for family, features, alpha in candidates:
        parts = []
        for held_out in sorted(development["year"].unique()):
            test = development[development["year"].eq(held_out)].copy()
            train = development[~development["year"].eq(held_out)]
            if family == "raw_identity":
                test["prediction"] = test["pbp_difficulty_pp"]
            elif family == "share_monotonic":
                model = _fit_monotonic_role_model(train, float(alpha))
                test["prediction"] = model.predict(_monotonic_role_features(test))
            else:
                model = _fit_role_model(train, features, float(alpha))
                test["prediction"] = model.predict(test[features])
            parts.append(test)
        scored = pd.concat(parts, ignore_index=True)
        scored["prediction"] = _center_by_season(scored, "prediction")
        metrics = weighted_metrics(scored["role"], scored["prediction"], scored["pbp_fga"])
        validation_rows.append(
            {"family": family, "alpha": alpha, "features": "|".join(features), **metrics}
        )
        scored["family"] = family
        scored["alpha"] = alpha
        prediction_rows.append(
            scored[["player_id", "feature_season", "role", "prediction", "pbp_fga", "family", "alpha"]]
        )
    validation = pd.DataFrame(validation_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    validation["eligible"] = validation["family"].eq("share_monotonic")
    selected = validation[validation["eligible"]].iloc[0]
    selected_family = str(selected["family"])
    selected_alpha = float(selected["alpha"]) if pd.notna(selected["alpha"]) else np.nan
    selected_features = str(selected["features"]).split("|")
    if selected_family == "raw_identity":
        proxy["estimated_role"] = proxy["pbp_difficulty_pp"]
        standardized_coefficients: dict[str, float] = {"pbp_difficulty_pp": 1.0}
    elif selected_family == "share_monotonic":
        final_model = _fit_monotonic_role_model(development, selected_alpha)
        proxy["estimated_role"] = final_model.predict(_monotonic_role_features(proxy))
        standardized_coefficients = {
            feature: float(sign * coefficient)
            for (feature, sign), coefficient in zip(
                MONOTONIC_ROLE_SIGNS.items(), final_model.named_steps["ridge"].coef_
            )
        }
    else:
        final_model = _fit_role_model(development, selected_features, selected_alpha)
        proxy["estimated_role"] = final_model.predict(proxy[selected_features])
        standardized_coefficients = {
            feature: float(coefficient)
            for feature, coefficient in zip(
                selected_features, final_model.named_steps["ridge"].coef_
            )
        }
    proxy["estimated_role"] = _center_by_season(proxy, "estimated_role")
    receipt = {
        "selected_family": selected_family,
        "selected_alpha": selected_alpha if np.isfinite(selected_alpha) else None,
        "selected_features": selected_features,
        "monotonic_role_signs": MONOTONIC_ROLE_SIGNS,
        "standardized_role_coefficients": standardized_coefficients,
        "development_rows": int(len(development)),
        "loso_rmse_pp": float(selected["rmse"]),
        "loso_r2": float(selected["r2"]),
        "loso_correlation": float(selected["correlation"]),
    }
    return proxy, validation, receipt


def build_raw_scoring_context(
    raw_dir: Path,
    box_features: pd.DataFrame,
    seasons: Iterable[int],
) -> pd.DataFrame:
    """Build the four pre-merged seasons from official PBP box events."""

    parts = []
    for season in seasons:
        path = raw_dir / f"NBA{season % 100:02d}.parquet"
        columns = [
            "game_id",
            "event_num",
            "event_type",
            "player1_id",
            "home_description",
            "neutral_description",
            "visitor_description",
        ]
        raw = pd.read_parquet(path, columns=columns).drop_duplicates().reset_index(drop=True)
        description = (
            raw["home_description"].fillna("").astype(str)
            + " "
            + raw["neutral_description"].fillna("").astype(str)
            + " "
            + raw["visitor_description"].fillna("").astype(str)
        )
        event_type = pd.to_numeric(raw["event_type"], errors="coerce")
        player_id = pd.to_numeric(raw["player1_id"], errors="coerce")
        valid = player_id.gt(0) & player_id.lt(1_000_000_000)
        fga = valid & event_type.isin([1, 2])
        made = valid & event_type.eq(1)
        three = fga & description.str.contains(r"\b3PT\b", case=False, regex=True)
        fta = valid & event_type.eq(3)
        ftm = fta & ~description.str.contains("MISS", case=False, na=False)
        counts = pd.DataFrame(
            {
                "nba_id": player_id,
                "fga": fga.astype(int),
                "fta": fta.astype(int),
                "points": made.astype(int) * (2 + three.astype(int)) + ftm.astype(int),
            }
        )[valid].groupby("nba_id", as_index=False).sum()
        counts["tsa"] = counts["fga"] + 0.44 * counts["fta"]
        counts["ts_pct"] = counts["points"] / (2.0 * counts["tsa"])
        league_ts = float(counts["points"].sum() / (2.0 * counts["tsa"].sum()))
        counts["rts"] = 100.0 * (counts["ts_pct"] - league_ts)
        names = box_features[box_features["season"].eq(season)][
            ["player_id", "player_name", "off_poss"]
        ].rename(columns={"player_id": "nba_id"})
        counts = counts.merge(names, on="nba_id", how="inner", validate="one_to_one")
        counts["tsa100"] = 100.0 * counts["tsa"] / counts["off_poss"]
        counts["year"] = int(season)
        counts["playtype_diff"] = np.nan
        counts["scoring_context_source"] = "raw_pbp_box_approximation"
        parts.append(counts)
    return pd.concat(parts, ignore_index=True)


def resolve_historical_player_names(
    scoring: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve 1997-2000 display names by exact NBA ID and season."""

    names = lookup[["nba_id", "year", "player_name"]].copy()
    if names.duplicated(["nba_id", "year"]).any():
        raise ValueError("Historical player-name lookup has duplicate nba_id/year keys")
    names = names.rename(columns={"player_name": "historical_full_name"})
    output = scoring.merge(names, on=["nba_id", "year"], how="left", validate="one_to_one")
    historical = output["year"].between(1997, 2000)
    has_full_name = output["historical_full_name"].fillna("").astype(str).str.strip().ne("")
    output.loc[historical & has_full_name, "player_name"] = output.loc[
        historical & has_full_name, "historical_full_name"
    ]
    return output.drop(columns="historical_full_name")


def load_scoring_context(
    merged_path: Path,
    raw_dir: Path,
    box_features_path: Path,
    historical_names_path: Path,
    first_season: int,
    last_season: int,
) -> pd.DataFrame:
    usecols = [
        "nba_id",
        "year",
        "player_name",
        "OffPoss",
        "Points",
        "FGA",
        "FTA",
        "TSA100",
        "TS_pct",
        "rTSPct",
        "playtype_diff",
    ]
    merged = pd.read_csv(merged_path, usecols=usecols, low_memory=False)
    merged = merged[merged["year"].between(max(first_season, 2001), last_season)].copy()
    merged = merged.rename(
        columns={
            "OffPoss": "off_poss",
            "Points": "points",
            "FGA": "fga",
            "FTA": "fta",
            "TSA100": "tsa100",
            "TS_pct": "ts_pct",
            "rTSPct": "rts",
        }
    )
    merged["tsa"] = merged["tsa100"] * merged["off_poss"] / 100.0
    merged["scoring_context_source"] = "core_merged_exact"
    parts = [merged]
    if first_season < 2001:
        features = pd.read_parquet(box_features_path)
        parts.append(
            build_raw_scoring_context(
                raw_dir,
                features,
                range(first_season, min(last_season, 2000) + 1),
            )
        )
    scoring = pd.concat(parts, ignore_index=True, sort=False)
    if first_season < 2001:
        historical_names = pd.read_csv(historical_names_path)
        scoring = resolve_historical_player_names(scoring, historical_names)
    return scoring


def validate_premerged_scoring_bridge(
    raw_dir: Path,
    box_features_path: Path,
    scoring_context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Audit the 1997-2000 reconstruction on the first exact overlap season."""

    features = pd.read_parquet(box_features_path)
    reconstructed = build_raw_scoring_context(raw_dir, features, [2001])[
        ["nba_id", "off_poss", "tsa100", "rts", "ts_pct"]
    ]
    exact = scoring_context[scoring_context["year"].eq(2001)][
        ["nba_id", "off_poss", "tsa100", "rts", "ts_pct"]
    ]
    common = reconstructed.merge(exact, on="nba_id", suffixes=("_raw", "_exact"))
    common["weight"] = np.minimum(common["off_poss_raw"], common["off_poss_exact"])
    common["rts_error_pp"] = common["rts_raw"] - common["rts_exact"]
    common["tsa100_error"] = common["tsa100_raw"] - common["tsa100_exact"]
    qualified = common[common["weight"].ge(QUAL_MIN_POSS)].copy()
    summary: dict[str, float | int] = {
        "season": 2001,
        "rows": int(len(common)),
        "qualified_rows": int(len(qualified)),
        "qualified_rts_rmse_pp": float(
            np.sqrt(np.average(np.square(qualified["rts_error_pp"]), weights=qualified["weight"]))
        ),
        "qualified_rts_mae_pp": float(
            np.average(np.abs(qualified["rts_error_pp"]), weights=qualified["weight"])
        ),
        "qualified_tsa100_rmse": float(
            np.sqrt(np.average(np.square(qualified["tsa100_error"]), weights=qualified["weight"]))
        ),
    }
    return common, summary


def apply_role_sources(
    scoring: pd.DataFrame,
    proxy: pd.DataFrame,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    proxy_cols = [
        "player_id",
        "feature_season",
        "pbp_fga",
        "pbp_difficulty_pp",
        "estimated_role",
        "pricing_status",
    ]
    output = scoring.merge(
        proxy[proxy_cols],
        left_on=["nba_id", "year"],
        right_on=["player_id", "feature_season"],
        how="left",
        validate="one_to_one",
    )
    observed = anchor.reindex(
        columns=["nba_id", "year", "role", "zts", "rts", "tsa100"]
    ).rename(
        columns={
            "role": "observed_role",
            "zts": "observed_zts",
            "rts": "observed_rts",
            "tsa100": "observed_tsa100",
        }
    )
    output = output.merge(observed, on=["nba_id", "year"], how="left", validate="one_to_one")
    output["observed_role"] = pd.to_numeric(output["observed_role"], errors="coerce")
    output["observed_rts"] = pd.to_numeric(output["observed_rts"], errors="coerce")
    output["observed_tsa100"] = pd.to_numeric(output["observed_tsa100"], errors="coerce")
    stale_modern_context = (
        output["year"].ge(MODERN_FIRST)
        & output["playtype_diff"].isna()
        & output["observed_role"].notna()
        & output["observed_rts"].notna()
        & output["observed_tsa100"].notna()
    )
    output.loc[stale_modern_context, "rts"] = output.loc[
        stale_modern_context, "observed_rts"
    ]
    output.loc[stale_modern_context, "tsa100"] = output.loc[
        stale_modern_context, "observed_tsa100"
    ]
    if "scoring_context_source" not in output:
        output["scoring_context_source"] = "unknown"
    output.loc[stale_modern_context, "scoring_context_source"] = "databallr_zts_observed"
    backcast_mask = output["year"].lt(ANCHOR_FIRST)
    output["role_difficulty"] = np.nan
    output["role_source"] = "databallr_synergy_observed_missing"
    output.loc[backcast_mask, "role_difficulty"] = output.loc[backcast_mask, "estimated_role"]
    output.loc[backcast_mask, "role_source"] = (
        "pbp_backcast_calibrated_on_databallr_2006_2013"
    )
    output.loc[~backcast_mask, "estimated_role"] = np.nan
    observed_mask = output["year"].ge(ANCHOR_FIRST) & output["observed_role"].notna()
    output.loc[observed_mask, "role_difficulty"] = output.loc[observed_mask, "observed_role"]
    output.loc[observed_mask, "role_source"] = "databallr_synergy_observed"
    output["zts"] = output["rts"] + output["role_difficulty"]
    output["zts_add_m4"] = 2.0 * output["tsa100"] * (output["zts"] - VALUE_BASELINE) / 100.0
    output["scoring_value100"] = VALUE_FIT_INTERCEPT + VALUE_FIT_SLOPE * output["zts_add_m4"]
    return output


def validate_observed_role_coverage(scoring: pd.DataFrame) -> pd.DataFrame:
    """Reject an observed-era season whose role snapshot failed completely."""

    observed_era = scoring[scoring["year"].ge(ANCHOR_FIRST)].copy()
    coverage = (
        observed_era.assign(
            observed=observed_era["role_source"].eq("databallr_synergy_observed")
        )
        .groupby("year", as_index=False)
        .agg(rows=("nba_id", "size"), observed_rows=("observed", "sum"))
    )
    missing_seasons = coverage.loc[coverage["observed_rows"].eq(0), "year"].astype(int).tolist()
    if missing_seasons:
        raise ValueError(
            "Databallr observed role coverage is zero for season(s): "
            + ", ".join(map(str, missing_seasons))
            + ". Refresh the observed-role snapshot; historical PBP fallback is forbidden "
            "from 2006 onward."
        )
    coverage["coverage"] = coverage["observed_rows"] / coverage["rows"]
    return coverage


def _window_tag(lo: int, hi: int) -> str:
    return f"{lo % 100:02d}_{hi % 100:02d}"


def _fit_origin_price(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, dict[str, float]]:
    denominator = float(np.sum(w * x * x))
    if denominator <= 0:
        raise ValueError("Turnover point-price fit has no weighted variation")
    price = float(np.sum(w * x * y) / denominator)
    pred = price * x
    metrics = weighted_metrics(y, pred, w)
    return price, metrics


def fit_turnover_point_prices(
    ledger: pd.DataFrame,
    component_dir: Path,
    first_end: int,
    last_end: int,
    bootstrap_draws: int,
    seed: int,
    alpha_tag: str = PRICE_TARGET_ALPHA_TAG,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for hi in range(first_end, last_end + 1):
        lo = hi - WINDOW + 1
        window = ledger[ledger["year"].between(lo, hi)].copy()
        window["sc_num"] = window["sc_excess"] * window["OffPoss"]
        window["bp_num"] = window["bp_excess"] * window["OffPoss"]
        roll = window.groupby("nba_id", as_index=False).agg(
            poss=("OffPoss", "sum"), sc_num=("sc_num", "sum"), bp_num=("bp_num", "sum")
        )
        roll["sc_saved100"] = -roll["sc_num"] / roll["poss"]
        roll["bp_saved100"] = -roll["bp_num"] / roll["poss"]
        tag = _window_tag(lo, hi)
        record: dict[str, object] = {"window_lo": lo, "window_hi": hi}
        for lane, source in (("scoring", "sc_saved100"), ("badpass", "bp_saved100")):
            path = component_dir / f"alt_{lane}_tov_net_value_{tag}_all_rb_se_{alpha_tag}_results.csv"
            target = pd.read_csv(path, usecols=["player_id", "off", "off_poss"]).rename(
                columns={"player_id": "nba_id", "off": "target", "off_poss": "target_poss"}
            )
            common = roll.merge(target, on="nba_id", how="inner").dropna(subset=[source, "target"])
            x = common[source].to_numpy(float)
            y = common["target"].to_numpy(float)
            w = np.minimum(common["poss"], common["target_poss"]).to_numpy(float)
            price, metrics = _fit_origin_price(x, y, w)
            draws = []
            for _ in range(bootstrap_draws):
                index = rng.integers(0, len(common), len(common))
                try:
                    draw, _ = _fit_origin_price(x[index], y[index], w[index])
                    draws.append(draw)
                except ValueError:
                    continue
            record.update(
                {
                    f"{lane}_point_price": price,
                    f"{lane}_point_price_p05": float(np.quantile(draws, 0.05)) if draws else np.nan,
                    f"{lane}_point_price_p95": float(np.quantile(draws, 0.95)) if draws else np.nan,
                    f"{lane}_r2": metrics["r2"],
                    f"{lane}_correlation": metrics["correlation"],
                    f"{lane}_rows": metrics["rows"],
                }
            )
        rows.append(record)
        print(
            f"turnover value {lo}-{hi}: scoring={record['scoring_point_price']:.4f}, "
            f"badpass={record['badpass_point_price']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows)


TURNOVER_COLUMNS = [
    "nba_id",
    "year",
    "OffPoss",
    "sc_excess",
    "bp_excess",
    "rztov",
    "tov_saved_100",
    "qualified",
    "be_source",
    "turnover_source",
]


def load_observed_rztov(path: Path) -> pd.DataFrame:
    """Observed rZTOV rows on the box-ledger column contract.

    Break-evens come from the trailing five-season window ending in the row's
    season; 2014-2017 inherit the first complete observed window, 2014-2018.
    """

    observed = pd.read_csv(path).rename(columns={"off_poss": "OffPoss"})
    if observed.duplicated(["nba_id", "year"]).any():
        raise ValueError("Observed rZTOV ledger has duplicate player-season keys")
    observed["be_source"] = observed["year"].clip(lower=OBSERVED_TOV_FIRST_WINDOW_END).astype(int)
    observed["turnover_source"] = "observed"
    return observed


def combine_turnover_ledgers(box: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Observed rows win from 2014; box rows fill pre-2014 and missing modern rows."""

    box = box.copy()
    box["turnover_source"] = "box_estimated"
    keys = observed.set_index(["nba_id", "year"]).index
    box_keys = box.set_index(["nba_id", "year"]).index
    keep_box = (box["year"] < OBSERVED_TOV_FIRST) | ~box_keys.isin(keys)
    combined = pd.concat(
        [box.loc[keep_box, TURNOVER_COLUMNS], observed[TURNOVER_COLUMNS]], ignore_index=True
    )
    if combined.duplicated(["nba_id", "year"]).any():
        raise ValueError("Combined turnover ledger has duplicate player-season keys")
    return combined.sort_values(["year", "nba_id"]).reset_index(drop=True)


def build_board(scoring: pd.DataFrame, ledger: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    board = ledger[TURNOVER_COLUMNS].merge(scoring, on=["nba_id", "year"], how="left")
    price_map = prices.set_index(["ledger_source", "window_hi"])
    keys = pd.MultiIndex.from_arrays([board["turnover_source"], board["be_source"]])
    board["scoring_tov_point_price"] = price_map["scoring_point_price"].reindex(keys).to_numpy()
    board["badpass_tov_point_price"] = price_map["badpass_point_price"].reindex(keys).to_numpy()
    if board[["scoring_tov_point_price", "badpass_tov_point_price"]].isna().any().any():
        raise ValueError("Some turnover rows have no matching lane point price")
    board["scoring_tov_value100"] = -board["sc_excess"] * board["scoring_tov_point_price"]
    board["badpass_tov_value100"] = -board["bp_excess"] * board["badpass_tov_point_price"]
    board["scoring_impact_no_badpass100"] = (
        board["scoring_value100"] + board["scoring_tov_value100"]
    )
    board["turnover_value100"] = board["scoring_tov_value100"] + board["badpass_tov_value100"]
    board["scoring_security_value100"] = board["scoring_value100"] + board["turnover_value100"]
    board["qualified_scoring_security"] = (
        board["OffPoss"].ge(QUAL_MIN_POSS)
        & board["zts"].notna()
        & board["turnover_value100"].notna()
    ).astype(int)
    return board


MIX_COLUMNS = ["playtype_mix_rapm"]


def attach_playtype_mix(board: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Join the one-column playtype-mix RAPM without touching any identity.

    ``playtype_mix_rapm`` is the time-decay lineup shot-diet RAPM priced in
    expected points per 100 offensive possessions (the mix board's
    ``mix_value100``): positive means the lineup's shot diet is easier while the
    player is on offense, negative means harder.  It is descriptive only and
    never enters scoring value, turnover value, or scoring security.
    """
    mix = pd.read_csv(path, usecols=["nba_id", "year", "mix_value100"])
    if mix.duplicated(["nba_id", "year"]).any():
        raise ValueError("Playtype-mix board has duplicate player-season keys")
    mix = mix.rename(columns={"mix_value100": "playtype_mix_rapm"})
    return board.merge(mix, on=["nba_id", "year"], how="left", validate="one_to_one")


LINEUP_TOV_LANES = ("scoring", "badpass")
LINEUP_TOV_PARTS = ("own", "teammate", "team")
LINEUP_TOV_COLUMNS = [
    *[
        f"lineup_{lane}_tov_{part}_value100"
        for lane in LINEUP_TOV_LANES
        for part in (*LINEUP_TOV_PARTS, "total")
    ],
    "lineup_scoring_tov_raw_value100",
    "lineup_badpass_tov_raw_value100",
    "lineup_tov_value100",
    "lineup_tov_point_price",
    "lineup_team_tov_coverage",
]


def attach_lineup_tov(board: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Join the time-decay lineup turnover view without touching any identity.

    The lineup board (``build_lineup_tov_board.py``) solves on-court lineup
    turnovers over a two-season TD550 window and splits each lane exactly into
    the player's own turnovers, teammates' turnovers, and unassigned team
    turnovers.  Here every piece is priced at the window's net-continuation
    value of a kept possession so it reads in the same pts/100 (+ good) units
    as ``scoring_tov_value100`` / ``badpass_tov_value100``.  ``*_raw_value100``
    is the unadjusted on-court rate against league.  These columns are a
    transparency comparison only and never enter any board identity.
    """
    source = pd.read_csv(path)
    if source.duplicated(["nba_id", "year"]).any():
        raise ValueError("Lineup turnover board has duplicate player-season keys")
    price = source["tov_point_price"]
    out = source[["nba_id", "year"]].copy()
    names = {"own": "own", "teammate": "teammate", "team": "unassigned", "total": "total"}
    for lane in LINEUP_TOV_LANES:
        for part, source_part in names.items():
            out[f"lineup_{lane}_tov_{part}_value100"] = source[f"{lane}_{source_part}_saved100"] * price
        out[f"lineup_{lane}_tov_raw_value100"] = source[f"{lane}_raw_total_saved100"] * price
    out["lineup_tov_value100"] = source["lineup_tov_value100"]
    out["lineup_tov_point_price"] = price
    out["lineup_team_tov_coverage"] = source["team_tov_coverage"]
    return board.merge(out[["nba_id", "year", *LINEUP_TOV_COLUMNS]], on=["nba_id", "year"], how="left", validate="one_to_one")


def lineup_tov_gaps(board: pd.DataFrame) -> dict[str, float]:
    """Exact split closure of the joined lineup view (rows with a lineup solve)."""
    rows = board[board["lineup_tov_value100"].notna()]
    gaps = {}
    for lane in LINEUP_TOV_LANES:
        parts = rows[[f"lineup_{lane}_tov_{part}_value100" for part in LINEUP_TOV_PARTS]].sum(axis=1)
        gaps[f"lineup_{lane}_split_gap"] = float((rows[f"lineup_{lane}_tov_total_value100"] - parts).abs().max())
    gaps["lineup_lane_sum_gap"] = float(
        (
            rows["lineup_tov_value100"]
            - rows["lineup_scoring_tov_total_value100"]
            - rows["lineup_badpass_tov_total_value100"]
        ).abs().max()
    )
    return {key: (0.0 if pd.isna(value) else value) for key, value in gaps.items()}


def lineup_pass_through(board: pd.DataFrame) -> pd.DataFrame:
    """Per-season league slope of each lineup turnover value on the box value.

    Qualified rows, OffPoss-weighted least squares with an intercept.  A slope
    of 1 means one point of box turnover value shows up one-for-one in the
    lineup's on-court turnovers; the adjusted slopes are ridge-shrunk, the raw
    slopes are not adjusted for teammates or opponents.
    """
    rows = []
    qualified = board[board["qualified_scoring_security"].eq(1) & board["lineup_tov_value100"].notna()]
    pairs = {
        "scoring_own": ("scoring_tov_value100", "lineup_scoring_tov_own_value100"),
        "scoring_total": ("scoring_tov_value100", "lineup_scoring_tov_total_value100"),
        "scoring_raw": ("scoring_tov_value100", "lineup_scoring_tov_raw_value100"),
        "badpass_own": ("badpass_tov_value100", "lineup_badpass_tov_own_value100"),
        "badpass_total": ("badpass_tov_value100", "lineup_badpass_tov_total_value100"),
        "badpass_raw": ("badpass_tov_value100", "lineup_badpass_tov_raw_value100"),
        "turnover_total": ("turnover_value100", "lineup_tov_value100"),
    }
    for year, group in qualified.groupby("year"):
        weight = group["OffPoss"].to_numpy(float)
        record: dict[str, object] = {"year": int(year), "players": int(len(group))}
        for key, (box_col, lineup_col) in pairs.items():
            x = group[box_col].to_numpy(float)
            y = group[lineup_col].to_numpy(float)
            xm = np.average(x, weights=weight)
            ym = np.average(y, weights=weight)
            var = np.average((x - xm) ** 2, weights=weight)
            cov = np.average((x - xm) * (y - ym), weights=weight)
            vary = np.average((y - ym) ** 2, weights=weight)
            record[f"{key}_slope"] = cov / var if var > 0 else np.nan
            record[f"{key}_corr"] = cov / np.sqrt(var * vary) if var > 0 and vary > 0 else np.nan
        record["team_tov_coverage"] = group["lineup_team_tov_coverage"].mode().iat[0]
        rows.append(record)
    return pd.DataFrame(rows)


def identity_gaps(board: pd.DataFrame) -> dict[str, float]:
    """Largest absolute gap for every additive identity on the board."""
    checks = {
        "zts_gap": (board["zts"] - board["rts"] - board["role_difficulty"]).abs().max(),
        "turnover_lane_gap": (
            board["turnover_value100"]
            - board["scoring_tov_value100"]
            - board["badpass_tov_value100"]
        ).abs().max(),
        "scoring_impact_no_badpass_gap": (
            board["scoring_impact_no_badpass100"]
            - board["scoring_value100"]
            - board["scoring_tov_value100"]
        ).abs().max(),
        "composite_gap": (
            board["scoring_security_value100"]
            - board["scoring_value100"]
            - board["turnover_value100"]
        ).abs().max(),
    }
    return {key: (0.0 if pd.isna(value) else float(value)) for key, value in checks.items()}


def validate_identities(board: pd.DataFrame) -> dict[str, float]:
    checks = identity_gaps(board)
    if max(checks.values()) > 1e-9:
        raise AssertionError(f"Historical scoring-security identity failure: {checks}")
    return checks


def observed_overlap(proxy: pd.DataFrame, observed: pd.DataFrame, label: str) -> dict[str, object]:
    common = proxy.merge(
        observed[["nba_id", "year", "role"]],
        left_on=["player_id", "feature_season"],
        right_on=["nba_id", "year"],
        how="inner",
    )
    common = common[common["role"].notna() & common["pbp_fga"].ge(MIN_CALIBRATION_FGA)].copy()
    metrics = weighted_metrics(common["role"], common["estimated_role"], common["pbp_fga"])
    return {"sample": label, **metrics}


def build(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_anchor_from_databallr:
        observed_roles = refresh_databallr_observed(args.databallr_env, args.synergy_anchor)
    else:
        if not args.synergy_anchor.exists():
            raise FileNotFoundError(
                f"Observed Synergy snapshot not found at {args.synergy_anchor}. Re-run with "
                "--refresh-anchor-from-databallr or pass --synergy-anchor."
            )
        observed_roles = pd.read_csv(args.synergy_anchor)
    required_anchor = {"nba_id", "year", "role", "zts", "rts", "tsa100", "minutes"}
    missing_anchor = required_anchor - set(observed_roles.columns)
    if missing_anchor:
        raise ValueError(f"Observed Synergy snapshot missing columns: {sorted(missing_anchor)}")
    observed_roles = observed_roles[
        observed_roles["year"].between(ANCHOR_FIRST, args.last_season)
    ].copy()
    if observed_roles.duplicated(["nba_id", "year"]).any():
        raise ValueError("Observed Synergy snapshot has duplicate player-season keys")
    calibration_anchor = observed_roles[
        observed_roles["year"].between(ANCHOR_FIRST, ANCHOR_LAST)
    ].copy()

    role_cache = args.output_dir / "pbp_role_proxy_1997_2026.parquet"
    price_cache = args.output_dir / "pbp_role_category_prices_1997_2026.csv"
    if args.reuse_role_cache:
        proxy = pd.read_parquet(role_cache)
        category_prices = pd.read_csv(price_cache)
    else:
        proxy, category_prices = build_pbp_role_panel(args.raw_dir, args.first_season, args.last_season)
        proxy.to_parquet(role_cache, index=False)
        category_prices.to_csv(price_cache, index=False)

    proxy, role_validation, role_receipt = calibrate_role_proxy(proxy, calibration_anchor)
    proxy.to_parquet(role_cache, index=False)
    role_validation.to_csv(args.output_dir / "role_calibration_loso.csv", index=False)

    scoring_context = load_scoring_context(
        args.merged,
        args.raw_dir,
        args.box_features,
        args.historical_names,
        args.first_season,
        args.last_season,
    )
    bridge_rows, bridge_receipt = validate_premerged_scoring_bridge(
        args.raw_dir, args.box_features, scoring_context
    )
    bridge_rows.round(8).to_csv(
        args.output_dir / "premerged_scoring_bridge_validation_2001.csv", index=False
    )
    scoring = apply_role_sources(scoring_context, proxy, observed_roles)
    observed_coverage = validate_observed_role_coverage(scoring)

    modern_observed = observed_roles[observed_roles["year"].ge(MODERN_FIRST)]
    overlap = pd.DataFrame(
        [
            observed_overlap(proxy, calibration_anchor, "anchor_2006_2013_in_sample_refit"),
            observed_overlap(proxy, modern_observed, "modern_2014_2026_out_of_anchor"),
        ]
    )
    overlap.to_csv(args.output_dir / "role_overlap_summary.csv", index=False)

    box_ledger = pd.read_csv(args.rztov_ledger)
    observed_ledger = load_observed_rztov(args.observed_rztov)
    ledger = combine_turnover_ledgers(box_ledger, observed_ledger)
    box_prices = fit_turnover_point_prices(
        box_ledger,
        args.component_dir,
        first_end=args.first_season + WINDOW - 1,
        last_end=args.last_season,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    box_prices.insert(0, "ledger_source", "box_estimated")
    observed_prices = fit_turnover_point_prices(
        observed_ledger,
        args.component_dir,
        first_end=OBSERVED_TOV_FIRST_WINDOW_END,
        last_end=args.last_season,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    observed_prices.insert(0, "ledger_source", "observed")
    prices = pd.concat([box_prices, observed_prices], ignore_index=True)
    prices.to_csv(args.output_dir / "turnover_point_price_path_1997_2026.csv", index=False)
    board = build_board(scoring, ledger, prices)
    board = attach_playtype_mix(board, args.playtype_mix)
    board = attach_lineup_tov(board, args.lineup_tov)
    identities = validate_identities(board)
    lineup_gaps = lineup_tov_gaps(board)
    # The lineup CSV is rounded to 6 dp before pricing, so closure here is to CSV
    # precision; the exact 1e-9 check runs inside build_lineup_tov_board.py.
    if max(lineup_gaps.values()) > 1e-5:
        raise AssertionError(f"Lineup turnover split failure: {lineup_gaps}")
    pass_through = lineup_pass_through(board)
    pass_through.round(6).to_csv(args.output_dir / "lineup_tov_pass_through_1997_2026.csv", index=False)

    ordered = [
        "nba_id",
        "player_name",
        "year",
        "OffPoss",
        "turnover_source",
        "scoring_context_source",
        "role_source",
        "pricing_status",
        "tsa100",
        "rts",
        "pbp_difficulty_pp",
        "estimated_role",
        "role_difficulty",
        "zts",
        "zts_add_m4",
        "scoring_value100",
        "rztov",
        "sc_excess",
        "bp_excess",
        "scoring_tov_point_price",
        "badpass_tov_point_price",
        "scoring_tov_value100",
        "scoring_impact_no_badpass100",
        "badpass_tov_value100",
        "turnover_value100",
        "scoring_security_value100",
        "qualified_scoring_security",
        *MIX_COLUMNS,
        *LINEUP_TOV_COLUMNS,
    ]
    board_path = args.output_dir / "historical_scoring_security_1997_2026.csv"
    board[ordered].sort_values(["year", "scoring_security_value100"], ascending=[True, False]).round(6).to_csv(
        board_path, index=False
    )
    historical_path = args.output_dir / "historical_scoring_security_1997_2013.csv"
    board[board["year"].le(2013)][ordered].sort_values(
        ["year", "scoring_security_value100"], ascending=[True, False]
    ).round(6).to_csv(historical_path, index=False)

    source_counts = (
        board.groupby(["year", "role_source"], dropna=False).size().rename("rows").reset_index()
    )
    source_counts.to_csv(args.output_dir / "role_source_coverage.csv", index=False)
    manifest: dict[str, object] = {
        "status": "research_historical_own_scoring_plus_ball_security_not_total_offense",
        "coverage": {
            "first_season": int(board["year"].min()),
            "last_season": int(board["year"].max()),
            "rows": int(len(board)),
            "qualified_rows": int(board["qualified_scoring_security"].sum()),
        },
        "role_contract": {
            "backcast_seasons": [1997, 2005],
            "databallr_synergy_observed_seasons": [2006, int(args.last_season)],
            "pbp_fallback_forbidden_from": 2006,
            "databallr_observed_rows": int(len(observed_roles)),
            "databallr_observed_role_rows": int(observed_roles["role"].notna().sum()),
            "databallr_observed_coverage_by_season": observed_coverage.to_dict("records"),
            "calibration_anchor_seasons": [2006, 2013],
            "calibration_anchor_rows": int(len(calibration_anchor)),
            "calibration_anchor_observed_roles": int(calibration_anchor["role"].notna().sum()),
            "first_season_pricing_fallback": "1997 same-season retrospective because NBA96 is absent",
            **role_receipt,
            "overlap": overlap.to_dict("records"),
        },
        "scoring_contract": {
            "identity": "zTS = rTS + role difficulty",
            "value": "SV/100 = -1.211295 + 0.800618 * (2*TSA100*(zTS+4)/100)",
            "pre_2001_context": "raw PBP box approximation using FGA + 0.44*FTA",
            "pre_2001_player_names": (
                "exact nba_id/year resolution from the Basketball-Reference FT bridge; "
                "no fuzzy surname matching"
            ),
            "2001_present_context": (
                "core merged exact scoring context; when a modern core role is absent, "
                "TSA100, rTS, and role come together from the observed Databallr snapshot"
            ),
            "premerged_bridge_validation": bridge_receipt,
        },
        "turnover_contract": {
            "source": (
                "observed rZTOV scoring and bad-pass excess lanes from 2014 (identical to the "
                "rZTOV explorer rows); annual box-estimated backcast before 2014 and for modern "
                "player-seasons without observed creation labels"
            ),
            "turnover_source_counts": board["turnover_source"].value_counts().to_dict(),
            "point_prices": (
                "rolling five-season through-origin fits to point-valued lane RAPMs, fit on the "
                "ledger that supplies each row; observed 2014-2017 rows use the 2014-2018 window"
            ),
            "bootstrap_draws_per_window": int(args.bootstrap_draws),
            "identities": [
                "scoring impact without bad-pass = scoring value + scoring-TOV value",
                "TOV value = scoring-TOV value + bad-pass-TOV value",
                "scoring security = scoring impact without bad-pass + bad-pass-TOV value",
            ],
        },
        "playtype_mix_contract": {
            "status": "descriptive column; never inside scoring value, turnover value, or scoring security",
            "source": str(args.playtype_mix),
            "column": "playtype_mix_rapm",
            "model": (
                "time-decay playtype-mix RAPM (build_td_playtype_mix_board.py): one row per FGA, "
                "lag-priced six-family PBP playtype difficulty, two-season window (feature season "
                "plus prior season), 550-day half-life, alphas 3000/5000/10000"
            ),
            "units": "expected points per 100 offensive possessions at the season's league FGA rate",
            "sign": "positive = easier lineup shot diet with the player on offense; negative = harder",
            "coverage_seasons": [1997, int(args.last_season)],
            "first_season_caveat": "1997 is priced on itself and solved on one season (no NBA96 raw file)",
            "rows_with_mix": int(board["playtype_mix_rapm"].notna().sum()),
            "admission": (
                "descriptive only; not production zTS, not a prior, not a VPM input. The earlier "
                "TD700 research spec failed the creation-controlled future-oTS increment "
                "(-0.0040 [-0.0233,+0.0136]); this two-season spec has not been gated"
            ),
        },
        "lineup_tov_contract": {
            "status": (
                "transparency comparison; never inside scoring value, turnover value, or scoring "
                "security, and never the default rank"
            ),
            "source": str(args.lineup_tov),
            "columns": LINEUP_TOV_COLUMNS,
            "model": (
                "build_lineup_tov_board.py: two-season window, 550-day half-life, joint O/D ridge "
                "(3000/5000) on season-centred scoring and bad-pass turnover indicators split into "
                "player-attributed and unassigned team turnovers; a CV-selected focal own-turnover "
                "contrast splits the attributed parent into own = (parent + 4c)/5 and teammates = "
                "4(parent - c)/5"
            ),
            "pricing": (
                "every piece times the window's TD-weighted mean non-turnover first-chance net "
                "value (one averted turnover = one kept possession)"
            ),
            "units": "pts/100 offensive possessions, positive = fewer turnovers with the player on offense",
            "own_share_caveat": (
                "lineup own is measured against a flat one-fifth share of the lineup's attributed "
                "turnovers, while box excess is measured against a workload allowance; creators read "
                "worse on lineup own than on box excess by construction"
            ),
            "team_tov_caveat": (
                "1997-2013 source possessions omit team turnovers (coverage 'absent'); 2014 is "
                "'partial'; 2015+ 'full'. The team column is not comparable across that seam"
            ),
            "rows_with_lineup": int(board["lineup_tov_value100"].notna().sum()),
            "split_gaps": lineup_gaps,
            "pass_through": str(args.output_dir / "lineup_tov_pass_through_1997_2026.csv"),
        },
        "identities": identities,
        "outputs": {
            "board": str(board_path),
            "historical_board": str(historical_path),
            "role_validation": str(args.output_dir / "role_calibration_loso.csv"),
            "role_overlap": str(args.output_dir / "role_overlap_summary.csv"),
            "turnover_price_path": str(args.output_dir / "turnover_point_price_path_1997_2026.csv"),
            "databallr_observed_snapshot": str(args.synergy_anchor),
            "historical_player_name_lookup": str(args.historical_names),
        },
        "limitations": [
            "1997-2005 role difficulty is a retrospective calibrated PBP backcast, not observed Synergy.",
            "Observed-era rows without a Databallr role remain missing and are never PBP-backfilled.",
            "1997-2000 rTS uses a raw-PBP FGA + 0.44*FTA reconstruction because the merged scoring file starts in 2001.",
            "Pre-2014 turnover values inherit the box-estimated historical rZTOV workload model.",
            "The composite covers own scoring and ball security only; it is not total offensive impact or playmaking value.",
        ],
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--last-season", type=int, default=LAST_SEASON)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--box-features", type=Path, default=DEFAULT_BOX_FEATURES)
    parser.add_argument("--rztov-ledger", type=Path, default=DEFAULT_RZTOV_LEDGER)
    parser.add_argument("--observed-rztov", type=Path, default=DEFAULT_OBSERVED_RZTOV)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--historical-names", type=Path, default=DEFAULT_HISTORICAL_NAMES)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--synergy-anchor", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--databallr-env", type=Path, default=DEFAULT_DATABALLR_ENV)
    parser.add_argument("--refresh-anchor-from-databallr", action="store_true")
    parser.add_argument("--playtype-mix", type=Path, default=DEFAULT_PLAYTYPE_MIX)
    parser.add_argument("--lineup-tov", type=Path, default=DEFAULT_LINEUP_TOV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-role-cache", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

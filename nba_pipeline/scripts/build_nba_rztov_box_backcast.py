#!/usr/bin/env python3
"""Estimate the three NBA rZTOV workload inputs from all-era PBP box features.

Observed rZTOV role labels begin in 2013-14.  This research builder learns two
player-season mappings on those labels:

* potentially assisted attempt share; and
* potential assists per 100 offensive possessions.

It then applies the mappings to a consistent raw-PBP feature contract available
from 1996-97 onward.  The shot split is constrained exactly:

    estimated self-created TSA + estimated assisted TSA = box TSA

Pre-2013-14 rows are retrospective backcasts, not observed tracking values and
not production rZTOV.  This script deliberately does not inherit modern
turnover break-even prices into the pre-label era.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PIPELINE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PIPELINE_DIR / "raw_data"
PROCESSED_DIR = PIPELINE_DIR / "processed"
DEFAULT_OBSERVED = PIPELINE_DIR / "results" / "rztov_seasons_2014_2026.csv"
DEFAULT_OUTPUT = PIPELINE_DIR / "results" / "nba_rztov_box_backcast"

FIRST_SEASON = 1997
LAST_SEASON = 2026
MIN_TRAIN_SEASONS = 3
MIN_TRAIN_POSS = 500.0
ASSIST_SHARE_PRIOR_MAKES = 40.0
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
WINDOW = 5
FIT_MIN_POSS = 2000.0
QUAL_MIN_POSS = 1000.0
QUAL_MIN_INVOLVE = 10.0
PRICE_ALPHA = 1500

REDUCED_SHARE_FEATURES = [
    "tsa100",
    "three_freq",
    "fta100",
    "assists100",
    "assist_points100",
    "turnovers100",
    "badpass_tov100",
    "usage_proxy",
]
OFFICIAL_SHARE_FEATURES = [
    *REDUCED_SHARE_FEATURES,
    "official_assisted_make_share_smoothed",
]
RICH_SHARE_FEATURES = [
    *OFFICIAL_SHARE_FEATURES,
    "rim_freq",
    "mid_freq",
    "rim_assists100",
]

REDUCED_POTAST_FEATURES = [
    "assists100",
    "assist_points100",
    "badpass_tov100",
    "turnovers100",
    "tsa100",
    "usage_proxy",
]
RICH_POTAST_FEATURES = [
    *REDUCED_POTAST_FEATURES,
    "rim_assists100",
    "three_freq",
    "rim_freq",
    "fta100",
    "official_assisted_make_share_smoothed",
]

FEATURE_FAMILIES = {
    "assist_opp_share": {
        "reduced_no_official": REDUCED_SHARE_FEATURES,
        "all_era_official": OFFICIAL_SHARE_FEATURES,
        "all_era_rich": RICH_SHARE_FEATURES,
        "official_only": ["official_assisted_make_share_smoothed"],
    },
    "potential_assists100": {
        "reduced": REDUCED_POTAST_FEATURES,
        "all_era_rich": RICH_POTAST_FEATURES,
    },
}

ASSIST_RE = re.compile(r"\(([^()]*)\s+\d+\s+AST\)", flags=re.IGNORECASE)
RIM_RE = re.compile(r"\b(?:layup|dunk|tip|alley[ -]?oop)\b", flags=re.IGNORECASE)
THREE_RE = re.compile(r"\b3PT\b", flags=re.IGNORECASE)
BADPASS_RE = re.compile(r"bad pass", flags=re.IGNORECASE)
NO_TURNOVER_RE = re.compile(r"no turnover", flags=re.IGNORECASE)
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def season_suffix(season: int) -> str:
    return f"{int(season) % 100:02d}"


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return " ".join(token for token in text.split() if token not in NAME_SUFFIXES)


def _descriptions(raw: pd.DataFrame) -> pd.Series:
    result = pd.Series("", index=raw.index, dtype="string")
    for column in ("home_description", "neutral_description", "visitor_description"):
        if column in raw:
            result = result.str.cat(raw[column].fillna("").astype(str), sep=" ")
    return result.str.strip()


def _modal_player_names(raw: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    names = raw[["player1_id", "player1_name"]].dropna().copy()
    names["player1_id"] = pd.to_numeric(names["player1_id"], errors="coerce")
    names = names[names["player1_id"].gt(0)].copy()
    names["player1_id"] = names["player1_id"].astype("int64")
    names["normalized"] = names["player1_name"].map(normalize_name)
    display: dict[int, str] = {}
    normalized: dict[int, str] = {}
    for player_id, group in names.groupby("player1_id", sort=False):
        display[int(player_id)] = str(group["player1_name"].value_counts().index[0])
        normalized[int(player_id)] = str(group["normalized"].value_counts().index[0])
    return display, normalized


def _same_name(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left.endswith(f" {right}") or right.endswith(f" {left}")


def resolve_assister_ids(raw: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Resolve made-shot assist tags to player IDs.

    Older NBA PBP carries ``player2_id`` on assisted makes.  Modern files in
    this repo do not, so the fallback parses the official ``(... N AST)`` tag
    and matches its name to the shooter's five-man lineup.  Ambiguous tags are
    left unresolved rather than assigned arbitrarily.
    """

    descriptions = _descriptions(raw)
    event_type = pd.to_numeric(raw["event_type"], errors="coerce")
    assisted = event_type.eq(1) & descriptions.str.contains(
        r"\bAST\)", case=False, regex=True, na=False
    )
    result = pd.Series(pd.NA, index=raw.index, dtype="Int64")

    player2 = pd.to_numeric(
        raw.get("player2_id", pd.Series(np.nan, index=raw.index)), errors="coerce"
    )
    direct = assisted & player2.gt(0)
    result.loc[direct] = player2.loc[direct].astype("int64")

    display_names, normalized_names = _modal_player_names(raw)
    del display_names
    home_columns = [column for column in raw if re.fullmatch(r"home_player[1-5]", column)]
    away_columns = [column for column in raw if re.fullmatch(r"away_player[1-5]", column)]
    unresolved = raw.index[assisted & result.isna()]
    parsed = 0
    ambiguous = 0
    unmatched = 0
    for index in unresolved:
        match = ASSIST_RE.search(str(descriptions.at[index]))
        if match is None:
            unmatched += 1
            continue
        token = normalize_name(match.group(1))
        shooter = pd.to_numeric(raw.at[index, "player1_id"], errors="coerce")
        if pd.isna(shooter):
            unmatched += 1
            continue
        shooter_id = int(shooter)

        def ids(columns: list[str]) -> list[int]:
            values = pd.to_numeric(raw.loc[index, columns], errors="coerce")
            return [int(value) for value in values.dropna() if int(value) > 0]

        home_ids = ids(home_columns)
        away_ids = ids(away_columns)
        candidates = (
            home_ids
            if shooter_id in home_ids
            else away_ids
            if shooter_id in away_ids
            else home_ids + away_ids
        )
        matches = {
            player_id
            for player_id in candidates
            if player_id != shooter_id
            and _same_name(normalized_names.get(player_id, ""), token)
        }
        if len(matches) == 1:
            result.at[index] = matches.pop()
            parsed += 1
        elif len(matches) > 1:
            ambiguous += 1
        else:
            unmatched += 1

    audit = {
        "assist_tags": int(assisted.sum()),
        "direct_player2_ids": int(direct.sum()),
        "parsed_lineup_ids": int(parsed),
        "ambiguous_tags": int(ambiguous),
        "unmatched_tags": int(unmatched),
        "resolved_tags": int(result.notna().sum()),
    }
    return result, audit


def _offensive_possessions(possessions: pd.DataFrame) -> pd.Series:
    columns = [column for column in possessions if re.fullmatch(r"O[1-5]", column)]
    if len(columns) != 5:
        raise ValueError(f"Expected O1-O5 in possession frame; found {columns}")
    values = pd.concat([pd.to_numeric(possessions[column], errors="coerce") for column in columns])
    values = values.dropna().astype("int64")
    values = values[(values > 0) & (values < 1_000_000_000)]
    return values.value_counts().astype(float)


def build_season_features_from_frames(
    raw: pd.DataFrame,
    possessions: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one season of stable box/PBP features from in-memory frames."""

    raw = raw.drop_duplicates().reset_index(drop=True).copy()
    raw["description"] = _descriptions(raw)
    raw["event_type_num"] = pd.to_numeric(raw["event_type"], errors="coerce")
    raw["player_id"] = pd.to_numeric(raw["player1_id"], errors="coerce")
    valid_player = raw["player_id"].gt(0) & raw["player_id"].lt(1_000_000_000)
    raw["player_id"] = raw["player_id"].fillna(0).astype("int64")

    is_make = valid_player & raw["event_type_num"].eq(1)
    is_miss = valid_player & raw["event_type_num"].eq(2)
    is_fga = is_make | is_miss
    is_three = is_fga & raw["description"].str.contains(THREE_RE, na=False)
    is_rim = is_fga & ~is_three & raw["description"].str.contains(RIM_RE, na=False)
    is_fta = valid_player & raw["event_type_num"].eq(3)
    is_turnover = (
        valid_player
        & raw["event_type_num"].eq(5)
        & ~raw["description"].str.contains(NO_TURNOVER_RE, na=False)
    )
    is_badpass = is_turnover & raw["description"].str.contains(BADPASS_RE, na=False)
    is_assisted_make = is_make & raw["description"].str.contains(
        r"\bAST\)", case=False, regex=True, na=False
    )

    # One attempt per multi-shot FT trip; and-ones/technical shots add none.
    ft_trip = is_fta & raw["description"].str.contains(
        r"\b1 of [23]\b", case=False, regex=True, na=False
    ) & ~raw["description"].str.contains("technical", case=False, na=False)
    event_counts = pd.DataFrame(
        {
            "player_id": raw["player_id"],
            "fga": is_fga.astype(int),
            "fgm": is_make.astype(int),
            "three_fga": is_three.astype(int),
            "rim_fga": is_rim.astype(int),
            "fta": is_fta.astype(int),
            "pbp_tsa": is_fga.astype(int) + ft_trip.astype(int),
            "turnovers": is_turnover.astype(int),
            "badpass_turnovers": is_badpass.astype(int),
            "assisted_fgm": is_assisted_make.astype(int),
        }
    )
    event_counts = event_counts[valid_player].groupby("player_id", as_index=True).sum()

    assister_id, assist_audit = resolve_assister_ids(raw)
    assist_rows = raw.loc[assister_id.notna()].copy()
    assist_rows["player_id"] = assister_id.loc[assister_id.notna()].astype("int64")
    assist_rows["assists"] = 1
    assist_rows["assist_points"] = np.where(
        assist_rows["description"].str.contains(THREE_RE, na=False), 3, 2
    )
    assist_rows["rim_assists"] = (
        ~assist_rows["description"].str.contains(THREE_RE, na=False)
        & assist_rows["description"].str.contains(RIM_RE, na=False)
    ).astype(int)
    assist_counts = assist_rows.groupby("player_id")[[
        "assists",
        "assist_points",
        "rim_assists",
    ]].sum()

    off_poss = _offensive_possessions(possessions).rename("off_poss")
    player_ids = off_poss.index.union(event_counts.index).union(assist_counts.index)
    features = pd.DataFrame(index=player_ids)
    features.index.name = "player_id"
    features = features.join(off_poss).join(event_counts).join(assist_counts).fillna(0.0)
    features = features[features["off_poss"].gt(0.0)].copy()

    display_names, _ = _modal_player_names(raw)
    features["player_name"] = [display_names.get(int(player_id), "") for player_id in features.index]
    features["season"] = int(season)
    denominator = features["off_poss"] / 100.0
    for source, target in (
        ("fga", "fga100"),
        ("fta", "fta100"),
        ("turnovers", "turnovers100"),
        ("badpass_turnovers", "badpass_tov100"),
        ("assists", "assists100"),
        ("assist_points", "assist_points100"),
        ("rim_assists", "rim_assists100"),
    ):
        features[target] = features[source] / denominator
    features["tsa100"] = (features["fga"] + 0.44 * features["fta"]) / denominator
    features["three_freq"] = np.where(
        features["fga"].gt(0), features["three_fga"] / features["fga"], 0.0
    )
    features["rim_freq"] = np.where(
        features["fga"].gt(0), features["rim_fga"] / features["fga"], 0.0
    )
    features["mid_freq"] = (
        1.0 - features["three_freq"] - features["rim_freq"]
    ).clip(0.0, 1.0)
    features["usage_proxy"] = features["tsa100"] + features["turnovers100"]

    league_makes = float(features["fgm"].sum())
    league_assisted = float(features["assisted_fgm"].sum())
    league_share = league_assisted / league_makes if league_makes > 0 else 0.6
    features["official_assisted_make_share_raw"] = np.where(
        features["fgm"].gt(0), features["assisted_fgm"] / features["fgm"], np.nan
    )
    features["official_assisted_make_share_smoothed"] = (
        features["assisted_fgm"] + ASSIST_SHARE_PRIOR_MAKES * league_share
    ) / (features["fgm"] + ASSIST_SHARE_PRIOR_MAKES)
    features["official_assisted_make_league_share"] = league_share
    features = features.reset_index()

    audit = {
        "season": int(season),
        "raw_rows": int(len(raw)),
        "possession_rows": int(len(possessions)),
        "players": int(len(features)),
        "off_possessions": float(features["off_poss"].sum()),
        "league_assisted_make_share": float(league_share),
        **assist_audit,
    }
    audit["assist_resolution_rate"] = (
        audit["resolved_tags"] / audit["assist_tags"] if audit["assist_tags"] else 1.0
    )
    return features, audit


def build_season_features(season: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = season_suffix(season)
    raw_path = RAW_DIR / f"NBA{suffix}.parquet"
    possession_path = PROCESSED_DIR / f"RAPM{suffix}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw PBP {raw_path}")
    if not possession_path.exists():
        raise FileNotFoundError(f"Missing resolved possession file {possession_path}")
    available = set(pq.read_schema(raw_path).names)
    wanted = {
        "game_id",
        "event_num",
        "source_actionNumber",
        "period",
        "minute_remaining_quarter",
        "seconds_remaining_quarter",
        "event_type",
        "player1_id",
        "player1_name",
        "player2_id",
        "home_description",
        "neutral_description",
        "visitor_description",
        *{f"home_player{i}" for i in range(1, 6)},
        *{f"away_player{i}" for i in range(1, 6)},
    }
    raw = pd.read_parquet(raw_path, columns=sorted(wanted & available))
    required = {"event_type", "player1_id", "player1_name"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{raw_path} is missing required fields {sorted(missing)}")
    possessions = pd.read_parquet(
        possession_path, columns=[f"O{i}" for i in range(1, 6)]
    )
    return build_season_features_from_frames(raw, possessions, season)


def build_feature_panel(first_season: int, last_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    features: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for season in range(first_season, last_season + 1):
        frame, audit = build_season_features(season)
        features.append(frame)
        audits.append(audit)
        print(
            f"{season}: {len(frame):4d} players; assist tags resolved "
            f"{audit['assist_resolution_rate']:.1%}",
            flush=True,
        )
    return pd.concat(features, ignore_index=True), pd.DataFrame(audits)


def load_observed_labels(path: Path) -> pd.DataFrame:
    observed = pd.read_csv(path)
    required = {"nba_id", "year", "off_poss", "v_self", "v_ast", "v_potast"}
    missing = required - set(observed)
    if missing:
        raise ValueError(f"Observed-role file is missing {sorted(missing)}")
    observed = observed.rename(
        columns={
            "nba_id": "player_id",
            "year": "season",
            "off_poss": "label_off_poss",
            "v_self": "actual_self_created_tsa100",
            "v_ast": "actual_assisted_tsa100",
            "v_potast": "potential_assists100",
        }
    )
    observed["player_id"] = pd.to_numeric(observed["player_id"], errors="coerce").astype("Int64")
    shot_total = observed["actual_self_created_tsa100"] + observed["actual_assisted_tsa100"]
    observed["assist_opp_share"] = np.where(
        shot_total.gt(0.0), observed["actual_assisted_tsa100"] / shot_total, np.nan
    )
    return observed


def build_training_panel(features: pd.DataFrame, observed_path: Path) -> pd.DataFrame:
    observed = load_observed_labels(observed_path)
    panel = observed.merge(
        features,
        on=["season", "player_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_observed", ""),
    )
    required = sorted(
        {
            "assist_opp_share",
            "potential_assists100",
            "label_off_poss",
            *[feature for families in FEATURE_FAMILIES.values() for values in families.values() for feature in values],
        }
    )
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    return panel[panel["label_off_poss"].ge(MIN_TRAIN_POSS)].copy()


def _model(features: list[str], alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def weighted_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    weight: Iterable[float],
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(predicted, dtype=float)
    weights = np.maximum(np.asarray(weight, dtype=float), 0.0)
    total_weight = max(float(weights.sum()), 1.0)
    mean = float(np.sum(weights * actual_array) / total_weight)
    mse = float(np.sum(weights * np.square(actual_array - prediction_array)) / total_weight)
    variance = float(np.sum(weights * np.square(actual_array - mean)) / total_weight)
    centered_actual = actual_array - mean
    prediction_mean = float(np.sum(weights * prediction_array) / total_weight)
    centered_prediction = prediction_array - prediction_mean
    denominator = float(
        np.sqrt(
            np.sum(weights * np.square(centered_actual))
            * np.sum(weights * np.square(centered_prediction))
        )
    )
    return {
        "rmse": float(np.sqrt(mse)),
        "r2": 1.0 - mse / variance if variance > 0.0 else np.nan,
        "correlation": (
            float(np.sum(weights * centered_actual * centered_prediction) / denominator)
            if denominator > 0.0
            else np.nan
        ),
    }


def chronological_predictions(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    alpha: float,
    lower: float,
    upper: float | None,
) -> pd.DataFrame:
    seasons = sorted(int(value) for value in panel["season"].unique())
    rows: list[dict[str, Any]] = []
    for test_position in range(MIN_TRAIN_SEASONS, len(seasons)):
        test_season = seasons[test_position]
        train_seasons = seasons[:test_position]
        train = panel[panel["season"].isin(train_seasons)]
        test = panel[panel["season"].eq(test_season)]
        fitted = _model(features, alpha)
        fitted.fit(
            train[features],
            train[target],
            ridge__sample_weight=train["label_off_poss"],
        )
        prediction = np.maximum(fitted.predict(test[features]), lower)
        if upper is not None:
            prediction = np.minimum(prediction, upper)
        for row, estimate in zip(test.itertuples(index=False), prediction):
            rows.append(
                {
                    "target": target,
                    "alpha": float(alpha),
                    "test_season": int(test_season),
                    "train_first": int(train_seasons[0]),
                    "train_last": int(train_seasons[-1]),
                    "player_id": int(row.player_id),
                    "weight": float(row.label_off_poss),
                    "actual": float(getattr(row, target)),
                    "prediction": float(estimate),
                }
            )
    if not rows:
        raise ValueError(
            f"Need more than {MIN_TRAIN_SEASONS} observed seasons for chronological validation"
        )
    return pd.DataFrame(rows)


def select_model_family(
    panel: pd.DataFrame,
    target: str,
    families: dict[str, list[str]],
    lower: float,
    upper: float | None,
) -> tuple[Pipeline, str, float, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for family, feature_names in families.items():
        for alpha in ALPHAS:
            predictions = chronological_predictions(
                panel, target, feature_names, alpha, lower, upper
            )
            predictions["family"] = family
            prediction_frames.append(predictions)
            metrics = weighted_metrics(
                predictions["actual"], predictions["prediction"], predictions["weight"]
            )
            summaries.append(
                {
                    "target": target,
                    "family": family,
                    "alpha": float(alpha),
                    "features": "|".join(feature_names),
                    "test_first": int(predictions["test_season"].min()),
                    "test_last": int(predictions["test_season"].max()),
                    "rows": int(len(predictions)),
                    **metrics,
                }
            )
    summary = pd.DataFrame(summaries).sort_values(["rmse", "alpha", "family"])
    selected = summary.iloc[0]
    selected_family = str(selected["family"])
    selected_alpha = float(selected["alpha"])
    summary["selected"] = (
        summary["family"].eq(selected_family) & summary["alpha"].eq(selected_alpha)
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selected_predictions = predictions[
        predictions["family"].eq(selected_family)
        & predictions["alpha"].eq(selected_alpha)
    ].copy()
    final = _model(families[selected_family], selected_alpha)
    final.fit(
        panel[families[selected_family]],
        panel[target],
        ridge__sample_weight=panel["label_off_poss"],
    )
    return final, selected_family, selected_alpha, summary, selected_predictions


def score_role_panel(
    features: pd.DataFrame,
    share_model: Pipeline,
    share_features: list[str],
    potast_model: Pipeline,
    potast_features: list[str],
    observed_path: Path,
) -> pd.DataFrame:
    scored = features.copy()
    # Keep the fitted role predictors frozen; change only the shot-volume basis.
    canonical_tsa = scored.get("canonical_tsa100", scored["tsa100"]).copy()
    scored["estimated_assisted_share"] = np.clip(
        share_model.predict(scored[share_features]), 0.0, 1.0
    )
    scored["estimated_potential_assists100"] = np.maximum(
        potast_model.predict(scored[potast_features]), 0.0
    )
    scored["tsa100"] = canonical_tsa
    scored["estimated_assisted_tsa100"] = canonical_tsa * scored["estimated_assisted_share"]
    scored["estimated_self_created_tsa100"] = canonical_tsa - scored["estimated_assisted_tsa100"]
    scored["shot_volume_identity_gap"] = (
        scored["estimated_self_created_tsa100"]
        + scored["estimated_assisted_tsa100"]
        - scored["tsa100"]
    )
    observed = load_observed_labels(observed_path)[
        [
            "season",
            "player_id",
            "actual_self_created_tsa100",
            "actual_assisted_tsa100",
            "potential_assists100",
            "assist_opp_share",
        ]
    ].rename(
        columns={
            "potential_assists100": "actual_potential_assists100",
            "assist_opp_share": "actual_assist_opp_share",
        }
    )
    scored = scored.merge(observed, on=["season", "player_id"], how="left", validate="one_to_one")
    first_observed = int(observed["season"].min())
    scored["estimate_status"] = np.select(
        [
            scored["season"].lt(first_observed),
            scored["actual_assist_opp_share"].notna(),
        ],
        [
            "pre_label_backcast_research_only",
            "estimated_with_observed_label_available",
        ],
        default="modern_estimate_below_observed_surface",
    )
    scored["qualified_role_estimate"] = scored["off_poss"].ge(MIN_TRAIN_POSS).astype(int)
    if float(scored["shot_volume_identity_gap"].abs().max()) > 1e-10:
        raise AssertionError("Estimated self-created plus assisted TSA does not close to TSA")
    return scored


def target_path(metric: str, lo: int, hi: int) -> Path:
    """Return the isolated count-target solve used by the price fit."""

    return PIPELINE_DIR / "results" / (
        f"{metric}_{lo % 100:02d}_{hi % 100:02d}_rs_se_"
        f"a{PRICE_ALPHA}_{PRICE_ALPHA}_results.csv"
    )


def estimated_role_panel(scored: pd.DataFrame) -> pd.DataFrame:
    """Map scored all-era features onto the pricing/ledger contract."""

    out = scored.rename(
        columns={
            "season": "year",
            "player_id": "nba_id",
            "player_name": "name",
            "off_poss": "OffPoss",
            "estimated_self_created_tsa100": "v_self",
            "estimated_assisted_tsa100": "v_ast",
            "estimated_potential_assists100": "v_potast",
            "turnovers100": "act",
            "badpass_tov100": "bp_100",
        }
    ).copy()
    out["sc_100"] = out["act"] - out["bp_100"]
    out["n_sc"] = (out["turnovers"] - out["badpass_turnovers"]).clip(lower=0)
    out["n_bp"] = out["badpass_turnovers"].clip(lower=0)
    out["involve"] = out["v_self"] + out["v_ast"] + out["v_potast"]
    keep = [
        "nba_id", "name", "year", "OffPoss", "tsa100", "v_self", "v_ast", "v_potast",
        "n_sc", "n_bp", "sc_100", "bp_100", "act", "involve",
    ]
    return out[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()


def observed_role_panel(path: Path) -> pd.DataFrame:
    """Load the modern observed-role ledger without changing the source artifact."""

    observed = pd.read_csv(path).rename(columns={"off_poss": "OffPoss"})
    required = {
        "nba_id", "year", "OffPoss", "v_self", "v_ast", "v_potast",
        "sc_100", "bp_100",
    }
    missing = required - set(observed)
    if missing:
        raise ValueError(f"Observed rZTOV ledger is missing {sorted(missing)}")
    if "name" not in observed:
        observed["name"] = ""
    observed["act"] = observed["sc_100"] + observed["bp_100"]
    observed["tsa100"] = observed["v_self"] + observed["v_ast"]
    observed["n_sc"] = observed["sc_100"] * observed["OffPoss"] / 100.0
    observed["n_bp"] = observed["bp_100"] * observed["OffPoss"] / 100.0
    observed["involve"] = observed["v_self"] + observed["v_ast"] + observed["v_potast"]
    keep = [
        "nba_id", "name", "year", "OffPoss", "tsa100", "v_self", "v_ast", "v_potast",
        "n_sc", "n_bp", "sc_100", "bp_100", "act", "involve",
    ]
    return observed[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()


def common_modern_panels(
    estimated: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use identical players, seasons, turnover rates, and possessions in the audit."""

    volume = estimated[["nba_id", "year", "tsa100", "v_self", "v_ast", "v_potast"]].rename(
        columns={c: f"estimated_{c}" for c in ("tsa100", "v_self", "v_ast", "v_potast")}
    )
    common = observed.merge(volume, on=["nba_id", "year"], how="inner", validate="one_to_one")
    observed_common = common[[
        "nba_id", "name", "year", "OffPoss", "tsa100", "v_self", "v_ast", "v_potast",
        "n_sc", "n_bp", "sc_100", "bp_100", "act", "involve",
    ]].copy()
    estimated_common = common[[
        "nba_id", "name", "year", "OffPoss", "n_sc", "n_bp", "sc_100", "bp_100", "act", "estimated_tsa100",
        "estimated_v_self", "estimated_v_ast", "estimated_v_potast",
    ]].rename(columns={
        "estimated_tsa100": "tsa100",
        "estimated_v_self": "v_self",
        "estimated_v_ast": "v_ast",
        "estimated_v_potast": "v_potast",
    })
    estimated_common["involve"] = (
        estimated_common["v_self"] + estimated_common["v_ast"]
        + estimated_common["v_potast"]
    )
    return estimated_common, observed_common


def window_frame(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Possession-weighted player aggregates for one rolling fit window."""

    window = df[df["year"].between(lo, hi)].copy()
    rows: list[dict[str, float | int]] = []
    for nba_id, group in window.groupby("nba_id", sort=False):
        poss = float(group["OffPoss"].sum())
        if poss <= 0:
            continue
        rows.append({
            "nba_id": int(nba_id),
            "poss": poss,
            "sc_100": 100.0 * float(group["n_sc"].sum()) / poss,
            "bp_100": 100.0 * float(group["n_bp"].sum()) / poss,
            **{
                column: float(np.average(group[column], weights=group["OffPoss"]))
                for column in ("v_self", "v_ast", "v_potast")
            },
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["act"] = result["sc_100"] + result["bp_100"]
    return result


def _read_target(metric: str, lo: int, hi: int, column: str) -> pd.DataFrame:
    path = target_path(metric, lo, hi)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing historical count-basis target {path}. Run rapm.py {metric.upper()} "
            f"{lo % 100:02d} {hi % 100:02d} RS --season-effects --off-alpha "
            f"{PRICE_ALPHA} --def-alpha {PRICE_ALPHA}. No turnover lane is imputed."
        )
    target = pd.read_csv(path, usecols=["player_id", "off"]).rename(
        columns={"player_id": "nba_id", "off": column}
    )
    target["nba_id"] = pd.to_numeric(target["nba_id"], errors="coerce").astype("Int64")
    return target.dropna()


def _joint_price_solve(g: pd.DataFrame) -> dict[str, float | int]:
    """Exact three-target/two-lane alternating solve used by rZTOV explorer."""

    wt = g["poss"].to_numpy(float)
    sc, bp, act = (g[c].to_numpy(float) for c in ("sc_100", "bp_100", "act"))
    vs, va, vp = (g[c].to_numpy(float) for c in ("v_self", "v_ast", "v_potast"))
    y1, y2, y3 = (g[c].to_numpy(float) for c in ("y_sc", "y_bp", "y_tov"))
    z = np.zeros_like(sc)
    sw = np.sqrt(wt)

    def origin(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]

    lane_sc = origin(y1, np.column_stack([sc, vs, va]))
    lane_bp = origin(y2, np.column_stack([bp, vp]))
    if abs(lane_sc[0]) < 1e-12 or abs(lane_bp[0]) < 1e-12:
        raise ValueError("Turnover lane target does not identify a finite break-even ratio")
    lane = (
        float(-lane_sc[1] / lane_sc[0]),
        float(-lane_sc[2] / lane_sc[0]),
        float(-lane_bp[1] / lane_bp[0]),
    )
    bs, ba, bpp = lane
    converged = False
    iterations = 0
    for iterations in range(1, 301):
        al_sc, al_bp = bs * vs + ba * va, bpp * vp
        r1, r2, r3 = sc - al_sc, bp - al_bp, act - al_sc - al_bp
        denominators = [np.sum(wt * r * r) for r in (r1, r2, r3)]
        if min(denominators) <= 1e-16:
            raise ValueError("Degenerate residual volume in joint rZTOV price solve")
        k1 = float(np.sum(wt * y1 * r1) / denominators[0])
        k2 = float(np.sum(wt * y2 * r2) / denominators[1])
        k3 = float(np.sum(wt * y3 * r3) / denominators[2])
        variances = [
            max(float(np.sum(wt * (y - k * r) ** 2) / np.sum(wt)), 1e-12)
            for y, k, r in ((y1, k1, r1), (y2, k2, r2), (y3, k3, r3))
        ]
        x = np.vstack([
            np.column_stack([-k1 * vs, -k1 * va, z]),
            np.column_stack([z, z, -k2 * vp]),
            np.column_stack([-k3 * vs, -k3 * va, -k3 * vp]),
        ])
        y_stack = np.concatenate([y1 - k1 * sc, y2 - k2 * bp, y3 - k3 * act])
        stack_weight = np.sqrt(np.concatenate([
            wt / variances[0], wt / variances[1], wt / variances[2]
        ]))
        previous = np.array([bs, ba, bpp])
        bs, ba, bpp = np.linalg.lstsq(
            x * stack_weight[:, None], y_stack * stack_weight, rcond=None
        )[0]
        if float(np.max(np.abs(np.array([bs, ba, bpp]) - previous))) < 1e-10:
            converged = True
            break

    def fit_quality(y: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
        denominator = float(np.sum(wt * residual * residual))
        kappa = float(np.sum(wt * y * residual) / denominator)
        error = float(np.sum(wt * np.square(y - kappa * residual)))
        total = float(np.sum(wt * np.square(y - np.average(y, weights=wt))))
        return kappa, 1.0 - error / total if total > 0 else np.nan

    al_sc, al_bp = bs * vs + ba * va, bpp * vp
    k_sc, r2_sc = fit_quality(y1, sc - al_sc)
    k_bp, r2_bp = fit_quality(y2, bp - al_bp)
    k_tot, r2_tot = fit_quality(y3, act - al_sc - al_bp)
    return {
        "self": float(bs), "ast": float(ba), "potast": float(bpp),
        "lane_self": lane[0], "lane_ast": lane[1], "lane_potast": lane[2],
        "k_sc": k_sc, "k_bp": k_bp, "kappa": k_tot,
        "r2_sc": r2_sc, "r2_bp": r2_bp, "r2": r2_tot,
        "iterations": iterations, "converged": int(converged),
        "n": int(len(g)), "poss": float(wt.sum()),
    }


def fit_price_window(
    df: pd.DataFrame, lo: int, hi: int, bootstrap_draws: int = 0, seed: int = 11
) -> dict[str, Any]:
    """Fit one rolling price window and optional player-cluster bootstrap bands."""

    g = window_frame(df, lo, hi)
    g = g[g["poss"].ge(FIT_MIN_POSS)].copy()
    for metric, column in (
        ("scoring_tov", "y_sc"), ("badpass_tov", "y_bp"), ("tov", "y_tov")
    ):
        g = g.merge(_read_target(metric, lo, hi, column), on="nba_id", how="inner")
    g = g.dropna(subset=["y_sc", "y_bp", "y_tov"])
    if len(g) < 20:
        raise ValueError(f"Only {len(g)} target-complete players in {lo}-{hi}")
    result: dict[str, Any] = {"window_lo": lo, "window_hi": hi, **_joint_price_solve(g)}
    if bootstrap_draws:
        rng = np.random.default_rng(seed + hi)
        draws: list[dict[str, float | int]] = []
        for _ in range(bootstrap_draws):
            sample = g.iloc[rng.integers(0, len(g), len(g))].reset_index(drop=True)
            try:
                draws.append(_joint_price_solve(sample))
            except (ValueError, np.linalg.LinAlgError):
                continue
        for price in ("self", "ast", "potast"):
            values = np.asarray([float(draw[price]) for draw in draws])
            result[f"{price}_p05"] = float(np.quantile(values, 0.05)) if len(values) else np.nan
            result[f"{price}_p95"] = float(np.quantile(values, 0.95)) if len(values) else np.nan
        result["bootstrap_requested"] = int(bootstrap_draws)
        result["bootstrap_successful"] = int(len(draws))
    return result


def fit_price_path(
    df: pd.DataFrame, first_end: int, last_end: int, bootstrap_draws: int = 0
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    fits: dict[int, dict[str, Any]] = {}
    for end in range(first_end, last_end + 1):
        fits[end] = fit_price_window(
            df, end - WINDOW + 1, end, bootstrap_draws=bootstrap_draws
        )
        print(
            f"price {end - WINDOW + 1}-{end}: self={fits[end]['self']:.5f} "
            f"ast={fits[end]['ast']:.5f} potast={fits[end]['potast']:.5f}",
            flush=True,
        )
    return fits, pd.DataFrame(fits.values()).sort_values("window_hi")


def score_ledger(
    df: pd.DataFrame, fits: dict[int, dict[str, Any]], series: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply rolling prices, season-center both lanes, and enforce all identities."""

    if not fits:
        raise ValueError("No price fits supplied")
    rows = df[df["OffPoss"].ge(MIN_TRAIN_POSS)].copy()
    earliest, latest = min(fits), max(fits)
    rows["be_source"] = rows["year"].clip(lower=earliest, upper=latest).astype(int)
    rows["early_window_inherited"] = rows["year"].lt(earliest).astype(int)
    for price in ("self", "ast", "potast"):
        rows[f"be_{price}"] = rows["be_source"].map(
            lambda end, price=price: float(fits[int(end)][price])
        )
        rows[f"allow_{price}"] = rows[f"be_{price}"] * rows[f"v_{price}"]
    rows["sc_allow"] = rows["allow_self"] + rows["allow_ast"]
    rows["bp_allow"] = rows["allow_potast"]
    centre_rows: list[dict[str, float | int]] = []
    for year, group in rows.groupby("year", sort=True):
        weight = group["OffPoss"].to_numpy(float)
        sc_centre = float(np.average(group["sc_100"] - group["sc_allow"], weights=weight))
        bp_centre = float(np.average(group["bp_100"] - group["bp_allow"], weights=weight))
        rows.loc[group.index, "centre_sc"] = sc_centre
        rows.loc[group.index, "centre_bp"] = bp_centre
        centre_rows.append({"year": int(year), "centre_sc": sc_centre, "centre_bp": bp_centre})
    rows["sc_allow"] += rows["centre_sc"]
    rows["bp_allow"] += rows["centre_bp"]
    rows["sc_excess"] = rows["sc_100"] - rows["sc_allow"]
    rows["bp_excess"] = rows["bp_100"] - rows["bp_allow"]
    rows["allowance"] = rows["sc_allow"] + rows["bp_allow"]
    rows["excess"] = rows["sc_excess"] + rows["bp_excess"]
    rows["act"] = rows["sc_100"] + rows["bp_100"]
    rows["involve"] = rows["v_self"] + rows["v_ast"] + rows["v_potast"]
    rows["plays"] = rows["involve"] + rows["act"]
    rows["rztov"] = 100.0 * rows["excess"] / rows["plays"].clip(lower=1.0)
    rows["tov_saved_100"] = -rows["excess"]
    rows["tov_saved"] = -rows["excess"] * rows["OffPoss"] / 100.0
    rows["qualified"] = (
        rows["OffPoss"].ge(QUAL_MIN_POSS) & rows["involve"].ge(QUAL_MIN_INVOLVE)
    ).astype(int)
    rows["series"] = series
    rows["shot_volume_identity_gap"] = rows["v_self"] + rows["v_ast"] - rows["tsa100"]
    lane_gap = (rows["excess"] - rows["sc_excess"] - rows["bp_excess"]).abs().max()
    saved_gap = (rows["tov_saved_100"] + rows["excess"]).abs().max()
    shot_gap = rows["shot_volume_identity_gap"].abs().max()
    if max(float(lane_gap), float(saved_gap), float(shot_gap)) > 1e-10:
        raise AssertionError("rZTOV shot/lane/saved identities do not close")
    for _, group in rows.groupby("year"):
        for lane in ("sc", "bp"):
            mean = float(np.average(group[f"{lane}_excess"], weights=group["OffPoss"]))
            if abs(mean) > 1e-10:
                raise AssertionError(f"{lane} lane does not center in {int(group['year'].iloc[0])}")
    return rows, pd.DataFrame(centre_rows)


def roll_up_ledger(rows: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    window = rows[rows["year"].between(lo, hi)].copy()
    values = ["v_self", "v_ast", "v_potast", "sc_allow", "bp_allow", "sc_100", "bp_100"]
    output: list[dict[str, float | int]] = []
    for nba_id, group in window.groupby("nba_id", sort=False):
        poss = float(group["OffPoss"].sum())
        if poss <= 0:
            continue
        output.append({
            "nba_id": int(nba_id), "poss": poss,
            **{column: float(np.average(group[column], weights=group["OffPoss"])) for column in values},
        })
    result = pd.DataFrame(output)
    result["act"] = result["sc_100"] + result["bp_100"]
    result["involve"] = result["v_self"] + result["v_ast"] + result["v_potast"]
    result["excess"] = result["act"] - result["sc_allow"] - result["bp_allow"]
    result["plays"] = result["involve"] + result["act"]
    result["rztov"] = 100.0 * result["excess"] / result["plays"].clip(lower=1.0)
    result["qualified"] = (
        result["poss"].ge(QUAL_MIN_POSS) & result["involve"].ge(QUAL_MIN_INVOLVE)
    ).astype(int)
    return result


def comparison_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | int]:
    common = left[["nba_id", "rztov", "poss", "qualified"]].merge(
        right[["nba_id", "rztov", "qualified"]],
        on="nba_id", suffixes=("_estimated", "_observed")
    )
    common = common[
        common["qualified_estimated"].eq(1) & common["qualified_observed"].eq(1)
    ]
    metrics = weighted_metrics(common["rztov_observed"], common["rztov_estimated"], common["poss"])
    metrics["rank_correlation"] = float(
        common["rztov_observed"].rank().corr(common["rztov_estimated"].rank())
    )
    metrics["rows"] = int(len(common))
    return metrics


def regular_season_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """NBA 002 game IDs only, including older integer-encoded game IDs."""
    game = frame["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    return frame.loc[game.str.startswith("002")].copy()


def canonical_tsa_counts(raw: pd.DataFrame) -> pd.DataFrame:
    """Count FGA plus two/three-shot FT trips on one explicit RS sample."""
    raw = regular_season_rows(raw.drop_duplicates())
    desc = _descriptions(raw)
    pid = pd.to_numeric(raw["player1_id"], errors="coerce").fillna(0).astype("int64")
    event = pd.to_numeric(raw["event_type"], errors="coerce")
    fga = event.isin([1, 2])
    fta = event.eq(3)
    trip = fta & desc.str.contains(r"\b1 of [23]\b", case=False, regex=True, na=False)
    trip &= ~desc.str.contains("technical", case=False, na=False)
    counts = pd.DataFrame({"player_id": pid, "pbp_tsa": fga.astype(int) + trip.astype(int),
                           "fga": fga.astype(int), "fta": fta.astype(int)})
    return counts[pid.between(1, 999999999)].groupby("player_id").sum()


def attach_canonical_tsa(features: pd.DataFrame) -> pd.DataFrame:
    """Use counts and exposure from the same complete regular-season PBP sample.

    Reader TSA rates may be a partial-season snapshot. Never multiply those rates
    by a different snapshot's possessions to reconstruct workload counts.
    """
    out = features.copy()
    out["canonical_tsa100"] = np.nan
    out["canonical_tsa_count"] = np.nan
    out["tsa_source"] = "same_sample_RS_raw_PBP_FGA_plus_multishot_FT_trips"
    for season in sorted(out["season"].unique()):
        suffix = season_suffix(int(season))
        raw = pd.read_parquet(RAW_DIR / f"NBA{suffix}.parquet")
        counts = canonical_tsa_counts(raw)
        possessions = regular_season_rows(pd.read_parquet(
            PROCESSED_DIR / f"RAPM{suffix}.parquet",
            columns=["game_id", *[f"O{i}" for i in range(1, 6)]],
        ))
        exposure = _offensive_possessions(possessions)
        mask = out["season"].eq(season)
        rows = out.loc[mask]
        off = rows["player_id"].map(exposure).fillna(0)
        if not np.allclose(off, rows["off_poss"], atol=1e-8):
            raise ValueError(f"{season}: cached exposure differs from RS PBP; rebuild feature cache")
        for column in ("fga", "fta"):
            actual = rows["player_id"].map(counts[column]).fillna(0)
            if not np.allclose(actual, rows[column], atol=1e-8):
                raise ValueError(f"{season}: cached {column} differs from RS PBP; rebuild feature cache")
        tsa = rows["player_id"].map(counts["pbp_tsa"]).fillna(0)
        out.loc[mask, "canonical_tsa_count"] = tsa
        out.loc[mask, "canonical_tsa100"] = 100.0 * tsa / off
    if out["canonical_tsa100"].isna().any():
        raise ValueError("Unresolved same-sample TSA coverage")
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = args.feature_cache or (
        args.output_dir / f"all_era_box_features_{args.first_season}_{args.last_season}.parquet"
    )
    audit_path = args.output_dir / "feature_source_audit.csv"
    if args.reuse_feature_cache:
        if not feature_cache.exists():
            raise FileNotFoundError(
                f"--reuse-feature-cache requested but {feature_cache} does not exist"
            )
        features = pd.read_parquet(feature_cache)
        if audit_path.exists():
            source_audit = pd.read_csv(audit_path)
        else:
            source_audit = pd.DataFrame()
    else:
        features, source_audit = build_feature_panel(args.first_season, args.last_season)
        features.to_parquet(feature_cache, index=False)
        source_audit.to_csv(audit_path, index=False)

    features = attach_canonical_tsa(features)
    panel = build_training_panel(features, args.observed_roles)
    share_model, share_family, share_alpha, share_summary, share_predictions = (
        select_model_family(
            panel,
            "assist_opp_share",
            FEATURE_FAMILIES["assist_opp_share"],
            lower=0.0,
            upper=1.0,
        )
    )
    potast_model, potast_family, potast_alpha, potast_summary, potast_predictions = (
        select_model_family(
            panel,
            "potential_assists100",
            FEATURE_FAMILIES["potential_assists100"],
            lower=0.0,
            upper=None,
        )
    )
    validation = pd.concat([share_summary, potast_summary], ignore_index=True)
    validation.to_csv(args.output_dir / "chronological_model_validation.csv", index=False)
    predictions = pd.concat([share_predictions, potast_predictions], ignore_index=True)
    predictions.to_csv(args.output_dir / "chronological_selected_predictions.csv", index=False)

    scored = score_role_panel(
        features,
        share_model,
        FEATURE_FAMILIES["assist_opp_share"][share_family],
        potast_model,
        FEATURE_FAMILIES["potential_assists100"][potast_family],
        args.observed_roles,
    )
    output_columns = [
        "season",
        "player_id",
        "player_name",
        "off_poss",
        "tsa100",
        "tsa_source",
        "canonical_tsa_count",
        "estimated_self_created_tsa100",
        "estimated_assisted_tsa100",
        "estimated_assisted_share",
        "estimated_potential_assists100",
        "official_assisted_make_share_raw",
        "official_assisted_make_share_smoothed",
        "actual_self_created_tsa100",
        "actual_assisted_tsa100",
        "actual_assist_opp_share",
        "actual_potential_assists100",
        "qualified_role_estimate",
        "estimate_status",
        "shot_volume_identity_gap",
    ]
    board_path = args.output_dir / (
        f"nba_rztov_box_role_backcast_{args.first_season}_{args.last_season}.csv"
    )
    scored[output_columns].sort_values(["season", "off_poss"], ascending=[True, False]).to_csv(
        board_path, index=False
    )

    # Historical prices are estimated from the same three independent count-basis
    # RAPM targets as the modern explorer. Nothing is copied backward from a modern
    # price path: each complete trailing-five-season window gets its own joint fit.
    estimated_panel = estimated_role_panel(scored)
    first_fit_end = args.first_season + WINDOW - 1
    historical_fits, historical_price_path = fit_price_path(
        estimated_panel,
        first_fit_end,
        args.last_season,
        bootstrap_draws=args.bootstrap_draws,
    )
    historical_price_path["role_source"] = "box_estimated"
    historical_price_path_path = args.output_dir / "historical_box_price_path_1997_2026.csv"
    historical_price_path.round(8).to_csv(historical_price_path_path, index=False)

    all_ledger, historical_centres = score_ledger(
        estimated_panel, historical_fits, "box_estimated_rztov"
    )
    ledger_columns = [
        "nba_id", "name", "year", "OffPoss", "series", "be_source",
        "early_window_inherited", "tsa100", "v_self", "v_ast", "v_potast", "involve",
        "plays", "sc_100", "bp_100", "act", "be_self", "be_ast", "be_potast",
        "sc_allow", "bp_allow", "sc_excess", "bp_excess", "allowance", "excess",
        "rztov", "tov_saved_100", "tov_saved", "qualified", "shot_volume_identity_gap",
    ]
    all_ledger_path = args.output_dir / "nba_box_estimated_rztov_1997_2026_diagnostic.csv"
    all_ledger[ledger_columns].sort_values(["year", "rztov"]).to_csv(
        all_ledger_path, index=False
    )
    historical_ledger = all_ledger[all_ledger["year"].lt(2014)].copy()
    historical_ledger_path = args.output_dir / "nba_box_estimated_rztov_1997_2013.csv"
    historical_ledger[ledger_columns].sort_values(["year", "rztov"]).to_csv(
        historical_ledger_path, index=False
    )
    historical_centres.to_csv(args.output_dir / "historical_box_lane_centres.csv", index=False)

    # The modern overlap changes only role volumes. Target solves, players, seasons,
    # possessions, and actual turnover counts are held identical between surfaces.
    observed_panel = observed_role_panel(args.observed_roles)
    modern_estimated, modern_observed = common_modern_panels(estimated_panel, observed_panel)
    overlap_first_end = max(2018, int(modern_observed["year"].min()) + WINDOW - 1)
    overlap_last_end = min(args.last_season, int(modern_observed["year"].max()))
    estimated_overlap_fits, estimated_overlap_prices = fit_price_path(
        modern_estimated, overlap_first_end, overlap_last_end, bootstrap_draws=0
    )
    observed_overlap_fits, observed_overlap_prices = fit_price_path(
        modern_observed, overlap_first_end, overlap_last_end, bootstrap_draws=0
    )
    compare_columns = ["window_hi", "window_lo", "self", "ast", "potast", "k_sc", "k_bp", "kappa", "n", "poss"]
    price_comparison = estimated_overlap_prices[compare_columns].merge(
        observed_overlap_prices[compare_columns],
        on=["window_hi", "window_lo"],
        suffixes=("_estimated", "_observed"),
        validate="one_to_one",
    )
    for price in ("self", "ast", "potast"):
        price_comparison[f"{price}_delta"] = (
            price_comparison[f"{price}_estimated"] - price_comparison[f"{price}_observed"]
        )
    price_comparison_path = args.output_dir / "modern_overlap_price_comparison.csv"
    price_comparison.round(8).to_csv(price_comparison_path, index=False)

    estimated_overlap_ledger, _ = score_ledger(
        modern_estimated, estimated_overlap_fits, "box_estimated_roles_common_modern"
    )
    observed_overlap_ledger, _ = score_ledger(
        modern_observed, observed_overlap_fits, "observed_roles_common_modern"
    )
    player_overlap = estimated_overlap_ledger[[
        "nba_id", "name", "year", "OffPoss", "rztov", "qualified",
        "v_self", "v_ast", "v_potast",
    ]].merge(
        observed_overlap_ledger[[
            "nba_id", "year", "rztov", "qualified", "v_self", "v_ast", "v_potast",
        ]],
        on=["nba_id", "year"],
        suffixes=("_estimated", "_observed"),
        validate="one_to_one",
    )
    player_overlap["rztov_delta"] = player_overlap["rztov_estimated"] - player_overlap["rztov_observed"]
    player_overlap_path = args.output_dir / "modern_overlap_player_seasons.csv"
    player_overlap.sort_values(["year", "rztov_observed"]).to_csv(
        player_overlap_path, index=False
    )
    season_metrics: list[dict[str, float | int | str]] = []
    for year, group in player_overlap.groupby("year"):
        qualified = group[
            group["qualified_estimated"].eq(1) & group["qualified_observed"].eq(1)
        ]
        metrics = weighted_metrics(
            qualified["rztov_observed"], qualified["rztov_estimated"], qualified["OffPoss"]
        )
        metrics["rank_correlation"] = float(
            qualified["rztov_observed"].rank().corr(qualified["rztov_estimated"].rank())
        )
        season_metrics.append({"scope": str(int(year)), "rows": int(len(qualified)), **metrics})
    board_lo = int(modern_observed["year"].min())
    board_hi = int(modern_observed["year"].max())
    estimated_rollup = roll_up_ledger(estimated_overlap_ledger, board_lo, board_hi)
    observed_rollup = roll_up_ledger(observed_overlap_ledger, board_lo, board_hi)
    pooled_overlap_metrics = comparison_metrics(estimated_rollup, observed_rollup)
    season_metrics.append({"scope": f"{board_lo}-{board_hi}_player_rollup", **pooled_overlap_metrics})
    overlap_summary = pd.DataFrame(season_metrics)
    overlap_summary_path = args.output_dir / "modern_overlap_summary.csv"
    overlap_summary.round(8).to_csv(overlap_summary_path, index=False)

    observed_reference = pd.read_csv(args.observed_roles).copy()
    observed_reference["series"] = "observed_modern_rztov_reference"
    observed_reference_path = args.output_dir / "nba_observed_rztov_2014_2026_reference.csv"
    observed_reference.to_csv(observed_reference_path, index=False)

    share_metrics = weighted_metrics(
        share_predictions["actual"],
        share_predictions["prediction"],
        share_predictions["weight"],
    )
    potast_metrics = weighted_metrics(
        potast_predictions["actual"],
        potast_predictions["prediction"],
        potast_predictions["weight"],
    )
    first_label = int(panel["season"].min())
    last_label = int(panel["season"].max())
    manifest = {
        "status": "research_only_box_estimated_rztov_backcast_not_production_rztov",
        "coverage": {
            "feature_first_season": int(features["season"].min()),
            "feature_last_season": int(features["season"].max()),
            "observed_label_first_season": first_label,
            "observed_label_last_season": last_label,
            "pre_label_backcast_seasons": [int(args.first_season), first_label - 1],
            "players_seasons": int(len(scored)),
            "training_rows": int(len(panel)),
        },
        "feature_contract": {
            "source": "repo raw NBA PBP plus resolved RAPM possessions",
            "official_assisted_share": (
                "made-only official assist tags, smoothed by 40 makes toward the "
                "same-season league rate; distinct from potential-assisted attempt share"
            ),
            "feature_cache": str(feature_cache),
            "source_audit": str(audit_path),
        },
        "validation": {
            "design": (
                "expanding chronological seasons; each scored season uses only earlier "
                "observed-label seasons; all observed seasons remain development evidence"
            ),
            "first_test_season": int(share_predictions["test_season"].min()),
            "last_test_season": int(share_predictions["test_season"].max()),
            "assist_opp_share": {
                "family": share_family,
                "alpha": share_alpha,
                **share_metrics,
                "rmse_percentage_points": 100.0 * share_metrics["rmse"],
            },
            "potential_assists100": {
                "family": potast_family,
                "alpha": potast_alpha,
                **potast_metrics,
            },
        },
        "identity": {
            "formula": (
                "estimated_self_created_tsa100 + estimated_assisted_tsa100 = tsa100"
            ),
            "max_abs_gap": float(scored["shot_volume_identity_gap"].abs().max()),
            "max_abs_lane_gap": float(
                (all_ledger["excess"] - all_ledger["sc_excess"] - all_ledger["bp_excess"])
                .abs().max()
            ),
            "max_abs_saved_gap": float(
                (all_ledger["tov_saved_100"] + all_ledger["excess"]).abs().max()
            ),
        },
        "historical_pricing": {
            "method": (
                "rolling five-season joint three-target/two-lane alternating solve "
                "against TOV, SCORING_TOV, and BADPASS_TOV RS+SE a1500/1500 targets"
            ),
            "first_fit_window": [args.first_season, first_fit_end],
            "last_fit_window": [args.last_season - WINDOW + 1, args.last_season],
            "early_seasons_inherit_earliest_fit": [args.first_season, first_fit_end - 1],
            "bootstrap_draws_per_window": int(args.bootstrap_draws),
            "price_path": str(historical_price_path_path),
        },
        "modern_overlap": {
            "design": (
                "estimated versus observed role volumes on identical players, windows, "
                "turnover counts, possessions, and count-basis target solves"
            ),
            "player_rollup_scope": [board_lo, board_hi],
            **pooled_overlap_metrics,
        },
        "limitations": [
            "Pre-2014 estimates are retrospective backcasts, not observed tracking values.",
            "Official assisted-make share is conditional on made field goals and is not the target.",
            "Modern passer IDs are parsed from official AST tags and unresolved ambiguities are dropped.",
            "The 1997-2000 seasons inherit the earliest complete 1997-2001 historical fit.",
            "All 1997-2013 rZTOV values are clearly labeled research-only box estimates.",
        ],
        "outputs": {
            "role_board": str(board_path),
            "historical_rztov": str(historical_ledger_path),
            "all_era_diagnostic": str(all_ledger_path),
            "observed_modern_reference": str(observed_reference_path),
            "modern_price_comparison": str(price_comparison_path),
            "modern_player_overlap": str(player_overlap_path),
            "modern_overlap_summary": str(overlap_summary_path),
            "validation": str(args.output_dir / "chronological_model_validation.csv"),
            "predictions": str(args.output_dir / "chronological_selected_predictions.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--last-season", type=int, default=LAST_SEASON)
    parser.add_argument("--observed-roles", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=100,
        help="player-cluster bootstrap draws per historical five-season price window",
    )
    return parser.parse_args()


def main() -> int:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

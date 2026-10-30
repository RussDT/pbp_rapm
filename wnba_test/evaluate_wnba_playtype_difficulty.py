#!/usr/bin/env python3
"""Test whether the WNBA PBP playtype-difficulty proxy carries shooting signal.

Two questions are answered on the same player-season panel.

1.  Contemporaneous: does prior-priced shot-mix difficulty explain observed
    true shooting in the same season?  A difficulty measure that does not is
    not measuring difficulty.
2.  Forward: does adding current-season difficulty to a current-efficiency
    baseline improve prediction of the *next* season's true shooting?

The forward test is the one that matters and it is deliberately hard: shot
role is highly persistent, so a difficulty adjustment applied to this season
partly cancels against next season's own role.  The candidate that isolates
the mechanism therefore also reports the role-projected variant, which credits
difficulty only through next season's projected role.

Every model is scored with leave-one-season-out ridge on identical rows,
weighted by next-season shooting volume, with player-clustered bootstrap
intervals.  Seasons are development data unless explicitly held out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PBP_DIR = ROOT / "nba_pipeline" / "external" / "gabriel_wnba" / "pbp_data"
DEFAULT_FEATURE_DIR = ROOT / "wnba_test" / "Results" / "playtype_difficulty"
RIDGE_ALPHAS = np.logspace(-2, 3, 11)
BOOTSTRAP_DRAWS = 2000


def build_box_shooting(pbp_dir: Path, season: int) -> pd.DataFrame:
    """Player-season shooting totals straight from regular-season PBP."""
    paths = sorted(pbp_dir.glob(f"102{season % 100:02d}*.csv"))
    if not paths:
        raise FileNotFoundError(f"No regular-season play-by-play for {season}")
    frames = []
    for path in paths:
        frames.append(
            pd.read_csv(
                path,
                low_memory=False,
                usecols=[
                    "EVENTMSGTYPE",
                    "HOMEDESCRIPTION",
                    "VISITORDESCRIPTION",
                    "PLAYER1_ID",
                    "PLAYER1_NAME",
                ],
            )
        )
    events = pd.concat(frames, ignore_index=True)
    events = events[events["EVENTMSGTYPE"].isin([1, 2, 3])]
    description = (
        events["HOMEDESCRIPTION"]
        .fillna("")
        .astype(str)
        .str.cat(events["VISITORDESCRIPTION"].fillna("").astype(str), sep=" ")
    )
    is_three = description.str.contains("3pt", case=False, regex=False)
    made = events["EVENTMSGTYPE"].eq(1)
    is_ft = events["EVENTMSGTYPE"].eq(3)
    is_fga = events["EVENTMSGTYPE"].isin([1, 2])
    # Free-throw rows carry EVENTMSGTYPE 3 for both results; misses are the
    # rows whose description opens with MISS.
    ft_made = is_ft & ~description.str.strip().str.upper().str.startswith("MISS")
    panel = pd.DataFrame(
        {
            "player_id": pd.to_numeric(events["PLAYER1_ID"], errors="coerce"),
            "player_name": events["PLAYER1_NAME"].fillna("").astype(str),
            "fga": is_fga.astype(float),
            "fta": is_ft.astype(float),
            "fg3a": (is_fga & is_three).astype(float),
            "points": np.where(
                made, np.where(is_three, 3.0, 2.0), np.where(ft_made, 1.0, 0.0)
            ),
        }
    )
    panel = panel[panel["player_id"].gt(0)]
    out = (
        panel.groupby("player_id", sort=False)
        .agg(
            player_name=("player_name", "last"),
            fga=("fga", "sum"),
            fta=("fta", "sum"),
            fg3a=("fg3a", "sum"),
            points=("points", "sum"),
        )
        .reset_index()
    )
    out["season"] = int(season)
    out["tsa"] = out["fga"] + 0.44 * out["fta"]
    out = out[out["tsa"] > 0].copy()
    out["ts_pct"] = 100.0 * out["points"] / (2.0 * out["tsa"])
    out["fg3a_rate"] = out["fg3a"] / out["fga"].clip(lower=1.0)
    out["fta_rate"] = out["fta"] / out["fga"].clip(lower=1.0)
    return out


def build_panel(
    pbp_dir: Path,
    feature_path: Path,
    *,
    first_season: int,
    last_season: int,
    min_feature_fga: float,
    min_target_tsa: float,
) -> pd.DataFrame:
    """Join difficulty, current shooting, and next-season shooting."""
    difficulty = pd.read_csv(feature_path)
    box = pd.concat(
        [
            build_box_shooting(pbp_dir, season)
            for season in range(first_season, last_season + 2)
        ],
        ignore_index=True,
    )
    current = box.rename(columns={"season": "feature_season"})
    future = box[["player_id", "season", "ts_pct", "tsa"]].rename(
        columns={
            "season": "feature_season",
            "ts_pct": "future_ts_pct",
            "tsa": "future_tsa",
        }
    )
    future["feature_season"] = future["feature_season"] - 1
    panel = (
        difficulty.drop(columns=["player_name"])
        .merge(current.drop(columns=["player_name"]), on=["player_id", "feature_season"], how="inner")
        .merge(future, on=["player_id", "feature_season"], how="inner")
    )
    # Next season's own role, used only for the role-projected diagnostic.
    next_role = difficulty[["player_id", "feature_season", "playtype_diff"]].rename(
        columns={"playtype_diff": "future_playtype_diff"}
    )
    next_role["feature_season"] = next_role["feature_season"] - 1
    panel = panel.merge(next_role, on=["player_id", "feature_season"], how="left")
    panel = panel.merge(
        difficulty[["player_id", "feature_season", "player_name"]],
        on=["player_id", "feature_season"],
        how="left",
    )
    panel = panel[
        panel["feature_season"].between(first_season, last_season)
        & panel["pbp_fga"].ge(min_feature_fga)
        & panel["future_tsa"].ge(min_target_tsa)
    ]
    return panel.reset_index(drop=True)


def _weighted_rmse(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average((actual - predicted) ** 2, weights=weights)))


def leave_one_season_out(
    panel: pd.DataFrame, features: list[str], target: str, weight: str
) -> np.ndarray:
    """Return out-of-fold predictions with the season held out."""
    predictions = np.full(len(panel), np.nan)
    for season in sorted(panel["feature_season"].unique()):
        train = panel["feature_season"].ne(season).to_numpy()
        test = ~train
        scaler = StandardScaler().fit(panel.loc[train, features])
        best_alpha, best_score = None, np.inf
        # Inner selection uses only training seasons.
        inner_seasons = sorted(panel.loc[train, "feature_season"].unique())
        for alpha in RIDGE_ALPHAS:
            errors = []
            for inner in inner_seasons:
                inner_train = train & panel["feature_season"].ne(inner).to_numpy()
                inner_test = train & panel["feature_season"].eq(inner).to_numpy()
                if inner_test.sum() < 10:
                    continue
                model = Ridge(alpha=alpha).fit(
                    scaler.transform(panel.loc[inner_train, features]),
                    panel.loc[inner_train, target],
                    sample_weight=panel.loc[inner_train, weight],
                )
                errors.append(
                    _weighted_rmse(
                        panel.loc[inner_test, target].to_numpy(),
                        model.predict(scaler.transform(panel.loc[inner_test, features])),
                        panel.loc[inner_test, weight].to_numpy(),
                    )
                )
            score = float(np.mean(errors)) if errors else np.inf
            if score < best_score:
                best_alpha, best_score = alpha, score
        model = Ridge(alpha=best_alpha).fit(
            scaler.transform(panel.loc[train, features]),
            panel.loc[train, target],
            sample_weight=panel.loc[train, weight],
        )
        predictions[test] = model.predict(scaler.transform(panel.loc[test, features]))
    return predictions


def clustered_bootstrap(
    panel: pd.DataFrame,
    base: np.ndarray,
    challenger: np.ndarray,
    target: str,
    weight: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 20260819,
) -> dict[str, float]:
    """Player-clustered bootstrap of the RMSE change (challenger minus base)."""
    rng = np.random.default_rng(seed)
    actual = panel[target].to_numpy()
    weights = panel[weight].to_numpy()
    players = panel["player_id"].to_numpy()
    unique = np.unique(players)
    index_by_player = {player: np.flatnonzero(players == player) for player in unique}
    changes = []
    for _ in range(draws):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_player[player] for player in drawn])
        changes.append(
            _weighted_rmse(actual[rows], challenger[rows], weights[rows])
            - _weighted_rmse(actual[rows], base[rows], weights[rows])
        )
    changes = np.asarray(changes)
    return {
        "rmse_change": _weighted_rmse(actual, challenger, weights)
        - _weighted_rmse(actual, base, weights),
        "ci_low": float(np.percentile(changes, 2.5)),
        "ci_high": float(np.percentile(changes, 97.5)),
        "p_improves": float((changes < 0).mean()),
    }


BASE_FEATURES = ["ts_pct", "tsa", "fg3a_rate", "fta_rate"]
ADJUSTED_FEATURES = ["adjusted_ts_pct", "tsa", "fg3a_rate", "fta_rate"]


def calibration(panel: pd.DataFrame) -> pd.DataFrame:
    """Is the proxy the right *size*, not merely the right ordering?

    ``playtype_diff`` is denominated in TS percentage points and signed so that
    a positive value means a harder-than-average diet.  If the scale is honest,
    then regressing next-season TS on next-season role should return a slope of
    -1.00: one point of extra diet difficulty should cost exactly one point of
    observed TS.  A slope near zero would mean the proxy only ranks; a slope
    far from -1 would mean it needs rescaling before it can be added to rTS.
    """
    rows = []
    specifications = {
        "future_role_slope": (
            ["adjusted_ts_pct", "tsa", "fg3a_rate", "fta_rate", "future_playtype_diff"],
            "future_playtype_diff",
        ),
        "current_role_slope": (
            ["ts_pct", "tsa", "fg3a_rate", "fta_rate", "playtype_diff"],
            "playtype_diff",
        ),
    }
    rng = np.random.default_rng(20260819)
    players = panel["player_id"].to_numpy()
    unique = np.unique(players)
    index_by_player = {player: np.flatnonzero(players == player) for player in unique}
    for name, (features, focus) in specifications.items():
        design = panel[features].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(design)), design])
        target = panel["future_ts_pct"].to_numpy(dtype=float)
        weights = panel["future_tsa"].to_numpy(dtype=float)
        position = features.index(focus) + 1

        def _fit(rows_index: np.ndarray) -> float:
            sqrt_w = np.sqrt(weights[rows_index])[:, None]
            coefficients, *_ = np.linalg.lstsq(
                design[rows_index] * sqrt_w,
                target[rows_index] * sqrt_w[:, 0],
                rcond=None,
            )
            return float(coefficients[position])

        point = _fit(np.arange(len(panel)))
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            drawn = rng.choice(unique, size=len(unique), replace=True)
            draws.append(
                _fit(np.concatenate([index_by_player[player] for player in drawn]))
            )
        draws = np.asarray(draws)
        rows.append(
            {
                "specification": name,
                "focus": focus,
                "slope": point,
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def run(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    models = {
        "box_baseline": BASE_FEATURES,
        "imposed_adjustment": ADJUSTED_FEATURES,
        "plus_difficulty": BASE_FEATURES + ["playtype_diff"],
        "plus_difficulty_and_future_role": BASE_FEATURES
        + ["playtype_diff", "future_playtype_diff"],
    }
    scored = panel.dropna(subset=["future_playtype_diff"]).reset_index(drop=True)
    scored["adjusted_ts_pct"] = scored["ts_pct"] + scored["playtype_diff"]
    predictions = {
        name: leave_one_season_out(scored, features, "future_ts_pct", "future_tsa")
        for name, features in models.items()
    }
    metrics = pd.DataFrame(
        [
            {
                "model": name,
                "rows": len(scored),
                "rmse": _weighted_rmse(
                    scored["future_ts_pct"].to_numpy(),
                    prediction,
                    scored["future_tsa"].to_numpy(),
                ),
            }
            for name, prediction in predictions.items()
        ]
    )
    comparisons = pd.DataFrame(
        [
            {"comparison": f"{name} vs box_baseline"}
            | clustered_bootstrap(
                scored,
                predictions["box_baseline"],
                predictions[name],
                "future_ts_pct",
                "future_tsa",
            )
            for name in models
            if name != "box_baseline"
        ]
    )
    per_season = pd.DataFrame(
        [
            {
                "feature_season": int(season),
                "rows": int(mask.sum()),
                **{
                    name: _weighted_rmse(
                        scored.loc[mask, "future_ts_pct"].to_numpy(),
                        prediction[mask.to_numpy()],
                        scored.loc[mask, "future_tsa"].to_numpy(),
                    )
                    for name, prediction in predictions.items()
                },
            }
            for season in sorted(scored["feature_season"].unique())
            for mask in [scored["feature_season"].eq(season)]
        ]
    )
    # Contemporaneous check: difficulty against same-season observed TS.
    contemporaneous = []
    for season, group in panel.groupby("feature_season"):
        if len(group) < 20:
            continue
        contemporaneous.append(
            {
                "feature_season": int(season),
                "rows": len(group),
                "corr_difficulty_vs_ts": float(
                    np.corrcoef(group["playtype_diff"], group["ts_pct"])[0, 1]
                ),
            }
        )
    return {
        "forward_metrics": metrics,
        "scale_calibration": calibration(scored),
        "forward_comparisons": comparisons,
        "forward_per_season": per_season,
        "contemporaneous": pd.DataFrame(contemporaneous),
        "panel": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp-dir", type=Path, default=DEFAULT_PBP_DIR)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=DEFAULT_FEATURE_DIR / "wnba_playtype_player_season.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_FEATURE_DIR / "evaluation")
    parser.add_argument("--first-season", type=int, default=2019)
    parser.add_argument("--last-season", type=int, default=2024)
    parser.add_argument("--min-feature-fga", type=float, default=100.0)
    parser.add_argument("--min-target-tsa", type=float, default=100.0)
    args = parser.parse_args()

    panel = build_panel(
        args.pbp_dir,
        args.feature_path,
        first_season=args.first_season,
        last_season=args.last_season,
        min_feature_fga=args.min_feature_fga,
        min_target_tsa=args.min_target_tsa,
    )
    artifacts = run(panel)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in artifacts.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
        if name != "panel":
            print(f"\n=== {name} ===")
            print(frame.round(4).to_string(index=False))
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "feature_seasons": [args.first_season, args.last_season],
                "target_seasons": [args.first_season + 1, args.last_season + 1],
                "min_feature_fga": args.min_feature_fga,
                "min_target_tsa": args.min_target_tsa,
                "rows": int(len(artifacts["panel"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test a scalar Title Lift over Contract challenger against the placement model.

The challenger compresses the production model into two one-dimensional curves:

    salary share -> expected salary-peer EPM
    player EPM - expected salary-peer EPM -> title lift

It is a diagnostic surrogate.  The title curve is trained on the full model's
output, so close agreement means the placement machinery can be summarized by
the scalar shortcut; it is not independent championship-model validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class MonotoneCurve:
    """Small public prediction wrapper around an isotonic regression."""

    model: IsotonicRegression

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(values, dtype=float)), dtype=float)


@dataclass(frozen=True)
class ZeroAnchoredCurve:
    """Monotone curve expressed as lift relative to zero input."""

    model: IsotonicRegression
    zero_value: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        predicted = np.asarray(self.model.predict(values), dtype=float) - self.zero_value
        predicted[values == 0.0] = 0.0
        return predicted


def _finite_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 2:
        raise ValueError("at least two finite observations are required")
    return left[keep], right[keep]


def fit_monotone_contract_curve(
    salary_share: np.ndarray,
    salary_peer_epm: np.ndarray,
) -> MonotoneCurve:
    """Fit the nondecreasing EPM normally occupying each salary-share slot."""

    salary_share, salary_peer_epm = _finite_pair(salary_share, salary_peer_epm)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(salary_share, salary_peer_epm)
    return MonotoneCurve(model=model)


def fit_zero_anchored_title_curve(
    surplus_epm: np.ndarray,
    title_lift: np.ndarray,
) -> ZeroAnchoredCurve:
    """Fit a nondecreasing title curve and express it relative to zero surplus."""

    surplus_epm, title_lift = _finite_pair(surplus_epm, title_lift)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(surplus_epm, title_lift)
    zero_value = float(model.predict(np.array([0.0]))[0])
    return ZeroAnchoredCurve(model=model, zero_value=zero_value)


REQUIRED_COLUMNS = (
    "season",
    "player_id",
    "salary_share",
    "own_dpm",
    "salary_peer_mean_epm",
    "salary_peer_mean_workload_share",
    "title_lift_over_contract",
)

LINEUP_OFFENSE_COLUMNS = ("O1", "O2", "O3", "O4", "O5")
LINEUP_DEFENSE_COLUMNS = ("D1", "D2", "D3", "D4", "D5")


def summarize_player_possessions(
    lineups: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Return exact player exposure, averaging offensive and defensive counts."""

    lineup_columns = (*LINEUP_OFFENSE_COLUMNS, *LINEUP_DEFENSE_COLUMNS)
    missing = sorted(set(lineup_columns) - set(lineups.columns))
    if missing:
        raise ValueError(f"lineup rows missing required columns: {', '.join(missing)}")

    def count_slots(columns: tuple[str, ...]) -> pd.Series:
        values = pd.to_numeric(
            pd.Series(lineups[list(columns)].to_numpy().ravel()),
            errors="coerce",
        ).dropna()
        return values.astype(np.int64).value_counts(sort=False)

    offense = count_slots(LINEUP_OFFENSE_COLUMNS)
    defense = count_slots(LINEUP_DEFENSE_COLUMNS)
    player_ids = offense.index.union(defense.index).astype(np.int64)
    output = pd.DataFrame({"player_id": player_ids})
    output["offensive_possessions"] = output["player_id"].map(offense).fillna(0).astype(int)
    output["defensive_possessions"] = output["player_id"].map(defense).fillna(0).astype(int)
    output["regular_season_possessions"] = (
        output["offensive_possessions"] + output["defensive_possessions"]
    ) / 2.0
    output.insert(0, "season", int(season))
    return output.sort_values("player_id", ignore_index=True)


def load_regular_season_possessions(
    processed_dir: Path,
    seasons: list[int],
) -> pd.DataFrame:
    """Load exact RS lineup exposure for the requested NBA seasons."""

    frames: list[pd.DataFrame] = []
    lineup_columns = [*LINEUP_OFFENSE_COLUMNS, *LINEUP_DEFENSE_COLUMNS]
    for season in sorted(set(int(value) for value in seasons)):
        path = processed_dir / f"RAPM{season % 100:02d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing regular-season possession file: {path}")
        lineups = pd.read_parquet(path, columns=lineup_columns)
        frames.append(summarize_player_possessions(lineups, season))
    return pd.concat(frames, ignore_index=True)


def attach_realized_surplus_points(
    scored: pd.DataFrame,
    possessions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach realized RS exposure and linear scoreboard points over contract."""

    required_scored = {"season", "player_id", "scalar_epm_surplus"}
    required_possessions = {
        "season",
        "player_id",
        "regular_season_possessions",
    }
    missing_scored = sorted(required_scored - set(scored.columns))
    missing_possessions = sorted(required_possessions - set(possessions.columns))
    if missing_scored:
        raise ValueError(f"scored rows missing required columns: {', '.join(missing_scored)}")
    if missing_possessions:
        raise ValueError(
            "possession rows missing required columns: "
            + ", ".join(missing_possessions)
        )

    output = scored.drop(
        columns=["regular_season_possessions", "realized_rs_points_over_contract"],
        errors="ignore",
    ).merge(
        possessions[list(required_possessions)],
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
    )
    output["realized_rs_points_over_contract"] = (
        pd.to_numeric(output["scalar_epm_surplus"], errors="coerce")
        * pd.to_numeric(output["regular_season_possessions"], errors="coerce")
        / 100.0
    )
    return output


def build_leave_one_season_out_challenger(frame: pd.DataFrame) -> pd.DataFrame:
    """Return held-out-season scalar predictions for every complete input row.

    Both scalar curves are refit without the season being scored.  This prevents
    a season's salary-peer outcomes or full-model lift from teaching its own
    prediction.
    """

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    work = frame.copy()
    numeric = [
        "season",
        "salary_share",
        "own_dpm",
        "salary_peer_mean_epm",
        "salary_peer_mean_workload_share",
        "title_lift_over_contract",
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    if work["season"].nunique() < 2:
        raise ValueError("at least two seasons are required for held-out scoring")

    scored_folds: list[pd.DataFrame] = []
    for season in sorted(work["season"].unique()):
        train = work.loc[work["season"] != season].copy()
        test = work.loc[work["season"] == season].copy()

        contract_curve = fit_monotone_contract_curve(
            train["salary_share"].to_numpy(),
            train["salary_peer_mean_epm"].to_numpy(),
        )
        train_contract_epm = contract_curve.predict(train["salary_share"].to_numpy())
        train_surplus_epm = train["own_dpm"].to_numpy() - train_contract_epm
        title_curve = fit_zero_anchored_title_curve(
            train_surplus_epm,
            train["title_lift_over_contract"].to_numpy(),
        )
        workload_curve = fit_monotone_contract_curve(
            train["salary_share"].to_numpy(),
            train["salary_peer_mean_workload_share"].to_numpy(),
        )
        train_expected_workload = workload_curve.predict(
            train["salary_share"].to_numpy()
        )
        train_surplus_team_rating = train_surplus_epm * train_expected_workload
        role_title_curve = fit_zero_anchored_title_curve(
            train_surplus_team_rating,
            train["title_lift_over_contract"].to_numpy(),
        )

        test["scalar_contract_epm"] = contract_curve.predict(
            test["salary_share"].to_numpy()
        )
        test["scalar_epm_surplus"] = (
            test["own_dpm"].to_numpy() - test["scalar_contract_epm"].to_numpy()
        )
        test["scalar_title_lift"] = title_curve.predict(
            test["scalar_epm_surplus"].to_numpy()
        )
        test["scalar_title_lift_unanchored"] = (
            test["scalar_title_lift"] + title_curve.zero_value
        )
        test["scalar_residual"] = (
            test["title_lift_over_contract"] - test["scalar_title_lift"]
        )
        test["scalar_expected_workload"] = workload_curve.predict(
            test["salary_share"].to_numpy()
        )
        test["scalar_surplus_team_rating"] = (
            test["scalar_epm_surplus"] * test["scalar_expected_workload"]
        )
        test["role_adjusted_title_lift"] = role_title_curve.predict(
            test["scalar_surplus_team_rating"].to_numpy()
        )
        test["role_adjusted_title_lift_unanchored"] = (
            test["role_adjusted_title_lift"] + role_title_curve.zero_value
        )
        test["role_adjusted_residual"] = (
            test["title_lift_over_contract"] - test["role_adjusted_title_lift"]
        )
        scored_folds.append(test)

    return pd.concat(scored_folds, ignore_index=True).sort_values(
        ["season", "player_id"], kind="stable", ignore_index=True
    )


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Evaluate a title-lift prediction, reporting probability errors in pp."""

    actual, predicted = _finite_pair(actual, predicted)
    error = actual - predicted
    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)
    return {
        "n": int(len(actual)),
        "pearson": float(actual_series.corr(predicted_series, method="pearson")),
        "spearman": float(actual_series.corr(predicted_series, method="spearman")),
        "mae_pp": round(float(np.mean(np.abs(error)) * 100.0), 12),
        "rmse_pp": round(float(np.sqrt(np.mean(np.square(error))) * 100.0), 12),
        "bias_actual_minus_predicted_pp": round(float(np.mean(error) * 100.0), 12),
    }


def build_career_table(scored: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["player_id"]
    if "player_name" in scored.columns:
        group_columns.append("player_name")
    career = scored.groupby(group_columns, as_index=False).agg(
        covered_seasons=("season", "size"),
        full_placement_titles=("title_lift_over_contract", "sum"),
        epm_only_titles=("scalar_title_lift", "sum"),
        role_adjusted_titles=("role_adjusted_title_lift", "sum"),
        role_adjusted_unanchored_titles=(
            "role_adjusted_title_lift_unanchored",
            "sum",
        ),
        surplus_epm_season_sum=("scalar_epm_surplus", "sum"),
        regular_season_possessions=("regular_season_possessions", "sum"),
        realized_rs_points_over_contract=(
            "realized_rs_points_over_contract",
            "sum",
        ),
    )
    career["full_minus_role_adjusted"] = (
        career["full_placement_titles"] - career["role_adjusted_titles"]
    )
    career["full_rank"] = career["full_placement_titles"].rank(
        method="min", ascending=False
    ).astype(int)
    career["role_adjusted_rank"] = career["role_adjusted_titles"].rank(
        method="min", ascending=False
    ).astype(int)
    career["realized_rs_points_rank"] = career[
        "realized_rs_points_over_contract"
    ].rank(method="min", ascending=False).astype(int)
    return career.sort_values("full_placement_titles", ascending=False, ignore_index=True)


def _format_career_markdown(career: pd.DataFrame, count: int = 15) -> str:
    lines = [
        "| Full rank | Player | Seasons | Full placements | EPM only | EPM × role | Realized RS points over contract | Points rank |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in career.head(count).itertuples(index=False):
        name = getattr(row, "player_name", str(row.player_id))
        lines.append(
            f"| {row.full_rank} | {name} | {row.covered_seasons} | "
            f"{row.full_placement_titles:.3f} | {row.epm_only_titles:.3f} | "
            f"{row.role_adjusted_titles:.3f} | "
            f"{row.realized_rs_points_over_contract:.1f} | "
            f"{row.realized_rs_points_rank} |"
        )
    return "\n".join(lines)


def build_report(scored: pd.DataFrame, career: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Build the concise human and machine-readable challenger receipts."""

    actual = scored["title_lift_over_contract"].to_numpy()
    models = {
        "epm_only_zero_anchored": evaluate_predictions(
            actual, scored["scalar_title_lift"].to_numpy()
        ),
        "role_adjusted_zero_anchored": evaluate_predictions(
            actual, scored["role_adjusted_title_lift"].to_numpy()
        ),
        "role_adjusted_unanchored_diagnostic": evaluate_predictions(
            actual, scored["role_adjusted_title_lift_unanchored"].to_numpy()
        ),
    }
    summary: dict[str, Any] = {
        "status": "research diagnostic; not a production replacement",
        "season_min": int(scored["season"].min()),
        "season_max": int(scored["season"].max()),
        "season_count": int(scored["season"].nunique()),
        "player_season_count": int(len(scored)),
        "models": models,
        "realized_points_over_contract": {
            "valuation_frame": "realized regular season",
            "impact_unit": "EPM points per 100 possessions",
            "exposure": "average of exact offensive and defensive lineup possessions",
            "identity": "scalar_epm_surplus * regular_season_possessions / 100",
        },
    }

    metric_lines = [
        "| Challenger | Pearson | Spearman | MAE (pp) | RMSE (pp) | Bias: full − shortcut (pp) |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "epm_only_zero_anchored": "EPM surplus only, zero anchored",
        "role_adjusted_zero_anchored": "EPM surplus × expected role, zero anchored",
        "role_adjusted_unanchored_diagnostic": "Role-adjusted, fitted intercept (diagnostic)",
    }
    for key, metrics in models.items():
        metric_lines.append(
            f"| {labels[key]} | {metrics['pearson']:.3f} | {metrics['spearman']:.3f} | "
            f"{metrics['mae_pp']:.3f} | {metrics['rmse_pp']:.3f} | "
            f"{metrics['bias_actual_minus_predicted_pp']:+.3f} |"
        )

    report = "\n".join(
        [
            "# Scalar Title Lift over Contract challenger",
            "",
            "Status: research diagnostic; not a production replacement.",
            "",
            "Every prediction is leave-one-season-out. The contract curve learns salary share → expected salary-peer EPM on other seasons. The literal shortcut sends EPM surplus through one monotone, zero-anchored title-lift curve. The role-adjusted shortcut first multiplies surplus EPM by the expected playoff workload attached to that salary slot.",
            "",
            "The curves are trained against the full placement model, so this tests compressibility—not independent championship accuracy.",
            "",
            "Realized regular-season points over contract are a separate linear accounting field: held-out EPM surplus × exact regular-season on-court possessions / 100. Offensive and defensive possession counts are averaged so the same team possession is not counted twice. The old sum of annual EPM-surplus rates is retained only as `surplus_epm_season_sum`; it is not a scoreboard-point total.",
            "",
            "## Out-of-season fit",
            "",
            *metric_lines,
            "",
            "The fitted-intercept row is diagnostic only: unlike the conceptual shortcut, it does not force zero expected title lift at zero estimated surplus.",
            "",
            "## Career comparison",
            "",
            _format_career_markdown(career),
            "",
        ]
    )
    return report, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("nba_pipeline/results/championship_help/player_season_estimates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nba_pipeline/results/contract_title_curve"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("nba_pipeline/processed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    scored = build_leave_one_season_out_challenger(frame)
    possessions = load_regular_season_possessions(
        args.processed_dir,
        scored["season"].astype(int).unique().tolist(),
    )
    scored = attach_realized_surplus_points(scored, possessions)
    career = build_career_table(scored)
    report, summary = build_report(scored, career)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "player_season_oof.csv", index=False)
    career.to_csv(args.output_dir / "career_comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

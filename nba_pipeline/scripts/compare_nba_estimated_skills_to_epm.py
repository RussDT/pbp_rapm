"""Compare NBA Kalman estimated skills with date-matched EPM E-Skills.

For 12 cached snapshot dates in the 2025-26 season, the script joins each
provider prediction and our latest Kalman posterior strictly before the
requested date to the player's pooled remaining-season realized rate. It
reports exposure-weighted player correlations, mean-aligns each predictor to
the common target, and uses a paired player bootstrap for the difference in
level-adjusted weighted MSE.

Only aggregate benchmark outputs are written. Raw provider values remain in
the gitignored cache and are never copied to the result artifacts.

Run:
    .venv/bin/python nba_pipeline/scripts/compare_nba_estimated_skills_to_epm.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
ESTIMATED_DIR = PIPELINE_ROOT / "results" / "estimated_skills"
PREDICTIONS_PATH = ESTIMATED_DIR / "predictions.parquet"
LINES_PATH = ESTIMATED_DIR / "player_game_lines.parquet"
EPM_DIR = PIPELINE_ROOT / "cache" / "epm"
OUTPUT_DIR = ESTIMATED_DIR / "epm_benchmark"
SEASON = 2026
DEFAULT_DRAWS = 2000
DEFAULT_SEED = 20260903

SNAPSHOT_DATES = (
    "2025-10-15",
    "2025-11-01",
    "2025-11-15",
    "2025-12-01",
    "2025-12-15",
    "2026-01-01",
    "2026-01-15",
    "2026-02-01",
    "2026-02-15",
    "2026-03-01",
    "2026-03-15",
    "2026-04-01",
)


@dataclass(frozen=True)
class SkillMap:
    skill: str
    numerator: str
    denominator: str
    epm_column: str
    epm_scale: float = 1.0


SKILLS = (
    SkillMap("rim_pct", "rim_fgm", "rim_fga", "p_fgpct_rim"),
    SkillMap("mid_pct", "mid_fgm", "mid_fga", "p_fgpct_mid"),
    SkillMap("three_pct", "three_fgm", "three_fga", "p_fg3pct"),
    SkillMap("ft_pct", "ftm", "fta", "p_ftpct"),
    SkillMap("rim_fga_rate", "rim_fga", "off_poss", "p_fga_rim_100", 0.01),
    SkillMap("mid_fga_rate", "mid_fga", "off_poss", "p_fga_mid_100", 0.01),
    SkillMap("three_fga_rate", "three_fga", "off_poss", "p_fg3a_100", 0.01),
    SkillMap("fta_rate", "fta", "off_poss", "p_fta_100", 0.01),
    SkillMap("pts_rate", "pts", "off_poss", "p_pts_100", 0.01),
    SkillMap("ast_rate", "ast", "off_poss", "p_ast_100", 0.01),
    SkillMap("tov_rate", "tov", "off_poss", "p_tov_100", 0.01),
    SkillMap("oreb_rate", "oreb", "off_poss", "p_orb_100", 0.01),
    SkillMap("usg_pct", "usg_events", "off_poss", "p_usg"),
    SkillMap("dreb_rate", "dreb", "def_poss", "p_drb_100", 0.01),
    SkillMap("stl_rate", "stl", "def_poss", "p_stl_100", 0.01),
    SkillMap("blk_rate", "blk", "def_poss", "p_blk_100", 0.01),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_correlation(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    total = float(weight.sum())
    if total <= 0:
        return float("nan")
    mean_x = float(np.sum(weight * x) / total)
    mean_y = float(np.sum(weight * y) / total)
    dx = x - mean_x
    dy = y - mean_y
    var_x = float(np.sum(weight * dx * dx))
    var_y = float(np.sum(weight * dy * dy))
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return float(np.sum(weight * dx * dy) / np.sqrt(var_x * var_y))


def paired_level_adjusted_bootstrap(
    ours: np.ndarray,
    epm: np.ndarray,
    actual: np.ndarray,
    weight: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    """Return ours WMSE, EPM WMSE, and paired ours-minus-EPM interval.

    Each predictor receives one weighted intercept shift on the full common
    player sample. The adjusted per-player squared errors are then kept fixed
    inside the paired bootstrap, matching the original benchmark protocol.
    """
    total = float(weight.sum())
    actual_mean = float(np.sum(weight * actual) / total)
    ours_adjusted = ours + actual_mean - float(np.sum(weight * ours) / total)
    epm_adjusted = epm + actual_mean - float(np.sum(weight * epm) / total)
    ours_error = (ours_adjusted - actual) ** 2
    epm_error = (epm_adjusted - actual) ** 2
    ours_wmse = float(np.sum(weight * ours_error) / total)
    epm_wmse = float(np.sum(weight * epm_error) / total)
    delta_error = ours_error - epm_error
    point = ours_wmse - epm_wmse
    if draws <= 0:
        return ours_wmse, epm_wmse, point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_players = len(actual)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        take = rng.integers(0, n_players, size=n_players)
        sample_weight = weight[take]
        samples[draw] = float(
            np.sum(sample_weight * delta_error[take]) / np.sum(sample_weight)
        )
    low, high = np.nanquantile(samples, [0.025, 0.975])
    return ours_wmse, epm_wmse, point, float(low), float(high)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    missing = [path for path in (PREDICTIONS_PATH, LINES_PATH) if not path.exists()]
    snapshots = {
        date: EPM_DIR / f"epm_all_{date}_go0.csv" for date in SNAPSHOT_DATES
    }
    missing.extend(path for path in snapshots.values() if not path.exists())
    if missing:
        raise FileNotFoundError(f"missing benchmark inputs: {[str(p) for p in missing]}")

    predictions = pd.read_parquet(
        PREDICTIONS_PATH,
        columns=["player_id", "game_date", "game_id", "skill", "post_rate"],
    )
    predictions["game_date"] = pd.to_datetime(predictions["game_date"])
    predictions["player_id"] = pd.to_numeric(
        predictions["player_id"], errors="raise"
    ).astype("int64")
    predictions = predictions.sort_values(
        ["skill", "player_id", "game_date", "game_id"], kind="mergesort"
    )

    needed = {
        "player_id",
        "game_date",
        "season",
        "fga",
        "fta",
        "tov",
        *(skill.numerator for skill in SKILLS if skill.numerator != "usg_events"),
        *(skill.denominator for skill in SKILLS),
    }
    lines = pd.read_parquet(LINES_PATH, columns=sorted(needed))
    lines["game_date"] = pd.to_datetime(lines["game_date"])
    lines["player_id"] = pd.to_numeric(lines["player_id"], errors="raise").astype(
        "int64"
    )
    lines["usg_events"] = lines["fga"] + 0.44 * lines["fta"] + lines["tov"]
    lines = lines[(lines["season"] == SEASON) & lines["game_date"].notna()].copy()
    return predictions, lines, snapshots


def latest_states(predictions: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    prior = predictions[predictions["game_date"] < date]
    latest = prior.groupby(["skill", "player_id"], sort=False).tail(1)
    return latest[["skill", "player_id", "post_rate"]].rename(
        columns={"post_rate": "ours"}
    )


def remaining_actuals(
    lines: pd.DataFrame, date: pd.Timestamp, skill: SkillMap
) -> pd.DataFrame:
    remaining = lines[(lines["game_date"] >= date) & (lines[skill.denominator] > 0)]
    actual = (
        remaining.groupby("player_id", as_index=False)
        .agg(numerator=(skill.numerator, "sum"), exposure=(skill.denominator, "sum"))
    )
    actual = actual[actual["exposure"] > 0].copy()
    actual["actual"] = actual["numerator"] / actual["exposure"]
    return actual[["player_id", "actual", "exposure"]]


def load_epm_snapshot(path: Path, date: str) -> pd.DataFrame:
    columns = ["requested_date", "player_id", *(skill.epm_column for skill in SKILLS)]
    snapshot = pd.read_csv(path, usecols=columns)
    requested = set(snapshot["requested_date"].astype(str).unique())
    if requested != {date}:
        raise ValueError(f"{path.name}: requested_date values {sorted(requested)}")
    snapshot["player_id"] = pd.to_numeric(snapshot["player_id"], errors="raise").astype(
        "int64"
    )
    if snapshot["player_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate player_id values")
    return snapshot


def run_benchmark(draws: int, seed: int) -> tuple[pd.DataFrame, dict]:
    predictions, lines, snapshots = load_inputs()
    result_rows: list[dict[str, object]] = []
    snapshot_hashes: dict[str, str] = {}

    for date_idx, date_text in enumerate(SNAPSHOT_DATES):
        date = pd.Timestamp(date_text)
        path = snapshots[date_text]
        snapshot_hashes[date_text] = sha256(path)
        epm_snapshot = load_epm_snapshot(path, date_text)
        states = latest_states(predictions, date)

        for skill_idx, skill in enumerate(SKILLS):
            ours = states[states["skill"] == skill.skill][["player_id", "ours"]]
            epm = epm_snapshot[["player_id", skill.epm_column]].rename(
                columns={skill.epm_column: "epm"}
            )
            epm["epm"] = pd.to_numeric(epm["epm"], errors="coerce") * skill.epm_scale
            actual = remaining_actuals(lines, date, skill)
            common = actual.merge(ours, on="player_id", how="inner").merge(
                epm, on="player_id", how="inner"
            )
            common = common.dropna(subset=["actual", "ours", "epm", "exposure"])
            if len(common) < 30:
                raise ValueError(
                    f"{date_text} {skill.skill}: only {len(common)} common players"
                )

            actual_values = common["actual"].to_numpy(dtype=float)
            ours_values = common["ours"].to_numpy(dtype=float)
            epm_values = common["epm"].to_numpy(dtype=float)
            exposure = common["exposure"].to_numpy(dtype=float)
            ours_r = weighted_correlation(ours_values, actual_values, exposure)
            epm_r = weighted_correlation(epm_values, actual_values, exposure)
            ours_wmse, epm_wmse, delta_wmse, low, high = paired_level_adjusted_bootstrap(
                ours_values,
                epm_values,
                actual_values,
                exposure,
                draws,
                seed + 100 * date_idx + skill_idx,
            )
            verdict = "tie"
            if high < 0:
                verdict = "ours"
            elif low > 0:
                verdict = "epm"
            result_rows.append(
                {
                    "requested_date": date_text,
                    "skill": skill.skill,
                    "epm_column": skill.epm_column,
                    "n_players": int(len(common)),
                    "target_exposure": float(exposure.sum()),
                    "ours_r": ours_r,
                    "epm_r": epm_r,
                    "delta_r_ours_minus_epm": ours_r - epm_r,
                    "ours_level_adjusted_wmse": ours_wmse,
                    "epm_level_adjusted_wmse": epm_wmse,
                    "delta_wmse_ours_minus_epm": delta_wmse,
                    "delta_wmse_ci_low": low,
                    "delta_wmse_ci_high": high,
                    "verdict": verdict,
                }
            )

    results = pd.DataFrame(result_rows)
    date_summary = (
        results.groupby("requested_date", as_index=False)
        .agg(
            mean_ours_r=("ours_r", "mean"),
            mean_epm_r=("epm_r", "mean"),
            mean_delta_r=("delta_r_ours_minus_epm", "mean"),
            min_players=("n_players", "min"),
            max_players=("n_players", "max"),
        )
    )
    counts = results["verdict"].value_counts().reindex(
        ["tie", "ours", "epm"], fill_value=0
    )
    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "season": SEASON,
        "snapshot_dates": list(SNAPSHOT_DATES),
        "n_skill_date_cells": int(len(results)),
        "n_skills": int(len(SKILLS)),
        "bootstrap": {
            "unit": "player",
            "draws": draws,
            "seed": seed,
            "level_adjustment": (
                "weighted intercept shift to the common actual mean, fixed "
                "before player resampling"
            ),
            "interval": (
                "paired percentile 95% for level-adjusted weighted-MSE "
                "difference, ours minus EPM; negative favors ours"
            ),
        },
        "verdict_counts": {key: int(value) for key, value in counts.items()},
        "date_summary": date_summary.to_dict(orient="records"),
        "input_hashes": {
            "predictions": sha256(PREDICTIONS_PATH),
            "player_game_lines": sha256(LINES_PATH),
            "epm_snapshots": snapshot_hashes,
        },
        "fences": [
            "benchmark-only; EPM values are not model features",
            "provider date-D predictions exclude games on D",
            "our state uses games strictly before D",
            "actual target uses games on or after D",
            "2025-26 is spent evidence and cannot promote a model",
        ],
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-write", action="store_true", help="Print results without writing artifacts"
    )
    args = parser.parse_args()
    if args.bootstrap_draws < 0:
        raise SystemExit("--bootstrap-draws must be nonnegative")

    results, summary = run_benchmark(args.bootstrap_draws, args.seed)
    date_summary = pd.DataFrame(summary["date_summary"])
    print("EPM E-Skills vs NBA Kalman estimated skills — 2025-26 rerun")
    print(date_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nverdicts: {summary['verdict_counts']}")

    skill_summary = (
        results.groupby("skill", as_index=False)
        .agg(
            mean_ours_r=("ours_r", "mean"),
            mean_epm_r=("epm_r", "mean"),
            mean_delta_r=("delta_r_ours_minus_epm", "mean"),
            ours_cells=("verdict", lambda values: int((values == "ours").sum())),
            epm_cells=("verdict", lambda values: int((values == "epm").sum())),
            ties=("verdict", lambda values: int((values == "tie").sum())),
        )
        .sort_values("mean_delta_r", ascending=False)
    )
    print("\nmean by skill across snapshot dates")
    print(skill_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if not args.no_write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUTPUT_DIR / "skill_date_results.csv", index=False)
        skill_summary.to_csv(OUTPUT_DIR / "skill_summary.csv", index=False)
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote aggregate artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

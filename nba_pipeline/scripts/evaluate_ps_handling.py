#!/usr/bin/env python3
"""Forward-prediction gate for playoff-handling candidates in rapm.py.

Compares six ways of handling playoff possessions in the pooled RAPM solve:

    rs_only             rapm.py RAPM S E RS
    all_pooled          rapm.py RAPM S E ALL
    all_se              rapm.py RAPM S E ALL --season-effects
    all_se_psprior      rapm.py RAPM S E ALL --season-effects --ps-prior
    all_se_psprioronly  rapm.py RAPM S E ALL --season-effects --ps-prior --ps-prior-stage2 ps-only
    all_se_a1500        rapm.py RAPM S E ALL --season-effects --off-alpha 1500 --def-alpha 1500
                        (shrinkage-dose control for the psprior scale expansion)

Each candidate's summed coefficient map is scored against held-out FUTURE
regular-season possessions (processed/RAPM{TT}.parquet for the test season).
Residuals are level-neutralized (both target and prediction are centered on
their test-set means) so candidates are compared purely on player-coefficient
quality: intercept/season-level differences and non-transferable season-effect
dummies do not pollute the comparison.

Metrics per (fold, candidate):
  - overall MSE of level-neutral residuals
  - paired per-game MSE delta vs the rs_only baseline (possession-weighted
    within game, then possession-weighted across games), in 1e-6 units, with
    a 95% CI from a seeded cluster bootstrap over games
  - fraction of games where the candidate strictly beats the baseline
    (ties count as non-wins, so the baseline's own win rate is 0.0)
  - slot coverage: fraction of valid test player-slots (id > 0) that have a
    coefficient in the candidate's results CSV

Bootstrap game resamples are shared across candidates within a fold (common
random numbers), so a candidate identical to the baseline yields paired
deltas of exactly 0 with CI [0, 0].

Usage:
    python evaluate_ps_handling.py                       # default folds
    python evaluate_ps_handling.py --folds "22-24:25" --force --cores 8

Solves are launched sequentially with sys.executable when a results CSV is
missing (or --force); existing CSVs are reused.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
RAPM_SCRIPT = SCRIPT_DIR / "rapm.py"

BASELINE_NAME = "rs_only"

# name -> (season_type, extra rapm.py flags, results-filename suffix).
# Suffix logic verified against build_output_suffix() in rapm.py: parts start
# with the season suffix (rs/all), then "se" for --season-effects, then
# "psprior"/"psprioronly" for --ps-prior, then "a{off}_{def}" for non-default
# alphas; default alphas 3000/3000 add no alpha token.
#
# all_se_a1500 is the shrinkage-dose control: the full-sample psprior stage 2
# expands the rating scale ~1.28x (equivalent to a weaker effective alpha), so
# a plain lower-alpha candidate is needed to tell "playoff-aware helps" apart
# from "less shrinkage helps". all_se_psprioronly avoids the expansion by
# fitting only playoff rows in stage 2.
CANDIDATES = {
    "rs_only": ("RS", [], "rs"),
    "all_pooled": ("ALL", [], "all"),
    "all_se": ("ALL", ["--season-effects"], "all_se"),
    "all_se_psprior": ("ALL", ["--season-effects", "--ps-prior"], "all_se_psprior"),
    "all_se_psprioronly": ("ALL", ["--season-effects", "--ps-prior", "--ps-prior-stage2", "ps-only"], "all_se_psprioronly"),
    "all_se_a1500": ("ALL", ["--season-effects", "--off-alpha", "1500", "--def-alpha", "1500"], "all_se_a1500_1500"),
}

OFF_SLOTS = [f"O{i}" for i in range(1, 6)]
DEF_SLOTS = [f"D{i}" for i in range(1, 6)]

OUTPUT_COLUMNS = [
    "fold",
    "candidate",
    "mse",
    "mean_paired_delta_vs_rs_only_e6",
    "ci_lo_e6",
    "ci_hi_e6",
    "game_win_rate",
    "slot_coverage",
    "n_test_rows",
    "n_test_games",
    "results_csv",
]


def expected_results_csv(start_yy: int, end_yy: int, candidate: str) -> Path:
    """Mirror rapm.py's output naming for a RAPM solve with default alphas."""
    suffix = CANDIDATES[candidate][2]
    if start_yy == end_yy:
        year_range = f"{start_yy:02d}"
    else:
        year_range = f"{start_yy:02d}_{end_yy:02d}"
    return RESULTS_DIR / f"rapm_{year_range}_{suffix}_results.csv"


def parse_folds(spec: str) -> list[dict]:
    """Parse "22-24:25,23-25:26" into fold dicts."""
    folds = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            window, test = chunk.split(":")
            start, end = window.split("-")
            fold = {
                "label": chunk,
                "start_yy": int(start),
                "end_yy": int(end),
                "test_yy": int(test),
            }
        except ValueError as exc:
            raise SystemExit(f"Bad --folds chunk {chunk!r}; expected S-E:T like 22-24:25") from exc
        folds.append(fold)
    if not folds:
        raise SystemExit("--folds parsed to zero folds")
    return folds


def ensure_solved(fold: dict, candidate: str, force: bool, cores: int | None) -> Path:
    """Reuse the candidate's results CSV or launch the solve sequentially."""
    out_csv = expected_results_csv(fold["start_yy"], fold["end_yy"], candidate)
    if out_csv.exists() and not force:
        logging.info("[%s/%s] REUSED existing results: %s", fold["label"], candidate, out_csv)
        return out_csv

    season_type, flags, _ = CANDIDATES[candidate]
    cmd = [
        sys.executable,
        str(RAPM_SCRIPT),
        "RAPM",
        str(fold["start_yy"]),
        str(fold["end_yy"]),
        season_type,
        *flags,
    ]
    if cores is not None:
        cmd += ["-j", str(cores)]
    logging.info("[%s/%s] LAUNCHING solve: %s", fold["label"], candidate, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))
    if not out_csv.exists():
        raise SystemExit(
            f"Solve finished but expected results CSV is missing: {out_csv} "
            "(naming drift between evaluate_ps_handling.py and rapm.py?)"
        )
    logging.info("[%s/%s] Solve wrote %s", fold["label"], candidate, out_csv)
    return out_csv


def load_coefficient_maps(results_csv: Path) -> tuple[pd.Series, pd.Series]:
    """Per-possession off/def coefficient maps keyed by player_id.

    Results CSVs store per-100 values; the solver's row prediction is
    X_off @ beta_off + X_def @ beta_def with beta as stored (net = off - def
    is only a reporting convention), so def enters the prediction with a
    + sign and no flip.
    """
    df = pd.read_csv(results_csv, usecols=["player_id", "off", "def"])
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(np.int64)
    df = df[df["player_id"] > 0]
    n_dupes = int(df["player_id"].duplicated().sum())
    if n_dupes:
        logging.warning("%s: %d duplicate player_id rows; keeping first", results_csv.name, n_dupes)
        df = df.drop_duplicates(subset="player_id", keep="first")
    off = pd.Series(df["off"].to_numpy(dtype=np.float64) / 100.0, index=df["player_id"])
    dfn = pd.Series(df["def"].to_numpy(dtype=np.float64) / 100.0, index=df["player_id"])
    return off, dfn


def load_test_set(test_yy: int) -> pd.DataFrame:
    """Load the future-season RS possession file (test set)."""
    path = PROCESSED_DIR / f"RAPM{test_yy:02d}.parquet"
    if not path.exists():
        raise SystemExit(f"Test parquet missing: {path}")
    cols = ["game_id"] + OFF_SLOTS + DEF_SLOTS + ["Off_Diff"]
    df = pd.read_parquet(path, columns=cols)
    n_before = len(df)
    df = df.dropna(subset=["game_id", "Off_Diff"]).reset_index(drop=True)
    if len(df) != n_before:
        logging.info("Test %02d: dropped %d rows with null game_id/Off_Diff", test_yy, n_before - len(df))
    logging.info("Test %02d: %d possessions, %d games (%s)", test_yy, len(df), df["game_id"].nunique(), path.name)
    return df


def predict_rows(test_df: pd.DataFrame, off_map: pd.Series, def_map: pd.Series) -> tuple[np.ndarray, int, int]:
    """Summed coefficient prediction per row plus slot-coverage counts."""
    pred = np.zeros(len(test_df), dtype=np.float64)
    matched_slots = 0
    valid_slots = 0
    for slots, coef_map in ((OFF_SLOTS, off_map), (DEF_SLOTS, def_map)):
        for col in slots:
            ids = pd.to_numeric(test_df[col], errors="coerce").fillna(0).astype(np.int64)
            valid = ids > 0
            contrib = ids.map(coef_map)
            matched = valid & contrib.notna()
            pred += contrib.fillna(0.0).to_numpy(dtype=np.float64)
            matched_slots += int(matched.sum())
            valid_slots += int(valid.sum())
    return pred, matched_slots, valid_slots


def evaluate_fold(
    fold_label: str,
    test_df: pd.DataFrame,
    candidate_files: dict[str, Path],
    baseline_name: str,
    n_bootstrap: int,
    seed: int,
    fold_index: int,
) -> dict[str, dict]:
    """Score every candidate on one fold. Returns per-candidate stats dicts."""
    y = test_df["Off_Diff"].to_numpy(dtype=np.float64)
    y_centered = y - y.mean()
    game_ids = test_df["game_id"]

    # One shared game order and one shared bootstrap index matrix per fold so
    # every candidate is paired on identical resamples (common random numbers).
    game_codes, game_index = pd.factorize(game_ids, sort=True)
    n_games = len(game_index)
    game_sizes = np.bincount(game_codes, minlength=n_games).astype(np.float64)
    rng = np.random.default_rng([seed, fold_index])
    boot_idx = rng.integers(0, n_games, size=(n_bootstrap, n_games))
    boot_w = game_sizes[boot_idx]
    boot_den = boot_w.sum(axis=1)

    per_game_mse = {}
    stats: dict[str, dict] = {}
    for name, csv_path in candidate_files.items():
        off_map, def_map = load_coefficient_maps(csv_path)
        pred, matched_slots, valid_slots = predict_rows(test_df, off_map, def_map)
        resid = y_centered - (pred - pred.mean())
        sq = resid * resid
        # Possession-weighted within game == per-game mean of squared residuals.
        game_sq_sums = np.bincount(game_codes, weights=sq, minlength=n_games)
        per_game_mse[name] = game_sq_sums / game_sizes
        stats[name] = {
            "mse": float(sq.mean()),
            "sq_sum": float(sq.sum()),
            "n_rows": int(len(sq)),
            "n_games": int(n_games),
            "matched_slots": matched_slots,
            "valid_slots": valid_slots,
            "results_csv": str(csv_path),
        }
        logging.info(
            "[%s/%s] mse=%.6f coverage=%.4f", fold_label, name,
            stats[name]["mse"], matched_slots / max(valid_slots, 1),
        )

    base_mse_g = per_game_mse[baseline_name]
    for name in candidate_files:
        d = per_game_mse[name] - base_mse_g  # exact zeros when name == baseline
        num = (d[boot_idx] * boot_w).sum(axis=1)
        reps = num / boot_den
        stats[name].update({
            "delta_per_game": d,
            "game_weights": game_sizes,
            "mean_delta": float(np.average(d, weights=game_sizes)),
            "boot_num": num,
            "boot_den": boot_den,
            "ci_lo": float(np.percentile(reps, 2.5)),
            "ci_hi": float(np.percentile(reps, 97.5)),
            # Strictly-better games only; ties (including baseline-vs-itself)
            # count as non-wins.
            "win_rate": float(np.mean(d < 0)),
        })
    return stats


def stats_to_row(fold_label: str, candidate: str, s: dict) -> dict:
    return {
        "fold": fold_label,
        "candidate": candidate,
        "mse": s["mse"],
        "mean_paired_delta_vs_rs_only_e6": s["mean_delta"] * 1e6,
        "ci_lo_e6": s["ci_lo"] * 1e6,
        "ci_hi_e6": s["ci_hi"] * 1e6,
        "game_win_rate": s["win_rate"],
        "slot_coverage": s["matched_slots"] / max(s["valid_slots"], 1),
        "n_test_rows": s["n_rows"],
        "n_test_games": s["n_games"],
        "results_csv": s.get("results_csv", ""),
    }


def build_pooled_rows(fold_stats: dict[str, dict[str, dict]], candidate_names: list[str]) -> list[dict]:
    """Two labeled pooled rows per candidate across folds.

    pooled_simple_avg:    simple average of per-fold deltas/MSEs; bootstrap CI
                          from the replicate-wise mean of per-fold replicates.
    pooled_poss_weighted: single possession-weighted delta over all games from
                          all folds; CI from summed replicate numerators over
                          summed replicate denominators.
    """
    rows = []
    fold_labels = list(fold_stats)
    for name in candidate_names:
        per_fold = [fold_stats[label][name] for label in fold_labels]
        n_rows = sum(s["n_rows"] for s in per_fold)
        n_games = sum(s["n_games"] for s in per_fold)
        matched = sum(s["matched_slots"] for s in per_fold)
        valid = sum(s["valid_slots"] for s in per_fold)

        simple_reps = np.mean([s["boot_num"] / s["boot_den"] for s in per_fold], axis=0)
        rows.append({
            "fold": "pooled_simple_avg",
            "candidate": name,
            "mse": float(np.mean([s["mse"] for s in per_fold])),
            "mean_paired_delta_vs_rs_only_e6": float(np.mean([s["mean_delta"] for s in per_fold])) * 1e6,
            "ci_lo_e6": float(np.percentile(simple_reps, 2.5)) * 1e6,
            "ci_hi_e6": float(np.percentile(simple_reps, 97.5)) * 1e6,
            "game_win_rate": float(np.mean([s["win_rate"] for s in per_fold])),
            "slot_coverage": matched / max(valid, 1),
            "n_test_rows": n_rows,
            "n_test_games": n_games,
            "results_csv": "",
        })

        all_d = np.concatenate([s["delta_per_game"] for s in per_fold])
        all_w = np.concatenate([s["game_weights"] for s in per_fold])
        pooled_num = np.sum([s["boot_num"] for s in per_fold], axis=0)
        pooled_den = np.sum([s["boot_den"] for s in per_fold], axis=0)
        pooled_reps = pooled_num / pooled_den
        rows.append({
            "fold": "pooled_poss_weighted",
            "candidate": name,
            "mse": sum(s["sq_sum"] for s in per_fold) / max(n_rows, 1),
            "mean_paired_delta_vs_rs_only_e6": float(np.average(all_d, weights=all_w)) * 1e6,
            "ci_lo_e6": float(np.percentile(pooled_reps, 2.5)) * 1e6,
            "ci_hi_e6": float(np.percentile(pooled_reps, 97.5)) * 1e6,
            "game_win_rate": float(np.mean(all_d < 0)),
            "slot_coverage": matched / max(valid, 1),
            "n_test_rows": n_rows,
            "n_test_games": n_games,
            "results_csv": "",
        })
    return rows


def print_table(df: pd.DataFrame) -> None:
    show = df.copy()
    show["mse"] = show["mse"].map(lambda v: f"{v:.6f}")
    for col in ["mean_paired_delta_vs_rs_only_e6", "ci_lo_e6", "ci_hi_e6"]:
        show[col] = show[col].map(lambda v: f"{v:+.2f}")
    show["game_win_rate"] = show["game_win_rate"].map(lambda v: f"{v:.3f}")
    show["slot_coverage"] = show["slot_coverage"].map(lambda v: f"{v:.4f}")
    show = show.rename(columns={
        "mean_paired_delta_vs_rs_only_e6": "delta_e6",
        "ci_lo_e6": "ci_lo",
        "ci_hi_e6": "ci_hi",
        "game_win_rate": "win_rate",
        "slot_coverage": "coverage",
        "n_test_rows": "rows",
        "n_test_games": "games",
    })
    cols = ["fold", "candidate", "mse", "delta_e6", "ci_lo", "ci_hi", "win_rate", "coverage", "rows", "games"]
    show = show[cols].astype(str)
    widths = {c: max(len(c), int(show[c].str.len().max())) for c in cols}
    header = "  ".join(c.rjust(widths[c]) for c in cols)
    print()
    print(header)
    print("-" * len(header))
    for _, row in show.iterrows():
        print("  ".join(str(row[c]).rjust(widths[c]) for c in cols))
    print()


def run_self_test(args: argparse.Namespace) -> int:
    """Validate the pairing/bootstrap path without running any solves.

    Uses one existing results CSV as both baseline and candidate on a fake
    test slice, so every paired per-game delta must be exactly 0, the CI must
    be exactly [0, 0], and (because ties are non-wins) win rate must be 0.0.
    """
    logging.info("SELF-TEST: no solves are launched; bootstrap forced to 200")
    fake_results = RESULTS_DIR / "rapm_22_26_all_rb_se_a2000_4000_results.csv"
    if not fake_results.exists():
        raise SystemExit(f"Self-test fixture missing: {fake_results}")
    test_df = load_test_set(26).head(20000).reset_index(drop=True)
    logging.info("SELF-TEST: fake test set = first %d rows of RAPM26.parquet", len(test_df))

    candidate_files = {"fake_baseline": fake_results, "fake_candidate": fake_results}
    stats = evaluate_fold(
        fold_label="selftest",
        test_df=test_df,
        candidate_files=candidate_files,
        baseline_name="fake_baseline",
        n_bootstrap=200,
        seed=args.seed,
        fold_index=0,
    )

    rows = [stats_to_row("selftest", name, stats[name]) for name in candidate_files]
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAGNOSTICS_DIR / "ps_handling_selftest.csv"
    out_df.to_csv(out_path, index=False)
    logging.info("SELF-TEST: wrote %s", out_path)
    print_table(out_df)

    cand = stats["fake_candidate"]
    checks = {
        "paired deltas all exactly 0": bool(np.all(cand["delta_per_game"] == 0.0)),
        "mean delta exactly 0": cand["mean_delta"] == 0.0,
        "CI exactly [0, 0]": cand["ci_lo"] == 0.0 and cand["ci_hi"] == 0.0,
        "win rate 0.0 (ties are non-wins)": cand["win_rate"] == 0.0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        logging.info("SELF-TEST check: %-35s %s", label, "PASS" if passed else "FAIL")
    if not ok:
        logging.error("SELF-TEST FAILED")
        return 1
    logging.info("SELF-TEST PASSED: zero-delta invariant holds exactly")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forward-prediction gate comparing playoff-handling RAPM candidates on held-out future possessions.",
    )
    parser.add_argument("--folds", default="22-24:25,23-25:26",
                        help='Comma list of trainStart-trainEnd:testSeason (two-digit years). Default: "22-24:25,23-25:26"')
    parser.add_argument("--force", action="store_true",
                        help="Re-run solves even if the expected results CSVs already exist (default: reuse)")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Cluster-bootstrap resamples over games (default: 2000)")
    parser.add_argument("--seed", type=int, default=20260808,
                        help="Bootstrap seed (default: 20260808)")
    parser.add_argument("--output", type=Path, default=DIAGNOSTICS_DIR / "ps_handling_eval.csv",
                        help="Output CSV path (default: nba_pipeline/results/diagnostics/ps_handling_eval.csv)")
    parser.add_argument("--cores", type=int, default=None,
                        help="Passed through to rapm.py -j when launching solves")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.self_test:
        return run_self_test(args)

    folds = parse_folds(args.folds)
    candidate_names = list(CANDIDATES)

    # Solve (or reuse) sequentially: never run rapm.py invocations in parallel.
    fold_files: dict[str, dict[str, Path]] = {}
    for fold in folds:
        fold_files[fold["label"]] = {
            name: ensure_solved(fold, name, force=args.force, cores=args.cores)
            for name in candidate_names
        }

    rows = []
    fold_stats: dict[str, dict[str, dict]] = {}
    for fold_index, fold in enumerate(folds):
        test_df = load_test_set(fold["test_yy"])
        stats = evaluate_fold(
            fold_label=fold["label"],
            test_df=test_df,
            candidate_files=fold_files[fold["label"]],
            baseline_name=BASELINE_NAME,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            fold_index=fold_index,
        )
        fold_stats[fold["label"]] = stats
        rows.extend(stats_to_row(fold["label"], name, stats[name]) for name in candidate_names)

    rows.extend(build_pooled_rows(fold_stats, candidate_names))

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logging.info("Wrote %s", args.output)
    print_table(out_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())

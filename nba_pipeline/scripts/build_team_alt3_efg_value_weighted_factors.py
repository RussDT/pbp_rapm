#!/usr/bin/env python3
"""Build team-level Alt3 EFG-value weighted factors.

This solves the same public Alt3 EFG-value component targets used by the
player weighted-factors bundle, but with team offense and team defense
indicators instead of player stints. Defense columns in the final artifact are
sign-flipped so positive means good defense, matching the player-facing public
schema.
"""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from rapm import (
    ALT_BASELINE_PREFIXES,
    ALT_FIRST_CHANCE_TOV_PREFIXES,
    compute_alt_first_chance_baselines,
    detect_file_type_and_prepare,
    expand_two_digit_year_window,
)
from team_rapm import build_game_team_map


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

ATOMIC_EFG = OrderedDict(
    [
        ("rim_freq", "ALT_EFG_RIM_FREQ"),
        ("rim_fg", "ALT_EFG_RIM_FG"),
        ("mid_freq", "ALT_EFG_MID_FREQ"),
        ("mid_fg", "ALT_EFG_MID_FG"),
        ("three_freq", "ALT_EFG_THREE_FREQ"),
        ("three_fg", "ALT_EFG_THREE_FG"),
    ]
)

OTHER_COMPONENTS = OrderedDict(
    [
        ("ft", "ALT_FT"),
        ("ft_freq", "ALT_FT_FREQ"),
        ("ft_severity", "ALT_FT_SEVERITY"),
        ("tov_value", "ALT_TOV_VALUE"),
        ("sc_clean", "SECOND_CHANCE_CLEAN"),
        ("badpass_tov_value", "ALT_BADPASS_TOV_VALUE"),
        ("scoring_tov_value", "ALT_SCORING_TOV_VALUE"),
        ("assist_points", "ASSIST_POINTS"),
        ("rim_assist", "RIM_ASSIST"),
    ]
)

FILE_PREFIX_MAP = {
    "ALT_EFG_RIM_FREQ": "FIRST_CHANCE",
    "ALT_EFG_RIM_FG": "FIRST_CHANCE",
    "ALT_EFG_MID_FREQ": "FIRST_CHANCE",
    "ALT_EFG_MID_FG": "FIRST_CHANCE",
    "ALT_EFG_THREE_FREQ": "FIRST_CHANCE",
    "ALT_EFG_THREE_FG": "FIRST_CHANCE",
    "ALT_FT": "FIRST_CHANCE",
    "ALT_FT_FREQ": "FIRST_CHANCE",
    "ALT_FT_SEVERITY": "FIRST_CHANCE",
    "ALT_TOV_VALUE": "FIRST_CHANCE",
    "ALT_BADPASS_TOV_VALUE": "FIRST_CHANCE",
    "ALT_SCORING_TOV_VALUE": "FIRST_CHANCE",
}


def year_range_label(years: list[int]) -> str:
    if len(years) == 1:
        return f"{years[0]:02d}"
    return f"{years[0]:02d}_{years[-1]:02d}"


def input_files_for_prefix(prefix: str, years: list[int], season_type: str) -> list[Path]:
    file_prefix = FILE_PREFIX_MAP.get(prefix, prefix)
    files: list[Path] = []
    for year in years:
        if season_type in {"RS", "ALL"}:
            files.append(PROCESSED_DIR / f"{file_prefix}{year:02d}.parquet")
        if season_type in {"PS", "ALL"}:
            files.append(PROCESSED_DIR / f"{file_prefix}{year:02d}_PS.parquet")
    return files


def load_metric_df(prefix: str, years: list[int], season_type: str, pure: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in input_files_for_prefix(prefix, years, season_type):
        if not path.exists():
            logging.warning("Skipping missing processed file: %s", path)
            continue
        df = pd.read_parquet(path)
        df["season_phase"] = "PS" if path.stem.endswith("_PS") else "RS"
        df["source_season_end_year"] = int(path.stem[-5:-3]) if path.stem.endswith("_PS") else int(path.stem[-2:])
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No processed files found for {prefix} {years} {season_type}")

    df = pd.concat(frames, ignore_index=True)
    logging.info("Loaded %s: %s rows from %d files", prefix, f"{len(df):,}", len(frames))

    if prefix in ALT_BASELINE_PREFIXES:
        baselines = compute_alt_first_chance_baselines(df, requested_prefixes=[prefix])
        return detect_file_type_and_prepare(df, pure=pure, prefix=prefix, alt_baselines=baselines)
    if prefix in ALT_FIRST_CHANCE_TOV_PREFIXES:
        return detect_file_type_and_prepare(df, pure=pure, prefix=prefix)
    return detect_file_type_and_prepare(df, pure=pure, prefix=prefix)


def transform_to_team_rows(
    df: pd.DataFrame,
    game_teams: dict[str, dict[str, str]],
    game_home_players: dict[str, set[int]],
) -> pd.DataFrame:
    gids = df["game_id"].astype(str)
    o1 = pd.to_numeric(df["O1"], errors="coerce").fillna(0).astype(int)

    home_team = gids.map(lambda gid: game_teams.get(gid, {}).get("home"))
    away_team = gids.map(lambda gid: game_teams.get(gid, {}).get("away"))
    home_sets = gids.map(game_home_players.get)

    mapped = home_team.notna() & away_team.notna() & home_sets.notna() & o1.gt(0)
    if not mapped.all():
        logging.warning("Dropping %s unmapped team rows", f"{int((~mapped).sum()):,}")

    gids_mapped = gids[mapped].reset_index(drop=True)
    o1_mapped = o1[mapped].reset_index(drop=True)
    home_team_mapped = home_team[mapped].reset_index(drop=True)
    away_team_mapped = away_team[mapped].reset_index(drop=True)
    home_sets_mapped = home_sets[mapped].reset_index(drop=True)
    source = df.loc[mapped].reset_index(drop=True)

    off_is_home = np.fromiter(
        (int(pid) in home_players for pid, home_players in zip(o1_mapped, home_sets_mapped)),
        dtype=bool,
        count=len(source),
    )
    off_team = np.where(off_is_home, home_team_mapped, away_team_mapped)
    def_team = np.where(off_is_home, away_team_mapped, home_team_mapped)

    return pd.DataFrame(
        {
            "game_id": gids_mapped,
            "off_team": off_team,
            "def_team": def_team,
            "target": pd.to_numeric(source["Off_Diff"], errors="coerce").fillna(0.0).astype(float),
        }
    )


def solve_team_component(team_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    teams = sorted(set(team_df["off_team"]) | set(team_df["def_team"]))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    n_teams = len(teams)
    off_idx = team_df["off_team"].map(team_to_idx).to_numpy()
    def_idx = team_df["def_team"].map(team_to_idx).to_numpy()
    y = team_df["target"].to_numpy(dtype=np.float64)
    y = y - float(y.mean())

    beta_off = np.zeros(n_teams, dtype=np.float64)
    beta_def = np.zeros(n_teams, dtype=np.float64)
    residual = y.copy()
    off_counts_arr = np.bincount(off_idx, minlength=n_teams).astype(np.float64)
    def_counts_arr = np.bincount(def_idx, minlength=n_teams).astype(np.float64)

    for iteration in range(200):
        prev_off = beta_off.copy()
        prev_def = beta_def.copy()

        residual += beta_off[off_idx]
        off_sums = np.bincount(off_idx, weights=residual, minlength=n_teams)
        beta_off = off_sums / (off_counts_arr + alpha)
        residual -= beta_off[off_idx]

        residual += beta_def[def_idx]
        def_sums = np.bincount(def_idx, weights=residual, minlength=n_teams)
        beta_def = def_sums / (def_counts_arr + alpha)
        residual -= beta_def[def_idx]

        delta = np.linalg.norm(beta_off - prev_off) + np.linalg.norm(beta_def - prev_def)
        if delta < 1e-8:
            logging.info("  Converged in %d iterations", iteration + 1)
            break

    off_poss = team_df["off_team"].value_counts().to_dict()
    def_poss = team_df["def_team"].value_counts().to_dict()
    total_weights = np.array([off_poss.get(team, 0) + def_poss.get(team, 0) for team in teams], dtype=float)
    if total_weights.sum() > 0:
        beta_off -= float(np.average(beta_off, weights=total_weights))
        beta_def -= float(np.average(beta_def, weights=total_weights))

    rows = []
    for team, idx in team_to_idx.items():
        off_raw = float(beta_off[idx] * 100.0)
        def_raw = float(beta_def[idx] * 100.0)
        rows.append(
            {
                "team": team,
                "off_raw": off_raw,
                "def_raw": def_raw,
                "net_raw": off_raw - def_raw,
                "off_poss": int(off_poss.get(team, 0)),
                "def_poss": int(def_poss.get(team, 0)),
                "possessions": int(off_poss.get(team, 0) + def_poss.get(team, 0)),
            }
        )
    return pd.DataFrame(rows)


def solve_prefix(
    prefix: str,
    years: list[int],
    season_type: str,
    game_teams: dict[str, dict[str, str]],
    game_home_players: dict[str, set[int]],
    alpha: float,
    pure: bool = False,
) -> pd.DataFrame:
    metric_df = load_metric_df(prefix, years, season_type, pure=pure)
    team_df = transform_to_team_rows(metric_df, game_teams, game_home_players)
    logging.info("Solving %s with alpha=%s on %s team rows", prefix, alpha, f"{len(team_df):,}")
    return solve_team_component(team_df, alpha)


def merge_component(out: pd.DataFrame, component: pd.DataFrame, name: str) -> pd.DataFrame:
    renamed = component[["team", "off_raw", "def_raw", "net_raw"]].rename(
        columns={
            "off_raw": f"o_{name}",
            "def_raw": f"d_{name}",
            "net_raw": f"n_{name}",
        }
    )
    return out.merge(renamed, on="team", how="inner", validate="one_to_one")


def build_team_alt3_bundle(
    start_year: int,
    end_year: int,
    season_type: str,
    alpha: float,
    output: Path,
    parquet_output: Path | None,
    decimals: int,
) -> pd.DataFrame:
    years = expand_two_digit_year_window(start_year, end_year)
    game_teams, game_home_players = build_game_team_map(years, season_type)

    rapm = solve_prefix(
        "RAPM",
        years,
        season_type,
        game_teams,
        game_home_players,
        alpha,
        pure=True,
    )
    out = rapm.rename(
        columns={"off_raw": "off", "def_raw": "def_raw", "net_raw": "net_rapm"}
    )[["team", "off", "def_raw", "net_rapm", "possessions", "off_poss", "def_poss"]].copy()

    for name, prefix in ATOMIC_EFG.items():
        out = merge_component(
            out,
            solve_prefix(prefix, years, season_type, game_teams, game_home_players, alpha),
            name,
        )
    for name, prefix in OTHER_COMPONENTS.items():
        out = merge_component(
            out,
            solve_prefix(prefix, years, season_type, game_teams, game_home_players, alpha),
            name,
        )

    for side in ["o", "d", "n"]:
        out[f"{side}_rim_value"] = out[f"{side}_rim_freq"] + out[f"{side}_rim_fg"]
        out[f"{side}_mid_value"] = out[f"{side}_mid_freq"] + out[f"{side}_mid_fg"]
        out[f"{side}_three_value"] = out[f"{side}_three_freq"] + out[f"{side}_three_fg"]
        out[f"{side}_shot_value"] = (
            out[f"{side}_rim_value"] + out[f"{side}_mid_value"] + out[f"{side}_three_value"]
        )
        out[f"{side}_tov_loss_split"] = out[f"{side}_badpass_tov_value"] + out[f"{side}_scoring_tov_value"]

    # Public weighted-factors convention: positive defensive values are good.
    raw_def_columns = ["def_raw"] + [col for col in out.columns if col.startswith("d_") or col.startswith("n_")]
    for col in raw_def_columns:
        if col.startswith("d_") or col == "def_raw":
            out[col] = -out[col]

    out = out.rename(columns={"def_raw": "def"})
    out["net_rapm"] = out["off"] + out["def"]
    out["oFC"] = out["o_shot_value"] + out["o_ft"] + out["o_tov_value"]
    out["dFC"] = out["d_shot_value"] + out["d_ft"] + out["d_tov_value"]
    out["oDISPLAY_SUM"] = out["oFC"] + out["o_sc_clean"]
    out["dDISPLAY_SUM"] = out["dFC"] + out["d_sc_clean"]
    out["nDISPLAY_SUM"] = out["oDISPLAY_SUM"] + out["dDISPLAY_SUM"]
    out["oRESID"] = out["off"] - out["oDISPLAY_SUM"]
    out["dRESID"] = out["def"] - out["dDISPLAY_SUM"]
    out["RESID"] = out["net_rapm"] - out["nDISPLAY_SUM"]
    out["alpha"] = alpha

    column_order = [
        "team",
        "alpha",
        "off",
        "def",
        "net_rapm",
        "o_shot_value",
        "o_rim_value",
        "o_rim_freq",
        "o_rim_fg",
        "o_mid_value",
        "o_mid_freq",
        "o_mid_fg",
        "o_three_value",
        "o_three_freq",
        "o_three_fg",
        "o_ft",
        "o_ft_freq",
        "o_ft_severity",
        "o_tov_value",
        "o_sc_clean",
        "oFC",
        "oDISPLAY_SUM",
        "oRESID",
        "o_badpass_tov_value",
        "o_scoring_tov_value",
        "o_tov_loss_split",
        "o_assist_points",
        "o_rim_assist",
        "d_shot_value",
        "d_rim_value",
        "d_rim_freq",
        "d_rim_fg",
        "d_mid_value",
        "d_mid_freq",
        "d_mid_fg",
        "d_three_value",
        "d_three_freq",
        "d_three_fg",
        "d_ft",
        "d_ft_freq",
        "d_ft_severity",
        "d_tov_value",
        "d_sc_clean",
        "dFC",
        "dDISPLAY_SUM",
        "dRESID",
        "d_badpass_tov_value",
        "d_scoring_tov_value",
        "d_tov_loss_split",
        "d_assist_points",
        "d_rim_assist",
        "n_shot_value",
        "n_rim_value",
        "n_rim_freq",
        "n_rim_fg",
        "n_mid_value",
        "n_mid_freq",
        "n_mid_fg",
        "n_three_value",
        "n_three_freq",
        "n_three_fg",
        "n_ft",
        "n_ft_freq",
        "n_ft_severity",
        "n_tov_value",
        "n_sc_clean",
        "nDISPLAY_SUM",
        "RESID",
        "n_badpass_tov_value",
        "n_scoring_tov_value",
        "n_tov_loss_split",
        "n_assist_points",
        "n_rim_assist",
        "possessions",
        "off_poss",
        "def_poss",
    ]
    out = out[column_order].sort_values("net_rapm", ascending=False).reset_index(drop=True)

    numeric_cols = out.select_dtypes(include="number").columns.difference(["alpha"])
    out[numeric_cols] = out[numeric_cols].round(decimals)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    logging.info("Wrote %s", output)
    if parquet_output is not None:
        parquet_output.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(parquet_output, index=False)
        logging.info("Wrote %s", parquet_output)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("season_type", choices=["RS", "PS", "ALL"])
    parser.add_argument("--alpha", type=float, default=25.0, help="Team ridge alpha. Default: 25")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--parquet-output", type=Path, default=None)
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = expand_two_digit_year_window(args.start_year, args.end_year)
    label = year_range_label(years)
    suffix = f"{args.season_type.lower()}_a{int(args.alpha)}"
    output = args.output or MASTER_RESULTS_DIR / f"team_weighted_factors_alt3_efg_value_{label}_{suffix}.csv"
    parquet_output = args.parquet_output
    if parquet_output is None and not args.no_parquet:
        parquet_output = output.with_suffix(".parquet")

    out = build_team_alt3_bundle(
        start_year=args.start_year,
        end_year=args.end_year,
        season_type=args.season_type,
        alpha=args.alpha,
        output=output,
        parquet_output=parquet_output,
        decimals=args.decimals,
    )

    logging.info("Top team values:")
    print(out[["team", "net_rapm", "off", "def", "o_shot_value", "o_ft", "o_tov_value", "o_sc_clean", "d_shot_value", "d_ft", "d_tov_value", "d_sc_clean"]].head(30).to_string(index=False))
    for col in ["oRESID", "dRESID", "RESID"]:
        logging.info("%s max_abs=%.6f mean_abs=%.6f", col, out[col].abs().max(), out[col].abs().mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

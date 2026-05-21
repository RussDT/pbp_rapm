#!/usr/bin/env python3
"""Build a weighted-factors style EFG-value alt3 bundle.

This converts the direct EFG-value RAPM result bundle into the public
weighted_factors sign/schema convention:
- offense columns are raw offensive coefficients
- defense columns are sign-flipped so positive means good defense
- player universe and Latest_Year come from the matching legacy alt3 file
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"


DEFAULT_BUNDLE = RESULTS_DIR / "alt_efg_value_display_bundle_24_26_all_rb_se_a2000_4000_results.csv"
DEFAULT_BASE = RESULTS_DIR / "weighted_factors_alt3_24_26_all_rb_se_a2000_4000.csv"
DEFAULT_OUTPUT_NAME = "weighted_factors_alt3_efg_value_24_26_all_rb_se_a2000_4000.csv"


OFF_VALUE_COLUMNS = {
    "oALT_EFG_VALUE": "o_shot_value",
    "oALT_EFG_BASELINE": "o_efg_baseline",
    "oALT_EFG_ATOMS_VALUE": "o_shot_atoms_value",
    "oALT_EFG_RIM": "o_rim_value",
    "oALT_EFG_RIM_FREQ": "o_rim_freq",
    "oALT_EFG_RIM_FG": "o_rim_fg",
    "oALT_EFG_MID": "o_mid_value",
    "oALT_EFG_MID_FREQ": "o_mid_freq",
    "oALT_EFG_MID_FG": "o_mid_fg",
    "oALT_EFG_THREE": "o_three_value",
    "oALT_EFG_THREE_FREQ": "o_three_freq",
    "oALT_EFG_THREE_FG": "o_three_fg",
    "oALT_FT": "o_ft",
    "oALT_FT_FREQ": "o_ft_freq",
    "oALT_FT_SEVERITY": "o_ft_severity",
    "oALT_TOV_VALUE": "o_tov_value",
    "oSC": "o_sc_clean",
    "oSC_CLEAN": "o_sc_clean",
    "oALT_TOV_LOSS_VALUE": "o_tov_loss_split",
    "oALT_TOV_VALUE_bp": "o_badpass_tov_value",
    "oALT_TOV_VALUE_sc": "o_scoring_tov_value",
    "oALT_TOV_VALUE_split": "o_tov_loss_split",
    "oALT_BADPASS_TOV_VALUE": "o_badpass_tov_value",
    "oALT_SCORING_TOV_VALUE": "o_scoring_tov_value",
    "oASSIST_POINTS": "o_assist_points",
    "oRIM_ASSIST": "o_rim_assist",
}


DEF_VALUE_COLUMNS = {
    "dALT_EFG_VALUE": "d_shot_value",
    "dALT_EFG_BASELINE": "d_efg_baseline",
    "dALT_EFG_ATOMS_VALUE": "d_shot_atoms_value",
    "dALT_EFG_RIM": "d_rim_value",
    "dALT_EFG_RIM_FREQ": "d_rim_freq",
    "dALT_EFG_RIM_FG": "d_rim_fg",
    "dALT_EFG_MID": "d_mid_value",
    "dALT_EFG_MID_FREQ": "d_mid_freq",
    "dALT_EFG_MID_FG": "d_mid_fg",
    "dALT_EFG_THREE": "d_three_value",
    "dALT_EFG_THREE_FREQ": "d_three_freq",
    "dALT_EFG_THREE_FG": "d_three_fg",
    "dALT_FT": "d_ft",
    "dALT_FT_FREQ": "d_ft_freq",
    "dALT_FT_SEVERITY": "d_ft_severity",
    "dALT_TOV_VALUE": "d_tov_value",
    "dSC": "d_sc_clean",
    "dSC_CLEAN": "d_sc_clean",
    "dALT_TOV_LOSS_VALUE": "d_tov_loss_split",
    "dALT_TOV_VALUE_bp": "d_badpass_tov_value",
    "dALT_TOV_VALUE_sc": "d_scoring_tov_value",
    "dALT_TOV_VALUE_split": "d_tov_loss_split",
    "dALT_BADPASS_TOV_VALUE": "d_badpass_tov_value",
    "dALT_SCORING_TOV_VALUE": "d_scoring_tov_value",
    "dASSIST_POINTS": "d_assist_points",
    "dRIM_ASSIST": "d_rim_assist",
}


NET_VALUE_COLUMNS = {
    "nALT_EFG_VALUE": "n_shot_value",
    "nALT_EFG_BASELINE": "n_efg_baseline",
    "nALT_EFG_ATOMS_VALUE": "n_shot_atoms_value",
    "nALT_EFG_RIM": "n_rim_value",
    "nALT_EFG_RIM_FREQ": "n_rim_freq",
    "nALT_EFG_RIM_FG": "n_rim_fg",
    "nALT_EFG_MID": "n_mid_value",
    "nALT_EFG_MID_FREQ": "n_mid_freq",
    "nALT_EFG_MID_FG": "n_mid_fg",
    "nALT_EFG_THREE": "n_three_value",
    "nALT_EFG_THREE_FREQ": "n_three_freq",
    "nALT_EFG_THREE_FG": "n_three_fg",
    "nALT_FT": "n_ft",
    "nALT_FT_FREQ": "n_ft_freq",
    "nALT_FT_SEVERITY": "n_ft_severity",
    "nALT_TOV_VALUE": "n_tov_value",
    "nSC": "n_sc_clean",
    "nSC_CLEAN": "n_sc_clean",
    "nALT_TOV_LOSS_VALUE": "n_tov_loss_split",
    "nALT_TOV_VALUE_bp": "n_badpass_tov_value",
    "nALT_TOV_VALUE_sc": "n_scoring_tov_value",
    "nALT_TOV_VALUE_split": "n_tov_loss_split",
    "nALT_BADPASS_TOV_VALUE": "n_badpass_tov_value",
    "nALT_SCORING_TOV_VALUE": "n_scoring_tov_value",
    "nASSIST_POINTS": "n_assist_points",
    "nRIM_ASSIST": "n_rim_assist",
}


def round_numeric(df: pd.DataFrame, columns: list[str], decimals: int) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)


def build_weighted_factors(
    bundle_path: Path,
    base_path: Path,
    output_name: str,
    decimals: int,
    copy_to_master: bool,
) -> tuple[Path, Path | None, pd.DataFrame]:
    bundle = pd.read_csv(bundle_path)
    base = pd.read_csv(base_path)

    required_base = ["player_id", "player_name", "Latest_Year"]
    missing_base = [col for col in required_base if col not in base.columns]
    if missing_base:
        raise ValueError(f"{base_path} is missing required columns: {missing_base}")

    needed_bundle = {"player_id", "off", "def", "net", "possessions", "off_poss", "def_poss"}
    needed_bundle.update(OFF_VALUE_COLUMNS.values())
    needed_bundle.update(DEF_VALUE_COLUMNS.values())
    needed_bundle.update(NET_VALUE_COLUMNS.values())
    missing_bundle = sorted(needed_bundle - set(bundle.columns))
    if missing_bundle:
        raise ValueError(f"{bundle_path} is missing required columns: {missing_bundle}")

    merged = base[["player_id", "player_name", "Latest_Year"]].merge(
        bundle,
        on="player_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_bundle"),
    )
    if merged["off"].isna().any():
        missing = merged.loc[merged["off"].isna(), ["player_id", "player_name"]]
        raise ValueError(f"Bundle is missing {len(missing)} base players:\n{missing.to_string(index=False)}")

    out = merged[["player_id", "player_name", "Latest_Year"]].copy()
    out["off"] = merged["off"]
    out["def"] = -merged["def"]
    out["net_rapm"] = merged["net"]

    for dst, src in OFF_VALUE_COLUMNS.items():
        out[dst] = merged[src]
    for dst, src in DEF_VALUE_COLUMNS.items():
        out[dst] = -merged[src]
    for dst, src in NET_VALUE_COLUMNS.items():
        out[dst] = merged[src]

    out["oFC"] = out["oALT_EFG_VALUE"] + out["oALT_FT"] + out["oALT_TOV_VALUE"]
    out["dFC"] = out["dALT_EFG_VALUE"] + out["dALT_FT"] + out["dALT_TOV_VALUE"]
    out["oDISPLAY_SUM"] = out["oFC"] + out["oSC"]
    out["dDISPLAY_SUM"] = out["dFC"] + out["dSC"]
    out["nDISPLAY_SUM"] = out["oDISPLAY_SUM"] + out["dDISPLAY_SUM"]
    out["oRESID"] = out["off"] - out["oDISPLAY_SUM"]
    out["dRESID"] = out["def"] - out["dDISPLAY_SUM"]
    out["RESID"] = out["net_rapm"] - out["nDISPLAY_SUM"]

    out["possessions"] = merged["possessions"]
    out["off_poss"] = merged["off_poss"]
    out["def_poss"] = merged["def_poss"]

    value_columns = [
        "off",
        "def",
        "net_rapm",
        *OFF_VALUE_COLUMNS.keys(),
        *DEF_VALUE_COLUMNS.keys(),
        *NET_VALUE_COLUMNS.keys(),
        "oFC",
        "dFC",
        "oDISPLAY_SUM",
        "dDISPLAY_SUM",
        "nDISPLAY_SUM",
        "oRESID",
        "dRESID",
        "RESID",
    ]
    round_numeric(out, value_columns, decimals)

    column_order = [
        "player_id",
        "player_name",
        "Latest_Year",
        "oALT_EFG_VALUE",
        "oALT_EFG_BASELINE",
        "oALT_EFG_ATOMS_VALUE",
        "oALT_EFG_RIM",
        "oALT_EFG_RIM_FREQ",
        "oALT_EFG_RIM_FG",
        "oALT_EFG_MID",
        "oALT_EFG_MID_FREQ",
        "oALT_EFG_MID_FG",
        "oALT_EFG_THREE",
        "oALT_EFG_THREE_FREQ",
        "oALT_EFG_THREE_FG",
        "oALT_FT",
        "oALT_FT_FREQ",
        "oALT_FT_SEVERITY",
        "oALT_TOV_VALUE",
        "oSC",
        "oSC_CLEAN",
        "oFC",
        "oDISPLAY_SUM",
        "oRESID",
        "oALT_TOV_LOSS_VALUE",
        "oALT_TOV_VALUE_bp",
        "oALT_TOV_VALUE_sc",
        "oALT_TOV_VALUE_split",
        "oALT_BADPASS_TOV_VALUE",
        "oALT_SCORING_TOV_VALUE",
        "oASSIST_POINTS",
        "oRIM_ASSIST",
        "dALT_EFG_VALUE",
        "dALT_EFG_BASELINE",
        "dALT_EFG_ATOMS_VALUE",
        "dALT_EFG_RIM",
        "dALT_EFG_RIM_FREQ",
        "dALT_EFG_RIM_FG",
        "dALT_EFG_MID",
        "dALT_EFG_MID_FREQ",
        "dALT_EFG_MID_FG",
        "dALT_EFG_THREE",
        "dALT_EFG_THREE_FREQ",
        "dALT_EFG_THREE_FG",
        "dALT_FT",
        "dALT_FT_FREQ",
        "dALT_FT_SEVERITY",
        "dALT_TOV_VALUE",
        "dSC",
        "dSC_CLEAN",
        "dFC",
        "dDISPLAY_SUM",
        "dRESID",
        "dALT_TOV_LOSS_VALUE",
        "dALT_TOV_VALUE_bp",
        "dALT_TOV_VALUE_sc",
        "dALT_TOV_VALUE_split",
        "dALT_BADPASS_TOV_VALUE",
        "dALT_SCORING_TOV_VALUE",
        "dASSIST_POINTS",
        "dRIM_ASSIST",
        "nALT_EFG_VALUE",
        "nALT_EFG_BASELINE",
        "nALT_EFG_ATOMS_VALUE",
        "nALT_EFG_RIM",
        "nALT_EFG_RIM_FREQ",
        "nALT_EFG_RIM_FG",
        "nALT_EFG_MID",
        "nALT_EFG_MID_FREQ",
        "nALT_EFG_MID_FG",
        "nALT_EFG_THREE",
        "nALT_EFG_THREE_FREQ",
        "nALT_EFG_THREE_FG",
        "nALT_FT",
        "nALT_FT_FREQ",
        "nALT_FT_SEVERITY",
        "nALT_TOV_VALUE",
        "nSC",
        "nSC_CLEAN",
        "nDISPLAY_SUM",
        "RESID",
        "nALT_TOV_LOSS_VALUE",
        "nALT_TOV_VALUE_bp",
        "nALT_TOV_VALUE_sc",
        "nALT_TOV_VALUE_split",
        "nALT_BADPASS_TOV_VALUE",
        "nALT_SCORING_TOV_VALUE",
        "nASSIST_POINTS",
        "nRIM_ASSIST",
        "off",
        "def",
        "net_rapm",
        "possessions",
        "off_poss",
        "def_poss",
    ]
    out = out[column_order].sort_values("net_rapm", ascending=False)

    output_path = RESULTS_DIR / output_name
    out.to_csv(output_path, index=False)

    master_path = None
    if copy_to_master:
        MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        master_path = MASTER_RESULTS_DIR / output_name
        shutil.copy2(output_path, master_path)

    return output_path, master_path, out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--no-master-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, master_path, out = build_weighted_factors(
        bundle_path=args.bundle,
        base_path=args.base,
        output_name=args.output_name,
        decimals=args.decimals,
        copy_to_master=not args.no_master_copy,
    )

    print(f"Wrote {output_path}")
    if master_path is not None:
        print(f"Copied {master_path}")
    print(f"Rows: {len(out):,}; columns: {len(out.columns):,}")
    for col in ["oRESID", "dRESID", "RESID"]:
        print(
            f"{col}: max_abs={out[col].abs().max():.6f} "
            f"mean_abs={out[col].abs().mean():.6f}"
        )


if __name__ == "__main__":
    main()

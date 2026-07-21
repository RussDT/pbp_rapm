#!/usr/bin/env python3
"""Build the public eight-component DECOMP weighted-factor table.

The public tree is five shot children plus free throws, point-valued
turnovers, and clean second chance. Offense is positive-good; defense is
sign-flipped from the solver so positive is also good.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"

SHOT_COMPONENTS = {
    "RIM_FREQ": "DECOMP_RIM_FREQ",
    "RIM_FG": "DECOMP_RIM_FG",
    "MID_FG": "DECOMP_MID_FG",
    "THREE_FREQ": "DECOMP_THREE_FREQ",
    "THREE_FG": "DECOMP_THREE_FG",
}
OTHER_COMPONENTS = {
    "EFG_PARENT": "DECOMP_EFG",
    "FT": "ALT_FT",
    "FT_FREQ": "ALT_FT_FREQ",
    "FT_SEVERITY": "ALT_FT_SEVERITY",
    "TOV_VALUE": "ALT_TOV_VALUE",
    "SC": "SECOND_CHANCE_CLEAN",
    "BADPASS_TOV_VALUE": "ALT_BADPASS_TOV_VALUE",
    "SCORING_TOV_VALUE": "ALT_SCORING_TOV_VALUE",
    "ASSIST_POINTS": "ASSIST_POINTS",
    "RIM_ASSIST": "RIM_ASSIST",
}


def result_filename(prefix: str, year_range: str, suffix: str) -> str:
    result_suffix = suffix.replace("all", "all_pure", 1) if prefix == "RAPM" else suffix
    return f"{prefix.lower()}_{year_range}_{result_suffix}_results.csv"


def find_result(prefix: str, year_range: str, suffix: str, results_dir: Path) -> Path:
    name = result_filename(prefix, year_range, suffix)
    direct = results_dir / name
    if direct.exists():
        return direct
    matches = sorted(results_dir.glob(f"**/{name}"))
    if not matches:
        raise FileNotFoundError(f"Missing DECOMP input: {name}")
    return matches[0]


def load_result(prefix: str, year_range: str, suffix: str, results_dir: Path) -> pd.DataFrame:
    path = find_result(prefix, year_range, suffix, results_dir)
    df = pd.read_csv(path)
    required = {"player_id", "off", "def", "net_rapm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    keep = ["player_id", "off", "def", "net_rapm"]
    for col in ["player_name", "possessions", "off_poss", "def_poss"]:
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()


def attach_component(
    out: pd.DataFrame,
    component: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    values = component[["player_id", "off", "def", "net_rapm"]].rename(
        columns={
            "off": f"oDECOMP_{label}",
            "def": f"dDECOMP_{label}_solver",
            "net_rapm": f"nDECOMP_{label}_solver",
        }
    )
    out = out.merge(values, on="player_id", how="left", validate="one_to_one")
    if out[f"oDECOMP_{label}"].isna().any():
        missing_ids = out.loc[out[f"oDECOMP_{label}"].isna(), "player_id"].tolist()[:10]
        raise ValueError(f"DECOMP {label} result is missing base players: {missing_ids}")
    out[f"dDECOMP_{label}"] = -out.pop(f"dDECOMP_{label}_solver")
    out[f"nDECOMP_{label}"] = out[f"oDECOMP_{label}"] + out[f"dDECOMP_{label}"]
    out.drop(columns=[f"nDECOMP_{label}_solver"], inplace=True)
    return out


def build_decomp_weighted_factors(
    year_range: str,
    suffix: str,
    base_path: Path,
    output_name: str,
    results_dir: Path = RESULTS_DIR,
    decimals: int = 3,
    transfer_multiplier: float = 1.0,
    copy_to_master: bool = True,
    write_parquet: bool = True,
) -> tuple[Path, Path | None, Path | None, Path | None, pd.DataFrame]:
    if transfer_multiplier <= 0:
        raise ValueError("turnover/second-chance transfer multiplier must be positive")
    base = pd.read_csv(base_path)
    required_base = {"player_id", "player_name", "Latest_Year"}
    missing_base = required_base - set(base.columns)
    if missing_base:
        raise ValueError(f"{base_path} is missing columns: {sorted(missing_base)}")

    rapm = load_result("RAPM", year_range, suffix, results_dir)
    rapm_keep = ["player_id", "off", "def", "net_rapm", "possessions", "off_poss", "def_poss"]
    missing_rapm = set(rapm_keep) - set(rapm.columns)
    if missing_rapm:
        raise ValueError(f"RAPM result is missing columns: {sorted(missing_rapm)}")
    out = base[["player_id", "player_name", "Latest_Year"]].merge(
        rapm[rapm_keep], on="player_id", how="left", validate="one_to_one"
    )
    if out["off"].isna().any():
        raise ValueError("RAPM result does not cover the weighted-factor base player universe")
    out["def"] = -out["def"]

    for label, prefix in {**SHOT_COMPONENTS, **OTHER_COMPONENTS}.items():
        out = attach_component(out, load_result(prefix, year_range, suffix, results_dir), label)

    policy = "full_possession_v1" if transfer_multiplier != 1.0 else "none"
    out.insert(3, "TOV_SC_ALLOCATION_POLICY", policy)
    out.insert(4, "TOV_SC_TRANSFER_MULTIPLIER", float(transfer_multiplier))

    for side in ["o", "d"]:
        # July 10 recovery: move the sample-specific full-possession share from
        # clean second chance into the parent and both turnover children while
        # preserving the displayed total exactly.
        old_tov = out[f"{side}DECOMP_TOV_VALUE"].copy()
        new_tov = old_tov * transfer_multiplier
        out[f"{side}DECOMP_TOV_VALUE"] = new_tov
        out[f"{side}DECOMP_BADPASS_TOV_VALUE"] *= transfer_multiplier
        out[f"{side}DECOMP_SCORING_TOV_VALUE"] *= transfer_multiplier
        out[f"{side}DECOMP_SC"] -= new_tov - old_tov

        out[f"{side}DECOMP_RIM"] = (
            out[f"{side}DECOMP_RIM_FREQ"] + out[f"{side}DECOMP_RIM_FG"]
        )
        out[f"{side}DECOMP_THREE"] = (
            out[f"{side}DECOMP_THREE_FREQ"] + out[f"{side}DECOMP_THREE_FG"]
        )
        children = (
            out[f"{side}DECOMP_RIM_FREQ"]
            + out[f"{side}DECOMP_RIM_FG"]
            + out[f"{side}DECOMP_MID_FG"]
            + out[f"{side}DECOMP_THREE_FREQ"]
            + out[f"{side}DECOMP_THREE_FG"]
        )
        out[f"{side}DECOMP_EFG_CHILDREN"] = children
        out[f"{side}DECOMP_EFG_PARENT_GAP"] = out[f"{side}DECOMP_EFG_PARENT"] - children
        out[f"{side}SC"] = out[f"{side}DECOMP_SC"]
        out[f"{side}SC_CLEAN"] = out[f"{side}DECOMP_SC"]
        rapm_total = out["off"] if side == "o" else out["def"]
        out[f"{side}DECOMP_EFG"] = (
            rapm_total
            - out[f"{side}DECOMP_FT"]
            - out[f"{side}DECOMP_TOV_VALUE"]
            - out[f"{side}SC"]
        )
        out[f"{side}DECOMP_EFG_DISPLAY_GAP"] = out[f"{side}DECOMP_EFG"] - children
        out[f"{side}FC"] = (
            out[f"{side}DECOMP_EFG"]
            + out[f"{side}DECOMP_FT"]
            + out[f"{side}DECOMP_TOV_VALUE"]
        )
        out[f"{side}DISPLAY_SUM"] = out[f"{side}FC"] + out[f"{side}SC"]
        out[f"{side}RESID"] = rapm_total - out[f"{side}DISPLAY_SUM"]
        out[f"{side}DECOMP_TOV_LOSS_VALUE"] = out[f"{side}DECOMP_TOV_VALUE"]
        out[f"{side}DECOMP_TOV_VALUE_bp"] = out[f"{side}DECOMP_BADPASS_TOV_VALUE"]
        out[f"{side}DECOMP_TOV_VALUE_sc"] = out[f"{side}DECOMP_SCORING_TOV_VALUE"]
        out[f"{side}DECOMP_TOV_VALUE_split"] = (
            out[f"{side}DECOMP_BADPASS_TOV_VALUE"]
            + out[f"{side}DECOMP_SCORING_TOV_VALUE"]
        )
        out[f"{side}ASSIST_POINTS"] = out[f"{side}DECOMP_ASSIST_POINTS"]
        out[f"{side}RIM_ASSIST"] = out[f"{side}DECOMP_RIM_ASSIST"]

    # Net columns are derived from the displayed offense/defense convention so
    # every public identity remains mechanically checkable after sign flipping.
    net_labels = [
        "EFG", "EFG_PARENT", "EFG_CHILDREN", "EFG_PARENT_GAP", "EFG_DISPLAY_GAP",
        "RIM", "RIM_FREQ", "RIM_FG", "MID_FG", "THREE", "THREE_FREQ", "THREE_FG",
        "FT", "FT_FREQ", "FT_SEVERITY", "TOV_VALUE", "SC",
        "TOV_LOSS_VALUE", "TOV_VALUE_bp", "TOV_VALUE_sc", "TOV_VALUE_split",
        "BADPASS_TOV_VALUE", "SCORING_TOV_VALUE", "ASSIST_POINTS", "RIM_ASSIST",
    ]
    for label in net_labels:
        out[f"nDECOMP_{label}"] = out[f"oDECOMP_{label}"] + out[f"dDECOMP_{label}"]
    out["nSC"] = out["oSC"] + out["dSC"]
    out["nSC_CLEAN"] = out["nSC"]
    out["nASSIST_POINTS"] = out["oASSIST_POINTS"] + out["dASSIST_POINTS"]
    out["nRIM_ASSIST"] = out["oRIM_ASSIST"] + out["dRIM_ASSIST"]
    out["nDISPLAY_SUM"] = out["oDISPLAY_SUM"] + out["dDISPLAY_SUM"]
    out["RESID"] = out["net_rapm"] - out["nDISPLAY_SUM"]

    side_order = [
        "DECOMP_EFG", "DECOMP_EFG_PARENT", "DECOMP_EFG_CHILDREN",
        "DECOMP_EFG_PARENT_GAP", "DECOMP_EFG_DISPLAY_GAP", "DECOMP_RIM",
        "DECOMP_RIM_FREQ", "DECOMP_RIM_FG", "DECOMP_MID_FG", "DECOMP_THREE",
        "DECOMP_THREE_FREQ", "DECOMP_THREE_FG", "DECOMP_FT", "DECOMP_FT_FREQ",
        "DECOMP_FT_SEVERITY", "DECOMP_TOV_VALUE", "SC", "SC_CLEAN", "FC",
        "DISPLAY_SUM", "RESID", "DECOMP_TOV_LOSS_VALUE", "DECOMP_TOV_VALUE_bp",
        "DECOMP_TOV_VALUE_sc", "DECOMP_TOV_VALUE_split", "DECOMP_BADPASS_TOV_VALUE",
        "DECOMP_SCORING_TOV_VALUE", "ASSIST_POINTS", "RIM_ASSIST",
    ]
    net_order = [item for item in side_order if item not in {"FC", "RESID"}]
    display_sum_index = net_order.index("DISPLAY_SUM") + 1
    net_before_resid = net_order[:display_sum_index]
    net_after_resid = net_order[display_sum_index:]
    order = [
        "player_id", "player_name", "Latest_Year", "TOV_SC_ALLOCATION_POLICY",
        "TOV_SC_TRANSFER_MULTIPLIER",
        *[f"o{item}" for item in side_order],
        *[f"d{item}" for item in side_order],
        *[f"n{item}" for item in net_before_resid],
        "RESID",
        *[f"n{item}" for item in net_after_resid],
        "off", "def", "net_rapm", "possessions", "off_poss", "def_poss",
    ]
    out = out[order].sort_values("net_rapm", ascending=False)
    numeric = out.select_dtypes(include="number").columns.difference(
        ["player_id", "Latest_Year", "TOV_SC_TRANSFER_MULTIPLIER"]
    )
    out[numeric] = out[numeric].round(decimals)

    output_path = RESULTS_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    parquet_path = output_path.with_suffix(".parquet") if write_parquet else None
    if parquet_path is not None:
        out.to_parquet(parquet_path, index=False)
    master_path = master_parquet_path = None
    if copy_to_master:
        MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        master_path = MASTER_RESULTS_DIR / output_path.name
        shutil.copy2(output_path, master_path)
        if parquet_path is not None:
            master_parquet_path = MASTER_RESULTS_DIR / parquet_path.name
            shutil.copy2(parquet_path, master_parquet_path)
    return output_path, master_path, parquet_path, master_parquet_path, out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year-range", required=True, help="Compact range such as 22_26")
    parser.add_argument("--suffix", required=True, help="Solver suffix such as all_rb_se_a2000_4000")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--tov-sc-transfer-multiplier", type=float, default=1.0)
    parser.add_argument("--no-master-copy", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_decomp_weighted_factors(
        year_range=args.year_range,
        suffix=args.suffix,
        base_path=args.base,
        output_name=args.output_name,
        results_dir=args.results_dir,
        decimals=args.decimals,
        transfer_multiplier=args.tov_sc_transfer_multiplier,
        copy_to_master=not args.no_master_copy,
        write_parquet=not args.no_parquet,
    )
    out = paths[-1]
    for path in paths[:-1]:
        if path is not None:
            print(f"Wrote {path}")
    print(f"Rows: {len(out):,}; columns: {len(out.columns):,}")
    print(f"Max display residual: {out[['oRESID', 'dRESID', 'RESID']].abs().max().max():.6f}")


if __name__ == "__main__":
    main()

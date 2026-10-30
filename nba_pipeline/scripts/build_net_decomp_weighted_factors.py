#!/usr/bin/env python3
"""Assemble the research net-continuation DECOMP player table."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PIPELINE_ROOT / "results" / "net_decomp_rolling" / "components"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results" / "net_decomp_rolling"

SHOT_COMPONENTS = {
    "RIM_FREQ": "DECOMP_NET_RIM_FREQ",
    "RIM_FG": "DECOMP_NET_RIM_FG",
    "MID_FG": "DECOMP_NET_MID_FG",
    "THREE_FREQ": "DECOMP_NET_THREE_FREQ",
    "THREE_FG": "DECOMP_NET_THREE_FG",
}
OTHER_COMPONENTS = {
    "EFG_PARENT": "DECOMP_NET_EFG",
    "FT": "ALT_FT",
    "TOV_VALUE": "ALT_TOV_NET_VALUE",
    "SC": "SECOND_CHANCE_NET",
    "BADPASS_TOV_VALUE": "ALT_BADPASS_TOV_NET_VALUE",
    "SCORING_TOV_VALUE": "ALT_SCORING_TOV_NET_VALUE",
}
MAX_SEPARATE_PARENT_GAP = 0.30


def result_filename(prefix: str, year_range: str, suffix: str) -> str:
    result_suffix = suffix.replace("all", "all_pure", 1) if prefix == "RAPM" else suffix
    return f"{prefix.lower()}_{year_range}_{result_suffix}_results.csv"


def load_result(prefix: str, year_range: str, suffix: str, results_dir: Path) -> pd.DataFrame:
    path = results_dir / result_filename(prefix, year_range, suffix)
    if not path.exists():
        raise FileNotFoundError(f"Missing net DECOMP input: {path}")
    frame = pd.read_csv(path)
    required = {"player_id", "off", "def", "net_rapm"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _attach(out: pd.DataFrame, component: pd.DataFrame, label: str) -> pd.DataFrame:
    values = component[["player_id", "off", "def"]].rename(
        columns={"off": f"oDECOMP_{label}", "def": f"dDECOMP_{label}_solver"}
    )
    out = out.merge(values, on="player_id", how="left", validate="one_to_one")
    missing = out[f"oDECOMP_{label}"].isna()
    if missing.any():
        ids = out.loc[missing, "player_id"].tolist()[:10]
        raise ValueError(f"Net DECOMP {label} is missing base players: {ids}")
    out[f"dDECOMP_{label}"] = -out.pop(f"dDECOMP_{label}_solver")
    out[f"nDECOMP_{label}"] = out[f"oDECOMP_{label}"] + out[f"dDECOMP_{label}"]
    return out


def build_net_decomp_weighted_factors(
    year_range: str,
    suffix: str,
    output_name: str,
    results_dir: Path = RESULTS_DIR,
    master_results_dir: Path = MASTER_RESULTS_DIR,
    latest_year: int | None = None,
    decimals: int = 3,
) -> tuple[Path, Path, Path, Path, pd.DataFrame]:
    rapm = load_result("RAPM", year_range, suffix, results_dir)
    required = {"player_id", "player_name", "off", "def", "net_rapm", "possessions", "off_poss", "def_poss"}
    missing = required - set(rapm.columns)
    if missing:
        raise ValueError(f"RAPM base is missing columns: {sorted(missing)}")
    if latest_year is None:
        end = int(year_range.split("_")[-1])
        latest_year = 2000 + end if end <= 50 else 1900 + end

    out = rapm[["player_id", "player_name", "off", "def", "net_rapm", "possessions", "off_poss", "def_poss"]].copy()
    out.insert(2, "Latest_Year", int(latest_year))
    out["def"] = -out["def"]
    out.insert(3, "DECOMP_PRICING", "net_continuation_kappa")

    for label, prefix in {**SHOT_COMPONENTS, **OTHER_COMPONENTS}.items():
        out = _attach(out, load_result(prefix, year_range, suffix, results_dir), label)

    for side in ("o", "d"):
        children = sum(out[f"{side}DECOMP_{label}"] for label in SHOT_COMPONENTS)
        out[f"{side}DECOMP_EFG_CHILDREN"] = children
        out[f"{side}DECOMP_EFG_PARENT_GAP"] = out[f"{side}DECOMP_EFG_PARENT"] - children
        out[f"{side}DECOMP_RIM"] = out[f"{side}DECOMP_RIM_FREQ"] + out[f"{side}DECOMP_RIM_FG"]
        out[f"{side}DECOMP_THREE"] = out[f"{side}DECOMP_THREE_FREQ"] + out[f"{side}DECOMP_THREE_FG"]
        out[f"{side}DECOMP_TOV_VALUE_split"] = (
            out[f"{side}DECOMP_BADPASS_TOV_VALUE"]
            + out[f"{side}DECOMP_SCORING_TOV_VALUE"]
        )
        out[f"{side}DECOMP_TOV_SPLIT_GAP"] = (
            out[f"{side}DECOMP_TOV_VALUE"] - out[f"{side}DECOMP_TOV_VALUE_split"]
        )
        out[f"{side}FC"] = children + out[f"{side}DECOMP_FT"] + out[f"{side}DECOMP_TOV_VALUE"]
        out[f"{side}DISPLAY_SUM"] = out[f"{side}FC"] + out[f"{side}DECOMP_SC"]
        total = out["off"] if side == "o" else out["def"]
        out[f"{side}RESID"] = total - out[f"{side}DISPLAY_SUM"]

    component_labels = [
        "EFG_PARENT", "EFG_CHILDREN", "EFG_PARENT_GAP", "RIM", "RIM_FREQ",
        "RIM_FG", "MID_FG", "THREE", "THREE_FREQ", "THREE_FG", "FT",
        "TOV_VALUE", "SC", "BADPASS_TOV_VALUE", "SCORING_TOV_VALUE",
        "TOV_VALUE_split", "TOV_SPLIT_GAP",
    ]
    for label in component_labels:
        out[f"nDECOMP_{label}"] = out[f"oDECOMP_{label}"] + out[f"dDECOMP_{label}"]
    out["nFC"] = out["oFC"] + out["dFC"]
    out["nDISPLAY_SUM"] = out["oDISPLAY_SUM"] + out["dDISPLAY_SUM"]
    out["RESID"] = out["net_rapm"] - out["nDISPLAY_SUM"]

    identity_cols = [
        "oDECOMP_EFG_PARENT_GAP", "dDECOMP_EFG_PARENT_GAP",
        "oDECOMP_TOV_SPLIT_GAP", "dDECOMP_TOV_SPLIT_GAP",
    ]
    # rapm.py exports each independently solved coefficient at two decimals.
    # The source rows close at machine precision, but adding five rounded child
    # CSVs can differ from the separately rounded parent by a few hundredths.
    if out[identity_cols].abs().max().max() > 0.075:
        raise ValueError(f"Net DECOMP coefficient identity gap is unexpectedly large: {out[identity_cols].abs().max().to_dict()}")
    # Short current windows can have very low-possession players whose
    # separately regularized parent and component fits differ slightly more
    # than the historical five-season family.  Keep the residual explicit and
    # fail above the observed/reviewed 0.30 ceiling rather than balancing it
    # away.  The source-row and child-parent identities are guarded separately.
    if out[["oRESID", "dRESID", "RESID"]].abs().max().max() > MAX_SEPARATE_PARENT_GAP:
        raise ValueError(
            "Net DECOMP separately solved component sum is unexpectedly far from RAPM: "
            f"{out[['oRESID', 'dRESID', 'RESID']].abs().max().to_dict()}"
        )

    numeric = out.select_dtypes(include="number").columns.difference(["player_id", "Latest_Year"])
    out[numeric] = out[numeric].round(decimals)
    out = out.sort_values("net_rapm", ascending=False).reset_index(drop=True)

    results_dir.parent.mkdir(parents=True, exist_ok=True)
    output_path = results_dir.parent / output_name
    parquet_path = output_path.with_suffix(".parquet")
    out.to_csv(output_path, index=False)
    out.to_parquet(parquet_path, index=False)
    master_results_dir.mkdir(parents=True, exist_ok=True)
    master_path = master_results_dir / output_name
    master_parquet_path = master_path.with_suffix(".parquet")
    shutil.copy2(output_path, master_path)
    shutil.copy2(parquet_path, master_parquet_path)
    return output_path, master_path, parquet_path, master_parquet_path, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year-range", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--master-results-dir", type=Path, default=MASTER_RESULTS_DIR)
    args = parser.parse_args()
    paths = build_net_decomp_weighted_factors(
        args.year_range,
        args.suffix,
        args.output_name,
        results_dir=args.results_dir,
        master_results_dir=args.master_results_dir,
    )
    print(f"Wrote {paths[0]} and {paths[1]} ({len(paths[-1]):,} rows)")


if __name__ == "__main__":
    main()

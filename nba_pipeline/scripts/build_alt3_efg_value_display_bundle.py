#!/usr/bin/env python3
"""Build an EFG-value display bundle from solved RAPM component files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PIPELINE_ROOT / "results"


ATOMIC_EFG = {
    "rim_freq": "ALT_EFG_RIM_FREQ",
    "rim_fg": "ALT_EFG_RIM_FG",
    "mid_freq": "ALT_EFG_MID_FREQ",
    "mid_fg": "ALT_EFG_MID_FG",
    "three_freq": "ALT_EFG_THREE_FREQ",
    "three_fg": "ALT_EFG_THREE_FG",
}

OTHER_COMPONENTS = {
    "ft": "ALT_FT",
    "ft_freq": "ALT_FT_FREQ",
    "ft_severity": "ALT_FT_SEVERITY",
    "tov_value": "ALT_TOV_VALUE",
    "sc_clean": "SECOND_CHANCE_CLEAN",
    "badpass_tov_value": "ALT_BADPASS_TOV_VALUE",
    "scoring_tov_value": "ALT_SCORING_TOV_VALUE",
    "assist_points": "ASSIST_POINTS",
    "rim_assist": "RIM_ASSIST",
}


def file_stem(prefix: str) -> str:
    return prefix.lower()


def result_path(prefix: str, year_range: str, suffix: str, results_dir: Path) -> Path:
    pure = "_pure" if prefix == "RAPM" else ""
    return results_dir / f"{file_stem(prefix)}_{year_range}_{suffix.replace('all', 'all' + pure, 1) if pure else suffix}_results.csv"


def load_component(prefix: str, year_range: str, suffix: str, results_dir: Path) -> pd.DataFrame:
    path = result_path(prefix, year_range, suffix, results_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"player_id", "player_name", "off", "def", "net_rapm", "possessions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    keep = ["player_id", "player_name", "off", "def", "net_rapm", "possessions"]
    if "off_poss" in df.columns:
        keep.append("off_poss")
    if "def_poss" in df.columns:
        keep.append("def_poss")
    return df[keep].copy()


def merge_component(
    out: pd.DataFrame,
    component: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    comp = component[["player_id", "off", "def", "net_rapm"]].rename(
        columns={
            "off": f"o_{name}",
            "def": f"d_{name}",
            "net_rapm": f"n_{name}",
        }
    )
    return out.merge(comp, on="player_id", how="inner", validate="one_to_one")


def build_bundle(
    year_range: str,
    suffix: str,
    results_dir: Path,
    output: Path,
    decimals: int,
) -> pd.DataFrame:
    rapm = load_component("RAPM", year_range, suffix, results_dir)
    out = rapm.rename(columns={"net_rapm": "net"})[
        ["player_id", "player_name", "possessions", "off_poss", "def_poss", "off", "def", "net"]
    ].copy()

    for name, prefix in ATOMIC_EFG.items():
        out = merge_component(out, load_component(prefix, year_range, suffix, results_dir), name)
    for name, prefix in OTHER_COMPONENTS.items():
        out = merge_component(out, load_component(prefix, year_range, suffix, results_dir), name)

    for side in ["o", "d", "n"]:
        out[f"{side}_rim_value"] = out[f"{side}_rim_freq"] + out[f"{side}_rim_fg"]
        out[f"{side}_mid_value"] = out[f"{side}_mid_freq"] + out[f"{side}_mid_fg"]
        out[f"{side}_three_value"] = out[f"{side}_three_freq"] + out[f"{side}_three_fg"]
        out[f"{side}_shot_atoms_value"] = (
            out[f"{side}_rim_value"] + out[f"{side}_mid_value"] + out[f"{side}_three_value"]
        )
        out[f"{side}_scoring_tov_value"] = out[f"{side}_tov_value"] - out[f"{side}_badpass_tov_value"]
        out[f"{side}_tov_loss_split"] = out[f"{side}_badpass_tov_value"] + out[f"{side}_scoring_tov_value"]

    out["o_efg_baseline"] = (
        out["off"] - out["o_shot_atoms_value"] - out["o_ft"] - out["o_tov_value"] - out["o_sc_clean"]
    )
    out["d_efg_baseline"] = (
        out["def"] - out["d_shot_atoms_value"] - out["d_ft"] - out["d_tov_value"] - out["d_sc_clean"]
    )
    out["n_efg_baseline"] = out["o_efg_baseline"] - out["d_efg_baseline"]

    for side in ["o", "d", "n"]:
        out[f"{side}_shot_value"] = out[f"{side}_efg_baseline"] + out[f"{side}_shot_atoms_value"]

    out["o_display_sum"] = out["o_shot_value"] + out["o_ft"] + out["o_tov_value"] + out["o_sc_clean"]
    out["d_display_sum"] = out["d_shot_value"] + out["d_ft"] + out["d_tov_value"] + out["d_sc_clean"]
    out["n_display_sum"] = out["o_display_sum"] - out["d_display_sum"]
    out["off_resid"] = out["off"] - out["o_display_sum"]
    out["def_resid"] = out["def"] - out["d_display_sum"]
    out["net_resid"] = out["net"] - out["n_display_sum"]

    column_order = [
        "player_id",
        "player_name",
        "possessions",
        "off_poss",
        "def_poss",
        "off",
        "def",
        "net",
        "o_shot_value",
        "o_efg_baseline",
        "o_shot_atoms_value",
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
        "o_display_sum",
        "off_resid",
        "d_shot_value",
        "d_efg_baseline",
        "d_shot_atoms_value",
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
        "d_display_sum",
        "def_resid",
        "n_shot_value",
        "n_efg_baseline",
        "n_shot_atoms_value",
        "n_ft",
        "n_ft_freq",
        "n_ft_severity",
        "n_tov_value",
        "n_sc_clean",
        "n_display_sum",
        "net_resid",
        "o_badpass_tov_value",
        "o_scoring_tov_value",
        "o_tov_loss_split",
        "d_badpass_tov_value",
        "d_scoring_tov_value",
        "d_tov_loss_split",
        "n_badpass_tov_value",
        "n_scoring_tov_value",
        "n_tov_loss_split",
        "o_assist_points",
        "d_assist_points",
        "n_assist_points",
        "o_rim_assist",
        "d_rim_assist",
        "n_rim_assist",
        "n_rim_freq",
        "n_rim_fg",
        "n_mid_freq",
        "n_mid_fg",
        "n_three_freq",
        "n_three_fg",
        "n_rim_value",
        "n_mid_value",
        "n_three_value",
    ]
    out = out[column_order].sort_values("net", ascending=False)
    numeric_cols = out.select_dtypes(include="number").columns.difference(["player_id"])
    out[numeric_cols] = out[numeric_cols].round(decimals)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year-range", required=True, help="Compact year range, e.g. 20_26")
    parser.add_argument("--suffix", required=True, help="Result suffix after year range, e.g. all_a2000_4000_td500")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_bundle(
        year_range=args.year_range,
        suffix=args.suffix,
        results_dir=args.results_dir,
        output=args.output,
        decimals=args.decimals,
    )
    print(f"Wrote {args.output}")
    print(f"Rows: {len(out):,}; columns: {len(out.columns):,}")
    for col in ["off_resid", "def_resid", "net_resid"]:
        print(f"{col}: max_abs={out[col].abs().max():.6f} mean_abs={out[col].abs().mean():.6f}")


if __name__ == "__main__":
    main()

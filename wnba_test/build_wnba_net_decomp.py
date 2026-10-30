#!/usr/bin/env python3
"""Build the isolated WNBA single-season net-continuation VPM board.

This is the WNBA analogue of the NBA kappa/net-DECOMP research lane.  It
reuses the frozen WNBA 5000/40000 likelihood penalties and 1.5 prior scale,
but refits every factor-specific box prior to net-continuation annual labels.
The live gross WNBA board is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import (
    DEFAULT_OUTPUT as GROSS_PRIOR_OUTPUT,
    build_annual_targets,
    build_box_priors_for_year,
    load_box_features,
    load_frozen_spec,
)
from wnba_test.clean_decomp_wnba import (
    PROCESSED_DIR,
    RESULTS_DIR,
    _component_series,
    _load_clean_total_df,
    _load_name_map,
    _prepare_direct_ridge_problem,
    run_metric,
    validate_net_first_chance_alias_identity,
)


NET_TARGET_PREFIXES = [
    "ALT_EFG_NET_RIM_FREQ",
    "ALT_EFG_NET_RIM_FG",
    "ALT_EFG_NET_MID_FREQ",
    "ALT_EFG_NET_MID_FG",
    "ALT_EFG_NET_THREE_FREQ",
    "ALT_EFG_NET_THREE_FG",
    "ALT_FT",
    "ALT_FT_FREQ",
    "ALT_FC_NET_COMPLETION",
    "ALT_TOV_NET_VALUE",
    "ALT_BADPASS_TOV_NET_VALUE",
    "SECOND_CHANCE_NET",
]

NET_ADDITIVE_PREFIXES = [
    "ALT_EFG_NET_RIM_FREQ",
    "ALT_EFG_NET_RIM_FG",
    "ALT_EFG_NET_MID_FREQ",
    "ALT_EFG_NET_MID_FG",
    "ALT_EFG_NET_THREE_FREQ",
    "ALT_EFG_NET_THREE_FG",
    "ALT_FT",
    "ALT_FC_NET_COMPLETION",
    "SECOND_CHANCE_NET",
]

DEFAULT_OUTPUT = RESULTS_DIR / "wnba_net_decomp"
DEFAULT_GROSS_BOARD = RESULTS_DIR / "current" / "wnba_current_metrics.csv"
DEFAULT_SOURCE_MANIFEST = RESULTS_DIR / "current" / "latest_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_component(
    results: dict[str, pd.DataFrame],
    prefix: str,
    side: str,
    player_ids: pd.Series,
) -> pd.Series:
    values = _component_series(results, prefix, side, player_ids)
    return values if side == "off" else -values


def _build_board(
    priors: dict[str, pd.DataFrame],
    spec: dict[str, Any],
    year: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    name_map = _load_name_map()
    years = [year % 100]
    results: dict[str, pd.DataFrame] = {}
    solve_prefixes = [
        "RAPM",
        *NET_ADDITIVE_PREFIXES,
        "ALT_FT_FREQ",
        "ALT_TOV_NET_VALUE",
        "ALT_BADPASS_TOV_NET_VALUE",
        "ALT_SCORING_TOV_NET_VALUE",
    ]
    for prefix in solve_prefixes:
        if prefix in results:
            continue
        def_alpha = (
            float(spec["direct_tov_def_alpha"])
            if prefix in {
                "ALT_TOV_NET_VALUE",
                "ALT_BADPASS_TOV_NET_VALUE",
                "ALT_SCORING_TOV_NET_VALUE",
            }
            else float(spec["def_alpha"])
        )
        logging.info("Solving WNBA net component %s", prefix)
        results[prefix] = run_metric(
            prefix,
            years,
            "RS",
            name_map,
            off_alpha=float(spec["off_alpha"]),
            def_alpha=def_alpha,
            time_decay_days=None,
            prior_results=priors.get(prefix),
            prior_scale=float(spec["prior_scale"]),
        )

    base = results["RAPM"].copy()
    base["player_id"] = base["player_id"].astype(str)
    out = base[["player_id", "player_name", "off_poss", "def_poss"]].copy()
    out["season"] = int(year)
    out["possessions"] = out["off_poss"] + out["def_poss"]
    out["off"] = pd.to_numeric(base["off"], errors="coerce")
    out["def"] = -pd.to_numeric(base["def"], errors="coerce")
    out["vpm"] = out["off"] + out["def"]
    out["net_rapm"] = out["vpm"]
    out["prior_off"] = pd.to_numeric(base["prior_off"], errors="coerce")
    out["prior_def"] = -pd.to_numeric(base["prior_def"], errors="coerce")
    out["prior_vpm"] = out["prior_off"] + out["prior_def"]
    out["lineup_update_off"] = pd.to_numeric(base["delta_off"], errors="coerce")
    out["lineup_update_def"] = -pd.to_numeric(base["delta_def"], errors="coerce")
    out["lineup_update_vpm"] = out["lineup_update_off"] + out["lineup_update_def"]
    out["has_prior"] = pd.to_numeric(base["has_prior"], errors="coerce").fillna(0).astype(int)

    prefix_to_name = {
        "ALT_EFG_NET_RIM_FREQ": "RIM_FREQ",
        "ALT_EFG_NET_RIM_FG": "RIM_FG",
        "ALT_EFG_NET_MID_FREQ": "MID_FREQ",
        "ALT_EFG_NET_MID_FG": "MID_FG",
        "ALT_EFG_NET_THREE_FREQ": "THREE_FREQ",
        "ALT_EFG_NET_THREE_FG": "THREE_FG",
        "ALT_FT": "FT",
        "ALT_FC_NET_COMPLETION": "COMPLETION",
        "SECOND_CHANCE_NET": "SC",
        "ALT_TOV_NET_VALUE": "TOV",
        "ALT_BADPASS_TOV_NET_VALUE": "BADPASS_TOV",
        "ALT_SCORING_TOV_NET_VALUE": "SCORING_TOV",
        "ALT_FT_FREQ": "FT_FREQ",
    }
    player_ids = out["player_id"]
    for prefix, name in prefix_to_name.items():
        for side in ("off", "def"):
            side_label = "o" if side == "off" else "d"
            out[f"{side_label}{name}"] = _positive_component(
                results, prefix, side, player_ids
            )

    for side, total in (("o", "off"), ("d", "def")):
        # Publish direct TOV value.  The exact completion-minus-direct-TOV
        # complement remains shot value, matching the corrected gross reader.
        out[f"{side}MID_FG"] += out[f"{side}COMPLETION"] - out[f"{side}TOV"]
        out[f"{side}FT_SEVERITY"] = out[f"{side}FT"] - out[f"{side}FT_FREQ"]
        out[f"{side}RIM"] = out[f"{side}RIM_FREQ"] + out[f"{side}RIM_FG"]
        out[f"{side}MID"] = out[f"{side}MID_FREQ"] + out[f"{side}MID_FG"]
        out[f"{side}THREE"] = out[f"{side}THREE_FREQ"] + out[f"{side}THREE_FG"]
        out[f"{side}SHOT"] = out[f"{side}RIM"] + out[f"{side}MID"] + out[f"{side}THREE"]
        out[f"{side}TS"] = out[f"{side}SHOT"] + out[f"{side}FT"]
        out[f"{side}TOV_SPLIT_GAP"] = (
            out[f"{side}TOV"]
            - out[f"{side}BADPASS_TOV"]
            - out[f"{side}SCORING_TOV"]
        )
        out[f"{side}DISPLAY_SUM"] = out[f"{side}TS"] + out[f"{side}TOV"] + out[f"{side}SC"]
        out[f"{side}RESID"] = out[total] - out[f"{side}DISPLAY_SUM"]

    component_names = [
        "RIM_FREQ", "RIM_FG", "MID_FREQ", "MID_FG", "THREE_FREQ",
        "THREE_FG", "FT", "TOV", "SC", "TS",
    ]
    for name in component_names:
        out[f"n{name}"] = out[f"o{name}"] + out[f"d{name}"]
    out["nDISPLAY_SUM"] = out["oDISPLAY_SUM"] + out["dDISPLAY_SUM"]
    out["RESID"] = out["vpm"] - out["nDISPLAY_SUM"]
    out["rank"] = out["vpm"].rank(method="min", ascending=False).astype(int)
    qualified = out["possessions"].ge(1000)
    out["qualified_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[qualified, "qualified_rank"] = (
        out.loc[qualified, "vpm"].rank(method="min", ascending=False).astype(int)
    )
    out["model_variant"] = "single_season_box_trueprior_netcont_v1"

    qc = {
        "offense_residual_max": float(out["oRESID"].abs().max()),
        "defense_residual_max": float(out["dRESID"].abs().max()),
        "net_residual_max": float(out["RESID"].abs().max()),
        "offense_tov_split_gap_max": float(out["oTOV_SPLIT_GAP"].abs().max()),
        "defense_tov_split_gap_max": float(out["dTOV_SPLIT_GAP"].abs().max()),
    }
    if max(qc.values()) > 0.01:
        raise RuntimeError(f"WNBA net board failed additive QC: {qc}")
    return out.sort_values(["vpm", "possessions"], ascending=[False, False]), qc


def _comparison(net: pd.DataFrame, gross_path: Path) -> pd.DataFrame:
    gross = pd.read_csv(gross_path, dtype={"player_id": str})
    gross["player_id"] = gross["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = ["player_id", "off", "def", "net_rapm"]
    gross = gross[keep].rename(
        columns={"off": "gross_off", "def": "gross_def", "net_rapm": "gross_vpm"}
    )
    compare = net.merge(gross, on="player_id", how="left", validate="one_to_one")
    compare["delta_off"] = compare["off"] - compare["gross_off"]
    compare["delta_def"] = compare["def"] - compare["gross_def"]
    compare["delta_vpm"] = compare["vpm"] - compare["gross_vpm"]
    return compare[
        [
            "player_id", "player_name", "possessions", "off", "def", "vpm",
            "gross_off", "gross_def", "gross_vpm", "delta_off", "delta_def",
            "delta_vpm", "oTS", "oTOV", "oSC", "dTS", "dTOV", "dSC",
        ]
    ].sort_values("vpm", ascending=False)


def _kappa_summary(years: range) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in years:
        suffixes = ["", "_PS"] if year < max(years) else [""]
        for suffix in suffixes:
            path = PROCESSED_DIR / f"FIRST_CHANCE{year % 100:02d}{suffix}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(
                path,
                usecols=["FC_KAPPA_RIM", "FC_KAPPA_MID", "FC_KAPPA_THREE"],
                nrows=1,
            )
            rows.append(
                {
                    "season": year,
                    "season_type": "PS" if suffix else "RS",
                    "rim": float(frame.iloc[0]["FC_KAPPA_RIM"]),
                    "mid": float(frame.iloc[0]["FC_KAPPA_MID"]),
                    "three": float(frame.iloc[0]["FC_KAPPA_THREE"]),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-report",
        type=Path,
        default=GROSS_PRIOR_OUTPUT / "report.json",
    )
    parser.add_argument("--gross-board", type=Path, default=DEFAULT_GROSS_BOARD)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument("--rebuild-targets", action="store_true")
    parser.add_argument("--year", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.year != 2026:
        raise RuntimeError("the frozen WNBA net-continuation lane currently supports 2026")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_frozen_spec(args.selection_report)

    # Prove the per-possession alias algebra on every annual source before any
    # target or prior is fit.
    alias_gaps: dict[str, float] = {}
    for year in range(2022, args.year + 1):
        suffixes = ["", "_PS"] if year < args.year else [""]
        for suffix in suffixes:
            path = PROCESSED_DIR / f"FIRST_CHANCE{year % 100:02d}{suffix}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            alias_gaps[f"{year}{suffix or '_RS'}"] = validate_net_first_chance_alias_identity(frame)

    build_annual_targets(
        args.output_dir,
        rebuild=args.rebuild_targets,
        target_off_alpha=float(spec["box_target_off_alpha"]),
        target_def_alpha=float(spec["box_target_def_alpha"]),
        target_prefixes=NET_TARGET_PREFIXES,
    )
    box_by_year = {
        year: load_box_features(year) for year in range(2022, args.year + 1)
    }
    current = _load_clean_total_df([args.year % 100], "RS")
    universe = list(_prepare_direct_ridge_problem(current)["all_players"])
    priors, readable, metadata = build_box_priors_for_year(
        args.output_dir,
        box_by_year,
        target_year=args.year,
        train_years=list(range(2022, args.year)),
        universe=universe,
        target_prefixes=NET_TARGET_PREFIXES,
        additive_prefixes=NET_ADDITIVE_PREFIXES,
    )
    priors_path = args.output_dir / f"box_priors_netcont_{args.year}.csv"
    readable.to_csv(priors_path, index=False)
    board, qc = _build_board(priors, spec, args.year)
    board_path = args.output_dir / f"wnba_single_season_vpm_netcont_{args.year}.csv"
    board.to_csv(board_path, index=False, float_format="%.6f")

    comparison_path = args.output_dir / f"gross_vs_netcont_{args.year}.csv"
    comparison = _comparison(board, args.gross_board)
    comparison.to_csv(comparison_path, index=False, float_format="%.6f")
    kappa = _kappa_summary(range(2022, args.year + 1))
    source_manifest = (
        json.loads(args.source_manifest.read_text())
        if args.source_manifest.exists()
        else {}
    )
    current_source = source_manifest.get("source", {}).get("after", {})
    manifest = {
        "status": "research_only",
        "model_variant": "single_season_box_trueprior_netcont_v1",
        "as_of": source_manifest.get("as_of"),
        "season": args.year,
        "season_type": "RS",
        "board_summary": {
            "players": int(len(board)),
            "qualified_1000_possessions": int(board["possessions"].ge(1000).sum()),
            "games": current_source.get("games"),
        },
        "solve": {
            "off_alpha": spec["off_alpha"],
            "def_alpha": spec["def_alpha"],
            "direct_tov_def_alpha": spec["direct_tov_def_alpha"],
            "prior_scale": spec["prior_scale"],
            "selection_folds": spec["selection_folds"],
        },
        "construction": {
            "shot_value": "actual points plus zone kappa on first-chance misses",
            "second_chance": "actual second-chance points minus the same kappa",
            "turnover": "direct loss priced at mean non-turnover FC_TS_NET",
            "gross_plus_second_chance_equals_net_plus_net_second_chance": True,
            "priors_refit_to_net_labels": True,
        },
        "row_alias_max_gaps": alias_gaps,
        "board_qc": qc,
        "kappa_by_season_type": kappa,
        "artifacts": {
            "board": {"path": str(board_path), "sha256": _sha256(board_path), "rows": len(board)},
            "comparison": {"path": str(comparison_path), "sha256": _sha256(comparison_path), "rows": len(comparison)},
            "priors": {"path": str(priors_path), "sha256": _sha256(priors_path), "rows": len(readable)},
            "gross_board": {"path": str(args.gross_board), "sha256": _sha256(args.gross_board)},
            "source_manifest": (
                {
                    "path": str(args.source_manifest),
                    "sha256": _sha256(args.source_manifest),
                    "source_sha256": current_source.get("sha256"),
                }
                if args.source_manifest.exists()
                else None
            ),
        },
        "model_metadata": metadata,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    logging.info("Wrote %s", board_path)
    logging.info("Wrote %s", comparison_path)
    logging.info("Wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

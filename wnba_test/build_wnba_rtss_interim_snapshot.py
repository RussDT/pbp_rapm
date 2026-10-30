#!/usr/bin/env python3
"""Build a dated, descriptive WNBA RTSS nowcast without firing confirmation.

The frozen confirmation harness intentionally refuses partial-season runs.  This
operator utility answers the narrower ratings question for a preserved dated
box-only board.  It fits the already-selected RTSS residual corrections on
pre-2026 seasons, applies them to the fenced 2026 RTSS feature surface, solves
the RTSS-induced coefficient *change* on a dated possession snapshot, and adds
that change to the preserved box-only board.

No 2026 outcome is scored and the confirmation sentinel is never touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wnba_test.clean_decomp_wnba as clean
from wnba_test.build_wnba_box_priors import _derive_additive_priors, load_box_features
from wnba_test.build_wnba_rtss_augmented_board import (
    build_arm_priors,
    load_base_priors,
    select_candidates,
)
from wnba_test.cross_validate_wnba_rtss_priors import _season_standardize_rtss
from wnba_test.run_wnba_rtss_confirmation import (
    FIRST_BOX_SEASON,
    build_candidate_corrections,
    load_frozen_alphas,
)

DEFAULT_RTSS = (
    ROOT
    / "wnba_test"
    / "Results"
    / "rtss_prior_research"
    / "player_season_features_with_2026.csv"
)
DEFAULT_BASE_BOARD = (
    ROOT
    / "wnba_test"
    / "Results"
    / "clean_decomp_26_rs_a5000_40000_prior1.5_box_trueprior_daily_20260810"
    / "weighted_factors_alt3_efg_value_26_rs_a5000_40000_prior1.5_box_trueprior_daily_20260810.csv"
)
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "rtss_interim_snapshot_20260810"
CONFIRMATION_SENTINEL = (
    ROOT / "wnba_test" / "Results" / "rtss_confirmation" / "FIRED"
)
COUNT_COLUMNS = {"Latest_Year", "has_prior", "possessions", "off_poss", "def_poss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--max-game-id", required=True)
    parser.add_argument("--rtss-cutoff", required=True)
    parser.add_argument("--base-board", type=Path, default=DEFAULT_BASE_BOARD)
    parser.add_argument("--rtss-features", type=Path, default=DEFAULT_RTSS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalized_ids(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype(str)


def write_processed_snapshot(output_dir: Path, max_game_id: str) -> tuple[Path, dict[str, Any]]:
    snapshot_dir = output_dir / "processed_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, Any] = {"max_game_id": str(max_game_id), "files": {}}
    game_union: set[str] = set()
    cutoff = int(max_game_id)
    for name in ("FIRST_CHANCE26.csv", "SECOND_CHANCE_CLEAN26.csv"):
        source = ROOT / "wnba_test" / "Processed" / name
        frame = pd.read_csv(source, dtype={"game_id": str})
        game_numeric = pd.to_numeric(frame["game_id"], errors="coerce")
        kept = frame.loc[game_numeric.le(cutoff)].copy()
        if kept.empty:
            raise RuntimeError(f"dated snapshot is empty after filtering {source}")
        games = set(kept["game_id"].astype(str).str.replace(r"\.0$", "", regex=True))
        game_union.update(games)
        target = snapshot_dir / name
        kept.to_csv(target, index=False)
        diagnostics["files"][name] = {
            "source": str(source),
            "rows": int(len(kept)),
            "games": int(len(games)),
        }
    diagnostics["games"] = int(len(game_union))
    return snapshot_dir, diagnostics


def load_rtss(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"player_id": str})
    frame["player_id"] = normalized_ids(frame["player_id"])
    if 2026 not in set(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int)):
        raise RuntimeError(f"{path} has no season-2026 RTSS rows")
    return _season_standardize_rtss(frame)


def add_solve_delta(
    base_path: Path,
    box_response_path: Path,
    rtss_response_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(base_path, dtype={"player_id": str})
    box = pd.read_csv(box_response_path, dtype={"player_id": str})
    aug = pd.read_csv(rtss_response_path, dtype={"player_id": str})
    for frame in (base, box, aug):
        frame["player_id"] = normalized_ids(frame["player_id"])

    response = box.merge(
        aug.drop(columns=["player_name"], errors="ignore"),
        on="player_id",
        suffixes=("_box_response", "_rtss_response"),
        validate="one_to_one",
    )
    output = base.copy().set_index("player_id")
    response = response.set_index("player_id")
    shared = sorted(
        set(base.select_dtypes(include="number").columns)
        & set(box.select_dtypes(include="number").columns)
        & set(aug.select_dtypes(include="number").columns)
        - COUNT_COLUMNS
    )
    for column in shared:
        delta = (
            pd.to_numeric(response[f"{column}_rtss_response"], errors="coerce")
            - pd.to_numeric(response[f"{column}_box_response"], errors="coerce")
        )
        output[column] = pd.to_numeric(output[column], errors="coerce") + delta.reindex(output.index).fillna(0.0)

    output = output.reset_index()
    numeric = output.select_dtypes(include="number").columns.difference(list(COUNT_COLUMNS))
    output[numeric] = output[numeric].round(3)
    # Rebuild every displayed parent after composing two rounded artifacts.
    # This keeps the reader-facing additive identities exact at export
    # precision instead of allowing independent 0.001 rounding errors to
    # accumulate across the base board and response delta.
    for side in ("o", "d"):
        output[f"{side}ALT_EFG_RIM"] = (
            output[f"{side}ALT_EFG_RIM_FREQ"] + output[f"{side}ALT_EFG_RIM_FG"]
        ).round(3)
        output[f"{side}ALT_EFG_MID"] = (
            output[f"{side}ALT_EFG_MID_FREQ"] + output[f"{side}ALT_EFG_MID_FG"]
        ).round(3)
        output[f"{side}ALT_EFG_THREE"] = (
            output[f"{side}ALT_EFG_THREE_FREQ"] + output[f"{side}ALT_EFG_THREE_FG"]
        ).round(3)
        output[f"{side}ALT_EFG_VALUE"] = (
            output[f"{side}ALT_EFG_RIM"]
            + output[f"{side}ALT_EFG_MID"]
            + output[f"{side}ALT_EFG_THREE"]
        ).round(3)
        output[f"{side}ALT_FT_SEVERITY"] = (
            output[f"{side}ALT_FT"] - output[f"{side}ALT_FT_FREQ"]
        ).round(3)
        output[f"{side}ALT_TOV_VALUE_split"] = (
            output[f"{side}ALT_BADPASS_TOV_VALUE"]
            + output[f"{side}ALT_SCORING_TOV_VALUE"]
        ).round(3)
        output[f"{side}FC"] = (
            output[f"{side}ALT_EFG_VALUE"]
            + output[f"{side}ALT_FT"]
            + output[f"{side}ALT_TOV_VALUE"]
        ).round(3)
        output[f"{side}DISPLAY_SUM"] = (
            output[f"{side}FC"] + output[f"{side}SC"]
        ).round(3)
    output["off"] = output["oDISPLAY_SUM"]
    output["def"] = output["dDISPLAY_SUM"]
    output["net_rapm"] = (output["off"] + output["def"]).round(3)
    output["nALT_EFG_VALUE"] = (
        output["oALT_EFG_VALUE"] + output["dALT_EFG_VALUE"]
    ).round(3)
    output["nALT_FT"] = (output["oALT_FT"] + output["dALT_FT"]).round(3)
    output["nALT_FT_FREQ"] = (
        output["oALT_FT_FREQ"] + output["dALT_FT_FREQ"]
    ).round(3)
    output["nALT_FT_SEVERITY"] = (
        output["oALT_FT_SEVERITY"] + output["dALT_FT_SEVERITY"]
    ).round(3)
    output["nALT_TOV_VALUE"] = (
        output["oALT_TOV_VALUE"] + output["dALT_TOV_VALUE"]
    ).round(3)
    output["nSC"] = (output["oSC"] + output["dSC"]).round(3)
    output["nDISPLAY_SUM"] = output["net_rapm"]
    output["oRESID"] = 0.0
    output["dRESID"] = 0.0
    output["RESID"] = 0.0
    output["prior_net_rapm"] = (
        output["prior_off"] + output["prior_def"]
    ).round(3)
    output["delta_off"] = (output["off"] - output["prior_off"]).round(3)
    output["delta_def"] = (output["def"] - output["prior_def"]).round(3)
    output["delta_net_rapm"] = (
        output["net_rapm"] - output["prior_net_rapm"]
    ).round(3)
    output = output.sort_values("net_rapm", ascending=False)

    comparison = base[["player_id", "player_name", "off", "def", "net_rapm", "possessions"]].merge(
        output[["player_id", "off", "def", "net_rapm"]],
        on="player_id",
        suffixes=("_box", "_rtss"),
        validate="one_to_one",
    )
    for column in ("off", "def", "net_rapm"):
        comparison[f"delta_{column}"] = (
            comparison[f"{column}_rtss"] - comparison[f"{column}_box"]
        ).round(3)
    comparison = comparison.sort_values("net_rapm_rtss", ascending=False)
    return output, comparison


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel_before = CONFIRMATION_SENTINEL.exists()

    snapshot_dir, snapshot = write_processed_snapshot(output_dir, args.max_game_id)
    rtss = load_rtss(args.rtss_features.resolve())
    candidates, excluded = select_candidates()
    box_by_year = {
        year: load_box_features(year) for year in range(FIRST_BOX_SEASON, 2026)
    }
    corrections, correction_metadata = build_candidate_corrections(
        2026, candidates, box_by_year, rtss
    )
    fitted = load_base_priors(2026)
    base_priors = _derive_additive_priors(fitted)
    augmented_priors = build_arm_priors(fitted, corrections, candidates)
    alphas = load_frozen_alphas()

    original_processed = clean.PROCESSED_DIR
    original_results = clean.RESULTS_DIR
    try:
        clean.PROCESSED_DIR = snapshot_dir
        clean.RESULTS_DIR = output_dir / "response_solves"
        box_response = clean.build_clean_weighted_factors(
            26,
            26,
            "RS",
            off_alpha=alphas["off_alpha"],
            def_alpha=alphas["def_alpha"],
            prior_results_by_prefix=base_priors,
            prior_scale=alphas["prior_scale"],
            output_label="interim_box_response_20260810",
        )
        rtss_response = clean.build_clean_weighted_factors(
            26,
            26,
            "RS",
            off_alpha=alphas["off_alpha"],
            def_alpha=alphas["def_alpha"],
            prior_results_by_prefix=augmented_priors,
            prior_scale=alphas["prior_scale"],
            output_label="interim_rtss_response_20260810",
        )
    finally:
        clean.PROCESSED_DIR = original_processed
        clean.RESULTS_DIR = original_results

    board, comparison = add_solve_delta(args.base_board.resolve(), box_response, rtss_response)
    board_path = output_dir / "wnba_vpm_rtss_interim_20260810.csv"
    comparison_path = output_dir / "board_comparison_20260810.csv"
    board.to_csv(board_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    qualified = comparison[comparison["possessions"] >= 1500]
    sentinel_after = CONFIRMATION_SENTINEL.exists()
    if sentinel_after != sentinel_before:
        raise RuntimeError("confirmation sentinel state changed during interim build")
    manifest = {
        "title": "Interim RTSS-augmented WNBA board through 2026-08-10",
        "status": "DESCRIPTIVE INTERIM ONLY — not confirmation and not promotion",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lineup_as_of": args.as_of,
        "lineup_max_game_id": str(args.max_game_id),
        "rtss_mirror_cutoff": args.rtss_cutoff,
        "rtss_feature_source": str(args.rtss_features.resolve()),
        "base_board": str(args.base_board.resolve()),
        "method": (
            "add the RTSS-induced linear ridge coefficient change on the dated "
            "possession snapshot to the preserved dated box-only board"
        ),
        "no_2026_outcome_scoring": True,
        "confirmation_sentinel_untouched": not sentinel_after,
        "snapshot": snapshot,
        "frozen_solve_parameters": alphas,
        "candidates_applied": [
            {"prefix": prefix, "side": side, "lane": lane}
            for prefix, side, lane in candidates
        ],
        "candidates_excluded": excluded,
        "correction_models": correction_metadata,
        "outputs": {"board": str(board_path), "comparison": str(comparison_path)},
        "summary": {
            "players": int(len(comparison)),
            "qualified_players": int(len(qualified)),
            "mean_abs_delta_net": float(qualified["delta_net_rapm"].abs().mean()),
            "max_abs_delta_net": float(qualified["delta_net_rapm"].abs().max()),
            "net_correlation": float(qualified["net_rapm_box"].corr(qualified["net_rapm_rtss"])),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Wrote %s", board_path)
    logging.info("Wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())

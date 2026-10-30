#!/usr/bin/env python3
"""Build the five-season WNBA single-season impact history.

The 2023-2026 boards use the frozen factor-specific box-prior specification
selected before the 2026 holdout was opened. The 2022 board is an explicitly
labeled pure-RAPM warm-up season because no earlier lineup season exists for
training a leakage-safe WNBA box-to-RAPM prior.
"""

from __future__ import annotations

import argparse
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
    DIRECT_TOV_DEF_ALPHA,
    DEFAULT_OUTPUT,
    build_annual_targets,
    build_box_priors_for_year,
    load_box_features,
)
from wnba_test.clean_decomp_wnba import (
    RESULTS_DIR,
    _load_clean_total_df,
    _prepare_direct_ridge_problem,
    build_clean_weighted_factors,
)


HISTORY_START = 2022
HISTORY_END = 2026
QUALIFYING_POSSESSIONS = 1000
PRIOR_AUDIT_COLUMNS = [
    "prior_off",
    "prior_def",
    "prior_net_rapm",
    "delta_off",
    "delta_def",
    "delta_net_rapm",
    "has_prior",
]
LEADERBOARD_COLUMNS = [
    "season",
    "season_type",
    "model_variant",
    "rank",
    "qualified_rank",
    "qualified",
    "player_id",
    "player_name",
    "wnba_impact",
    "wnba_offense",
    "wnba_defense",
    "possessions",
    "minutes",
    "box_prior_source",
    "box_net",
    "prior_net_rapm",
    "delta_net_rapm",
    "nALT_EFG_VALUE",
    "nALT_FT",
    "nALT_TOV_VALUE",
    "nSC",
]


def _normalize_id(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
    )


def _season_type(year: int) -> str:
    return "ALL" if year < HISTORY_END else "RS"


def _load_selected_spec(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} is missing; run build_wnba_box_priors.py first"
        )
    report = json.loads(report_path.read_text())
    spec = report.get("selected_spec")
    if not isinstance(spec, dict):
        raise ValueError(f"{report_path} has no selected_spec")
    required = {
        "off_alpha",
        "def_alpha",
        "prior_scale",
        "box_target_off_alpha",
        "box_target_def_alpha",
    }
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"selected_spec is missing {sorted(missing)}")
    return {
        **spec,
        "direct_tov_def_alpha": float(
            spec.get("direct_tov_def_alpha", DIRECT_TOV_DEF_ALPHA)
        ),
    }


def _add_reader_columns(
    frame: pd.DataFrame,
    year: int,
    season_type: str,
    model_variant: str,
    prior_table: pd.DataFrame | None,
) -> pd.DataFrame:
    out = frame.copy()
    out["player_id"] = _normalize_id(out["player_id"])
    for column in PRIOR_AUDIT_COLUMNS:
        if column not in out:
            if column == "has_prior":
                out[column] = 0
            elif column.startswith("delta_"):
                source = {
                    "delta_off": "off",
                    "delta_def": "def",
                    "delta_net_rapm": "net_rapm",
                }[column]
                out[column] = pd.to_numeric(out[source], errors="coerce")
            else:
                out[column] = np.nan

    out["season"] = int(year)
    out["season_type"] = season_type
    out["model_variant"] = model_variant
    out["wnba_offense"] = pd.to_numeric(out["off"], errors="coerce")
    out["wnba_defense"] = pd.to_numeric(out["def"], errors="coerce")
    out["wnba_impact"] = pd.to_numeric(out["net_rapm"], errors="coerce")
    out["rank"] = (
        out["wnba_impact"].rank(method="min", ascending=False).astype("Int64")
    )
    out["qualified"] = (
        pd.to_numeric(out["possessions"], errors="coerce")
        >= QUALIFYING_POSSESSIONS
    )
    qualified_rank = out.loc[out["qualified"], "wnba_impact"].rank(
        method="min", ascending=False
    )
    out["qualified_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[out["qualified"], "qualified_rank"] = qualified_rank.astype("Int64")

    if prior_table is not None:
        readable = prior_table.copy()
        readable["player_id"] = _normalize_id(readable["player_id"])
        keep = [
            "player_id",
            "box_prior_source",
            "minutes",
            "box_off",
            "box_def",
            "box_net",
        ]
        out = out.merge(
            readable[keep],
            on="player_id",
            how="left",
            validate="one_to_one",
        )
    else:
        out["box_prior_source"] = "none_2022_warmup"
        out["minutes"] = np.nan
        out["box_off"] = np.nan
        out["box_def"] = np.nan
        out["box_net"] = np.nan

    leading = [
        "season",
        "season_type",
        "model_variant",
        "rank",
        "qualified_rank",
        "qualified",
        "player_id",
        "player_name",
        "wnba_impact",
        "wnba_offense",
        "wnba_defense",
        "possessions",
        "off_poss",
        "def_poss",
        "minutes",
        "box_prior_source",
        "box_off",
        "box_def",
        "box_net",
        *PRIOR_AUDIT_COLUMNS,
    ]
    remaining = [column for column in out.columns if column not in leading]
    return out[leading + remaining].sort_values(
        ["rank", "possessions"], ascending=[True, False]
    )


def _validate_history(history: pd.DataFrame) -> dict[str, Any]:
    duplicate_keys = int(history.duplicated(["season", "player_id"]).sum())
    missing_core = int(
        history[
            [
                "season",
                "player_id",
                "player_name",
                "wnba_offense",
                "wnba_defense",
                "wnba_impact",
                "possessions",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )
    net_identity = (
        history["wnba_impact"]
        - history["wnba_offense"]
        - history["wnba_defense"]
    ).abs()
    display_gap = pd.concat(
        [
            pd.to_numeric(history["oRESID"], errors="coerce").abs(),
            pd.to_numeric(history["dRESID"], errors="coerce").abs(),
            pd.to_numeric(history["RESID"], errors="coerce").abs(),
        ],
        ignore_index=True,
    )
    report = {
        "rows": int(len(history)),
        "seasons": sorted(history["season"].astype(int).unique().tolist()),
        "duplicate_player_season_keys": duplicate_keys,
        "missing_core_values": missing_core,
        "max_net_identity_gap": float(net_identity.max()),
        "max_display_residual": float(display_gap.max()),
        "season_rows": {
            str(int(year)): int(count)
            for year, count in history.groupby("season").size().items()
        },
        "qualified_rows": {
            str(int(year)): int(group["qualified"].sum())
            for year, group in history.groupby("season")
        },
    }
    if duplicate_keys or missing_core:
        raise ValueError(f"Historical WNBA metric validation failed: {report}")
    if report["max_net_identity_gap"] > 0.002:
        raise ValueError(f"Net identity failed: {report}")
    if report["max_display_residual"] > 0.002:
        raise ValueError(f"Display identity failed: {report}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--history-dir",
        type=Path,
        help="Defaults to OUTPUT_DIR/history.",
    )
    parser.add_argument("--rebuild-targets", action="store_true")
    parser.add_argument(
        "--reuse-season-outputs",
        action="store_true",
        help="Rebuild only the consolidated history from existing season files.",
    )
    return parser.parse_args()


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _existing_season_path(year: int, spec: dict[str, Any]) -> Path:
    if year == HISTORY_START:
        suffix = (
            f"all_a{int(spec['box_target_off_alpha'])}_"
            f"{int(spec['box_target_def_alpha'])}_history_warmup_pure_v1"
        )
    else:
        suffix = (
            f"{_season_type(year).lower()}_a{int(spec['off_alpha'])}_"
            f"{int(spec['def_alpha'])}_prior{float(spec['prior_scale']):g}_"
            "box_trueprior_history_v1"
        )
    directory = RESULTS_DIR / f"clean_decomp_{year % 100:02d}_{suffix}"
    return directory / (
        f"weighted_factors_alt3_efg_value_{year % 100:02d}_{suffix}.csv"
    )


def main() -> int:
    args = parse_args()
    history_dir = args.history_dir or args.output_dir / "history"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    history_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_selected_spec(args.output_dir)
    selected_report = json.loads((args.output_dir / "report.json").read_text())
    build_annual_targets(
        args.output_dir,
        args.rebuild_targets,
        float(spec["box_target_off_alpha"]),
        float(spec["box_target_def_alpha"]),
    )

    box_by_year = {
        year: load_box_features(year)
        for year in range(HISTORY_START, HISTORY_END + 1)
    }
    priors_by_year: dict[int, dict[str, pd.DataFrame]] = {}
    readable_by_year: dict[int, pd.DataFrame] = {}

    for year in range(HISTORY_START + 1, HISTORY_END + 1):
        season_type = _season_type(year)
        clean = _load_clean_total_df([year % 100], season_type)
        problem = _prepare_direct_ridge_problem(clean)
        priors, readable, _ = build_box_priors_for_year(
            args.output_dir,
            box_by_year,
            year,
            list(range(HISTORY_START, year)),
            list(problem["all_players"]),
        )
        priors_by_year[year] = priors
        readable_by_year[year] = readable
        readable.to_csv(args.output_dir / f"box_priors_{year}.csv", index=False)

    season_paths: dict[int, Path] = {}
    season_frames: list[pd.DataFrame] = []

    if args.reuse_season_outputs:
        warmup_path = _existing_season_path(2022, spec)
        if not warmup_path.exists():
            raise FileNotFoundError(warmup_path)
    else:
        warmup_path = build_clean_weighted_factors(
            22,
            22,
            "ALL",
            off_alpha=float(spec["box_target_off_alpha"]),
            def_alpha=float(spec["box_target_def_alpha"]),
            output_label="history_warmup_pure_v1",
        )
    season_paths[2022] = warmup_path
    season_frames.append(
        _add_reader_columns(
            pd.read_csv(warmup_path),
            2022,
            "ALL",
            "pure_warmup_no_prior",
            None,
        )
    )

    for year in range(2023, HISTORY_END + 1):
        season_type = _season_type(year)
        if args.reuse_season_outputs:
            path = _existing_season_path(year, spec)
            if not path.exists():
                raise FileNotFoundError(path)
        else:
            path = build_clean_weighted_factors(
                year % 100,
                year % 100,
                season_type,
                off_alpha=float(spec["off_alpha"]),
                def_alpha=float(spec["def_alpha"]),
                prior_results_by_prefix=priors_by_year[year],
                prior_scale=float(spec["prior_scale"]),
                output_label="box_trueprior_history_v1",
                direct_tov_def_alpha=float(spec["direct_tov_def_alpha"]),
            )
        season_paths[year] = path
        season_frames.append(
            _add_reader_columns(
                pd.read_csv(path),
                year,
                season_type,
                "single_season_box_trueprior_v1",
                readable_by_year[year],
            )
        )

    history = pd.concat(season_frames, ignore_index=True, sort=False)
    history = history.sort_values(["season", "rank", "possessions"])
    history_path = history_dir / "wnba_single_season_metrics_2022_2026.csv"
    history.to_csv(history_path, index=False)
    leaderboard_path = (
        history_dir / "wnba_single_season_leaderboard_2022_2026.csv"
    )
    history[LEADERBOARD_COLUMNS].to_csv(leaderboard_path, index=False)
    validation = _validate_history(history)

    manifest = {
        "title": "WNBA Single-Season Impact History, 2022-2026",
        "metric": "single-season factor-specific box-prior true-prior RAPM",
        "units": "points per 100 possessions",
        "defense_sign": "positive is good",
        "selected_spec": spec,
        "qualifying_possessions": QUALIFYING_POSSESSIONS,
        "season_policy": {
            "2022": "ALL, pure clean RAPM warm-up; no earlier WNBA season exists for a leakage-safe box prior",
            "2023-2025": "ALL, single-season true-prior likelihood with box models trained only on earlier seasons",
            "2026": "RS through the current completed data cutoff, same true-prior architecture",
        },
        "proprietary_metrics_in_training": False,
        "forward_validation": {
            "selected_pre_2026": selected_report.get("selected_pre_2026"),
            "held_out_2026": selected_report.get("held_out_2026"),
            "held_out_interpretation": (
                "The frozen 2025 rating was applied to observed 2026 lineups; "
                "the live 2026 player-season board was not used."
            ),
        },
        "season_outputs": {
            str(year): _manifest_path(path) for year, path in season_paths.items()
        },
        "consolidated_output": _manifest_path(history_path),
        "compact_leaderboard_output": _manifest_path(leaderboard_path),
        "validation": validation,
    }
    manifest_path = history_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float))
    logging.info("Wrote %s (%d player-seasons)", history_path, len(history))
    logging.info("Wrote %s", leaderboard_path)
    logging.info("Wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

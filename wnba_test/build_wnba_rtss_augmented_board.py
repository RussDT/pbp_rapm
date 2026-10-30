#!/usr/bin/env python3
"""Development-only RTSS-augmented WNBA board, side by side with box-only.

Answers "what do the ratings look like with RTSS in the prior?" in rating
units instead of component RMSE. It solves the same nine additive components
through the same prior-centered ridge the live board uses, twice: once around
the frozen box-only prior and once around the same prior plus the RTSS
corrections, then reports per-player rating deltas.

This is NOT a promotion path and NOT the confirmation. The one-shot promotion
decision lives in ``run_wnba_rtss_confirmation.py`` and is scored on the
untouched 2026 season. This script may never read season-2026 tracking
(``MAX_RTSS_FEATURE_YEAR``), so no run of it consumes the holdout mirror.

Prior contract mirrors the live nowcast: corrections are fit on cross-fitted
box residuals from seasons STRICTLY BEFORE the feature year and applied to
that year's RTSS features. For the default 2025 board that means corrections
fit on 2023-2024 and applied to 2025.

Carry-forward mode (``--rtss-feature-year`` strictly earlier than ``--year``)
carries those corrections onto a later season's box prior by player id. It is
the only way to see a 2026 board, since 2026 tracking is off limits, and it is
also the construction a live deployment needs before the current season has
accumulated tracking. Players with no row in the feature year keep their
box-only prior, so coverage is reported rather than assumed.

Candidate lanes are read at runtime from the v2 cross-validation selections
(``cross_validate_wnba_rtss_priors.py`` after the August 10, 2026 defensive
feature expansion) rather than hard-coded, and are filtered two ways:

1. The corrected quantity must map onto one of the nine additive solve
   components. ``ALT_TOV_VALUE`` and ``ALT_BADPASS_TOV_VALUE`` win the CV on
   defense but are not additive children of this tree, and inventing a parent
   for them would be a modeling claim this script has no standing to make, so
   they are dropped and recorded in the manifest as excluded.
2. At most one correction per (parent component, side). ``ALT_FT_FREQ`` is an
   exact additive child of ``ALT_FT``; when both win on the same side, the
   parent is applied and the child dropped, because applying both would add
   the same defensive free-throw signal to one prior twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import (
    ADDITIVE_PREFIXES,
    TARGET_PREFIXES,
    _derive_additive_priors,
    load_box_features,
)
from wnba_test.clean_decomp_wnba import RESULTS_DIR, build_clean_weighted_factors
from wnba_test.run_wnba_rtss_confirmation import (
    BOOTSTRAP_SEED,
    BOX_DIR,
    FIRST_BOX_SEASON,
    ComponentSumEvaluator,
    build_candidate_corrections,
    load_frozen_alphas,
    load_standardized_rtss,
    paired_game_bootstrap,
    rating_year_intercept,
)
from wnba_test.tune_wnba_component_alphas import _prior_frame

CV_BEST_PATH = RESULTS_DIR / "rtss_prior_cross_validation_v2" / "best_by_component.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "rtss_augmented_board"

# Hard fence on RTSS CONSUMPTION. 2026 is the untouched confirmation season
# and its tracking mirror is reserved for the one-shot harness, so no run of
# this script may read season-2026 RTSS features, with or without a flag.
#
# The BOARD year may exceed this in carry-forward mode
# (``--rtss-feature-year`` strictly earlier than ``--year``), where a player's
# prior correction comes from their previous season's tracking. That is the
# only construction available for a season whose own tracking is off limits,
# and it is also what a live deployment must do before the current season has
# accumulated tracking data.
MAX_RTSS_FEATURE_YEAR = 2025
DEFAULT_BOARD_YEAR = 2025

# Additive-tree mapping for corrected quantities that are not themselves
# solve components. ALT_FT_FREQ is an exact additive child of ALT_FT.
CANDIDATE_PARENT = {"ALT_FT_FREQ": "ALT_FT"}
QUALIFYING_POSSESSIONS = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=DEFAULT_BOARD_YEAR)
    parser.add_argument(
        "--rtss-feature-year",
        type=int,
        default=None,
        help=(
            "season whose RTSS features the corrections are applied to "
            "(default: the board year). Set it strictly earlier than --year "
            "for carry-forward mode, which reads no tracking from the board "
            "season at all."
        ),
    )
    parser.add_argument("--season-type", default="ALL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--qualifying-possessions", type=int, default=QUALIFYING_POSSESSIONS
    )
    parser.add_argument(
        "--forward-check",
        action="store_true",
        help=(
            "also score the development forward fold (board year - 1) -> "
            "(board year) so the rating deltas can be read against predictive "
            "evidence rather than on their own"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def forward_check(
    board_year: int,
    candidates: tuple[tuple[str, str, str], ...],
    box_by_year: dict[int, pd.DataFrame],
    rtss: pd.DataFrame,
    alphas: dict[str, Any],
    bootstrap_iterations: int,
) -> dict[str, Any]:
    """Development forward fold: rate on ``board_year - 1``, score on
    ``board_year`` actual possessions.

    Same construction as the confirmation harness's primary endpoint, one year
    earlier and on the same candidate set the board applies. This is
    development evidence about the candidate set, not the promotion decision.
    """
    rating_year = board_year - 1
    corrections, metadata = build_candidate_corrections(
        rating_year, candidates, box_by_year, rtss
    )
    fitted = {
        prefix: _prior_frame(BOX_DIR, rating_year, prefix)
        for prefix in TARGET_PREFIXES
    }
    base = _derive_additive_priors(fitted)
    augmented = build_arm_priors(fitted, corrections, candidates)

    evaluator = ComponentSumEvaluator(
        rating_year,
        board_year,
        rating_year_intercept(rating_year),
        asof_bound=board_year,
    )
    y = evaluator.validation_problem["y"]
    predictions = {
        name: evaluator.arm_predictions(
            priors, alphas["off_alpha"], alphas["def_alpha"], alphas["prior_scale"]
        )
        for name, priors in (("box_only", base), ("rtss_augmented", augmented))
    }
    mse = {
        name: float(((y - pred) ** 2).mean())
        for name, pred in predictions.items()
    }
    bootstrap = paired_game_bootstrap(
        y,
        predictions["rtss_augmented"],
        predictions["box_only"],
        evaluator.game_vector,
        bootstrap_iterations,
        BOOTSTRAP_SEED,
    )
    return {
        "label": (
            "DEVELOPMENT EVIDENCE ONLY: forward fold "
            f"{rating_year}->{board_year}; not the 2026 confirmation"
        ),
        "rating_year": rating_year,
        "validation_year": board_year,
        "correction_fit_years": sorted(
            {year for entry in metadata for year in entry["rtss_fit_years"]}
        ),
        "possessions": int(len(y)),
        "joint_mse": mse,
        "rtss_beats_box": bool(mse["rtss_augmented"] < mse["box_only"]),
        "paired_game_clustered_bootstrap": bootstrap,
    }


def select_candidates() -> tuple[
    tuple[tuple[str, str, str], ...], list[dict[str, Any]]
]:
    """Read the v2 CV selections and keep only additively-mappable winners."""
    if not CV_BEST_PATH.exists():
        raise FileNotFoundError(
            f"{CV_BEST_PATH} missing; run cross_validate_wnba_rtss_priors.py first"
        )
    best = pd.read_csv(CV_BEST_PATH)
    winners = best[best["selected_prior_lane"] == best["lane"]]

    kept: list[tuple[str, str, str]] = []
    excluded: list[dict[str, Any]] = []
    claimed: dict[tuple[str, str], str] = {}

    # Parents first so a parent always wins a (component, side) collision.
    ordered = sorted(
        winners.itertuples(),
        key=lambda row: (row.prefix in CANDIDATE_PARENT, row.prefix, row.side),
    )
    for row in ordered:
        prefix, side, lane = str(row.prefix), str(row.side), str(row.lane)
        parent = CANDIDATE_PARENT.get(prefix, prefix)
        if parent not in ADDITIVE_PREFIXES:
            excluded.append(
                {
                    "prefix": prefix,
                    "side": side,
                    "lane": lane,
                    "reason": (
                        "not an additive solve component and no defensible "
                        "parent mapping exists in this tree"
                    ),
                }
            )
            continue
        key = (parent, side)
        if key in claimed:
            excluded.append(
                {
                    "prefix": prefix,
                    "side": side,
                    "lane": lane,
                    "reason": (
                        f"{claimed[key]} already corrects {parent}/{side}; "
                        "applying both would double-count one signal"
                    ),
                }
            )
            continue
        claimed[key] = prefix
        kept.append((prefix, side, lane))
    return tuple(kept), excluded


def load_base_priors(year: int) -> dict[str, pd.DataFrame]:
    """All twelve fitted prior frames for ``year`` in solver sign convention.

    The board solve needs more than the nine additive components: the display
    tree also solves ``ALT_FT_FREQ``/``ALT_FT_SEVERITY``, the turnover-value
    trio, and the total ``RAPM``. Those are exact derived identities of the
    fitted twelve, which is why the live board closes with ``RESID == 0``.
    """
    return {prefix: _prior_frame(BOX_DIR, year, prefix) for prefix in TARGET_PREFIXES}


def build_arm_priors(
    base: dict[str, pd.DataFrame],
    corrections: dict[tuple[str, str], pd.Series],
    candidates: tuple[tuple[str, str, str], ...],
) -> dict[str, pd.DataFrame]:
    """Apply corrections to the fitted priors, then re-derive the identities.

    Corrections land on fitted prefixes only; ``_derive_additive_priors`` then
    rebuilds ``ALT_FT_SEVERITY``, ``ALT_SCORING_TOV_VALUE``, and the additive
    ``RAPM`` total from the corrected parents, so the augmented board keeps the
    same exact-closure contract as the box-only board.
    """
    priors = {prefix: frame.copy() for prefix, frame in base.items()}
    for prefix, side, _lane in candidates:
        parent = CANDIDATE_PARENT.get(prefix, prefix)
        frame = priors[parent]
        mapped = frame["player_id"].astype(str).map(corrections[(prefix, side)])
        frame[side] = frame[side] + mapped.fillna(0.0)
    return _derive_additive_priors(priors)


def main() -> int:
    args = parse_args()
    feature_year = (
        args.year if args.rtss_feature_year is None else args.rtss_feature_year
    )
    if feature_year > MAX_RTSS_FEATURE_YEAR:
        raise SystemExit(
            f"confirmation-season fence: RTSS feature year {feature_year} "
            f"exceeds {MAX_RTSS_FEATURE_YEAR}. The 2026 tracking mirror is "
            "reserved for run_wnba_rtss_confirmation.py. For a 2026 board use "
            "carry-forward mode: --year 2026 --rtss-feature-year 2025."
        )
    if feature_year > args.year:
        raise SystemExit(
            f"--rtss-feature-year {feature_year} is later than the board year "
            f"{args.year}; a prior may not be built from the future."
        )
    carry_forward = feature_year < args.year
    if carry_forward and args.forward_check:
        raise SystemExit(
            "--forward-check is unavailable in carry-forward mode: scoring the "
            f"{args.year} fold would compute the confirmation harness's "
            "scoring season outside the one-shot harness."
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = load_frozen_alphas()
    candidates, excluded = select_candidates()
    print(f"RTSS candidates applied ({len(candidates)}):", flush=True)
    for prefix, side, lane in candidates:
        print(f"  {prefix}/{side} <- {lane}", flush=True)
    for row in excluded:
        print(
            f"  EXCLUDED {row['prefix']}/{row['side']}: {row['reason']}",
            flush=True,
        )

    print("Loading box features and standardized RTSS features...", flush=True)
    box_by_year = {
        year: load_box_features(year)
        for year in range(FIRST_BOX_SEASON, feature_year)
    }
    rtss = load_standardized_rtss()
    if int(rtss["season"].max()) > MAX_RTSS_FEATURE_YEAR:
        raise SystemExit(
            "RTSS feature file contains seasons past "
            f"{MAX_RTSS_FEATURE_YEAR}; refusing to run against a feature "
            "surface that has been rebuilt with --include-confirmation-season"
        )

    print(
        f"Fitting corrections on seasons < {feature_year}, applying to "
        f"{feature_year} features...",
        flush=True,
    )
    corrections, correction_metadata = build_candidate_corrections(
        feature_year, candidates, box_by_year, rtss
    )

    fitted_priors = load_base_priors(args.year)
    base_priors = _derive_additive_priors(fitted_priors)
    augmented_priors = build_arm_priors(fitted_priors, corrections, candidates)

    label_suffix = f"_rtss{feature_year}" if carry_forward else ""
    print(f"Solving box-only board for {args.year}...", flush=True)
    box_path = build_clean_weighted_factors(
        args.year % 100,
        args.year % 100,
        args.season_type,
        off_alpha=alphas["off_alpha"],
        def_alpha=alphas["def_alpha"],
        prior_results_by_prefix=base_priors,
        prior_scale=alphas["prior_scale"],
        output_label=f"rtss_board_box_only_{args.year}{label_suffix}",
    )
    print(f"Solving RTSS-augmented board for {args.year}...", flush=True)
    rtss_path = build_clean_weighted_factors(
        args.year % 100,
        args.year % 100,
        args.season_type,
        off_alpha=alphas["off_alpha"],
        def_alpha=alphas["def_alpha"],
        prior_results_by_prefix=augmented_priors,
        prior_scale=alphas["prior_scale"],
        output_label=f"rtss_board_augmented_{args.year}{label_suffix}",
    )

    keep = [
        "player_id",
        "player_name",
        "off",
        "def",
        "net_rapm",
        "possessions",
    ]
    box = pd.read_csv(box_path)[keep]
    aug = pd.read_csv(rtss_path)[keep]
    comparison = box.merge(
        aug.drop(columns=["player_name", "possessions"]),
        on="player_id",
        suffixes=("_box", "_rtss"),
    )
    for column in ("off", "def", "net_rapm"):
        comparison[f"delta_{column}"] = (
            comparison[f"{column}_rtss"] - comparison[f"{column}_box"]
        )
    comparison = comparison.sort_values("net_rapm_rtss", ascending=False)
    comparison_path = (
        output_dir / f"board_comparison_{args.year}{label_suffix}.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    qualified = comparison[
        comparison["possessions"] >= args.qualifying_possessions
    ]
    summary = {
        "players": int(len(comparison)),
        "qualified_players": int(len(qualified)),
        "qualifying_possessions": int(args.qualifying_possessions),
        "mean_abs_delta_net": float(qualified["delta_net_rapm"].abs().mean()),
        "max_abs_delta_net": float(qualified["delta_net_rapm"].abs().max()),
        "mean_abs_delta_off": float(qualified["delta_off"].abs().mean()),
        "mean_abs_delta_def": float(qualified["delta_def"].abs().mean()),
        "net_correlation": float(
            qualified["net_rapm_box"].corr(qualified["net_rapm_rtss"])
        ),
        "def_correlation": float(
            qualified["def_box"].corr(qualified["def_rtss"])
        ),
        "qualified_players_with_correction": int(
            (qualified["delta_net_rapm"].abs() > 1e-9).sum()
        ),
    }

    forward: dict[str, Any] | None = None
    if args.forward_check:
        print(
            f"Scoring development forward fold {args.year - 1}->{args.year}...",
            flush=True,
        )
        forward = forward_check(
            args.year,
            candidates,
            box_by_year,
            rtss,
            alphas,
            args.bootstrap_iterations,
        )

    manifest = {
        "title": f"RTSS-augmented vs box-only WNBA board, {args.year}",
        "status": "DEVELOPMENT ONLY — not a promotion, not the confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "board_year": int(args.year),
        "rtss_feature_year": int(feature_year),
        "carry_forward": bool(carry_forward),
        "season_type": args.season_type,
        "confirmation_rtss_mirror_read": False,
        "max_rtss_feature_year": MAX_RTSS_FEATURE_YEAR,
        "prior_contract": (
            f"corrections fit on cross-fitted box residuals from seasons "
            f"strictly before {feature_year} and applied to {feature_year} "
            "RTSS features"
            + (
                f", then carried onto the {args.year} box prior by player id; "
                f"no season-{args.year} tracking is read and players without a "
                f"{feature_year} tracking row keep their box-only prior"
                if carry_forward
                else ", mirroring the live nowcast prior-application contract"
            )
        ),
        "frozen_solve_parameters": alphas,
        "candidate_source": str(CV_BEST_PATH),
        "candidates_applied": [
            {"prefix": prefix, "side": side, "lane": lane}
            for prefix, side, lane in candidates
        ],
        "candidates_excluded": excluded,
        "correction_models": correction_metadata,
        "box_only_board": str(box_path),
        "rtss_augmented_board": str(rtss_path),
        "comparison_output": str(comparison_path),
        "summary": summary,
        "forward_check": forward,
        "interpretation": (
            "Rating deltas show how much the RTSS corrections move players, "
            "not whether they improve prediction. Predictive evidence comes "
            "only from the forward joint component-sum test."
        ),
    }
    (output_dir / f"manifest_{args.year}{label_suffix}.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n"
    )

    print(json.dumps(summary, indent=2), flush=True)
    if forward is not None:
        print(json.dumps(forward, indent=2, default=float), flush=True)
    print(f"Wrote {comparison_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

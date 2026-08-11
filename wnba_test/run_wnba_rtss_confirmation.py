#!/usr/bin/env python3
"""Guarded one-shot 2026 RTSS component-prior confirmation harness.

Binding spec: ``nba_pipeline/docs/WNBA_RTSS_CONFIRMATION_PREREG.md`` (frozen
August 10, 2026; amended the same day by design review before any 2026
evaluation ran). This harness fires exactly once, on the untouched 2026
season, and only when every fire condition passes:

1. 2026 regular season complete: cached CDN final-game evidence, a documented
   completed-season heuristic (final-game count >= expected schedule AND the
   latest final game is at least ``SEASON_COMPLETE_QUIET_DAYS`` days old),
   PLUS the explicit operator assertion ``--season-complete``.
2. RTSS mirror completeness: every 2026 regular-season game in the mirror
   manifest under ``downloads/rtss-data/wnba/`` is ENDED with gamedata and a
   non-trivial 2D ``basketball-markings.json``, and the mirror covers at least
   as many games as the CDN boxscore cache reports final. The 2026 mirror is
   consumed only by the SECONDARY descriptive nowcast diagnostic; the primary
   forward endpoint never reads it.
3. As-of audit: a mechanical assertion hook (``assert_no_confirmation_rows``)
   runs inside the fit path on every fit-input frame and fails closed on any
   frame it cannot positively classify; no season-2026 rows may enter any
   box-backbone or RTSS-residual fit. 2026 rows are used only for scoring and
   for the secondary diagnostic's prior *application* features, mirroring the
   live box-prior nowcast contract.
4. The sentinel ``wnba_test/Results/rtss_confirmation/FIRED`` is absent.

Modes:
- ``--check``: print PASS/BLOCKED per fire condition; exit nonzero if blocked.
- ``--dev-smoke``: run the full pipeline shape on development data only,
  one year earlier than the real fold: primary forward fold 2024->2025
  (corrections trained on 2023 residuals only, applied to 2024 features),
  secondary nowcast on 2025. Writes to ``Results/rtss_confirmation_dev_smoke/``;
  never touches 2026 data or the sentinel. Development evidence only.
- real mode (no flag): refuses unless all fire conditions pass; on completion
  writes the verdict JSON, bootstrap CIs, and the FIRED sentinel exactly once.

Frozen procedure (prereg steps 1-8, as amended August 10, 2026):

PRIMARY endpoint — strictly forward fold 2025 -> 2026, the exact
``tune_wnba_component_alphas`` construction: solve the nine additive
components on rating season 2025 (season type ALL) around (a) the box-only
``box_priors_2025.csv`` backbone (trained on seasons <= 2024) and (b) the
RTSS-augmented priors whose corrections are fit on 2023-2024 cross-fitted box
residuals ONLY (strictly before rating year 2025) and applied to 2025 RTSS
features. Both arms' summed player coefficient maps then predict actual 2026
possession points; the promotion decision is the paired game-clustered
bootstrap of the joint component-sum MSE difference on 2026 games. The frozen
live alphas (offense 5000, defense from
``Results/alpha_grid_extension/decision.json`` when present else 40000) and
prior scale 1.5 are used symmetrically in BOTH arms. Promotion: an arm passes
iff it beats box-only joint MSE on the primary endpoint; both pass -> lower
MSE; neither -> null promotion.

SECONDARY (descriptive only, never a promotion input): the in-sample 2026
nowcast comparison — corrections trained on 2023-2025 residuals applied to
2026 RTSS features, nine components solved on 2026 around each arm and scored
on the same 2026 possessions. It mirrors the live nowcast prior-application
contract and is the only consumer of the 2026 RTSS mirror. Per-component
RMSE/correlation is likewise secondary and descriptive only.

Frozen candidate lanes are the recorded ``best_by_component.csv`` selections;
the harness asserts the hard-coded lane spec equals that file at runtime and
fails loudly on drift. ``ALT_FT_FREQ`` is an exact additive child of
``ALT_FT``, so its correction is applied to the ``ALT_FT`` prior; every other
candidate maps to itself.

Standing fences: market-blind, no proprietary ratings, 2026 never informs
selection/tuning/fitting, no credentials in artifacts, generated outputs stay
out of git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WNBA_TEST_DIR = Path(__file__).resolve().parent
if str(WNBA_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(WNBA_TEST_DIR))

import wnba_gameday_common as common

from wnba_test.build_wnba_box_priors import (
    ADDITIVE_PREFIXES,
    DEFAULT_OUTPUT as BOX_DIR,
    RIDGE_ALPHAS as BOX_ALPHAS,
    feature_columns,
    load_box_features,
    load_target,
    target_path,
)
from wnba_test.clean_decomp_wnba import (
    RESULTS_DIR,
    _attach_date_proxy,
    _files_for_metric,
    _infer_year_from_path,
    _load_clean_total_df,
    _load_name_map,
    _prepare_direct_ridge_problem,
    _prepare_metric_df,
    _solve_direct_ridge_problem,
    run_metric,
)
from wnba_test.cross_validate_wnba_rtss_priors import (
    RESIDUAL_ALPHAS,
    _fit,
    _lane_features,
    _rtss_rows,
    _season_standardize_rtss,
    _select_alpha_by_year,
    _target_rows,
    _weighted_metrics,
)
from wnba_test.tune_wnba_component_alphas import _prior_frame, _result_vector


CONFIRMATION_YEAR = 2026
DEV_SMOKE_YEAR = 2025
FIRST_BOX_SEASON = 2022
FIRST_RTSS_SEASON = 2023

OUTPUT_DIR = RESULTS_DIR / "rtss_confirmation"
DEV_OUTPUT_DIR = RESULTS_DIR / "rtss_confirmation_dev_smoke"
SENTINEL_PATH = OUTPUT_DIR / "FIRED"
ALPHA_DECISION_PATH = RESULTS_DIR / "alpha_grid_extension" / "decision.json"
RTSS_FEATURES_PATH = RESULTS_DIR / "rtss_prior_research" / "player_season_features.csv"
CV_BEST_PATH = RESULTS_DIR / "rtss_prior_cross_validation" / "best_by_component.csv"
MIRROR_ROOT = ROOT / "downloads" / "rtss-data" / "wnba"
MIRROR_SCHEDULES = MIRROR_ROOT / "api" / "portal" / "schedules.json"

FROZEN_OFF_ALPHA = 5000.0
FALLBACK_DEF_ALPHA = 40000.0
FROZEN_PRIOR_SCALE = 1.5
TARGET_OFF_ALPHA = 750.0
TARGET_DEF_ALPHA = 1000.0

# Completed-season heuristic: 2026 has 15 teams x 44 games -> 330 RS games.
FALLBACK_EXPECTED_RS_GAMES = 330
SEASON_COMPLETE_QUIET_DAYS = 3
MIN_MARKINGS_BYTES = 1024

MIN_RTSS_CHANCES = 100.0
MIN_TARGET_POSS = 250.0
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260810

# Frozen candidate bundle: (prefix, side, lane). Lanes are the recorded CV
# selections in best_by_component.csv; assert_frozen_lanes_match_recorded_cv()
# fails loudly if this spec ever drifts from the recorded file.
TIER_A: tuple[tuple[str, str, str], ...] = (
    ("ALT_EFG_THREE_FREQ", "def", "tracking_core"),
    ("SECOND_CHANCE_CLEAN", "off", "tracking_core"),
    ("ALT_FT_FREQ", "def", "tracking_core"),
    ("ALT_EFG_MID_FREQ", "def", "tracking_core_plus_q"),
    ("ALT_EFG_MID_FG", "off", "tracking_core"),
)
TIER_B: tuple[tuple[str, str, str], ...] = (
    ("ALT_FT", "def", "tracking_core"),
    ("ALT_EFG_RIM_FREQ", "off", "tracking_core"),
    ("ALT_EFG_MID_FREQ", "off", "creation_roles_only"),
    ("ALT_EFG_THREE_FREQ", "off", "tracking_core_plus_q"),
)
ARMS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "tier_a": TIER_A,
    "tier_a_plus_b": TIER_A + TIER_B,
}
# ALT_FT_FREQ is an exact additive child of ALT_FT; every other candidate
# corrects its own additive solve component.
CANDIDATE_PARENT = {"ALT_FT_FREQ": "ALT_FT"}

# Genuine calendar-season keys written by this lane's own loaders.
SEASON_COLUMN_CANDIDATES = ("year", "season")
WNBA_GAME_ID_PATTERN = r"10[24]\d{7}"


# ---------------------------------------------------------------------------
# As-of audit hook (fire condition 3)
# ---------------------------------------------------------------------------


def assert_no_confirmation_rows(
    frame: pd.DataFrame,
    confirmation_year: int,
    context: str,
    allow_unclassified: bool = False,
) -> pd.DataFrame:
    """Runtime assertion: no confirmation-season rows in any fit input frame.

    WNBA game ids (``102YY``/``104YY``) are the canonical season evidence.
    The embedded ``Season`` column in processed possession CSVs is NOT
    canonical (it uses the NBA-style ending-year convention, e.g. the 2024
    files carry ``Season=2025``), so it is consulted only when no game-id
    evidence exists in the frame.

    Fail-closed contract: every non-empty frame must be positively classified
    by at least one recognized season-evidence source, and every ``game_id``
    value must be WNBA-shaped. Anything the audit cannot positively classify
    raises unless the call site explicitly whitelists the frame with
    ``allow_unclassified=True`` (reserved for frames known to carry no
    season-keyed rows; no current call site needs it).
    """
    if frame.empty:
        return frame
    classified = False
    if "game_id" in frame.columns:
        ids = frame["game_id"].astype(str)
        wnba_mask = ids.str.fullmatch(WNBA_GAME_ID_PATTERN)
        unrecognized = int((~wnba_mask).sum())
        if unrecognized and not allow_unclassified:
            samples = ids[~wnba_mask].unique()[:5].tolist()
            raise RuntimeError(
                f"as-of audit cannot classify {context}: {unrecognized} game_id "
                f"rows are not WNBA-shaped (samples: {samples}); refusing to "
                "fail open"
            )
        if wnba_mask.any():
            classified = True
            seasons = ids[wnba_mask].map(common.game_id_season)
            bad = int((seasons >= confirmation_year).sum())
            if bad:
                raise RuntimeError(
                    f"as-of audit violation in {context}: {bad} rows with "
                    f"game_id season >= {confirmation_year}"
                )
    for column in SEASON_COLUMN_CANDIDATES:
        if column in frame.columns:
            classified = True
            seasons = pd.to_numeric(frame[column], errors="coerce").dropna()
            bad = int((seasons >= confirmation_year).sum())
            if bad:
                raise RuntimeError(
                    f"as-of audit violation in {context}: {bad} rows with "
                    f"{column} >= {confirmation_year}"
                )
    if not classified and "Season" in frame.columns:
        classified = True
        seasons = pd.to_numeric(frame["Season"], errors="coerce").dropna()
        # Ending-year convention: Season == confirmation_year labels the
        # season that STARTS in confirmation_year - 1, which is still a
        # legitimate pre-confirmation fit season.
        bad = int((seasons > confirmation_year).sum())
        if bad:
            raise RuntimeError(
                f"as-of audit violation in {context}: {bad} rows with "
                f"embedded Season > {confirmation_year} and no game-id evidence"
            )
    if not classified and not allow_unclassified:
        raise RuntimeError(
            f"as-of audit cannot positively classify season evidence in "
            f"{context} (no WNBA game ids, no {SEASON_COLUMN_CANDIDATES} "
            "column, no embedded Season); refusing to fail open"
        )
    return frame


def assert_frozen_lanes_match_recorded_cv() -> None:
    """Fail loudly if the hard-coded Tier A/B lane spec drifts from the
    recorded CV selections in best_by_component.csv."""
    if not CV_BEST_PATH.exists():
        raise RuntimeError(
            f"recorded CV selection file missing: {CV_BEST_PATH}; cannot "
            "verify the frozen candidate lanes"
        )
    best = pd.read_csv(CV_BEST_PATH).set_index(["prefix", "side"])
    problems: list[str] = []
    for prefix, side, lane in TIER_A + TIER_B:
        key = (prefix, side)
        if key not in best.index:
            problems.append(f"{prefix}/{side}: absent from recorded CV file")
            continue
        row = best.loc[key]
        recorded_lane = str(row["lane"])
        selected_lane = str(row["selected_prior_lane"])
        if lane != recorded_lane:
            problems.append(
                f"{prefix}/{side}: frozen lane {lane!r} != recorded CV lane "
                f"{recorded_lane!r}"
            )
        if selected_lane != lane:
            problems.append(
                f"{prefix}/{side}: recorded selected_prior_lane "
                f"{selected_lane!r} does not select the frozen lane {lane!r}"
            )
    if problems:
        raise RuntimeError(
            "frozen candidate lanes drifted from recorded CV selections "
            f"({CV_BEST_PATH}): " + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# Fire conditions
# ---------------------------------------------------------------------------


def _cached_final_rs_games(year: int) -> tuple[int, str | None]:
    prefix = f"{common.RS_PREFIX}{year % 100:02d}"
    count = 0
    latest: str | None = None
    for game_id in common.iter_cached_game_ids():
        if not game_id.startswith(prefix):
            continue
        payload = common.load_cached_boxscore(game_id)
        if payload is None or not common.boxscore_is_final(payload):
            continue
        count += 1
        stamp = (payload.get("game") or {}).get("gameTimeUTC")
        if stamp and (latest is None or stamp > latest):
            latest = stamp
    return count, latest


def _mirror_rs_group(year: int) -> list[dict[str, Any]] | None:
    if not MIRROR_SCHEDULES.exists():
        return None
    try:
        schedules = json.loads(MIRROR_SCHEDULES.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    wanted = f"{year} Regular Season"
    for entry in schedules:
        if wanted in str(entry.get("name", "")):
            group_path = (
                MIRROR_ROOT / "api" / "portal" / "games"
                / f"{entry['gameGroupId']}.json"
            )
            if group_path.exists():
                return json.loads(group_path.read_text())
    return None


def check_season_complete(season_complete_asserted: bool) -> tuple[bool, str]:
    final_count, latest_stamp = _cached_final_rs_games(CONFIRMATION_YEAR)
    group = _mirror_rs_group(CONFIRMATION_YEAR)
    expected = len(group) if group else FALLBACK_EXPECTED_RS_GAMES
    quiet_ok = False
    if latest_stamp is not None:
        latest_dt = datetime.fromisoformat(latest_stamp.replace("Z", "+00:00"))
        quiet_ok = datetime.now(timezone.utc) - latest_dt >= timedelta(
            days=SEASON_COMPLETE_QUIET_DAYS
        )
    heuristic_ok = final_count >= expected and quiet_ok
    passed = heuristic_ok and season_complete_asserted
    detail = (
        f"cached final {CONFIRMATION_YEAR} RS games={final_count}/{expected} "
        f"expected, latest final gameTimeUTC={latest_stamp}, "
        f"quiet>={SEASON_COMPLETE_QUIET_DAYS}d={quiet_ok}, "
        f"operator --season-complete={season_complete_asserted}"
    )
    return passed, detail


def check_mirror_complete() -> tuple[bool, str]:
    group = _mirror_rs_group(CONFIRMATION_YEAR)
    if group is None:
        return False, "mirror manifest/RS game group missing under downloads/rtss-data/wnba/"
    scheduled = [game for game in group if game.get("status") == "SCHEDULED"]
    ended = [game for game in group if game.get("status") != "SCHEDULED"]
    covered = 0
    missing_markings: list[str] = []
    for game in ended:
        game_id = str(game["id"])
        gamedata = MIRROR_ROOT / "api" / "portal" / "gamedata" / f"{game_id}.json"
        markings = MIRROR_ROOT / "downloads" / game_id / "basketball-markings.json"
        if (
            gamedata.exists()
            and markings.exists()
            and markings.stat().st_size >= MIN_MARKINGS_BYTES
        ):
            covered += 1
        else:
            missing_markings.append(game_id)
    cdn_final, _ = _cached_final_rs_games(CONFIRMATION_YEAR)
    features_2026 = False
    if RTSS_FEATURES_PATH.exists():
        seasons = pd.read_csv(RTSS_FEATURES_PATH, usecols=["season"])["season"]
        features_2026 = bool((seasons == CONFIRMATION_YEAR).any())
    passed = (
        not scheduled
        and not missing_markings
        and covered >= cdn_final
        and covered >= len(group)
    )
    detail = (
        f"mirror RS group games={len(group)}, ENDED={len(ended)}, "
        f"still SCHEDULED={len(scheduled)}, 2D-covered={covered}, "
        f"CDN cache final={cdn_final}, missing markings={len(missing_markings)}; "
        f"season-{CONFIRMATION_YEAR} rows in feature file={features_2026} "
        "(2026 mirror feeds only the secondary descriptive nowcast arm)"
    )
    return passed, detail


def check_asof_audit() -> tuple[bool, str]:
    primary_rating_year = CONFIRMATION_YEAR - 1
    box_fit_years = list(range(FIRST_BOX_SEASON, CONFIRMATION_YEAR))
    rtss_fit_years = [
        year for year in range(FIRST_RTSS_SEASON, CONFIRMATION_YEAR)
    ]
    primary_rtss_fit_years = [
        year for year in range(FIRST_RTSS_SEASON, primary_rating_year)
    ]
    problems: list[str] = []
    if max(box_fit_years) >= CONFIRMATION_YEAR or max(rtss_fit_years) >= CONFIRMATION_YEAR:
        problems.append("fit-year list reaches the confirmation season")
    if max(primary_rtss_fit_years) >= primary_rating_year:
        problems.append("primary correction fit years reach the rating season")
    for year in box_fit_years:
        for prefix in ("ALT_EFG_THREE_FREQ", "ALT_FT_FREQ", "SECOND_CHANCE_CLEAN"):
            if not target_path(BOX_DIR, year, prefix).exists():
                problems.append(f"missing annual target {year}/{prefix}")
    if RTSS_FEATURES_PATH.exists():
        seasons = pd.read_csv(RTSS_FEATURES_PATH, usecols=["season"])["season"]
        fit_frame = pd.DataFrame({"season": seasons[seasons.isin(rtss_fit_years)]})
        try:
            assert_no_confirmation_rows(
                fit_frame, CONFIRMATION_YEAR, "static feature-file fit-subset scan"
            )
        except RuntimeError as error:
            problems.append(str(error))
    else:
        problems.append(f"missing RTSS feature file {RTSS_FEATURES_PATH}")
    try:
        assert_frozen_lanes_match_recorded_cv()
    except RuntimeError as error:
        problems.append(str(error))
    passed = not problems
    detail = (
        f"primary forward fold {primary_rating_year}->{CONFIRMATION_YEAR} with "
        f"correction fit years {primary_rtss_fit_years}; secondary nowcast fit "
        f"years box={box_fit_years} rtss={rtss_fit_years}; runtime assertion "
        "hook assert_no_confirmation_rows() armed fail-closed on every fit "
        "input frame; frozen lanes verified against recorded CV selections"
        + ("" if passed else f"; problems={problems}")
    )
    return passed, detail


def check_sentinel_absent() -> tuple[bool, str]:
    if SENTINEL_PATH.exists():
        return False, f"sentinel exists: {SENTINEL_PATH} (harness already fired)"
    return True, f"sentinel absent: {SENTINEL_PATH}"


def run_fire_checks(season_complete_asserted: bool) -> list[dict[str, Any]]:
    checks = [
        ("season_complete", check_season_complete(season_complete_asserted)),
        ("rtss_mirror_complete", check_mirror_complete()),
        ("asof_audit", check_asof_audit()),
        ("sentinel_absent", check_sentinel_absent()),
    ]
    return [
        {"condition": name, "passed": passed, "detail": detail}
        for name, (passed, detail) in checks
    ]


# ---------------------------------------------------------------------------
# Frozen solve parameters
# ---------------------------------------------------------------------------


def load_frozen_alphas() -> dict[str, Any]:
    def_alpha = FALLBACK_DEF_ALPHA
    source = "prereg fallback 40000"
    if ALPHA_DECISION_PATH.exists():
        decision = json.loads(ALPHA_DECISION_PATH.read_text())
        def_alpha = float(decision["selected_def_alpha"])
        source = str(ALPHA_DECISION_PATH)
    return {
        "off_alpha": FROZEN_OFF_ALPHA,
        "def_alpha": def_alpha,
        "def_alpha_source": source,
        "prior_scale": FROZEN_PRIOR_SCALE,
    }


# ---------------------------------------------------------------------------
# RTSS corrections (prereg step 2)
# ---------------------------------------------------------------------------


def load_standardized_rtss() -> pd.DataFrame:
    rtss = pd.read_csv(RTSS_FEATURES_PATH, dtype={"player_id": str})
    rtss["player_id"] = (
        pd.to_numeric(rtss["player_id"], errors="coerce").astype("Int64").astype(str)
    )
    return _season_standardize_rtss(rtss)


def build_candidate_corrections(
    apply_year: int,
    candidates: tuple[tuple[str, str, str], ...],
    box_by_year: dict[int, pd.DataFrame],
    rtss: pd.DataFrame,
) -> tuple[dict[tuple[str, str], pd.Series], list[dict[str, Any]]]:
    """Fit cross-fitted box-residual RTSS models on seasons strictly before
    ``apply_year`` (asserted) and predict corrections for ``apply_year``.

    For the primary forward endpoint ``apply_year`` is the rating season
    (2025 real / 2024 dev-smoke), so no fit sees the rating season, let alone
    the scoring season. For the secondary nowcast diagnostic ``apply_year`` is
    the confirmation season itself; fits stay strictly earlier and only the
    prior-application features come from the confirmation season.
    """
    box_fit_years = list(range(FIRST_BOX_SEASON, apply_year))
    rtss_fit_years = [
        year
        for year in sorted(rtss["season"].unique())
        if FIRST_RTSS_SEASON <= int(year) < apply_year
    ]
    if not rtss_fit_years:
        raise RuntimeError(
            f"no RTSS seasons strictly before {apply_year} available for fitting"
        )
    apply_rows_all = rtss[rtss["season"] == apply_year]
    if apply_rows_all.empty:
        raise RuntimeError(
            f"no season-{apply_year} rows in {RTSS_FEATURES_PATH}; rebuild "
            "player_season_features with build_wnba_rtss_prior_features.py first"
        )

    corrections: dict[tuple[str, str], pd.Series] = {}
    metadata: list[dict[str, Any]] = []
    for prefix, side, lane in candidates:
        context = f"{prefix}/{side}/{lane}"
        box_features = feature_columns(prefix, side)
        box_rows: dict[int, pd.DataFrame] = {}
        for year in box_fit_years:
            rows = _target_rows(
                year, prefix, side, box_by_year, BOX_DIR, MIN_TARGET_POSS
            )
            assert_no_confirmation_rows(
                rows, apply_year, f"box fit rows {year} {context}"
            )
            box_rows[year] = rows
        box_alpha = _select_alpha_by_year(
            box_rows, box_fit_years, box_features, BOX_ALPHAS
        )

        residual_by_year: dict[int, pd.DataFrame] = {}
        for residual_year in rtss_fit_years:
            inner_years = [year for year in box_fit_years if year != residual_year]
            inner_model = _fit(
                pd.concat([box_rows[year] for year in inner_years], ignore_index=True),
                box_features,
                box_alpha,
            )
            rows = _rtss_rows(
                box_rows[residual_year],
                rtss[rtss["season"] == residual_year],
                side,
                MIN_RTSS_CHANCES,
            )
            assert_no_confirmation_rows(
                rows, apply_year, f"RTSS residual rows {residual_year} {context}"
            )
            rows = rows.copy()
            rows["target"] = rows["target"] - inner_model.predict(rows[box_features])
            residual_by_year[residual_year] = rows

        lane_map = _lane_features(prefix, side)
        if lane not in lane_map:
            raise KeyError(f"frozen lane {lane!r} unavailable for {prefix} {side}")
        tracking_features = lane_map[lane]
        residual_alpha = _select_alpha_by_year(
            residual_by_year, list(rtss_fit_years), tracking_features, RESIDUAL_ALPHAS
        )
        residual_model = _fit(
            pd.concat(list(residual_by_year.values()), ignore_index=True),
            tracking_features,
            residual_alpha,
        )
        apply_rows = apply_rows_all[
            apply_rows_all[f"{side}_chances"] >= MIN_RTSS_CHANCES
        ]
        correction = pd.Series(
            residual_model.predict(apply_rows[tracking_features]),
            index=apply_rows["player_id"].astype(str),
        )
        corrections[(prefix, side)] = correction
        metadata.append(
            {
                "prefix": prefix,
                "side": side,
                "lane": lane,
                "apply_year": int(apply_year),
                "box_alpha": box_alpha,
                "residual_alpha": residual_alpha,
                "box_fit_years": box_fit_years,
                "rtss_fit_years": [int(year) for year in rtss_fit_years],
                "tracking_features": [
                    feature.replace("z_", "", 1) for feature in tracking_features
                ],
                "corrected_players": int(len(correction)),
                "correction_mean": float(correction.mean()),
                "correction_std": float(correction.std(ddof=0)),
            }
        )
    return corrections, metadata


def build_arm_priors(
    base_priors: dict[str, pd.DataFrame],
    corrections: dict[tuple[str, str], pd.Series],
    candidates: tuple[tuple[str, str, str], ...],
) -> dict[str, pd.DataFrame]:
    priors = {prefix: frame.copy() for prefix, frame in base_priors.items()}
    for prefix, side, _lane in candidates:
        parent = CANDIDATE_PARENT.get(prefix, prefix)
        frame = priors[parent]
        mapped = frame["player_id"].astype(str).map(corrections[(prefix, side)])
        frame[side] = frame[side] + mapped.fillna(0.0)
    return priors


# ---------------------------------------------------------------------------
# Joint component-sum evaluation (prereg steps 3-4)
# ---------------------------------------------------------------------------


def _season_type(year: int) -> str:
    return "RS" if year == CONFIRMATION_YEAR else "ALL"


def _component_problem(
    prefix: str,
    year: int,
    season_type: str,
    asof_bound: int | None = None,
) -> dict[str, Any]:
    files = [
        path
        for path in _files_for_metric(prefix, [year % 100], season_type)
        if path.exists()
    ]
    if not files:
        raise FileNotFoundError(f"No {prefix} files for {year} {season_type}")
    frames = [
        _attach_date_proxy(pd.read_csv(path), _infer_year_from_path(path))
        for path in files
    ]
    combined = pd.concat(frames, ignore_index=True)
    if asof_bound is not None:
        assert_no_confirmation_rows(
            combined, asof_bound, f"{prefix} rating-season solve rows {year}"
        )
    prepared = _prepare_metric_df(combined, prefix)
    return _prepare_direct_ridge_problem(prepared)


def possession_game_ids(df: pd.DataFrame, problem: dict[str, Any]) -> np.ndarray:
    """Recover per-possession game ids in the transformed row order.

    ``transform_to_home_away_format`` iterates games in first-appearance order
    and keeps each game's rows in source order. The reordering is verified
    mechanically by matching the reordered ``Off_Diff`` against the problem's
    ``y`` vector before any bootstrap uses the clusters.
    """
    ids = df["game_id"].astype(str)
    codes = pd.Categorical(ids, categories=pd.unique(ids)).codes
    order = np.argsort(codes, kind="stable")
    game_vector = ids.to_numpy()[order]
    y_expected = (
        pd.to_numeric(df["Off_Diff"], errors="coerce").fillna(0.0).to_numpy()[order]
    )
    y_actual = np.asarray(problem["y"], dtype=float)
    if len(y_expected) != len(y_actual) or not np.allclose(y_expected, y_actual):
        raise RuntimeError(
            "game-cluster alignment failed: reordered source rows do not match "
            "the transformed possession target vector"
        )
    return game_vector


def rating_year_intercept(rating_year: int) -> float:
    """Tuner-construction intercept: mean clean-total possession points of the
    rating season (``JointAlphaEvaluator`` uses ``np.mean(total_train['y'])``)."""
    problem = _prepare_direct_ridge_problem(
        _load_clean_total_df([rating_year % 100], _season_type(rating_year))
    )
    return float(np.mean(np.asarray(problem["y"], dtype=float)))


def training_intercept(confirmation_year: int) -> float:
    total = 0.0
    rows = 0
    for year in range(FIRST_BOX_SEASON, confirmation_year):
        frame = _load_clean_total_df([year % 100], "ALL")
        assert_no_confirmation_rows(
            frame, confirmation_year, f"intercept training season {year}"
        )
        target = pd.to_numeric(frame["Net_Diff"], errors="coerce").fillna(0.0)
        total += float(target.sum())
        rows += len(target)
    return total / rows


def _prior_hash(frame: pd.DataFrame) -> str:
    payload = frame[["player_id", "off", "def"]].round(9).to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class ComponentSumEvaluator:
    """Solve the nine additive components on the rating season around a prior
    set and score the summed coefficient maps on the validation season's
    possessions (the ``tune_wnba_component_alphas`` fold construction).

    The primary endpoint uses a strictly forward fold
    (rating_year < validation_year); the secondary descriptive nowcast reuses
    the same machinery with rating_year == validation_year and no fit-side
    as-of bound, because it deliberately solves on the confirmation season.
    """

    def __init__(
        self,
        rating_year: int,
        validation_year: int,
        intercept: float,
        asof_bound: int | None,
    ) -> None:
        self.rating_year = rating_year
        self.validation_year = validation_year
        self.rating_season_type = _season_type(rating_year)
        self.validation_season_type = _season_type(validation_year)
        self.intercept = float(intercept)
        self.name_map = _load_name_map()
        print(
            f"Preparing validation clean-total problem {validation_year} "
            f"{self.validation_season_type}...",
            flush=True,
        )
        validation_df = _load_clean_total_df(
            [validation_year % 100], self.validation_season_type
        )
        self.validation_problem = _prepare_direct_ridge_problem(validation_df)
        self.game_vector = possession_game_ids(validation_df, self.validation_problem)
        self.component_problems: dict[str, dict[str, Any]] = {}
        for prefix in ADDITIVE_PREFIXES:
            print(
                f"Preparing component problem {prefix} ({rating_year} "
                f"{self.rating_season_type})...",
                flush=True,
            )
            self.component_problems[prefix] = _component_problem(
                prefix, rating_year, self.rating_season_type, asof_bound
            )
        self._solve_cache: dict[tuple[str, str], np.ndarray] = {}

    def component_vector(
        self,
        prefix: str,
        prior_frame: pd.DataFrame,
        off_alpha: float,
        def_alpha: float,
        prior_scale: float,
    ) -> np.ndarray:
        key = (prefix, _prior_hash(prior_frame))
        if key not in self._solve_cache:
            result = _solve_direct_ridge_problem(
                self.component_problems[prefix],
                self.name_map,
                off_alpha=off_alpha,
                def_alpha=def_alpha,
                prior_results=prior_frame,
                prior_scale=prior_scale,
            )
            self._solve_cache[key] = _result_vector(result, self.validation_problem)
        return self._solve_cache[key]

    def arm_predictions(
        self,
        priors: dict[str, pd.DataFrame],
        off_alpha: float,
        def_alpha: float,
        prior_scale: float,
    ) -> np.ndarray:
        coefficient_sum = np.zeros(self.validation_problem["X"].shape[1], dtype=float)
        for prefix in ADDITIVE_PREFIXES:
            coefficient_sum += self.component_vector(
                prefix, priors[prefix], off_alpha, def_alpha, prior_scale
            )
        return self.intercept + self.validation_problem["X"] @ coefficient_sum


def paired_game_bootstrap(
    y: np.ndarray,
    arm_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    game_vector: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    se_arm = np.square(y - arm_prediction)
    se_base = np.square(y - baseline_prediction)
    frame = pd.DataFrame(
        {"game": game_vector, "se_arm": se_arm, "se_base": se_base, "n": 1.0}
    )
    grouped = frame.groupby("game", sort=False).sum()
    arm_sums = grouped["se_arm"].to_numpy()
    base_sums = grouped["se_base"].to_numpy()
    counts = grouped["n"].to_numpy()
    n_games = len(grouped)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n_games, size=(iterations, n_games))
    boot_diff = (
        arm_sums[index].sum(axis=1) - base_sums[index].sum(axis=1)
    ) / counts[index].sum(axis=1)
    point = float((arm_sums.sum() - base_sums.sum()) / counts.sum())
    lower, upper = np.percentile(boot_diff, [2.5, 97.5])
    return {
        "games": int(n_games),
        "iterations": int(iterations),
        "seed": int(seed),
        "delta_mse": point,
        "delta_mse_x1e6": point * 1e6,
        "ci95_delta_mse_x1e6": [float(lower * 1e6), float(upper * 1e6)],
        "bootstrap_fraction_below_zero": float(np.mean(boot_diff < 0.0)),
    }


def evaluate_arms(
    evaluator: ComponentSumEvaluator,
    base_priors: dict[str, pd.DataFrame],
    corrections: dict[tuple[str, str], pd.Series],
    alphas: dict[str, Any],
    bootstrap_iterations: int,
) -> dict[str, Any]:
    """Score box-only and both corrected arms on the evaluator's fold."""
    y = np.asarray(evaluator.validation_problem["y"], dtype=float)
    predictions = {
        "box_only": evaluator.arm_predictions(
            base_priors, alphas["off_alpha"], alphas["def_alpha"], alphas["prior_scale"]
        )
    }
    for arm_name, arm_candidates in ARMS.items():
        arm_prior_set = build_arm_priors(base_priors, corrections, arm_candidates)
        predictions[arm_name] = evaluator.arm_predictions(
            arm_prior_set, alphas["off_alpha"], alphas["def_alpha"], alphas["prior_scale"]
        )
    mse = {name: float(np.mean(np.square(y - pred))) for name, pred in predictions.items()}
    arm_reports: dict[str, Any] = {"box_only": {"joint_mse": mse["box_only"]}}
    for arm_name in ARMS:
        bootstrap = paired_game_bootstrap(
            y,
            predictions[arm_name],
            predictions["box_only"],
            evaluator.game_vector,
            bootstrap_iterations,
            BOOTSTRAP_SEED,
        )
        arm_reports[arm_name] = {
            "joint_mse": mse[arm_name],
            "passes": bool(mse[arm_name] < mse["box_only"]),
            "paired_game_clustered_bootstrap": bootstrap,
        }
    return {
        "rating_year": evaluator.rating_year,
        "rating_season_type": evaluator.rating_season_type,
        "validation_year": evaluator.validation_year,
        "validation_season_type": evaluator.validation_season_type,
        "intercept": evaluator.intercept,
        "possessions": int(len(y)),
        "games": int(len(np.unique(evaluator.game_vector))),
        "arms": arm_reports,
    }


# ---------------------------------------------------------------------------
# Secondary descriptive per-component metrics (prereg step 7)
# ---------------------------------------------------------------------------


def _confirmation_target(
    confirmation_year: int,
    prefix: str,
    season_type: str,
    output_dir: Path,
) -> pd.DataFrame:
    if target_path(BOX_DIR, confirmation_year, prefix).exists():
        return load_target(BOX_DIR, confirmation_year, prefix)
    cache = output_dir / "confirmation_targets" / f"{prefix}.csv"
    if not cache.exists():
        result = run_metric(
            prefix,
            [confirmation_year % 100],
            season_type,
            _load_name_map(),
            off_alpha=TARGET_OFF_ALPHA,
            def_alpha=TARGET_DEF_ALPHA,
            time_decay_days=None,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False)
    frame = pd.read_csv(cache)
    frame["player_id"] = (
        pd.to_numeric(frame["player_id"], errors="coerce").astype("Int64").astype(str)
    )
    return frame


def secondary_component_metrics(
    confirmation_year: int,
    candidates: tuple[tuple[str, str, str], ...],
    corrections: dict[tuple[str, str], pd.Series],
    season_type: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix, side, lane in candidates:
        target = _confirmation_target(confirmation_year, prefix, season_type, output_dir)
        target = target[["player_id", side, f"{side}_poss"]].rename(
            columns={side: "target", f"{side}_poss": "target_poss"}
        )
        prior = _prior_frame(BOX_DIR, confirmation_year, prefix).rename(
            columns={side: "box_prior"}
        )[["player_id", "box_prior"]]
        merged = target.merge(prior, on="player_id", how="inner")
        correction = corrections[(prefix, side)]
        merged["correction"] = merged["player_id"].map(correction)
        merged = merged[
            (merged["target_poss"] >= MIN_TARGET_POSS)
            & merged["correction"].notna()
        ]
        if merged.empty:
            rows.append({"prefix": prefix, "side": side, "lane": lane, "rows": 0})
            continue
        weights = np.sqrt(merged["target_poss"].to_numpy(dtype=float))
        actual = merged["target"].to_numpy(dtype=float)
        box_metrics = _weighted_metrics(
            actual, merged["box_prior"].to_numpy(dtype=float), weights
        )
        augmented_metrics = _weighted_metrics(
            actual,
            (merged["box_prior"] + merged["correction"]).to_numpy(dtype=float),
            weights,
        )
        rows.append(
            {
                "prefix": prefix,
                "side": side,
                "lane": lane,
                "rows": int(len(merged)),
                "box_rmse": box_metrics["rmse"],
                "augmented_rmse": augmented_metrics["rmse"],
                "relative_rmse_change_pct": 100.0
                * (augmented_metrics["rmse"] - box_metrics["rmse"])
                / box_metrics["rmse"],
                "box_correlation": box_metrics["correlation"],
                "augmented_correlation": augmented_metrics["correlation"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Confirmation run (both real and dev-smoke shapes)
# ---------------------------------------------------------------------------


def recorded_cv_lanes() -> dict[str, str]:
    if not CV_BEST_PATH.exists():
        return {}
    best = pd.read_csv(CV_BEST_PATH)
    return {
        f"{row.prefix}/{row.side}": str(row.lane)
        for row in best.itertuples()
    }


def run_confirmation(
    confirmation_year: int,
    output_dir: Path,
    dev_smoke: bool,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    """``confirmation_year`` is the scoring/holdout season (2026 real,
    2025 dev-smoke); the primary rating season is the year before it."""
    if dev_smoke and confirmation_year >= CONFIRMATION_YEAR:
        raise RuntimeError("dev smoke must not touch the 2026 confirmation season")
    rating_year = confirmation_year - 1
    season_type = _season_type(confirmation_year)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_frozen_lanes_match_recorded_cv()
    alphas = load_frozen_alphas()

    for prior_year in (rating_year, confirmation_year):
        box_prior_path = BOX_DIR / f"box_priors_{prior_year}.csv"
        if not box_prior_path.exists():
            raise FileNotFoundError(
                f"{box_prior_path} missing; build it with build_wnba_box_priors.py "
                "(leakage-safe backbone trained strictly on earlier seasons)"
            )

    print("Loading box features and standardized RTSS features...", flush=True)
    box_by_year = {
        year: load_box_features(year)
        for year in range(FIRST_BOX_SEASON, confirmation_year)
    }
    for year, frame in box_by_year.items():
        assert_no_confirmation_rows(
            frame, confirmation_year, f"box feature season {year}"
        )
    rtss = load_standardized_rtss()

    candidates_all = ARMS["tier_a_plus_b"]

    # PRIMARY (promotion decision): strictly forward fold rating -> scoring.
    # Corrections fit strictly before the rating season, applied to
    # rating-season features; the scoring season's RTSS mirror is never read.
    print(
        f"Building PRIMARY corrections (fit < {rating_year}, applied to "
        f"{rating_year} features)...",
        flush=True,
    )
    primary_corrections, primary_correction_metadata = build_candidate_corrections(
        rating_year, candidates_all, box_by_year, rtss
    )
    primary_base_priors = {
        prefix: _prior_frame(BOX_DIR, rating_year, prefix)
        for prefix in ADDITIVE_PREFIXES
    }
    primary_evaluator = ComponentSumEvaluator(
        rating_year,
        confirmation_year,
        rating_year_intercept(rating_year),
        asof_bound=confirmation_year,
    )
    primary = evaluate_arms(
        primary_evaluator,
        primary_base_priors,
        primary_corrections,
        alphas,
        bootstrap_iterations,
    )

    # Prereg step 6 promotion rule, applied to the primary endpoint only.
    primary_mse = {name: report["joint_mse"] for name, report in primary["arms"].items()}
    passing = [name for name in ARMS if primary["arms"][name]["passes"]]
    if len(passing) == 2:
        promoted = min(passing, key=lambda name: primary_mse[name])
    elif len(passing) == 1:
        promoted = passing[0]
    else:
        promoted = None

    # SECONDARY (descriptive only): in-sample nowcast on the confirmation
    # season, mirroring the live nowcast prior-application contract. This is
    # the only consumer of the confirmation season's RTSS mirror.
    print(
        f"Building SECONDARY nowcast corrections (fit < {confirmation_year}, "
        f"applied to {confirmation_year} features)...",
        flush=True,
    )
    nowcast_corrections, nowcast_correction_metadata = build_candidate_corrections(
        confirmation_year, candidates_all, box_by_year, rtss
    )
    nowcast_base_priors = {
        prefix: _prior_frame(BOX_DIR, confirmation_year, prefix)
        for prefix in ADDITIVE_PREFIXES
    }
    nowcast_evaluator = ComponentSumEvaluator(
        confirmation_year,
        confirmation_year,
        training_intercept(confirmation_year),
        asof_bound=None,
    )
    nowcast = evaluate_arms(
        nowcast_evaluator,
        nowcast_base_priors,
        nowcast_corrections,
        alphas,
        bootstrap_iterations,
    )
    for report in nowcast["arms"].values():
        report.pop("passes", None)

    secondary = secondary_component_metrics(
        confirmation_year, candidates_all, nowcast_corrections, season_type, output_dir
    )

    verdict: dict[str, Any] = {
        "mode": "dev_smoke" if dev_smoke else "confirmation",
        "label": (
            "DEVELOPMENT EVIDENCE ONLY: forward-fold pseudo-confirmation "
            "2024->2025 on development data; proves pipeline shape, not the "
            "2026 verdict"
            if dev_smoke
            else "one-shot 2026 RTSS component-prior confirmation"
        ),
        "prereg": "nba_pipeline/docs/WNBA_RTSS_CONFIRMATION_PREREG.md",
        "confirmation_year": confirmation_year,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_parameters": {
            **alphas,
            "primary_box_backbone": str(BOX_DIR / f"box_priors_{rating_year}.csv"),
            "secondary_box_backbone": str(
                BOX_DIR / f"box_priors_{confirmation_year}.csv"
            ),
            "min_rtss_chances": MIN_RTSS_CHANCES,
            "min_target_poss": MIN_TARGET_POSS,
            "candidate_parent_mapping": CANDIDATE_PARENT,
            "frozen_lanes": {
                f"{prefix}/{side}": lane
                for prefix, side, lane in candidates_all
            },
            "recorded_cv_lanes_at_runtime": {
                key: value
                for key, value in recorded_cv_lanes().items()
                if key in {f"{p}/{s}" for p, s, _ in candidates_all}
            },
        },
        "primary_endpoint": {
            "construction": (
                "strictly forward fold: nine additive components solved on "
                f"rating season {rating_year} around each prior arm "
                "(tune_wnba_component_alphas construction, rating-season mean "
                "intercept), summed player coefficient maps predicting actual "
                f"{confirmation_year} possession points, paired game-clustered "
                f"bootstrap on {confirmation_year} games; corrections fit on "
                f"seasons strictly before {rating_year} and applied to "
                f"{rating_year} RTSS features (the {confirmation_year} RTSS "
                "mirror is not consumed)"
            ),
            **primary,
        },
        "promotion": {
            "promoted_arm": promoted,
            "null_result": promoted is None,
            "rule": (
                "arm passes iff it beats box-only joint MSE on the PRIMARY "
                "forward endpoint; both pass -> lower MSE; neither -> promote "
                "nothing; the secondary diagnostics cannot rescue or demote"
            ),
        },
        "secondary_nowcast_descriptive_only": {
            "label": (
                "SECONDARY DESCRIPTIVE DIAGNOSTIC: in-sample nowcast solved "
                f"and scored on the same {confirmation_year} possessions, "
                "mirroring the live nowcast prior-application contract; NOT "
                "the promotion decision"
            ),
            **nowcast,
        },
        "correction_models": {
            "primary": primary_correction_metadata,
            "secondary_nowcast": nowcast_correction_metadata,
        },
        "secondary_per_component_descriptive_only": secondary,
    }

    verdict_path = output_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, default=float) + "\n")
    print(f"Wrote {verdict_path}", flush=True)

    if not dev_smoke:
        # Fire exactly once: 'x' mode fails if the sentinel already exists.
        with SENTINEL_PATH.open("x") as handle:
            handle.write(
                json.dumps(
                    {
                        "fired_at_utc": verdict["generated_at_utc"],
                        "verdict": str(verdict_path),
                        "promoted_arm": promoted,
                    }
                )
                + "\n"
            )
        print(f"Wrote sentinel {SENTINEL_PATH}", flush=True)
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="print PASS/BLOCKED per fire condition; exit nonzero if blocked",
    )
    mode.add_argument(
        "--dev-smoke",
        action="store_true",
        help=(
            "run the full pipeline shape on development data only "
            "(forward fold 2024->2025 plus 2025 nowcast diagnostic)"
        ),
    )
    parser.add_argument(
        "--season-complete",
        action="store_true",
        help="explicit operator assertion that the 2026 regular season is complete",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    args = parser.parse_args()

    if args.dev_smoke:
        verdict = run_confirmation(
            DEV_SMOKE_YEAR,
            DEV_OUTPUT_DIR,
            dev_smoke=True,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print("PRIMARY forward fold arms:")
        print(json.dumps(verdict["primary_endpoint"]["arms"], indent=2, default=float))
        print(json.dumps(verdict["promotion"], indent=2))
        print("SECONDARY nowcast (descriptive only) arms:")
        print(
            json.dumps(
                verdict["secondary_nowcast_descriptive_only"]["arms"],
                indent=2,
                default=float,
            )
        )
        return 0

    checks = run_fire_checks(args.season_complete)
    for check in checks:
        status = "PASS" if check["passed"] else "BLOCKED"
        print(f"[{status}] {check['condition']}: {check['detail']}")
    blocked = [check["condition"] for check in checks if not check["passed"]]

    if args.check:
        print(
            "fire conditions: "
            + ("ALL PASS" if not blocked else f"BLOCKED: {', '.join(blocked)}")
        )
        return 0 if not blocked else 1

    if blocked:
        print(
            "REFUSING to fire the one-shot 2026 RTSS confirmation: blocked "
            f"fire conditions: {', '.join(blocked)}. This harness fires exactly "
            "once, only when every fire condition passes (see "
            "nba_pipeline/docs/WNBA_RTSS_CONFIRMATION_PREREG.md). Use "
            "--dev-smoke to exercise the pipeline on development data."
        )
        return 2

    verdict = run_confirmation(
        CONFIRMATION_YEAR,
        OUTPUT_DIR,
        dev_smoke=False,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print("PRIMARY forward fold arms:")
    print(json.dumps(verdict["primary_endpoint"]["arms"], indent=2, default=float))
    print(json.dumps(verdict["promotion"], indent=2))
    print("SECONDARY nowcast (descriptive only) arms:")
    print(
        json.dumps(
            verdict["secondary_nowcast_descriptive_only"]["arms"],
            indent=2,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

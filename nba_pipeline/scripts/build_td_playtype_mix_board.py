#!/usr/bin/env python3
"""Reader board for the time-decay playtype-mix RAPM, 1997-2026.

This is the descriptive companion to production zTS.  It reuses the exact
research solver in ``analyze_td_overall_playtype_mix_ots.py`` (one row per
complete-lineup regular-season FGA, lag-priced outcome-invariant playtype
difficulty, offense/defense/role alphas 3000/5000/10000, shot-share-centred
shooter interaction) with the owner-selected reader window:

* every feature season is solved on itself plus the prior season (two
  seasons) with a 550-day half-life, so the prior season carries roughly
  ``2 ** (-365 / 550)`` ~ 0.63 of current weight and every year is solved on
  the same short history;
* the attempt panel runs 1997-2026; each season is priced on the prior regular
  season, except 1997 (no 1996 raw file), which is priced on itself and solved
  on one season (``history_seasons == 1``, flagged in the manifest);
* names come from the observed rZTOV / scoring-security ledgers, then the
  curated exact-``nba_id`` ``autocomplete_map.csv`` for small-exposure players
  off those ledgers (``ID_<n>`` placeholders stay blank), never raw-PBP
  surnames; and
* the difficulty rates are converted to expected points per 100 offensive
  possessions with the measured league FGA-per-possession rate of the season.

Pre-2009 play-by-play descriptions almost never tag step-back / pull-up
threes, so the early-era mix mostly separates rim, other twos, threes, and
transition.  Each season's target is centred on its own mean difficulty
inside the solve, so descriptor drift moves the scale of the categories, not
the league level.

Positive ``mix_rapm`` means the lineup's overall shot diet is *harder* while
the player is on offense.  The exact identity

    mix_rapm = own_mix_contribution + teammate_mix_contribution
             = shot_share * own_diet + (1 - shot_share) * teammate_diet

is preserved from the solver.  ``*_value100`` fields are the same quantities
priced in expected points per 100 offensive possessions with positive good
(an easier diet), so they sit on the scoring-value scale.  They are prices of
the shot diet at prior-season league rates, not realized points.  This
two-season TD550 spec is ungated and the TD700 predecessor did not clear the
future-oTS admission gate as an increment: the board is descriptive only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import analyze_td_overall_playtype_mix_ots as mix
except ImportError:  # Direct script execution.
    import analyze_td_overall_playtype_mix_ots as mix

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "td_playtype_mix_board"
DEFAULT_OBSERVED_RZTOV = PIPELINE_ROOT / "results" / "rztov_seasons_2014_2026.csv"
DEFAULT_NAME_MAP = PIPELINE_ROOT.parent / "autocomplete_map.csv"
DEFAULT_SCORING = (
    PIPELINE_ROOT
    / "results"
    / "historical_scoring_security"
    / "historical_scoring_security_1997_2026.csv"
)
WINDOW_SEASONS = 2
DIFFICULTY_POINTS_PER_PP = 1.0 / 50.0  # difficulty_pp = 50 * points/FGA

LOGGER = logging.getLogger("td_playtype_mix_board")


def build_panel_with_counts(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the attempt panel and record complete-lineup coverage per season."""

    counts: list[dict[str, int]] = []
    original = mix._lineup_payload

    def counting_payload(frame: pd.DataFrame):
        kept, offense = original(frame)
        season = int(pd.to_numeric(frame.get("Season", pd.Series([0])), errors="coerce").iloc[0] or 0)
        counts.append({"all_fga": int(len(frame)), "complete_lineup_fga": int(len(kept)), "season_hint": season})
        return kept, offense

    mix._lineup_payload = counting_payload
    try:
        attempts, prices = mix.build_attempt_panel(
            pricing_start_season=args.pricing_start_season,
            end_season=args.last_feature_season,
            raw_dir=args.raw_dir,
            transition_dir=args.transition_dir,
            game_dates_path=args.game_dates,
            price_first_season_on_itself=True,
        )
    finally:
        mix._lineup_payload = original
    seasons = list(range(args.pricing_start_season, args.last_feature_season + 1))
    if len(counts) != len(seasons):
        raise RuntimeError("Lineup coverage counter did not see one call per season")
    coverage = pd.DataFrame(counts)
    coverage["season"] = seasons
    coverage["complete_lineup_share"] = coverage["complete_lineup_fga"] / coverage["all_fga"]
    return attempts, prices, coverage.drop(columns="season_hint")


def load_names_and_possessions(
    observed_path: Path, scoring_path: Path, name_map_path: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(observed_path, usecols=["nba_id", "name", "year", "off_poss"])
    scoring = pd.read_csv(scoring_path, usecols=["nba_id", "player_name", "year", "OffPoss"])
    names = pd.concat(
        [
            observed.rename(columns={"name": "player_name"})[["nba_id", "player_name", "year"]],
            scoring[["nba_id", "player_name", "year"]],
        ],
        ignore_index=True,
    ).dropna(subset=["player_name"])
    names = (
        names.sort_values("year")
        .drop_duplicates("nba_id", keep="last")[["nba_id", "player_name"]]
        .rename(columns={"nba_id": "player_id"})
    )
    if name_map_path is not None and name_map_path.exists():
        fallback = pd.read_csv(name_map_path, usecols=["nba_id", "player_name"]).dropna()
        fallback = fallback[~fallback["player_name"].str.startswith("ID_")]
        fallback = fallback.drop_duplicates("nba_id").rename(columns={"nba_id": "player_id"})
        names = pd.concat(
            [names, fallback[~fallback["player_id"].isin(names["player_id"])]], ignore_index=True
        )
    poss = scoring.rename(columns={"OffPoss": "off_poss"})[["nba_id", "year", "off_poss"]]
    return names, poss


def league_fga_rate(coverage: pd.DataFrame, poss: pd.DataFrame) -> pd.DataFrame:
    """League FGA per 100 team offensive possessions by season.

    Every on-court offensive possession is counted once for each of the five
    players, so team possessions are the player possession sum divided by five.
    """

    team_poss = poss.groupby("year")["off_poss"].sum().div(5.0).rename("team_off_poss")
    frame = coverage.merge(team_poss, left_on="season", right_index=True, how="left")
    frame["league_fga_per100"] = 100.0 * frame["all_fga"] / frame["team_off_poss"]
    return frame


def solve_board(
    attempts: pd.DataFrame,
    *,
    first: int,
    last: int,
    args: argparse.Namespace,
    names: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    parts = []
    metadata = []
    for feature_season in range(first, last + 1):
        lo = feature_season - args.window_seasons + 1
        fit = attempts[attempts["Season"].between(lo, feature_season)].copy()
        LOGGER.info("Solving feature season %d on %d-%d (%d FGA)", feature_season, fit["Season"].min(), feature_season, len(fit))
        result, meta, _ = mix.build_td_results(
            fit,
            feature_season=feature_season,
            half_life_days=args.half_life,
            off_alpha=args.off_alpha,
            def_alpha=args.def_alpha,
            role_alpha=args.role_alpha,
            names=names,
        )
        meta["first_fit_season"] = int(fit["Season"].min())
        result["history_seasons"] = int(feature_season - meta["first_fit_season"] + 1)
        parts.append(result)
        metadata.append(meta)
    return pd.concat(parts, ignore_index=True), metadata


def to_reader(board: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    rate = rates.set_index("season")["league_fga_per100"]
    out = pd.DataFrame(
        {
            "nba_id": board["player_id"].astype(int),
            "player_name": board["player_name"],
            "year": board["feature_season"].astype(int),
            "mix_rapm": board["td_overall_playtype_difficulty_rapm"],
            "own_diet": board["conditional_own_playtype_difficulty"],
            "teammate_diet": board["conditional_teammate_playtype_difficulty"],
            "own_mix_contribution": board["own_mix_contribution"],
            "teammate_mix_contribution": board["teammate_mix_contribution"],
            "td_shot_share": board["td_shot_share"],
            "td_fga_exposure": board["weighted_offensive_fga_exposure"],
            "def_mix_rapm": board["def_playtype_difficulty_rapm"],
            "history_seasons": board["history_seasons"].astype(int),
        }
    )
    out["league_fga_per100"] = out["year"].map(rate)
    price = DIFFICULTY_POINTS_PER_PP * out["league_fga_per100"]
    out["mix_value100"] = -out["mix_rapm"] * price
    out["own_mix_value100"] = -out["own_mix_contribution"] * price
    out["teammate_mix_value100"] = -out["teammate_mix_contribution"] * price
    return out


def season_on_court_fga(attempts: pd.DataFrame) -> pd.DataFrame:
    """Team FGA taken with each player on offense in each season (focal or teammate)."""
    long = attempts.melt(
        id_vars=["Season"], value_vars=["Focal_Player", "O1", "O2", "O3", "O4"], value_name="nba_id"
    )
    out = long.groupby(["Season", "nba_id"]).size().rename("season_on_court_fga").reset_index()
    return out.rename(columns={"Season": "year"}).astype({"nba_id": int, "year": int})


def validate(reader: pd.DataFrame) -> dict[str, float]:
    checks = {
        "parent_split_gap": float(
            (reader["own_mix_contribution"] + reader["teammate_mix_contribution"] - reader["mix_rapm"]).abs().max()
        ),
        "share_identity_gap": float(
            (
                reader["td_shot_share"] * reader["own_diet"]
                + (1.0 - reader["td_shot_share"]) * reader["teammate_diet"]
                - reader["mix_rapm"]
            ).abs().max()
        ),
        "value_split_gap": float(
            (reader["own_mix_value100"] + reader["teammate_mix_value100"] - reader["mix_value100"]).abs().max()
        ),
    }
    if max(checks.values()) > 1e-9:
        raise AssertionError(f"Playtype-mix identities failed: {checks}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pricing-start-season", type=int, default=1997)
    parser.add_argument("--first-feature-season", type=int, default=1997)
    parser.add_argument("--last-feature-season", type=int, default=2026)
    parser.add_argument("--window-seasons", type=int, default=WINDOW_SEASONS)
    parser.add_argument("--half-life", type=float, default=550.0)
    parser.add_argument("--off-alpha", type=float, default=3000.0)
    parser.add_argument("--def-alpha", type=float, default=5000.0)
    parser.add_argument("--role-alpha", type=float, default=10000.0)
    parser.add_argument("--raw-dir", type=Path, default=mix.DEFAULT_RAW_DIR)
    parser.add_argument("--transition-dir", type=Path, default=mix.DEFAULT_TRANSITION_DIR)
    parser.add_argument("--game-dates", type=Path, default=mix.DEFAULT_GAME_DATES)
    parser.add_argument("--observed-rztov", type=Path, default=DEFAULT_OBSERVED_RZTOV)
    parser.add_argument("--scoring-board", type=Path, default=DEFAULT_SCORING)
    parser.add_argument("--name-map", type=Path, default=DEFAULT_NAME_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-attempts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    span = f"{args.first_feature_season}_{args.last_feature_season}"
    attempts_path = args.output_dir / f"attempt_panel_{span}.parquet"
    prices_path = args.output_dir / "lag_category_prices.csv"
    coverage_path = args.output_dir / "lineup_coverage.csv"
    if attempts_path.exists() and coverage_path.exists() and not args.rebuild_attempts:
        attempts = pd.read_parquet(attempts_path)
        prices = pd.read_csv(prices_path)
        coverage = pd.read_csv(coverage_path)
    else:
        attempts, prices, coverage = build_panel_with_counts(args)
        attempts.to_parquet(attempts_path, index=False)
        prices.to_csv(prices_path, index=False)
        coverage.to_csv(coverage_path, index=False)

    names, poss = load_names_and_possessions(args.observed_rztov, args.scoring_board, args.name_map)
    rates = league_fga_rate(coverage, poss)
    board, metadata = solve_board(
        attempts,
        first=args.first_feature_season,
        last=args.last_feature_season,
        args=args,
        names=names,
    )
    reader = to_reader(board, rates)
    season_fga = season_on_court_fga(attempts)
    reader = reader.merge(season_fga, on=["nba_id", "year"], how="left")
    reader["season_on_court_fga"] = reader["season_on_court_fga"].fillna(0).astype(int)
    dropped_inactive = int(reader["season_on_court_fga"].eq(0).sum())
    reader = reader[reader["season_on_court_fga"].gt(0)].reset_index(drop=True)
    checks = validate(reader)
    reader = reader.sort_values(["year", "mix_rapm"], ascending=[True, False])
    reader.to_csv(args.output_dir / f"td_playtype_mix_board_{span}.csv", index=False)
    rates.to_csv(args.output_dir / "league_fga_rate.csv", index=False)

    manifest = {
        "status": "descriptive_only",
        "not": "not production zTS, not a prior, not a VPM input; ungated at this spec (the TD700 predecessor failed the future-oTS admission gate as an increment)",
        "solver": "analyze_td_overall_playtype_mix_ots.build_td_results (unchanged)",
        "sign": "mix_rapm positive = harder lineup shot diet with the player on offense; *_value100 positive = easier diet (more expected points)",
        "units": {
            "mix_rapm": "difficulty points per team FGA (50 * expected points per FGA; TS-point scale)",
            "value100": "expected points per 100 offensive possessions at prior-season league playtype prices",
        },
        "identity": "mix_rapm = own_mix_contribution + teammate_mix_contribution = td_shot_share*own_diet + (1-td_shot_share)*teammate_diet",
        "half_life_days": args.half_life,
        "alphas": {"offense": args.off_alpha, "defense": args.def_alpha, "role": args.role_alpha},
        "window_seasons": args.window_seasons,
        "pricing_start_season": args.pricing_start_season,
        "feature_seasons": [args.first_feature_season, args.last_feature_season],
        "history_caveat": (
            f"each feature season fits itself plus the prior {args.window_seasons - 1} season(s) at a "
            f"{args.half_life:g}-day half-life; {args.pricing_start_season} has no prior raw season, "
            "so it is priced on itself and solved alone (history_seasons == 1)"
        ),
        "same_season_pricing_fallback": [args.pricing_start_season],
        "descriptor_caveat": (
            "pre-2009 descriptions rarely tag step-back/pull-up threes, so SELF_CREATED_3 is near "
            "empty early; each season is centred on its own mean difficulty inside the solve"
        ),
        "spec_change": (
            "owner-selected 2026-09-26: two-season window, 550-day half-life, every season 1997-2026; "
            "supersedes the eight-season TD700 2015-2026 reader, whose future-oTS gate result "
            "(-0.0040 [-0.0233,+0.0136]) was measured on the TD700 research spec"
        ),
        "transition_caveat": "pre-2025 transition labels come from the classifier frozen on the 2025 labeled era; retrospective measurement",
        "identity_checks": checks,
        "rows": int(len(reader)),
        "rows_missing_player_name": int(reader["player_name"].isna().sum()),
        "active_filter": (
            "only player-seasons with season_on_court_fga > 0 are written; the trailing-window "
            "solve also returns inactive players, dropped here"
        ),
        "dropped_inactive_rows": dropped_inactive,
        "solve_metadata": metadata,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Wrote %d player-seasons; identity gaps %s", len(reader), checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

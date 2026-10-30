#!/usr/bin/env python3
"""Time-decay lineup turnover board, 1997-2026: what lineup data implies.

The box rZTOV ledger prices a player's *own* turnovers against a workload
allowance.  The lane point prices that convert it to points (about 0.54 per
scoring turnover and 0.77 per bad pass in 2022-26) are regressions of lineup
turnover RAPM on the box excess.  They are low mostly because a player's own
avoided turnover is only partly avoided by the team: some of it resurfaces as
a teammate or team turnover.  This board shows that lineup reading directly
instead of leaving it inside a price.

For every feature season it solves the owner's reader window (the feature
season plus the prior regular season, 550-day half-life as of the day after
the feature season's last game), exactly like the playtype-mix column, on
one standard regular-season possession per row:

* four 0/1 targets per possession: scoring turnover committed by one of the
  five offensive players, scoring turnover with no on-court offensive actor
  (team turnovers such as shot-clock, 5- and 8-second violations, plus the
  rare actor-lineup mismatch), and the same two for bad passes;
* a joint offense/defense parent ridge (alphas 3000/5000, the Daniel reader
  turnover parent) on each season-centred target with positive = saved;
* a centred focal own-turnover contrast on each attributed lane (five focal
  rows per possession, frozen role alpha from pooled weighted grouped-game
  CV), giving ``own = (parent + 4*contrast) / 5`` and
  ``teammates = 4*(parent - contrast) / 5``.

Ridge solutions are linear in the target, so on the identical design every
identity is exact:

    total = own + teammates + unassigned          (per lane, turnovers/100)
    lineup TOV saved = scoring total + bad-pass total

Points use the window's time-decay-weighted net-continuation first-chance
value of a non-turnover possession (``FC_TS_NET_Diff`` baseline, the
``ALT_TOV_NET_VALUE`` price): one averted turnover is one possession kept.
Raw on-court rates (time-decay weighted, league-relative) are written beside
the adjusted values so a reader can see what the ridge adjusted.

Source caveat: the 1997-2013 Gabriel-converted TOV possessions omit
team-actor turnovers, so their unassigned lane holds only lineup mismatches
and totals miss about 4% of turnovers.  ``team_tov_coverage`` is ``absent``
for windows with only pre-2014 seasons, ``partial`` for the 2014 window, and
``full`` from 2015.

Descriptive research only: not rZTOV, not a price refit, not added to any
scoring-board identity, not a prior or VPM input, and ungated.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, hstack
from sklearn.model_selection import GroupKFold

try:
    from . import build_v2_six_shooting_split as base
    from .build_td_playtype_mix_board import (
        DEFAULT_NAME_MAP,
        DEFAULT_OBSERVED_RZTOV,
        DEFAULT_SCORING,
        load_names_and_possessions,
    )
except ImportError:  # Direct script execution.
    import build_v2_six_shooting_split as base
    from build_td_playtype_mix_board import (
        DEFAULT_NAME_MAP,
        DEFAULT_OBSERVED_RZTOV,
        DEFAULT_SCORING,
        load_names_and_possessions,
    )

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "lineup_tov_board"
PROCESSED = PIPELINE_ROOT / "processed"
RAW = PIPELINE_ROOT / "raw_data"
NETCONT = PIPELINE_ROOT / "processed_netcont_lane"
FIRST_SEASON = 1997
LAST_SEASON = 2026
WINDOW_SEASONS = 2
HALF_LIFE = 550.0
OFF_ALPHA = 3000.0
DEF_ALPHA = 5000.0
ROLE_GRID = [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 1e12]
CV_WINDOWS = (2006, 2016, 2026)
TEAM_TOV_FIRST_SEASON = 2014
TEAM_ID_RANGE = (1610612000, 1610613000)
LANES = ("scoring", "badpass")
TARGETS = ("scoring_att", "scoring_un", "badpass_att", "badpass_un")
EPOCH = pd.Timestamp("1990-01-01")

LOGGER = logging.getLogger("lineup_tov_board")


def season_key(year: int) -> str:
    return f"{year % 100:02d}"


def load_season(year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """One standard RS possession per row with the turnover actor classified."""
    cols = ["game_id", "event_num", *base.OFF, *base.DEF, "Is_Turnover", "Is_BadPass_TOV", "game_date"]
    d = base.unique(base.norm(pd.read_parquet(PROCESSED / f"TOV{season_key(year)}.parquet", columns=cols)))
    raw = pd.read_parquet(
        RAW / f"NBA{season_key(year)}.parquet", columns=["game_id", "event_num", "event_type", "player1_id"]
    )
    raw = base.unique(base.norm(raw))
    events = raw.loc[raw["event_type"].eq(5), base.KEY + ["player1_id"]].rename(columns={"player1_id": "actor"})
    d = d.merge(events, on=base.KEY, how="left", validate="one_to_one")
    line = d[base.OFF + base.DEF].to_numpy(dtype=float)
    valid = (
        np.isfinite(line).all(axis=1)
        & (line > 0).all(axis=1)
        & (np.diff(np.sort(line, axis=1), axis=1) > 0).all(axis=1)
    )
    tov = d["Is_Turnover"].eq(1).to_numpy()
    bad = d["Is_BadPass_TOV"].eq(1).to_numpy() & tov
    actor = pd.to_numeric(d["actor"], errors="coerce").fillna(-1).to_numpy()
    on_court = (line[:, :5] == actor[:, None]).sum(axis=1) == 1
    team_actor = tov & (actor >= TEAM_ID_RANGE[0]) & (actor < TEAM_ID_RANGE[1])
    attributed = tov & on_court
    unassigned = tov & ~on_court
    audit = {
        "season": year,
        "input_possessions": int(len(d)),
        "invalid_lineup_possessions": int((~valid).sum()),
        "retained_possessions": int(valid.sum()),
        "turnovers": int(tov.sum()),
        "retained_turnovers": int((tov & valid).sum()),
        "attributed_share": float(attributed[valid].sum() / max((tov & valid).sum(), 1)),
        "team_actor_turnovers": int((team_actor & valid).sum()),
        "actor_lineup_mismatch_turnovers": int((unassigned & ~team_actor & valid).sum()),
        "badpass_share": float(bad[valid].sum() / max((tov & valid).sum(), 1)),
    }
    out = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": year,
            "day": (pd.to_datetime(d["game_date"]) - EPOCH).dt.days.to_numpy(),
            "actor": np.where(attributed, actor, 0).astype(np.int64),
            "scoring_att": (attributed & ~bad).astype(np.float64),
            "scoring_un": (unassigned & ~bad).astype(np.float64),
            "badpass_att": (attributed & bad).astype(np.float64),
            "badpass_un": (unassigned & bad).astype(np.float64),
        }
    )
    for column in base.OFF + base.DEF:
        out[column] = d[column].to_numpy(dtype=np.int64)
    return out[valid].reset_index(drop=True), audit


def net_baseline_rows(year: int, dates: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Non-turnover first-chance net value rows with game days for TD weighting."""
    fc = pd.read_parquet(
        NETCONT / f"FIRST_CHANCE{season_key(year)}.parquet", columns=["game_id", "Is_Turnover", "FC_TS_NET_Diff"]
    )
    fc = fc[fc["Is_Turnover"].eq(0)].copy()
    fc["game_id"] = pd.to_numeric(fc["game_id"], errors="raise").astype("int64")
    fc = fc.merge(dates, on="game_id", how="left", validate="many_to_one")
    undated = fc["day"].isna()
    if undated.mean() > 0.05:
        raise ValueError(f"FIRST_CHANCE{season_key(year)}: {undated.mean():.1%} of rows have no TOV game date")
    return fc.loc[~undated, ["day", "FC_TS_NET_Diff"]], int(undated.sum())


def design(frame: pd.DataFrame) -> tuple[object, object, np.ndarray]:
    ids = np.unique(frame[base.OFF + base.DEF].to_numpy())
    n = len(ids)
    rows = np.repeat(np.arange(len(frame)), 5)
    ones = np.ones(len(rows))
    off = coo_matrix(
        (ones, (rows, np.searchsorted(ids, frame[base.OFF].to_numpy().ravel()))), shape=(len(frame), n)
    ).tocsr()
    de = coo_matrix(
        (ones, (rows, np.searchsorted(ids, frame[base.DEF].to_numpy().ravel()))), shape=(len(frame), n)
    ).tocsr()
    return off, de, ids


def cholesky_solve(gram: np.ndarray, rhs: np.ndarray, ridge: np.ndarray | float) -> np.ndarray:
    matrix = gram.copy()
    matrix.flat[:: len(matrix) + 1] += ridge
    return cho_solve(cho_factor(matrix, lower=True, check_finite=False), rhs, check_finite=False)


def contrast_stats(frame: pd.DataFrame, off, ids: np.ndarray, weight: np.ndarray, lane: str):
    """Weighted normal equations for the centred focal own-turnover contrast.

    Five focal rows per possession with target ``-I(focal committed) + t/5``
    (centred within the possession, positive = fewer own turnovers than the
    lineup share).  Returns gram, right-hand side, and the target sum of squares.
    """
    t = frame[f"{lane}_att"].to_numpy()
    exposure = np.asarray(off.T @ weight).ravel()
    gram = np.diag(exposure) - (off.T @ off.multiply(weight[:, None])).toarray() / 5.0
    take = t == 1
    own = np.bincount(
        np.searchsorted(ids, frame["actor"].to_numpy()[take]), weights=weight[take], minlength=len(ids)
    )
    rhs = -own + np.asarray(off.T @ (weight * t)).ravel() / 5.0
    return gram, rhs, float(0.8 * np.sum(weight * t))


def td_weights(days: np.ndarray, asof: float, half_life: float) -> np.ndarray:
    return np.power(0.5, (asof - days) / half_life)


def window_frame(seasons: dict[int, pd.DataFrame], year: int, window: int) -> pd.DataFrame:
    parts = [seasons[s] for s in range(year - window + 1, year + 1) if s in seasons]
    return pd.concat(parts, ignore_index=True)


def select_role_alphas(seasons: dict[int, pd.DataFrame], args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    """Pooled grouped-game CV for the contrast penalty, one frozen alpha per lane.

    Three GroupKFold folds in each of the CV windows; training rows carry the
    time-decay weights, validation rows carry time-decay weights normalised to
    sum to one.  The largest alpha within one paired SE of the minimum wins.
    """
    losses: dict[str, list[list[float]]] = {lane: [] for lane in LANES}
    for year in args.cv_windows:
        frame = window_frame(seasons, year, args.window_seasons)
        asof = float(frame.loc[frame["season"].eq(year), "day"].max() + 1)
        weight = td_weights(frame["day"].to_numpy(dtype=float), asof, args.half_life)
        off, _, ids = design(frame)
        for train, valid in GroupKFold(3).split(frame, groups=frame["game_id"]):
            tr, va = frame.iloc[train], frame.iloc[valid]
            vw = weight[valid] / weight[valid].sum()
            for lane in LANES:
                gram, rhs, _ = contrast_stats(tr, off[train], ids, weight[train], lane)
                vgram, vrhs, const = contrast_stats(va, off[valid], ids, vw, lane)
                row = []
                for alpha in ROLE_GRID:
                    beta = cholesky_solve(gram, rhs, alpha)
                    row.append(float(const - 2 * beta @ vrhs + beta @ vgram @ beta))
                losses[lane].append(row)
        LOGGER.info("role CV window %s done", year)
    chosen, records = {}, []
    for lane in LANES:
        loss = np.array(losses[lane])
        means = loss.mean(axis=0)
        best = int(means.argmin())
        delta = loss - loss[:, [best]]
        se = delta.std(axis=0, ddof=1) / np.sqrt(len(loss))
        eligible = means - means[best] <= se + 1e-15
        chosen[lane] = max(a for a, ok in zip(ROLE_GRID, eligible) if ok)
        for alpha, mean, err in zip(ROLE_GRID, means, se):
            records.append(
                {
                    "lane": lane,
                    "alpha": alpha,
                    "mean_loss": mean,
                    "paired_se_vs_min": err,
                    "loss_vs_no_contrast": mean - means[-1],
                    "selected": alpha == chosen[lane],
                }
            )
    return chosen, pd.DataFrame(records)


def solve_window(
    seasons: dict[int, pd.DataFrame],
    baselines: dict[int, pd.DataFrame],
    year: int,
    role_alphas: dict[str, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = window_frame(seasons, year, args.window_seasons)
    history = sorted(frame["season"].unique().tolist())
    asof = float(frame.loc[frame["season"].eq(year), "day"].max() + 1)
    weight = td_weights(frame["day"].to_numpy(dtype=float), asof, args.half_life)
    off, de, ids = design(frame)
    n = len(ids)

    # Season-centred, positive = saved targets, weighted season means.
    season = frame["season"].to_numpy()
    targets = frame[list(TARGETS)].to_numpy()
    means = np.zeros_like(targets)
    for s in history:
        mask = season == s
        means[mask] = np.average(targets[mask], axis=0, weights=weight[mask])
    y = -(targets - means)

    stacked = hstack([off, de], format="csr")
    gram = (stacked.T @ stacked.multiply(weight[:, None])).toarray()
    rhs = np.asarray(stacked.T @ (weight[:, None] * y))
    penalty = np.concatenate([np.full(n, args.off_alpha), np.full(n, args.def_alpha)])
    parent = cholesky_solve(gram, rhs, penalty)[:n]  # offense block, one column per target

    exposure = np.asarray(off.T @ weight).ravel()
    parent = parent - np.average(parent, axis=0, weights=exposure)
    contrast = {}
    for lane in LANES:
        cgram, crhs, _ = contrast_stats(frame, off, ids, weight, lane)
        beta = cholesky_solve(cgram, crhs, role_alphas[lane])
        contrast[lane] = beta - np.average(beta, weights=exposure)

    # Raw, league-relative on-court rates (time-decay weighted), positive = saved.
    raw_saved = np.asarray(off.T @ (weight[:, None] * y)) / exposure[:, None]
    own_rows = {}
    for lane in LANES:
        t = frame[f"{lane}_att"].to_numpy()
        take = t == 1
        own_rows[lane] = np.bincount(
            np.searchsorted(ids, frame["actor"].to_numpy()[take]), weights=weight[take], minlength=n
        ) / exposure
    lane_mean_share = {}
    for lane in LANES:
        col = TARGETS.index(f"{lane}_att")
        lane_mean_share[lane] = np.asarray(off.T @ (weight * means[:, col])).ravel() / exposure

    fc = pd.concat([baselines[s] for s in history], ignore_index=True)
    fc_weight = td_weights(fc["day"].to_numpy(dtype=float), asof, args.half_life)
    price = float(np.average(fc["FC_TS_NET_Diff"].to_numpy(), weights=fc_weight))

    feature = frame["season"].eq(year).to_numpy()
    season_poss = np.asarray(off[feature].sum(axis=0)).ravel()

    out = pd.DataFrame({"nba_id": ids, "year": year})
    for lane in LANES:
        p_att = parent[:, TARGETS.index(f"{lane}_att")]
        p_un = parent[:, TARGETS.index(f"{lane}_un")]
        c = contrast[lane]
        out[f"{lane}_own_saved100"] = 100 * (p_att + 4 * c) / 5
        out[f"{lane}_teammate_saved100"] = 100 * 4 * (p_att - c) / 5
        out[f"{lane}_unassigned_saved100"] = 100 * p_un
        out[f"{lane}_total_saved100"] = 100 * (p_att + p_un)
        out[f"{lane}_role_contrast100"] = 100 * c
        raw_att = raw_saved[:, TARGETS.index(f"{lane}_att")]
        raw_own = lane_mean_share[lane] / 5 - own_rows[lane]
        out[f"{lane}_raw_own_saved100"] = 100 * raw_own
        out[f"{lane}_raw_teammate_saved100"] = 100 * (raw_att - raw_own)
        out[f"{lane}_raw_unassigned_saved100"] = 100 * raw_saved[:, TARGETS.index(f"{lane}_un")]
        out[f"{lane}_raw_total_saved100"] = 100 * (raw_att + raw_saved[:, TARGETS.index(f"{lane}_un")])
    out["lineup_tov_saved100"] = out["scoring_total_saved100"] + out["badpass_total_saved100"]
    out["tov_point_price"] = price
    for lane in LANES:
        out[f"{lane}_total_value100"] = out[f"{lane}_total_saved100"] * price
    out["lineup_tov_value100"] = out["lineup_tov_saved100"] * price
    out["td_off_exposure"] = exposure
    out["season_off_poss"] = season_poss.astype(int)
    out["history_seasons"] = len(history)
    out["team_tov_coverage"] = (
        "full" if min(history) >= TEAM_TOV_FIRST_SEASON else ("partial" if year >= TEAM_TOV_FIRST_SEASON else "absent")
    )
    league = {
        "year": year,
        "history": history,
        "rows": int(len(frame)),
        "players": int(n),
        "tov_point_price": price,
        **{
            f"league_{name}_per100": float(100 * np.average(frame.loc[feature, name], weights=weight[feature]))
            for name in TARGETS
        },
    }
    return out[out["season_off_poss"].gt(0)].reset_index(drop=True), league


def validate(board: pd.DataFrame) -> dict[str, float]:
    gaps = {}
    for lane in LANES:
        parts = board[[f"{lane}_own_saved100", f"{lane}_teammate_saved100", f"{lane}_unassigned_saved100"]].sum(axis=1)
        gaps[f"{lane}_split_gap"] = float((board[f"{lane}_total_saved100"] - parts).abs().max())
        raw_parts = board[
            [f"{lane}_raw_own_saved100", f"{lane}_raw_teammate_saved100", f"{lane}_raw_unassigned_saved100"]
        ].sum(axis=1)
        gaps[f"{lane}_raw_split_gap"] = float((board[f"{lane}_raw_total_saved100"] - raw_parts).abs().max())
        gaps[f"{lane}_value_gap"] = float(
            (board[f"{lane}_total_value100"] - board[f"{lane}_total_saved100"] * board["tov_point_price"]).abs().max()
        )
    gaps["lane_sum_gap"] = float(
        (board["lineup_tov_saved100"] - board["scoring_total_saved100"] - board["badpass_total_saved100"]).abs().max()
    )
    gaps["value_sum_gap"] = float(
        (board["lineup_tov_value100"] - board["scoring_total_value100"] - board["badpass_total_value100"]).abs().max()
    )
    if max(gaps.values()) > 1e-9:
        raise AssertionError(f"Lineup turnover identity failure: {gaps}")
    return gaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--last-season", type=int, default=LAST_SEASON)
    parser.add_argument("--window-seasons", type=int, default=WINDOW_SEASONS)
    parser.add_argument("--half-life", type=float, default=HALF_LIFE)
    parser.add_argument("--off-alpha", type=float, default=OFF_ALPHA)
    parser.add_argument("--def-alpha", type=float, default=DEF_ALPHA)
    parser.add_argument("--cv-windows", type=int, nargs="+", default=list(CV_WINDOWS))
    parser.add_argument("--role-alpha-scoring", type=float, default=None, help="skip CV and freeze this alpha")
    parser.add_argument("--role-alpha-badpass", type=float, default=None, help="skip CV and freeze this alpha")
    parser.add_argument("--observed-rztov", type=Path, default=DEFAULT_OBSERVED_RZTOV)
    parser.add_argument("--scoring-board", type=Path, default=DEFAULT_SCORING)
    parser.add_argument("--name-map", type=Path, default=DEFAULT_NAME_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    seasons, baselines, audits = {}, {}, []
    first_loaded = max(FIRST_SEASON, args.first_season - args.window_seasons + 1)
    for year in range(first_loaded, args.last_season + 1):
        frame, audit = load_season(year)
        seasons[year] = frame
        audits.append(audit)
        dates = frame[["game_id", "day"]].drop_duplicates("game_id")
        baselines[year], audit["price_rows_without_valid_lineup_game"] = net_baseline_rows(year, dates)
    coverage = pd.DataFrame(audits)
    coverage.to_csv(args.output_dir / "coverage.csv", index=False)
    LOGGER.info("loaded %d seasons in %.0fs", len(seasons), time.time() - start)

    if args.role_alpha_scoring is not None and args.role_alpha_badpass is not None:
        role_alphas = {"scoring": args.role_alpha_scoring, "badpass": args.role_alpha_badpass}
        cv_table = None
    else:
        role_alphas, cv_table = select_role_alphas(seasons, args)
        cv_table.to_csv(args.output_dir / "role_alpha_cv.csv", index=False)
    LOGGER.info("role alphas %s", role_alphas)

    boards, league = [], []
    for year in range(args.first_season, args.last_season + 1):
        board, summary = solve_window(seasons, baselines, year, role_alphas, args)
        boards.append(board)
        league.append(summary)
        LOGGER.info("solved %s (%d players, price %.4f)", year, len(board), summary["tov_point_price"])
    board = pd.concat(boards, ignore_index=True)
    checks = validate(board)

    names, _ = load_names_and_possessions(args.observed_rztov, args.scoring_board, args.name_map)
    board = board.merge(names.rename(columns={"player_id": "nba_id"}), on="nba_id", how="left")
    lead = ["nba_id", "player_name", "year", "season_off_poss", "td_off_exposure", "history_seasons", "team_tov_coverage"]
    board = board[lead + [c for c in board.columns if c not in lead]]
    span = f"{args.first_season}_{args.last_season}"
    board_path = args.output_dir / f"lineup_tov_board_{span}.csv"
    board.sort_values(["year", "lineup_tov_value100"], ascending=[True, False]).round(6).to_csv(board_path, index=False)
    league_frame = pd.DataFrame(league)
    league_frame["history"] = league_frame["history"].map(lambda h: "-".join(map(str, h)))
    league_frame.to_csv(args.output_dir / "league_summary.csv", index=False)

    manifest = {
        "status": "descriptive_research_only",
        "not": (
            "not rZTOV, not a refit of the scoring-board lane prices, never added to scoring value, turnover "
            "value, or scoring security; not a prior or VPM input; ungated"
        ),
        "question": "with the player on offense, how many team turnovers per 100 does the lineup avoid, and whose?",
        "window": {
            "seasons": args.window_seasons,
            "half_life_days": args.half_life,
            "asof": "day after the feature season's last regular-season game",
            "first_season_caveat": f"{FIRST_SEASON} is solved on one season (no NBA96 raw file)",
        },
        "parent": {
            "alphas": {"offense": args.off_alpha, "defense": args.def_alpha},
            "targets": list(TARGETS),
            "centering": "each target centred on its time-decay-weighted season mean; coefficients centred on TD exposure",
        },
        "role_contrast": {
            "alphas": role_alphas,
            "cv": (
                "three grouped-game folds in each of windows "
                f"{list(args.cv_windows)}, TD-weighted; largest alpha within one paired SE"
                if cv_table is not None
                else "frozen from the command line"
            ),
        },
        "identities": [
            "lane total = own + teammates + unassigned",
            "own = (parent_attributed + 4*contrast)/5, teammates = 4*(parent_attributed - contrast)/5",
            "lineup TOV saved = scoring total + bad-pass total",
            "value = saved * TD-weighted net non-turnover first-chance baseline",
        ],
        "sign": "positive = fewer turnovers than average with the player on offense (saved)",
        "units": "turnovers per 100 of the player's offensive possessions; value in points per 100",
        "team_tov_source_caveat": (
            "1997-2013 TOV possessions (Gabriel conversion) omit team-actor turnovers; unassigned holds only "
            "actor-lineup mismatches there. team_tov_coverage = absent (<=2013), partial (2014), full (2015+)"
        ),
        "identity_checks": checks,
        "rows": int(len(board)),
        "rows_missing_player_name": int(board["player_name"].isna().sum()),
        "outputs": {
            "board": str(board_path),
            "coverage": str(args.output_dir / "coverage.csv"),
            "league_summary": str(args.output_dir / "league_summary.csv"),
            "role_alpha_cv": str(args.output_dir / "role_alpha_cv.csv"),
        },
        "runtime_seconds": round(time.time() - start, 1),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOGGER.info("wrote %s (%d rows) in %.0fs", board_path, len(board), time.time() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score pre-tip WNBA forecasts against finals and closing odds.

Contract: verify both hash-chained logs first (broken chain = hard refuse,
exit 2). Scoring unit = per ``(game_date, home_tricode, away_tricode)``, the
LAST ``pretip_forecast`` record that passes the pre-tip fence: ``espn_state``
must be ``'pre'`` (a missing field on legacy records is accepted but counted
under ``espn_state_unknown_selected``) AND ``emitted_at_utc`` must be strictly
before the tipoff timestamp when one is available from the captured odds
records or the games table. Records failing the fence are excluded and tallied
(``post_tip_excluded`` / ``nonpre_state_excluded``); a key with no tipoff
timestamp falls back to its last eligible record and is counted under
``tipoff_unknown_keys``; a key whose EVERY record fails the fence becomes a
REFUSED row. Unknown extra fields on log records are ignored (forward-
compatible schema). Finals join the same key against ``status == 3`` rows of
the games table; odds join takes the last ``state == "pre"`` snapshot as the
close proxy. Any ambiguous join (duplicate final keys, conflicting tipoffs,
impossible/missing final scores, malformed forecast fields) emits verdict
REFUSED with a reason — never a guess; refused rows are excluded from metrics
but counted loudly. A forecast whose key matches only a not-yet-final game is
PENDING; a key absent from the games table entirely is PENDING when its
game_date is within one day of the table's max game_date (the table holds only
finals, so a just-played game simply has not landed yet), REFUSED otherwise.

Metrics: per-game margin error (``margin_mean - actual_margin``) / abs /
squared, pooled RMSE and MAE, win log-loss and Brier from ``p_home_win``,
PIT = Phi((actual - mean) / sigma) with one-sample KS distance from uniform,
90% central PIT coverage, and mean closed-form normal CRPS of
``N(margin_mean, sigma)`` against the realized margin. Market comparison
(odds rows only): model vs ``market_home_margin`` RMSE, log-loss of
``p_home_win`` vs ``p_home_implied_devig``, CRPS of the close
(``N(market_home_margin, same model sigma)``, outcome space, matched
dispersion), and a flat-stake spread record (bet home when
``margin_mean - market_home_margin >= threshold``, away when
``<= -threshold``; push = spread exactly hit). The simulation is log-scoring
only — no staking output, no bankroll math. Ratings and forecasts upstream
are market-blind; this scoring layer alone sees odds.

Outputs ``{output_dir}/report.json`` and ``scored_forecasts.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wnba_gameday_common as common

TIPOFF_CONFLICT_HOURS = 6.0
PROB_EPS = 1e-12
PUSH_TOL = 1e-9

CSV_COLUMNS = [
    "game_date",
    "home_tricode",
    "away_tricode",
    "verdict",
    "reason",
    "espn_event_id",
    "cdn_game_id",
    "game_id",
    "emitted_at_utc",
    "tipoff_utc",
    "espn_state",
    "model_version",
    "minutes_source",
    "low_coverage",
    "default_params",
    "margin_mean",
    "margin_sigma",
    "p_home_win",
    "home_score",
    "away_score",
    "actual_margin",
    "home_win",
    "margin_error",
    "abs_error",
    "sq_error",
    "win_log_loss",
    "brier",
    "pit",
    "in_90_central",
    "crps",
    "has_odds",
    "odds_captured_at_utc",
    "odds_provider",
    "spread_home",
    "market_home_margin",
    "home_ml",
    "away_ml",
    "over_under",
    "p_home_implied_devig",
    "crps_close",
    "edge_vs_market",
    "bet_side",
    "bet_result",
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def as_float(value: Any) -> Optional[float]:
    """Lenient scalar -> float; None for missing/NaN/unparseable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def parse_utc(value: Any) -> Optional[datetime]:
    """ISO-ish timestamp -> naive UTC datetime, or None if unparseable."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_crps(mu: float, sigma: float, realized: float) -> float:
    """Closed-form CRPS of N(mu, sigma^2) against a realized value.

    Gneiting & Raftery (2007): CRPS = sigma * (z*(2*Phi(z)-1) + 2*phi(z)
    - 1/sqrt(pi)) with z = (realized - mu) / sigma. Lower is better; equals
    absolute error in the sigma -> 0 limit.
    """
    z = (realized - mu) / sigma
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def parse_iso_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return None


def clip_prob(p: float) -> float:
    return min(max(p, PROB_EPS), 1.0 - PROB_EPS)


def log_loss_one(p: float, y: int) -> float:
    p = clip_prob(p)
    return -(math.log(p) if y == 1 else math.log(1.0 - p))


def mean(values: list) -> Optional[float]:
    return sum(values) / len(values) if values else None


def rmse(values: list) -> Optional[float]:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else None


def ks_uniform(pits: list) -> Optional[float]:
    """One-sample KS distance of pits from Uniform(0,1)."""
    if not pits:
        return None
    xs = sorted(pits)
    n = len(xs)
    d = 0.0
    for i, x in enumerate(xs, start=1):
        d = max(d, abs(i / n - x), abs(x - (i - 1) / n))
    return d


def r6(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 6)


def join_key(game_date: Any, home: Any, away: Any) -> tuple:
    return (
        str(game_date).strip(),
        str(home).strip().upper(),
        str(away).strip().upper(),
    )


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def read_log_records(path: Path) -> list:
    """Read a (pre-verified) hash-chained JSONL log, stripping chain fields."""
    records = []
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("prev_hash", None)
            row.pop("record_hash", None)
            records.append(row)
    return records


def last_per_key(records: list) -> dict:
    """Last record per join key, in first-seen key order (append order wins)."""
    out: dict = {}
    for record in records:
        key = join_key(
            record.get("game_date"), record.get("home_tricode"), record.get("away_tricode")
        )
        out[key] = record
    return out


def build_tipoff_lookup(odds_records: list, games_tipoffs: dict) -> dict:
    """key -> tipoff datetime for the pre-tip fence.

    Games-table times seed the lookup; captured odds snapshots (any state)
    override because they were recorded independently of the forecast being
    fenced. The forecast's own self-reported tipoff is deliberately not used.
    """
    lookup = dict(games_tipoffs)
    for record in odds_records:
        if record.get("type") != "odds_snapshot":
            continue
        key = join_key(
            record.get("game_date"), record.get("home_tricode"), record.get("away_tricode")
        )
        tip = parse_utc(record.get("tipoff_utc"))
        if tip is not None:
            lookup[key] = tip
    return lookup


def select_pretip_forecasts(forecasts: list, tipoff_by_key: dict) -> tuple:
    """Apply the pre-tip fence and pick one record per key.

    Eligible = ``espn_state`` is 'pre' or absent (legacy, flagged) AND, when a
    tipoff timestamp is known for the key, ``emitted_at_utc`` parses and is
    strictly before it (fail closed on unparseable emission times). Returns
    ``(selected_by_key, fenced_out_by_key, fence_counts)`` where
    ``fenced_out_by_key`` maps keys whose EVERY record failed the fence to
    ``(n_records, last_record)`` so they surface as REFUSED rows, never
    silently vanish.
    """
    records_by_key: dict = {}
    for record in forecasts:
        key = join_key(
            record.get("game_date"), record.get("home_tricode"), record.get("away_tricode")
        )
        records_by_key.setdefault(key, []).append(record)

    selected: dict = {}
    fenced_out: dict = {}
    counts = {
        "post_tip_excluded": 0,
        "nonpre_state_excluded": 0,
        "tipoff_unknown_keys": 0,
        "espn_state_unknown_selected": 0,
        "superseded_forecast_records": 0,
    }
    for key, records in records_by_key.items():
        tipoff = tipoff_by_key.get(key)
        eligible = []
        for record in records:
            state = record.get("espn_state")
            if state is not None and str(state) != "pre":
                counts["nonpre_state_excluded"] += 1
                continue
            if tipoff is not None:
                emitted = parse_utc(record.get("emitted_at_utc"))
                if emitted is None or emitted >= tipoff:
                    counts["post_tip_excluded"] += 1
                    continue
            eligible.append(record)
        if not eligible:
            fenced_out[key] = (len(records), records[-1])
            continue
        if tipoff is None:
            counts["tipoff_unknown_keys"] += 1
        counts["superseded_forecast_records"] += len(eligible) - 1
        chosen = eligible[-1]
        if chosen.get("espn_state") is None:
            counts["espn_state_unknown_selected"] += 1
        selected[key] = chosen
    return selected, fenced_out, counts


def load_games_table(path: Path) -> tuple:
    """Games table -> (finals_map key->[rows], nonfinal_keys, n_final, missing,
    tipoffs key->datetime, max_game_date or None)."""
    finals_map: dict = {}
    nonfinal_keys: set = set()
    tipoffs: dict = {}
    max_game_date: Optional[date] = None
    if not path.exists():
        return finals_map, nonfinal_keys, 0, True, tipoffs, max_game_date
    games = pd.read_csv(path, dtype={"game_id": str})
    n_final = 0
    for _, row in games.iterrows():
        key = join_key(row.get("game_date"), row.get("home_tricode"), row.get("away_tricode"))
        status = as_float(row.get("status"))
        if status is not None and int(status) == 3:
            finals_map.setdefault(key, []).append(row.to_dict())
            n_final += 1
        else:
            nonfinal_keys.add(key)
        tip = parse_utc(row.get("game_time_utc"))
        if tip is not None and key not in tipoffs:
            tipoffs[key] = tip
        row_date = parse_iso_date(row.get("game_date"))
        if row_date is not None and (max_game_date is None or row_date > max_game_date):
            max_game_date = row_date
    return finals_map, nonfinal_keys, n_final, False, tipoffs, max_game_date


# --------------------------------------------------------------------------
# Per-forecast evaluation
# --------------------------------------------------------------------------


def fence_refused_row(key: tuple, n_records: int, last_record: dict) -> dict:
    """REFUSED row for a key whose every record failed the pre-tip fence."""
    out: dict = {column: None for column in CSV_COLUMNS}
    out["game_date"], out["home_tricode"], out["away_tricode"] = key
    for field in (
        "espn_event_id",
        "cdn_game_id",
        "emitted_at_utc",
        "tipoff_utc",
        "espn_state",
        "model_version",
        "minutes_source",
        "low_coverage",
        "default_params",
    ):
        out[field] = last_record.get(field)
    out["has_odds"] = False
    out["verdict"] = "REFUSED"
    out["reason"] = (
        f"all {n_records} record(s) for this key failed the pre-tip fence "
        f"(post-tip emission or non-pre espn_state)"
    )
    return out


def evaluate_forecast(
    key: tuple,
    forecast: dict,
    finals_map: dict,
    nonfinal_keys: set,
    games_missing: bool,
    recent_cutoff: Optional[date],
    odds_map: dict,
    edge_threshold: float,
) -> dict:
    """Score one forecast key. FAIL CLOSED: ambiguity -> REFUSED, never a guess."""
    out: dict = {column: None for column in CSV_COLUMNS}
    out["game_date"], out["home_tricode"], out["away_tricode"] = key
    for field in (
        "espn_event_id",
        "cdn_game_id",
        "emitted_at_utc",
        "tipoff_utc",
        "espn_state",
        "model_version",
        "minutes_source",
        "low_coverage",
        "default_params",
    ):
        out[field] = forecast.get(field)
    out["margin_mean"] = as_float(forecast.get("margin_mean"))
    out["margin_sigma"] = as_float(forecast.get("margin_sigma"))
    out["p_home_win"] = as_float(forecast.get("p_home_win"))
    out["has_odds"] = False

    def refused(reason: str) -> dict:
        out["verdict"] = "REFUSED"
        out["reason"] = reason
        return out

    # -- finals join ------------------------------------------------------
    if games_missing:
        return refused("games table missing; cannot join finals")
    final_rows = finals_map.get(key)
    if final_rows is None:
        if key in nonfinal_keys:
            out["verdict"] = "PENDING"
            out["reason"] = "matching game exists but is not final (status != 3)"
            return out
        # The games table holds only finals, so an absent key can simply mean
        # the game has not finished/landed yet. PENDING when the forecast date
        # is within one day of the table's horizon; REFUSED (fail closed) for
        # genuinely unmatchable past games.
        forecast_date = parse_iso_date(key[0])
        if forecast_date is None:
            return refused("unparseable forecast game_date; cannot classify as pending")
        if recent_cutoff is None or forecast_date >= recent_cutoff:
            out["verdict"] = "PENDING"
            out["reason"] = (
                "no matching game yet; game_date is within one day of the "
                "games table horizon (likely not final yet)"
            )
            return out
        return refused("no matching game in games table for (game_date, home, away)")
    if len(final_rows) > 1:
        return refused(f"duplicate final rows in games table ({len(final_rows)} rows share key)")
    final = final_rows[0]
    out["game_id"] = final.get("game_id")
    home_score = as_float(final.get("home_score"))
    away_score = as_float(final.get("away_score"))
    if home_score is None or away_score is None:
        return refused("final game row missing home/away score")
    if home_score < 0 or away_score < 0:
        return refused("final game row has negative score")
    if home_score == away_score:
        return refused("final game row has tied score (impossible for a final)")
    forecast_tip = parse_utc(forecast.get("tipoff_utc"))
    games_tip = parse_utc(final.get("game_time_utc"))
    if forecast_tip is not None and games_tip is not None:
        gap_hours = abs((forecast_tip - games_tip).total_seconds()) / 3600.0
        if gap_hours > TIPOFF_CONFLICT_HOURS:
            return refused(
                f"conflicting tipoffs: forecast vs games table differ by {gap_hours:.1f}h"
            )

    # -- forecast field sanity -------------------------------------------
    margin_mean = out["margin_mean"]
    margin_sigma = out["margin_sigma"]
    p_home_win = out["p_home_win"]
    if margin_mean is None or margin_sigma is None or margin_sigma <= 0:
        return refused("malformed forecast: missing margin_mean or nonpositive margin_sigma")
    if p_home_win is None or not (0.0 <= p_home_win <= 1.0):
        return refused("malformed forecast: p_home_win missing or outside [0, 1]")

    # -- odds join (close proxy = last pre snapshot) ---------------------
    odds = odds_map.get(key)
    if odds is not None:
        odds_tip = parse_utc(odds.get("tipoff_utc"))
        if forecast_tip is not None and odds_tip is not None:
            gap_hours = abs((forecast_tip - odds_tip).total_seconds()) / 3600.0
            if gap_hours > TIPOFF_CONFLICT_HOURS:
                return refused(
                    f"conflicting tipoffs: forecast vs odds snapshot differ by {gap_hours:.1f}h"
                )
        out["has_odds"] = True
        out["odds_captured_at_utc"] = odds.get("captured_at_utc")
        out["odds_provider"] = odds.get("provider")
        out["spread_home"] = as_float(odds.get("spread_home"))
        out["market_home_margin"] = as_float(odds.get("market_home_margin"))
        out["home_ml"] = as_float(odds.get("home_ml"))
        out["away_ml"] = as_float(odds.get("away_ml"))
        out["over_under"] = as_float(odds.get("over_under"))
        out["p_home_implied_devig"] = as_float(odds.get("p_home_implied_devig"))

    # -- scoring ----------------------------------------------------------
    actual_margin = float(home_score - away_score)
    home_win = 1 if actual_margin > 0 else 0
    out["home_score"] = int(home_score)
    out["away_score"] = int(away_score)
    out["actual_margin"] = actual_margin
    out["home_win"] = home_win
    out["margin_error"] = margin_mean - actual_margin
    out["abs_error"] = abs(out["margin_error"])
    out["sq_error"] = out["margin_error"] ** 2
    out["win_log_loss"] = log_loss_one(p_home_win, home_win)
    out["brier"] = (p_home_win - home_win) ** 2
    pit = norm_cdf((actual_margin - margin_mean) / margin_sigma)
    out["pit"] = pit
    out["in_90_central"] = bool(0.05 <= pit <= 0.95)
    out["crps"] = normal_crps(margin_mean, margin_sigma, actual_margin)

    market_margin = out["market_home_margin"]
    if market_margin is not None:
        # Close CRPS in outcome space at matched dispersion: the market's
        # implied margin (already -spread_home in the odds records) with the
        # MODEL's sigma, so model-vs-close differs only in the mean.
        out["crps_close"] = normal_crps(market_margin, margin_sigma, actual_margin)
        edge = margin_mean - market_margin
        out["edge_vs_market"] = edge
        if edge >= edge_threshold:
            out["bet_side"] = "home"
        elif edge <= -edge_threshold:
            out["bet_side"] = "away"
        if out["bet_side"] is not None:
            if math.isclose(actual_margin, market_margin, abs_tol=PUSH_TOL):
                out["bet_result"] = "push"
            elif (actual_margin > market_margin) == (out["bet_side"] == "home"):
                out["bet_result"] = "win"
            else:
                out["bet_result"] = "loss"

    out["verdict"] = "SCORED"
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def build_report(
    rows: list,
    chain_info: dict,
    counts: dict,
    edge_threshold: float,
    inputs: dict,
    notes: list,
) -> dict:
    scored = [row for row in rows if row["verdict"] == "SCORED"]
    refused = [row for row in rows if row["verdict"] == "REFUSED"]
    pending = [row for row in rows if row["verdict"] == "PENDING"]

    errors = [row["margin_error"] for row in scored]
    pits = [row["pit"] for row in scored]
    model_metrics = {
        "n_scored": len(scored),
        "n_home_wins": sum(row["home_win"] for row in scored),
        "margin_rmse": r6(rmse(errors)),
        "margin_mae": r6(mean([abs(e) for e in errors])),
        "margin_mean_error": r6(mean(errors)),
        "win_log_loss": r6(mean([row["win_log_loss"] for row in scored])),
        "brier": r6(mean([row["brier"] for row in scored])),
        "pit_ks_distance": r6(ks_uniform(pits)),
        "coverage_90_central": r6(mean([1.0 if row["in_90_central"] else 0.0 for row in scored])),
        "crps_mean": r6(mean([row["crps"] for row in scored])),
    }

    odds_rows = [row for row in scored if row["has_odds"]]
    margin_rows = [row for row in odds_rows if row["market_home_margin"] is not None]
    prob_rows = [row for row in odds_rows if row["p_home_implied_devig"] is not None]
    bets = [row for row in margin_rows if row["bet_side"] is not None]
    wins = sum(1 for row in bets if row["bet_result"] == "win")
    losses = sum(1 for row in bets if row["bet_result"] == "loss")
    pushes = sum(1 for row in bets if row["bet_result"] == "push")
    market_comparison = {
        "n_scored_with_odds": len(odds_rows),
        "n_with_market_margin": len(margin_rows),
        "model_margin_rmse_odds_rows": r6(rmse([row["margin_error"] for row in margin_rows])),
        "market_margin_rmse": r6(
            rmse([row["market_home_margin"] - row["actual_margin"] for row in margin_rows])
        ),
        "n_with_market_prob": len(prob_rows),
        "model_win_log_loss_odds_rows": r6(
            mean([log_loss_one(row["p_home_win"], row["home_win"]) for row in prob_rows])
        ),
        "market_win_log_loss": r6(
            mean([log_loss_one(row["p_home_implied_devig"], row["home_win"]) for row in prob_rows])
        ),
        "crps_note": (
            "close CRPS uses N(market_home_margin, model sigma): outcome-space "
            "model-vs-close comparison at matched dispersion, not just CLV"
        ),
        "model_crps_odds_rows": r6(mean([row["crps"] for row in margin_rows])),
        "market_close_crps": r6(mean([row["crps_close"] for row in margin_rows])),
        "spread_simulation": {
            "note": "flat-stake log-scoring only; no staking output, no bankroll math",
            "edge_threshold_pts": edge_threshold,
            "n_bets": len(bets),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "record": f"{wins}-{losses}-{pushes}",
            "hit_rate": r6(wins / (wins + losses)) if (wins + losses) > 0 else None,
        },
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": inputs,
        "log_chains": chain_info,
        "counts": dict(
            counts,
            scored=len(scored),
            refused=len(refused),
            pending=len(pending),
            scored_with_odds=len(odds_rows),
            scored_low_coverage=sum(1 for row in scored if bool(row["low_coverage"])),
        ),
        "refused_rows": [
            {
                "game_date": row["game_date"],
                "home_tricode": row["home_tricode"],
                "away_tricode": row["away_tricode"],
                "reason": row["reason"],
            }
            for row in refused
        ],
        "pending_rows": [
            {
                "game_date": row["game_date"],
                "home_tricode": row["home_tricode"],
                "away_tricode": row["away_tricode"],
            }
            for row in pending
        ],
        "model_metrics": model_metrics,
        "market_comparison": market_comparison,
        "notes": notes,
    }
    if not scored:
        report["notes"] = list(notes) + ["0 scoreable forecasts; all metrics are null"]
    return report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--forecast-log",
        type=Path,
        default=common.FORECAST_LOG_DIR / "forecasts_2026.jsonl",
        help="hash-chained pretip_forecast JSONL",
    )
    parser.add_argument(
        "--odds-log",
        type=Path,
        default=common.ODDS_LOG_DIR / "odds_2026.jsonl",
        help="hash-chained odds_snapshot JSONL",
    )
    parser.add_argument(
        "--games",
        type=Path,
        default=common.GAMES_TABLE_PATH,
        help="games.csv finals table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=common.GAMEDAY_DIR / "scoring",
        help="where report.json and scored_forecasts.csv are written",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=2.0,
        help="points of model-vs-market margin edge required to log a simulated bet",
    )
    args = parser.parse_args(argv)

    # Hard refuse on any broken chain before reading a single record.
    chain_info = {}
    for label, path in (("forecast", args.forecast_log), ("odds", args.odds_log)):
        ok, n_records, reason = common.verify_log_chain(path)
        chain_info[label] = {
            "path": str(path),
            "chain_ok": ok,
            "n_records": n_records,
            "exists": path.exists(),
        }
        if not ok:
            print(
                f"HARD REFUSE: {label} log hash chain broken at {path}: {reason}",
                file=sys.stderr,
            )
            return 2

    forecast_records = read_log_records(args.forecast_log)
    odds_records = read_log_records(args.odds_log)
    forecasts = [row for row in forecast_records if row.get("type") == "pretip_forecast"]
    pre_odds = [
        row
        for row in odds_records
        if row.get("type") == "odds_snapshot" and row.get("state") == "pre"
    ]
    odds_by_key = last_per_key(pre_odds)
    finals_map, nonfinal_keys, n_final, games_missing, games_tipoffs, max_game_date = (
        load_games_table(args.games)
    )
    recent_cutoff = max_game_date - timedelta(days=1) if max_game_date else None

    tipoff_by_key = build_tipoff_lookup(odds_records, games_tipoffs)
    forecast_by_key, fenced_out, fence_counts = select_pretip_forecasts(
        forecasts, tipoff_by_key
    )

    notes: list = []
    if games_missing:
        notes.append(f"games table not found at {args.games}; every forecast refused")
    n_ignored = len(forecast_records) - len(forecasts)
    if n_ignored:
        notes.append(f"{n_ignored} non-pretip_forecast records ignored in forecast log")
    if fence_counts["post_tip_excluded"]:
        notes.append(
            f"{fence_counts['post_tip_excluded']} forecast record(s) excluded by "
            "the pre-tip fence (emitted at or after tipoff)"
        )
    if fence_counts["nonpre_state_excluded"]:
        notes.append(
            f"{fence_counts['nonpre_state_excluded']} forecast record(s) excluded "
            "for non-pre espn_state"
        )

    rows = [
        evaluate_forecast(
            key,
            forecast,
            finals_map,
            nonfinal_keys,
            games_missing,
            recent_cutoff,
            odds_by_key,
            args.edge_threshold,
        )
        for key, forecast in forecast_by_key.items()
    ]
    rows.extend(
        fence_refused_row(key, n_records, last_record)
        for key, (n_records, last_record) in fenced_out.items()
    )

    counts = {
        "forecast_log_records": len(forecast_records),
        "pretip_forecast_records": len(forecasts),
        "unique_forecast_keys": len(forecast_by_key) + len(fenced_out),
        "superseded_forecast_records": fence_counts["superseded_forecast_records"],
        "post_tip_excluded": fence_counts["post_tip_excluded"],
        "nonpre_state_excluded": fence_counts["nonpre_state_excluded"],
        "tipoff_unknown_keys": fence_counts["tipoff_unknown_keys"],
        "espn_state_unknown_selected": fence_counts["espn_state_unknown_selected"],
        "pretip_fence_refused_keys": len(fenced_out),
        "odds_log_records": len(odds_records),
        "pre_odds_snapshots": len(pre_odds),
        "unique_pre_odds_keys": len(odds_by_key),
        "final_games_in_table": n_final,
    }
    inputs = {
        "forecast_log": str(args.forecast_log),
        "odds_log": str(args.odds_log),
        "games": str(args.games),
        "edge_threshold_pts": args.edge_threshold,
    }
    report = build_report(rows, chain_info, counts, args.edge_threshold, inputs, notes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    csv_path = args.output_dir / "scored_forecasts.csv"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)

    # Loud console summary — refusals are never silent.
    print(
        "scored {scored} | pending {pending} | REFUSED {refused} "
        "(of {unique} unique forecast keys)".format(
            scored=report["counts"]["scored"],
            pending=report["counts"]["pending"],
            refused=report["counts"]["refused"],
            unique=counts["unique_forecast_keys"],
        )
    )
    for row in report["refused_rows"]:
        print(
            "  REFUSED {game_date} {away_tricode} @ {home_tricode}: {reason}".format(**row)
        )
    for row in report["pending_rows"]:
        print("  pending {game_date} {away_tricode} @ {home_tricode}".format(**row))
    fence_bits = report["counts"]
    if fence_bits["post_tip_excluded"] or fence_bits["nonpre_state_excluded"]:
        print(
            f"  pre-tip fence: {fence_bits['post_tip_excluded']} post-tip and "
            f"{fence_bits['nonpre_state_excluded']} non-pre-state record(s) excluded"
        )
    metrics = report["model_metrics"]
    if metrics["n_scored"]:
        print(
            f"margin RMSE {metrics['margin_rmse']} MAE {metrics['margin_mae']} | "
            f"log-loss {metrics['win_log_loss']} Brier {metrics['brier']} | "
            f"PIT KS {metrics['pit_ks_distance']} 90% cov {metrics['coverage_90_central']} | "
            f"CRPS {metrics['crps_mean']}"
        )
        sim = report["market_comparison"]["spread_simulation"]
        print(
            f"market rows {report['market_comparison']['n_with_market_margin']} | "
            f"model/market margin RMSE "
            f"{report['market_comparison']['model_margin_rmse_odds_rows']}/"
            f"{report['market_comparison']['market_margin_rmse']} | "
            f"model/close CRPS "
            f"{report['market_comparison']['model_crps_odds_rows']}/"
            f"{report['market_comparison']['market_close_crps']} | "
            f"spread sim {sim['record']} (hit {sim['hit_rate']})"
        )
    else:
        print("0 scoreable forecasts; wrote clean empty report")
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Clock-aware game-state features for RAPM rubberband adjustment.

The production RAPM target is points scored by the offense on a completed
possession.  Rubberband features therefore use the offensive team's margin at
the *start* of that possession and the exact amount of game time remaining.

Every basis in this module is anchored to zero at a tied score.  That keeps the
RAPM intercept/season baseline interpretable and prevents the nuisance block
from assigning a generic scoring constant to all possessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


LEGACY_RUBBERBAND_BINS = (
    ("neg3", None, -15.0),
    ("neg2", -15.0, -10.0),
    ("neg1", -10.0, -5.0),
    ("pos1", 5.0, 10.0),
    ("pos2", 10.0, 15.0),
    ("pos3", 15.0, None),
)

MARGIN_KNOTS = (0.0, 5.0, 10.0, 20.0, 30.0)
MARGIN_CAP = 45.0
REGULATION_PROGRESS_KNOTS = (0.0, 0.25, 0.50, 0.75, 0.8958333333, 0.9583333333)
PRESSURE_KNOTS = (0.0, 0.5, 1.0, 2.0, 4.0)
PRESSURE_CAP = 8.0
SECONDS_PER_COMBINED_POSSESSION = 14.4


@dataclass(frozen=True)
class RubberbandBasis:
    matrix: csr_matrix
    feature_names: tuple[str, ...]


def _clock_strings_to_seconds(values: pd.Series) -> np.ndarray:
    text = values.fillna("").astype(str)
    parts = text.str.extract(r"^\s*(\d{1,2}):(\d{2})\s*$")
    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce")
    parsed = (minutes * 60.0 + seconds).to_numpy(dtype=np.float64)
    return parsed


def compute_pre_possession_clock_state(df: pd.DataFrame) -> pd.DataFrame:
    """Return exact start-of-possession clock variables aligned to ``df``.

    Processed RAPM rows describe completed possessions, so ``time_quarter`` is
    the clock at the end of the row.  The next possession starts at that clock.
    At a new period we use 12:00 in regulation and 5:00 in overtime.
    """

    required = {"period", "time_quarter"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rubberband clock state missing columns: {sorted(missing)}")

    game_col = "game_id" if "game_id" in df.columns else "gameid"
    if game_col not in df.columns:
        raise ValueError("Rubberband clock state requires game_id or gameid.")

    periods = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int).to_numpy()
    game_ids = df[game_col].astype(str).to_numpy()
    end_clock = _clock_strings_to_seconds(df["time_quarter"])
    if np.isnan(end_clock).any():
        bad = int(np.isnan(end_clock).sum())
        raise ValueError(f"Could not parse time_quarter on {bad} possession rows.")

    same_period = np.zeros(len(df), dtype=bool)
    if len(df) > 1:
        same_period[1:] = (game_ids[1:] == game_ids[:-1]) & (periods[1:] == periods[:-1])

    period_length = np.where(periods <= 4, 720.0, 300.0)
    pre_period_clock = period_length.copy()
    if len(df) > 1:
        pre_period_clock[1:] = np.where(same_period[1:], end_clock[:-1], period_length[1:])
    pre_period_clock = np.clip(pre_period_clock, 0.0, period_length)

    regulation = periods <= 4
    seconds_remaining_game = np.where(
        regulation,
        np.maximum(0.0, (4 - periods) * 720.0 + pre_period_clock),
        pre_period_clock,
    )
    regulation_progress = np.where(
        regulation,
        np.clip(1.0 - seconds_remaining_game / 2880.0, 0.0, 1.0),
        1.0,
    )
    overtime_progress = np.where(
        regulation,
        0.0,
        np.clip(1.0 - pre_period_clock / 300.0, 0.0, 1.0),
    )

    return pd.DataFrame(
        {
            "pre_period_clock_seconds": pre_period_clock,
            "seconds_remaining_game": seconds_remaining_game,
            "regulation_progress": regulation_progress,
            "is_overtime": (~regulation).astype(np.int8),
            "overtime_progress": overtime_progress,
        },
        index=df.index,
    )


def _normalized_hinges(values: np.ndarray, knots: tuple[float, ...], cap: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, cap)
    columns = []
    for knot in knots:
        scale = max(cap - knot, 1.0)
        columns.append(np.maximum(clipped - knot, 0.0) / scale)
    return np.column_stack(columns)


def _progress_hinges(progress: np.ndarray) -> np.ndarray:
    progress = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    columns = []
    for knot in REGULATION_PROGRESS_KNOTS:
        scale = max(1.0 - knot, 1e-6)
        columns.append(np.maximum(progress - knot, 0.0) / scale)
    return np.column_stack(columns)


def _tensor_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left[:, :, None] * right[:, None, :]).reshape(len(left), -1)


def _legacy_basis(df: pd.DataFrame) -> RubberbandBasis:
    margin = pd.to_numeric(df["score_diff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    period = np.clip(pd.to_numeric(df["period"], errors="coerce").fillna(1).to_numpy(dtype=int), 1, 4)
    columns = []
    names = []
    for label, low, high in LEGACY_RUBBERBAND_BINS:
        if low is None:
            margin_mask = margin <= float(high)
        elif high is None:
            margin_mask = margin > float(low)
        else:
            margin_mask = (margin > float(low)) & (margin <= float(high))
        for quarter in range(1, 5):
            columns.append((margin_mask & (period == quarter)).astype(np.float64))
            names.append(f"rb_legacy_{label}_period{quarter}")
    matrix = np.column_stack(columns) if columns else np.empty((len(df), 0))
    return RubberbandBasis(csr_matrix(matrix), tuple(names))


def _pressure_basis(df: pd.DataFrame) -> RubberbandBasis:
    margin = pd.to_numeric(df["score_diff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    seconds = pd.to_numeric(df["seconds_remaining_game"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    remaining_combined_possessions = np.maximum(seconds / SECONDS_PER_COMBINED_POSSESSION, 1.0)
    pressure = margin / np.sqrt(remaining_combined_possessions)
    lead = _normalized_hinges(np.maximum(pressure, 0.0), PRESSURE_KNOTS, PRESSURE_CAP)
    trail = _normalized_hinges(np.maximum(-pressure, 0.0), PRESSURE_KNOTS, PRESSURE_CAP)
    matrix = np.column_stack([trail, lead])
    names = tuple(
        [f"rb_pressure_trail_k{k:g}" for k in PRESSURE_KNOTS]
        + [f"rb_pressure_lead_k{k:g}" for k in PRESSURE_KNOTS]
    )
    return RubberbandBasis(csr_matrix(matrix), names)


def _smooth_surface_basis(df: pd.DataFrame) -> RubberbandBasis:
    margin = pd.to_numeric(df["score_diff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    progress = pd.to_numeric(df["regulation_progress"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    is_ot = pd.to_numeric(df["is_overtime"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ot_progress = pd.to_numeric(df["overtime_progress"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    trail_margin = _normalized_hinges(np.maximum(-margin, 0.0), MARGIN_KNOTS, MARGIN_CAP)
    lead_margin = _normalized_hinges(np.maximum(margin, 0.0), MARGIN_KNOTS, MARGIN_CAP)
    time_basis = _progress_hinges(progress)

    trail_reg = _tensor_columns(trail_margin * (1.0 - is_ot[:, None]), time_basis)
    lead_reg = _tensor_columns(lead_margin * (1.0 - is_ot[:, None]), time_basis)

    ot_time = np.column_stack([is_ot, is_ot * ot_progress])
    trail_ot = _tensor_columns(trail_margin, ot_time)
    lead_ot = _tensor_columns(lead_margin, ot_time)
    matrix = np.column_stack([trail_reg, lead_reg, trail_ot, lead_ot])

    names: list[str] = []
    for side in ("trail", "lead"):
        for margin_knot in MARGIN_KNOTS:
            for time_knot in REGULATION_PROGRESS_KNOTS:
                names.append(f"rb_smooth_{side}_m{margin_knot:g}_t{time_knot:.6f}")
    for side in ("trail", "lead"):
        for margin_knot in MARGIN_KNOTS:
            for time_name in ("base", "progress"):
                names.append(f"rb_smooth_ot_{side}_m{margin_knot:g}_{time_name}")

    return RubberbandBasis(csr_matrix(matrix), tuple(names))


def build_rubberband_basis(df: pd.DataFrame, model: str) -> RubberbandBasis:
    """Build one deterministic rubberband nuisance basis.

    Supported models are ``none``, ``legacy``, ``pressure``, and ``smooth``.
    The clock-aware models expect columns returned by
    :func:`compute_pre_possession_clock_state` to be attached to ``df``.
    """

    model = str(model).lower()
    if model == "none":
        return RubberbandBasis(csr_matrix((len(df), 0), dtype=np.float64), tuple())
    if model == "legacy":
        return _legacy_basis(df)
    if model == "pressure":
        return _pressure_basis(df)
    if model == "smooth":
        return _smooth_surface_basis(df)
    raise ValueError(f"Unknown rubberband model: {model}")


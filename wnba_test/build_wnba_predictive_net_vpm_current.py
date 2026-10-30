"""Emit the as-of-current WNBA predictive net-DECOMP research board.

Same lane and frozen dev-selected configuration as
``build_wnba_predictive_net_vpm.py`` (read from the lane's ``report.json``:
Kalman arm off/def ``2000/5000``, no decay, prior scale ``1.0``), but the
likelihood window now includes the live 2026 rows on disk (through the
latest processed game) instead of stopping at the 2025/2026 fold boundary.
By default the prior stays the phase-1 season-start 2026 Kalman prior
(``prior_predictions_kalman_2026.csv``, trained on 2023-2025 labels only).
``--prior-mode fresh`` opts into the phase-2 fresh-blend prior at the frozen
``w*`` from ``fresh_blend/selection.json`` (2024+2025 mid-season-cut
selection; w*=1.00 means today's Kalman states through the same frozen
year-forward models, with the pre-registered debutant/carry-forward coverage
rule) and writes ``*_fresh`` output files so the phase-1 board is preserved.
A labeled 550-day-decay sensitivity column reflects the spent-2026 recency
observation only; the frozen 2027 gate candidate remains the no-decay solve.

Research-only: the live published WNBA VPM (box-prior gross lane) is
untouched and remains canonical. Output stays out of git.

Run from the repo root:
``.venv/bin/python wnba_test/build_wnba_predictive_net_vpm_current.py``
``.venv/bin/python wnba_test/build_wnba_predictive_net_vpm_current.py --prior-mode fresh``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test import build_wnba_predictive_net_vpm as lane  # noqa: E402

OUTPUT_DIR = lane.DEFAULT_OUTPUT_DIR
SENSITIVITY_HALF_LIFE_DAYS = 550.0


def build_current_fold(ordered: pd.DataFrame, transformed: pd.DataFrame):
    """Full-window PriorFold over every row on disk (2022 through live 2026).

    Mirrors ``lane.build_fold`` without a validation split: the validation
    fields are inert one-row dummies and are never scored.
    """
    prepared, target_names = lane.build_component_targets(ordered, "net")
    mapping = lane._player_mapping(transformed)
    design, _ = lane.build_design_matrix(transformed, mapping)
    off_indices, def_indices, off_players, def_players = lane._mapping_sides(mapping)
    cutoff = pd.Timestamp(pd.to_datetime(ordered["game_date"]).max()) + pd.Timedelta(
        days=1
    )
    fold = lane.PriorFold(
        label="current",
        validation_year=0,
        train_years=sorted(ordered["season_end_year"].astype(int).unique().tolist()),
        cutoff_date=cutoff,
        train_dates=ordered["game_date"].to_numpy(),
        train_x=design,
        train_targets=prepared[target_names].to_numpy(float),
        validation_x=design[:1],
        validation_actual=np.zeros(1),
        validation_games=np.asarray(["_unused"]),
        mapping=mapping,
        off_indices=off_indices,
        def_indices=def_indices,
        off_players=off_players,
        def_players=def_players,
    )
    return fold, target_names


def build_fresh_prior(report, ordered, transformed, short_names):
    """Frozen-w* fresh-blend prior: today's Kalman states through the frozen
    year-forward models, phase-2 protocol exactly (selection.json is the
    authority; the season-start reproduction and same-model checks re-run)."""
    from wnba_test import research_wnba_fresh_blend_phase2 as p2

    if not p2.SELECTION_PATH.exists():
        raise SystemExit(
            "--prior-mode fresh requires fresh_blend/selection.json; run "
            "research_wnba_fresh_blend_phase2.py --phase select first"
        )
    selection = json.loads(p2.SELECTION_PATH.read_text())
    w_star = float(selection["frozen_w_star"])
    labels_by_year, label_targets = p2.solve_labels(
        ordered, transformed, (2023, 2024, 2025)
    )
    label_short = [name.replace("target_", "") for name in label_targets]
    if label_short != short_names:
        raise ValueError("label target order diverged from fold target order")
    starts = p2.season_boundaries(ordered, (2023, 2024, 2025, 2026))
    p2.assert_lane_boundaries(starts, report)
    start_snapshots, _ = lane.build_kalman_snapshots(
        starts, lane.KALMAN_BOX_PATH, lane.SKILLS_PATH
    )
    mu_start, start_meta = p2.fit_season_prior(
        2026,
        start_snapshots[2026],
        start_snapshots,
        labels_by_year,
        short_names,
        arm="kalman_start",
    )
    reproduction = p2.check_start_reproduction(2026, mu_start)
    boundary = pd.Timestamp(
        pd.to_datetime(ordered["game_date"]).max()
    ) + pd.Timedelta(days=1)
    fresh_snapshots, fresh_audit = lane.build_kalman_snapshots(
        {2026: boundary}, lane.KALMAN_BOX_PATH, lane.SKILLS_PATH
    )
    mu_fresh, fresh_meta = p2.fit_season_prior(
        2026,
        fresh_snapshots[2026],
        start_snapshots,
        labels_by_year,
        short_names,
        arm="kalman_fresh",
    )
    p2.assert_same_models(start_meta, fresh_meta)
    b0, b1, groups = p2.build_blend_endpoints(mu_start, mu_fresh)
    prior_wide = b0 + w_star * (b1 - b0)
    desc = (
        f"fresh-blend Kalman prior at frozen w*={w_star:.2f} "
        f"(phase-2 selection; Kalman states strictly before {boundary.date()})"
    )
    extras = {
        "frozen_w_star": w_star,
        "fresh_state_boundary": str(boundary.date()),
        "coverage_groups": groups,
        "start_reproduction_max_abs_diff": reproduction,
        "fresh_state_audit": fresh_audit,
    }
    logging.info(
        "Fresh prior at w*=%.2f: %s players (%s both, %s debutant fresh-only, "
        "%s start-only carry-forward)",
        w_star,
        groups["n_union"],
        groups["n_both_union"],
        groups["n_fresh_only_union"],
        groups["n_start_only_union"],
    )
    return prior_wide, desc, extras


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-mode",
        choices=["season-start", "fresh"],
        default="season-start",
        help=(
            "season-start = phase-1 prior (default); fresh = phase-2 frozen-w* "
            "fresh-blend prior, written to *_fresh output files"
        ),
    )
    args = parser.parse_args()
    report = json.loads((OUTPUT_DIR / "report.json").read_text())
    selected = report["selected"]["kalman"]
    off_alpha = float(selected["off_alpha"])
    def_alpha = float(selected["def_alpha"])
    prior_scale = float(selected["prior_scale"])
    if selected["half_life_days"] is not None:
        raise ValueError(
            "selected config gained a decay term; update this runner deliberately"
        )

    years = list(range(2022, 2027))
    rows, sources = lane.load_target_rows(
        years,
        target_family="net",
        processed_dir=lane.DEFAULT_PROCESSED_DIR,
        sq_dir=lane.DEFAULT_SQ_DIR,
        include_playoffs=True,
    )
    if int(rows["season_end_year"].max()) > 2026:
        raise ValueError("2027 is embargoed everywhere in this lane")
    ordered, transformed = lane._load_or_build_ordered(
        OUTPUT_DIR, sources, rows, use_cache=True
    )
    as_of = pd.Timestamp(pd.to_datetime(ordered["game_date"]).max()).date()
    n_2026 = int(ordered["season_end_year"].eq(2026).sum())
    logging.info(
        "Current window: %s rows (%s live 2026 rows) through %s",
        len(ordered),
        n_2026,
        as_of,
    )

    fold, target_names = build_current_fold(ordered, transformed)
    short_names = [name.replace("target_", "") for name in target_names]

    if args.prior_mode == "fresh":
        prior_wide, prior_desc, prior_extras = build_fresh_prior(
            report, ordered, transformed, short_names
        )
    else:
        prior_wide = pd.read_csv(
            OUTPUT_DIR / "prior_predictions_kalman_2026.csv",
            dtype={"player_id": str},
        ).set_index("player_id")
        prior_desc = "season-start 2026 Kalman (phase-1 protocol; no fresh blend)"
        prior_extras = None
    prior_matrix, matched = lane.build_prior_matrix(fold, prior_wide, short_names)
    logging.info(
        "%s prior matched %s of %s mapped players",
        args.prior_mode,
        matched,
        len(fold.off_players),
    )

    coeff, _, centered_prior = lane.solve_fold(
        fold, off_alpha, def_alpha, None, prior_matrix, prior_scale
    )
    pure_coeff, _, _ = lane.solve_fold(fold, off_alpha, def_alpha, None)
    decay_coeff, _, _ = lane.solve_fold(
        fold, off_alpha, def_alpha, SENSITIVITY_HALF_LIFE_DAYS, prior_matrix, prior_scale
    )

    name_map = lane._load_name_map()
    exposure = fold.raw_exposure()
    poss_2026 = None
    mask_2026 = ordered["season_end_year"].eq(2026).to_numpy()
    if mask_2026.any():
        poss_2026 = np.asarray(fold.train_x[mask_2026].sum(axis=0)).reshape(-1)

    board_rows = []
    component_rows = []
    for off_row, player in zip(fold.off_indices, fold.off_players):
        def_row = fold.mapping[f"{player}_def"]
        off_total = float(coeff[off_row, :].sum()) * 100.0
        def_total = float(coeff[def_row, :].sum()) * 100.0
        comp = {
            "player_id": player,
            "player_name": name_map.get(player, f"ID_{player}"),
            "off_poss_2026": float(poss_2026[off_row]) if poss_2026 is not None else 0.0,
        }
        for t, name in enumerate(short_names):
            comp[f"off_{name}"] = float(coeff[off_row, t]) * 100.0
            comp[f"def_{name}_raw"] = float(coeff[def_row, t]) * 100.0
            comp[f"net_{name}"] = comp[f"off_{name}"] - comp[f"def_{name}_raw"]
        comp["off100"] = off_total
        comp["def100_raw"] = def_total
        comp["net100"] = off_total - def_total
        if abs(sum(comp[f"off_{n}"] for n in short_names) - off_total) > 1e-9:
            raise ValueError(f"component off sum does not close for {player}")
        if abs(sum(comp[f"def_{n}_raw"] for n in short_names) - def_total) > 1e-9:
            raise ValueError(f"component def sum does not close for {player}")
        component_rows.append(comp)
        board_rows.append({
            "player_id": player,
            "player_name": name_map.get(player, f"ID_{player}"),
            "off_poss_window": float(exposure[off_row]),
            "def_poss_window": float(exposure[def_row]),
            "off_poss_2026": float(poss_2026[off_row]) if poss_2026 is not None else 0.0,
            "off100": off_total,
            "def100_raw": def_total,
            "net100": off_total - def_total,
            "net100_decay550_sensitivity": (
                float(decay_coeff[off_row, :].sum())
                - float(decay_coeff[def_row, :].sum())
            ) * 100.0,
            "prior_off100": float((centered_prior[off_row, :]).sum()) * 100.0,
            "prior_def100_raw": float((centered_prior[def_row, :]).sum()) * 100.0,
            "pure_off100": float(pure_coeff[off_row, :].sum()) * 100.0,
            "pure_def100_raw": float(pure_coeff[def_row, :].sum()) * 100.0,
            "has_prior": int(np.abs(prior_matrix[off_row, :]).sum() > 0),
        })

    suffix = "_fresh" if args.prior_mode == "fresh" else ""
    board = pd.DataFrame(board_rows).sort_values("net100", ascending=False)
    out_path = OUTPUT_DIR / f"research_board_current{suffix}.csv"
    board.to_csv(out_path, index=False, float_format="%.4f")
    components = pd.DataFrame(component_rows).sort_values("net100", ascending=False)
    components_path = OUTPUT_DIR / f"research_board_current{suffix}_components.csv"
    components.to_csv(components_path, index=False, float_format="%.4f")

    meta = {
        "as_of_latest_game": str(as_of),
        "window_seasons": fold.train_years,
        "rows_total": int(len(ordered)),
        "rows_2026": n_2026,
        "config": {
            "off_alpha": off_alpha,
            "def_alpha": def_alpha,
            "half_life_days": None,
            "prior_scale": prior_scale,
            "prior": prior_desc,
        },
        "decay_sensitivity_half_life_days": SENSITIVITY_HALF_LIFE_DAYS,
        "matched_prior_players": int(matched),
        "note": (
            "research-only current reading of the dev-selected Kalman-prior "
            "lane; not the published WNBA VPM, not a 2027 gate result"
        ),
    }
    if prior_extras is not None:
        meta["fresh_prior"] = prior_extras
    (OUTPUT_DIR / f"research_board_current{suffix}_meta.json").write_text(
        json.dumps(meta, indent=2)
    )

    active = board[board["off_poss_2026"] >= 500]
    print(f"\nTop 15 (>=500 offensive 2026 possessions), as of {as_of}:")
    cols = ["player_name", "net100", "off100", "def100_raw", "off_poss_2026",
            "net100_decay550_sensitivity", "has_prior"]
    print(active.head(15)[cols].to_string(index=False))
    print(
        f"\nwrote {out_path}, research_board_current{suffix}_components.csv, "
        f"and research_board_current{suffix}_meta.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

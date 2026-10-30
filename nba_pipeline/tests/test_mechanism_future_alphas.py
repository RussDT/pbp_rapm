from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_mechanism_future_alphas.py"
SPEC = importlib.util.spec_from_file_location("mechanism_future_alphas", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_future_blocks_are_complete_nonoverlapping_and_strictly_later() -> None:
    blocks = MODULE.nonoverlapping_blocks(list(range(1997, 2027)), 5, 2, 4)
    assert blocks == [
        ([2014, 2015, 2016, 2017, 2018], [2019, 2020]),
        ([2016, 2017, 2018, 2019, 2020], [2021, 2022]),
        ([2018, 2019, 2020, 2021, 2022], [2023, 2024]),
        ([2020, 2021, 2022, 2023, 2024], [2025, 2026]),
    ]
    future_years = [year for _, future in blocks for year in future]
    assert len(future_years) == len(set(future_years))
    assert all(max(train) < min(future) for train, future in blocks)


def test_missing_year_breaks_a_block_instead_of_silent_backfill() -> None:
    years = [year for year in range(2014, 2027) if year != 2021]
    blocks = MODULE.nonoverlapping_blocks(years, 5, 2, None)
    assert all(2021 not in train + future for train, future in blocks)


def test_paired_one_se_chooses_most_shrunk_eligible_pair() -> None:
    rows = []
    losses = {
        (100, 100): [1.000, 1.000, 1.000],
        (1000, 1000): [1.002, 0.999, 1.001],
        (3000, 3000): [1.020, 1.018, 1.019],
    }
    for (off_alpha, def_alpha), block_losses in losses.items():
        for block, loss in enumerate(block_losses, start=1):
            rows.append(
                {
                    "off_alpha": off_alpha,
                    "def_alpha": def_alpha,
                    "candidate": f"{off_alpha}_{def_alpha}",
                    "block": block,
                    "future_mse": loss,
                }
            )
    selected, report = MODULE.paired_one_se(pd.DataFrame(rows))
    assert selected == (1000, 1000)
    assert not report.set_index("off_alpha").loc[3000, "within_paired_one_se"]


def test_full_alpha_grid_checks_offense_and_defense_separately() -> None:
    mechanism = MODULE.Mechanism("example", "EXAMPLE", "EXAMPLE", 2000, 5000)
    pairs = MODULE.candidate_pairs(mechanism, full_alpha_grid=True)
    pair_keys = {(off_alpha, def_alpha) for off_alpha, def_alpha, _ in pairs}
    assert len(pair_keys) == 28
    assert (100, 10000) in pair_keys
    assert (10000, 100) in pair_keys
    assert (2000, 5000) in pair_keys
    assert (2000, 3000) in pair_keys
    assert (3000, 5000) in pair_keys


def test_current_pair_label_survives_when_current_is_3000_3000() -> None:
    mechanism = MODULE.Mechanism("example", "EXAMPLE", "EXAMPLE", 3000, 3000)
    labels = {(off, defense): label for off, defense, label in MODULE.candidate_pairs(mechanism, True)}
    assert labels[(3000, 3000)] == "current_game_cv"

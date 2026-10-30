import numpy as np
import pandas as pd
import pytest

from nba_pipeline.scripts.analyze_zts_stability import METRICS, evaluate, make_pairs


def test_pairs_require_consecutive_years_and_preserve_player_identity():
    panel = pd.DataFrame({
        "nba_id": [1, 1, 1, 2, 2], "year": [2020, 2021, 2023, 2020, 2021],
        "role": [1, 2, 3, 4, 5], "single_team": [True]*5,
        "team": ["BOS", "LAL", "LAL", "MIA", "MIA"], "TSA": [300]*5,
    })
    pairs = make_pairs(panel)
    assert list(zip(pairs.nba_id, pairs.year_a, pairs.year_b)) == [(1, 2020, 2021), (2, 2020, 2021)]
    assert pairs.changed_team.tolist() == [True, False]
    assert pairs.role_change.tolist() == [1, 1]


def test_repeatability_does_not_confuse_correlation_with_absolute_stability():
    # Scaling preserves ordering perfectly but doubles year-to-year changes.
    x = np.linspace(-4, 4, 40)
    p = pd.DataFrame({"nba_id": np.repeat(np.arange(20), 2), "target_year": [2020, 2021]*20,
                      "harmonic_tsa": np.ones(40), "role_change": x})
    for metric in METRICS:
        p[metric+"_a"] = x
        p[metric+"_b"] = 2*x
    p["zts_a"], p["zts_b"] = 2*x, 4*x
    rows, contrasts, decomposition = evaluate(p, "synthetic", reps=100)
    r = {row["metric"]: row for row in rows}
    assert r["zts"]["pearson"] == pytest.approx(r["rts"]["pearson"])
    assert r["zts"]["mae_pp"] == pytest.approx(2*r["rts"]["mae_pp"])
    assert decomposition[0]["closure_max"] < 1e-10
    assert rows[0]["players"] == 20
    assert len(contrasts) == 9


def test_identical_metrics_have_exact_zero_paired_uncertainty():
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(2, 40))
    p = pd.DataFrame({"nba_id": np.repeat(np.arange(20), 2), "target_year": [2020, 2021]*20,
                      "harmonic_tsa": np.ones(40), "role_change": np.zeros(40)})
    for metric in METRICS:
        p[metric+"_a"], p[metric+"_b"] = x, y
    _, contrasts, decomposition = evaluate(p, "identical", reps=100)
    for row in contrasts:
        assert row["zts_minus_baseline"] == pytest.approx(0)
        assert row["lo"] == pytest.approx(0)
        assert row["hi"] == pytest.approx(0)
    assert decomposition[0]["closure_max"] < 1e-10

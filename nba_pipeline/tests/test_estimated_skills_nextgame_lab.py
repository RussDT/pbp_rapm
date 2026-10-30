"""Engine-level checks for the next-game estimated-skills lab.

The lab engine must be the v2 OU filter exactly when every lab option is off,
and the six OU params must be jointly scale-invariant in (s_mu, s_d, q_off, r)
for predictions (the flat DE direction that motivates the ``fixr`` variant).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2  # noqa: E402
from research_skills_nextgame_lab import run_filter_lab  # noqa: E402


def _panel(seed=0, n_players=12, n_games=40):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        talent = 0.55 + 0.08 * rng.standard_normal()
        for k in range(n_games):
            n = int(rng.integers(2, 12))
            made = rng.binomial(n, min(max(talent, 0.05), 0.95))
            rows.append(dict(player_idx=p, step_k=k, exposure=n, obs_rate=made / n,
                             days_gap=float(rng.integers(1, 5)) if k else 0.0,
                             season_cross=float(k == 20)))
    return pd.DataFrame(rows)


PARAMS = [900.0, 0.004, 12.0, 0.002, 0.001, 0.24]


def test_lab_engine_equals_v2_when_options_off():
    df = _panel(); lay = LayoutV2(df); obs = df["obs_rate"].to_numpy()
    tr = 0.01 * np.sin(np.arange(len(df))); ls = np.full(len(df), 0.003)
    mu0 = np.full(len(df), 0.52)
    a = run_filter_v2(lay, obs, PARAMS, 0.55, mu0=mu0, transient=tr, level_shift=ls)
    b = run_filter_lab(lay, obs, PARAMS, 0.55, mu0=mu0, transient=tr, level_shift=ls)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-14)


def test_variance_params_are_jointly_scale_invariant_for_predictions():
    df = _panel(1); lay = LayoutV2(df); obs = df["obs_rate"].to_numpy()
    base = run_filter_lab(lay, obs, PARAMS, 0.55)[0]
    for c in (0.1, 7.5, 40.0):
        p = list(PARAMS); p[1] *= c; p[3] *= c; p[4] *= c; p[5] *= c
        np.testing.assert_allclose(run_filter_lab(lay, obs, p, 0.55)[0], base, rtol=1e-10)


def test_lab_options_change_predictions_and_stay_finite():
    df = _panel(2); lay = LayoutV2(df); obs = df["obs_rate"].to_numpy()
    base = run_filter_lab(lay, obs, PARAMS, 0.55)[0]
    p = list(PARAMS); p[5] = 1.0
    binom = run_filter_lab(lay, obs, p, 0.55, noise="binom")[0]
    clip = run_filter_lab(lay, obs, PARAMS, 0.55, clip_k=1.5)[0]
    mult = run_filter_lab(lay, obs, PARAMS, 0.55, mult=np.full(len(df), 0.05))[0]
    for arr in (binom, clip, mult):
        assert np.isfinite(arr).all()
        assert not np.allclose(arr, base)
    # multiplicative context scales the level: rows with a prior obs move up
    hp = df["step_k"].to_numpy() > 0
    assert np.all(mult[hp] > base[hp])
    # clipping at a huge k is a no-op
    np.testing.assert_allclose(run_filter_lab(lay, obs, PARAMS, 0.55, clip_k=1e6)[0], base)

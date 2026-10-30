import json

import numpy as np
import pandas as pd

from wnba_test.build_wnba_box_priors import (
    ADDITIVE_PREFIXES,
    TARGET_PREFIXES,
    _derive_additive_priors,
    _smoothed_rate,
    _smoothed_share,
    audit_box_exposure_conflicts,
    classify_box_prior_sources,
    load_frozen_spec,
)


def _prior_frame(off, defense):
    return pd.DataFrame(
        {
            "player_id": ["1", "2"],
            "player_name": ["A", "B"],
            "off": off,
            "def": defense,
        }
    )


def test_stabilized_box_features_are_finite_at_zero_exposure():
    rate = _smoothed_rate(
        pd.Series([0.0, 4.0]),
        pd.Series([0.0, 100.0]),
        prior_exposure=250.0,
    )
    share = _smoothed_share(
        pd.Series([0.0, 3.0]),
        pd.Series([0.0, 10.0]),
        prior_attempts=75.0,
    )

    assert np.isfinite(rate).all()
    assert np.isfinite(share).all()
    assert rate.iloc[0] == rate.iloc[1]
    assert share.iloc[0] == share.iloc[1]


def test_box_component_priors_preserve_derived_and_total_identities():
    base = {}
    for index, prefix in enumerate(TARGET_PREFIXES, start=1):
        base[prefix] = _prior_frame(
            [index / 10.0, -index / 20.0],
            [-index / 30.0, index / 40.0],
        )

    priors = _derive_additive_priors(base)

    for side in ("off", "def"):
        np.testing.assert_allclose(
            priors["ALT_FT"][side],
            priors["ALT_FT_FREQ"][side] + priors["ALT_FT_SEVERITY"][side],
        )
        np.testing.assert_allclose(
            priors["ALT_TOV_VALUE"][side],
            priors["ALT_BADPASS_TOV_VALUE"][side]
            + priors["ALT_SCORING_TOV_VALUE"][side],
        )
        expected_total = sum(priors[prefix][side] for prefix in ADDITIVE_PREFIXES)
        np.testing.assert_allclose(priors["RAPM"][side], expected_total)

    # The helper must not mutate the fitted base-component dictionary.
    assert "ALT_FT_SEVERITY" not in base
    assert "ALT_SCORING_TOV_VALUE" not in base
    assert "RAPM" not in base


def test_current_build_reuses_only_a_pre_2026_frozen_selection(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "selected_spec": {
                    "off_alpha": 5000.0,
                    "def_alpha": 40000.0,
                    "prior_scale": 1.5,
                    "box_target_off_alpha": 750.0,
                    "box_target_def_alpha": 1000.0,
                    "selection_folds": [2024, 2025],
                }
            }
        )
    )

    spec = load_frozen_spec(report_path)

    assert spec["def_alpha"] == 40000.0
    assert spec["direct_tov_def_alpha"] == 10000.0
    assert spec["selection_folds"] == [2024, 2025]

    report_path.write_text(
        json.dumps(
            {
                "selected_spec": {
                    **spec,
                    "selection_folds": [2024, 2025, 2026],
                }
            }
        )
    )
    try:
        load_frozen_spec(report_path)
    except RuntimeError as exc:
        assert "2026" in str(exc)
    else:
        raise AssertionError("a spec selected with 2026 was accepted as frozen")


def test_current_box_source_uses_possessions_when_minutes_are_corrupt():
    current = pd.DataFrame(
        {
            "player_id": ["1", "2"],
            "minutes": [-3.0, 20.0],
            "off_poss": [100.0, 0.0],
            "def_poss": [102.0, 0.0],
        }
    )
    previous = pd.DataFrame({"player_id": ["1", "2"]})

    sources = classify_box_prior_sources(current, previous, ["1", "2", "3"])

    assert sources.to_dict() == {
        "1": "current_single_season",
        "2": "previous_season_fallback",
        "3": "imputed_zero",
    }


def test_box_exposure_audit_flags_nonpositive_minutes_with_possessions():
    current = pd.DataFrame(
        {
            "player_id": ["1", "2"],
            "player_name": ["Corrupt Minutes", "No Exposure"],
            "minutes": [-3.0, 0.0],
            "off_poss": [100.0, 0.0],
            "def_poss": [102.0, 0.0],
        }
    )

    audit = audit_box_exposure_conflicts(current)

    assert audit == {
        "count": 1,
        "players": [
            {
                "player_id": "1",
                "player_name": "Corrupt Minutes",
                "minutes": -3.0,
                "off_poss": 100.0,
                "def_poss": 102.0,
            }
        ],
    }

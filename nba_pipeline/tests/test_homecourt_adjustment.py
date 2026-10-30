import numpy as np
import pandas as pd

import pytest

from nba_pipeline.scripts.research_homecourt_adjustment import (
    centered_home_columns,
    validate_home_label_balance,
)


def test_centered_home_columns_use_training_share_and_preserve_contrast():
    train = pd.DataFrame(
        {
            "home_poss": [1, 1, 0, 1, 0],
            "home_orientation_resolved": [1, 1, 1, 1, 1],
        }
    )
    test = pd.DataFrame(
        {"home_poss": [0, 1], "home_orientation_resolved": [1, 1]}
    )
    train_column, test_column, share = centered_home_columns(train, test)

    assert share == 0.6
    assert abs(train_column.mean()) < 1e-12
    np.testing.assert_allclose(test_column, [-0.6, 0.4])
    assert test_column[1] - test_column[0] == 1.0


def test_home_label_balance_rejects_nonvenue_orientation():
    valid = pd.DataFrame(
        {
            "home_poss": [0, 1] * 50,
            "home_orientation_resolved": [1] * 100,
        }
    )
    invalid = pd.DataFrame(
        {
            "home_poss": [0] * 90 + [1] * 10,
            "home_orientation_resolved": [1] * 100,
        }
    )

    assert validate_home_label_balance(valid, 2026) == 0.5
    with pytest.raises(ValueError, match="not trustworthy"):
        validate_home_label_balance(invalid, 2017)


def test_unresolved_home_rows_are_neutral_and_not_used_for_centering():
    train = pd.DataFrame(
        {
            "home_poss": [1, 0, 0, 1],
            "home_orientation_resolved": [1, 1, 0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "home_poss": [1, 0, 1],
            "home_orientation_resolved": [1, 1, 0],
        }
    )
    train_column, test_column, share = centered_home_columns(train, test)

    assert share == 0.5
    np.testing.assert_allclose(train_column, [0.5, -0.5, 0.0, 0.0])
    np.testing.assert_allclose(test_column, [0.5, -0.5, 0.0])


def test_home_label_balance_rejects_low_resolution():
    frame = pd.DataFrame(
        {
            "home_poss": [0, 1] * 50,
            "home_orientation_resolved": [1] * 97 + [0] * 3,
        }
    )

    with pytest.raises(ValueError, match="resolves venue orientation"):
        validate_home_label_balance(frame, 1997)

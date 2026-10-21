"""Annual model availability fences for the optional OU history inputs."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from historical_features_ou import legal_vintage


def test_no_later_ou_model_is_borrowed_for_early_history():
    assert legal_vintage('2014-01-01') is None
    assert legal_vintage('2015-09-30') is None
    assert legal_vintage('2015-10-01')==2015


def test_delayed_2020_preseason_uses_existing_2019_model_until_december():
    assert legal_vintage('2020-10-01')==2019
    assert legal_vintage('2020-11-30')==2019
    assert legal_vintage('2020-12-01')==2020


def test_2026_carries_available_2025_model_without_fabricating_refit():
    assert legal_vintage('2026-09-01')==2025
    assert legal_vintage('2026-10-01')==2025

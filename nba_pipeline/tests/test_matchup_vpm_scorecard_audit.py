"""Regression coverage for independent veteran-override audit comparisons."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec = importlib.util.spec_from_file_location('score_audit',Path(__file__).parents[1]/'scripts/audit_matchup_vpm_scorecard.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_veteran_vector_compares_to_scalar_replacement_without_shape_false_alarm():
    assert m.comparison_gap(np.full(28936,-2.),-2.,'veteran') == 0
    assert m.comparison_gap([-2.,-1.],-2.,'veteran') == 1

def test_missing_veteran_is_not_hidden_by_scalar_reference():
    with pytest.raises(ValueError,match='missingness'):
        m.comparison_gap([-2.,np.nan],-2.,'veteran')
    assert m.comparison_gap([1.,np.nan],[1.,np.nan],'scores') == 0

def test_nonscalar_shape_mismatch_is_not_silently_broadcast():
    with pytest.raises(ValueError,match='shapes differ'):
        m.comparison_gap(np.ones((3,1)),np.ones(3),'matrix')
    with pytest.raises(ValueError,match='infinity'):
        m.comparison_gap([np.inf],[np.inf],'scores')

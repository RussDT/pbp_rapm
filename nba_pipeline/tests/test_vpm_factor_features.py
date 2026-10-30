import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from export_vpm_factor_features import select_snapshot

def test_factor_snapshot_excludes_future_and_allows_exact_exclusive_cut():
 snapshots=[('2020-10-01','old',{}),('2021-01-01','future',{})]
 assert select_snapshot(snapshots,pd.Timestamp('2020-12-01'))[1]=='old'
 assert select_snapshot(snapshots,pd.Timestamp('2021-01-01'))[1]=='future'
 with pytest.raises(ValueError):select_snapshot(snapshots,'2019-10-01')

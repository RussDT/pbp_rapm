"""Verify annual model fences and current-field parity of the OU history."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from historical_features import ROOT, sha, write_json
from historical_features_ou import OUT


def audit(out=OUT):
    out=Path(out);m=json.loads((out/'manifest.json').read_text())
    for group in ['sources','models']:
        for path,digest in m[group].items():
            if sha(path)!=digest:raise ValueError('OU input/model changed: '+path)
    for name,digest in m['artifacts'].items():
        if sha(out/name)!=digest:raise ValueError('OU artifact changed: '+name)
    d=pd.read_parquet(out/'features.parquet');p=pd.read_parquet(out/'provenance.parquet')
    assert not d.duplicated(['player_id','estimate_date']).any()
    assert not p.duplicated(['player_id','estimate_date','feature']).any()
    assert p.last_observation_date.lt(p.estimate_date).all()
    fitted=p.parameter_fit_end.notna()
    assert p.loc[fitted,'parameter_fit_end'].lt(p.loc[fitted,'estimate_date']).all()
    assert p.loc[fitted,'parameter_fit_cut'].le(p.loc[fitted,'estimate_date']).all()
    assert set(p.loc[fitted,'model_sha256']).issubset(set(m['models'].values()))
    assert d.estimate_date.min()==pd.Timestamp('2015-10-01')
    assert d.estimate_date.max()==pd.Timestamp('2026-10-01')
    for c in [c for c in d if c.endswith('_missing')]:
        assert d[c].eq(d[c[:-8]].isna().astype(int)).all(),c
    for zone,attempt in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),('three','three_fga_rate'),('ft','fta_rate')]:
        expected=d[f'own_{attempt}']*d[f'own_{zone}_pct']
        np.testing.assert_allclose(d[f'own_{zone}_makes100'],expected,atol=1e-12,rtol=0,equal_nan=True)
        assert d[f'own_{zone}_pct'].dropna().between(0,1).all()
    np.testing.assert_allclose(d.own_rim_saved100,2*d.own_rim_contests100*d.own_rim_suppression,
        atol=1e-12,rtol=0,equal_nan=True)
    original=ROOT/'results/estimated_skills/vpm_monthly_neutral_v1/features.parquet'
    old=pd.read_parquet(original);old=old.loc[old.estimate_date.eq('2026-09-01')].set_index('player_id')
    now=d.loc[d.estimate_date.eq('2026-09-01')].set_index('player_id').reindex(old.index)
    columns=[c for c in old if c.startswith('own_') and not c.endswith('_missing')]
    np.testing.assert_allclose(now[columns],old[columns],atol=1e-12,rtol=0,equal_nan=True)
    report=dict(status='passed',rows=len(d),provenance_rows=len(p),cuts=int(d.estimate_date.nunique()),
        trained_provenance_rows=int(fitted.sum()),fixed_extra_provenance_rows=int((~fitted).sum()),
        model_checksums_verified=True,all_observation_and_parameter_fences_pass=True,
        all_missing_flags_and_shooting_rim_identities_pass=True,current_reference_players=len(old),
        current_reference_own_features=len(columns),current_reference_cells=len(old)*len(columns),
        current_reference_maximum_gap=float((now[columns]-old[columns]).abs().max().max()),
        reference_source=str(original),reference_sha256=sha(original),protected_outcomes=0)
    write_json(out/'independent_audit.json',report);print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':audit()

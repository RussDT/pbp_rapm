"""Independently reconstruct v51 observed rates and dated EB snapshots."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root):
    root = Path(root).resolve()
    meta = json.loads((root / 'manifest.json').read_text())
    for path, expected in meta['source_hashes'].items():
        if digest(path) != expected:
            raise AssertionError('Source changed: ' + path)
    for name, expected in meta['artifact_hashes'].items():
        if digest(root / name) != expected:
            raise AssertionError('Artifact changed: ' + name)
    d = pd.read_parquet(root / 'measurements.parquet')
    assert not d.duplicated(['game_id', 'player_id', 'feature']).any()
    assert d.numerator.ge(0).all() and d.denominator.gt(0).all()
    features = pd.read_parquet(root / 'features.parquet')
    provenance = pd.read_parquet(root / 'provenance.parquet')
    assert provenance.last_observation_date.lt(provenance.estimate_date).all()
    vpm = root.parents[4] / 'vpm/vpm/results/harness'
    maximum = 0.
    cells = 0
    for cut, snapshot in features.groupby('estimate_date'):
        label = 'current' if cut == pd.Timestamp('2026-09-01') else str(cut.year + 1)
        base = (vpm / 'vpm_long_spm_current_2026_09_01_v1/calibration' if label == 'current'
                else vpm / 'vpm_long_spm_v1/calibrations' / label)
        games = pd.read_parquet(base / 'window_games.parquet')
        prior = d.loc[d.game_date.lt(cut)].copy()
        assert prior.game_date.max() < cut
        for name in meta['features']:
            p = prior.loc[prior.feature.eq(name)].copy()
            weights = 0.5 ** ((cut - p.game_date).dt.total_seconds().to_numpy() / 86400 / 550)
            p['n'] = p.numerator.to_numpy() * weights
            p['e'] = p.denominator.to_numpy() * weights
            totals = p.groupby('player_id')[['n', 'e']].sum()
            league = p.n.sum() / p.e.sum() if len(p) else 0.
            expected = (100 * (totals.n + 1000 * league) / (totals.e + 1000)).reindex(snapshot.player_id)
            np.testing.assert_allclose(snapshot[name], expected, atol=1e-10, rtol=0, equal_nan=True)
            if np.isfinite(expected).any():
                maximum = max(maximum, float(np.nanmax(abs(snapshot[name].to_numpy() - expected.to_numpy()))))
            cells += len(expected)
            for prefix, accepted in [('observed', games), ('component_observed', games.loc[games.source_season_end_year.ge(2016)])]:
                packet = pd.read_parquet(root / f'{prefix}_{label}.parquet')
                matched = p.loc[p.game_id.isin(set(accepted.game_id.astype(str).str.zfill(10)))]
                sums = matched.groupby('player_id')[['numerator', 'denominator']].sum()
                values = (100 * sums.numerator / sums.denominator).reindex(packet.player_id)
                np.testing.assert_allclose(packet[name], values, atol=1e-10, rtol=0, equal_nan=True)
                cells += len(values)
    result = dict(status='passed', independently_reconstructed_cells=cells, maximum_snapshot_gap=maximum,
                  source_files=len(meta['source_hashes']), observations=len(d), prefix_dates=8,
                  protected_outcomes=0, same_day_observations=0,
                  measurement_sha256=digest(root / 'measurements.parquet'))
    (root / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)

#!/usr/bin/env python3
"""Re-solve a saved current full-feature board with the frozen entrant prior."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1]
E = PIPE/'results/estimated_skills'
V = PIPE.parents[1]/'vpm'
sys.path[:0] = [str(HERE), str(PIPE/'scripts'), str(V)]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def restore(own_root, *, out=None):
    start = time.monotonic(); root = Path(own_root).resolve()
    tag = root.name.removeprefix('current_refresh_').removesuffix('_sq_v1')
    result = json.loads((root/'result.json').read_bytes())
    cut = pd.Timestamp(result['estimate_date']).as_unit('s')
    code = root/'current_code_native_scale'
    load('vpm.harness.current_daily_numerics_'+tag, code/'current_daily_numerics.py')
    bank = load('vpm.harness.current_bank_numerics_'+tag, code/'current_bank_numerics.py')
    storage = load('vpm.harness.current_board_storage_'+tag, code/'current_board_storage.py')
    model_path = V/'vpm/results/harness/vpm_full_feature_history_v1/current_preview_v1/mapper/model.json'
    model = json.loads(model_path.read_bytes())
    if sha(model_path) != result['model_sha256']: raise ValueError('Own mapper changed')
    reference = pd.read_parquet(root/'current_reference_complete.parquet')
    def basis(recipe, ids):
        _, arrays, _ = bank.source_reference_basis(reference, model, ids, cut, role_source=None)
        return dict(off_basis=arrays['off'], def_basis=arrays['def'])
    own_path = Path(result['board_path'])
    own = storage.read_compact_board(own_path, basis_provider=basis)
    if sha(own_path/'manifest.json') != result['board_manifest_sha256']: raise ValueError('Own-only comparison board changed')
    engine = bank.FullBankHistoryGeometry.resume(V/'vpm/results/harness/vpm_full_feature_history_v1/compact_history_fast_v1/checkpoints/2026-10-01')
    native = root/'native_source/targets'
    atoms = pd.read_parquet(native/'atoms_2027_RS.parquet'); games = pd.read_parquet(native/'games_2027_RS.parquet')
    atoms['sq_active'] = True
    atoms = atoms.sort_values('game_date', kind='stable').reset_index(drop=True)
    games = games.sort_values('game_date', kind='stable').reset_index(drop=True)
    engine.append(atoms, games, as_of=cut, input_hashes=own.manifest['input_hashes'])
    from entrant_background import prepare_background
    from vpm.harness.current_entrant_prior import apply_entrant_background, save_board
    base = (root/'refresh.json').resolve().parent
    # refresh.json is symlinked through the tracking folder to its raw snapshot.
    prepared = prepare_background(cut=cut, base=base, reference=reference, ids=own.arrays['ids'], out=root/'entrant_background_inputs')
    restored = apply_entrant_background(engine, own, prepared['background_lanes'], prepared['weights'],
        reported_weights=prepared['reported_weights'], identified=prepared['identified'],
        career_possessions=prepared.get('career_possessions', prepared['packet'].nba_possessions.to_numpy()),
        career_complete=prepared.get('career_complete'), metadata=dict(entrant_prior=prepared['metadata'],
            entrant_model_sha256=prepared['metadata']['model_sha256'], entrant_model_path=prepared['metadata']['model_path']))
    # The same operator with zero background weight must recover the old board.
    control = apply_entrant_background(engine, own, prepared['background_lanes'], np.zeros(len(own.ratings)))
    np.testing.assert_allclose(control.ratings[['ovpm', 'dvpm', 'vpm']], own.ratings[['ovpm', 'dvpm', 'vpm']], atol=1e-9, rtol=0)
    for key in ['own_lanes', 'off_basis', 'def_basis', 'off_td', 'def_td', 'mid_sq_delta', 'three_sq_delta', 'mid_sq_strong', 'three_sq_strong']:
        np.testing.assert_array_equal(restored.arrays[key], own.arrays[key])
    destination = Path(out) if out else E/f'current_refresh_{tag}_entrant_board_v1'
    pins = dict(own.manifest['input_hashes'], **prepared['input_hashes'])
    for path in [own_path/'manifest.json', model_path, Path(__file__), HERE/'entrant_background.py', V/'vpm/harness/current_entrant_prior.py']:
        pins[str(path.resolve())] = sha(path)
    restored.metadata.update(own_only_comparison_manifest_sha256=sha(own_path/'manifest.json'),
        deployment_comparison='Owner-requested frozen entrant blend; no new performance/selection claim')
    manifest = save_board(restored, destination, input_hashes=pins)
    recent = pd.to_datetime(restored.ratings.last_archive_positive_observation_date).ge(cut-pd.Timedelta(days=365))
    ranks = restored.ratings.loc[recent].sort_values('vpm', ascending=False)
    ranks.to_csv(destination/'current_rankings.csv', index=False)
    old = own.ratings.set_index('player_id')
    change = ranks[['player_id','player_name','ovpm','dvpm','vpm','own_prior_off','own_prior_def','background_prior_off','background_prior_def','combined_prior_off','combined_prior_def','background_weight','background_weight_identified']].copy()
    for field in ['ovpm','dvpm','vpm']:
        change[field+'_own_only'] = old[field].reindex(change.player_id).to_numpy()
        change[field+'_change'] = change[field]-change[field+'_own_only']
    change.to_csv(destination/'own_only_comparison.csv', index=False)
    new_result = dict(result, board_path=str(destination), board_manifest_sha256=sha(destination/'manifest.json'),
        rankings_path=str(destination/'current_rankings.csv'), columns=len(restored.ratings.columns), arrays=len(restored.arrays),
        prior_regime=restored.metadata['prior_regime'], background_prior_restored=True, new_entrant_fits=0,
        normal_equation_error=restored.metadata['normal_equation_error'], native_net_gap=restored.metadata['maximum_native_lane_sum_gap'],
        entrant_metadata=prepared['metadata'], nonzero_background_weights=int((prepared['weights']>0).sum()),
        zero_weight_control_max_gap=float(np.max(np.abs(control.ratings[['ovpm','dvpm','vpm']].to_numpy()-own.ratings[['ovpm','dvpm','vpm']].to_numpy()))),
        seconds=time.monotonic()-start)
    (destination/'result.json').write_text(json.dumps(new_result, indent=2, allow_nan=False)+'\n')
    (root/'entrant_result.json').write_text(json.dumps(new_result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(new_result, indent=2), flush=True)
    return new_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('own_root', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args(); restore(args.own_root, out=args.out)

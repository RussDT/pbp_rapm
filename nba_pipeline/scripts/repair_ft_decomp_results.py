#!/usr/bin/env python3
"""Apply audited target-difference ridge corrections to existing DECOMP solves.

The job manifest explicitly names old/new target directories, source results,
penalties, time decay, and rubberband model. Original files are backed up;
receipts make each job idempotent. This does not change model specifications.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import splu
try:
    from . import rapm
except ImportError:
    import rapm

KEYS = ['game_id', 'event_num'] + [f'{s}{i}' for s in 'OD' for i in range(1, 6)]


def ridge_target_corrections(X, delta, penalties, weights=None):
    """Solve all changed RHS together with the same penalized linear operator."""
    weights = np.ones(X.shape[0]) if weights is None else np.asarray(weights)
    delta = np.asarray(delta, dtype=float)
    if delta.ndim == 1:
        delta = delta[:, None]
    centered = delta - np.average(delta, axis=0, weights=weights)
    weighted_X = X.multiply(weights[:, None])
    normal = (X.T @ weighted_X).tocsc() + diags(penalties, format='csc')
    rhs = np.asarray(X.T @ (weights[:, None] * centered))
    beta = splu(normal).solve(rhs)
    if not np.allclose(normal @ beta, rhs, atol=1e-8, rtol=1e-8):
        raise ValueError('Target-correction normal equation did not converge')
    return beta


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backup(path, root):
    path = Path(path).resolve()
    destination = root / 'backup' / path.relative_to(Path.cwd())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def load_family(directory, family, start, end, tags=None, columns=None):
    frames = []
    if tags is None:
        tags = [f'{year % 100:02d}{suffix}' for year in range(start, end+1) for suffix in ['', '_PS']]
    for tag in tags:
        year = (1900 if int(tag[:2]) >= 97 else 2000) + int(tag[:2])
        path = Path(directory) / f'{family}{tag}.parquet'
        if columns is not None:
            import pyarrow.parquet as pq
            names = pq.read_schema(path).names
            frame = pd.read_parquet(path, columns=[name for name in names if name in columns])
        else:
            frame = pd.read_parquet(path)
        frame['season_phase'] = 'PS' if tag.endswith('_PS') else 'RS'
        frame['source_season_end_year'] = year
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def target(frame, component, half_life):
    baseline = None
    if component in rapm.FIRST_CHANCE_ALIAS_PREFIXES:
        baseline = rapm.compute_alt_first_chance_baselines(
            frame, timedecay=bool(half_life), half_life=half_life,
            requested_prefixes=[component])
    needed = {'Net_Diff','Off_Diff','Def_Diff','Is_Turnover','Is_BadPass_TOV','SC_NET_CONT_Diff'}
    if component in rapm.ALT_COMPONENT_COLUMNS:
        needed.add(rapm.ALT_COMPONENT_COLUMNS[component])
    compact = frame[[column for column in frame if column in needed]]
    return rapm.detect_file_type_and_prepare(
        compact, pure=component == 'RAPM', prefix=component,
        alt_baselines=baseline)['Off_Diff'].to_numpy(dtype=float)


def run_job(job, root):
    receipt_path = root / 'receipts' / f"{job['id']}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for record in receipt['changes']:
            if file_hash(record['path']) != record['after_sha256']:
                raise ValueError(f"Previously corrected result changed: {record['path']}")
        return receipt
    start, end = job['start'], job['end']
    half_life = job.get('half_life')
    raw = load_family(job['rapm_dir'], 'RAPM', start, end, job.get('tags'))
    old, new = {}, {}
    deltas, components = [], []
    for component, spec in job['components'].items():
        family = spec['family']
        if family not in old:
            wanted = set(KEYS) | {'score','period','time_quarter','away_score','home_score','game_date','Season',
                'Net_Diff','Off_Diff','Def_Diff','Is_Turnover','Is_BadPass_TOV','SC_NET_CONT_Diff','FC_TS_NET_Diff'}
            wanted.update(rapm.ALT_COMPONENT_COLUMNS[c] for c, entry in job['components'].items()
                          if entry['family']==family and c in rapm.ALT_COMPONENT_COLUMNS)
            old[family] = load_family(job['old_dir'], family, start, end, job.get('tags'), wanted)
            new[family] = load_family(job['new_dir'], family, start, end, job.get('tags'), wanted)
            if not raw[KEYS].apply(pd.to_numeric).equals(old[family][KEYS].apply(pd.to_numeric)) or not raw[KEYS].apply(pd.to_numeric).equals(new[family][KEYS].apply(pd.to_numeric)):
                raise ValueError(f'Possession universe mismatch: {family}')
            for column in ['score','period','time_quarter','away_score','home_score','game_date']:
                if not old[family][column].equals(new[family][column]):
                    raise ValueError(f'Model covariate changed: {family} {column}')
        delta = target(new[family], component, half_life) - target(old[family], component, half_life)
        if np.max(np.abs(delta)) > 1e-10:
            deltas.append(delta)
            components.append(component)
    del old, new
    receipt = {'id': job['id'], 'specification': job, 'changes': [], 'changed_components': components}
    if components:
        raw['Off_Diff'] = raw['Net_Diff']
        if job['rubberband_model'] == 'legacy':
            context = dict(vars(rapm))
            exec(Path(job['legacy_transform']).read_text(), context)
            transformed = context['transform_to_home_away_format'](raw)
        else:
            transformed = rapm.transform_to_home_away_format(raw)
        players = sorted({str(int(x)) for col in KEYS[2:] for x in raw[col].unique() if x > 0})
        mapping = {f'{p}_{side}': 2*i+j for i,p in enumerate(players) for j,side in enumerate(['off','def'])}
        builder = rapm.build_design_matrix_with_rubberband if job['rubberband_model']=='legacy' else rapm.build_design_matrix_with_smooth_rubberband
        X, _, _ = builder(transformed, mapping, season_effects=True)
        weights = rapm.compute_time_decay_weights_from_dates(raw.game_date, rapm.half_life_to_decay_base(half_life)) if half_life else None
        counts = {side: pd.concat([raw[f'{side}{i}'] for i in range(1,6)]).astype(int).astype(str).value_counts().to_dict() for side in 'OD'}
        delta_matrix = np.column_stack(deltas)
        for alphas in sorted({tuple(job['components'][c]['alphas']) for c in components}):
            indices = [i for i,c in enumerate(components) if tuple(job['components'][c]['alphas'])==alphas]
            penalties = np.zeros(X.shape[1])
            for name,index in mapping.items():
                if name.endswith('_off'): penalties[index] = alphas[0]
                elif name.endswith('_def'): penalties[index] = alphas[1]
                elif name.startswith('rb_'): penalties[index] = 10 if job['rubberband_model']=='legacy' else 100
            corrections = ridge_target_corrections(X, delta_matrix[:,indices], penalties, weights)
            for j,i in enumerate(indices):
                component = components[i]
                correction = corrections[:,j]
                rapm.recenter_player_coefficients_by_side_possessions(correction,mapping,players,defaultdict(int,counts['O']),defaultdict(int,counts['D']))
                path = Path(job['components'][component]['path'])
                source = backup(path,root)
                frame = pd.read_csv(source)
                for side in ['off','def']:
                    lookup = {p:100*correction[mapping[f'{p}_{side}']] for p in players}
                    changes = frame.player_id.astype(str).map(lookup)
                    if changes.isna().any(): raise ValueError('Missing player in correction')
                    frame[side] = frame[side] + changes
                frame['net_rapm'] = frame['off'] - frame['def']
                frame.to_csv(path,index=False)
                receipt['changes'].append({'component':component,'path':str(path),'before_sha256':file_hash(source),'after_sha256':file_hash(path),'max_abs_coefficient_correction':float(np.max(np.abs(correction[:2*len(players)]))*100)})
                # Rebuild nuisance exports from their old coefficients plus the same correction.
                stem = path.name.removesuffix('_results.csv')
                rb_path = path.with_name(stem + '_rubberband_coefficients.csv')
                if rb_path.exists():
                    rb_old = pd.read_csv(backup(rb_path, root))
                    rb_beta = correction.copy()
                    for row in rb_old.itertuples():
                        rb_beta[mapping[row.feature]] += row.coefficient / 100.0
                    for ancillary in path.parent.glob(stem + '_rubberband_*.csv'):
                        backup(ancillary, root)
                    rapm.save_rubberband_analysis(rb_beta, mapping, path.parent, stem,
                                                 rubberband_model=job['rubberband_model'])
                season_path = path.with_name(stem + '_season_effects.csv')
                if season_path.exists():
                    a = pd.read_csv(backup(season_path, root))
                    a['effect_per_100'] += [
                        100*correction[mapping[f'season_fe_{int(row.season)}_{row.season_phase}']]
                        if f'season_fe_{int(row.season)}_{row.season_phase}' in mapping else 0
                        for row in a.itertuples()]
                    a.to_csv(season_path, index=False)
        receipt['max_target_change'] = float(np.max(np.abs(delta_matrix)))
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    receipt=run_job(json.loads(args.job.read_text()),args.root)
    print(json.dumps({'id':receipt['id'],'changed_components':receipt['changed_components']}),flush=True)

if __name__=='__main__': main()

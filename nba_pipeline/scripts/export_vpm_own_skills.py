"""Export neutral own-skill states for VPM research, without retuning filters.

Fixed 2020-2025 skill parameters make historical replay DEVELOPMENT ONLY.
The exported state is the latest covered postgame state strictly before a cut;
it is not advanced across idle days. No 2026/2027 prediction errors are scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_nba_estimated_skills import SKILLS, load_panel, weighted_mse
from research_skills_nextgame_lab import prepare, assemble, X, CTX_PATH, AGE_PATH

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/estimated_skills'
OUT = SOURCE / 'vpm_neutral_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot_rows(states, cuts, *, max_age_days=730):
    """Latest strictly pre-cut observed state; preserve stale values as missing."""
    keys = ['player_id', 'game_id'] if 'game_id' in states else ['player_id', 'game_date']
    if states.duplicated(keys).any():
        raise ValueError('Duplicate player/game states')
    states = states.sort_values(['player_id', 'game_date'], kind='stable')
    frames = []
    for cut in cuts:
        cut = pd.Timestamp(cut)
        d = states.loc[states.game_date < cut].groupby('player_id').tail(1).copy()
        d['estimate_date'] = cut
        d['last_observation_date'] = d.pop('game_date')
        d['age_days'] = (cut - d.last_observation_date).dt.days
        d['is_missing'] = d.age_days.gt(max_age_days) | d.value.isna()
        d.loc[d.is_missing, ['value', 'state_variance']] = np.nan
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def build(cuts, out=OUT, rim_directory=None):
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    cuts = sorted({pd.Timestamp(x).normalize() for x in cuts})
    if not cuts or max(cuts) > pd.Timestamp('2026-09-14'):
        raise ValueError('This frozen development export supports cuts through 2026-09-14')
    df = load_panel()
    # 2026 is filtered for export only. 2027 cannot enter any helper.
    df = df.loc[df.season.le(2026) & df.game_date.lt(max(cuts))].copy()
    df['game_id'] = df.game_id.astype(str).str.zfill(10)
    ctx = pd.read_parquet(CTX_PATH)
    df = df.merge(ctx, on=['game_id', 'player_id'], how='left', validate='one_to_one')
    age = pd.read_parquet(AGE_PATH)
    a_path = SOURCE / 'v2_stage_a.json'
    v_path = SOURCE / 'v4_nextgame.json'
    stage_a = json.loads(a_path.read_text())
    v4 = json.loads(v_path.read_text())
    absent = X.absent_teammates(df, ctx)
    all_rows, closure, definitions = [], {}, {}
    for spec in SKILLS:
        prep = prepare(spec, df, ctx, age, absent, stage_a, v4, context_reference_end=2025)
        run, _, _, _ = assemble(prep, set())
        pred, _, post, post_var, _ = run(prep['v4_params'])
        loss = weighted_mse(pred, prep['obs'], prep['expo'], prep['dev'])
        gap = abs(loss - prep['v4_obj'])
        # Stable ordering of same-date state ties repairs a tiny legacy
        # context difference. Record it; never silently claim bitwise closure.
        if gap > 1e-8:
            raise ValueError(f'{spec.name}: v4 reconstruction differs by {gap}')
        closure[spec.name] = gap
        sub = prep['sub']
        scale = 1.0 if spec.kind == 'pct' else 100.0
        name = 'own_' + spec.name
        states = sub[['player_id', 'game_id', 'game_date']].copy()
        states['value'] = post * scale
        states['state_variance'] = post_var * scale**2
        states['observed_exposure'] = sub.groupby('player_id').exposure.cumsum()
        rows = snapshot_rows(states, cuts)
        rows['feature'] = name
        rows['model_version'] = 'pbp_v4_neutral_post'
        rows['parameter_fit_end'] = pd.Timestamp('2025-06-30')
        rows['model_sha256'] = sha(v_path)
        all_rows.append(rows)
        definitions[name] = {
            'units': 'fraction' if spec.kind == 'pct' else f'events per 100 {spec.denom}',
            'numerator': spec.numer, 'denominator': spec.denom,
            'state': 'post = mu + delta + league center; context removed from innovations',
            'variance': 'raw uncalibrated latent-state variance; not a confidence interval',
        }
        print(f'{name}: {len(rows)} snapshots, closure {gap:.2g}', flush=True)

    rim_directory = Path(rim_directory).resolve() if rim_directory is not None else SOURCE / 'predictive_rim_saved'
    rim_path = rim_directory / 'dated_predictions.parquet'
    rim_frozen = rim_directory / 'frozen.json'
    dfga_path = SOURCE / 'tracking_next_game/dfga/predictions.parquet'
    dfga_frozen = SOURCE / 'tracking_next_game/dfga/frozen.json'
    rim = pd.read_parquet(rim_path)
    dfga = pd.read_parquet(dfga_path)
    selected = json.loads(dfga_frozen.read_text())['selected']
    if not dfga.dfga_valid.all() or not dfga.dfga_reason.eq('observed').all():
        raise ValueError('DFGA state source includes non-measured observations')
    for tab, feature, col, exposure, version, frozen, fit_end in [
        (rim, 'own_rim_saved100', 'post_saved100', 'rim_dfga', 'rim_lagleague_h730_k100', rim_frozen, '2023-06-30'),
        (rim, 'own_rim_contests100', 'post_contests100', 'def_poss', 'rim_lagleague_h730_k100', rim_frozen, '2023-06-30'),
        (rim, 'own_rim_suppression', 'post_suppression', 'rim_dfga', 'rim_lagleague_h730_k100', rim_frozen, '2023-06-30'),
        (dfga, 'own_dfga100', selected + '_postgame100', 'exposure', 'dfga_' + selected, dfga_frozen, '2022-06-30'),
    ]:
        tab = tab.sort_values(['player_id', 'game_date']).copy()
        tab['value'] = tab[col]
        tab['state_variance'] = np.nan  # these published states do not expose variance
        tab['observed_exposure'] = tab.groupby('player_id')[exposure].cumsum()
        rows = snapshot_rows(tab[['player_id', 'game_id', 'game_date', 'value', 'state_variance', 'observed_exposure']], cuts)
        rows['feature'] = feature
        rows['model_version'] = version
        rows['parameter_fit_end'] = pd.Timestamp(fit_end)
        rows['model_sha256'] = sha(frozen)
        all_rows.append(rows)
        definitions[feature] = {'source_field': col, 'source': str(frozen.relative_to(ROOT)),
                                'units': 'fraction' if feature.endswith('suppression') else 'per 100 defensive possessions'}
    long = pd.concat(all_rows, ignore_index=True)
    if long.duplicated(['player_id', 'estimate_date', 'feature']).any():
        raise ValueError('Duplicate player/date/feature')
    wide = long.pivot(index=['player_id', 'estimate_date'], columns='feature', values='value').reset_index()
    for zone, rate in [('rim', 'rim_fga_rate'), ('mid', 'mid_fga_rate'), ('three', 'three_fga_rate'), ('ft', 'fta_rate')]:
        wide[f'own_{zone}_makes100'] = wide[f'own_{rate}'] * wide[f'own_{zone}_pct']
        definitions[f'own_{zone}_makes100'] = {'formula': f'own_{rate} * own_{zone}_pct', 'units': 'makes per 100 off_poss'}
    # Known missingness has an explicit channel; no zero-filled measurements.
    for feature in definitions:
        wide[feature + '_missing'] = wide[feature].isna().astype(int)
    complete = wide.dropna(subset=['own_rim_saved100', 'own_rim_contests100', 'own_rim_suppression'])
    gap = (complete.own_rim_saved100 - 2 * complete.own_rim_contests100 * complete.own_rim_suppression).abs().max()
    if gap > 1e-10:
        raise ValueError('Rim savings identity broken')
    sources = [a_path, v_path, Path(CTX_PATH), Path(AGE_PATH), SOURCE / 'player_game_lines.parquet',
               rim_path, rim_frozen, dfga_path, dfga_frozen, Path(__file__)]
    sources += [Path(__file__).parent / x for x in ['research_skills_nextgame_lab.py', 'build_nba_estimated_skills.py',
                'build_nba_estimated_skills_v2.py', 'build_nba_estimated_skills_v2_stageb.py',
                'estimated_skills_context.py', 'estimated_skills_corrections.py']]
    out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out / 'feature_provenance.parquet', index=False)
    wide.to_parquet(out / 'features.parquet', index=False)
    manifest = dict(schema_version=1, status='development_integration_not_admitted',
                    cuts=[str(x.date()) for x in cuts], rows=len(wide),
                    feature_definitions=definitions, v4_closure=closure,
                    observation_rule='strictly before cut; latest covered state held, not advanced',
                    max_state_age_days=730, context_in_prior=False,
                    historical_parameter_replay=True, nested_chronological_fit=False,
                    fit_end='2025-06-30', confirmation='2027 protected', scored_seasons=[],
                    limitations=['Fixed skill parameters selected using 2020-2025; historical replay is development only',
                                 'Biographical/source availability is retrospective; no real-time availability guarantee',
                                 'Rim coverage follows the hashed source; recovered cumulative counts may exclude ambiguous gaps; stale states become missing after 730 days',
                                 'Total personal fouls use off_poss + def_poss, not defensive possessions',
                                 'Neutral state has not passed an impact or game-prediction admission gate'],
                    sources={str(p): sha(p) for p in sources},
                    artifacts={n: sha(out / n) for n in ['features.parquet', 'feature_provenance.parquet']})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(out), 'rows': len(wide), 'features': len(definitions)}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuts', required=True, help='Comma-separated estimate dates')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--rim-directory', type=Path, help='Explicit versioned rim state package; default preserves original source')
    args = parser.parse_args()
    build(args.cuts.split(','), args.out, args.rim_directory)

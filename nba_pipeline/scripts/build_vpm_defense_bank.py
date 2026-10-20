"""Measured defensive counts -> matched-window rates and dated talent summaries.

RAPM retains player-game measurements. VPM receives only aggregated player
packets, with explicit coverage and strictly pre-cut observation dates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / 'results/estimated_skills'
OUT = SKILLS / 'defense_bank_v51_shooting_pf_repaired'
FEATURES = ['deflections100', 'contested_dreb100', 'contested_shots100',
            'charges_drawn100', 'off_fouls_drawn100', 'shooting_pf100', 'nonoff_pf100']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(frame):
    d = frame.copy()
    d['game_id'] = d.game_id.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    d['game_date'] = pd.to_datetime(d.game_date)
    return d


def measured_rows(frame, feature, numerator, valid):
    d = frame.loc[valid, ['player_id', 'game_id', 'game_date', 'def_poss', numerator]].copy()
    d = d.rename(columns={numerator: 'numerator', 'def_poss': 'denominator'})
    if d.duplicated(['player_id', 'game_id']).any():
        raise ValueError('Ambiguous measured identity: ' + feature)
    d = d.loc[np.isfinite(d.numerator) & d.numerator.ge(0) & d.denominator.gt(0)]
    d['feature'] = 'def_' + feature
    return d


def shooting_foul_events(raw):
    """Recognize observed foul labels; archived action subtype is often zero."""
    descriptions = raw[['home_description', 'neutral_description', 'visitor_description']].fillna('')
    text = descriptions.astype(str).agg(' '.join, axis=1).str.upper()
    shooting = text.str.contains(r'\bS\.FOUL\b|\bSHOOTING(?:\s+BLOCK)?\s+FOUL\b', regex=True)
    return raw.loc[raw.event_type.eq(6) & raw.player1_id.gt(0) & shooting]


def summarize(measurements, players, cut, game_ids=None, half_life=None, prior_exposure=1000.):
    """Unsmoothed matched calibration rates, or dated EB/decayed talent rates.

    No observed history means missing, including when a population mean exists.
    A measured zero updates both numerator and exposure. Calendar gaps decay
    evidence; missing games are never observations.
    """
    cut = pd.Timestamp(cut)
    d = measurements.loc[pd.to_datetime(measurements.game_date).lt(cut)].copy()
    if game_ids is not None:
        d = d.loc[d.game_id.isin(game_ids)]
    result = pd.DataFrame({'player_id': np.asarray(players, int)})
    history = []
    for name in ['def_' + f for f in FEATURES]:
        g = d.loc[d.feature.eq(name)].copy()
        if half_life is None:
            g['weight'] = 1.
        else:
            g['weight'] = np.exp2(-(cut - pd.to_datetime(g.game_date)).dt.total_seconds() / (86400 * half_life))
        g['n'] = g.numerator * g.weight
        g['e'] = g.denominator * g.weight
        sums = g.groupby('player_id')[['n', 'e']].sum().reindex(result.player_id)
        if half_life is None:
            value = 100 * sums.n / sums.e.where(sums.e.gt(0))
        else:
            league = float(g.n.sum() / g.e.sum()) if len(g) and g.e.sum() > 0 else 0.
            value = 100 * (sums.n + prior_exposure * league) / (sums.e + prior_exposure)
            value = value.where(sums.e.gt(0))
        result[name] = value.to_numpy()
        result[name + '__exposure'] = sums.e.to_numpy()
        last = g.groupby('player_id').game_date.max().reindex(result.player_id)
        for player, date, exposure in zip(result.player_id, last, sums.e):
            if pd.notna(date):
                history.append(dict(player_id=int(player), feature=name, estimate_date=cut,
                                    last_observation_date=date, measured_exposure=float(exposure)))
    return result, pd.DataFrame(history)


def prepare(out=OUT):
    out = Path(out)
    if out.resolve() == (SKILLS / 'defense_bank_v51').resolve():
        raise ValueError('Preserve the original invalid shooting-foul source; use the repaired output root')
    out.mkdir(parents=True, exist_ok=True)
    linepath = SKILLS / 'monthly_source_2014_2026/player_game_lines.parquet'
    trackingpath = SKILLS / 'monthly_defense_v1/dfga_counts.parquet'
    deflectionpath = SKILLS / 'deflections/panel.parquet'
    foulpath = SKILLS / 'fouls_drawn/fouls_drawn_game_lines.parquet'
    paths = [linepath, trackingpath, deflectionpath, foulpath, Path(__file__)]
    lines = ids(pd.read_parquet(linepath))
    if lines.duplicated(['game_id', 'player_id']).any() or lines.season.max() > 2026:
        raise ValueError('Invalid source lines/protected season')
    parts = []
    d = ids(pd.read_parquet(deflectionpath))
    parts.append(measured_rows(d, 'deflections100', 'deflections', d.valid))
    d = ids(pd.read_parquet(trackingpath))
    for short in ['contested_dreb', 'contested_shots']:
        parts.append(measured_rows(d, short + '100', short + '_value', d[short + '_valid'] & d[short + '_reason'].eq('observed')))
    d = ids(pd.read_parquet(foulpath))
    d = d.merge(lines[['game_id', 'player_id', 'def_poss', 'minutes']], on=['game_id', 'player_id'],
                validate='one_to_one', suffixes=('', '_canonical'))
    valid = d.fouls_drawn_is_measured & d.minutes.sub(d.minutes_canonical).abs().le(.5)
    for short in ['charges_drawn', 'off_fouls_drawn']:
        parts.append(measured_rows(d, short + '100', short, valid))
    d = lines.copy()
    d['nonoff_pf'] = d.pf_committed - d.off_fouls_committed
    parts.append(measured_rows(d, 'nonoff_pf100', 'nonoff_pf', d.nonoff_pf.ge(0)))
    audits = []
    for year in range(2014, 2027):
        for phase in ['RS', 'PS']:
            path = ROOT / 'raw_data' / f'NBA{year % 100:02d}{"_PS" if phase == "PS" else ""}.parquet'
            paths.append(path)
            raw = pd.read_parquet(path).drop_duplicates()
            raw['game_id'] = raw.game_id.astype(str).str.zfill(10)
            f = shooting_foul_events(raw)
            if raw.event_type.eq(6).sum() > 50 and f.empty:
                raise ValueError('Degenerate shooting-foul extraction from a complete source file')
            if f.duplicated(['game_id', 'event_num']).any():
                raise ValueError('Ambiguous shooting foul event identity')
            counts = f.groupby(['game_id', 'player1_id']).size().rename('shooting_pf')
            d = lines.loc[lines.season.eq(year) & lines.season_type.eq(phase.lower())].copy()
            idx = pd.MultiIndex.from_frame(d[['game_id', 'player_id']])
            d['shooting_pf'] = counts.reindex(idx, fill_value=0).to_numpy()
            valid = d.game_id.isin(set(raw.game_id)) & d.shooting_pf.le(d.pf_committed)
            audits.append(dict(season=year, phase=phase, source_events=len(f),
                               admitted_events=int(d.loc[valid, 'shooting_pf'].sum()),
                               rejected_player_games=int((~valid).sum())))
            parts.append(measured_rows(d, 'shooting_pf100', 'shooting_pf', valid))
    measured = pd.concat(parts, ignore_index=True)
    # Independently enforce canonical exposure/date for every accepted measurement.
    joined = measured.merge(lines[['game_id', 'player_id', 'game_date', 'def_poss']],
                            on=['game_id', 'player_id'], validate='many_to_one', suffixes=('', '_canonical'))
    if len(joined) != len(measured):
        raise ValueError('Measured rows absent from canonical lines')
    np.testing.assert_array_equal(joined.game_date, joined.game_date_canonical)
    np.testing.assert_allclose(joined.denominator, joined.def_poss, atol=0, rtol=0)
    if measured.duplicated(['game_id', 'player_id', 'feature']).any():
        raise ValueError('Duplicate feature observations')
    measured.to_parquet(out / 'measurements.parquet', index=False)
    vpm = ROOT.parents[1] / 'vpm/vpm/results/harness'
    packets = []
    histories = []
    for label, cut in [(str(y), pd.Timestamp(f'{y-1}-12-01' if y == 2021 else f'{y-1}-10-01'))
                       for y in range(2020, 2027)] + [('current', pd.Timestamp('2026-09-01'))]:
        base = (vpm / 'vpm_long_spm_v1/calibrations' / label if label != 'current'
                else vpm / 'vpm_long_spm_current_2026_09_01_v1/calibration')
        games = pd.read_parquet(base / 'window_games.parquet')
        players = pd.read_parquet(base / 'window_features.parquet').player_id.to_numpy()
        observed, _ = summarize(measured, players, cut, set(games.game_id.astype(str).str.zfill(10)))
        observed.to_parquet(out / f'observed_{label}.parquet', index=False)
        component_games = games.loc[games.source_season_end_year.ge(2016)]
        observed, _ = summarize(measured, players, cut, set(component_games.game_id.astype(str).str.zfill(10)))
        observed.to_parquet(out / f'component_observed_{label}.parquet', index=False)
        universe = lines.loc[lines.game_date.lt(cut), 'player_id'].unique()
        snapshot, history = summarize(measured, universe, cut, half_life=550)
        snapshot['estimate_date'] = cut
        packets.append(snapshot)
        histories.append(history)
    pd.concat(packets, ignore_index=True).to_parquet(out / 'features.parquet', index=False)
    pd.concat(histories, ignore_index=True).to_parquet(out / 'provenance.parquet', index=False)
    coverage = measured.assign(year=measured.game_date.dt.year).groupby(['feature', 'year']).size()
    outputs = sorted(out.glob('*.parquet'))
    manifest = dict(status='passed', source_hashes={str(p): sha(p) for p in paths},
                    artifact_hashes={p.name: sha(p) for p in outputs}, shooting_foul_audits=audits,
                    measurement_rows=len(measured), features=['def_' + x for x in FEATURES],
                    snapshot_half_life=550, prior_exposure=1000,
                    calibration='raw same-window matched measured counts',
                    shooting_fouls='description-labelled S.FOUL/shooting foul; archived numeric subtype is not authoritative',
                    snapshots='fixed TD550, K1000, past-prefix population prior; no fitted future parameters',
                    coverage={str(k): int(v) for k, v in coverage.items()}, protected_outcomes=0)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='passed', rows=len(measured), output=str(out)), indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    prepare(parser.parse_args().out)

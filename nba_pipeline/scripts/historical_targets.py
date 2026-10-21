"""Versioned, date-fenced actual-point/Net targets for quarterly historical VPM.

This adapter never fits SQ. The audited clean SQ residuals are available from
2016; earlier residuals are explicitly inactive (zero correction), not estimated
from a retrospectively trained shot model. RAPM remains owner of row algebra.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_vpm_net_lane_atoms import build as canonical_build
from vpm_net_lane_targets import LANES, REFERENCES, compact, targets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/vpm_historical_targets_v1'
EXISTING = ROOT / 'results/vpm_net_lane_atoms_v1'
CLEAN_SQ = ROOT / 'results/monthly_vpm_targets_clean_sq_v2'
QUARANTINE = ROOT / 'results/long_spm_actual_warmup_v1/score_accounting.json'
SLOTS = [f'{s}{i}' for s in ['O', 'D'] for i in range(1, 6)]
KEYS = ['source_season_end_year', 'season_phase', 'game_id']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quarantine_ids(path=QUARANTINE):
    return {int(v) for v in json.loads(Path(path).read_text())['quarantined_game_ids']}


def validate_population(atoms, games, cut=None):
    """Pricing population must match exactly: no future/extra/duplicate games."""
    if atoms.empty or games.empty:
        raise ValueError('Empty historical target population')
    if games.duplicated(KEYS).any():
        raise ValueError('Duplicate historical pricing games')
    if atoms.duplicated(KEYS + ['event_num']).any():
        raise ValueError('Duplicate canonical possession identity')
    ad = pd.to_datetime(atoms.game_date, errors='raise')
    gd = pd.to_datetime(games.game_date, errors='raise')
    if ad.isna().any() or gd.isna().any():
        raise ValueError('Missing historical game dates')
    if cut is not None and (ad.ge(pd.Timestamp(cut)).any() or gd.ge(pd.Timestamp(cut)).any()):
        raise ValueError('Historical target/pricing date fence violated')
    ap = atoms[KEYS].drop_duplicates()
    gp = games[KEYS]
    if len(ap) != len(gp) or not ap.merge(gp, on=KEYS, how='outer', indicator=True)._merge.eq('both').all():
        raise ValueError('Historical target/pricing game populations differ')
    dates = atoms.assign(game_date=ad).groupby(KEYS).game_date.agg(['min', 'max'])
    if not dates['min'].equals(dates['max']):
        raise ValueError('A historical game has inconsistent dates')
    join = games.assign(game_date=gd).merge(dates['min'].rename('atom_date'), on=KEYS, validate='one_to_one')
    if not join.game_date.eq(join.atom_date).all():
        raise ValueError('Historical target/pricing dates differ')
    actual = atoms.actual_points.to_numpy(float)
    if not np.isfinite(actual).all() or not np.allclose(actual, np.rint(actual), atol=1e-9, rtol=0):
        raise ValueError('Historical parent must be finite actual FG/FT points')
    np.testing.assert_allclose(atoms.First_Chance_Diff + atoms.Second_Chance_Diff, actual, atol=1e-9, rtol=0)


def compare_scores(raw, exported):
    """Independent event accounting; MISS must be a token, never surname Missi."""
    raw = raw.drop_duplicates().copy()
    raw["game_id"] = pd.to_numeric(raw.game_id, errors="raise").astype("int64")
    text = raw.home_description.fillna("") + " " + raw.visitor_description.fillna("")
    event = pd.to_numeric(raw.event_type, errors="raise")
    missed = text.str.contains(r"\bMISS\b", case=False, regex=True)
    points = np.where(event.eq(1), np.where(text.str.contains("3PT", case=False), 3, 2),
                      np.where(event.eq(3) & ~missed, 1, 0))
    expected = pd.Series(points, index=raw.index).groupby(raw.game_id).sum()
    exported = exported.copy()
    exported["game_id"] = pd.to_numeric(exported.game_id, errors="raise").astype("int64")
    actual = exported.groupby("game_id").actual_points.sum()
    rows = []
    for game in sorted(set(expected.index) | set(actual.index)):
        rp = None if game not in expected else float(expected[game])
        cp = None if game not in actual else float(actual[game])
        delta = None if rp is None or cp is None else cp - rp
        rows.append(dict(game_id=str(game).zfill(10), raw_event_points=rp,
                         canonical_actual_points=cp, difference=delta, accepted=delta == 0))
    return pd.DataFrame(rows)


def build(year, phase, out=OUT):
    """Build/reuse canonical measurements, with a new immutable scoring gate."""
    if not 1997 <= int(year) <= 2026 or phase not in ['RS', 'PS']:
        raise ValueError('Only completed 1997–2026 historical sources admitted')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    name = f'atoms_{year}_{phase}.parquet'
    dest = out / name
    receipt = out / f'historical_{year}_{phase}.json'
    suffix = '_PS' if phase == 'PS' else ''
    raw = ROOT / f'raw_data/NBA{year % 100:02d}{suffix}.parquet'
    canonical = ROOT / f'processed/RAPM{year % 100:02d}{suffix}.parquet'
    if receipt.exists():
        meta = json.loads(receipt.read_text())
        for p, expected in meta['artifacts'].items():
            if sha(out / p) != expected:
                raise ValueError(f'Frozen historical target artifact changed: {p}')
        for p, expected in meta['source_files'].items():
            if sha(p) != expected:
                raise ValueError(f'Frozen historical target source changed: {p}')
        return meta
    if not dest.exists():
        src = EXISTING / name
        if src.exists():
            old = json.loads(src.with_suffix('.json').read_text())
            for n, h in old['artifacts'].items():
                if sha(EXISTING / n) != h:
                    raise ValueError(f'Canonical Net measurement changed: {n}')
            # Symlinks retain provenance; no duplicate large generated files.
            for n in old['artifacts']:
                (out / n).symlink_to((EXISTING / n).resolve())
            lineage = dict(kind='audited_v61_measurement_archive', receipt=str(src.with_suffix('.json')),
                           receipt_sha256=sha(src.with_suffix('.json')))
        else:
            canonical_build(year, phase, out)
            lineage = dict(kind='new_canonical_price_free_measurements', receipt=str(dest.with_suffix('.json')),
                           receipt_sha256=sha(dest.with_suffix('.json')))
    else:
        prior_receipt = dest.resolve().with_suffix('.json')
        lineage = dict(kind='canonical_measurements_prepared_before_receipt', receipt=str(prior_receipt),
                       receipt_sha256=sha(prior_receipt))
    d = pd.read_parquet(dest)
    gp = out / f'games_{year}_{phase}.parquet'; g = pd.read_parquet(gp)
    validate_population(d, g)
    accounting = compare_scores(pd.read_parquet(raw), d)
    rejected = accounting.loc[~accounting.accepted]
    registered = quarantine_ids()
    excluded = registered | {int(v) for v in rejected.game_id}
    # Additional early-era score failures are quarantined, never silently repaired.
    good = d.loc[~pd.to_numeric(d.game_id).isin(excluded)].reset_index(drop=True)
    gg = g.loc[~pd.to_numeric(g.game_id).isin(excluded)].reset_index(drop=True)
    validate_population(good, gg)
    y, pricing = targets(good, gg)
    primitive_gap = float(np.max(abs(y[:, :len(LANES)].sum(axis=1) - good.actual_points)))
    standard_gap = float(np.max(abs(compact(y).sum(axis=1) - good.actual_points)))
    meta = dict(schema_version=1, year=year, phase=phase, rows=len(d), games=int(d.game_id.nunique()),
                admitted_rows=len(good), admitted_games=int(good.game_id.nunique()),
                independently_checked_games=len(accounting), scoring_mismatches=rejected.astype(object).where(pd.notna(rejected), None).to_dict(orient='records'),
                excluded_game_ids=sorted(excluded), actual_FT=True, price_free=True,
                max_primitive_closure_gap=primitive_gap, max_standard_midrange_closure_gap=standard_gap,
                sq_policy='audited_clean_dated_residuals_from_2016; earlier_correction_inactive', lineage=lineage,
                source_files={str(p): sha(p) for p in [raw, canonical, Path(__file__), QUARANTINE]},
                artifacts={dest.name: sha(dest), gp.name: sha(gp)}, protected_outcomes=0)
    write(receipt, meta)
    print(f'Historical target checked {year} {phase}: {len(good):,} rows, {len(rejected)} scoring quarantines', flush=True)
    return meta


def load_before_cut(root, cut, years, phases=('RS', 'PS'), exclude_games=(), clean_sq_root=CLEAN_SQ,
                    start=None, sq=True):
    """Load admitted rows strictly before cut; optional start makes a label window.

    Returns (atoms, games). `mid/three_gross_residual` are compatible with
    build_monthly_vpm_targets.reprice_before_cut. `sq_active` records which rows
    have audited SQ rather than pretending pre-2016 residuals were observed zero.
    """
    root = Path(root); cut = pd.Timestamp(cut)
    if cut > pd.Timestamp('2026-10-21'):
        raise ValueError('Historical request enters protected 2027 outcome period')
    all_d, all_g = [], []
    for year in years:
        if year > 2026:
            raise ValueError('Protected 2027 outcomes requested')
        for phase in phases:
            p = root / f'atoms_{year}_{phase}.parquet'
            meta = json.loads((root / f'historical_{year}_{phase}.json').read_text())
            for n, h in meta['artifacts'].items():
                if sha(root / n) != h:
                    raise ValueError(f'Historical target changed: {n}')
            d = pd.read_parquet(p); g = pd.read_parquet(root / f'games_{year}_{phase}.parquet')
            d.game_date = pd.to_datetime(d.game_date); g.game_date = pd.to_datetime(g.game_date)
            excluded = set(map(int, exclude_games)) | set(meta['excluded_game_ids'])
            for frame in [d, g]:
                # Use full-game selection on both sides of the exact match.
                frame.drop(frame.index[frame.game_date.ge(cut) | pd.to_numeric(frame.game_id).isin(excluded)], inplace=True)
                if start is not None:
                    frame.drop(frame.index[frame.game_date.lt(pd.Timestamp(start))], inplace=True)
            if d.empty:
                if not g.empty:
                    raise ValueError('Pricing exists without canonical historical possessions')
                continue
            validate_population(d, g, cut)
            d['sq_active'] = False
            for zone in ['mid', 'three']:
                d[zone + '_gross_residual'] = 0.
            if sq and year >= 2016:
                sp = Path(clean_sq_root) / p.name
                sm = json.loads(sp.with_suffix('.json').read_text())
                if sha(sp) != sm['output_sha256']:
                    raise ValueError('Audited clean SQ target changed')
                q = pd.read_parquet(sp)
                q = q.loc[pd.to_numeric(q.game_id).isin(pd.to_numeric(d.game_id))].reset_index(drop=True)
                d = d.reset_index(drop=True)
                if len(q) != len(d):
                    raise ValueError('Historical clean SQ possession count differs')
                for c in ['game_id', 'event_num', *SLOTS]:
                    np.testing.assert_array_equal(pd.to_numeric(q[c]), pd.to_numeric(d[c]))
                np.testing.assert_array_equal(pd.to_datetime(q.game_date), d.game_date)
                np.testing.assert_allclose(q.actual_points, d.actual_points, atol=1e-12, rtol=0)
                for zone in ['mid', 'three']:
                    d[zone + '_gross_residual'] = q[zone + '_gross_residual'].to_numpy()
                d['sq_active'] = True
            all_d.append(d); all_g.append(g)
    if not all_d:
        raise ValueError('No admitted historical games before requested cut')
    d = pd.concat(all_d, ignore_index=True); g = pd.concat(all_g, ignore_index=True)
    validate_population(d, g, cut)
    return d, g


def target_matrix(atoms, games, cut, weights=None, standard=False):
    """Price exactly the admitted dated population; preserve original lane algebra."""
    validate_population(atoms, games, cut)
    y, pricing = targets(atoms, games, weights)
    for zone in ['mid', 'three']:
        atoms[zone + '_residual'] = 0.
    for rec in pricing['files']:
        use = atoms.source_season_end_year.eq(rec['season']) & atoms.season_phase.eq(rec['phase'])
        for zone, points in [('mid', 2.), ('three', 3.)]:
            atoms.loc[use, zone + '_residual'] = atoms.loc[use, zone + '_gross_residual'] * (1 - rec['kappa'][zone] / points)
    pricing['sq_active_rows'] = int(atoms.sq_active.sum())
    pricing['sq_inactive_rows'] = int((~atoms.sq_active).sum())
    return (compact(y) if standard else y), pricing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--years', default=','.join(map(str, range(2008, 2027))))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args(); tasks = [(y, phase, a.out) for y in map(int, a.years.split(',')) for phase in ['RS', 'PS']]
    if a.workers == 1:
        records = [build(*t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            records = [f.result() for f in as_completed([pool.submit(build, *t) for t in tasks])]
    write(a.out / 'manifest.json', dict(files=records, total_rows=sum(r['admitted_rows'] for r in records),
                                       protected_outcomes=0, status='historical_measurements_and_date_fenced_target_adapter'))


if __name__ == '__main__':
    main()

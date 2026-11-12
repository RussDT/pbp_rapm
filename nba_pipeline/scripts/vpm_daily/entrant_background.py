"""Frozen clean-v61 entrant component prior for current full-feature VPM.

No fitting or rating solve. Career counts are separate from feature-admission
masks. Fresh completed college observations use the original feature estimator;
the fixed 2025 scaler/coefficients and actual-draft source stay unchanged.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[3]
V = R.parent / 'vpm'
ENTRANTS = V / 'vpm/data/entrants'
MODEL = V / 'vpm/results/harness/vpm_net_lane_clean_v1/entrant_models/2025.json'
MONTHLY = R / 'nba_pipeline/results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet'
SLOTS = [f'{side}{i}' for side in ('O', 'D') for i in range(1, 6)]
COLLEGE_FIELDS = ['id', 'player', 'player_id', 'team', 'year', 'nba_id', 'games', 'mpg',
                  'orpg', 'drpg', 'apg', 'spg', 'bpg', 'topg', 'ftm', 'fta',
                  'two_m', 'two_a', 'three_m', 'three_a', 'birthdate', 'espn_id', 'updated_at']


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + '\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError('Preserved entrant preparation differs: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(raw)


def _college_snapshot(out):
    """Bounded read-only public table fetch, selecting no impact or draft-pick fields."""
    path = out / 'college_2026_raw_snapshot.json'
    if path.exists():
        snapshot = json.loads(path.read_bytes())
        if snapshot['table'] != 'public.draft_ready_data' or snapshot['year'] != 2026 or snapshot['columns'] != COLLEGE_FIELDS:
            raise ValueError('Existing college snapshot query differs')
        return path, snapshot['rows']
    import requests
    from dotenv import dotenv_values
    env = {**dotenv_values(R / '.env'), **os.environ}
    url = env.get('SUPABASE_URL')
    key = env.get('SUPABASE_SERVICE_ROLE_KEY') or env.get('SUPABASE_KEY')
    if not url or not key:
        raise RuntimeError('Supabase URL/service credential required for completed college observations')
    rows = []
    for offset in range(0, 20000, 1000):
        response = requests.get(url.rstrip('/') + '/rest/v1/draft_ready_data',
            headers={'apikey': key, 'Authorization': 'Bearer ' + key},
            params={'select': ','.join(COLLEGE_FIELDS), 'year': 'eq.2026',
                    'order': 'id.asc', 'limit': 1000, 'offset': offset}, timeout=60)
        if response.status_code != 200:
            raise RuntimeError('Completed-college snapshot request failed, HTTP ' + str(response.status_code))
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError('College rows must be a list')
        rows.extend(batch)
        if len(batch) < 1000:
            break
    else:
        raise ValueError('College snapshot exceeds bounded 20000 rows')
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate college snapshot rows')
    _json(path, dict(table='public.draft_ready_data', year=2026, columns=COLLEGE_FIELDS,
                    fetched_at=datetime.now(timezone.utc).isoformat(), rows=rows,
                    draft_pick_used=False, impact_fields_selected=False))
    return path, rows


def _names(base, out, normalize):
    path = R / 'autocomplete_map.csv'
    names = pd.read_csv(path).drop_duplicates('nba_id').set_index('nba_id').player_name.copy()
    inputs = {str(path): _sha(path)}
    # Missing current identities can be supplied by the same admitted player objects.
    for p in sorted((base / 'objects/player').glob('*.json')):
        if not p.stem.isdigit():
            continue
        obj = json.loads(p.read_bytes())
        name = obj.get('name')
        if not isinstance(name, str) or not name.strip():
            continue
        pid = int(p.stem)
        if pid not in names.index:
            names.loc[pid] = name
            inputs[str(p)] = _sha(p)
    aliases = []
    if 1642905 in names.index:
        observed = names.loc[1642905]
        if normalize(observed) not in ('hansenyang', 'yanghansen'):
            raise ValueError('Known Yang/Hansen identity no longer matches player1642905')
        names.loc[1642905] = 'Yang Hansen'
        aliases.append(dict(player_id=1642905, canonical_observed=observed,
                            frozen_verified_name='Yang Hansen', rule='one explicit known-ID name-order alias'))
    copied = out / 'entrant_canonical_names.csv'
    raw = names.rename('player_name').rename_axis('nba_id').reset_index().to_csv(index=False).encode()
    if copied.exists():
        if copied.read_bytes() != raw:
            raise FileExistsError('Preserved canonical-name copy differs')
    else:
        copied.write_bytes(raw)
    inputs[str(copied)] = _sha(copied)
    return names, copied, aliases, inputs


def _career(base, cut):
    """Original pre2014 history + dated2014-26 + exact current native slot counts."""
    from vpm.harness.monthly_entrants import dated_history
    history_path = ENTRANTS / 'nba_history.parquet'
    history = pd.read_parquet(history_path)
    lines = pd.read_parquet(MONTHLY, columns=['player_id', 'season', 'game_id', 'game_date', 'off_poss', 'def_poss'])
    if history.season.max() > 2026 or lines.season.max() > 2026:
        raise ValueError('Historical career source unexpectedly includes current season')
    lines = lines.loc[lines.season.between(2014, 2026)].copy()
    # This original helper checks finite nonnegative exact unmasked counts and duplicates.
    career = dated_history(history, lines, cut)
    first = pd.concat([history.groupby('player_id').season.min(),
                       lines.loc[pd.to_datetime(lines.game_date, utc=True).lt(pd.Timestamp(cut).tz_localize('UTC'))]
                            .groupby('player_id').season.min()]).groupby(level=0).min()
    native_path = base / 'native_source/RAPM27_current.parquet'
    current = pd.read_parquet(native_path)
    if not set(SLOTS + ['game_id', 'game_date']) <= set(current):
        raise ValueError('Current career source lacks original ten-slot identities')
    dates = pd.to_datetime(current.game_date, utc=True)
    fence = pd.Timestamp(cut).tz_localize('UTC')
    if dates.isna().any():
        raise ValueError('Missing current native game date')
    current = current.loc[dates.lt(fence)].copy()
    if len(current) and not pd.to_datetime(current.game_date).dt.year.between(2026, 2027).all():
        raise ValueError('Current season native date fence differs')
    a = current[SLOTS].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if not np.isfinite(a).all() or (a <= 0).any() or not np.equal(a, np.floor(a)).all() or any(len(set(row)) != 10 for row in a):
        raise ValueError('Current career counts require canonical distinct ten-player native rows')
    if 'event_num' in current and current.duplicated(['game_id', 'event_num']).any():
        raise ValueError('Duplicate current native possession address')
    counts = pd.Series(a.ravel().astype(np.int64)).value_counts() / 2
    date_rows = pd.DataFrame({'player_id': a.ravel().astype(np.int64),
                              'last_game': np.repeat(pd.to_datetime(current.game_date, utc=True).to_numpy(), 10)})
    last = date_rows.groupby('player_id').last_game.max()
    now = pd.DataFrame({'player_id': counts.index, 'possessions': counts.values,
                        'last_game': last.reindex(counts.index).to_numpy()})
    career = pd.concat([career, now], ignore_index=True)
    current_first = pd.Series(2027, index=counts.index)
    first = pd.concat([first, current_first]).groupby(level=0).min()
    return career, first, counts, {str(history_path): _sha(history_path), str(MONTHLY): _sha(MONTHLY),
                                  str(native_path): _sha(native_path)}, dict(current_native_rows=len(current),
                                  current_native_games=int(current.game_id.nunique()),
                                  current_positive_players=len(counts), current_career_source='canonical ten-slot raw NBA possessions, independent of stat-admission masks')


def _add_college(sources, rows, names, ids, first, cut, normalize):
    """Exact unique full-name join only; never infer an actual NBA draft pick."""
    canonical = names.map(normalize)
    unique = canonical.loc[~canonical.duplicated(keep=False)]
    reverse = dict(zip(unique, unique.index.astype(int)))
    candidates = []
    rejected = []
    birthdates = {}
    for row in rows:
        if int(row['year']) != 2026:
            raise ValueError('Only completed2026 college rows were fetched')
        key = normalize(row['player'])
        pid = reverse.get(key)
        if pid is None or pid not in ids:
            continue
        explicit = row.get('nba_id')
        if explicit is not None and (not np.isfinite(float(explicit)) or int(explicit) != float(explicit) or int(explicit) != pid):
            rejected.append(dict(player_id=pid, source_row_id=row['id'], reason='conflicting_nonnull_nba_id'))
            continue
        if pd.Timestamp('2026-07-01') >= pd.Timestamp(cut) or 2026 >= first.get(pid, np.inf):
            continue
        birth = pd.to_datetime(row.get('birthdate'), errors='coerce')
        if pd.notna(birth):
            if not pd.Timestamp('1900-01-01') < birth < pd.Timestamp('2026-07-01'):
                rejected.append(dict(player_id=pid, source_row_id=row['id'], reason='invalid_birthdate'))
                continue
            birthdates[pid] = str(birth.date())
        candidates.append((pid, row))
    by_player = {}
    for pid, row in candidates:
        by_player.setdefault(pid, []).append(row)
    converted = []
    records = []
    for pid, group in sorted(by_player.items()):
        if len(group) != 1:
            rejected.append(dict(player_id=pid, source_row_ids=[r['id'] for r in group], reason='ambiguous_player_season_rows'))
            continue
        row = group[0]
        needed = ['games', 'mpg', 'two_m', 'two_a', 'three_m', 'three_a', 'ftm', 'fta',
                  'apg', 'topg', 'orpg', 'drpg', 'spg', 'bpg']
        values = pd.to_numeric(pd.Series({k: row.get(k) for k in needed}), errors='coerce')
        if (not np.isfinite(values.to_numpy()).all() or (values < 0).any()
                or values['games'] <= 0 or values['mpg'] <= 0
                or any(values[m] > values[a] for m, a in [('two_m', 'two_a'), ('three_m', 'three_a'), ('ftm', 'fta')])):
            rejected.append(dict(player_id=pid, source_row_id=row['id'], reason='missing_or_invalid_completed_college_counts'))
            continue
        f = dict(player_id=pid, season=2026, available_at=pd.Timestamp('2026-07-01', tz='UTC'),
                 minutes=float(row['games']) * float(row['mpg']))
        for key in ['two_m', 'two_a', 'three_m', 'three_a', 'ftm', 'fta']:
            f[key] = float(row[key]) if row[key] is not None else np.nan
        for raw, key in [('apg', 'ast'), ('topg', 'tov'), ('orpg', 'oreb'), ('drpg', 'dreb'), ('spg', 'stl'), ('bpg', 'blk')]:
            f[key] = float(row[raw]) * float(row['games']) if row[raw] is not None and row['games'] is not None else np.nan
        converted.append(f)
        records.append(dict(player_id=pid, source_row_id=row['id'], player=row['player'], birthdate=row['birthdate'],
                            identity='unique normalized full name including suffix; nonnull NBA ID consistent', available_at='2026-07-01'))
    if converted:
        fresh = pd.DataFrame(converted)
        sources.college = sources.college.loc[~(sources.college.season.eq(2026) & sources.college.player_id.isin(fresh.player_id))]
        sources.college = pd.concat([sources.college, fresh], ignore_index=True)
    return records, rejected, birthdates


def prepare_background(*, cut, base, reference, ids, out):
    """Return fixed component ability and compact-support entrant blend weights."""
    if str(V) not in sys.path:
        sys.path.insert(0, str(V))
    from vpm.harness.entrant_priors import load_sources, normalize_name, feature_packet, transformed_features
    from vpm.harness.monthly_vpm import entrant_weight
    from vpm.harness.quarterly_history import LANES, REFERENCES
    cut = pd.Timestamp(cut)
    if cut.tz is not None or pd.isna(cut) or cut != cut.normalize() or cut.year != 2026:
        raise ValueError('Explicit completed current2027 date in2026 required')
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids) or (ids <= 0).any():
        raise ValueError('Ordered unique positive integer player IDs required')
    base, out = Path(base).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names, copied, aliases, inputs = _names(base, out, normalize_name)
    sources = load_sources(ENTRANTS, copied)
    source_files = ['draft_identity_snapshot_20260926.json', 'college_snapshot_20260926.json',
                    'college_crosswalk_20260926.json', 'verified_draft_metadata.json',
                    'verified_identity_overrides.json', 'gleague_player_games.parquet']
    inputs.update({str(ENTRANTS / n): _sha(ENTRANTS / n) for n in source_files})
    career, first, current_counts, career_pins, career_meta = _career(base, cut)
    inputs.update(career_pins)
    snapshot, raw = _college_snapshot(out)
    inputs[str(snapshot)] = _sha(snapshot)
    college_records, rejected, births = _add_college(sources, raw, names, set(map(int, ids)), first, cut, normalize_name)
    # feature_packet retains all original masks, rates, decay550 and reliability500.
    packet = feature_packet(sources, career, ids, cut, first)
    model = json.loads(MODEL.read_bytes())
    if model['components'] != LANES + REFERENCES or model['parent']['full_own_possessions'] != 2000:
        raise ValueError('Frozen clean entrant schema/weight differs')
    if pd.Timestamp(model['parent']['fit_cut']) > cut or pd.Timestamp(model['parent']['fit_end']) >= cut:
        raise ValueError('Entrant fit is not past-only')
    features = transformed_features(packet, model['parent']['scaler'])
    p = features.shape[1]
    coef = np.asarray(model['coefficients'], float)
    if coef.shape != (2 * p, 11) or not np.isfinite(coef).all():
        raise ValueError('Frozen 11-component entrant coefficient shape differs')
    background = np.stack([features @ coef[:p], features @ coef[p:]], axis=1)
    parent = np.asarray(model['parent']['coef'], float)
    total = np.column_stack([features @ parent[:p], features @ parent[p:]])
    gap = float(np.max(np.abs(background[:, :, :len(LANES)].sum(axis=2) - total), initial=0))
    if gap > 1e-8 or not np.isfinite(background).all():
        raise ValueError('Frozen background scalar/component closure failed')
    exposure = packet.nba_possessions.to_numpy(float)
    earliest = first.reindex(ids).to_numpy(float)
    # Careers first seen at the1997 left boundary remain unknown below the
    # compact-support threshold. >=2000 is enough to identify weight0 exactly.
    identified = np.isfinite(earliest) & ((earliest > 1997) | (exposure >= 2000))
    for i, pid in enumerate(ids):
        if not identified[i] and int(pid) in births and pd.Timestamp(births[int(pid)]) > pd.Timestamp('1997-01-01'):
            identified[i] = True
    original_weights = entrant_weight(exposure, 2000.)
    weights = np.where(identified, original_weights, 0.)
    reported = np.where(identified, original_weights, np.nan)
    packet = packet.copy()
    packet['career_exposure_identified'] = identified
    packet['reported_background_weight'] = reported
    packet['applied_background_weight'] = weights
    packet['current_2027_nba_possessions'] = current_counts.reindex(ids, fill_value=0).to_numpy()
    inputs[str(MODEL)] = _sha(MODEL)
    for module in [Path(__file__), Path(sys.modules[load_sources.__module__].__file__),
                   Path(sys.modules[entrant_weight.__module__].__file__),
                   V / 'vpm/harness/monthly_entrants.py']:
        inputs[str(module.resolve())] = _sha(module)
    # The passed full source reference determines target identity only; its own
    # imputed/masked feature exposure is deliberately NOT career evidence.
    if reference.player_id.duplicated().any() or not set(map(int, ids)) <= set(reference.player_id.astype(int)):
        raise ValueError('Complete full-feature reference must cover target IDs once')
    packet_path = out / 'background_packet.parquet'
    if packet_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(packet_path), packet, check_exact=True)
    else:
        packet.to_parquet(packet_path, index=True)
    inputs[str(packet_path)] = _sha(packet_path)
    metadata = dict(estimate_date=str(cut.date()), model_path=str(MODEL), model_sha256=inputs[str(MODEL)],
        fit_cut=model['parent']['fit_cut'], fit_end=model['parent']['fit_end'], components=model['components'],
        full_own_possessions=2000, new_fits=0, new_features=0, draft_pick_from_fresh_college_used=False,
        frozen_draft_max_year=int(sources.draft.draft_year.max()),
        players=len(ids), weighted_players=int((weights > 0).sum()), unidentified_legacy_players=int((~identified).sum()),
        individualized_college_players=int(packet.college_covered.sum()), gleague_players=int(packet.gleague_covered.sum()),
        prior_sources=packet.prior_source.value_counts().to_dict(), maximum_background_scalar_gap=gap,
        career_rule='completed1997-2013 NBA history + dated2014-2026 source + current2027 canonical native ten-slot counts once; no stat-mask/TD weights',
        historical_left_boundary='1997 low-career unidentified: reportNaN and effective own-only; >=2000 possessions proves weight0',
        identity_aliases=aliases, fresh_college_matches=college_records, rejected_college_rows=rejected,
        fresh_college_snapshot=str(snapshot), current_career=career_meta,
        qualification='Fixed clean-v61 entrant component ability transported to full_split_raw own skill prior; owner-requested current blend, no refit or predictive-win assertion')
    _json(out / 'metadata.json', metadata)
    return dict(background_lanes=background, weights=weights, reported_weights=reported,
                identified=identified, packet=packet, input_hashes=inputs, metadata=metadata)

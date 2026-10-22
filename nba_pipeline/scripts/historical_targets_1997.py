"""Immutable 1997--2026 actual-point/Net target extension; no model fitting.

Existing 2000--2026 measurements are hash-verified links. Six early files are
rebuilt with actual FT via the unchanged canonical helper. Only independently
proved, exact season/phase/game source-ID aliases are translated in NEW atoms.
Incomplete/admin-ID rows stay visible in source files and are censored by the
public loader. Prices retain the raw first-chance population of retained games.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import historical_targets as H
from historical_targets_identities import build_identity, complete_rows
from vpm_net_lane_targets import compact, targets

ROOT = H.ROOT
OUT = ROOT / 'results/vpm_historical_targets_1997_v1'
OLD = H.OUT
CORE = ROOT / 'results/estimated_skills/full_history_v2/source/player_game'
SLOTS = H.SLOTS
YEARS = tuple(range(1997, 2027))
MANDATORY_EARLY_SCORE_EXCLUSIONS = (29600172, 29600370, 49600063)
ALIAS_CANDIDATES = ((471, 1489, 'Lionel Simmons', 'Simmons', 1610612758),
                    (775, 511, 'Melvin Booker', 'Booker', 1610612744))
COUNT_FIELDS = ('pts', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'tov')


def write_once(path, value):
    path = Path(path)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Immutable extension artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def read(path):
    return json.loads(Path(path).read_text())


def verify_pins(pins):
    for name, digest in pins.items():
        if not Path(name).is_absolute() or H.sha(name) != digest:
            raise ValueError('Frozen target-extension input changed: ' + name)


def raw_counts(frame):
    """Independent integer made-FG/FT accounting; MISS is a word token."""
    frame = frame.drop_duplicates()
    event = pd.to_numeric(frame.event_type, errors='raise')
    text = frame.home_description.fillna('') + ' ' + frame.visitor_description.fillna('')
    three = text.str.contains('3PT', case=False, regex=False)
    made = event.eq(1)
    ft = event.eq(3)
    hit = ft & ~text.str.contains(r'\bMISS\b', case=False, regex=True)
    return dict(pts=int((made * np.where(three, 3, 2) + hit).sum()),
                fga=int(event.isin([1, 2]).sum()), fgm=int(made.sum()),
                fg3a=int((event.isin([1, 2]) & three).sum()),
                fg3m=int((made & three).sum()), fta=int(ft.sum()),
                ftm=int(hit.sum()), tov=int(event.eq(5).sum()))


def prove_alias(raw, core, official, source_id, canonical_id, name, surname, team_id):
    """Return exact game-scoped aliases only after independent identity/count QA.

    The archived core is a separately parsed native game/player source. Its
    possession counts are NOT copied into targets or used to repair geometry.
    """
    official = official.loc[pd.to_numeric(official.GP).gt(0)]
    identity = official.loc[official.PLAYER_ID.eq(canonical_id)]
    if (len(identity) != 1 or identity.PLAYER_NAME.iloc[0] != name
            or int(identity.TEAM_ID.iloc[0]) != team_id
            or source_id in set(official.PLAYER_ID)):
        raise ValueError('Alias lacks exact official annual identity/team proof')
    # The surname must be unique in the official year, never a fuzzy name join.
    surnames = official.PLAYER_NAME.str.split().str[-1]
    if int(surnames.eq(surname).sum()) != 1:
        raise ValueError('Ambiguous official alias surname')
    source = raw.loc[pd.to_numeric(raw.player1_id).eq(source_id)]
    if source.empty:
        raise ValueError('Expected source alias has no actor evidence')
    games = sorted(map(int, source.game_id.unique()))
    records = []
    for game in games:
        s = source.loc[pd.to_numeric(source.game_id).eq(game)]
        if set(pd.to_numeric(s.player1_team_id).dropna().astype(int)) != {team_id}:
            raise ValueError('Alias actor team is not independently identified')
        text = s.home_description.fillna('') + ' ' + s.visitor_description.fillna('')
        shots = pd.to_numeric(s.event_type).isin([1, 2, 3])
        if not shots.any() or not text.loc[shots].str.contains(r'\b' + re.escape(surname) + r'\b', regex=True).all():
            raise ValueError('Alias lacks exact surname on all shooting actions')
        c = core.loc[pd.to_numeric(core.game_id).eq(game) & core.person_id.eq(canonical_id)]
        wrong = core.loc[pd.to_numeric(core.game_id).eq(game) & core.person_id.eq(source_id)]
        if (len(c) != 1 or len(wrong) or int(c.team_id.iloc[0]) != team_id
                or not bool(c.lineups_clean.iloc[0]) or int(c.provably_wrong_rows.iloc[0])
                or int(c.rows_without_10_players.iloc[0])):
            raise ValueError('Alias lacks clean exact canonical native game/player key')
        if int(c.season.iloc[0]) != 1997 or bool(c.playoffs.iloc[0]):
            raise ValueError('Alias proof leaves exact 1997 RS epoch')
        dates = pd.to_datetime(s.game_date).dt.normalize().unique()
        native_date = pd.to_datetime(str(int(c.date.iloc[0])), format='%Y%m%d')
        if len(dates) != 1 or pd.Timestamp(dates[0]) != native_date:
            raise ValueError('Alias game dates differ')
        # Sum both source aliases if the raw game contains canonical-ID actions.
        combined = raw.loc[pd.to_numeric(raw.game_id).eq(game)
                           & pd.to_numeric(raw.player1_id).isin([source_id, canonical_id])]
        counts = raw_counts(combined)
        if any(counts[k] != int(c[k].iloc[0]) for k in COUNT_FIELDS):
            raise ValueError('Alias independent shooting/FT/TOV counts differ')
        candidates = core.loc[pd.to_numeric(core.game_id).eq(game) & core.team_id.eq(team_id)]
        exact = np.ones(len(candidates), dtype=bool)
        for k in COUNT_FIELDS:
            exact &= candidates[k].to_numpy() == counts[k]
        if int(exact.sum()) != 1:
            raise ValueError('Alias count signature is not unique in native team/game')
        records.append(dict(year=1997, phase='RS', game_id=game, source_id=source_id,
            canonical_id=canonical_id, canonical_name=name, team_id=team_id,
            counts=counts, native_counts_exact=True, native_clean_census=True,
            game_date=str(native_date.date()), source_actor_rows=len(s),
            substitution_actor_rows=int(pd.to_numeric(s.event_type).eq(8).sum()),
            official_identity_and_team_exact=True, admitted=True))
    return records


def apply_aliases(frame, year, phase, packet):
    """Exact scope only; does not repair or fill any other slot/count."""
    result = frame.copy()
    for rec in packet['aliases']:
        if rec['year'] != year or rec['phase'] != phase or not rec['admitted']:
            continue
        use = pd.to_numeric(result.game_id).eq(rec['game_id'])
        result.loc[use, SLOTS] = result.loc[use, SLOTS].replace(rec['source_id'], rec['canonical_id'])
    # A translation cannot merge two simultaneously represented players.
    before = complete_rows(frame[SLOTS].to_numpy())
    after = complete_rows(result[SLOTS].to_numpy())
    if (before & ~after).any():
        raise ValueError('Admitted alias creates a duplicate/colliding lineup')
    return result


def identity_filter(atoms, games, identity):
    keep = complete_rows(atoms[SLOTS].to_numpy())
    for year, indices in atoms.groupby('source_season_end_year').indices.items():
        allowed = identity['seasons'][str(int(year))]['admitted_ids']
        keep[indices] &= np.isin(atoms.iloc[indices][SLOTS].to_numpy(), allowed).all(axis=1)
    data = atoms.loc[keep].copy().reset_index(drop=True)
    keys = data[H.KEYS].drop_duplicates()
    pricing = games.merge(keys, on=H.KEYS, how='inner', validate='one_to_one')
    if not data.empty:
        H.validate_population(data, pricing)
    elif not pricing.empty:
        raise ValueError('Identity census has pricing without admitted targets')
    return data, pricing, keep


def source_files():
    files = {Path(__file__).resolve(), Path(H.__file__).resolve(),
             ROOT/'scripts/historical_targets_identities.py',
             ROOT/'scripts/export_vpm_net_lane_atoms.py', ROOT/'scripts/vpm_net_lane_targets.py',
             ROOT/'tests/test_historical_targets_1997.py', H.QUARANTINE}
    files.update((ROOT/'scripts/process_rapm_blocks').rglob('*.py'))
    files.update(CORE.glob('*.parquet'))
    for year in YEARS:
        for phase, suffix, sheet_suffix in [('RS', '', ''), ('PS', '_PS', '_ps')]:
            files.update([ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet',
                          ROOT/f'processed/RAPM{year%100:02d}{suffix}.parquet',
                          ROOT.parent/f'player_sheets/year_totals/{year}{sheet_suffix}.csv'])
            if year >= 2000:
                files.add(OLD/f'historical_{year}_{phase}.json')
                for name in read(OLD/f'historical_{year}_{phase}.json')['artifacts']:
                    files.add(OLD/name)
    files.update([OLD/'manifest.json', OLD/'identity_admission.json'])
    return sorted({p.resolve() for p in files})


def source_census_row(raw_path, canonical_path, year, phase):
    # Modern raw event schemas need not contain dates. Canonical dates are the
    # target clock and must be present; raw and canonical game sets must agree.
    rr = pd.read_parquet(raw_path, columns=['game_id'])
    cc = pd.read_parquet(canonical_path, columns=['game_id', 'game_date'])
    raw_games = set(pd.to_numeric(rr.game_id)); canonical_games = set(pd.to_numeric(cc.game_id))
    dates = pd.to_datetime(cc.game_date)
    if raw_games != canonical_games or dates.isna().any():
        raise ValueError('Raw/canonical game census or dates differ')
    return dict(year=year, phase=phase, raw_rows=len(rr), canonical_rows=len(cc),
        raw_games=len(raw_games), canonical_games=len(canonical_games),
        first_date=str(dates.min().date()),last_date=str(dates.max().date()),
        game_date_source='canonical_RAPM_exact_target_clock',
        source_census_complete_season_claim=False)


def preflight(out=OUT):
    out = Path(out).resolve()
    if out == OLD.resolve() or OLD.resolve() in out.parents:
        raise ValueError('New extension root must not modify the old target package')
    if (out/'registration.json').exists():
        registered = read(out/'registration.json'); verify_pins(registered['input_hashes'])
        return registered
    out.mkdir(parents=True, exist_ok=True)
    inputs = {str(p): H.sha(p) for p in source_files()}
    for year in range(2000,2027):
        for phase in ['RS','PS']:
            verify_pins(read(OLD/f'historical_{year}_{phase}.json')['source_files'])
    identity = build_identity(years=YEARS)
    original_identity = read(OLD/'identity_admission.json')
    for year in range(2000, 2027):
        if identity['seasons'][str(year)] != original_identity['seasons'][str(year)]:
            raise ValueError('Existing historical annual identity packet changed')
    raw = pd.read_parquet(ROOT/'raw_data/NBA97.parquet')
    core = pd.concat([pd.read_parquet(p, filters=[('season', '=', 1997)])
                      for p in sorted(CORE.glob('*.parquet'))], ignore_index=True)
    official = pd.read_csv(ROOT.parent/'player_sheets/year_totals/1997.csv')
    aliases = []
    for candidate in ALIAS_CANDIDATES:
        aliases.extend(prove_alias(raw, core, official, *candidate))
    alias_packet = dict(status='passed_exact_independent_early_alias_admission', aliases=aliases,
        policy='exact1997RSgame/sourceID only; raw shooting/FT/TOV signature + dated native canonical key + official unique identity/team; admin IDs never inferred',
        counts_or_exposures_spliced=False, input_hashes={n:h for n,h in inputs.items()
            if '/full_history_v2/source/player_game/' in n or n.endswith('/NBA97.parquet')
            or n.endswith('/1997.csv') or n.endswith('/historical_targets_1997.py')})
    write_once(out/'alias_admission.json', alias_packet)
    write_once(out/'identity_source_packet.json', identity)
    census = []
    for year in YEARS:
        for phase, suffix in [('RS', ''), ('PS', '_PS')]:
            rp = ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet'
            cp = ROOT/f'processed/RAPM{year%100:02d}{suffix}.parquet'
            census.append(source_census_row(rp,cp,year,phase))
    write_once(out/'preflight_source_census.json', dict(status='passed_raw_canonical_60_file_census', files=census))
    for p in [out/'alias_admission.json', out/'identity_source_packet.json', out/'preflight_source_census.json']:
        inputs[str(p)] = H.sha(p)
    registered = dict(schema_version=1, status='frozen_1997_target_extension_before_build',
        years=list(YEARS), phases=['RS', 'PS'], input_hashes=inputs,
        old_target_root=str(OLD.resolve()), source_alias_scope='exact admitted game/year/phase/sourceID',
        mandatory_early_score_exclusions=list(MANDATORY_EARLY_SCORE_EXCLUSIONS),
        actual_FT=True, earlier_SQ_inactive=True, clean_SQ_supported_from=2016,
        raw_source_immutable=True, selected_vpm_changed=False, protected_2027_outcomes=0)
    verify_pins(inputs)
    write_once(out/'registration.json', registered)
    return registered


def early_build(args):
    year, phase, out = args; out = Path(out)
    reg = read(out/'registration.json')
    packet = read(out/'alias_admission.json')
    destination = out/f'historical_{year}_{phase}.json'
    if destination.exists():
        meta = read(destination)
        for name, digest in meta['artifacts'].items():
            if H.sha(out/name) != digest: raise ValueError('Existing early artifact changed')
        return meta
    unaliased = out/'raw_measurements'
    source = H.build(year, phase, unaliased)
    p = unaliased/f'atoms_{year}_{phase}.parquet'
    atoms = apply_aliases(pd.read_parquet(p), year, phase, packet)
    games = pd.read_parquet(unaliased/f'games_{year}_{phase}.parquet')
    excluded = sorted(set(source['excluded_game_ids']) | set(MANDATORY_EARLY_SCORE_EXCLUSIONS))
    good = atoms.loc[~pd.to_numeric(atoms.game_id).isin(excluded)].reset_index(drop=True)
    gg = games.loc[~pd.to_numeric(games.game_id).isin(excluded)].reset_index(drop=True)
    H.validate_population(good, gg)
    y, _ = targets(good, gg)
    if np.max(np.abs(compact(y).sum(axis=1) - good.actual_points)) > 1e-10:
        raise ValueError('Early standard Net row algebra does not close')
    ap = out/f'atoms_{year}_{phase}.parquet'; gp = out/f'games_{year}_{phase}.parquet'
    if ap.exists() or gp.exists(): raise ValueError('Partial early product must be inspected; no overwrite')
    atoms.to_parquet(ap, index=False); gp.symlink_to((unaliased/gp.name).resolve())
    meta = {**source, 'admitted_rows':len(good), 'admitted_games':int(good.game_id.nunique()),
        'excluded_game_ids':excluded, 'artifacts':{ap.name:H.sha(ap),gp.name:H.sha(gp)},
        'lineage':dict(kind='new_actual_FT_measurements_with_exact_independent_identity_aliases',
            unaliased_receipt=str((unaliased/destination.name).resolve()),
            unaliased_receipt_sha256=H.sha(unaliased/destination.name)),
        'input_hashes':{str(out/'registration.json'):H.sha(out/'registration.json'),
            str(out/'alias_admission.json'):H.sha(out/'alias_admission.json'),
            str((unaliased/destination.name).resolve()):H.sha(unaliased/destination.name)},
        'target_counts_or_exposure_replaced_from_native_core':False}
    write_once(destination, meta)
    return meta


def build(out=OUT, workers=1):
    out = Path(out).resolve(); reg = read(out/'registration.json'); verify_pins(reg['input_hashes'])
    for year in range(2000, 2027):
        for phase in ['RS', 'PS']:
            rp = OLD/f'historical_{year}_{phase}.json'; meta = read(rp)
            for name, digest in meta['artifacts'].items():
                if H.sha(OLD/name) != digest: raise ValueError('Old target artifact changed')
            for original in [rp, *[OLD/name for name in meta['artifacts']]]:
                destination = out/original.name
                if destination.exists():
                    if H.sha(destination) != H.sha(original): raise ValueError('Reused historical target differs')
                else: destination.symlink_to(original.resolve())
    tasks = [(y, p, str(out)) for y in range(1997, 2000) for p in ['RS', 'PS']]
    if workers == 1:
        results = [early_build(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(early_build, tasks))
    verify_pins(reg['input_hashes'])
    print(json.dumps(dict(status='six_new_actual_measurement_pairs_built', files=len(results))), flush=True)


def audit(out=OUT):
    out = Path(out).resolve(); reg = read(out/'registration.json'); verify_pins(reg['input_hashes'])
    identity = read(out/'identity_source_packet.json'); aliases = read(out/'alias_admission.json')
    files = []; outputs = {}; maxgap = 0.; max_weighted_gap = 0.
    for year in YEARS:
        for phase, suffix in [('RS', ''), ('PS', '_PS')]:
            rp = out/f'historical_{year}_{phase}.json'; meta = read(rp)
            for name, digest in meta['artifacts'].items():
                if H.sha(out/name) != digest: raise ValueError('Historical extension artifact changed')
                outputs[str(out/name)] = digest
            outputs[str(rp)] = H.sha(rp)
            a = pd.read_parquet(out/f'atoms_{year}_{phase}.parquet')
            g = pd.read_parquet(out/f'games_{year}_{phase}.parquet')
            canonical = pd.read_parquet(ROOT/f'processed/RAPM{year%100:02d}{suffix}.parquet',
                columns=['game_id','event_num','game_date',*SLOTS])
            reference = apply_aliases(canonical,year,phase,aliases)
            for c in ['game_id','event_num',*SLOTS]:
                np.testing.assert_array_equal(pd.to_numeric(a[c]),pd.to_numeric(reference[c]))
            np.testing.assert_array_equal(pd.to_datetime(a.game_date),pd.to_datetime(reference.game_date))
            H.validate_population(a,g)
            if year < 2000:
                # Independently repeat raw-made-event accounting after the build.
                raw = pd.read_parquet(ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet')
                accounting = H.compare_scores(raw,a)
                bad = set(pd.to_numeric(accounting.loc[~accounting.accepted,'game_id']).astype(int))
                if not bad <= set(meta['excluded_game_ids']): raise ValueError('Unquarantined raw/actual score mismatch')
                if not set(MANDATORY_EARLY_SCORE_EXCLUSIONS) <= set(meta['excluded_game_ids']):
                    raise ValueError('Mandatory raw scoreboard conflicts lost')
            use = ~pd.to_numeric(a.game_id).isin(meta['excluded_game_ids'])
            aa = a.loc[use].reset_index(drop=True)
            gg = g.loc[~pd.to_numeric(g.game_id).isin(meta['excluded_game_ids'])].reset_index(drop=True)
            H.validate_population(aa,gg)
            complete = complete_rows(aa[SLOTS].to_numpy())
            d,prices,retained = identity_filter(aa,gg,identity)
            y,_ = targets(d,prices)
            gap=float(np.max(np.abs(compact(y).sum(axis=1)-d.actual_points)))
            maxgap=max(maxgap,gap)
            w=np.exp2(-(pd.to_datetime(d.game_date).max()-pd.to_datetime(d.game_date)).dt.total_seconds().to_numpy()/(86400*550))
            yw,_=targets(d,prices,w)
            wg=float(np.max(np.abs(compact(yw).sum(axis=1)-d.actual_points)))
            max_weighted_gap=max(max_weighted_gap,wg)
            if max(gap,wg)>1e-10: raise ValueError('Historical Net lane closure failed')
            lost=sorted(set(pd.to_numeric(aa.game_id))-set(pd.to_numeric(d.game_id)))
            files.append(dict(year=year,phase=phase,raw_target_rows=len(a),raw_target_games=int(a.game_id.nunique()),
                scoring_admitted_rows=len(aa),scoring_admitted_games=int(aa.game_id.nunique()),
                scoring_excluded_game_ids=sorted(set(pd.to_numeric(a.game_id))&set(meta['excluded_game_ids'])),
                complete_positive_5v5_rows=int(complete.sum()),
                identity_rejected_complete_rows=int((complete&~retained).sum()),
                retained_rows=len(d),retained_games=int(d.game_id.nunique()),
                whole_games_removed_by_identity=list(map(int,lost)),
                first_date=str(pd.to_datetime(d.game_date).min().date()),last_date=str(pd.to_datetime(d.game_date).max().date()),
                maximum_standard_Net_gap=gap,maximum_TD550_standard_Net_gap=wg))
    census=dict(status='passed_60_file_actual_Net_identity_source_census',files=files,
        raw_rows=sum(x['raw_target_rows'] for x in files),raw_games=sum(x['raw_target_games'] for x in files),
        scoring_admitted_rows=sum(x['scoring_admitted_rows'] for x in files),
        scoring_admitted_games=sum(x['scoring_admitted_games'] for x in files),
        retained_rows=sum(x['retained_rows'] for x in files),retained_games=sum(x['retained_games'] for x in files),
        whole_games_removed_by_identity=sum(len(x['whole_games_removed_by_identity']) for x in files),
        maximum_standard_Net_gap=maxgap,maximum_TD550_standard_Net_gap=max_weighted_gap,
        input_hashes={str(out/'registration.json'):H.sha(out/'registration.json')},output_hashes=outputs,
        source_census_complete_season_claim=False,protected_2027_outcomes=0)
    write_once(out/'source_identity_census.json',census)
    identity={**identity,'status':'passed','census':census,
        'alias_admission':str(out/'alias_admission.json'),'alias_admission_sha256':H.sha(out/'alias_admission.json')}
    write_once(out/'identity_admission.json',identity)
    outputs.update({str(out/n):H.sha(out/n) for n in ['alias_admission.json','identity_source_packet.json',
        'preflight_source_census.json','source_identity_census.json','identity_admission.json']})
    verify_pins(reg['input_hashes'])
    manifest=dict(status='passed_1997_2026_actual_Net_target_extension',schema_version=1,
        years=list(YEARS),phases=['RS','PS'],files=60,new_measurement_pairs=6,hash_identical_reused_pairs=54,
        aliases_admitted=len(aliases['aliases']),raw_rows=census['raw_rows'],raw_games=census['raw_games'],
        retained_rows=census['retained_rows'],retained_games=census['retained_games'],
        maximum_standard_Net_gap=maxgap,maximum_TD550_standard_Net_gap=max_weighted_gap,
        input_hashes={str(out/'registration.json'):H.sha(out/'registration.json')},output_hashes=outputs,
        no_native_counter_splice=True,actual_FT=True,clean_SQ_supported_from=2016,earlier_SQ_inactive=True,
        no_mapper_fit=True,selected_vpm_changed=False,protected_2027_outcomes=0)
    write_once(out/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['input_hashes','output_hashes']}),flush=True)
    return manifest


def load_before_cut(root,cut,years,**kwargs):
    """The shared target/feature identity contract; never fit source/admin IDs."""
    root=Path(root).resolve();manifest=read(root/'manifest.json')
    if manifest['status']!='passed_1997_2026_actual_Net_target_extension':
        raise ValueError('Completed audited historical target extension required')
    for path,digest in manifest['input_hashes'].items():
        if H.sha(path)!=digest:raise ValueError('Historical extension registration changed')
    years=list(years)
    if not years or not set(years)<=set(YEARS):raise ValueError('Only1997–2026 historical targets admitted')
    identity=read(root/'identity_admission.json')
    if H.sha(root/'identity_admission.json')!=manifest['output_hashes'][str(root/'identity_admission.json')]:
        raise ValueError('Historical identity admission changed')
    atoms,games=H.load_before_cut(root,cut,years,**kwargs)
    atoms,games,_=identity_filter(atoms,games,identity)
    return atoms,games


target_matrix=H.target_matrix
validate_population=H.validate_population


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=OUT)
    ap.add_argument('--years',required=True,help='Explicit full1997,1998,...,2026 source registry')
    ap.add_argument('--stage',choices=['preflight','build','audit','all'],default='all')
    ap.add_argument('--workers',type=int,default=1)
    args=ap.parse_args();years=tuple(map(int,args.years.split(',')))
    if years!=YEARS or not 1<=args.workers<=2:raise ValueError('Exact completed1997–2026 registry and1–2workers required')
    if args.stage in ['preflight','all']:preflight(args.root)
    if args.stage in ['build','all']:build(args.root,args.workers)
    if args.stage in ['audit','all']:audit(args.root)


if __name__=='__main__':main()

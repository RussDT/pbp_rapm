"""Past-only game-date defensive queries of frozen selected-library recipes.

Retain original, validated tracking measurement denominators. The new core is
the query/identity census, not permission to reinterpret a tracking numerator.
Missing history stays missing; mapper-aware aggregate fallbacks are downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_monthly_own_skills as C
from build_pregame_neutral_skills import compile_kernel, replay

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / 'results/estimated_skills'
OUT = SKILLS / 'pregame_defensive_v2'
SOURCE = SKILLS / 'pregame_source_validated_v1'
DEFENSE = SKILLS / 'monthly_defense_v1'
MEASUREMENTS = SKILLS / 'defense_bank_v51_shooting_pf_repaired/measurements.parquet'
TRACKING = ['def_' + x + '100' for x in ['deflections', 'contested_dreb',
    'contested_shots', 'charges_drawn', 'off_fouls_drawn', 'shooting_pf', 'nonoff_pf']]
KEY = ['player_id', 'game_id']


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def normalize(frame):
    d = frame.copy()
    d['game_id'] = d.game_id.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    d['game_date'] = pd.to_datetime(d.game_date)
    if d.game_date.ge('2026-09-01').any() or ('season' in d and d.season.gt(2026).any()):
        raise ValueError('Unexpected protected/future source')
    return d


def reconcile(measurements, census, denominator):
    """Exact dated identities and positive census exposure; retain source E."""
    d = measurements.merge(census[KEY + ['game_date', 'def_poss']], on=KEY,
        how='left', validate='many_to_one', suffixes=('', '_census'), indicator=True)
    matched = d._merge.eq('both')
    if (matched & d.game_date.ne(d.game_date_census)).any():
        raise ValueError('Measurement date differs from validated census')
    e = 'def_poss_census' if denominator == 'def_poss' else 'def_poss'
    keep = matched & d[e].gt(0) & np.isfinite(d[e])
    audit = dict(rows=len(d), missing_identity_rows=int((~matched).sum()),
        unsafe_census_exposure_rows=int((matched & ~keep).sum()),
        differing_positive_denominators=int((keep & d[denominator].ne(d[e])).sum()),
        retained_rows=int(keep.sum()), denominator_rule='original_selected_measurement_denominator')
    drop = ['game_date_census', '_merge'] + ([e] if e != denominator else [])
    return d.loc[keep].drop(columns=drop), audit


def eb_queries(queries, observations, half_life=550., prior=1000.):
    """Prefix-only decayed ratio with common prefix population shrinkage.

    Query and observation arrays share exact identities. Every date is queried
    before any same-date updates. Query-only rows never advance a player state.
    """
    q = queries.reset_index(drop=True)
    z = q[KEY + ['game_date']].merge(observations[KEY + ['numerator', 'denominator']],
        on=KEY, how='left', validate='one_to_one')
    ids, unique = pd.factorize(q.player_id)
    n = np.zeros(len(unique)); e = n.copy(); mass = n.copy()
    last = np.full(len(unique), -1, dtype=int)
    values = np.full(len(q), np.nan); ex = values.copy(); last_out = np.full(len(q), -1, dtype=int)
    days = (q.game_date - pd.Timestamp('1970-01-01')).dt.days.to_numpy()
    num, den = z.numerator.to_numpy(float), z.denominator.to_numpy(float)
    valid = np.isfinite(num) & (num >= 0) & np.isfinite(den) & (den > 0)
    ln = le = 0.; ld = None
    order = np.argsort(days, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(days[order])) + 1)
    for index in groups:
        day = days[index[0]]
        if ld is not None:
            decay = np.exp2(-(day-ld)/half_life); ln *= decay; le *= decay
        p = ids[index]; decay = np.where(last[p] >= 0, np.exp2(-(day-last[p])/half_life), 0.)
        pn, pe = n[p]*decay, e[p]*decay
        league = ln/le if le > 0 else np.nan
        values[index] = np.where(pe > 0, 100*(pn+prior*league)/(pe+prior), np.nan)
        ex[index] = pe; last_out[index] = last[p]
        take = index[valid[index]]
        if len(take):
            players = np.unique(ids[take])
            decay = np.where(last[players] >= 0, np.exp2(-(day-last[players])/half_life), 0.)
            n[players] *= decay; e[players] *= decay
            np.add.at(n, ids[take], num[take]); np.add.at(e, ids[take], den[take])
            np.add.at(mass, ids[take], den[take]); last[players] = day
            ln += num[take].sum(); le += den[take].sum()
        ld = day
    return values, ex, last_out, valid


def rim_queries(queries, observations, model):
    """Selected decayed contest volume × lagged-league finishing suppression."""
    q = queries.reset_index(drop=True)
    z = q[KEY].merge(observations[KEY + ['rim_dfga', 'rim_dfgm', 'def_poss']], on=KEY,
        how='left', validate='one_to_one')
    ids, unique = pd.factorize(q.player_id)
    vn = np.zeros(len(unique)); ve = vn.copy(); sn = vn.copy(); sr = vn.copy()
    last = np.full(len(unique), -1, dtype=int)
    days = (q.game_date-pd.Timestamp('1970-01-01')).dt.days.to_numpy()
    a, m, e = (z[c].to_numpy() for c in ['rim_dfga', 'rim_dfgm', 'def_poss'])
    valid = np.isfinite(a) & np.isfinite(m) & np.isfinite(e) & (a >= m) & (m >= 0) & (e > 0)
    out = np.full((len(q), 5), np.nan); last_out = np.full(len(q), -1, dtype=int)
    total_a = total_m = 0.
    order = np.argsort(days, kind='stable')
    for index in np.split(order, np.flatnonzero(np.diff(days[order])) + 1):
        day = days[index[0]]; p = ids[index]
        qleague = (total_m+65)/(total_a+100)
        vd = np.where(last[p] >= 0, np.exp2(-(day-last[p])/model['volume_half']), 0.)
        sd = np.where(last[p] >= 0, np.exp2(-(day-last[p])/model['selected']['half_life']), 0.)
        volume = (vn[p]*vd+1000*model['mean'])/(ve[p]*vd+1000)
        suppress = np.clip(sr[p]*sd/(sn[p]*sd+model['selected']['prior_attempts']), qleague-1, qleague)
        seen = last[p] >= 0
        out[index, 0] = np.where(seen, 100*volume, np.nan)
        out[index, 1] = np.where(seen, suppress, np.nan)
        out[index, 2] = np.where(seen, 200*volume*suppress, np.nan)
        out[index, 3] = ve[p]*vd; out[index, 4] = sn[p]*sd; last_out[index] = last[p]
        take = index[valid[index]]
        if len(take):
            players = np.unique(ids[take]); gap = np.where(last[players] >= 0, day-last[players], 0.)
            vd, sd = np.exp2(-gap/model['volume_half']), np.exp2(-gap/model['selected']['half_life'])
            vn[players] *= vd; ve[players] *= vd; sn[players] *= sd; sr[players] *= sd
            np.add.at(vn, ids[take], a[take]); np.add.at(ve, ids[take], e[take])
            np.add.at(sn, ids[take], a[take]); np.add.at(sr, ids[take], qleague*a[take]-m[take])
            last[players] = day
            total_a += a[take].sum(); total_m += m[take].sum()
    return out, last_out, valid


def mpg_queries(queries):
    """Mean positive RS appearance minutes in [date−365days,date)."""
    q = queries.reset_index(drop=True)
    values = np.full(len(q), np.nan); games = np.zeros(len(q), dtype=int)
    last_out = np.full(len(q), -1, dtype=int)
    days = (q.game_date-pd.Timestamp('1970-01-01')).dt.days.to_numpy()
    for _, g in q.groupby('player_id', sort=False):
        index = g.index.to_numpy(); dates = days[index]
        valid = g.season_type.str.lower().eq('rs') & np.isfinite(g.minutes) & g.minutes.gt(0)
        measured_days = dates[valid]; minutes = g.loc[valid,'minutes'].to_numpy()
        if not len(measured_days): continue
        sums = np.r_[0., minutes.cumsum()]
        end = np.searchsorted(measured_days, dates, side='left')
        start = np.searchsorted(measured_days, dates-365, side='left')
        count = end-start; seen = count > 0
        values[index[seen]] = (sums[end[seen]]-sums[start[seen]])/count[seen]
        games[index] = count
        last_out[index[seen]] = measured_days[end[seen]-1]
    return values, games, last_out


def source_inputs(source):
    q = normalize(pd.read_parquet(Path(source)/'player_game_lines.parquet'))
    q = q.sort_values(['player_id', 'game_date', 'game_id'], kind='stable').reset_index(drop=True)
    if q.duplicated(KEY).any():
        raise ValueError('Duplicate validated query identity')
    rim, ra = reconcile(normalize(pd.read_parquet(DEFENSE/'rim_counts.parquet')), q, 'def_poss')
    dfga, da = reconcile(normalize(pd.read_parquet(DEFENSE/'dfga_counts.parquet')), q, 'def_poss')
    dfga = dfga.loc[dfga.dfga_valid & dfga.dfga_reason.eq('observed')].copy()
    measured, ma = reconcile(normalize(pd.read_parquet(MEASUREMENTS)), q, 'denominator')
    return q, rim, dfga, measured, dict(rim=ra, dfga=da, tracking=ma)


def build_year(out, year, q, rim, dfga, measured, snapshot_cut=None, snapshot_ids=None, model_file=None):
    cut, end = C.vintage_cut(year), C.vintage_cut(year+1)
    if year == 2025:
        end = pd.Timestamp('2026-09-01')
    if snapshot_cut is not None:
        snapshot_cut=pd.Timestamp(snapshot_cut)
        if snapshot_cut < cut or snapshot_cut > pd.Timestamp('2026-09-01'):
            raise ValueError('Snapshot outside legal parameter/protected date fence')
        q=q.loc[q.game_date.lt(snapshot_cut)].copy()
        extra=pd.DataFrame(dict(player_id=np.asarray(snapshot_ids,dtype=int),game_id='__snapshot__',
            game_date=snapshot_cut,season=snapshot_cut.year+int(snapshot_cut.month>=9),
            season_type='query',minutes=np.nan))
        q=pd.concat([q,extra],ignore_index=True).sort_values(['player_id','game_date','game_id'],kind='stable').reset_index(drop=True)
    else:
        q = q.loc[q.game_date.lt(end)].copy().reset_index(drop=True)
    model_path = Path(model_file) if model_file else DEFENSE/'models'/f'{year}.json'
    model = json.loads(model_path.read_text())
    if pd.Timestamp(model['fit_cut']) != cut or any(pd.Timestamp(model[k]['fit_end']) >= cut for k in ['rim', 'dfga']):
        raise ValueError('Illegal defensive parameter vintage')
    take = q.game_id.eq('__snapshot__') if snapshot_cut is not None else q.game_date.ge(cut)
    result = q.loc[take, KEY + ['game_date', 'season']].copy()
    def add(name, value, exposure, last, observed):
        result[name] = np.asarray(value)[take]
        result[name+'_missing'] = ~np.isfinite(result[name])
        result[name+'__prior_exposure'] = np.asarray(exposure)[take]
        result[name+'__last_observation_day'] = np.asarray(last)[take]
        if observed is not None:
            result[name+'__observed_this_game'] = np.asarray(observed)[take]
    for name in TRACKING:
        z = measured.loc[measured.feature.eq(name)]
        value, exposure, last, valid = eb_queries(q, z)
        add(name, value, exposure, last, valid)
    arr, last, valid = rim_queries(q, rim, model['rim'])
    for name, col, exposure in [('own_rim_contests100',0,3), ('own_rim_suppression',1,4), ('own_rim_saved100',2,4)]:
        add(name, arr[:, col], arr[:, exposure], last, valid)
    z = q[KEY].merge(dfga[KEY + ['dfga_value', 'def_poss']], on=KEY, how='left', validate='one_to_one')
    valid = np.isfinite(z.dfga_value) & z.dfga_value.ge(0) & z.def_poss.gt(0)
    frame = q[KEY+['season']].copy()
    frame['day'] = (q.game_date-pd.Timestamp('1970-01-01')).dt.days
    frame['exposure'] = z.def_poss.where(valid,0.)
    frame['obs_rate'] = z.dfga_value.div(z.def_poss).where(valid)
    mean = model['dfga']['mean']; params = model['dfga']['params']; zeros = np.zeros(len(q))
    a = replay(frame, params, mean, np.full(len(q),mean), zeros, zeros, compile_kernel(out))
    add('own_dfga100', np.where(a[4]>0, 100*np.maximum(a[0],0), np.nan), a[4], a[5].astype(int), valid)
    mpg, games, last = mpg_queries(q)
    add('ctx_mpg', mpg, games, last, None); result['ctx_games'] = games[take]
    result['parameter_vintage'] = year
    result['dfga_parameter_fit_end'] = pd.Timestamp(model['dfga']['fit_end'])
    result['rim_parameter_fit_end'] = pd.Timestamp(model['rim']['fit_end'])
    days = (result.game_date-pd.Timestamp('1970-01-01')).dt.days.to_numpy()
    for name in [c for c in result if c.endswith('__last_observation_day')]:
        if not ((result[name].to_numpy() == -1) | (result[name].to_numpy() < days)).all():
            raise ValueError('Same-date/future measurement in query')
    if snapshot_cut is not None:
        result['estimate_date']=snapshot_cut
        return result
    dest = Path(out)/'vintages'/f'{year}.parquet'; dest.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(dest, index=False)
    receipt = dict(status='passed_defensive_queries', year=year, rows=len(result), sha256=C.sha(dest),
        model_sha256=C.sha(model_path), run_lock_sha256=C.sha(Path(out)/'run_lock.json'),
        availability={name:int(result[name].notna().sum()) for name in TRACKING+['own_rim_contests100','own_rim_saved100','own_dfga100','ctx_mpg']})
    write_json(dest.with_suffix('.json'), receipt)
    return receipt


def defensive_snapshot(source, cut, player_ids, work_dir, model_path=None):
    """Pure annual-cut queries, including never-observed requested players."""
    cut=pd.Timestamp(cut)
    if len(np.unique(player_ids)) != len(player_ids):
        raise ValueError('Duplicate snapshot player identity')
    if model_path is None:
        choices=[(C.vintage_cut(y),DEFENSE/'models'/f'{y}.json') for y in range(2015,2026)
                 if C.vintage_cut(y)<=cut]
        if not choices:raise ValueError('No legal annual defensive parameter vintage')
        _,model_path=max(choices)
    model_path=Path(model_path);model=json.loads(model_path.read_text());year=int(model['vintage'])
    q,rim,dfga,measured,audit=source_inputs(source)
    result=build_year(work_dir,year,q,rim,dfga,measured,snapshot_cut=cut,snapshot_ids=player_ids,model_file=model_path)
    result=result.set_index('player_id').reindex(np.asarray(player_ids,dtype=int)).reset_index()
    sources=[Path(source)/'player_game_lines.parquet',DEFENSE/'rim_counts.parquet',
             DEFENSE/'dfga_counts.parquet',MEASUREMENTS,model_path,Path(__file__),
             Path(__file__).with_name('build_pregame_neutral_skills.py'),
             Path(__file__).with_name('pregame_neutral_kernel.cpp')]
    receipt=dict(status='passed_defensive_snapshot',estimate_date=str(cut.date()),players=len(result),
        source_hashes={str(p.resolve()):C.sha(p) for p in sources},model_sha256=C.sha(model_path),
        parameter_vintage=year,source_reconciliation=audit,raw_missing_history_preserved=True,
        fallback_policy='requires downstream admitted-and-observed model-aware transform',
        latest_measurement_fence='strictly before query date; all same-day measurements excluded')
    return result,receipt


def audit_prefix(out, year, q, rim, dfga, measured, date='2020-01-22'):
    """Perturb current/later outcomes and exercise the complete export adapter."""
    cutoff = pd.Timestamp(date)
    diagnostic = Path(out)/'prefix_counterfactual'
    if diagnostic.exists():
        raise FileExistsError('Preserve diagnostic audit; use a fresh output')
    diagnostic.mkdir()
    write_json(diagnostic/'run_lock.json',dict(diagnostic_only=True, admitted=False,
        parent_run_lock_sha256=C.sha(Path(out)/'run_lock.json'),mutation_start=date))
    q2=q.copy();q2.loc[q2.game_date.ge(cutoff),'minutes']=1.
    rim2=rim.copy();rim2.loc[rim2.game_date.ge(cutoff),'rim_dfgm']=0.
    dfga2=dfga.copy();dfga2.loc[dfga2.game_date.ge(cutoff),'dfga_value']=0.
    measured2=measured.copy();measured2.loc[measured2.game_date.ge(cutoff),'numerator']=0.
    build_year(diagnostic,year,q2,rim2,dfga2,measured2)
    a=pd.read_parquet(Path(out)/'vintages'/f'{year}.parquet')
    b=pd.read_parquet(diagnostic/'vintages'/f'{year}.parquet')
    pd.testing.assert_frame_equal(a[KEY+['game_date']],b[KEY+['game_date']])
    prefix=a.game_date.le(cutoff)
    fields=TRACKING+['own_rim_contests100','own_rim_suppression','own_rim_saved100','own_dfga100','ctx_mpg']
    changes={}
    for field in fields:
        np.testing.assert_array_equal(a.loc[prefix,field],b.loc[prefix,field])
        for suffix in ['_missing','__prior_exposure','__last_observation_day']:
            np.testing.assert_array_equal(a.loc[prefix,field+suffix],b.loc[prefix,field+suffix])
        diff=np.abs(a.loc[~prefix,field]-b.loc[~prefix,field])
        changes[field]=int(diff.gt(1e-12).sum())
    receipt=dict(status='passed_complete_defensive_prefix_audit',year=year,mutation_start=date,
        prefix_queries=int(prefix.sum()),maximum_prefix_gap=0.,later_changed_rows=changes,
        reference_sha256=C.sha(Path(out)/'vintages'/f'{year}.parquet'),
        counterfactual_sha256=C.sha(diagnostic/'vintages'/f'{year}.parquet'),diagnostic_admitted=False)
    write_json(Path(out)/'prefix_audit.json',receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT); ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--years', nargs='+', type=int, default=list(range(2015,2026)))
    ap.add_argument('--audit-prefix', action='store_true', help='Perturb2019 deployment outcomes from2020-01-22')
    args = ap.parse_args()
    if any(y not in range(2015,2026) for y in args.years):
        raise ValueError('Only legal annual vintages2015–2025')
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('build_pregame_neutral_skills.py'),
        Path(__file__).with_name('pregame_neutral_kernel.cpp'), args.source/'player_game_lines.parquet',
        DEFENSE/'rim_counts.parquet', DEFENSE/'dfga_counts.parquet', MEASUREMENTS]
    sources += sorted((DEFENSE/'models').glob('*.json'))
    if (args.source/'manifest.json').exists(): sources.append(args.source/'manifest.json')
    lock = dict(source_directory=str(args.source.resolve()), source_hashes={str(p.resolve()):C.sha(p) for p in sources},
        semantics='selected measurement denominators; validated identity census; frozen annual parameters; missing until observed',
        full_recipe_ready=False, source_clean_parameter_refit=False)
    lock_path = args.out/'run_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError('Changed inputs/code require a new immutable output directory')
    write_json(lock_path,lock)
    q,rim,dfga,measured,audit = source_inputs(args.source)
    write_json(args.out/'source_audit.json', audit)
    for year in args.years:
        receipt = args.out/'vintages'/f'{year}.json'
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['run_lock_sha256'] != C.sha(lock_path) or old['sha256'] != C.sha(receipt.with_suffix('.parquet')):
                raise ValueError('Cached export differs from immutable lock')
            print(json.dumps(dict(year=year,status='cached')), flush=True)
        else:
            print(json.dumps(build_year(args.out,year,q,rim,dfga,measured)),flush=True)
    if args.audit_prefix:
        print(json.dumps(audit_prefix(args.out,2019,q,rim,dfga,measured)),flush=True)


if __name__ == '__main__':
    main()

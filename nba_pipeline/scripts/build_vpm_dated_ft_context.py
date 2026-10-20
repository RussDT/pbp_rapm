"""RAPM-owned v50 seam: pregame expected FT and repaired possession context.

Uses frozen annual own-FT models, never full-season shooter percentages.
Existing actual/SQ atoms and all published outputs are read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_monthly_own_skills import (ROOT, OUT as SKILLS, AGE, B, FastOU,
    load_panel, context_columns, build_corrections, vintage_cut, sha, write_json)

OUT = ROOT/'results/vpm_dated_ft_context_v1'
ATOMS = ROOT/'results/monthly_vpm_targets_clean_sq_v2'
SLOTS = [f'{s}{i}' for s in ['O', 'D'] for i in range(1, 6)]
CONTEXT = ['home_poss','home_orientation_resolved','score_diff','period',
    'pre_period_clock_seconds','seconds_remaining_game','regulation_progress',
    'is_overtime','overtime_progress']


def normal(values):
    return pd.to_numeric(values, errors='raise').astype('int64').astype(str).str.zfill(10)


def predict_ft_games(model, end):
    """Neutral prior state, before assimilation of this game's FT outcomes."""
    spec = next(s for s in B.SKILLS if s.name == 'ft_pct')
    df, ctx = load_panel(SKILLS, pd.Timestamp(end))
    sub = B.skill_subpanel(df, spec)
    if sub.duplicated(['player_id','game_date']).any():
        raise ValueError('Same-day shooter rows require batched prior states')
    B.WARMUP_SEASON = int(sub.season.min())
    league, _ = B.league_center(sub, spec)
    centers = league if model['centered'] else np.zeros(len(sub))
    mean = 0. if model['centered'] else model['mean_rate']
    response = sub.obs_rate.to_numpy()-centers
    engine = FastOU(sub, SKILLS/'kernel')
    base = engine.run(response, model['stage_a']['params'], mean)[0]+centers
    hp = sub.step_k.to_numpy() > 0
    residual = np.where(hp, sub.obs_rate.to_numpy()-base, np.nan)
    cols = context_columns(sub, spec, base, residual, league, None,
        model['context_c'], model['reference_year'])
    context, _ = B_unused_expanding(cols, residual, sub, hp)
    init, shift, transient, _ = build_corrections(sub, spec, pd.read_parquet(AGE),
        model['mean_rate'], sorted(sub.season.unique()))
    pars = model['params']
    pred = engine.run(response, pars[:6], mean, init-centers, shift,
        transient, context, pars[6])[0]
    # Remove next-game context/transient from the prediction; retain neutral
    # skill state after correcting earlier observed outcomes for their context.
    neutral = pred-transient-pars[6]*context+centers
    d = sub[['game_id','game_date','player_id','ftm','fta']].copy()
    d['ft_expectation'] = np.clip(neutral, 0., 1.)
    d['model_fit_end'] = pd.Timestamp(model['fit_end'])
    d['model_fit_cut'] = pd.Timestamp(model['fit_cut'])
    d['last_shooter_observation'] = sub.groupby('player_id').game_date.shift()
    return d.loc[d.game_date.ge(pd.Timestamp(model['fit_cut']))].reset_index(drop=True)


def B_unused_expanding(cols, residual, sub, hp):
    from estimated_skills_context import expanding_linear_context
    return expanding_linear_context(cols, residual, sub.exposure.to_numpy(),
        sub.season.to_numpy(), hp, sorted(sub.season.unique())[1:])


def build_predictions(out):
    dest = out/'ft_game_predictions.parquet'
    if dest.exists():
        return pd.read_parquet(dest)
    parts = []; sources = {}
    for year in range(2015, 2026):
        path = SKILLS/'models'/str(year)/'ft_pct.json'
        model = json.loads(path.read_text()); sources[str(path)] = sha(path)
        end = min(vintage_cut(year+1), pd.Timestamp('2026-07-01'))
        d = predict_ft_games(model, end)
        if not d.model_fit_end.lt(d.game_date).all():
            raise ValueError('FT parameters reach the predicted game')
        if not (d.last_shooter_observation.isna() | d.last_shooter_observation.lt(d.game_date)).all():
            raise ValueError('FT shooter observations reach the predicted game')
        d['model_vintage'] = year; parts.append(d)
        print(f'FT pregame vintage {year}: {len(d):,} shooter games', flush=True)
    d = pd.concat(parts, ignore_index=True); d.game_id = normal(d.game_id)
    if d.duplicated(['game_id','player_id']).any():
        raise ValueError('Duplicate FT prediction')
    d.to_parquet(dest, index=False)
    sources.update({str(p):sha(p) for p in [Path(__file__),AGE,
        SKILLS/'game_context.parquet',ROOT/'results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet']})
    write_json(out/'ft_predictions_manifest.json', dict(sources=sources,
        rows=len(d),output_sha256=sha(dest),same_day_outcomes_excluded=True,
        target='neutral pregame own FT skill; earlier-game observations only'))
    return d


def build_context(year, phase, out):
    import rapm
    suffix = '_PS' if phase == 'PS' else ''
    path = ROOT/f'processed/RAPM{year%100:02d}{suffix}.parquet'
    dest = out/f'context_{year}_{phase}.parquet'
    if dest.exists():
        return
    d = pd.read_parquet(path); d['source_season_end_year'] = year
    c = rapm.transform_to_home_away_format(d)
    result = d[['game_id','event_num',*SLOTS]].copy()
    result.game_id = normal(result.game_id)
    for key in CONTEXT:
        result[key] = c[key].to_numpy()
    result.to_parquet(dest, index=False)
    write_json(dest.with_suffix('.json'), dict(source_sha256=sha(path),
        rows=len(d),resolved_share=float(c.home_orientation_resolved.mean()),
        output_sha256=sha(dest),source=str(path)))


def build_ft_atoms(year, phase, predictions, out):
    from process_rapm_blocks import clean_chance_utils as clean
    from process_rapm_blocks.common import _finalize_df
    suffix = '_PS' if phase == 'PS' else ''
    source = ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet'
    original = ATOMS/f'atoms_{year}_{phase}.parquet'
    dest = out/f'ft_{year}_{phase}.parquet'
    if dest.exists():
        return
    probs = predictions.set_index(['game_id','player_id']).ft_expectation
    attach = clean._attach_rapm_scoring; fetch = clean.fetch_player_stats_supabase
    counts = {}
    def dated_attach(frame, *args):
        d = attach(frame, *args)
        mask = (d.event_type.eq('FreeThrow') |
            d.home_description.fillna('').str.contains('Free Throw',case=False) |
            d.visitor_description.fillna('').str.contains('Free Throw',case=False))
        keys = pd.MultiIndex.from_arrays([normal(d.loc[mask,'game_id']),
            pd.to_numeric(d.loc[mask,'player1_id'],errors='raise').astype('int64')])
        p = probs.reindex(keys).to_numpy()
        if not np.isfinite(p).all():
            missing = keys[~np.isfinite(p)]
            raise ValueError(f'Missing dated FT prediction: {list(missing[:10])}')
        actual = d.loc[mask,'ActualNet'].to_numpy().copy()
        d.loc[mask,'ActualNet'] = p; d.loc[mask,'ExpFT'] = p
        counts.update(ft_events=int(mask.sum()),expected_sum=float(p.sum()),actual_sum=float(actual.sum()))
        return d
    try:
        clean.fetch_player_stats_supabase = lambda *args:None
        clean._attach_rapm_scoring = dated_attach
        base = clean.build_clean_chance_base_py(str(source),year%100,
            f'{year-1}-{year%100:02d}','DATED_FT',include_continuation_atoms=False)
    finally:
        clean._attach_rapm_scoring = attach; clean.fetch_player_stats_supabase = fetch
    fc = _finalize_df(base,['Actual_Diff','First_Chance_Diff','Second_Chance_Diff'],year%100)
    old = pd.read_parquet(original)
    for key in ['game_id','event_num',*SLOTS]:
        np.testing.assert_array_equal(pd.to_numeric(fc[key]),pd.to_numeric(old[key]))
    result = old[['game_id','event_num']].copy();result.game_id = normal(result.game_id)
    result['expected_ft_points'] = fc.Actual_Diff.to_numpy()
    result['ft_delta'] = result.expected_ft_points-old.actual_points.to_numpy()
    closure = float(abs(fc.Actual_Diff-fc.First_Chance_Diff-fc.Second_Chance_Diff).max())
    if closure > 1e-9:
        raise ValueError('Expected-FT parent does not close')
    result.to_parquet(dest,index=False)
    write_json(dest.with_suffix('.json'), dict(rows=len(old),**counts,
        max_fc_sc_closure=closure,actual_atoms_unchanged=True,
        source_hashes={str(p):sha(p) for p in [source,original,Path(__file__),Path(clean.__file__),out/'ft_game_predictions.parquet']},
        output_sha256=sha(dest)))
    print(f'Dated FT atoms {year} {phase}: {len(old):,} rows',flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=['predictions','atoms','context','all'],default='all')
    args = ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    predictions = None
    if args.stage in ['predictions','atoms','all']:
        predictions = build_predictions(args.out)
    if args.stage in ['atoms','all']:
        for year in range(2016,2027):
            for phase in ['RS','PS']:
                build_ft_atoms(year,phase,predictions,args.out)
    if args.stage in ['context','all']:
        for year in range(2014,2027):
            for phase in ['RS','PS']:
                build_context(year,phase,args.out)


if __name__ == '__main__':
    main()

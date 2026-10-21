"""Strictly dated fixed-EB box/PBP and available tracking inputs for VPM history.

This is an explicitly different input estimator from the selected annual-OU
research model. No later fitted skill parameters are carried into early dates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_nba_player_game_lines as L
from build_vpm_defense_bank import summarize, sha
from build_predictive_rim_saved import lagged_league

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/estimated_skills/vpm_historical_features_v1'
QUARANTINE = {'0021300316', '0041300305', '0021400843', '0021401185',
    '0029901052', '0040100131', '0040100132', '0040100133', '0040100134',
    '0040100201', '0040100202', '0040100203', '0040100204', '0040100205'}
SCORE_REVIEWS = [ROOT/'results/vpm_historical_targets_v1'/f'{s}_score_conflicts_final_score_check.json'
                 for s in ['early','modern']]
RATE_SPECS = {
    'own_rim_fga_rate': ('rim_fga', 'off_poss'),
    'own_mid_fga_rate': ('mid_fga', 'off_poss'),
    'own_three_fga_rate': ('three_fga', 'off_poss'),
    'own_fta_rate': ('fta', 'off_poss'), 'own_pts_rate': ('pts', 'off_poss'),
    'own_ast_rate': ('ast', 'off_poss'), 'own_tov_rate': ('tov', 'off_poss'),
    'own_oreb_rate': ('oreb', 'off_poss'), 'own_usg_pct': ('usage_events', 'off_poss'),
    'own_dreb_rate': ('dreb', 'def_poss'), 'own_stl_rate': ('stl', 'def_poss'),
    'own_blk_rate': ('blk', 'def_poss'),
    'own_pf_committed_rate': ('pf_committed', 'total_poss'),
}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def cuts():
    """52 quarterly report cuts plus matured-label training anchors."""
    return sorted(set(pd.date_range('2014-01-01', '2026-10-01', freq='QS')) |
                  {pd.Timestamp(f'{y}-10-01') for y in range(2008, 2026)} |
                  {pd.Timestamp('2020-12-01'), pd.Timestamp('2026-09-01')})


def normalize_clock(raw):
    """Historical seconds field already contains the full clock; parse MM:SS."""
    result = raw.copy()
    pieces = result.time_quarter.astype(str).str.extract(r'^(\d+):(\d+(?:\.\d+)?)$')
    if pieces.isna().any(axis=None):
        raise ValueError('Unparseable historical quarter clock')
    result['minute_remaining_quarter'] = pieces[0].astype(float)
    result['seconds_remaining_quarter'] = pieces[1].astype(float)
    return result


class RawReader:
    """Local module-only pandas facade; leaves canonical raw files untouched."""
    def __getattr__(self, name):
        return getattr(pd, name)

    def read_parquet(self, path, columns=None, **kwargs):
        if columns == L.RAW_COLS:
            frame = pd.read_parquet(path, columns=[*columns, 'time_quarter'], **kwargs)
            return normalize_clock(frame)[columns]
        return pd.read_parquet(path, columns=columns, **kwargs)


def extract_legacy(season, phase, common):
    raw = ROOT/'raw_data'/f'NBA{season%100:02d}{"_PS" if phase=="ps" else ""}.parquet'
    previous = L.pd
    try:
        L.pd = RawReader()
        packet = L.extract_season(raw, season, phase, common)
    finally:
        L.pd = previous
    observations = packet['lines']
    exposure = L.load_possessions(season, phase, common)
    # A possession-only appearance is missing counts, not an observed zero.
    joined = observations.merge(exposure, on=['game_id', 'player_id'], how='outer', indicator=True)
    joined['counts_observed'] = joined['_merge'].ne('right_only')
    joined['season'] = season
    joined['season_type'] = phase
    joined[['off_poss', 'def_poss']] = joined[['off_poss', 'def_poss']].fillna(0)
    dates = exposure.groupby('game_id').game_date.first()
    joined['game_date'] = joined.game_date.fillna(joined.game_id.map(dates))
    if joined.game_date.isna().any():
        raise ValueError('Undated raw player-game')
    joined = joined.drop(columns='_merge')
    recon = packet['recon']
    recon.update(season=season, phase=phase, rows=len(joined),
                 possession_only_rows=int((~joined.counts_observed).sum()),
                 clock_rule='parsed time_quarter MM:SS before existing extractor',
                 measured_minutes=float(joined.minutes.sum()),
                 complete_season_claim=False)
    return joined, recon


def build_lines(out=OUT, start=2000):
    out = Path(out)
    cache = out/'lines'
    cache.mkdir(parents=True, exist_ok=True)
    common = L.load_common()
    for year in range(start, 2014):
        for phase in ['rs', 'ps']:
            dest = cache/f'{year}_{phase}.parquet'
            stamp = dest.with_suffix('.json')
            raw = ROOT/'raw_data'/f'NBA{year%100:02d}{"_PS" if phase=="ps" else ""}.parquet'
            canonical = ROOT/'processed'/f'RAPM{year%100:02d}{"_PS" if phase=="ps" else ""}.parquet'
            signature = {str(p): sha(p) for p in [raw, canonical, Path(L.__file__), Path(__file__)]}
            if dest.exists() and stamp.exists():
                meta = json.loads(stamp.read_text())
                if meta['source_hashes'] != signature or meta['sha256'] != sha(dest):
                    raise ValueError('Changed historical line source; use a new version')
                continue
            frame, recon = extract_legacy(year, phase, common)
            frame.to_parquet(dest, index=False)
            write_json(stamp, dict(**recon, source_hashes=signature, sha256=sha(dest)))
            print('historical_lines', year, phase, len(frame), flush=True)


def load_lines(out):
    paths = sorted((Path(out)/'lines').glob('*.parquet'))
    years = {int(p.stem.split('_')[0]) for p in paths}
    if not set(range(2000, 2014)).issubset(years):
        raise ValueError('Need complete 2000–2013 warmup file set')
    modern = ROOT/'results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet'
    frames = [pd.read_parquet(p) for p in paths]
    m = pd.read_parquet(modern)
    m['counts_observed'] = True
    frames.append(m)
    d = pd.concat(frames, ignore_index=True)
    d['game_id'] = d.game_id.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    d['game_date'] = pd.to_datetime(d.game_date)
    if d.season.max() > 2026 or d.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Protected season or duplicate player-game')
    d = d.loc[~d.game_id.isin(QUARANTINE)].copy()
    # The legacy archive never supplied steal/block tags. Their extractor zeros
    # are unavailable fields, not measured zero activity.
    d.loc[d.season.lt(2014), ['stl', 'blk']] = np.nan
    d['usage_events'] = d.fga + .44*d.fta + d.tov
    d['total_poss'] = d.off_poss + d.def_poss
    for year in range(2000,2014):
        names = pd.read_csv(ROOT.parent/'player_sheets/year_totals'/f'{year}.csv',
            usecols=['PLAYER_ID','PLAYER_NAME']).set_index('PLAYER_ID').PLAYER_NAME
        selected = d.season.eq(year)
        d.loc[selected,'player_name'] = d.loc[selected,'player_id'].map(names)
    return d, [*paths, modern]


def annual_defense_observations():
    """Completed RS box STL/BLK with full NBA Advanced POSS, not partial PBP E.

    The date is an explicit conservative availability marker: July 1 after
    the completed regular season. These are annual aggregate measurements,
    never asserted to be player-game rows or exact defensive possessions.
    """
    frames = []; paths = []; audits = []
    for year in range(2000, 2014):
        path = ROOT.parent/'player_sheets/year_totals'/f'{year}.csv'
        columns = ['PLAYER_ID','PLAYER_NAME','year','GP','MIN','POSS','STL','BLK']
        d = pd.read_csv(path, usecols=columns)
        if d.PLAYER_ID.duplicated().any() or not d['year'].eq(year).all():
            raise ValueError('Ambiguous annual NBA identity or duplicated team stints')
        for c in ['PLAYER_ID','GP','POSS','STL','BLK']:
            if not np.isfinite(d[c]).all() or not np.equal(d[c], np.floor(d[c])).all() or d[c].lt(0).any():
                raise ValueError('Invalid annual box integer count: '+c)
        if not d.PLAYER_ID.between(1,1610612736).all() or not d.GP.between(1,85).all() or not d.MIN.gt(0).all():
            raise ValueError('Invalid annual box identity/exposure')
        no_possessions = d.POSS.eq(0)
        if d.loc[no_possessions,['STL','BLK']].ne(0).any(axis=None):
            raise ValueError('Annual positive counts without official opportunities')
        dropped_zero_possessions = int(no_possessions.sum())
        d = d.loc[~no_possessions].copy()
        pace = 48*d.POSS.sum()/d.MIN.sum()
        if not 70 < pace < 120 or d.STL.sum()<1000 or d.BLK.sum()<1000:
            raise ValueError('Degenerate annual box source or non-total POSS')
        # June30 is the aggregate's conservative end marker; availabilityJuly1
        # is checked separately, allowing use in the July1 report.
        frame = pd.DataFrame(dict(player_id=d.PLAYER_ID.astype(int),
            game_date=pd.Timestamp(f'{year}-06-30'),
            available_date=pd.Timestamp(f'{year}-07-01'),
            annual_poss=d.POSS.astype(float), stl=d.STL.astype(float), blk=d.BLK.astype(float)))
        frames.append(frame); paths.append(path)
        audits.append(dict(season=year,players=len(d),duplicate_player_ids=0,
            zero_possession_rows_excluded=dropped_zero_possessions,
            steals=int(d.STL.sum()),blocks=int(d.BLK.sum()),official_possessions=int(d.POSS.sum()),
            implied_pace=float(pace),available_date=f'{year}-07-01',
            denominator='NBA Advanced POSS, full annual total; not canonical defensive possession count'))
    return pd.concat(frames,ignore_index=True), paths, audits


def eb_feature(frame, players, cut, numerator, denominator, k, scale=100., half_life=550.):
    """Prefix-only population mean; unknown rows never contribute exposure."""
    z = frame.loc[frame.game_date.lt(cut) & frame[denominator].gt(0) &
                  np.isfinite(frame[numerator]) & frame[numerator].ge(0)].copy()
    w = np.exp2(-(cut-z.game_date).dt.total_seconds()/(86400*half_life))
    z['n'] = z[numerator]*w
    z['e'] = z[denominator]*w
    sums = z.groupby('player_id')[['n', 'e']].sum().reindex(players)
    league = float(z.n.sum()/z.e.sum()) if z.e.sum() else np.nan
    value = scale*(sums.n+k*league)/(sums.e+k)
    value = value.where(sums.e.gt(0))
    last = z.groupby('player_id').game_date.max().reindex(players)
    return value.to_numpy(), sums.e.to_numpy(), last.to_numpy(), league


def feature_snapshot(lines, cut, measurements=None, rim=None, dfga=None, annual_defense=None):
    cut = pd.Timestamp(cut)
    prior = lines.loc[lines.game_date.lt(cut)].copy()
    players = np.sort(prior.player_id.unique().astype(int))
    d = pd.DataFrame({'player_id': players, 'estimate_date': cut})
    observed = prior.loc[prior.counts_observed].copy()
    records = []

    def add(name, values, exposure, last, league, recipe='fixed_eb_td550', prior_k=1000.):
        d[name] = values
        for pid, v, e, t in zip(players, values, exposure, last):
            if pd.notna(t):
                records.append(dict(player_id=int(pid), estimate_date=cut, feature=name,
                    value=float(v) if np.isfinite(v) else np.nan, measured_exposure=float(e),
                    last_observation_date=t, estimator=recipe, prior_exposure=prior_k,
                    prefix_population_mean=league, parameter_fit_end=pd.NaT))

    for name, (n, e) in RATE_SPECS.items():
        frame = observed
        recipe = 'fixed_eb_td550'
        if annual_defense is not None and name in ['own_blk_rate','own_stl_rate']:
            annual = annual_defense.loc[annual_defense.available_date.le(cut),
                ['player_id','game_date',n,'annual_poss']].rename(columns={'annual_poss':e})
            frame = pd.concat([observed[['player_id','game_date',n,e]],annual],ignore_index=True)
            recipe = 'fixed_eb_td550_annual_official_poss_pre2014_daily_defposs_from2014'
        v, ex, last, league = eb_feature(frame, players, cut, n, e, 1000.)
        add(name, v, ex, last, league, recipe=recipe)
    for zone, n, e, attempt in [('rim','rim_fgm','rim_fga','own_rim_fga_rate'),
        ('mid','mid_fgm','mid_fga','own_mid_fga_rate'),
        ('three','three_fgm','three_fga','own_three_fga_rate'),('ft','ftm','fta','own_fta_rate')]:
        v, ex, last, league = eb_feature(observed, players, cut, n, e, 100., scale=1.)
        add(f'own_{zone}_pct', v, ex, last, league, prior_k=100.)
        # An observed zero-attempt history still has coherent zero expected makes.
        rate = d[attempt].to_numpy()
        makes = rate*np.where(np.isfinite(v), v, league)
        add(f'own_{zone}_makes100', makes, ex, last, league,
            recipe='fixed_eb_attempt_rate_times_pct', prior_k=100.)
    recent = observed.loc[observed.season_type.eq('rs') & observed.minutes.gt(0) &
                           observed.game_date.ge(cut-pd.Timedelta(days=365))]
    mpg = recent.groupby('player_id').agg(value=('minutes','mean'), exposure=('game_id','size'),
                                        last=('game_date','max')).reindex(players)
    add('ctx_mpg', mpg.value.to_numpy(), mpg.exposure.to_numpy(), mpg['last'].to_numpy(),
        np.nan, recipe='observed_trailing365_rs_positive_appearance_mpg', prior_k=0.)
    d['ctx_games'] = mpg.exposure.to_numpy()
    for side in ['off','def']:
        d[side+'_source_possessions'] = prior.groupby('player_id')[side+'_poss'].sum().reindex(players).fillna(0).to_numpy()
    if measurements is not None:
        extras, history = summarize(measurements, players, cut, half_life=550)
        for name in [c for c in extras if c.startswith('def_') and not c.endswith('__exposure')]:
            h = (history.loc[history.feature.eq(name)].set_index('player_id').reindex(players)
                 if 'feature' in history else pd.DataFrame(index=players))
            last = h.last_observation_date.to_numpy() if 'last_observation_date' in h else np.full(len(players), np.datetime64('NaT'))
            add(name, extras[name].to_numpy(), extras[name+'__exposure'].to_numpy(), last,
                np.nan, recipe='canonical_measurements_fixed_eb_td550')
    if dfga is not None:
        z = dfga.loc[dfga.dfga_valid & dfga.dfga_reason.eq('observed')]
        v, ex, last, league = eb_feature(z, players, cut, 'dfga_value', 'def_poss', 1000.)
        add('own_dfga100', v, ex, last, league)
    if rim is not None:
        z = rim.loc[rim.game_date.lt(cut)].copy()
        z['residual'] = z.league_pct*z.rim_dfga-z.rim_dfgm
        volume, ex, last, league = eb_feature(z, players, cut, 'rim_dfga', 'def_poss', 1000.)
        add('own_rim_contests100', volume, ex, last, league)
        w = np.exp2(-(cut-z.game_date).dt.total_seconds()/(86400*550.))
        z['wr'] = z.residual*w
        z['wn'] = z.rim_dfga*w
        sums = z.groupby('player_id')[['wr','wn']].sum().reindex(players)
        suppression = sums.wr/(sums.wn+100.)
        add('own_rim_suppression', suppression.to_numpy(), sums.wn.to_numpy(), last, 0., prior_k=100.)
        add('own_rim_saved100', 2*volume*suppression.to_numpy(), sums.wn.to_numpy(), last, 0.,
            recipe='twice_fixed_eb_contests_times_suppression', prior_k=100.)
    for name in [c for c in d if c.startswith('own_') or c.startswith('def_') or c=='ctx_mpg']:
        if name.endswith('_source_possessions'):
            continue
        d[name+'_missing'] = d[name].isna().astype(int)
    d['feature_recipe'] = 'fixed_eb_td550_k1000_pctk100_v1'
    p = pd.DataFrame(records)
    if len(p) and not pd.to_datetime(p.last_observation_date).lt(p.estimate_date).all():
        raise ValueError('Historical feature date fence')
    return d, p


def build_features(out=OUT):
    out = Path(out)
    lines, source_paths = load_lines(out)
    annual, annual_paths, annual_audit = annual_defense_observations()
    lines.to_parquet(out/'player_game_lines.parquet', index=False)
    tracking_root = ROOT/'results/estimated_skills/monthly_defense_v1'
    paths = [ROOT/'results/estimated_skills/defense_bank_v51_shooting_pf_repaired/measurements.parquet',
             tracking_root/'rim_counts.parquet', tracking_root/'dfga_counts.parquet']
    measurements, rim, dfga = [pd.read_parquet(p) for p in paths]
    for frame in [measurements, rim, dfga]:
        frame['game_date'] = pd.to_datetime(frame.game_date)
        frame['game_id'] = frame.game_id.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        frame.drop(frame.loc[frame.game_id.isin(QUARANTINE)].index, inplace=True)
        if frame.game_date.ge('2026-07-01').any():
            raise ValueError('Unexpected source date beyond completed2026')
    rim['league_pct'] = lagged_league(rim)
    packets = []; provenance = []; coverage = []
    for cut in cuts():
        d, p = feature_snapshot(lines, cut, measurements, rim, dfga, annual)
        packets.append(d); provenance.append(p)
        coverage.append(dict(estimate_date=str(cut.date()), players=len(d),
            available={c:int(d[c].notna().sum()) for c in d if c.startswith('own_') or c.startswith('def_') if not c.endswith('_missing')}))
        print('historical_features', str(cut.date()), len(d), flush=True)
    wide = pd.concat(packets, ignore_index=True)
    long = pd.concat(provenance, ignore_index=True)
    if wide.duplicated(['player_id','estimate_date']).any() or long.duplicated(['player_id','estimate_date','feature']).any():
        raise ValueError('Duplicate historical features')
    wide.to_parquet(out/'features.parquet', index=False)
    long.to_parquet(out/'provenance.parquet', index=False)
    source_paths += paths + annual_paths + SCORE_REVIEWS + [Path(__file__), Path(L.__file__),
        ROOT.parent/'player_sheets/gen_totals/averages_scrape.py']
    manifest = dict(schema_version=1, status='passed_historical_fixed_eb_input_candidate',
        feature_recipe='fixed_eb_td550_k1000_pctk100_v1', rows=len(wide), provenance_rows=len(long),
        cuts=[str(c.date()) for c in cuts()], quarterly_cuts=[str(c.date()) for c in pd.date_range('2014-01-01','2026-10-01',freq='QS')],
        parameters=dict(half_life_days=550, rate_prior_possessions=1000, percentage_prior_attempts=100),
        parameter_rule='fixed declared research design; no fitted future skill parameters',
        raw_date_rule='strictly earlier than cut', source_seasons=sorted(map(int,lines.season.unique())),
        first_observation=str(lines.game_date.min().date()), last_observation=str(lines.game_date.max().date()),
        source_hashes={str(p):sha(p) for p in source_paths},
        artifact_hashes={n:sha(out/n) for n in ['features.parquet','provenance.parquet','player_game_lines.parquet']},
        quarantined_games=sorted(QUARANTINE), coverage=coverage, protected_outcomes=0,
        annual_defense_audit=annual_audit,
        raw_source_quarantine=dict(confirmed_duplicate_2002_playoffs=sorted(g for g in QUARANTINE if g.startswith('00401')),
            raw_final_discrepancy=['0029901052'],
            previous_pinned_2014_2015=['0021300316','0041300305','0021400843','0021401185'],
            rule='Preserve raw/final matches despite canonical mismatch; exclude confirmed raw duplicates and unresolved raw/final conflict.',
            scoreboard_review='For zero last-score strings inspect numeric scoreboard maxima: 0029900739=146 raw146,0020200077=214 raw214,0021500359=208 raw208 retained;0029901052=188 raw189 excluded.'),
        limitations=['Different state estimator from current selected annual OU VPM; requires separately calibrated historical mapper',
            'Historical archives not certified contemporaneous publication vintages',
            '2011 raw archive has one missing RS game; no fabricated zero observations or complete-season claim',
            'Tracking unavailable before first validated measurements remains missing, never zero',
            'Parameters are conventional fixed research choices; not historically optimized or publication-vintage settings',
            'MPG is trailing365day RS mean over positive-minute appearances, not future availability'])
    write_json(out/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--stage', choices=['lines','features','all'], required=True)
    args = ap.parse_args()
    if args.stage in ['lines','all']:
        build_lines(args.out)
    if args.stage in ['features','all']:
        build_features(args.out)

"""Independent source/formula/date audit for completed-season BAM exports.

Reads no forecasts, model scores or protected-season outcomes. Does not import
the feature builder. Partial-download audits remain explicitly non-admissible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/estimated_skills/lagged_bam_v1'
FALLBACK = {2018:1.9197, 2019:2.1359, 2020:1.9674, 2021:1.9447,
            2022:1.7925, 2023:1.9128, 2024:1.6476, 2025:1.624}
FEATURE = 'matchup_dated_bam'
KEY = ['year', 'nba_id_off', 'nba_id_def']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_values(actual, expected, label, tolerance=1e-10):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True):
        raise ValueError('Independent BAM mismatch: ' + label)
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.


def pair_formula(rows):
    """Independently evaluate scalar legacy terms with an explicit admission mask."""
    fga, fta, tov, pts = (rows[column].to_numpy(float) for column in
                            ['MATCHUP_FGA', 'MATCHUP_FTA', 'MATCHUP_TOV', 'PLAYER_PTS'])
    raw_denom, eff_fga = fga + .44 * fta, fga + 1.05 * tov
    raw_ts = np.divide(pts, 2 * raw_denom, out=np.zeros(len(rows)), where=raw_denom != 0)
    eff_ts = np.divide(pts, 2 * (eff_fga + .44 * fta), out=np.zeros(len(rows)), where=(eff_fga + .44 * fta) != 0)
    ts = rows.TS_PCT.to_numpy(float)
    load = np.where(raw_ts > ts, rows.year.map(FALLBACK), rows.Offensive_Score).astype(float)
    # Source-year positions only. No BPM, career-profile boost or current-year
    # TS enters this completed-period construction.
    roles = rows.position.map(lambda value: value.upper().split('-')[0] if isinstance(value, str) else '')
    boost = roles.map({'PG':.3025, 'SG':.3025, 'G':.3025, 'SF':.25, 'PF':.25, 'F':.25, 'C':.125}).to_numpy(float)
    total_fga = rows.groupby(['year', 'nba_id_def']).MATCHUP_FGA.transform('sum').to_numpy(float)
    observed_ts = rows.observed_ts.to_numpy(float)
    verified_ts = (ts >= 0) & (np.abs(ts - observed_ts) <= .00051)
    raw_inputs = np.column_stack([fga, fta, tov, pts, rows.PARTIAL_POSS, ts, load])
    mask = ((fga > 0) & (fta >= 0) & (tov >= 0) & (pts >= 0)
            & (rows.PARTIAL_POSS.to_numpy(float) > 0) & (total_fga > 0)
            & verified_ts & np.isfinite(boost) & (load >= 0)
            & np.isfinite(raw_inputs).all(axis=1)
            & rows.pair_source_counts_valid.to_numpy(bool) & rows.pair_identity_verified.to_numpy(bool))
    result = -10 * (eff_ts - ts) * load * eff_fga / total_fga * (1 + boost)
    return dict(literal_raw_ts=raw_ts, literal_effective_fga=eff_fga,
                literal_effective_ts=eff_ts, literal_adjusted_load=load,
                all_defender_fga=total_fga, dated_boost=boost,
                pair_verified=mask, dated_bam_pair=np.where(mask, result, np.nan))


def season_reference(pairs):
    records = []
    for (year, player), group in pairs.groupby(['year', 'nba_id_def']):
        good = group.pair_verified & group.dated_bam_pair.notna()
        exposure = float(group.loc[good, 'PARTIAL_POSS'].sum())
        all_exposure = float(group.PARTIAL_POSS.sum())
        records.append(dict(year=year, player_id=player,
            season_raw=float(group.loc[good, 'dated_bam_pair'].sum()) if exposure > 0 else np.nan,
            season_exposure=exposure, all_source_exposure=all_exposure,
            source_coverage=exposure/all_exposure if all_exposure > 0 else np.nan,
            source_rs_end=group.source_rs_end.max(),
            position_available_at=group.position_available_at.max()))
    result = pd.DataFrame(records)
    result['season_z'] = np.nan
    for _, group in result.groupby('year'):
        usable = group.season_exposure.gt(0) & group.season_raw.notna()
        sample = group.loc[usable]
        if len(sample) < 2:
            continue
        weights, values = sample.season_exposure.to_numpy(), sample.season_raw.to_numpy()
        center = np.dot(values, weights) / weights.sum()
        variance = np.dot((values-center)**2, weights) / weights.sum()
        if variance > 0:
            result.loc[sample.index, 'season_z'] = (values-center)/np.sqrt(variance)
    return result


def snapshot_reference(seasons, cuts, ends):
    records = []
    for value in cuts:
        cut = pd.Timestamp(value)
        if cut >= pd.Timestamp('2026-10-01'):
            raise ValueError('Protected 2027 emission')
        allowed_years = sorted(year for year, day in ends.items() if pd.Timestamp(day) < cut)[-4:]
        eligible = seasons.loc[seasons.year.isin(allowed_years) & pd.to_datetime(seasons.source_rs_end).lt(cut)]
        for player, group in eligible.groupby('player_id'):
            decay = 2. ** (-(cut-pd.to_datetime(group.source_rs_end)).dt.total_seconds().to_numpy()/(86400*550))
            usable = group.season_exposure.gt(0).to_numpy() & group.season_z.notna().to_numpy()
            weight = group.season_exposure.to_numpy()*decay
            exposure = float(weight[usable].sum())
            numerator = float(np.dot(weight[usable], group.season_z.to_numpy()[usable]))
            all_exposure = float(np.dot(group.all_source_exposure, decay))
            good = group.loc[usable]
            last = good.source_rs_end.max()
            position_date = pd.to_datetime(good.position_available_at).max()
            fit_end = max(last, position_date) if pd.notna(last) and pd.notna(position_date) else last
            if pd.notna(fit_end) and fit_end >= cut:
                raise ValueError('Position publication at/after emission')
            records.append(dict(player_id=int(player), estimate_date=cut,
                matchup_exposure=exposure, matchup_dated_bam__exposure=exposure,
                matchup_dated_bam=numerator/(exposure+1000) if exposure > 0 else np.nan,
                matchup_dated_bam_unshrunk=numerator/exposure if exposure > 0 else np.nan,
                source_seasons=int(good.year.nunique()), latest_source_year=good.year.max(),
                source_coverage=exposure/all_exposure if all_exposure > 0 else 0.,
                reliability=exposure/(exposure+1000),
                last_observation_date=last, parameter_fit_end=fit_end))
    return pd.DataFrame(records)


def align(frame, keys):
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate export key ' + str(keys))
    return frame.set_index(keys).sort_index()


def compare_frames(actual, expected, keys, fields):
    actual, expected = align(actual, keys), align(expected, keys)
    if not actual.index.equals(expected.index):
        raise ValueError('Independent BAM row census mismatch: ' + str(keys))
    return {field: assert_values(actual[field], expected[field], field) for field in fields}


def audit(folder=DEFAULT):
    folder = Path(folder)
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = manifest['source_hashes']
    for path, checksum in sources.items():
        if sha(path) != checksum:
            raise ValueError('Source changed since BAM build: ' + path)
    for name, checksum in manifest['artifact_hashes'].items():
        if sha(folder/name) != checksum:
            raise ValueError('BAM export changed: ' + name)
    find = lambda suffix: next(Path(path) for path in sources if path.endswith(suffix))
    annual = pd.read_parquet(find('matchups_two_rs_2018_2026.parquet'))
    positions = pd.read_parquet(find('player_seasons.parquet'))
    lines = pd.read_parquet(find('player_game_lines.parquet'))
    census = pd.read_parquet(find('game_census.parquet'))
    gate_path = find('annual_native_concordance.json')
    gate = json.loads(gate_path.read_text())
    if gate['annual_sha256'] != sha(find('matchups_two_rs_2018_2026.parquet')):
        raise ValueError('Annual source not pinned by gate')
    if sorted(gate['verified_years']) != sorted(manifest['verified_source_years']):
        raise ValueError('BAM source-year admission differs from source gate')
    chunks = []
    for year in gate['verified_years']:
        if year not in range(2018,2026):
            raise ValueError('Protected/incomplete annual year admitted')
        name = f'native_{year}_rs_pairs_verified.parquet'
        path = find(name)
        if sha(path) != gate['artifacts'][name]:
            raise ValueError('Native aggregate not pinned by source gate')
        native = pd.read_parquet(path); native['year'] = year; chunks.append(native)
    native = pd.concat(chunks, ignore_index=True)
    pairs = pd.read_parquet(folder/'pairs_dated_bam.parquet')
    # Preserve all primitive rows and counts, including excluded pair FGA.
    gaps = compare_frames(pairs, native, KEY, ['PARTIAL_POSS','MATCHUP_FGA','MATCHUP_FTA',
        'MATCHUP_TOV','PLAYER_PTS','pair_source_counts_valid','pair_identity_verified'])
    pricing = annual.loc[annual.year.between(2018,2025)].groupby(['year','nba_id_off'])[['TS_PCT','Offensive_Score']].first().reset_index()
    ref = native.merge(pricing, on=['year','nba_id_off'], how='left', validate='many_to_one')
    pos = positions.rename(columns={'year':'source_year','nba_id':'nba_id_off'})
    ref = ref.merge(pos[['source_year','nba_id_off','position','earliest_use_date']],
                    left_on=['year','nba_id_off'], right_on=['source_year','nba_id_off'], how='left', validate='many_to_one')
    observed = lines.loc[lines.season_type.eq('rs')].groupby(['season','player_id'])[['pts','fga','fta']].sum()
    observed['observed_ts'] = observed.pts/(2*(observed.fga+.44*observed.fta).where((observed.fga+.44*observed.fta).gt(0)))
    observed.index.names = ['year','nba_id_off']
    ref = ref.merge(observed[['observed_ts']].reset_index(), on=['year','nba_id_off'], how='left', validate='many_to_one')
    ends = census.loc[census.season_type.eq('rs') & census.season.between(2018,2025)].groupby('season').game_date.max().to_dict()
    ref['source_rs_end'] = pd.to_datetime(ref.year.map(ends))
    ref['position_available_at'] = pd.to_datetime(ref.earliest_use_date)
    for field, values in pair_formula(ref).items():
        ref[field] = values
    gaps.update(compare_frames(pairs, ref, KEY, list(pair_formula(ref))))
    expected_seasons = season_reference(ref)
    seasons = pd.read_parquet(folder/'defender_seasons.parquet')
    gaps.update(compare_frames(seasons, expected_seasons, ['year','player_id'],
        ['season_raw','season_exposure','all_source_exposure','source_coverage','season_z']))
    expected_features = snapshot_reference(expected_seasons, manifest['cuts'], ends)
    features = pd.read_parquet(folder/'features.parquet')
    feature_fields = [column for column in expected_features if column not in
        ['player_id','estimate_date','last_observation_date','parameter_fit_end']]
    gaps.update(compare_frames(features, expected_features, ['player_id','estimate_date'], feature_fields))
    provenance = pd.read_parquet(folder/'feature_provenance.parquet')
    p, e = align(provenance,['player_id','estimate_date']), align(expected_features,['player_id','estimate_date'])
    if not p.index.equals(e.index) or not p.feature.eq(FEATURE).all():
        raise ValueError('BAM provenance row/feature mismatch')
    for field in ['last_observation_date','parameter_fit_end']:
        actual, expected = pd.to_datetime(p[field]), pd.to_datetime(e[field])
        if not (actual.eq(expected) | (actual.isna() & expected.isna())).all():
            raise ValueError('Independent BAM provenance mismatch: ' + field)
        if actual.ge(p.index.get_level_values('estimate_date')).any():
            raise ValueError('BAM source not strictly before emission')
    for cut in manifest['cuts']:
        # Prefix construction with future source years removed must emit the
        # identical cut, including four-season calendar rather than data fallback.
        day = pd.Timestamp(cut)
        prefix = expected_seasons.loc[pd.to_datetime(expected_seasons.source_rs_end).lt(day)]
        check = snapshot_reference(prefix, [cut], ends)
        selected = expected_features.loc[expected_features.estimate_date.eq(day)]
        if not selected.empty:
            compare_frames(selected, check, ['player_id','estimate_date'], feature_fields)
    if manifest['feature_groups'] != {'bam':[FEATURE]}:
        raise ValueError('Unexpected BAM feature admission (archival diagnostics must remain separate)')
    if manifest['protected_outcomes'] != 0:
        raise ValueError('Protected outcomes read')
    for path, checksum in sources.items():
        if sha(path) != checksum:
            raise ValueError('Source changed during independent BAM audit')
    result = dict(status='passed_independent_formula_and_chronology',
        source_gate_complete=manifest['source_gate_complete'],
        model_admissible_source=bool(manifest['source_gate_complete'] and gate['status']=='passed_source_concordance'),
        verified_years=gate['verified_years'], pair_rows=len(pairs), season_rows=len(seasons),
        feature_rows=len(features), provenance_rows=len(provenance),
        maximum_numerical_gap=max(gaps.values()), protected_outcomes=0,
        manifest_sha256=sha(manifest_path), auditor_sha256=sha(__file__),
        note='Formula audit success does not waive incomplete source years or certify predictive value/publication vintage.')
    (folder/'independent_bam_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, default=DEFAULT)
    audit(parser.parse_args().export)

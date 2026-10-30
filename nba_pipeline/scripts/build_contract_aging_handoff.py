"""Build a private, source-audited contract-aging input panel; fit no model.

Run from any directory with the repo's .venv/bin/python. No network or secrets
are needed. Vendor-derived outputs remain in the gitignored results directory.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd
from build_epm_game_value import reconstruct_player_minutes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'nba_pipeline/results/contract_aging_handoff'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []

    def record(path):
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    frames = []
    for year in range(1997, 2027):
        path = ROOT / f'player_sheets/year_totals/{year}.csv'
        d = pd.read_csv(path, usecols=['PLAYER_ID', 'PLAYER_NAME', 'year', 'AGE', 'GP', 'MIN'])
        assert d.year.eq(year).all() and not d.PLAYER_ID.duplicated().any(), path
        assert d[['GP', 'MIN']].notna().all().all() and d.GP.gt(0).all(), path
        assert d.MIN.ge(0).all() and d.GP.le(85).all(), path
        d = d.rename(columns={'PLAYER_ID': 'player_id', 'PLAYER_NAME': 'player_name',
                             'year': 'season', 'AGE': 'season_age', 'GP': 'rs_gp', 'MIN': 'rs_minutes'})
        d['rs_source'] = str(path)
        frames.append(d)
        record(path)
    observed = pd.concat(frames, ignore_index=True)
    # Keep every elapsed year, including gaps followed by a return and all
    # post-last-appearance years through the observed endpoint. Not retirement labels.
    grid = pd.DataFrame([(int(pid), y) for pid, first in observed.groupby('player_id').season.min().items()
                         for y in range(int(first), 2027)], columns=['player_id', 'season'])
    panel = grid.merge(observed, on=['player_id', 'season'], how='left', validate='one_to_one')
    panel['observed_rs_row'] = panel.rs_gp.notna()
    panel['rs_gp'] = panel.rs_gp.fillna(0).astype(int)
    panel['rs_minutes'] = panel.rs_minutes.fillna(0)
    panel['rs_mpg'] = panel.rs_minutes / panel.rs_gp.where(panel.rs_gp.gt(0))
    panel['player_name'] = panel.player_name.fillna(panel.player_id.map(observed.groupby('player_id').player_name.last()))
    panel['no_appearance_in_source'] = ~panel.observed_rs_row
    panel['season_observation_end'] = 2026
    panel['zero_exposure_certified'] = False
    panel['source_available_date'] = pd.NaT

    teams = []
    for path in sorted((ROOT / 'player_sheets/team_totals').glob('*.csv')):
        if not path.stem.isdigit() or int(path.stem) < 1610000000:
            continue
        d = pd.read_csv(path, usecols=['year', 'GamesPlayed', 'SecondsPlayed'])
        d['team_id'] = int(path.stem)
        d = d.rename(columns={'year': 'season', 'GamesPlayed': 'actual_team_games',
                              'SecondsPlayed': 'actual_team_seconds'})
        teams.append(d)
        record(path)
    teams = pd.concat(teams, ignore_index=True)
    assert not teams.duplicated(['season', 'team_id']).any()
    teams.to_csv(OUT / 'team_seasons.csv', index=False)
    census = teams.groupby('season').agg(teams=('team_id', 'nunique'),
                                        team_games=('actual_team_games', 'sum'),
                                        team_seconds=('actual_team_seconds', 'sum'))
    census['player_minutes'] = observed.groupby('season').rs_minutes.sum()
    census['player_minutes_minus_team_minutes_x5'] = census.player_minutes - census.team_seconds * 5 / 60
    census.to_csv(OUT / 'season_census_audit.csv')
    raw_audits = []
    for year in range(2002, 2027):
        path = ROOT / f'nba_pipeline/raw_data/NBA{year % 100:02d}.parquet'
        minutes = reconstruct_player_minutes(path)
        season_rows = observed[observed.season.eq(year)]
        compared = season_rows.merge(minutes, on='player_id', how='outer', indicator=True)
        games = pd.read_parquet(path, columns=['game_id', 'period']).groupby('game_id').period.max()
        expected_minutes = len(games) * 480 + (games - 4).clip(lower=0).sum() * 50
        player_gap = season_rows.rs_minutes.sum() - expected_minutes
        known_team_games = census.loc[year, 'team_games'] / 2
        missing = compared[compared._merge.eq('right_only')]
        # This certifies the any-positive-minute player census, not exact
        # per-player clock reconstruction or contemporaneous archival provenance.
        source_without_raw = int(compared._merge.eq('left_only').sum())
        certified = len(missing) == 0 and source_without_raw == 0 and len(games) == known_team_games
        aggregate_clock_pass = abs(player_gap) < 2
        panel.loc[panel.season.eq(year), 'zero_exposure_certified'] = certified
        raw_audits.append({'season': year, 'raw_games': len(games), 'team_games_div2': known_team_games,
                           'raw_positive_minute_players': len(minutes), 'source_players': len(season_rows),
                           'raw_players_missing_from_source': len(missing),
                           'source_players_without_positive_raw_minutes': source_without_raw,
                           'source_minutes_minus_game_clock_minutes': player_gap,
                           'max_player_minutes_disagreement': float((compared.rs_minutes - compared.minutes).abs().max()),
                           'gp_disagreements': int((compared.rs_gp != compared.gp).sum()),
                           'source_gp_minutes_nonmissing': bool(season_rows[['rs_gp', 'rs_minutes']].notna().all().all()),
                           'source_aggregate_clock_within_2_minutes': aggregate_clock_pass,
                           'zero_exposure_certified': certified})
        compared.to_csv(OUT / f'{year}_raw_census_comparison.csv', index=False)
        record(path)
        print(f'RS census {year}: {len(games)} games, {len(missing)} omitted positive-minute players; certified={certified}', flush=True)
    pd.DataFrame(raw_audits).to_csv(OUT / 'raw_census_audit.csv', index=False)

    # Title cache has monthly EPM-only history; full cache has additional dates
    # and skills. Preserve provenance and prefer full cache only after overlap audit.
    snapshots = []
    inventory = []
    for dirname in ['epm_title', 'epm']:
        for path in sorted((ROOT / f'nba_pipeline/cache/{dirname}').glob('epm_all_*_go0.csv')):
            d = pd.read_csv(path, usecols=['requested_date', 'date', 'game_dt', 'player_id',
                                         'player_name', 'game_optimized', 'oepm', 'depm', 'epm'])
            assert not d.player_id.duplicated().any() and d.game_optimized.eq(0).all(), path
            assert d.requested_date.nunique() == 1 and d.date.le(d.requested_date).all(), path
            manifest_path = path.with_suffix('.manifest.json')
            manifest = json.loads(manifest_path.read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['csv_sha256'], path
            d['snapshot_source'] = str(path)
            d['retrieved_at_utc'] = manifest.get('retrieved_at_utc')
            inventory.append({'path': str(path), 'requested_date': d.requested_date.iloc[0],
                              'effective_min': d.date.min(), 'effective_max': d.date.max(), 'rows': len(d)})
            snapshots.append(d)
            record(path)
            record(manifest_path)
    snapshots = pd.concat(snapshots, ignore_index=True)
    conflicts = []
    duplicate = snapshots[snapshots.duplicated(['requested_date', 'player_id'], keep=False)]
    for key, group in duplicate.groupby(['requested_date', 'player_id']):
        if any(group[c].nunique(dropna=False) > 1 for c in ['date', 'game_dt', 'oepm', 'depm', 'epm']):
            conflicts.append({'requested_date': key[0], 'player_id': int(key[1])})
    if conflicts:
        raise ValueError(f'Conflicting overlapping snapshots: {conflicts[:10]}')
    snapshots = snapshots.drop_duplicates(['requested_date', 'player_id'], keep='last')
    snapshots = snapshots.rename(columns={'date': 'provider_effective_date', 'game_dt': 'provider_last_player_game_date'})
    snapshots = snapshots.sort_values(['requested_date', 'player_id'])
    # These are post-season predictive states, not RS actual EPM or season averages.
    panel['epm_requested_date'] = panel.season.map(lambda y: '2020-11-01' if y == 2020 else
                                                  '2021-08-01' if y == 2021 else f'{y}-07-01')
    required_dates = set(panel.loc[panel.season.ge(2002), 'epm_requested_date'])
    assert required_dates <= set(snapshots.requested_date), required_dates - set(snapshots.requested_date)
    cols = ['player_id', 'requested_date', 'provider_effective_date', 'provider_last_player_game_date',
            'oepm', 'depm', 'epm', 'snapshot_source']
    panel = panel.merge(snapshots[cols].rename(columns={'requested_date': 'epm_requested_date'}),
                        on=['player_id', 'epm_requested_date'], how='left', validate='many_to_one')
    # A provider can retain an inactive player. EPM presence never defines participation.
    panel['epm_kind'] = 'postseason_predictive_state_go0'
    panel.to_parquet(OUT / 'player_seasons.parquet', index=False)
    panel.to_csv(OUT / 'player_seasons.csv', index=False)
    snapshots.to_parquet(OUT / 'predictive_epm_snapshots.parquet', index=False)
    snapshots.to_csv(OUT / 'predictive_epm_snapshots.csv', index=False)
    pd.DataFrame(inventory).to_csv(OUT / 'snapshot_inventory.csv', index=False)
    coverage = panel.groupby('season').agg(rows=('player_id', 'size'), rs_players=('observed_rs_row', 'sum'),
                                          rs_minutes=('rs_minutes', 'sum'))
    coverage['active_epm_matches'] = panel[panel.observed_rs_row & panel.epm.notna()].groupby('season').size()
    coverage['active_age_missing'] = panel[panel.observed_rs_row & panel.season_age.isna()].groupby('season').size()
    coverage = coverage.fillna(0)
    coverage.to_csv(OUT / 'coverage.csv')
    receipt = {'source_files': sources, 'rows': len(panel), 'snapshot_rows': len(snapshots),
               'snapshot_dates': snapshots.requested_date.nunique(), 'snapshot_min': snapshots.requested_date.min(),
               'snapshot_max': snapshots.requested_date.max(), 'overlap_conflicts': len(conflicts),
               'regular_seasons': [1997, 2026], 'postseason_epm_seasons': [2002, 2026],
               'notes': ['No-appearance rows are not confirmed retirement or injury labels.',
                         'Zero-exposure certification is narrowly a source/player census gate; see raw_census_audit.csv.',
                         'Source available dates are unknown; never use filesystem timestamps as as-of dates.',
                         'Team games are observed, not a known preseason schedule. Player exposure denominator needs a stated convention.',
                         'Age is reported season age; no exact birthday or imputed gap age.',
                         'MIN is season total minutes; MPG is MIN/GP.',
                         'Predictive EPM can include playoffs; never label this actual RS EPM.',
                         'Requested-date history is retrospectively retrieved, not archived model vintages.',
                         'Pre-2002 EPM is missing, not zero. Future seasons are not materialized.']}
    (OUT / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_files'}, indent=2))


if __name__ == '__main__':
    main()

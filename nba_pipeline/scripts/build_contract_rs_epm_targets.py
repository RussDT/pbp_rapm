"""Private RS minutes-weighted pregame predictive-EPM step targets, 2002–2026.

Requires the contract aging handoff and existing dated player-game lines.
This is a vendor predictive-state target, never realized/actual season EPM.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'nba_pipeline/results/contract_aging_handoff'


def historical_minutes(year, date_map):
    """Same event-clock allocation as build_nba_player_game_lines, without box stats."""
    from build_nba_player_game_lines import load_common, HOME_LINEUP, AWAY_LINEUP, TEAM_ID_MIN
    common = load_common()
    path = ROOT / f'nba_pipeline/raw_data/NBA{year % 100:02d}.parquet'
    columns = ['game_id', 'event_num', 'period', 'time_quarter', *HOME_LINEUP, *AWAY_LINEUP]
    df = pd.read_parquet(path, columns=columns)
    df['game_id'] = common.normalize_game_id_series(df.game_id)
    # Before 2014 seconds_remaining_quarter is already the full clock;
    # adding minutes*60 doubles it. Parse the explicit MM:SS source instead.
    clock_parts = df.time_quarter.astype('string').str.extract(r'^(\d+):(\d+(?:\.\d+)?)$')
    df['minute_remaining_quarter'] = pd.to_numeric(clock_parts[0], errors='raise')
    df['seconds_remaining_quarter'] = pd.to_numeric(clock_parts[1], errors='raise')
    assert df[['minute_remaining_quarter', 'seconds_remaining_quarter']].notna().all().all()
    df = common.sort_raw_pbp_chronologically(df.drop_duplicates())
    clock = (pd.to_numeric(df.minute_remaining_quarter, errors='coerce').fillna(0) * 60
             + pd.to_numeric(df.seconds_remaining_quarter, errors='coerce').fillna(0)).to_numpy(float)
    gid = df.game_id.to_numpy()
    period = df.period.to_numpy()
    same = (gid == np.roll(gid, -1)) & (period == np.roll(period, -1))
    elapsed = np.where(same, clock - np.roll(clock, -1), 0.)
    elapsed = np.where((elapsed > 0) & (elapsed <= 720), elapsed, 0.)
    parts = []
    for col in HOME_LINEUP + AWAY_LINEUP:
        part = pd.DataFrame({'game_id': gid, 'player_id': pd.to_numeric(df[col], errors='coerce'), 'seconds': elapsed})
        part = part[part.player_id.gt(0) & part.player_id.lt(TEAM_ID_MIN) & part.seconds.gt(0)]
        parts.append(part)
    result = pd.concat(parts).groupby(['game_id', 'player_id'], as_index=False).seconds.sum()
    result['player_id'] = result.player_id.astype('int64')
    result['minutes'] = result.pop('seconds') / 60
    result['season'] = year
    result['season_type'] = 'rs'
    result['game_date'] = result.game_id.map(date_map)
    assert result.game_date.notna().all(), f'Undated games in {year}'
    game_lengths = 480 + (df.groupby('game_id').period.max() - 4).clip(lower=0) * 50
    game_allocated = result.groupby('game_id').minutes.sum()
    assert (game_allocated - game_lengths).max() < .01, f'Overallocated clock minutes in {year}'
    minute_audit = pd.DataFrame({'expected_player_minutes': game_lengths, 'allocated_player_minutes': game_allocated})
    minute_audit.to_csv(OUT / f'{year}_target_clock_audit.csv')
    print(f'Historical game minutes {year}: {result.game_id.nunique()} games', flush=True)
    return result


def main():
    games_path = ROOT / 'nba_pipeline/results/estimated_skills/player_game_lines.parquet'
    snapshots_path = OUT / 'predictive_epm_snapshots.csv'
    games = pd.read_parquet(games_path, columns=['player_id', 'season', 'season_type', 'game_id', 'game_date', 'minutes'])
    games = games[games.season_type.eq('rs') & games.minutes.gt(0)].copy()
    dates_path = ROOT / 'player_sheets/gen_totals/game_dates.csv'
    dates = pd.read_csv(dates_path, usecols=['GAME_ID', 'date', 'season', 'playoffs'])
    dates = dates[dates.playoffs.eq(False)].copy()
    dates['game_id'] = dates.GAME_ID.astype(str).str.zfill(10)
    dates['game_date'] = pd.to_datetime(dates.date.astype(str), format='%Y%m%d')
    assert dates.groupby('game_id').game_date.nunique().eq(1).all()
    date_map = dates.drop_duplicates('game_id').set_index('game_id').game_date
    older = [historical_minutes(year, date_map) for year in range(2002, 2019)]
    games = pd.concat([*older, games], ignore_index=True)
    games.to_csv(OUT / 'rs_player_game_minutes_2002_2026.csv', index=False)
    assert not games.duplicated(['player_id', 'game_id']).any()
    assert games.game_date.notna().all()
    games['game_date'] = pd.to_datetime(games.game_date).dt.normalize()
    snapshots = pd.read_csv(snapshots_path, parse_dates=['requested_date', 'provider_effective_date', 'provider_last_player_game_date'])
    cols = ['player_id', 'requested_date', 'provider_effective_date', 'provider_last_player_game_date', 'epm', 'snapshot_source']
    joined = pd.merge_asof(games.sort_values('game_date'), snapshots[cols].sort_values('requested_date'),
                           left_on='game_date', right_on='requested_date', by='player_id', direction='backward',
                           allow_exact_matches=False)
    joined['snapshot_age_days'] = (joined.game_date - joined.requested_date).dt.days
    joined['date_gate_pass'] = (joined.requested_date.lt(joined.game_date)
                                & joined.provider_effective_date.le(joined.requested_date)
                                & (joined.provider_last_player_game_date.lt(joined.game_date)
                                   | joined.provider_last_player_game_date.isna()))
    joined['covered'] = joined.date_gate_pass & joined.epm.notna() & joined.snapshot_age_days.le(35)
    joined['covered_minutes'] = joined.minutes.where(joined.covered, 0)
    joined['epm_minutes'] = (joined.epm * joined.minutes).where(joined.covered, 0)
    joined['covered_snapshot_age'] = joined.snapshot_age_days.where(joined.covered)
    target = joined.groupby(['player_id', 'season'], as_index=False).agg(
        rs_game_minutes=('minutes', 'sum'), covered_minutes=('covered_minutes', 'sum'),
        epm_minutes=('epm_minutes', 'sum'), rs_gp_from_game_lines=('game_id', 'nunique'),
        target_through_date=('game_date', 'max'), first_game_date=('game_date', 'min'),
        max_covered_snapshot_age_days=('covered_snapshot_age', 'max'))
    target['target_epm'] = target.epm_minutes / target.covered_minutes.where(target.covered_minutes.gt(0))
    target['coverage_fraction'] = target.covered_minutes / target.rs_game_minutes
    target['target_kind'] = 'rs_minutes_weighted_pregame_predictive_epm_step_35d'
    target['coverage_at_least_95pct'] = target.coverage_fraction.ge(.95)
    # Entire target-season closure, rather than a player's last game, determines
    # when a season-level workload/target observation has fully matured.
    target['season_rs_last_game_date'] = target.season.map(games.groupby('season').game_date.max())
    target['target_available_not_before'] = target.season_rs_last_game_date + pd.Timedelta(days=1)
    teams = pd.read_csv(OUT / 'team_seasons.csv')
    expected_games = teams.groupby('season').actual_team_games.sum() / 2
    actual_games = games.groupby('season').game_id.nunique()
    target['season_game_coverage'] = target.season.map(actual_games / expected_games)
    target['season_game_schedule_complete'] = target.season.map(actual_games.eq(expected_games))
    target.to_csv(OUT / 'rs_pregame_epm_targets_2002_2026.csv', index=False)
    target[target.season.ge(2019)].to_csv(OUT / 'rs_pregame_epm_targets_2019_2026.csv', index=False)
    joined.to_parquet(OUT / 'rs_pregame_epm_game_audit.parquet', index=False)
    coverage = target.groupby('season').agg(players=('player_id', 'size'), minutes=('rs_game_minutes', 'sum'),
                                           covered_minutes=('covered_minutes', 'sum'),
                                           players_95pct=('coverage_at_least_95pct', 'sum'))
    coverage['coverage_fraction'] = coverage.covered_minutes / coverage.minutes
    coverage.to_csv(OUT / 'rs_pregame_epm_target_coverage.csv')
    receipt = {'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in [games_path, snapshots_path, dates_path,
                                    *[ROOT / f'nba_pipeline/raw_data/NBA{y % 100:02d}.parquet' for y in range(2002, 2019)]]],
               'rows': len(target), 'game_rows': len(joined), 'max_snapshot_age_days': 35,
               'date_gate_failed_game_rows': int((joined.epm.notna() & ~joined.date_gate_pass).sum()),
               'future_requested_date_rows': int(joined.requested_date.gt(joined.game_date).sum()),
               'method': 'Nearest available player snapshot requested strictly BEFORE game date, held constant until next snapshot; reject >35-day old requests.',
               'limitations': ['Predictive-state proxy, not realized player impact; model can learn provider persistence.',
                               'Nonuniform archive frequency; preserve source dates and run monthly-only sensitivity.',
                               'Minutes from PBP allocation; use official season GP/MIN for workload labels.',
                               '95% coverage is a transparent screen, not a statistically selected optimal threshold.',
                               'Original vendor model vintage unavailable; provider date contract is relied on.',
                               '2011 lacks one raw game; season_game_schedule_complete must gate complete-season targets.',
                               'Historical PBP lineup allocation has larger individual minute errors; census audit supplies scale.']}
    (OUT / 'rs_pregame_epm_target_provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(coverage.to_string())
    print(json.dumps({k: v for k, v in receipt.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()

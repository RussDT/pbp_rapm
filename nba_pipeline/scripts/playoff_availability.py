"""Retrospective playoff participation; cross-fitted imputation, not injury odds."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

SEASON_FIELDS = [
    'playoff_games_played', 'team_playoff_games', 'playoff_participation_actual',
    'playoff_participation_estimated', 'playoff_participation_used',
    'playoff_participation_source', 'playoff_participation_team_source',
    'playoff_participation_team', 'playoff_participation_prior_games',
    'playoff_participation_history_games', 'participation_scaled_title_lift_v2',
    'participation_scaled_title_lift_v1', 'participation_scaled_playoff_prior_title_lift',
]
CAREER_FIELDS = ['participation_scaled_title_lift_v2_championships',
                 'participation_scaled_title_lift_v1_championships',
                 'participation_scaled_playoff_prior_title_lift_championships',
                 'participation_seasons_covered', 'participation_seasons_estimated',
                 'participation_seasons_inferred_roster']
K_GRID = np.array([0., 1., 2., 4., 8., 16., 32., 64., 128., 256.])


def game_ids(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)


def count_participation(raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One appearance per exact player/team/game; raw lineups retain brief stints."""
    raw = raw.copy(); raw['game_id'] = game_ids(raw['game_id'])
    games = games.copy(); games['game_id'] = game_ids(games['game_id'])
    if games['game_id'].duplicated().any():
        raise ValueError('Duplicate playoff schedule games')
    parts = []
    for side in ['home', 'away']:
        ids = raw[['game_id'] + [f'{side}_player{i}' for i in range(1, 6)]].melt(
            id_vars='game_id', value_name='player_id').drop(columns='variable')
        ids['player_id'] = pd.to_numeric(ids.player_id, errors='coerce')
        ids = ids.loc[ids.player_id.gt(0)].drop_duplicates()
        schedule = games[['game_id', f'{side}_team_abbr']].rename(
            columns={f'{side}_team_abbr': 'team_abbr'})
        parts.append(ids.merge(schedule, on='game_id', validate='many_to_one'))
    appearances = pd.concat(parts).drop_duplicates(['game_id', 'team_abbr', 'player_id'])
    return appearances.groupby(['player_id', 'team_abbr'], as_index=False).agg(
        playoff_games_played=('game_id', 'nunique'))


def build_participation_evidence(seasons, games, regular_roster, raw_dir, roster_path):
    evidence = []
    verified = pd.DataFrame(json.loads(Path(roster_path).read_text()),
                            columns=['season', 'player_id', 'team_abbr', 'source_url', 'note'])
    if verified.duplicated(['season', 'player_id']).any():
        raise ValueError('Duplicate verified playoff roster overrides')
    for year in sorted(seasons.season.unique()):
        schedule = games.loc[games.season.eq(year)]
        counts = pd.concat([schedule.home_team_abbr, schedule.away_team_abbr]).value_counts()
        raw = pd.read_parquet(Path(raw_dir) / f'NBA{int(year)%100:02d}_PS.parquet',
            columns=['game_id'] + [f'{s}_player{i}' for s in ['home','away'] for i in range(1,6)])
        played = count_participation(raw, schedule)
        played = played.loc[played.player_id.isin(seasons.loc[seasons.season.eq(year), 'player_id'])]
        if played.player_id.duplicated().any():
            raise ValueError(f'Modeled player appears on multiple playoff teams in {year}; audit raw identities')
        played = played.set_index('player_id')
        rs = regular_roster.loc[regular_roster.season.eq(year)].sort_values(
            ['last_game_date', 'player_exposures'], kind='stable').drop_duplicates('player_id', keep='last').set_index('player_id')
        cert = verified.loc[verified.season.eq(year)].set_index('player_id')
        for row in seasons.loc[seasons.season.eq(year)].itertuples():
            pid = row.player_id; team = None; team_source = 'unresolved'; n = np.nan; d = np.nan
            if pid in played.index:
                team = played.loc[pid, 'team_abbr']; n = played.loc[pid, 'playoff_games_played']; team_source = 'observed_playoff'
            elif pid in cert.index:
                team = cert.loc[pid, 'team_abbr']; n = 0; team_source = 'verified_roster'
            elif pid in rs.index and rs.loc[pid, 'team_abbr'] == row.team_abbr:
                team = row.team_abbr; team_source = 'regular_season_team_proxy'
                if team in counts.index:
                    n = 0
            if team in counts.index:
                d = int(counts[team]); source = 'actual' if team_source != 'regular_season_team_proxy' else 'inferred_roster'
            elif team is not None:
                source = 'estimated_nonplayoff'
            else:
                source = 'unresolved_roster'
            evidence.append(dict(season=year, player_id=pid, playoff_games_played=n,
                team_playoff_games=d, playoff_participation_actual=n/d if d>0 else np.nan,
                playoff_participation_source=source, playoff_participation_team_source=team_source,
                playoff_participation_team=team))
    result = pd.DataFrame(evidence)
    if result.duplicated(['season','player_id']).any():
        raise ValueError('Participation requires unique player-seasons')
    return result


def predict_from_history(target, training, k):
    """Training excludes the entire target year, including league baseline."""
    denominator = training.team_playoff_games.sum()
    if denominator <= 0:
        return np.full(len(target), np.nan), np.zeros(len(target)), np.nan
    league = training.playoff_games_played.sum() / denominator
    totals = training.groupby('player_id')[['playoff_games_played','team_playoff_games']].sum()
    successes = target.player_id.map(totals.playoff_games_played).fillna(0).to_numpy()
    exposure = target.player_id.map(totals.team_playoff_games).fillna(0).to_numpy()
    estimate = np.full(len(target), league, dtype=float)
    np.divide(successes + k*league, exposure+k, out=estimate, where=exposure+k>0)
    return estimate, exposure, float(league)


def add_participation_adjustment(seasons, evidence):
    seasons = seasons.drop(columns=[c for c in SEASON_FIELDS if c in seasons], errors='ignore')
    output = seasons.merge(evidence, on=['season','player_id'], how='left', validate='one_to_one')
    # Do not teach the predictor that an uncertified roster proxy is an observed absence.
    known = evidence.loc[evidence.team_playoff_games.notna()]
    if not (known.team_playoff_games.gt(0) & known.playoff_games_played.ge(0)
            & known.playoff_games_played.le(known.team_playoff_games)).all():
        raise ValueError('Invalid observed playoff game counts')
    observed = evidence.loc[evidence.playoff_participation_source.eq('actual')].copy()
    output['playoff_participation_estimated'] = np.nan
    output['playoff_participation_history_games'] = np.nan
    output['playoff_participation_prior_games'] = np.nan
    diagnostics=[]
    for year, target in output.groupby('season'):
        training = observed.loc[observed.season.ne(year)]
        losses=[]
        for inner_year, validation in training.groupby('season'):
            inner = training.loc[training.season.ne(inner_year)]
            if inner.empty: continue
            losses.append([np.mean((predict_from_history(validation,inner,k)[0]-validation.playoff_participation_actual.to_numpy())**2) for k in K_GRID])
        if losses:
            scores=np.mean(losses,axis=0); best=np.flatnonzero(np.isclose(scores, np.nanmin(scores), rtol=0, atol=1e-12))[-1]; k=float(K_GRID[best])
        else:
            scores=np.full(len(K_GRID),np.nan); k=float(K_GRID[-1])
        estimate, history, league=predict_from_history(target,training,k)
        output.loc[target.index,'playoff_participation_estimated']=estimate
        output.loc[target.index,'playoff_participation_history_games']=history
        output.loc[target.index,'playoff_participation_prior_games']=k
        for value,score in zip(K_GRID,scores):
            diagnostics.append(dict(held_out_season=int(year),prior_games=float(value),inner_season_mean_mse=float(score),selected=value==k,league_baseline=league))
    # Unknown roster is not silently converted into a nonplayoff season.
    output['playoff_participation_used']=output.playoff_participation_actual
    nonplayoff=output.playoff_participation_source.eq('estimated_nonplayoff')
    output.loc[nonplayoff,'playoff_participation_used']=output.loc[nonplayoff,'playoff_participation_estimated']
    for source,target in [('title_lift_over_contract_v2','participation_scaled_title_lift_v2'),
                          ('title_lift_over_contract','participation_scaled_title_lift_v1'),
                          ('playoff_adjusted_title_lift_over_contract_v2','participation_scaled_playoff_prior_title_lift')]:
        output[target]=output[source]*output.playoff_participation_used
    return output, pd.DataFrame(diagnostics)


def career_participation_summary(seasons):
    return seasons.assign(_estimated=seasons.playoff_participation_source.eq('estimated_nonplayoff'),
        _inferred=seasons.playoff_participation_source.eq('inferred_roster')).groupby('player_id',as_index=False).agg(
        participation_scaled_title_lift_v2_championships=('participation_scaled_title_lift_v2',lambda x:x.sum(min_count=1)),
        participation_scaled_title_lift_v1_championships=('participation_scaled_title_lift_v1',lambda x:x.sum(min_count=1)),
        participation_scaled_playoff_prior_title_lift_championships=('participation_scaled_playoff_prior_title_lift',lambda x:x.sum(min_count=1)),
        participation_seasons_covered=('participation_scaled_title_lift_v2','count'),
        participation_seasons_estimated=('_estimated','sum'),
        participation_seasons_inferred_roster=('_inferred','sum'))


def refresh_participation(seasons, games, regular_roster, raw_dir, roster_path):
    evidence=build_participation_evidence(seasons,games,regular_roster,raw_dir,roster_path)
    result,cv=add_participation_adjustment(seasons,evidence)
    actual=result.loc[result.playoff_participation_source.eq('actual')].copy()
    baseline=cv.loc[cv.selected].set_index('held_out_season').league_baseline
    model_mse=actual.assign(error=(actual.playoff_participation_actual-actual.playoff_participation_estimated)**2).groupby('season').error.mean().mean()
    league_mse=actual.assign(error=(actual.playoff_participation_actual-actual.season.map(baseline))**2).groupby('season').error.mean().mean()
    report=dict(status='retrospective participation sensitivity, not injury title probability loss',
        model='nested leave-one-season-out shrinkage toward game-weighted league participation',
        selection='minimum inner equal-season mean player-season squared error; ties favor stronger shrinkage',
        future_history='other career seasons, including later years; not a past-only forecast',
        source_counts=result.playoff_participation_source.value_counts().to_dict(),
        outer_season_mean_mse=float(model_mse), league_only_outer_season_mean_mse=float(league_mse),
        roster_training='uncertified zero-appearance roster proxies excluded from predictor fitting; observed-participant selection limits the fallback',
        roster_policy='observed playoff team or cited roster overrides; matching latest RS team and focal team is an explicit proxy; other assignments unresolved',
        adjustment='season lift times participation, including signed negative lift; a participation scaling, not a contract-cost penalty',
        career_rule='sum adjusted seasons; missing stays missing; report coverage')
    return result, cv, report

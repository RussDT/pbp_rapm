"""Research measurement layer: six-foot contest savings and assignment burden.

Never promotes estimates to the current skill table or VPM. Provider normal
shooting percentages are retrospective; no dated predictor is made from them.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'nba_pipeline/results/estimated_skills/rim_defense_matchups'
MATCHUPS = ROOT.parent / 'vpm/vpm/data/matchup/matchups_two_rs_2018_2026.parquet'


def rim_values(attempts, makes, baseline):
    return 2 * (attempts * baseline - makes)


def assignment_summary(pairs, strength):
    """Strength is frozen before assignment season; uncovered opponents stay missing."""
    p = pairs.merge(strength, on=['year', 'nba_id_off'], how='left', validate='many_to_one')
    p['covered'] = p.PARTIAL_POSS.where(p.pts100.notna(), 0)
    p['pts_n'] = p.covered * p.pts100.fillna(0)
    p['ast_n'] = p.covered * p.ast100.fillna(0)
    g = p.groupby(['year', 'nba_id_def']).agg(assignment_poss=('PARTIAL_POSS','sum'),
        covered_poss=('covered','sum'), pts_n=('pts_n','sum'), ast_n=('ast_n','sum'))
    g['opponent_scoring_load100'] = g.pts_n / g.covered_poss.replace(0, np.nan)
    g['opponent_assist_load100'] = g.ast_n / g.covered_poss.replace(0, np.nan)
    g['strength_coverage'] = g.covered_poss / g.assignment_poss
    for metric, numerator in [('scoring','pts_n'), ('assist','ast_n')]:
        league = g.groupby('year')[numerator].transform('sum') / g.groupby('year').covered_poss.transform('sum')
        g[f'{metric}_burden_above_league100'] = g[f'opponent_{metric}_load100'] - league
    return g.drop(columns=['pts_n','ast_n']).reset_index().rename(columns={'year':'season','nba_id_def':'player_id'})


def recover_cumulative_counts(tracking, appearances):
    """Difference season/player snapshots only across consecutive known appearances.

    Never difference percentages. Missing appearances and invalid increments stay
    missing; their snapshots can still anchor the following consecutive game.
    """
    keys = ['season', 'season_type', 'player_id']
    t = tracking.sort_values(keys + ['game_date']).copy()
    if t.duplicated(keys + ['game_date']).any():
        raise ValueError('Duplicate cumulative snapshot')
    a = appearances.loc[appearances.minutes.gt(0)].sort_values(keys + ['game_date']).copy()
    if a.duplicated(keys + ['game_date']).any():
        raise ValueError('Ambiguous player appearance dates')
    a['previous_appearance'] = a.groupby(keys).game_date.shift()
    if 'team_id' in a:
        changed = a.team_id.ne(a.groupby(keys).team_id.transform('first')) | a.team_id.isna()
        a['after_team_change'] = changed.groupby([a[k] for k in keys]).cummax()
    else:
        a['after_team_change'] = False
    t['previous_snapshot'] = t.groupby(keys).game_date.shift()
    t = t.merge(a[keys + ['game_date', 'previous_appearance', 'after_team_change']], on=keys + ['game_date'],
                how='left', validate='one_to_one', indicator='_appearance')
    consecutive = (t.previous_snapshot.eq(t.previous_appearance) |
                   (t.previous_snapshot.isna() & t.previous_appearance.isna())) & t._appearance.eq('both')
    for col in ['rim_dfga', 'rim_dfgm']:
        t['cumulative_' + col] = t[col]
        previous = t.groupby(keys)[col].shift()
        t[col] = t[col] - previous.where(t.previous_snapshot.notna(), 0)
    valid = (consecutive & ~t.after_team_change.fillna(True).astype(bool) & t.rim_dfga.ge(0) & t.rim_dfgm.ge(0) &
             t.rim_dfgm.le(t.rim_dfga) & t.rim_dfga.mod(1).eq(0) & t.rim_dfgm.mod(1).eq(0))
    t['recovery_status'] = np.where(valid, 'recovered_consecutive_snapshot',
                                  np.where(consecutive, 'invalid_count_increment', 'appearance_gap'))
    t.loc[t.after_team_change.fillna(False).astype(bool), 'recovery_status'] = 'post_trade_unverified_cumulative'
    t.loc[~valid, ['rim_dfga', 'rim_dfgm']] = np.nan
    # A season-to-date opponent baseline is not a single-game baseline.
    t['provider_normal_pct'] = np.nan
    t['field_family'] = 'cumulative_lt6ft_differenced'
    return t.drop(columns=['_appearance'])


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    receipts, rows = [], []
    pbp_path = ROOT / "nba_pipeline/results/estimated_skills/player_game_lines.parquet"
    appearances = pd.read_parquet(pbp_path)
    context_path = ROOT / "nba_pipeline/results/estimated_skills/game_context.parquet"
    context = pd.read_parquet(context_path)[["game_id", "player_id", "team_id"]]
    appearances = appearances.merge(context, on=["game_id", "player_id"], how="left", validate="one_to_one")
    appearances = appearances.loc[~appearances.duplicated(["season", "season_type", "player_id", "game_date"], keep=False)]
    for y in range(2019, 2027):
        for suffix in ['', 'ps']:
            path = ROOT / f'player_sheets/game_report/year_files/{y}{suffix}_games.csv'
            keep = ['PLAYER_ID','PLAYER_NAME','date','MIN','AST','FGA','TEAM_ID']
            d = pd.read_csv(path, usecols=lambda c: c in keep or 'LT_06' in c, low_memory=False)
            receipts.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            d['season'], d['season_type'] = y, 'ps' if suffix else 'rs'
            # Select one complete field family per file; never splice a baseline
            # from one endpoint into attempts from another endpoint.
            prefixes = ['lt6ft_totals_', 'less_6ft_def_', '']
            prefix = next((p for p in prefixes if all(p+c in d for c in ['FGA_LT_06','FGM_LT_06','NS_LT_06_PCT'])), None)
            if prefix is None:
                raise ValueError(f'No explicit defensive six-foot family in {path}')
            d['field_family'] = prefix or 'cumulative_pending_validation'
            for target, c in [('rim_dfga','FGA_LT_06'),('rim_dfgm','FGM_LT_06'),('provider_normal_pct','NS_LT_06_PCT')]:
                d[target] = pd.to_numeric(d[prefix+c], errors='coerce')
            d['game_date'] = pd.to_datetime(d.date.astype(str), format='%Y%m%d')
            d = d.rename(columns={'PLAYER_ID':'player_id','PLAYER_NAME':'tracking_name'})
            if prefix == '':
                if y not in [2020, 2021, 2022] or suffix:
                    raise ValueError('Unvalidated cumulative source family')
                d = recover_cumulative_counts(d, appearances)
            rows.append(d[['season','season_type','player_id','game_date','tracking_name','MIN','AST','FGA','TEAM_ID','field_family','rim_dfga','rim_dfgm','provider_normal_pct'] + [c for c in ['recovery_status','cumulative_rim_dfga','cumulative_rim_dfgm'] if c in d]])
    t = pd.concat(rows, ignore_index=True)
    keys = ['season','season_type','player_id','game_date']
    if t.duplicated(keys).any():
        raise ValueError('Tracking duplicate keys')
    pbp_path = ROOT / 'nba_pipeline/results/estimated_skills/player_game_lines.parquet'
    p = pd.read_parquet(pbp_path)
    ambiguous = p.duplicated(keys, keep=False)
    p = p.loc[~ambiguous].copy()
    d = p.merge(t, on=keys, how='left', validate='one_to_one')
    d['box_join_valid'] = d.FGA.eq(d.fga) & d.AST.eq(d.ast) & (d.MIN-d.minutes).abs().lt(.5)
    d['rim_counts_valid'] = (d.box_join_valid & d.def_poss.gt(0) & d.rim_dfga.ge(0)
        & d.rim_dfgm.ge(0) & d.rim_dfgm.le(d.rim_dfga))
    # An entire team/date with no reported contests is not measured zero activity.
    # Recovered cumulative files carry retrospective TEAM_ID; use PBP team
    # identity for their team/date outage gate. Preserve original daily sources.
    dated_teams = appearances.set_index(['game_id', 'player_id']).team_id
    recovered = d.field_family.eq('cumulative_lt6ft_differenced')
    d['rim_team_id'] = d.TEAM_ID
    d.loc[recovered, 'rim_team_id'] = dated_teams.reindex(
        pd.MultiIndex.from_frame(d.loc[recovered, ['game_id', 'player_id']])).to_numpy()
    team_total = d.groupby(['game_date','rim_team_id']).rim_dfga.transform('sum')
    d['rim_counts_valid'] &= team_total.gt(0)
    audit = d.loc[d.rim_counts_valid].groupby('season')[['rim_dfga','def_poss']].sum()
    if (audit.rim_dfga / audit.def_poss).gt(.5).any():
        raise ValueError('Implausible season contest volume; inspect repeated aggregate fields')
    d['rim_baseline_valid'] = d.rim_counts_valid & (d.provider_normal_pct.between(0,1) | d.rim_dfga.eq(0))
    d['rim_expected_makes'] = (d.rim_dfga * d.provider_normal_pct).where(d.rim_dfga.ne(0), 0).where(d.rim_baseline_valid)
    d['rim_points_saved'] = (2*(d.rim_expected_makes-d.rim_dfgm)).where(d.rim_baseline_valid)
    d['rim_points_saved100'] = 100*d.rim_points_saved/d.def_poss
    d.to_parquet(OUT/'player_games.parquet', index=False)
    valid = d.loc[d.rim_baseline_valid].copy()
    g = valid.groupby(['season','season_type','player_id']).agg(
        games=('game_id','size'), covered_def_poss=('def_poss','sum'),
        rim_dfga=('rim_dfga','sum'),rim_dfgm=('rim_dfgm','sum'),
        expected_makes=('rim_expected_makes','sum'),rim_points_saved=('rim_points_saved','sum'))
    g['rim_dfga100'] = 100*g.rim_dfga/g.covered_def_poss
    g['rim_points_saved100'] = 100*g.rim_points_saved/g.covered_def_poss
    g['rim_allowed_pct'] = 100*g.rim_dfgm/g.rim_dfga.replace(0,np.nan)
    g['opponent_baseline_pct'] = 100*g.expected_makes/g.rim_dfga.replace(0,np.nan)
    g['saved_per100_contests'] = 100*g.rim_points_saved/g.rim_dfga.replace(0,np.nan)
    total = d.groupby(['season','season_type','player_id']).def_poss.sum().rename('total_def_poss')
    g = g.join(total)
    g['rim_coverage'] = g.covered_def_poss/g.total_def_poss
    g.reset_index().to_csv(OUT/'rim_seasons.csv',index=False)
    # Homegrown preseason scoring/passing projections from the existing filter,
    # not the opponent's results against the defender being evaluated.
    pred_path = ROOT/'nba_pipeline/results/estimated_skills/predictions.parquet'
    pred = pd.read_parquet(pred_path)
    print('prediction variants',pred.variant.unique(), flush=True)
    pred = pred.loc[pred.skill.isin(['pts_rate','ast_rate']) & pred.variant.eq('uncentered')].copy()
    if pred.empty:
        raise ValueError('Expected uncentered skill predictions')
    strength = pred.sort_values('game_date').groupby(['season','player_id','skill']).tail(1)
    strength = strength.pivot(index=['season','player_id'],columns='skill',values='post_rate').reset_index()
    strength['year'] = strength.season+1
    strength = strength.rename(columns={'player_id':'nba_id_off','pts_rate':'pts100','ast_rate':'ast100'})
    strength[['pts100','ast100']] *= 100
    pairs = pd.read_parquet(MATCHUPS)
    if pairs.duplicated(['year','nba_id_off','nba_id_def']).any():
        raise ValueError('Duplicate matchup pairs')
    if not np.isfinite(pairs.PARTIAL_POSS).all() or pairs.PARTIAL_POSS.lt(0).any():
        raise ValueError('Invalid matchup exposure')
    burden = assignment_summary(pairs, strength[['year','nba_id_off','pts100','ast100']])
    burden.to_csv(OUT/'matchup_seasons.csv',index=False)
    names = d.sort_values('game_date').groupby('player_id').tracking_name.last().rename('player_name')
    board = g.reset_index().query("season == 2026 and season_type == 'rs'").merge(
        burden.query('season == 2026'), on=['season','player_id'],how='outer',validate='one_to_one').join(names,on='player_id')
    board['matchup_source_status'] = '2026 incomplete season snapshot; date unavailable'
    board.to_csv(OUT/'current_research.csv',index=False)
    display = ['player_name','games','rim_dfga100','rim_allowed_pct','opponent_baseline_pct','rim_points_saved100','rim_coverage','scoring_burden_above_league100','assist_burden_above_league100','strength_coverage']
    shown = board.loc[board.covered_def_poss.ge(1000) & board.rim_coverage.ge(.8)].sort_values('rim_points_saved100',ascending=False)
    html = '<meta charset="utf-8"><title>Rim defense and matchup burden</title><style>body{font:15px system-ui;margin:32px;background:#111;color:#eee}td,th{padding:8px;text-align:right}th{position:sticky;top:0;background:#222}input{padding:10px;width:300px}</style><h1>Rim defense and matchup burden</h1><p>Research · 2026 regular season · observed contests within six feet. Positive saved is better; positive burden means tougher scoring/passing assignments.</p><p>Matchup archive is incomplete for 2026. Opponent strengths are frozen from prior-season homegrown estimates. These are descriptive measurements, not validated predictive skills. Rim rates use covered possessions only. Display requires 1,000 covered defensive possessions and 80% rim coverage. Coverage is a fraction. Matchup burden uses scoring/passing volume, not total offensive impact.</p><input placeholder="Search player" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))">'
    html += shown[display].to_html(index=False,float_format=lambda x:f'{x:.3f}',na_rep='—')
    (OUT/'index.html').write_text(html)
    coverage = d.groupby(['season','season_type']).agg(rows=('game_id','size'),valid_counts=('rim_counts_valid','sum'),valid_savings=('rim_baseline_valid','sum'))
    coverage.to_csv(OUT/'coverage.csv')
    manifest = {'status':'research measurement layer; not predictive admission','rim_definition':'NBA defense less than six feet', 'formula':'2*(FGA*provider_normal_pct-FGM)', 'baseline_timing':'provider season-normal percentage is not verified as-of; retrospective only','matchup_definition':'partial-possession-weighted prior-season homegrown scoring and assist rates, centered on common covered assignments', 'matchup_2026':'incomplete source; no exact extraction date', 'strength_timing':'observations prior season only; legacy filter hyperparameters historically tuned through 2025', 'ambiguous_pbp_rows_excluded':int(ambiguous.sum()),'sources':receipts}
    for path in [pbp_path,pred_path,MATCHUPS,context_path,Path(__file__)]:
        manifest['sources'].append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(coverage.to_string()); print(shown[display].head(10).to_string(index=False))

if __name__ == '__main__':
    main()

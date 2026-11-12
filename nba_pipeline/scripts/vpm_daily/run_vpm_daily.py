#!/usr/bin/env python3
"""Build one dated full-feature VPM board from the Cloudflare R2 feeds.

The date is an argument; everything date-shaped comes from it or from that
run's own snapshot. No model is fitted. Any failed gate stops the run before a
board can be published. See nba_pipeline/docs/VPM_DAILY_RUNNER.md.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render import date_values, render  # noqa: E402

PIPE = HERE.parents[1]
E = PIPE / 'results/estimated_skills'
VPM = PIPE.parents[1] / 'vpm'
SCRIPTS = PIPE / 'scripts'
SEASON = 2027
NOV3_INPUTS = ['refresh.json', 'game_manifest.json', 'source_inventory.json', 'player_core.parquet',
               'team_core.parquet', 'player_report.parquet', 'team_report.parquet',
               'player_game_catalog.json', 'player_game_report_catalog.json']
sys.path[:0] = [str(SCRIPTS), str(VPM)]
ENV = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(SCRIPTS), str(VPM)])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def say(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def dirs(tag):
    return dict(base=E/f'current_refresh_{tag}_v1', track=E/f'current_refresh_{tag}_tracking_v1',
                sq=E/f'current_sq_{tag}_v1', emit=E/f'current_refresh_{tag}_sq_v1',
                board=E/f'current_refresh_{tag}_sq_board_v1')


def run(script, args, log):
    say('started', script=Path(script).name)
    with Path(log).open('x') as handle:
        subprocess.run([sys.executable, str(script), *map(str, args)], stdout=handle,
                       stderr=subprocess.STDOUT, check=True, env=ENV, cwd=PIPE)
    say('complete', script=Path(script).name)


def write_rendered(name, destination, values):
    text = render((HERE/'templates'/(name+'.tmpl')).read_text(), values)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        handle.write(text)
    return destination


def espn_finals(day):
    response = requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
                            params=dict(dates=day.strftime('%Y%m%d')), timeout=30)
    response.raise_for_status()
    states = [e['status']['type']['name'] for e in response.json()['events']]
    return states.count('STATUS_FINAL'), len(states)


def r2_game_counts():
    from databallr_r2_read import R2Reader
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'manifest.json'
        R2Reader().download('databallr-datasets', f'nba/pbp-analytics/v1/{SEASON}/rs/manifest.json', path)
        return pd.Series([int(g['date']) for g in json.loads(path.read_bytes())['games']]).value_counts()


def wait_until_complete(cut, hours):
    """Last night counts as complete when R2 holds every game ESPN calls final."""
    day = cut-pd.Timedelta(days=1)
    deadline = time.monotonic()+hours*3600
    while True:
        finals, scheduled = espn_finals(day)
        have = int(r2_game_counts().get(int(day.strftime('%Y%m%d')), 0))
        say('completeness', day=str(day.date()), espn_final=finals, espn_scheduled=scheduled, r2_games=have)
        if finals == scheduled and have == finals:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f'R2 has {have} games for {day.date()}; ESPN has {finals}/{scheduled} final')
        time.sleep(600)


def clear_unfinished(paths):
    """An interrupted attempt is rebuilt; a finished board is never touched."""
    if (paths['emit']/'result.json').exists():
        raise FileExistsError('A finished board already exists for this date: '+str(paths['emit']))
    for path in paths.values():
        if path.exists():
            say('clearing_unfinished_attempt', path=path.name)
            shutil.rmtree(path)


def snapshot(paths, cut, values, hours, attempts=3):
    base = paths['base']
    for attempt in range(1, attempts+1):
        script = write_rendered('cloud_refresh.py', base/'pipeline_code/cloud_refresh.py', values)
        run(script, ['--out', base, '--cut', values['CUT'], '--season', SEASON], base/'cloud_refresh.log')
        refresh = json.loads((base/'refresh.json').read_bytes())
        if not refresh['player_missing_game_ids'] and not refresh['team_missing_game_ids']:
            return refresh
        say('incomplete_logs', attempt=attempt, player=refresh['player_missing_game_ids'],
            team=refresh['team_missing_game_ids'])
        shutil.rmtree(base)
        if attempt < attempts:
            time.sleep(1800)
    raise RuntimeError('Player/team logs stayed incomplete')


def copy_saved_snapshot(saved, paths):
    """Reproduction only: start from a saved R2 snapshot instead of downloading."""
    base = paths['base']
    (base/'native_source').mkdir(parents=True)
    for name in NOV3_INPUTS:
        shutil.copy2(saved/name, base/name)
    shutil.copytree(saved/'objects', base/'objects')
    shutil.copytree(saved/'native_source/games', base/'native_source/games')
    text = (base/'refresh.json').read_text().replace(str(saved), str(base))
    (base/'refresh.json').write_text(text)
    return json.loads(text)


def emit(paths, values, refresh, base_values):
    B, T, Q, S = paths['base'], paths['track'], paths['sq'], paths['emit']
    for name in ['run_refresh.py', 'refresh_vpm_native_source.py', 'native_atom_adapter.py',
                 'prepare_current_vpm_features.py', 'prepare_current_vpm_tracking.py']:
        write_rendered(name, B/'pipeline_code'/name, values)
    run(B/'pipeline_code/run_refresh.py', [], B/'run.log')
    Q.mkdir()
    script = write_rendered('score_current_sq.py', Q/'score_current_sq.py', values)
    run(script, [], Q/'actual_inference.log')
    S.mkdir()
    (S/'native_source/targets').mkdir(parents=True)
    for name in ['refresh.json', 'player_core.parquet', 'objects', 'current_reference_core_native_scale.parquet',
                 'report_snapshot.json', 'tracking_refresh.json', 'measurements.parquet', 'defense',
                 'current_contested_split_measurements.parquet']:
        (S/name).symlink_to(T/name, target_is_directory=(T/name).is_dir())
    shutil.copytree(T/'current_code_native_scale', S/'current_code_native_scale')
    for name in ['NBA27_current.parquet', 'RAPM27_current.parquet', 'games', 'score_comparison.csv']:
        (S/'native_source'/name).symlink_to(B/'native_source'/name, target_is_directory=(B/'native_source'/name).is_dir())
    (S/'native_source/targets/games_2027_RS.parquet').symlink_to(B/'native_source/targets/games_2027_RS.parquet')
    (S/'sq_atom_pipeline/raw_data').mkdir(parents=True)
    (S/'sq_atom_pipeline/raw_data/NBA27.parquet').symlink_to(B/'native_source/NBA27_current.parquet')
    from rebuild_monthly_sq_atoms import rebuild_one
    report = rebuild_one((S/'sq_atom_pipeline', B/'native_source/targets', S/'native_source/targets',
                          Q/'clean_ev.parquet', SEASON, 'RS'))
    if report['actual_parent_max_change'] != 0 or report['other_atom_fields_max_change'] != 0:
        raise ValueError('SQ atoms changed an actual or non-SQ field')
    receipt = json.loads((Q/'receipt.json').read_bytes())
    pins = dict(base_values, SQ_ROWS=receipt['scored_rows'], SQ_RECEIPT_SHA=sha(Q/'receipt.json'),
                SQ_EV_SHA=sha(Q/'clean_ev.parquet'))
    script = write_rendered('emit_current_vpm_sq_refresh.py', S/'emit_current_vpm_sq_refresh.py', pins)
    shutil.copy2(B/'pipeline_code/prepare_current_vpm_tracking.py', S/'prepare_current_vpm_tracking.py')
    run(script, [S, '--tracking-refresh'], S/'emit.log')
    return json.loads((S/'result.json').read_bytes())


def build_preview(paths, values, result):
    S, cut = paths['emit'], values['CUT']
    games = int(values['GAMES'])
    if not (result['current_season_sq_available'] and result['current_tracking_reports_available']):
        raise ValueError('Board lacks current tracking or SQ')
    if result['games_appended'] != games or result['estimate_date'] != cut:
        raise ValueError('Board date or game count differs')
    if result['normal_equation_error'] > 1e-10 or result['native_net_gap'] > 1e-9:
        raise ValueError('Board fails its numerical closure gates')
    if result['new_mapper_fits'] != 0 or result['new_objective_tuning'] != 0 or result['selected_pointer_changed']:
        raise ValueError('Board changed a frozen choice')
    d = pd.read_csv(Path(result.get('rankings_path', S/'current_rankings.csv')), float_precision='round_trip').rename(columns={'player_id': 'nba_id'})
    if not d.nba_id.is_unique or len(d) != result['recent_players'] or not d.estimate_date.eq(cut).all():
        raise ValueError('Rankings identity differs')
    if (d.ovpm+d.dvpm-d.vpm).abs().max() >= 1e-9:
        raise ValueError('O+D does not equal total')
    d.insert(0, 'rank', range(1, len(d)+1))
    metadata = dict(schema='full_feature_vpm_preview_v1', estimate_date=cut, game_through=values['LAST_GAME'],
        season_end_year=SEASON, current_season_games=games, source_players=result['players'], player_count=len(d),
        selection='Positive playing exposure within365days before estimate date',
        source_manifest_sha256=result['board_manifest_sha256'], model_sha256=result['model_sha256'], tracking=True,
        current_sq=True, preview=True,
        qualification='Research preview; conditional fixed fits and historical source vintages; not evidence of predictive superiority.',
        sq_policy='Frozen2026SQ models applied without refitting; predictive EPM skill inputs use cached pregame snapshots, latest request2026-09-20; original missingness retained.')
    metadata.update(prior_regime=result.get('prior_regime', 'uniform_own_only'),
        background_prior_restored=result.get('background_prior_restored', False),
        entrant_metadata=({key: result['entrant_metadata'][key] for key in [
            'model_sha256', 'fit_cut', 'fit_end', 'full_own_possessions', 'frozen_draft_max_year', 'qualification']}
            if result.get('entrant_metadata') else None),
        deployment_comparison=('Frozen entrant/background blend added by owner request; no new predictive-performance claim'
                               if result.get('background_prior_restored') else 'Own-only research control'))
    rows = d.astype(object).where(pd.notnull(d), None).to_dict(orient='records')
    (S/'supabase_preview.json').write_text(json.dumps(dict(snapshot_date=cut, player_count=len(rows), metadata=metadata,
        players=rows), ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    return S/'supabase_preview.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cut', help='Estimate date; games strictly before it. Default: today in Los Angeles.')
    parser.add_argument('--tag', help='Output-folder tag. Default: the date as YYYYMMDD.')
    parser.add_argument('--from-saved', type=Path, help='Reproduction only: reuse this saved R2 snapshot folder.')
    parser.add_argument('--wait-hours', type=float, default=4.0)
    parser.add_argument('--prior-mode', choices=['entrant', 'own-only'], default='entrant', help='Restore the frozen entrant prior by default; own-only is a comparison control.')
    args = parser.parse_args()
    cut = pd.Timestamp(args.cut or pd.Timestamp.now(tz='America/Los_Angeles').date())
    if cut.year+(cut.month >= 7) != SEASON:
        raise SystemExit(f'{cut.date()} is outside the {SEASON} season; rolling over is a deliberate step')
    if shutil.disk_usage(E).free < 15*2**30:
        raise OSError('Under 15 GiB free')
    values = date_values(cut, args.tag)
    paths = dirs(values['TAG'])
    clear_unfinished(paths)
    if args.from_saved:
        refresh = copy_saved_snapshot(args.from_saved.resolve(), paths)
    else:
        wait_until_complete(cut, args.wait_hours)
        refresh = snapshot(paths, cut, values, args.wait_hours)
    if refresh['player_missing_game_ids'] or refresh['team_missing_game_ids'] or refresh['cut'] != values['CUT']:
        raise ValueError('Snapshot incomplete or for another date')
    manifest = json.loads((paths['base']/'game_manifest.json').read_bytes())['games']
    dates = [int(g['date']) for g in manifest if int(g['date']) < int(values['CUT8'])]
    if len(dates) != refresh['manifest_games'] or not dates:
        raise ValueError('Manifest and snapshot game counts differ')
    values.update(GAMES=refresh['manifest_games'], LAST_GAME=str(pd.to_datetime(str(max(dates))).date()))
    result = emit(paths, values, refresh, values)
    if args.prior_mode == 'entrant':
        run(HERE/'restore_entrant_background.py', [paths['emit']], paths['emit']/'restore_entrant.log')
        result = json.loads((paths['emit']/'entrant_result.json').read_bytes())
    preview = build_preview(paths, values, result)
    say('READY_TO_PUBLISH', preview=str(preview), estimate_date=values['CUT'], games=values['GAMES'],
        players=result['recent_players'], manifest=result['board_manifest_sha256'])


if __name__ == '__main__':
    main()

"""Run the authorized matchup backfill, feature export and v47 audits in order.

Each stage is restartable in its owning tool. A failed download/admission/audit
stops the chain. This is an ordinary foreground job, not a scheduled task.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
VPM = ROOT.parents[1] / 'vpm' / 'vpm'
ARCHIVE = ROOT / 'cache/matchup_games_v1'
FEATURES = ROOT / 'results/estimated_skills/matchup_defense_v1'
EXPERIMENT = VPM / 'results/harness/vpm_matchup_defense_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(workers=4, interval=.15):
    destination = ARCHIVE / 'research_run'
    destination.mkdir(parents=True, exist_ok=True)
    config = VPM / 'configs/research_harness_v47_matchup_defense.json'
    stages = [
        ('fetch', [SCRIPTS/'fetch_matchup_game_archive.py', '--stage', 'all', '--workers', str(workers), '--interval', str(interval)]),
        ('archive_audit', [SCRIPTS/'audit_matchup_game_archive.py']),
        ('features', [SCRIPTS/'build_matchup_defensive_features.py']),
        ('feature_audit', [SCRIPTS/'audit_matchup_defensive_features.py', FEATURES]),
        ('vpm', [VPM/'run_harness.py', '--config', config, 'matchup-defense', '--stage', 'all']),
        ('score_audit', [SCRIPTS/'audit_matchup_vpm_scorecard.py']),
        ('field_audit', [VPM/'scripts/validate_field_benchmark.py', EXPERIMENT]),
    ]
    code = [config, Path(__file__), *[command[0] for _,command in stages]]
    code += sorted((VPM/'harness').glob('matchup_defense*.py'))
    code += [VPM/'harness/long_spm_runner.py', VPM/'harness/forward_prior.py',
             VPM/'harness/field_benchmark.py', VPM/'harness/own_sq_parent.py']
    frozen = {str(path):sha(path) for path in code}
    receipt = dict(started_at=datetime.now(timezone.utc).isoformat(), status='running',
                   code_hashes=frozen, stages=[], protected_season=2027)
    progress = destination/'progress.json'

    def save():
        temporary = progress.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        temporary.replace(progress)

    save()
    for name, command in stages:
        for filename, expected in frozen.items():
            if sha(filename) != expected:
                receipt.update(status='failed', error='Code changed during run: '+filename)
                save()
                raise RuntimeError(receipt['error'])
        stage = dict(name=name, started_at=datetime.now(timezone.utc).isoformat(),
                     command=[str(value) for value in command], status='running')
        receipt['stages'].append(stage);save()
        print(json.dumps(dict(stage=name,status='started',log=str(destination/(name+'.log')))),flush=True)
        start=time.monotonic()
        with (destination/(name+'.log')).open('a') as handle:
            completed=subprocess.run([sys.executable,'-u',*[str(value) for value in command]],
                                     cwd=ROOT.parent,stdout=handle,stderr=subprocess.STDOUT)
        stage.update(status='passed' if completed.returncode==0 else 'failed',
                     exit_code=completed.returncode, elapsed_seconds=time.monotonic()-start)
        save();print(json.dumps(stage),flush=True)
        if completed.returncode:
            receipt.update(status='failed',failed_stage=name);save()
            raise RuntimeError('Stage failed; inspect '+str(destination/(name+'.log')))
    receipt.update(status='passed',completed_at=datetime.now(timezone.utc).isoformat());save()
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--interval',type=float,default=.15)
    args=parser.parse_args();run(args.workers,args.interval)

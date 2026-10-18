"""Run admitted literal BAM source, feature and v48 model audits in order.

This is an ordinary foreground job, not an automation. It never downloads
matchups or opens a model before complete 2018–2025 source and feature gates.
The optional wait observes the existing v47 job without restarting it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
VPM = ROOT.parents[1] / 'vpm' / 'vpm'
ARCHIVE = ROOT / 'cache/matchup_games_v1'
DESTINATION = ARCHIVE / 'lagged_bam_run'
MAIN_PROGRESS = ARCHIVE / 'research_run/progress.json'
FEATURES = ROOT / 'results/estimated_skills/lagged_bam_v1'
CONCORDANCE = ROOT / 'results/legacy_bam_lineage_audit_v1/annual_native_concordance.json'
ANNUAL = VPM / 'data/matchup/matchups_two_rs_2018_2026.parquet'
CONFIG = VPM / 'configs/research_harness_v48_lagged_bam_defense.json'
EXPERIMENT = VPM / 'results/harness/vpm_lagged_bam_defense_v1'
REQUIRED_YEARS = list(range(2018,2026))
REQUIRED_CUTS = ['2018-10-01','2019-10-01','2020-12-01',
                 '2021-10-01','2022-10-01','2023-10-01','2024-10-01','2025-10-01']
FEATURE = 'matchup_dated_bam'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _hash_mapping(mapping):
    for name,expected in mapping.items():
        if sha(name)!=expected:raise ValueError('Pinned BAM source changed: '+str(name))


def assert_source_gate(path=CONCORDANCE,annual_path=ANNUAL):
    """Require every declared season, not merely a passed subset request."""
    path=Path(path);gate=json.loads(path.read_text())
    if gate.get('status')!='passed_source_concordance':
        raise ValueError('Annual/native BAM concordance has not passed')
    for key in ['verified_years','requested_years','fully_resolved_years']:
        if sorted(map(int,gate.get(key,[])))!=REQUIRED_YEARS:
            raise ValueError('Incomplete BAM source years in '+key)
    if gate.get('annual_sha256')!=sha(annual_path):
        raise ValueError('BAM concordance does not pin the current annual source')
    if gate.get('protected_outcomes')!=0:raise ValueError('Protected outcomes in source gate')
    details={int(row['year']):row for row in gate.get('per_year_details',[])}
    for year in REQUIRED_YEARS:
        admission=details.get(year,{}).get('gate',{})
        if not admission.get('passed') or not admission.get('all_games_resolved'):
            raise ValueError('Incomplete native source-year gate: '+str(year))
        name=f'native_{year}_rs_pairs_verified.parquet'
        if name not in gate.get('artifacts',{}):raise ValueError('Missing verified native artifact: '+name)
    _hash_mapping(gate.get('source_hashes',{}))
    for name,expected in gate.get('artifacts',{}).items():
        if sha(path.parent/name)!=expected:raise ValueError('Verified native BAM artifact changed: '+name)
    if sha(gate['audit_path'])!=gate['audit_sha256']:
        raise ValueError('BAM source audit receipt changed')
    return dict(status='passed',verified_years=REQUIRED_YEARS,
                concordance_sha256=sha(path),annual_sha256=sha(annual_path),
                source_audit_sha256=gate['audit_sha256'],protected_outcomes=0)


def assert_feature_gate(folder=FEATURES,concordance_path=CONCORDANCE,annual_path=ANNUAL):
    """Require a matching independent formula/chronology admission receipt."""
    source=assert_source_gate(concordance_path,annual_path)
    folder=Path(folder);manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='research_candidate_not_admitted' or not manifest.get('source_gate_complete'):
        raise ValueError('BAM feature export is partial or source-unadmitted')
    if manifest.get('pending_source_years') or manifest.get('missing_games'):
        raise ValueError('BAM feature source is incomplete')
    for key in ['verified_source_years','source_years']:
        if sorted(map(int,manifest.get(key,[])))!=REQUIRED_YEARS:
            raise ValueError('Unexpected BAM feature source years: '+key)
    if sorted(manifest.get('cuts',[]))!=sorted(REQUIRED_CUTS):
        raise ValueError('BAM feature cuts differ from registered warm-up/evaluation dates')
    if manifest.get('feature_groups')!={'bam':[FEATURE]} or FEATURE in manifest.get('unavailable_features',[]):
        raise ValueError('Literal dated BAM is missing or extra feature groups were admitted')
    if manifest.get('protected_outcomes')!=0:raise ValueError('Protected outcomes in BAM features')
    _hash_mapping(manifest.get('source_hashes',{}))
    for name,expected in manifest.get('artifact_hashes',{}).items():
        if sha(folder/name)!=expected:raise ValueError('BAM feature artifact changed: '+name)
    for name in ['features.parquet','feature_provenance.parquet']:
        if name not in manifest.get('artifact_hashes',{}):raise ValueError('Missing feature artifact receipt: '+name)
    pinned=manifest.get('source_hashes',{}).get(str(Path(concordance_path).resolve()))
    if pinned!=source['concordance_sha256']:
        raise ValueError('Feature export does not pin the admitted source gate')
    audit_path=folder/'independent_bam_audit.json';audit=json.loads(audit_path.read_text())
    if (audit.get('status')!='passed_independent_formula_and_chronology'
            or not audit.get('model_admissible_source') or not audit.get('source_gate_complete')):
        raise ValueError('Independent BAM formula/chronology audit has not admitted the model source')
    if sorted(map(int,audit.get('verified_years',[])))!=REQUIRED_YEARS:
        raise ValueError('Independent BAM audit covers incomplete source years')
    if audit.get('manifest_sha256')!=sha(manifest_path) or audit.get('protected_outcomes')!=0:
        raise ValueError('Independent BAM audit is stale or uses protected outcomes')
    if audit.get('auditor_sha256')!=sha(SCRIPTS/'audit_lagged_bam_features.py'):
        raise ValueError('Independent BAM auditor changed since its receipt')
    return dict(status='passed',**{k:v for k,v in source.items() if k!='status'},
        feature_manifest_sha256=sha(manifest_path),independent_bam_audit_sha256=sha(audit_path),
        feature_groups=manifest['feature_groups'],cuts=manifest['cuts'])


def stage_plan():
    return [
        ('source_audit',[SCRIPTS/'audit_legacy_bam_lineage.py','--years',','.join(map(str,REQUIRED_YEARS))]),
        ('features',[SCRIPTS/'build_lagged_bam_features.py','--require-admitted']),
        ('feature_audit',[SCRIPTS/'audit_lagged_bam_features.py','--export',FEATURES]),
        ('vpm',[VPM/'run_harness.py','--config',CONFIG,'matchup-defense','--stage','all']),
        ('score_audit',[SCRIPTS/'audit_matchup_vpm_scorecard.py','--root',EXPERIMENT]),
        ('field_audit',[VPM/'scripts/validate_field_benchmark.py',EXPERIMENT]),
    ]


def frozen_files():
    files=[Path(__file__),CONFIG,*[command[0] for _,command in stage_plan()]]
    files+=sorted((VPM/'harness').glob('matchup_defense*.py'))
    files+=[VPM/'harness'/name for name in ['cli.py','long_spm_runner.py','long_spm_features.py',
        'forward_prior.py','forward_prior_runner.py','field_benchmark.py','own_sq_parent.py',
        'monthly_fit.py','monthly_vpm.py','paths.py','config.py','artifacts.py','provenance.py']]
    return list(dict.fromkeys(files))


def wait_for_main(path=MAIN_PROGRESS,*,check_code=lambda:None,on_update=lambda value:None,sleep=time.sleep):
    """Observe an existing ordinary job; never restart or duplicate its work."""
    previous=None
    while True:
        check_code()
        record=json.loads(Path(path).read_text())
        status=record.get('status')
        if status!=previous:on_update(record);previous=status
        if status=='passed':return record
        if status in ['failed','interrupted','cancelled']:
            raise RuntimeError('Existing main matchup job ended with status '+str(status))
        if status not in ['running','waiting_main']:
            raise RuntimeError('Unrecognized main matchup job status: '+str(status))
        sleep(30)


def run(*,wait_main=False):
    destination=DESTINATION;destination.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid())
    history=destination/'receipt_history'/run_id;history.mkdir(parents=True,exist_ok=False)
    # Keep previous receipts byte-for-byte before their owning restartable
    # source/feature stages update the canonical research directories.
    for name,path in [('previous_progress',destination/'progress.json'),
                      ('concordance',CONCORDANCE),
                      ('source_audit',CONCORDANCE.parent/'all_year_lineage_audit.json'),
                      ('feature_manifest',FEATURES/'manifest.json'),
                      ('feature_audit',FEATURES/'independent_bam_audit.json')]:
        if path.exists():shutil.copy2(path,history/(name+'.json'))
    frozen={str(path):sha(path) for path in frozen_files()}
    receipt=dict(run_id=run_id,started_at=datetime.now(timezone.utc).isoformat(),status='running',
        code_hashes=frozen,stages=[],protected_season=2027,wait_main=wait_main,
        experiment=str(EXPERIMENT),feature_root=str(FEATURES),ordinary_foreground_job=True)
    progress=destination/'progress.json'
    def save():
        payload=json.dumps(receipt,indent=2,allow_nan=False)+'\n'
        temporary=progress.with_suffix('.json.tmp');temporary.write_text(payload);temporary.replace(progress)
        (history/'progress.json').write_text(payload)
    def check_code():_hash_mapping(frozen)
    try:
        save()
        if wait_main:
            receipt['status']='waiting_main';save()
            wait_for_main(check_code=check_code,on_update=lambda record:print(
                json.dumps(dict(wait_main_status=record.get('status'))),flush=True))
            receipt['status']='running';receipt['main_completed_before_start']=True;save()
        for name,command in stage_plan():
            check_code()
            if name=='features':receipt['source_admission']=assert_source_gate();save()
            if name in ['vpm','score_audit','field_audit']:
                receipt['feature_admission']=assert_feature_gate();save()
            stage=dict(name=name,started_at=datetime.now(timezone.utc).isoformat(),
                       command=[str(v) for v in command],status='running')
            receipt['stages'].append(stage);save()
            print(json.dumps(dict(stage=name,status='started',log=str(destination/(name+'.log')))),flush=True)
            start=time.monotonic()
            with (destination/(name+'.log')).open('a') as handle:
                completed=subprocess.run([sys.executable,'-u',*[str(v) for v in command]],
                    cwd=ROOT.parent,stdout=handle,stderr=subprocess.STDOUT)
            stage.update(status='passed' if completed.returncode==0 else 'failed',
                         exit_code=completed.returncode,elapsed_seconds=time.monotonic()-start)
            save();print(json.dumps(stage),flush=True)
            if completed.returncode:raise RuntimeError('Stage failed; inspect '+str(destination/(name+'.log')))
            check_code()
            if name=='source_audit':receipt['source_admission']=assert_source_gate();save()
            if name=='feature_audit':receipt['feature_admission']=assert_feature_gate();save()
        receipt.update(status='passed',completed_at=datetime.now(timezone.utc).isoformat());save()
        return receipt
    except BaseException as error:
        receipt.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                       error=str(error),failed_stage=receipt['stages'][-1]['name'] if receipt['stages'] else 'wait_main')
        save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-main',action='store_true',help='Observe the existing main matchup job every30seconds; abort if it fails')
    run(wait_main=parser.parse_args().wait_main)

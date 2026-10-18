"""Admission/orchestration tests: no network, source build or model execution."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_lagged_bam_research as runner


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def admission(tmp_path):
    annual=tmp_path/'annual.parquet';annual.write_bytes(b'annual-source-fixture')
    folder=tmp_path/'source';folder.mkdir();audit=folder/'audit.json';write(audit,{'fixture':True})
    artifacts={}
    for year in runner.REQUIRED_YEARS:
        path=folder/f'native_{year}_rs_pairs_verified.parquet';path.write_bytes(str(year).encode())
        artifacts[path.name]=runner.sha(path)
    gate=dict(status='passed_source_concordance',verified_years=runner.REQUIRED_YEARS,
        requested_years=runner.REQUIRED_YEARS,fully_resolved_years=runner.REQUIRED_YEARS,
        annual_sha256=runner.sha(annual),protected_outcomes=0,source_hashes={str(annual):runner.sha(annual)},
        artifacts=artifacts,per_year_details=[dict(year=y,gate=dict(passed=True,all_games_resolved=True))
                                           for y in runner.REQUIRED_YEARS],
        audit_path=str(audit),audit_sha256=runner.sha(audit))
    path=folder/'annual_native_concordance.json';write(path,gate)
    return path,annual,gate


def test_source_gate_requires_all_eight_complete_years(admission):
    path,annual,gate=admission
    result=runner.assert_source_gate(path,annual)
    assert result['verified_years']==list(range(2018,2026))
    gate['verified_years']=gate['verified_years'][:-1];write(path,gate)
    with pytest.raises(ValueError,match='Incomplete BAM source years'):
        runner.assert_source_gate(path,annual)


def test_source_gate_rechecks_native_and_audit_bytes(admission):
    path,annual,gate=admission
    native=path.parent/'native_2020_rs_pairs_verified.parquet';native.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        runner.assert_source_gate(path,annual)


def test_feature_gate_requires_matching_independent_admission(tmp_path,admission):
    source_path,annual,_=admission;folder=tmp_path/'features';folder.mkdir()
    artifacts={}
    for name in ['features.parquet','feature_provenance.parquet']:
        path=folder/name;path.write_bytes(name.encode());artifacts[name]=runner.sha(path)
    manifest=dict(status='research_candidate_not_admitted',source_gate_complete=True,
        pending_source_years=[],missing_games=[],verified_source_years=runner.REQUIRED_YEARS,
        source_years=runner.REQUIRED_YEARS,cuts=runner.REQUIRED_CUTS,
        feature_groups={'bam':[runner.FEATURE]},unavailable_features=[],protected_outcomes=0,
        source_hashes={str(source_path.resolve()):runner.sha(source_path)},artifact_hashes=artifacts)
    write(folder/'manifest.json',manifest)
    audit=dict(status='passed_independent_formula_and_chronology',model_admissible_source=True,
        source_gate_complete=True,verified_years=runner.REQUIRED_YEARS,protected_outcomes=0,
        manifest_sha256=runner.sha(folder/'manifest.json'),auditor_sha256=runner.sha(runner.SCRIPTS/'audit_lagged_bam_features.py'))
    write(folder/'independent_bam_audit.json',audit)
    result=runner.assert_feature_gate(folder,source_path,annual)
    assert result['feature_groups']=={'bam':[runner.FEATURE]}
    audit['model_admissible_source']=False;write(folder/'independent_bam_audit.json',audit)
    with pytest.raises(ValueError,match='has not admitted'):
        runner.assert_feature_gate(folder,source_path,annual)
    audit['model_admissible_source']=True;write(folder/'independent_bam_audit.json',audit)
    manifest['additional_receipt_note']='changed after audit';write(folder/'manifest.json',manifest)
    with pytest.raises(ValueError,match='audit is stale'):
        runner.assert_feature_gate(folder,source_path,annual)


def test_wait_observes_only_existing_job_and_aborts_failed(tmp_path):
    path=tmp_path/'progress.json';write(path,{'status':'running'})
    sleeps=[]
    def finish(seconds):
        sleeps.append(seconds);write(path,{'status':'passed'})
    result=runner.wait_for_main(path,sleep=finish)
    assert result['status']=='passed' and sleeps==[30]
    write(path,{'status':'failed'})
    with pytest.raises(RuntimeError,match='status failed'):
        runner.wait_for_main(path,sleep=lambda _:pytest.fail('Failed job must not wait'))


def test_stage_plan_never_downloads_and_passes_v48_explicitly():
    stages=dict(runner.stage_plan())
    assert list(stages)==['source_audit','features','feature_audit','vpm','score_audit','field_audit']
    assert all('fetch_' not in str(command[0]) for command in stages.values())
    assert '--require-admitted' in stages['features']
    assert runner.CONFIG in stages['vpm'] and 'v48' in runner.CONFIG.name
    assert stages['score_audit'][-2:]==['--root',runner.EXPERIMENT]
    assert stages['field_audit'][-1]==runner.EXPERIMENT


def test_wait_stops_before_sleep_when_frozen_code_changes(tmp_path):
    progress=tmp_path/'progress.json';write(progress,{'status':'running'})
    code=tmp_path/'model.py';code.write_text('frozen source')
    lock={str(code):runner.sha(code)}
    code.write_text('changed source')
    with pytest.raises(ValueError,match='source changed'):
        runner.wait_for_main(progress,check_code=lambda:runner._hash_mapping(lock),
                             sleep=lambda _:pytest.fail('Changed code must stop before waiting'))

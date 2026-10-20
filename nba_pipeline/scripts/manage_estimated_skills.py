"""Inventory and pin estimated-skill recipes; preflight dated daily adapter outputs.

This coordinator never runs a research builder, fits parameters, fetches games,
or installs a scheduler. Missing daily adapters remain an explicit blocker.
"""
from __future__ import annotations
import argparse,csv,hashlib,html,json,shutil,tempfile
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'nba_pipeline/config/estimated_skills_operations.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def validate(config):
    if config.get('schema_version')!=1:raise ValueError('Unknown operations schema')
    families=config['families'];ids=[f['id'] for f in families]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate family')
    keys=[]
    for f in families:
        if f['profile'] not in config['profiles']:raise ValueError('Unknown profile')
        if not f['frozen_files'] or not f['fields']:raise ValueError('Family lacks fields or frozen files')
        for s in f['fields']:
            if s['id']!=f['profile']+':'+s['key']:raise ValueError('Unqualified stat identity')
            keys.append(s['id'])
        if f['live_adapter'] is not None:raise ValueError('No approved executable live adapters are registered yet')
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate qualified stat')
    known=set(keys)
    for f in families:
        for s in f['fields']:
            if any(f['profile']+':'+d not in known for d in s['dependencies']):raise ValueError('Unregistered stat dependency')
    return config


def inventory(config,root=ROOT):
    rows=[]
    for f in config['families']:
        pins={p:sha(root/p) if (root/p).is_file() else None for p in f['frozen_files']}
        for s in f['fields']:
            rows.append(dict(profile=f['profile'],family=f['id'],stat=s['key'],label=s['label'],unit=s['unit'],
                             current_vpm_mapper='+'.join(s.get('mapper',[])),status=s['status'],
                             daily_adapter='not_wired',parameters_available=all(pins.values())))
    return dict(id=config['id'],inventory_as_of=config['inventory_as_of'],season=config['season_end_year'],
                status='organized_not_live_wired',fields=len(rows),families=len(config['families']),
                profiles={p:sum(r['profile']==p for r in rows) for p in config['profiles']},rows=rows,
                family_pins={f['id']:{p:sha(root/p) if (root/p).is_file() else None for p in f['frozen_files']} for f in config['families']},
                remaining_work={f['id']:f['remaining_work'] for f in config['families']},
                research_lanes=config['research_lanes'],scheduler_installed=False)


def save_inventory(config,report,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);write(out/'inventory.json',report)
    with (out/'inventory.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(report['rows'][0]));w.writeheader();w.writerows(report['rows'])
    esc=html.escape
    parts=['<!doctype html><meta charset="utf-8"><title>Estimated skills · Daily operations</title>',
           '<style>body{font:15px Arial;background:#f7f8f3;color:#252d25;max-width:1300px;margin:36px auto;padding:0 22px}a{color:#3d5e24}p,li{line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd6c3;text-align:left;padding:10px}details{margin:18px 0}summary{cursor:pointer;font-size:20px}small{color:#5d6b52}</style>',
           '<a href="skills.html">Predictive stats</a> · <a href="ratings.html">Current ratings</a><h1>Estimated skills: daily operations inventory</h1>',
           '<p><strong>Organized; daily data adapters are not yet wired.</strong> No automatic nightly update is running. '+str(report['families'])+' source/estimator groups cover '+str(report['fields'])+' routed outputs across the existing profiles. These are not new model versions or independent VPM coefficients.</p>',
           '<p>Daily work updates states using pinned parameters after completed measured games. Research tuning is separate. Missing stays missing; delayed tracking is visible. New NBA IDs must be admitted. Skill exports and VPM rating refresh are separate operations.</p>']
    for profile,label in config['profiles'].items():
        parts.append('<h2>'+esc(profile.replace('_',' '))+'</h2><p>'+esc(label)+'</p>')
        for f in config['families']:
            if f['profile']!=profile:continue
            parts.append('<details><summary>'+esc(f['label'])+' · '+str(len(f['fields']))+' outputs</summary><p><strong>Daily adapter: not wired.</strong></p><p>Sources: '+esc('; '.join(f['source_inputs']))+'</p><ul>'+''.join('<li>'+esc(t)+'</li>' for t in f['remaining_work'])+'</ul><table><tr><th>Stat</th><th>Units</th><th>Current VPM mapper</th><th>Recipe</th></tr>')
            for s in f['fields']:
                parts.append('<tr><td>'+esc(s['label'])+'<br><small>'+esc(s['key'])+'</small></td><td>'+esc(s['unit'])+'</td><td>'+esc('+'.join(s.get('mapper',[])) or 'outside current mapper')+'</td><td>'+esc(s.get('table_model',s['status']))+'</td></tr>')
            parts.append('</table></details>')
    parts.append('<h2>Research inventory</h2><p>All20 existing research lanes remain visible. Failed and diagnostic lanes are not scheduled as daily models.</p><ul>'+''.join('<li>'+esc(str(l['lane'])+' · '+l['name']+' · '+l['status'])+'</li>' for l in report['research_lanes'])+'</ul>')
    (out/'index.html').write_text(''.join(parts))


def freeze(config,config_path,out,season,root=ROOT):
    """Copy exact currently exported parameter recipes, without fitting/adopting."""
    out=Path(out)
    if out.exists():raise FileExistsError('Bundle is immutable; use a new output directory')
    report=inventory(config,root)
    missing=[p for group in report['family_pins'].values() for p,h in group.items() if h is None]
    if missing:raise FileNotFoundError('Missing recipe pins: '+str(sorted(set(missing))))
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='.skill-freeze-') as temp:
        temp=Path(temp);(temp/'objects').mkdir();files={}
        for group in report['family_pins'].values():
            for p,h in group.items():
                dest=temp/'objects'/h
                if not dest.exists():shutil.copyfile(root/p,dest)
                if sha(dest)!=h:raise ValueError('Copy changed during freeze')
                files[p]=h
        shutil.copyfile(config_path,temp/'operations.json')
        manifest=dict(schema_version=1,season=int(season),recipe_status='carry-forward recipes currently exported; not a newly trained preseason vintage',
                      config_sha256=sha(temp/'operations.json'),files=files,family_pins=report['family_pins'],
                      no_new_fit=True,daily_adapters_wired=False)
        write(temp/'manifest.json',manifest);temp.rename(out)
    return manifest


def verify_bundle(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if sha(bundle/'operations.json')!=m['config_sha256']:raise ValueError('Frozen config changed')
    for h in m['files'].values():
        if sha(bundle/'objects'/h)!=h:raise ValueError('Frozen parameter/code changed')
    return validate(read(bundle/'operations.json')),m


def check_receipt(family,receipt_dir,bundle,asof):
    """Validate an upstream adapter's dated normalized outputs, not research scores."""
    path=Path(receipt_dir)/(family['id']+'.json')
    if not path.exists():return dict(family=family['id'],status='blocked',reason='daily adapter receipt missing')
    try:
        r=read(path);cut=pd.Timestamp(asof)
        if r['family']!=family['id'] or int(r['season'])!=bundle['season']:raise ValueError('Wrong family/season')
        if pd.Timestamp(r['as_of'])!=cut:raise ValueError('Stale or mismatched as-of')
        if r['parameter_bundle_sha256']!=bundle['bundle_sha256']:raise ValueError('Wrong frozen recipe bundle')
        if r['source_status']!='complete' or not r['completed_game_census_checked']:raise ValueError('Source delayed or game census unchecked')
        if not r['identities_checked'] or not r['missingness_checked']:raise ValueError('Identity/missingness checks missing')
        target=(path.parent/r['artifact']).resolve()
        if path.parent.resolve() not in target.parents:raise ValueError('Artifact must be inside receipt directory')
        if sha(target)!=r['artifact_sha256']:raise ValueError('Artifact checksum changed')
        d=pd.read_parquet(target);required={'player_id','stat_key','value','last_observation_date','missing_reason'}
        if not required.issubset(d):raise ValueError('Incomplete state schema')
        expected={s['key'] for s in family['fields']}
        if set(d.stat_key)!=expected or d.duplicated(['player_id','stat_key']).any():raise ValueError('Missing/extra stats or duplicate player/stat')
        pid=pd.to_numeric(d.player_id,errors='coerce')
        if not pid.notna().all() or not (pid>0).all() or not np.equal(pid,np.floor(pid)).all():raise ValueError('Invalid NBA identity')
        if set(pid.astype(int))!=set(r['expected_player_ids']):raise ValueError('Player census does not match declared daily universe')
        if not d.groupby('player_id').stat_key.nunique().eq(len(expected)).all():raise ValueError('Missing per-player stat rows')
        dates=pd.to_datetime(d.last_observation_date,errors='raise');values=pd.to_numeric(d.value,errors='raise');present=values.notna()
        if not np.isfinite(values[present]).all():raise ValueError('Nonfinite observed state')
        if not dates.dropna().lt(cut).all() or dates[present].isna().any():raise ValueError('Observation date fence or missing provenance')
        if d.loc[~present,'missing_reason'].fillna('').str.strip().eq('').any():raise ValueError('Missing value has no reason')
        if d.loc[present,'missing_reason'].fillna('').str.strip().ne('').any():raise ValueError('Present value incorrectly marked missing')
        return dict(family=family['id'],status='passed',rows=len(d),players=d.player_id.nunique(),missing=int((~present).sum()),artifact_sha256=r['artifact_sha256'])
    except (ValueError,KeyError,TypeError,OSError) as e:return dict(family=family['id'],status='blocked',reason=str(e))


def check(bundle,receipts,asof,out):
    config,m=verify_bundle(bundle);m['bundle_sha256']=sha(Path(bundle)/'manifest.json')
    rows=[check_receipt(f,receipts,m,asof) for f in config['families']]
    report=dict(as_of=str(pd.Timestamp(asof).date()),season=m['season'],status='passed' if all(r['status']=='passed' for r in rows) else 'blocked',
                families=rows,parameter_bundle_sha256=m['bundle_sha256'],states_updated_by_this_command=False,scheduler_installed=False)
    Path(out).mkdir(parents=True,exist_ok=True);write(Path(out)/'daily_check.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('inventory');a.add_argument('--out',type=Path,required=True);a.add_argument('--reader-dir',type=Path)
    a=sub.add_parser('freeze');a.add_argument('--out',type=Path,required=True);a.add_argument('--season',type=int,required=True)
    a=sub.add_parser('check');a.add_argument('--bundle',type=Path,required=True);a.add_argument('--receipts',type=Path,required=True);a.add_argument('--as-of',required=True);a.add_argument('--out',type=Path,required=True)
    args=p.parse_args();config=validate(read(args.config))
    if args.command=='inventory':
        r=inventory(config);save_inventory(config,r,args.out)
        if args.reader_dir:
            args.reader_dir.mkdir(parents=True,exist_ok=True);shutil.copyfile(args.out/'index.html',args.reader_dir/'skills-operations.html')
        print(json.dumps({k:r[k] for k in ['status','fields','families','profiles']}));return 0
    if args.command=='freeze':freeze(config,args.config,args.out,args.season);print(args.out);return 0
    r=check(args.bundle,args.receipts,args.as_of,args.out);print(json.dumps(r));return 0 if r['status']=='passed' else 2

if __name__=='__main__':raise SystemExit(main())

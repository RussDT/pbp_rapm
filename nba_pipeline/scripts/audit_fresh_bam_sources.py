"""Fresh native-only and official Synergy source gates for the new BAM test.

The failed annual archive, its action counts and all its metadata are unused.
The native source gate preserves all raw FGA while exposing conservative pair
count/identity masks. Historical publication vintage is not claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse only the tested raw-count/PBP-identity aggregation utilities. No legacy
# annual archive is opened and the old concordance gate is never consulted.
from audit_legacy_bam_lineage import identity_evidence,pair_quality,aggregate_native,LINES,CONTEXT

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'cache/matchup_games_v1'
ARCHIVE_AUDIT=ROOT/'results/matchup_game_archive_audit_v1'
INPUTS=ROOT/'cache/matchup_bam_fresh_inputs_v1'
OUT=ROOT/'results/fresh_bam_source_audit_v1'
TYPES=['Isolation','Postup','PRBallHandler']
YEARS=list(range(2018,2026))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def require_hashes(sources):
    for path,checksum in sources.items():
        if sha(path)!=checksum:raise ValueError('Fresh BAM source changed: '+path)


def require_complete_statuses(statuses):
    if not statuses or any(status not in ['ok','empty'] for status in statuses) or 'ok' not in statuses:
        raise ValueError('Unresolved/empty-only native RS source year')


def native_gate(archive=ARCHIVE,archive_audit=ARCHIVE_AUDIT,out=OUT):
    archive,archive_audit,out=map(Path,[archive,archive_audit,out]);out.mkdir(parents=True,exist_ok=True)
    ap=archive_audit/'manifest.json';audit=json.loads(ap.read_text())
    if (audit['status']!='passed_native_identity_fidelity; field_coverage_retained_as_measurement'
            or audit['pending_games']!=0 or audit['protected_outcomes']!=0):
        raise ValueError('Full native archive source audit has not passed')
    require_hashes(audit['sources'])
    for name,checksum in audit['artifacts'].items():
        if sha(archive_audit/name)!=checksum:raise ValueError('Native source audit artifact changed')
    census_path=archive/'game_census.parquet';census=pd.read_parquet(census_path)
    census.game_id=census.game_id.astype(str).str.zfill(10)
    if census.game_id.duplicated().any() or census.season.gt(2026).any():raise ValueError('Native census identity/protected year')
    checked=pd.read_parquet(archive_audit/'game_audits.parquet')
    if set(checked.game_id.astype(str))!=set(census.game_id):raise ValueError('Archive audit not whole-census complete')
    context=pd.read_parquet(CONTEXT);lines=pd.read_parquet(LINES)
    evidence=identity_evidence(lines,context)
    sources={str(p.resolve()):sha(p) for p in [ap,census_path,LINES,CONTEXT,Path(__file__),Path(__file__).with_name('audit_legacy_bam_lineage.py')]}
    artifacts={};details=[]
    for year in YEARS:
        selected=census.loc[census.season.eq(year)&census.season_type.eq('rs')]
        if selected.empty:raise ValueError('Missing native RS census year')
        parts=[];statuses=[]
        for row in selected.itertuples():
            mp=archive/'metadata'/f'{row.game_id}.json';meta=json.loads(mp.read_text())
            status=meta.get('status');statuses.append(status)
            if status not in ['ok','empty']:raise ValueError('Unresolved native RS game '+row.game_id)
            raw=archive/'raw'/f'{row.game_id}.json'
            if sha(raw)!=meta['sha256'] or audit['sources'].get(str(raw.resolve()))!=sha(raw):
                raise ValueError('Raw native game not pinned by full source audit')
            if status=='empty':continue
            path=archive/'pairs'/f'{row.game_id}.parquet'
            if audit['sources'].get(str(path.resolve()))!=sha(path):raise ValueError('Native pair cache not source-audited')
            frame=pd.read_parquet(path)
            if (not frame.game_id.astype(str).eq(row.game_id).all() or not frame.season.eq(year).all()
                or not frame.season_type.eq('rs').all() or not pd.to_datetime(frame.game_date).eq(pd.Timestamp(row.game_date)).all()):
                raise ValueError('Mixed native game/date/season/phase')
            parts.append(frame)
        require_complete_statuses(statuses)
        raw_pairs=pd.concat(parts,ignore_index=True)
        if raw_pairs.duplicated(['game_id','offense_player_id','defense_player_id']).any():raise ValueError('Duplicate native game/pair')
        qualified=pair_quality(raw_pairs,evidence)
        aggregate=aggregate_native(qualified)
        path=out/f'native_{year}_rs_pairs_verified.parquet';aggregate.reset_index().to_parquet(path,index=False)
        artifacts[path.name]=sha(path)
        details.append(dict(year=year,RS_games=len(selected),ok_games=statuses.count('ok'),empty_games=statuses.count('empty'),
            pair_game_rows=len(qualified),positive_fga_pairs=len(aggregate),
            raw_count_invalid_rows=int((~qualified.pair_source_counts_valid).sum()),
            identity_unverified_rows=int((~qualified.pair_identity_verified).sum()),
            MATCHUP_FGA=float(aggregate.MATCHUP_FGA.sum()),MATCHUP_TOV=float(aggregate.MATCHUP_TOV.sum()),
            masked_pair_FGA=float(aggregate.loc[~(aggregate.pair_source_counts_valid&aggregate.pair_identity_verified),'MATCHUP_FGA'].sum())))
        print(json.dumps(details[-1]),flush=True)
    require_hashes(audit['sources']);require_hashes(sources)
    receipt=dict(status='passed_native_source_gate',verified_years=YEARS,artifacts=artifacts,
        source_hashes=sources,full_archive_audit_sha256=sha(ap),full_archive_source_hashes=audit['sources'],
        per_year=details,legacy_annual_archive_used=False,protected_outcomes=0,
        positive_fga_pair_rule='Raw zero-FGA pairs removed only after aggregation; all raw positive-FGA denominator counts retained before pair quality masks',
        pair_identity_rule='Every constituent game has exact PBP offensive/defensive exposure and matching own/opponent team IDs; unknown pairs masked, not remapped',
        historical_publication_vintage_certified=False)
    save(out/'native_source_gate.json',receipt);return receipt


def official_table(payload,year,play_type):
    expected={'LeagueID':'00','SeasonYear':f'{year-1}-{year%100:02d}','SeasonType':'Regular Season',
              'PerMode':'PerGame','PlayerOrTeam':'P','PlayType':play_type,'TypeGrouping':'offensive'}
    params=payload.get('parameters',{})
    if any(params.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong official response parameter lineage')
    sets=payload.get('resultSets',[])
    if len(sets)!=1:raise ValueError('Ambiguous official result set')
    frame=pd.DataFrame(sets[0]['rowSet'],columns=sets[0]['headers'])
    columns=['SEASON_ID','PLAYER_ID','TEAM_ID','PLAY_TYPE','TYPE_GROUPING','PPP','POSS','GP']
    if frame.empty or not set(columns).issubset(frame):raise ValueError('Missing fresh Synergy schema/cohort')
    for column in ['SEASON_ID','PLAYER_ID','TEAM_ID','PPP','POSS','GP']:
        frame[column]=pd.to_numeric(frame[column],errors='raise')
    if not frame.SEASON_ID.eq(20000+year-1).all():raise ValueError('Wrong Synergy season ID')
    if not frame.PLAY_TYPE.eq(play_type).all() or not frame.TYPE_GROUPING.str.lower().eq('offensive').all():
        raise ValueError('Wrong Synergy row playtype/side')
    if frame.duplicated(['PLAYER_ID','TEAM_ID','PLAY_TYPE']).any():raise ValueError('Duplicate fresh Synergy identity')
    if not np.isfinite(frame[['PPP','POSS','GP','PLAYER_ID','TEAM_ID']].to_numpy(float)).all():
        raise ValueError('Nonfinite fresh Synergy measurement')
    if frame[['PPP','POSS']].lt(0).any().any() or not frame[['GP','PLAYER_ID','TEAM_ID']].gt(0).all().all():
        raise ValueError('Negative production or invalid Synergy identity/exposure')
    frame['year']=year
    return frame


def synergy_audit(inputs=INPUTS,out=OUT):
    inputs,out=Path(inputs),Path(out);out.mkdir(parents=True,exist_ok=True)
    path=inputs/'manifest.json';manifest=json.loads(path.read_text());require_hashes(manifest['source_hashes'])
    if (manifest['status']!='completed_official_RS_inputs' or manifest['years']!=YEARS
        or manifest['play_types']!=TYPES or manifest['protected_outcomes']!=0
        or manifest['legacy_annual_counts_or_metadata_used']):raise ValueError('Fresh input contract changed')
    for name,checksum in manifest['artifact_hashes'].items():
        if sha(inputs/name)!=checksum:raise ValueError('Fresh Synergy parquet changed')
    chunks=[];details=[];sources={str(path.resolve()):sha(path),str(Path(__file__).resolve()):sha(__file__)}
    if len(manifest['receipts'])!=24:raise ValueError('Missing completed-year/playtype response')
    for year in YEARS:
        for play_type in TYPES:
            matches=[r for r in manifest['receipts'] if r['year']==year and r['play_type']==play_type]
            if len(matches)!=1:raise ValueError('Missing/ambiguous source year/playtype receipt')
            row=matches[0];raw,meta=Path(row['raw']),Path(row['metadata'])
            if sha(raw)!=row['sha256'] or sha(meta)!=row['metadata_sha256']:raise ValueError('Fresh response changed')
            md=json.loads(meta.read_text())
            request={'LeagueID':'00','PerMode':'PerGame','PlayerOrTeam':'P','SeasonYear':f'{year-1}-{year%100:02d}',
                     'SeasonType':'Regular Season','PlayType':play_type,'TypeGrouping':'offensive'}
            if md['endpoint']!='SynergyPlayTypes' or md['parameters']!=request or md['sha256']!=sha(raw):
                raise ValueError('Fresh request metadata mismatch')
            fetched=pd.Timestamp(md['fetched_at'])
            if fetched.year<year:raise ValueError('Impossible completed-season source fetch date')
            frame=official_table(json.loads(raw.read_text()),year,play_type)
            if len(frame)!=row['rows']:raise ValueError('Fresh response row census changed')
            chunks.append(frame);sources[str(raw.resolve())]=sha(raw);sources[str(meta.resolve())]=sha(meta)
            details.append(dict(year=year,play_type=play_type,rows=len(frame),source_fetched_at=md['fetched_at']))
    expected=pd.concat(chunks,ignore_index=True)
    stored=pd.read_parquet(inputs/'synergy_player_playtypes.parquet')
    keys=['year','PLAYER_ID','TEAM_ID','PLAY_TYPE']
    a,b=(frame.set_index(keys).sort_index() for frame in [expected,stored])
    if not a.index.equals(b.index):raise ValueError('Raw/parquet fresh Synergy identity mismatch')
    if not np.allclose(a[['PPP','POSS','GP','SEASON_ID']].to_numpy(float),
                       b[['PPP','POSS','GP','SEASON_ID']].to_numpy(float),rtol=0,atol=0):
        raise ValueError('Raw/parquet Synergy measurement mismatch')
    require_hashes(manifest['source_hashes']);require_hashes(sources)
    receipt=dict(status='passed_fresh_official_synergy_source_audit',years=YEARS,play_types=TYPES,
        rows=len(expected),responses=details,source_hashes=sources,protected_outcomes=0,
        historical_publication_vintage_certified=False,legacy_annual_archive_used=False,
        interpretation='Current official completed-season RS per-game measurements; no pristine historical extraction/publication vintage claimed')
    save(out/'synergy_source_audit.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['native','synergy','all'],default='all')
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--inputs',type=Path,default=INPUTS)
    args=parser.parse_args()
    if args.stage in ['native','all']:native_gate(out=args.out)
    if args.stage in ['synergy','all']:synergy_audit(args.inputs,args.out)

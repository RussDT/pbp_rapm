"""Independent formula and chronological audit of the fresh literal-BAM export.

No old annual matchup archive/metadata is opened. No VPM outcome is read.
Season normalization and snapshot references are independent-auditor utilities,
not feature-builder functions. Traded-team per-game loads deliberately retain
legacy unweighted summation; GP-weighted values are diagnostic only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_lagged_bam_features import (sha,assert_values,compare_frames,season_reference,
    snapshot_reference,align,FEATURE,KEY)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/estimated_skills/fresh_bam_v1'
YEARS=list(range(2018,2026))
TYPES=['Isolation','Postup','PRBallHandler']


def creation_reference(rows):
    positive=rows.loc[rows.PPP.gt(0)&rows.POSS.gt(0)].copy()
    positive['term']=positive.PPP*positive.POSS
    # This deliberately sums each team's per-game term, matching the literal
    # original implementation; no stint GP weighting or totals conversion.
    loads=positive.groupby(['year','PLAYER_ID']).term.sum().rename('Offensive_Score')
    loads.index.names=['year','nba_id_off']
    metadata={int(year):dict(median=float(g.median()),fifth_percentile=float(g.quantile(.05)),
                           cohort_players=len(g)) for year,g in loads.groupby(level=0)}
    return loads,metadata


def actual_ts_reference(lines):
    rows=lines.loc[lines.season.isin(YEARS)&lines.season_type.eq('rs')]
    if rows.duplicated(['game_id','player_id']).any():raise ValueError('Ambiguous actual TS player/game identity')
    if not np.isfinite(rows[['pts','fga','fta']].to_numpy(float)).all() or rows[['pts','fga','fta']].lt(0).any().any():
        raise ValueError('Invalid actual TS counts')
    totals=rows.groupby(['season','player_id'])[['pts','fga','fta']].sum()
    totals.index.names=['year','nba_id_off']
    totals=totals.rename(columns={'pts':'actual_pts','fga':'actual_fga','fta':'actual_fta'})
    denom=2*(totals.actual_fga+.44*totals.actual_fta)
    totals['actual_ts_unrounded']=totals.actual_pts/denom.where(denom.gt(0))
    totals['source_ts']=totals.actual_ts_unrounded.round(3)
    totals['ts_verified']=np.isfinite(totals.source_ts)
    return totals


def fresh_pair_formula(rows):
    fga,fta,tov,pts=(rows[f].to_numpy(float) for f in ['MATCHUP_FGA','MATCHUP_FTA','MATCHUP_TOV','PLAYER_PTS'])
    denominator=fga+.44*fta;effective=fga+1.05*tov
    raw=np.divide(pts,2*denominator,out=np.zeros(len(rows)),where=denominator!=0)
    adjusted=np.where(raw>rows.source_ts.to_numpy(float),rows.median_fallback_load,rows.Offensive_Score)
    ets=np.divide(pts,2*(effective+.44*fta),out=np.zeros(len(rows)),where=effective+.44*fta!=0)
    roles=rows.position.map(lambda p:p.upper().split('-')[0] if isinstance(p,str) else '')
    boost=roles.map({'PG':.3025,'SG':.3025,'G':.3025,'SF':.25,'PF':.25,'F':.25,'C':.125})
    allfga=rows.groupby(['year','nba_id_def']).MATCHUP_FGA.transform('sum')
    fields=np.column_stack([fga,fta,tov,pts,rows.PARTIAL_POSS,rows.source_ts,adjusted])
    valid=((fga>0)&(fta>=0)&(tov>=0)&(pts>=0)&rows.PARTIAL_POSS.gt(0)&allfga.gt(0)
        &rows.ts_verified.fillna(False)&np.isfinite(boost)&(adjusted>=0)&np.isfinite(fields).all(axis=1)
        &rows.pair_source_counts_valid&rows.pair_identity_verified)
    contribution=-10*(ets-rows.source_ts)*adjusted*effective/allfga*(1+boost)
    return dict(TS_PCT=rows.source_ts,literal_raw_denom=denominator,literal_raw_ts=raw,
        literal_effective_fga=effective,literal_effective_denom=effective+.44*fta,
        literal_effective_ts=ets,literal_adjusted_load=adjusted,dated_boost=boost,
        all_defender_fga=allfga,pair_verified=valid,dated_bam_pair=contribution.where(valid))


def audit(folder=DEFAULT):
    folder=Path(folder);mp=folder/'manifest.json';manifest=json.loads(mp.read_text())
    if (not manifest['source_gate_complete'] or manifest['verified_source_years']!=YEARS
        or manifest['pending_source_years'] or manifest['protected_outcomes']!=0):
        raise ValueError('Fresh BAM export source gate incomplete/protected')
    sources=manifest['source_hashes']
    if any('matchups_two' in path for path in sources):raise ValueError('Legacy annual matchup source admitted')
    for path,checksum in sources.items():
        if sha(path)!=checksum:raise ValueError('Fresh BAM source changed: '+path)
    for name,checksum in manifest['artifact_hashes'].items():
        if sha(folder/name)!=checksum:raise ValueError('Fresh BAM artifact changed: '+name)
    def find(suffix):
        matches=[Path(path) for path in sources if path.endswith(suffix)]
        if len(matches)!=1:raise ValueError('Missing/ambiguous pinned fresh BAM source: '+suffix)
        return matches[0]
    gate_path=find('native_source_gate.json');gate=json.loads(gate_path.read_text())
    if gate['status']!='passed_native_source_gate' or gate['verified_years']!=YEARS or gate['legacy_annual_archive_used']:
        raise ValueError('Native source gate identity changed')
    source_receipt=ROOT/'results/fresh_bam_source_audit_v1/synergy_source_audit.json'
    sr=json.loads(source_receipt.read_text())
    if sr['status']!='passed_fresh_official_synergy_source_audit' or sr['years']!=YEARS or sr['legacy_annual_archive_used']:
        raise ValueError('Fresh Synergy independent source audit incomplete')
    for path,checksum in sr['source_hashes'].items():
        if sha(path)!=checksum:raise ValueError('Fresh Synergy audit source changed')
    native=[]
    for year in YEARS:
        name=f'native_{year}_rs_pairs_verified.parquet';path=find(name)
        if sha(path)!=gate['artifacts'][name]:raise ValueError('Native artifact differs from source gate')
        d=pd.read_parquet(path);d['year']=year;native.append(d)
    ref=pd.concat(native,ignore_index=True)
    # Independently parse all24 raw JSONs, never builder offensive_inputs.
    chunks=[]
    for year in YEARS:
        for play_type in TYPES:
            path=find(f'{year}_{play_type}.json')
            raw=json.loads(path.read_text());s=raw['resultSets'][0]
            d=pd.DataFrame(s['rowSet'],columns=s['headers']);d['year']=year;chunks.append(d)
    synergy=pd.concat(chunks,ignore_index=True)
    loads,loadmeta=creation_reference(synergy)
    actual=actual_ts_reference(pd.read_parquet(find('player_game_lines.parquet')))
    offensive=actual.join(loads,how='outer').reset_index()
    offensive['creation_load_observed']=offensive.Offensive_Score.notna()
    offensive['median_fallback_load']=offensive.year.map({year:value['median'] for year,value in loadmeta.items()})
    offensive['fifth_percentile_load']=offensive.year.map({year:value['fifth_percentile'] for year,value in loadmeta.items()})
    saved_inputs=pd.read_parquet(folder/'offensive_inputs.parquet')
    gaps=compare_frames(saved_inputs,offensive,['year','nba_id_off'],
        ['actual_pts','actual_fga','actual_fta','actual_ts_unrounded','source_ts','Offensive_Score',
         'creation_load_observed','median_fallback_load','fifth_percentile_load'])
    for year,values in loadmeta.items():
        saved=manifest['fresh_creation_population'][str(year)]
        for field in ['median','fifth_percentile','cohort_players']:
            assert_values(saved[field],values[field],f'fresh full creation cohort {year}/{field}')
    cohort=pd.read_parquet(folder/'synergy_cohort.parquet')
    compare_frames(cohort,loads.reset_index(),['year','nba_id_off'],['Offensive_Score'])
    # Source identity is always a left join so missing-price pair FGA remains
    # in the original defender denominator before admission masks.
    ref=ref.merge(offensive,on=['year','nba_id_off'],how='left',validate='many_to_one')
    ref['Offensive_Score']=ref.Offensive_Score.fillna(ref.fifth_percentile_load)
    positions=pd.read_parquet(find('player_seasons.parquet'))
    ref=ref.merge(positions[['year','nba_id','position','earliest_use_date']],left_on=['year','nba_id_off'],
                  right_on=['year','nba_id'],how='left',validate='many_to_one')
    census=pd.read_parquet(find('game_census.parquet'))
    ends=census.loc[census.season.isin(YEARS)&census.season_type.eq('rs')].groupby('season').game_date.max().to_dict()
    ref['source_rs_end']=pd.to_datetime(ref.year.map(ends));ref['position_available_at']=pd.to_datetime(ref.earliest_use_date)
    formula=fresh_pair_formula(ref)
    for field,values in formula.items():ref[field]=values
    pairs=pd.read_parquet(folder/'pairs_dated_bam.parquet')
    compare_fields=['PARTIAL_POSS','MATCHUP_FGA','MATCHUP_FTA','MATCHUP_TOV','PLAYER_PTS',
        'pair_source_counts_valid','pair_identity_verified','source_ts','Offensive_Score',
        'creation_load_observed','median_fallback_load','fifth_percentile_load',*formula]
    gaps.update(compare_frames(pairs,ref,KEY,compare_fields))
    expected_seasons=season_reference(ref);seasons=pd.read_parquet(folder/'defender_seasons.parquet')
    gaps.update(compare_frames(seasons,expected_seasons,['year','player_id'],
        ['season_raw','season_exposure','all_source_exposure','source_coverage','season_z']))
    expected=snapshot_reference(expected_seasons,manifest['cuts'],ends)
    features=pd.read_parquet(folder/'features.parquet')
    numeric=[name for name in expected if name not in ['player_id','estimate_date','last_observation_date','parameter_fit_end']]
    gaps.update(compare_frames(features,expected,['player_id','estimate_date'],numeric))
    provenance=pd.read_parquet(folder/'feature_provenance.parquet')
    a,b=align(provenance,['player_id','estimate_date']),align(expected,['player_id','estimate_date'])
    if not a.index.equals(b.index) or not a.feature.eq(FEATURE).all():raise ValueError('Fresh BAM provenance row census')
    for field in ['last_observation_date','parameter_fit_end']:
        x,y=pd.to_datetime(a[field]),pd.to_datetime(b[field])
        if not (x.eq(y)|(x.isna()&y.isna())).all() or x.ge(a.index.get_level_values('estimate_date')).any():
            raise ValueError('Fresh BAM strict date fence: '+field)
    for cut in manifest['cuts']:
        day=pd.Timestamp(cut);selected=expected.loc[expected.estimate_date.eq(day)]
        prefix=expected_seasons.loc[pd.to_datetime(expected_seasons.source_rs_end).lt(day)]
        if not selected.empty:
            compare_frames(selected,snapshot_reference(prefix,[cut],ends),['player_id','estimate_date'],numeric)
    if manifest['feature_groups']!={'bam':[FEATURE]}:raise ValueError('Unexpected fresh BAM feature admission')
    # Report source convention without replacing it: multiple team/type rows
    # deliberately inflate a literal per-game sum relative to GP weighting.
    multiteam=synergy.groupby(['year','PLAYER_ID','PLAY_TYPE']).TEAM_ID.nunique().gt(1)
    for path,checksum in sources.items():
        if sha(path)!=checksum:raise ValueError('Fresh source changed during audit')
    result=dict(status='passed_independent_fresh_BAM_formula_and_chronology',source_gate_complete=True,
        source_years=YEARS,pairs=len(pairs),defender_seasons=len(seasons),features=len(features),provenance=len(provenance),
        fresh_synergy_rows=len(synergy),multi_team_player_type_years=int(multiteam.sum()),
        creation_convention='literal unweighted sum(PPP*PerGamePOSS) over all team/type rows; GP-weighted diagnostic is not a feature',
        maximum_numerical_gap=max(gaps.values()),protected_outcomes=0,legacy_annual_archive_used=False,
        manifest_sha256=sha(mp),auditor_sha256=sha(__file__),independent_synergy_source_audit_sha256=sha(source_receipt),
        limitations=['Historical publication vintage uncertified; retrospective completed-season inputs.',
                    'Formula/source audit does not establish predictive value or waive frozen forward evaluation.'])
    (folder/'independent_fresh_bam_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--export',type=Path,default=DEFAULT)
    audit(parser.parse_args().export)

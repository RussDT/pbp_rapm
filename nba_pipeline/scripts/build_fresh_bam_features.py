"""Literal dated BAM from fresh native matchup, actual TS and Synergy inputs.

No legacy season matchup table or its offensive metadata is read. Counts come
from independently verified native RS game aggregates. Source-year positions
and completed RS offensive measurements are used only after that RS ends.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name('build_lagged_bam_features.py')
_spec = importlib.util.spec_from_file_location('fixed_lagged_bam_helpers', HELPER)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
sha, write_json = H.sha, H.write_json
POSITIONS, LINES, CENSUS, CUTS, BOOST = H.POSITIONS, H.LINES, H.CENSUS, H.CUTS, H.BOOST
SOURCE_GATE = ROOT / 'results/fresh_bam_source_audit_v1/native_source_gate.json'
SYNERGY_GATE = ROOT / 'results/fresh_bam_source_audit_v1/synergy_source_audit.json'
NATIVE = ROOT / 'results/fresh_bam_source_audit_v1'
SYNERGY = ROOT / 'cache/matchup_bam_fresh_inputs_v1'
OUT = ROOT / 'results/estimated_skills/fresh_bam_v1'
YEARS = list(range(2018, 2026))
PLAYTYPES = ('Isolation', 'Postup', 'PRBallHandler')
FEATURE = H.FEATURE


def actual_ts(lines):
    """Same completed-RS actual-count TS at original three-decimal precision."""
    d = lines.loc[lines.season_type.eq('rs') & lines.season.isin(YEARS)].copy()
    if d.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Ambiguous actual player-game TS source')
    numeric = d[['pts', 'fga', 'fta']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric).all(axis=None) or numeric.lt(0).any(axis=None):
        raise ValueError('Invalid actual RS TS primitive')
    d[['pts', 'fga', 'fta']] = numeric
    totals = d.groupby(['season', 'player_id'], as_index=False)[['pts', 'fga', 'fta']].sum()
    totals = totals.rename(columns={'season':'year', 'player_id':'nba_id_off',
        'pts':'actual_pts', 'fga':'actual_fga', 'fta':'actual_fta'})
    denom = 2 * (totals.actual_fga + .44 * totals.actual_fta)
    totals['actual_ts_unrounded'] = totals.actual_pts / denom.where(denom.gt(0))
    totals['source_ts'] = totals.actual_ts_unrounded.round(3)
    totals['ts_verified'] = totals.source_ts.notna() & np.isfinite(totals.source_ts)
    return totals


def creation_loads(rows):
    """Literal unweighted team-stint sum over full positive-PPP/POSS cohort."""
    need = {'year','play_type','PLAYER_ID','TEAM_ID','PPP','POSS','GP'}
    if not need <= set(rows):raise ValueError('Missing Synergy columns: '+str(need-set(rows)))
    d = rows.loc[rows.year.isin(YEARS)].copy()
    if not d.play_type.isin(PLAYTYPES).all():raise ValueError('Unregistered Synergy play type')
    if d.duplicated(['year','play_type','PLAYER_ID','TEAM_ID']).any():raise ValueError('Duplicate Synergy player/team/type/year')
    for column in ['PLAYER_ID','TEAM_ID','PPP','POSS','GP']:
        d[column] = pd.to_numeric(d[column], errors='raise')
    if not np.isfinite(d[['PLAYER_ID','TEAM_ID','PPP','POSS','GP']]).all(axis=None):raise ValueError('Invalid Synergy primitive')
    if not d[['PLAYER_ID','TEAM_ID','GP']].gt(0).all(axis=None) or d[['PPP','POSS']].lt(0).any(axis=None):
        raise ValueError('Invalid Synergy identity/exposure/production')
    positive = d.loc[d.PPP.gt(0) & d.POSS.gt(0)].copy()
    positive['load_term'] = positive.PPP * positive.POSS
    cohort = positive.groupby(['year','PLAYER_ID'], as_index=False).agg(
        Offensive_Score=('load_term','sum'),positive_source_rows=('load_term','size'),
        source_teams=('TEAM_ID','nunique'))
    # Diagnostic only. The primary hypothesis deliberately retains the original
    # unweighted sum of stint-level PerGame rates, including traded-player
    # inflation, rather than silently changing the literal BAM convention.
    positive['gp_weighted_term']=positive.load_term*positive.GP
    type_diagnostic=positive.groupby(['year','PLAYER_ID','play_type'],as_index=False).agg(
        numerator=('gp_weighted_term','sum'),denominator=('GP','sum'))
    type_diagnostic['gp_weighted_creation_load_diagnostic']=type_diagnostic.numerator/type_diagnostic.denominator
    diagnostic=type_diagnostic.groupby(['year','PLAYER_ID'],as_index=False).gp_weighted_creation_load_diagnostic.sum()
    cohort=cohort.merge(diagnostic,on=['year','PLAYER_ID'],validate='one_to_one')
    cohort = cohort.rename(columns={'PLAYER_ID':'nba_id_off'})
    metadata = {}
    for year, g in cohort.groupby('year'):
        metadata[int(year)] = dict(median=float(g.Offensive_Score.median()),
            fifth_percentile=float(g.Offensive_Score.quantile(.05)),cohort_players=len(g),
            multi_team_players=int(g.source_teams.gt(1).sum()))
    return cohort, metadata


def synergy_table(payload):
    sets = payload.get('resultSets', payload.get('resultSet', []))
    if isinstance(sets, dict):sets=[sets]
    candidates=[s for s in sets if {'PLAYER_ID','PPP','POSS'} <= set(s.get('headers', []))]
    if len(candidates)!=1:raise ValueError('Expected exactly one player Synergy result set')
    return pd.DataFrame(candidates[0]['rowSet'], columns=candidates[0]['headers'])


def load_synergy(root):
    """Read only request-certified, fresh official RS PerGame response files."""
    root=Path(root); manifest_path=root/'manifest.json'
    if not manifest_path.exists():raise ValueError('Missing fresh Synergy request manifest')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='completed_official_RS_inputs':raise ValueError('Fresh Synergy fetch incomplete')
    sources={str(manifest_path.resolve()):sha(manifest_path)}; chunks=[]; seen=set()
    for entry in manifest.get('receipts', []):
        year=int(entry['year']); play=entry['play_type']
        if year not in YEARS:continue
        if play not in PLAYTYPES:raise ValueError('Unregistered fresh play type')
        if (year,play) in seen:raise ValueError('Duplicate fresh Synergy request')
        seen.add((year,play))
        path=Path(entry['raw']);meta_path=Path(entry['metadata'])
        if not path.is_absolute():path=root/path
        if not meta_path.is_absolute():meta_path=root/meta_path
        if sha(path)!=entry['sha256'] or sha(meta_path)!=entry['metadata_sha256']:
            raise ValueError('Fresh Synergy response/request checksum changed')
        metadata=json.loads(meta_path.read_text())
        expected=dict(LeagueID='00',PerMode='PerGame',PlayerOrTeam='P',SeasonYear=f'{year-1}-{year%100:02d}',
            SeasonType='Regular Season',PlayType=play,TypeGrouping='offensive')
        if metadata.get('parameters')!=expected or metadata.get('sha256')!=sha(path):
            raise ValueError('Fresh Synergy must be request-certified offensive PerGame Regular Season')
        frame=synergy_table(json.loads(path.read_text()))
        if frame.empty:raise ValueError('Empty fresh Synergy request')
        if not pd.to_numeric(frame.SEASON_ID,errors='raise').eq(20000+year-1).all():
            raise ValueError('Wrong response completed RS source year')
        if not frame.PLAY_TYPE.eq(play).all() or not frame.TYPE_GROUPING.str.lower().eq('offensive').all():
            raise ValueError('Wrong response playtype/side')
        frame['year']=year;frame['play_type']=play;chunks.append(frame)
        for input_path in [path,meta_path]:sources[str(input_path.resolve())]=sha(input_path)
    required={(year,play) for year in YEARS for play in PLAYTYPES}
    if seen!=required:raise ValueError('Fresh Synergy missing complete year/types: '+str(sorted(required-seen)))
    # Pin the fetcher's derived table/code as lineage, but always parse the
    # original raw responses above rather than using unverified derived loads.
    for path,digest in manifest.get('source_hashes',{}).items():
        if sha(path)!=digest:raise ValueError('Fresh fetcher lineage checksum changed')
        sources[str(Path(path).resolve())]=digest
    return pd.concat(chunks,ignore_index=True),sources


def dated_pairs(native, offensive, cohort, load_metadata, positions, ends):
    """Preserve all defender FGA before any missing-price or identity mask."""
    d=native.loc[native.year.isin(YEARS)].copy()
    if d.duplicated(['year','nba_id_off','nba_id_def']).any():raise ValueError('Duplicate native pair/year')
    primitives=['MATCHUP_FGA','MATCHUP_TOV','MATCHUP_FTA','PLAYER_PTS','PARTIAL_POSS']
    for col in primitives:d[col]=pd.to_numeric(d[col],errors='raise')
    # Never infer zero missing counts and never redistribute an excluded row.
    d['all_defender_fga']=d.groupby(['year','nba_id_def']).MATCHUP_FGA.transform('sum')
    d['source_rs_end']=d.year.map(ends)
    if d.source_rs_end.isna().any():raise ValueError('Missing RS completion date')
    d=d.merge(offensive,on=['year','nba_id_off'],how='left',validate='many_to_one')
    d=d.merge(cohort,on=['year','nba_id_off'],how='left',validate='many_to_one')
    d['creation_load_observed']=d.Offensive_Score.notna()
    d['fifth_percentile_load']=d.year.map({y:m['fifth_percentile'] for y,m in load_metadata.items()})
    d['median_fallback_load']=d.year.map({y:m['median'] for y,m in load_metadata.items()})
    d['Offensive_Score']=d.Offensive_Score.fillna(d.fifth_percentile_load)
    pos=positions.copy()
    if pos.duplicated(['year','nba_id']).any():raise ValueError('Ambiguous completed-year position')
    pos['dated_position_role']=pos.position.map(H.position_role)
    pos['dated_boost']=pos.dated_position_role.map(BOOST)
    pos['position_available_at']=(pd.to_datetime(pos.earliest_use_date) if 'earliest_use_date' in pos
        else pd.to_datetime(pos.source_last_game_date)+pd.Timedelta(days=1))
    d=d.merge(pos[['year','nba_id','position','dated_position_role','dated_boost','position_available_at']],
        left_on=['year','nba_id_off'],right_on=['year','nba_id'],how='left',validate='many_to_one')
    d['TS_PCT']=d.source_ts
    d['literal_raw_denom']=d.MATCHUP_FGA+.44*d.MATCHUP_FTA
    d['literal_raw_ts']=np.divide(d.PLAYER_PTS,2*d.literal_raw_denom,out=np.zeros(len(d)),where=d.literal_raw_denom.ne(0))
    d['literal_effective_fga']=d.MATCHUP_FGA+1.05*d.MATCHUP_TOV
    d['literal_effective_denom']=d.literal_effective_fga+.44*d.MATCHUP_FTA
    d['literal_effective_ts']=np.divide(d.PLAYER_PTS,2*d.literal_effective_denom,out=np.zeros(len(d)),where=d.literal_effective_denom.ne(0))
    d['literal_adjusted_load']=np.where(d.literal_raw_ts.gt(d.TS_PCT),d.median_fallback_load,d.Offensive_Score)
    valid=(d.MATCHUP_FGA.gt(0)&d.MATCHUP_TOV.ge(0)&d.MATCHUP_FTA.ge(0)&d.PLAYER_PTS.ge(0)
        &d.PARTIAL_POSS.gt(0)&d.all_defender_fga.gt(0)&d.ts_verified.eq(True)
        &d.dated_boost.notna()&d.literal_adjusted_load.ge(0))
    valid &= np.isfinite(d[primitives+['TS_PCT','literal_adjusted_load']]).all(axis=1)
    for column in ['pair_source_counts_valid','pair_identity_verified']:
        if column not in d:raise ValueError('Native aggregate lacks '+column)
        valid &= d[column].fillna(False)
    d['pair_verified']=valid
    d['dated_bam_pair']=(-10*(d.literal_effective_ts-d.TS_PCT)*d.literal_adjusted_load
        *(d.literal_effective_fga/d.all_defender_fga)*(1+d.dated_boost)).where(valid)
    return d


def load_native(folder, gate_path):
    gate_path=Path(gate_path);gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
    complete=(gate.get('status')=='passed_native_source_gate' and set(gate.get('verified_years',[]))==set(YEARS))
    native=[];sources={}
    if gate_path.exists():sources[str(gate_path.resolve())]=sha(gate_path)
    for year in YEARS:
        path=Path(folder)/f'native_{year}_rs_pairs_verified.parquet'
        if not path.exists():raise ValueError('Missing native aggregate '+str(year))
        digest=sha(path)
        if complete and gate.get('artifacts',{}).get(path.name)!=digest:
            raise ValueError('Native-only source gate checksum differs '+str(year))
        frame=pd.read_parquet(path);frame['year']=year;native.append(frame);sources[str(path.resolve())]=digest
    return pd.concat(native,ignore_index=True),sources,complete


def synergy_admission(path):
    path=Path(path)
    if not path.exists():return False,{}
    receipt=json.loads(path.read_text())
    complete=(receipt.get('status')=='passed_fresh_official_synergy_source_audit'
        and set(receipt.get('years',[]))==set(YEARS)
        and set(receipt.get('play_types',[]))==set(PLAYTYPES))
    sources={str(path.resolve()):sha(path)}
    if complete:
        for name,digest in receipt.get('source_hashes',{}).items():
            if sha(name)!=digest:raise ValueError('Independent Synergy source audit checksum changed')
            sources[str(Path(name).resolve())]=digest
    return complete,sources


def build(native_path=NATIVE, synergy_path=SYNERGY, positions_path=POSITIONS, lines_path=LINES,
          census_path=CENSUS, source_gate=SOURCE_GATE, out=OUT, cuts=CUTS, *,require_admitted=False,synergy_gate=SYNERGY_GATE):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    inputs=[Path(positions_path),Path(lines_path),Path(census_path),Path(__file__),HELPER]
    sources={str(path.resolve()):sha(path) for path in inputs}
    position_manifest=Path(positions_path).parent/'manifest.json'
    if position_manifest.exists():sources[str(position_manifest.resolve())]=sha(position_manifest)
    native,hashes,native_complete=load_native(native_path,source_gate);sources.update(hashes)
    synergy_complete,hashes=synergy_admission(synergy_gate);sources.update(hashes)
    complete=native_complete and synergy_complete
    if require_admitted and not complete:raise ValueError('Independent native/Synergy source gate incomplete')
    synergy,hashes=load_synergy(synergy_path);sources.update(hashes)
    cohort,metadata=creation_loads(synergy)
    if set(metadata)!=set(YEARS):raise ValueError('Missing fresh offensive-score source year')
    offensive=actual_ts(pd.read_parquet(lines_path))
    ends=H.completion_dates(pd.read_parquet(census_path))
    pairs=dated_pairs(native,offensive,cohort,metadata,pd.read_parquet(positions_path),ends)
    seasons,normalization=H.normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    features,provenance=H.lagged_snapshots(seasons,cuts,completion_calendar=ends)
    provenance['recipe']='freshnative+actualRSroundedTS+fresh3typeSynergy;sourceyearweightedz;RS-endTD550;latest4;oneK1000'
    offensive_export=offensive.merge(cohort,on=['year','nba_id_off'],how='outer',validate='one_to_one')
    offensive_export['creation_load_observed']=offensive_export.Offensive_Score.notna()
    offensive_export['median_fallback_load']=offensive_export.year.map({y:m['median'] for y,m in metadata.items()})
    offensive_export['fifth_percentile_load']=offensive_export.year.map({y:m['fifth_percentile'] for y,m in metadata.items()})
    artifacts={'pairs_dated_bam.parquet':pairs,'defender_seasons.parquet':seasons,
        'features.parquet':features,'feature_provenance.parquet':provenance,
        'offensive_inputs.parquet':offensive_export,'synergy_cohort.parquet':cohort}
    for name,frame in artifacts.items():frame.to_parquet(out/name,index=False)
    if any(sha(path)!=digest for path,digest in sources.items()):raise ValueError('Fresh BAM source changed during build')
    manifest=dict(status='research_candidate_not_admitted' if complete else 'partial_source_gate_smoke_only',
        source_gate_complete=complete,native_gate_complete=native_complete,synergy_gate_complete=synergy_complete,verified_source_years=YEARS if complete else [],pending_source_years=[] if complete else YEARS,
        missing_games=[],feature_groups={'bam':[FEATURE]},source_hashes=sources,code_sha256=sha(__file__),
        artifact_hashes={name:sha(out/name) for name in artifacts},cuts=list(cuts),rows=len(features),
        provenance_rows=len(provenance),protected_outcomes=0,source_years=YEARS,excluded_incomplete_years=[2026],
        unavailable_features=[FEATURE] if features[FEATURE].isna().all() else [],
        event_count_source='fresh native RS game aggregates; no legacy annual matchup table or metadata',
        source_rs_ending_dates={str(y):str(pd.Timestamp(day).date()) for y,day in ends.items()},
        fresh_creation_population=metadata,normalization=normalization,
        dated_position_mapping={'PG':'G','SG':'G','SF':'F','PF':'F','C':'C','hybrids':'first-listed'},position_boosts=BOOST,
        formula='-10*(PTS/[2*(FGA+1.05TOV+.44FTA)]-round(actual_RS_TS,3))*adjusted_creation_load*(FGA+1.05TOV)/ALLdefenderFGA*(1+sourceyearattackerpositionboost)',
        creation_branch='rawmatchupTS>sameyearTS: full positivePPP/POSS three-type cohort median; otherwise player sum(PPP*POSS), missing player fullcohort fifth percentile',
        stabilization='covered-exposureweightedsourceyearz;exactRS-endTD550;latest4completedRS;oneK1000',
        historical_publication_vintage_certified=False,
        limitations=['Fresh official retrospective measurement extraction, not certified original historical publication vintages',
            'Completed source-year actual TS and Synergy load, never unfinished same-year or current-game forecasts',
            'Native historical FT coverage can be incomplete; literal legacy formula retains measured source counts',
            'All defender FGA retained before excluded identity/position/TS pair masks; coverage explicit; missing stays missing',
            '2018-2025 only; no incomplete2026 source and no2027 outcomes',
            'Literal unweighted sum(PPP*POSS) across team stints inflates some traded-player loads; GP-weighted load is diagnostic only',
            'V47 conditional-FG residual and failed V48 annual-source build remain separate and unchanged'])
    write_json(out/'manifest.json',manifest)
    print(json.dumps({key:manifest[key] for key in ['status','rows','verified_source_years','pending_source_years']}))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in [('native',NATIVE),('synergy',SYNERGY),('positions',POSITIONS),('lines',LINES),
                         ('census',CENSUS),('source-gate',SOURCE_GATE),('synergy-gate',SYNERGY_GATE),('out',OUT)]:
        parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--cuts',nargs='+',default=CUTS)
    parser.add_argument('--require-admitted',action='store_true')
    a=parser.parse_args();build(a.native,a.synergy,a.positions,a.lines,a.census,a.source_gate,a.out,a.cuts,require_admitted=a.require_admitted,synergy_gate=a.synergy_gate)

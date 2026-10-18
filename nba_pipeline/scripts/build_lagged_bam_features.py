"""Completed-season literal BAM with dated positions, behind a source gate.

This is distinct from the frozen per-game conditional-FG matchup_bam feature.
The legacy formula is retained, including its nonlinear creation-load branch,
turnover adjustment, total-defender-FGA denominator and attacker position
boost. Only source-year positions replace retrospective current profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ANNUAL=ROOT.parents[1]/'vpm/vpm/data/matchup/matchups_two_rs_2018_2026.parquet'
POSITIONS=ROOT/'external/matchup_bpm_v1/player_seasons.parquet'
LINES=ROOT/'results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet'
CENSUS=ROOT/'cache/matchup_games_v1/game_census.parquet'
CONCORDANCE=ROOT/'results/legacy_bam_lineage_audit_v1/annual_native_concordance.json'
OUT=ROOT/'results/estimated_skills/lagged_bam_v1'
CUTS=['2018-10-01','2019-10-01','2020-12-01',*[f'{year}-10-01' for year in range(2021,2026)]]
BOOST={'G':.3025,'G-F':.29,'F-G':.275,'F':.25,'F-C':.225,'C-F':.175,'C':.125}
FEATURE='matchup_dated_bam'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')


def position_role(value):
    if not isinstance(value,str):return None
    # Historical BRef rows here use five roles. If a future archived row has
    # an ordered hybrid, retain its primary (first-listed) role explicitly.
    return {'PG':'G','SG':'G','G':'G','SF':'F','PF':'F','F':'F','C':'C'}.get(value.upper().split('-')[0])


def completion_dates(census):
    frame=census.loc[census.season_type.eq('rs')&census.season.between(2018,2025)].copy()
    if frame.game_id.duplicated().any():raise ValueError('Ambiguous completed RS census')
    frame['game_date']=pd.to_datetime(frame.game_date).astype('datetime64[ns]')
    dates=frame.groupby('season').game_date.max()
    if set(dates.index)!=set(range(2018,2026)):raise ValueError('Missing 2018–2025 RS completion dates')
    return dates


def source_ts_checks(annual,lines,tolerance=.00051):
    """Verify rounded same-year advanced TS against raw actual RS counts."""
    d=lines.loc[lines.season_type.eq('rs')&lines.season.between(2018,2025)]
    sums=d.groupby(['season','player_id'])[['pts','fga','fta']].sum()
    observed=sums.pts/(2*(sums.fga+.44*sums.fta).where((sums.fga+.44*sums.fta).gt(0)))
    observed.index.names=['year','nba_id_off']
    source=annual.groupby(['year','nba_id_off']).TS_PCT.agg(['first','nunique'])
    if source['nunique'].gt(1).any():raise ValueError('Attacker TS varies within source year')
    source=source.join(observed.rename('observed_ts'))
    source['ts_gap']=source['first']-source.observed_ts
    source['ts_verified']=source['first'].ge(0)&source.ts_gap.abs().le(tolerance)
    return source.reset_index().rename(columns={'first':'source_ts'})


def dated_pairs(native,annual,positions,lines,ends):
    """Reconstruct every literal term before filtering unverified pair rows."""
    source=annual.loc[annual.year.between(2018,2025)&annual.playoffs.eq(0)].copy()
    for name in ['TS_PCT','Offensive_Score','Adjusted_Offensive_Score','Matchup_TS%']:
        source[name]=pd.to_numeric(source[name],errors='raise')
    d=native.loc[native.year.between(2018,2025)].copy()
    if d.duplicated(['year','nba_id_off','nba_id_def']).any():raise ValueError('Duplicate annual attacker/defender identity')
    for name in ['MATCHUP_FGA','MATCHUP_TOV','MATCHUP_FTA','PLAYER_PTS','PARTIAL_POSS']:
        d[name]=pd.to_numeric(d[name],errors='raise')
    if source.groupby(['year','nba_id_off']).Offensive_Score.nunique().gt(1).any():
        raise ValueError('Creation load varies within source player-year')
    pricing=source.groupby(['year','nba_id_off'],as_index=False)[['TS_PCT','Offensive_Score']].first()
    d=d.merge(pricing,on=['year','nba_id_off'],how='left',validate='many_to_one')
    # These legacy archives omit zero-FGA pairs. Keep all rows in the full
    # defender denominator before any identity, TS, position or score masking.
    d['all_defender_fga']=d.groupby(['year','nba_id_def']).MATCHUP_FGA.transform('sum')
    d['source_rs_end']=d.year.map(ends)
    if d.source_rs_end.isna().any():raise ValueError('Missing source completion date')
    pos=positions.rename(columns={'nba_id':'player_id','year':'source_year'}).copy()
    if pos.duplicated(['source_year','player_id']).any():raise ValueError('Ambiguous dated position identity')
    pos['dated_position_role']=pos.position.map(position_role)
    pos['dated_boost']=pos.dated_position_role.map(BOOST)
    available=(pd.to_datetime(pos['earliest_use_date']) if 'earliest_use_date' in pos
               else pd.to_datetime(pos.source_last_game_date)+pd.Timedelta(days=1))
    pos['position_available_at']=available
    keep=['source_year','player_id','position','dated_position_role','dated_boost','position_available_at']
    d=d.merge(pos[keep],left_on=['year','nba_id_off'],right_on=['source_year','player_id'],how='left',validate='many_to_one')
    checks=source_ts_checks(source,lines)
    d=d.merge(checks[['year','nba_id_off','observed_ts','ts_gap','ts_verified']],on=['year','nba_id_off'],how='left',validate='many_to_one')
    d['literal_raw_denom']=d.MATCHUP_FGA+.44*d.MATCHUP_FTA
    d['literal_raw_ts']=np.divide(d.PLAYER_PTS,2*d.literal_raw_denom,out=np.zeros(len(d)),where=d.literal_raw_denom.ne(0))
    d['literal_effective_fga']=d.MATCHUP_FGA+1.05*d.MATCHUP_TOV
    d['literal_effective_denom']=d.literal_effective_fga+.44*d.MATCHUP_FTA
    d['literal_effective_ts']=np.divide(d.PLAYER_PTS,2*d.literal_effective_denom,out=np.zeros(len(d)),where=d.literal_effective_denom.ne(0))
    medians={}
    for year,g in source.groupby('year'):
        # Original median used the full positive-PPP Synergy cohort, not the
        # observed matchup attackers. Recover its unique stored source-year
        # branch value rather than silently changing that population.
        branch=g.loc[pd.to_numeric(g['Matchup_TS%'],errors='coerce').gt(g.TS_PCT),'Adjusted_Offensive_Score'].dropna()
        values=np.unique(branch.to_numpy(float))
        if len(values)!=1 or not np.isfinite(values[0]) or values[0]<0:
            raise ValueError(f'Cannot recover unique legacy median creation load for {year}')
        medians[int(year)]=float(values[0])
    d['median_fallback_load']=d.year.map(medians)
    d['literal_adjusted_load']=np.where(d.literal_raw_ts.gt(d.TS_PCT),d.median_fallback_load,d.Offensive_Score)
    valid=(d.MATCHUP_FGA.gt(0)&d.MATCHUP_TOV.ge(0)&d.MATCHUP_FTA.ge(0)&d.PLAYER_PTS.ge(0)
           &d.PARTIAL_POSS.gt(0)&d.all_defender_fga.gt(0)&d.ts_verified.fillna(False)
           &d.dated_boost.notna()&d.literal_adjusted_load.ge(0))
    numeric=['MATCHUP_FGA','MATCHUP_TOV','MATCHUP_FTA','PLAYER_PTS','PARTIAL_POSS','TS_PCT','literal_adjusted_load']
    valid&=np.isfinite(d[numeric]).all(axis=1)
    for name in ['pair_source_counts_valid','pair_identity_verified']:
        if name not in d:raise ValueError('Verified native BAM aggregate lacks '+name)
        valid&=d[name].fillna(False)
    d['pair_verified']=valid
    d['dated_bam_pair']=-10*(d.literal_effective_ts-d.TS_PCT)*d.literal_adjusted_load*(d.literal_effective_fga/d.all_defender_fga)*(1+d.dated_boost)
    d['dated_bam_pair']=d.dated_bam_pair.where(valid)
    return d,checks,medians


def normalize_seasons(pairs,column,*,verified_only):
    d=pairs.copy()
    d['covered_exposure']=d.PARTIAL_POSS.where(d[column].notna()&(d.pair_verified if verified_only else True),0.)
    d['contribution']=d[column].where(d.covered_exposure.gt(0),0.)
    output=d.groupby(['year','nba_id_def'],as_index=False).agg(
        season_raw=('contribution','sum'),season_exposure=('covered_exposure','sum'),
        all_source_exposure=('PARTIAL_POSS','sum'),source_pairs=('PARTIAL_POSS','size'),
        source_rs_end=('source_rs_end','max'),position_available_at=('position_available_at','max'))
    output['source_coverage']=output.season_exposure/output.all_source_exposure.where(output.all_source_exposure.gt(0))
    output.loc[output.season_exposure.le(0),'season_raw']=np.nan
    output['season_z']=np.nan;receipts=[]
    for year,index in output.groupby('year').groups.items():
        group=output.loc[index];good=group.season_exposure.gt(0)&group.season_raw.notna()
        if good.sum()<2:
            receipts.append(dict(year=int(year),center=None,scale=None,verified_defenders=int(good.sum()),
                                 exposure=float(group.season_exposure.sum()),column=column,status='insufficient_verified_defenders'))
            continue
        values=group.loc[good,'season_raw'].to_numpy();weights=group.loc[good,'season_exposure'].to_numpy()
        center=float(np.average(values,weights=weights))
        scale=float(np.sqrt(np.average((values-center)**2,weights=weights)))
        if not np.isfinite(scale) or scale<=0:
            receipts.append(dict(year=int(year),center=center,scale=None,verified_defenders=int(good.sum()),
                                 exposure=float(weights.sum()),column=column,status='zero_verified_dispersion'))
            continue
        output.loc[group.loc[good].index,'season_z']=(values-center)/scale
        receipts.append(dict(year=int(year),center=center,scale=scale,verified_defenders=int(good.sum()),
                             exposure=float(weights.sum()),column=column))
    return output.rename(columns={'nba_id_def':'player_id'}),receipts


def lagged_snapshots(seasons,cuts,*,half_life=550.,shrinkage=1000.,max_seasons=4,completion_calendar=None):
    rows=[];provenance=[]
    for value in cuts:
        cut=pd.Timestamp(value)
        if cut>=pd.Timestamp('2026-10-01'):raise ValueError('2027 emission protected')
        eligible=seasons.loc[seasons.source_rs_end.lt(cut)&seasons.year.between(2018,2025)].copy()
        years=(sorted(completion_calendar.loc[completion_calendar.lt(cut)].index)[-max_seasons:]
               if completion_calendar is not None else sorted(eligible.year.unique())[-max_seasons:])
        eligible=eligible.loc[eligible.year.isin(years)]
        if eligible.empty:continue
        decay=np.exp2(-(cut-pd.to_datetime(eligible.source_rs_end)).dt.total_seconds().to_numpy()/(86400*half_life))
        eligible['effective_exposure']=eligible.season_exposure*decay
        eligible['all_effective_exposure']=eligible.all_source_exposure*decay
        eligible['numerator']=eligible.season_z*eligible.effective_exposure
        for player,g in eligible.groupby('player_id'):
            good=g.effective_exposure.gt(0)&g.season_z.notna()
            exposure=float(g.loc[good,'effective_exposure'].sum())
            rawtotal=float(g.all_effective_exposure.sum())
            score=float(g.loc[good,'numerator'].sum()/(exposure+shrinkage)) if exposure>0 else np.nan
            last=g.loc[good,'source_rs_end'].max()
            position_date=pd.to_datetime(g.loc[good,'position_available_at']).max()
            fit_end=max(last,position_date) if pd.notna(last) and pd.notna(position_date) else last
            if pd.notna(fit_end) and fit_end>=cut:raise ValueError('Source positions not available by emission cutoff')
            rows.append(dict(player_id=int(player),estimate_date=cut,matchup_exposure=exposure,
                matchup_dated_bam=score,matchup_dated_bam__exposure=exposure,
                matchup_dated_bam__numerator=float(g.loc[good,'numerator'].sum()),
                all_effective_source_exposure=rawtotal,
                matchup_dated_bam_unshrunk=float(g.loc[good,'numerator'].sum()/exposure) if exposure>0 else np.nan,
                source_seasons=int(g.loc[good,'year'].nunique()),latest_source_year=int(g.loc[good,'year'].max()) if good.any() else None,
                source_coverage=exposure/rawtotal if rawtotal>0 else 0.,reliability=exposure/(exposure+shrinkage)))
            provenance.append(dict(player_id=int(player),estimate_date=cut,feature=FEATURE,last_observation_date=last,
                parameter_fit_end=fit_end,exposure=exposure,source_coverage=exposure/rawtotal if rawtotal>0 else 0.,
                recipe='completedRS;source-yearweightedz;exactRS-endTD550;atmost4years;oneK1000shrink'))
    if not rows:raise ValueError('No eligible completed BAM history')
    return pd.DataFrame(rows),pd.DataFrame(provenance)


def admission(path,annual_hash,required=range(2018,2026)):
    if not Path(path).exists():return [],list(required),None
    receipt=json.loads(Path(path).read_text())
    if receipt.get('annual_sha256')!=annual_hash:raise ValueError('Concordance receipt belongs to another annual artifact')
    verified=sorted(set(map(int,receipt.get('verified_years',[]))))
    details={int(row['year']):row for row in receipt.get('per_year_details',[])}
    for year in verified:
        gate=details.get(year,{}).get('gate',{})
        if not gate.get('passed') or not gate.get('all_games_resolved'):
            raise ValueError('Concordance verified-year list conflicts with its complete-year source gate')
    return verified,sorted(set(required)-set(verified)),receipt


def native_sources(receipt,folder):
    if receipt is None or not receipt.get('verified_years'):raise ValueError('No fully verified native RS source years')
    chunks=[];hashes={}
    for year in receipt['verified_years']:
        if int(year) not in range(2018,2026):continue
        name=f'native_{year}_rs_pairs_verified.parquet';path=Path(folder)/name
        if sha(path)!=receipt['artifacts'][name]:raise ValueError('Native aggregate changed '+str(year))
        d=pd.read_parquet(path);d['year']=int(year);chunks.append(d);hashes[str(path.resolve())]=sha(path)
    return pd.concat(chunks,ignore_index=True),hashes


def build(annual_path=ANNUAL,positions_path=POSITIONS,lines_path=LINES,census_path=CENSUS,
          concordance_path=CONCORDANCE,out=OUT,cuts=CUTS,*,require_admitted=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    inputs=[Path(annual_path),Path(positions_path),Path(lines_path),Path(census_path),Path(__file__)]
    sources={str(p.resolve()):sha(p) for p in inputs}
    for path in [Path(annual_path).with_suffix('.manifest.json'),Path(positions_path).parent/'manifest.json']:
        if path.exists():sources[str(path.resolve())]=sha(path)
    verified,pending,receipt=admission(concordance_path,sha(annual_path))
    if Path(concordance_path).exists():sources[str(Path(concordance_path).resolve())]=sha(concordance_path)
    if pending and require_admitted:raise ValueError(f'Native annual concordance pending for {pending}; no model runs admitted')
    annual=pd.read_parquet(annual_path);annual=annual.loc[annual.year.between(2018,2025)].copy()
    positions=pd.read_parquet(positions_path);lines=pd.read_parquet(lines_path)
    ends=completion_dates(pd.read_parquet(census_path))
    native,native_hashes=native_sources(receipt,Path(concordance_path).parent);sources.update(native_hashes)
    pairs,checks,medians=dated_pairs(native,annual,positions,lines,ends)
    seasons,normalization=normalize_seasons(pairs,'dated_bam_pair',verified_only=True)
    features,provenance=lagged_snapshots(seasons,cuts,completion_calendar=ends)
    archived=annual.copy();archived['archival_bam_pair_diagnostic']=pd.to_numeric(archived.BAM_inverted,errors='coerce')
    archived['source_rs_end']=archived.year.map(ends);archived['position_available_at']=archived.source_rs_end+pd.Timedelta(days=1)
    archival,arch_norm=normalize_seasons(archived,'archival_bam_pair_diagnostic',verified_only=False)
    diagnostic,_=lagged_snapshots(archival,cuts,completion_calendar=ends)
    diagnostic=diagnostic.rename(columns={name:'archival_'+name for name in diagnostic if name.startswith('matchup_')})
    artifacts={'pairs_dated_bam.parquet':pairs,'defender_seasons.parquet':seasons,
        'features.parquet':features,'feature_provenance.parquet':provenance,'ts_verification.parquet':checks,
        'archival_diagnostics.parquet':diagnostic,'archival_defender_seasons.parquet':archival}
    for name,frame in artifacts.items():frame.to_parquet(out/name,index=False)
    if any(sha(Path(path))!=value for path,value in sources.items()):raise ValueError('Source changed during BAM build')
    manifest=dict(status='partial_download_smoke_only' if pending else 'research_candidate_not_admitted',
        source_gate_complete=not pending,verified_source_years=verified,pending_source_years=pending,missing_games=[],
        feature_groups={'bam':[FEATURE]},unavailable_features=[FEATURE] if features[FEATURE].isna().all() else [],
        source_hashes=sources,code_sha256=sha(__file__),artifact_hashes={name:sha(out/name) for name in artifacts},
        cuts=list(cuts),rows=len(features),provenance_rows=len(provenance),protected_outcomes=0,
        source_years=list(range(2018,2026)),excluded_incomplete_years=[2026],
        event_count_source='freshnativeRSgameaggregatesonly;annualarchiveusedonlyforsourceyearTS/PPPload/mediananddiagnostic',
        source_rs_ending_dates={str(int(y)):str(pd.Timestamp(day).date()) for y,day in ends.items()},
        source_median_fallback_load=medians,normalization=normalization,archival_normalization=arch_norm,
        dated_position_mapping={'PG':'G','SG':'G','SF':'F','PF':'F','C':'C','hybrids':'first-listedBRefrole'},
        position_boosts=BOOST,formula='negative10*(PTS/[2*(FGA+1.05TOV+.44FTA)]-sameyearTS)*adjusted_creation_load*(FGA+1.05TOV)/ALLdefenderFGA*(1+sourceyearattackerpositionboost)',
        creation_branch='rawmatchupTS>sameyearTSusesoriginalsourceyearPPPpopulationmedian;otherwisearchivedsourceyearOffensive_Score',
        ts_verification_tolerance=.00051,stabilization='within-sourceyearcovered-exposureweightedz;exactRS-endTD550;latest4completedRS;oneK1000shrinkymean0',
        historical_publication_vintage_certified=False,
        limitations=['Notv47conditionalFGmatchup_bam;literallegacyformulaexceptdatedattackerpositions',
            'Annualsourcehasnointraseasondates;onlycompletedRSadmission',
            'AnnualarchivecontainspositiveFGA-pairsonly;partialpossessionsarenotfullassignmentcoverage',
            'ALLdefenderFGAdenominatorretainedbeforeunverifiedpairmasking;validexposureandcoverageexplicit',
            'Archivalstatic-positionBAMisdiagnosticonlyandnotinthefeature_groups',
            '2025sourcequantitiesrounded;archivalBAMdiagnosticisnotexactlyreproduciblefromroundedterms',
            'Source-yearTSandSynergycreationloadarecompletedperiodmeasurements,notpregamecurrent-seasonforecasts'])
    write_json(out/'manifest.json',manifest)
    print(json.dumps({key:manifest[key] for key in ['status','rows','verified_source_years','pending_source_years']}))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in [('annual',ANNUAL),('positions',POSITIONS),('lines',LINES),('census',CENSUS),('concordance',CONCORDANCE),('out',OUT)]:
        parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--cuts',nargs='+',default=CUTS);parser.add_argument('--require-admitted',action='store_true')
    a=parser.parse_args();build(a.annual,a.positions,a.lines,a.census,a.concordance,a.out,a.cuts,require_admitted=a.require_admitted)

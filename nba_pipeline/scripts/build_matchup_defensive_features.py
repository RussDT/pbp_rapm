"""Research-only dated defender summaries from native per-game matchups.

Raw matchup pairs remain RAPM-owned. VPM receives only stabilized player-cut
summaries and explicit coverage/provenance. Assignment is observational, not
causal defensive credit. Every attacker price excludes the entire game day.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'cache/matchup_games_v1'
OUT = ROOT/'results/estimated_skills/matchup_defense_v1'
CUTS = ['2018-10-01','2019-10-01','2020-12-01',*[f'{y}-10-01' for y in range(2021,2026)],'2026-09-01']
FEATURE_GROUPS = {
    'assignment':['matchup_assignment_difficulty','matchup_assignment_bpm','matchup_assignment_obpm','matchup_assignment_pos_difficulty',
        'matchup_assignment_assist_load100','matchup_assignment_usage_load100'],
    'suppression':['matchup_points_saved100','matchup_fg_points_saved100','matchup_ft_points_saved100',
        'matchup_fg_points_allowed100','matchup_points_allowed100',
        'matchup_shot_frequency_saved100','matchup_three_frequency_saved100','matchup_assists_saved100',
        'matchup_turnovers_above_expected100'],
    'bam':['matchup_bam'],
    'events':['matchup_blocks100','matchup_help_blocks100','matchup_shooting_fouls100',
        'matchup_potential_assists_allowed100','matchup_ft_frequency_saved100'],
    'workload':['matchup_primary_share','matchup_concentration','matchup_coverage'],
    'position_mix':['matchup_guard_share','matchup_wing_share','matchup_big_share'],
}
FEATURES = sum(FEATURE_GROUPS.values(),[])
COUNTS = ['partialPossessions','matchupFieldGoalsMade','matchupFieldGoalsAttempted',
          'matchupThreePointersMade','matchupThreePointersAttempted','matchupFreeThrowsMade',
          'matchupFreeThrowsAttempted','matchupAssists','matchupPotentialAssists','matchupTurnovers',
          'matchupBlocks','helpBlocks','shootingFouls','playerPoints']
PRICE_COUNTS = ['off_poss','fg2m','fg2a','three_fgm','three_fga','ftm','fta','fga','ast','tov']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(series):
    return series.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')


def previous_day_evidence(lines, requests, *, prior_possessions=1000., prior_shots=100.):
    """Price attackers from raw PBP counts strictly before each request day.

    Player and league counts are independently joined as-of, so an idle
    player's population fallback can update without ingesting today's games.
    Initial requests without any earlier league evidence remain unpriced.
    """
    d=lines.copy();d['game_date']=pd.to_datetime(d.game_date).dt.normalize().astype('datetime64[ns]')
    d=d.loc[d.off_poss.gt(0)].copy()
    d['fg2m']=d.rim_fgm+d.mid_fgm;d['fg2a']=d.rim_fga+d.mid_fga
    if d.duplicated(['game_id','player_id']).any():raise ValueError('Duplicate own player-game counts')
    daily=d.groupby(['player_id','game_date'],as_index=False)[PRICE_COUNTS].sum().sort_values(['player_id','game_date'])
    anchor=pd.Timestamp('2013-01-01')
    factor=np.exp2((daily.game_date-anchor).dt.total_seconds().to_numpy()/(86400*550.))
    daily[PRICE_COUNTS]=daily[PRICE_COUNTS].multiply(factor,axis=0)
    daily[PRICE_COUNTS]=daily.groupby('player_id')[PRICE_COUNTS].cumsum()
    daily=daily.rename(columns={'game_date':'player_evidence_day',**{c:'p_'+c for c in PRICE_COUNTS}})
    r=requests[['offense_player_id','game_date']].drop_duplicates().copy()
    r['game_date']=pd.to_datetime(r.game_date).dt.normalize().astype('datetime64[ns]');r['order']=np.arange(len(r))
    r=r.rename(columns={'offense_player_id':'player_id'})
    merged=pd.merge_asof(r.sort_values('game_date'),daily.sort_values('player_evidence_day'),
        left_on='game_date',right_on='player_evidence_day',by='player_id',allow_exact_matches=False,direction='backward')
    league=d.groupby('game_date')[PRICE_COUNTS].sum().sort_index().reset_index()
    factor=np.exp2((league.game_date-anchor).dt.total_seconds().to_numpy()/(86400*550.))
    league[PRICE_COUNTS]=league[PRICE_COUNTS].multiply(factor,axis=0).cumsum()
    league=league.rename(columns={'game_date':'league_evidence_day',**{c:'l_'+c for c in PRICE_COUNTS}})
    merged=pd.merge_asof(merged.sort_values('game_date'),league.sort_values('league_evidence_day'),
        left_on='game_date',right_on='league_evidence_day',allow_exact_matches=False,direction='backward')
    inverse=np.exp2(-(merged.game_date-anchor).dt.total_seconds().to_numpy()/(86400*550.))
    for prefix in ['p_','l_']:
        columns=[prefix+c for c in PRICE_COUNTS]
        merged[columns]=merged[columns].multiply(inverse,axis=0)
    for c in PRICE_COUNTS:merged['p_'+c]=merged['p_'+c].fillna(0.)
    for name,numerator,denominator,k in [
        ('fg_points','fg_points','off_poss',prior_possessions),('ft_points','ftm','off_poss',prior_possessions),
        ('fga','fga','off_poss',prior_possessions),('three_fga','three_fga','off_poss',prior_possessions),
        ('fta','fta','off_poss',prior_possessions),('ast','ast','off_poss',prior_possessions),
        ('tov','tov','off_poss',prior_possessions),('fg2pct','fg2m','fg2a',prior_shots),
        ('fg3pct','three_fgm','three_fga',prior_shots)]:
        if numerator=='fg_points':
            pn=2*merged.p_fg2m+3*merged.p_three_fgm;ln=2*merged.l_fg2m+3*merged.l_three_fgm
        else:pn=merged['p_'+numerator];ln=merged['l_'+numerator]
        league_rate=ln/merged['l_'+denominator].where(merged['l_'+denominator].gt(0))
        merged['expected_'+name]=(pn+k*league_rate)/(merged['p_'+denominator]+k)
    merged['expected_total_points']=merged.expected_fg_points+merged.expected_ft_points
    merged['opponent_price_reliability']=merged.p_off_poss/(merged.p_off_poss+prior_possessions)
    for c in ['player_evidence_day','league_evidence_day']:
        if not (merged[c].isna()|merged[c].lt(merged.game_date)).all():raise ValueError('Same-day price leakage')
    columns=['player_id','game_date','player_evidence_day','league_evidence_day','opponent_price_reliability']
    columns += [c for c in merged if c.startswith('expected_')]
    return merged.sort_values('order')[columns].rename(columns={'player_id':'offense_player_id'})


def lagged_metadata(pairs, table=None):
    """Exact IDs and preceding completed season only; no fuzzy or live joins.

    Optional input schema: player_id,season,bpm,position. Each player-season
    must already be unique. Only the immediately preceding completed season
    is admitted; no prior record yields missing BPM and position.
    """
    r=pairs[['offense_player_id','season','game_date']].drop_duplicates().copy()
    r['game_date']=pd.to_datetime(r.game_date).astype('datetime64[ns]')
    r['opponent_bpm']=np.nan;r['opponent_position']=pd.Series(None,index=r.index,dtype=object)
    r['opponent_obpm']=np.nan
    r['opponent_bpm_position_mean']=np.nan;r['metadata_season']=np.nan
    if table is None:return r
    t=table.copy().rename(columns={'nba_id':'player_id','year':'season'})
    if t.duplicated(['player_id','season']).any():raise ValueError('Ambiguous BPM/position player-season')
    for season in sorted(r.season.unique()):
        previous=t.loc[t.season.eq(season-1)].set_index('player_id')
        ix=r.season.eq(season)
        for target,source in [('opponent_bpm','bpm'),('opponent_obpm','obpm'),('opponent_position','position'),('metadata_season','season')]:
            if source in previous:r.loc[ix,target]=r.loc[ix,'offense_player_id'].map(previous[source])
        if 'bpm' in previous and 'position' in previous:
            # Means are fitted on exactly the same earlier metadata cohort.
            good=previous.bpm.notna()&previous.position.notna()
            weights=previous['minutes'] if 'minutes' in previous else pd.Series(1.,index=previous.index)
            good&=weights.gt(0)
            num=(previous.loc[good,'bpm']*weights.loc[good]).groupby(previous.loc[good,'position']).sum()
            den=weights.loc[good].groupby(previous.loc[good,'position']).sum()
            means=num/den
            r.loc[ix,'opponent_bpm_position_mean']=r.loc[ix,'opponent_position'].map(means)
        for date_column in ['source_last_game_date','earliest_use_date']:
            if date_column in previous:
                date=pd.to_datetime(r.loc[ix,'offense_player_id'].map(previous[date_column]))
                available=date.isna()|date.lt(r.loc[ix,'game_date'])
                if not available.all():raise ValueError('Opponent metadata reaches matchup game day')
    if not (r.metadata_season.isna()|r.metadata_season.lt(r.season)).all():raise ValueError('Current-season metadata')
    # Current BPM2.0 backhistory predates the metric's publication. Position
    # remains usable, but pricing by modern BPM starts at a conservative floor.
    historical=r.game_date.lt('2020-03-01')
    r.loc[historical,['opponent_bpm','opponent_obpm','opponent_bpm_position_mean']]=np.nan
    return r


def coverage_audit(pairs, lines, *, fga_tolerance=.15):
    """Lane-specific source coverage, rather than interpreting absent counts as zero."""
    official=lines.groupby('game_id')[['off_poss','def_poss','fga','fta','ast','tov','blk','pf_committed']].sum()
    sums=pairs.groupby('game_id')[COUNTS].sum(min_count=1)
    g=sums.join(official,how='left')
    g['fga_ratio']=g.matchupFieldGoalsAttempted/g.fga.where(g.fga.gt(0))
    g['fta_ratio']=g.matchupFreeThrowsAttempted/g.fta.where(g.fta.gt(0))
    g['assist_ratio']=g.matchupAssists/g.ast.where(g.ast.gt(0))
    g['turnover_ratio']=g.matchupTurnovers/g.tov.where(g.tov.gt(0))
    g['block_ratio']=g.matchupBlocks/g.blk.where(g.blk.gt(0))
    g['assignment_ratio']=g.partialPossessions/g.off_poss.where(g.off_poss.gt(0))
    valid=pairs.groupby('game_id').source_counts_valid.all() if 'source_counts_valid' in pairs else pd.Series(True,index=g.index)
    if 'row_counts_valid' in pairs:
        valid&=pairs.groupby('game_id').row_counts_valid.all()
    g['assignment_valid']=valid&g.assignment_ratio.between(.5,1.15)
    g['fg_valid']=g.assignment_valid&g.fga_ratio.between(1-fga_tolerance,1+fga_tolerance)
    g['fg_tight_valid']=g.assignment_valid&g.fga_ratio.between(.95,1.05)
    g['fg_overcount']=g.fga_ratio.gt(1.)
    g['ft_valid']=g.assignment_valid&((g.fta.eq(0)&g.matchupFreeThrowsAttempted.eq(0))|g.fta_ratio.between(.70,1.05))
    g['ft_structurally_missing']=g.fta.gt(0)&g.matchupFreeThrowsAttempted.eq(0)
    g['assist_valid']=g.assignment_valid&g.matchupAssists.gt(0)&g.matchupAssists.le(g.ast*1.15)
    g['potential_assist_valid']=g.assignment_valid&g.matchupPotentialAssists.gt(0)
    g['tov_valid']=g.assignment_valid&g.matchupTurnovers.gt(0)&g.matchupTurnovers.le(g.tov*1.15)
    g['block_valid']=g.assignment_valid&g.matchupBlocks.notna()&g.matchupBlocks.le(g.blk*1.15)&(g.matchupBlocks.gt(0)|g.blk.eq(0))
    g['help_block_valid']=g.block_valid&g.helpBlocks.gt(0)
    g['foul_valid']=g.assignment_valid&g.shootingFouls.gt(0)&g.shootingFouls.le(g.pf_committed*1.15)
    g['counts_valid']=valid
    return g.reset_index()


def defender_game_atoms(pairs, lines, *, metadata=None, game_context=None, fga_tolerance=.15):
    """Private diagnostic atoms; only snapshots are admitted to the VPM seam."""
    p=pairs.copy();p.game_id=ids(p.game_id);p.game_date=pd.to_datetime(p.game_date).dt.normalize()
    p=p.loc[p.season_type.isin(['rs','ps'])&p.season.le(2026)].copy()
    if p.empty:raise ValueError('No ordinary RS/PS matchup rows')
    if p.duplicated(['game_id','offense_player_id','defense_player_id']).any():raise ValueError('Duplicate native matchup pair')
    for c in COUNTS:
        if c not in p:p[c]=np.nan
        p[c]=pd.to_numeric(p[c],errors='raise')
    p['row_counts_valid']=~p[COUNTS].lt(0).any(axis=1)
    for m,a in [('matchupFieldGoalsMade','matchupFieldGoalsAttempted'),('matchupThreePointersMade','matchupThreePointersAttempted'),('matchupFreeThrowsMade','matchupFreeThrowsAttempted')]:
        p['row_counts_valid']&=~p[m].gt(p[a])
    p['row_counts_valid']&=~p.matchupThreePointersAttempted.gt(p.matchupFieldGoalsAttempted)
    p['row_counts_valid']&=~p.matchupThreePointersMade.gt(p.matchupFieldGoalsMade)
    official=lines.copy();official.game_id=ids(official.game_id)
    index=official.groupby(['game_id','player_id'])[['off_poss','def_poss']].sum()
    off_index=pd.MultiIndex.from_arrays([p.game_id,p.offense_player_id],names=['game_id','player_id'])
    def_index=pd.MultiIndex.from_arrays([p.game_id,p.defense_player_id],names=['game_id','player_id'])
    p['identity_valid']=(index.off_poss.reindex(off_index).to_numpy()>0)&(index.def_poss.reindex(def_index).to_numpy()>0)
    if game_context is not None:
        context=game_context.copy();context.game_id=ids(context.game_id)
        if context.duplicated(['game_id','player_id']).any():raise ValueError('Ambiguous PBP player-team identity')
        teams=context.set_index(['game_id','player_id']).team_id
        p['identity_valid']&=(teams.reindex(off_index).to_numpy()==p.offense_team_id.to_numpy())&(teams.reindex(def_index).to_numpy()==p.defense_team_id.to_numpy())
    audit=coverage_audit(p.loc[p.identity_valid],official,fga_tolerance=fga_tolerance)
    unknown=p.loc[~p.identity_valid].groupby('game_id').partialPossessions.sum()
    audit['unverified_identity_partial_possessions']=audit.game_id.map(unknown).fillna(0.)
    audit['invalid_count_pairs']=audit.game_id.map(p.loc[~p.row_counts_valid].groupby('game_id').size()).fillna(0).astype(int)
    p=p.loc[p.identity_valid].copy()
    if p.empty:raise ValueError('No exact PBP-verified attacker/defender pairs')
    p=p.merge(audit[['game_id',*[c for c in audit if c.endswith('_valid')]]],on='game_id',validate='many_to_one')
    price=previous_day_evidence(official,p)
    p=p.merge(price,on=['offense_player_id','game_date'],how='left',validate='many_to_one')
    p=p.merge(lagged_metadata(p,metadata),on=['offense_player_id','season','game_date'],how='left',validate='many_to_one')
    p['fg_points']=2*p.matchupFieldGoalsMade+p.matchupThreePointersMade
    p['fg_conditional_expected']=2*(p.matchupFieldGoalsAttempted-p.matchupThreePointersAttempted)*p.expected_fg2pct+3*p.matchupThreePointersAttempted*p.expected_fg3pct
    pp=p.partialPossessions;priced=p.expected_total_points.notna();assignment=p.assignment_valid&pp.gt(0)&p.identity_valid
    masks={};numerators={};scales={}

    def metric(name, numerator, mask, scale=100.):
        masks[name]=assignment&mask&np.isfinite(numerator)
        numerators[name]=numerator;scales[name]=scale

    metric('matchup_assignment_difficulty',pp*p.expected_total_points,priced)
    metric('matchup_assignment_assist_load100',pp*p.expected_ast,priced)
    metric('matchup_assignment_usage_load100',pp*(p.expected_fga+.44*p.expected_fta+p.expected_tov),priced)
    metric('matchup_assignment_bpm',pp*p.opponent_bpm,p.opponent_bpm.notna(),1.)
    metric('matchup_assignment_obpm',pp*p.opponent_obpm,p.opponent_obpm.notna(),1.)
    metric('matchup_assignment_pos_difficulty',pp*(p.opponent_bpm-p.opponent_bpm_position_mean),p.opponent_bpm_position_mean.notna(),1.)
    metric('matchup_fg_points_saved100',pp*p.expected_fg_points-p.fg_points,p.fg_valid&priced)
    metric('matchup_ft_points_saved100',pp*p.expected_ft_points-p.matchupFreeThrowsMade,p.ft_valid&priced)
    metric('matchup_points_saved100',pp*p.expected_total_points-p.fg_points-p.matchupFreeThrowsMade,p.fg_valid&p.ft_valid&priced)
    metric('matchup_fg_points_allowed100',p.fg_points,p.fg_valid)
    metric('matchup_points_allowed100',p.fg_points+p.matchupFreeThrowsMade,p.fg_valid&p.ft_valid)
    metric('matchup_shot_frequency_saved100',pp*p.expected_fga-p.matchupFieldGoalsAttempted,p.fg_valid&priced)
    metric('matchup_three_frequency_saved100',pp*p.expected_three_fga-p.matchupThreePointersAttempted,p.fg_valid&priced)
    metric('matchup_assists_saved100',pp*p.expected_ast-p.matchupAssists,p.assist_valid&priced)
    metric('matchup_turnovers_above_expected100',p.matchupTurnovers-pp*p.expected_tov,p.tov_valid&priced)
    metric('matchup_bam',p.fg_conditional_expected-p.fg_points,p.fg_valid&priced)
    metric('matchup_blocks100',p.matchupBlocks,p.block_valid)
    metric('matchup_help_blocks100',p.helpBlocks,p.help_block_valid&p.helpBlocks.notna())
    metric('matchup_shooting_fouls100',p.shootingFouls,p.foul_valid)
    metric('matchup_potential_assists_allowed100',p.matchupPotentialAssists,p.potential_assist_valid)
    metric('matchup_ft_frequency_saved100',pp*p.expected_fta-p.matchupFreeThrowsAttempted,p.ft_valid&priced)
    position=p.opponent_position.fillna('').astype(str).str.upper()
    mapped=position.isin(['PG','SG','G','SF','PF','F','C','FC','F-C','C-F','G-F','F-G'])
    for name,choices in [('guard',['PG','SG','G']),('wing',['SF','PF','F','G-F','F-G']),('big',['C','FC','F-C','C-F'])]:
        metric('matchup_'+name+'_share',pp*position.isin(choices),mapped,1.)
    keys=['game_id','game_date','season','season_type','defense_player_id']
    p['assignment_exposure']=pp.where(assignment,0.)
    for name in masks:
        p[name+'__numerator']=numerators[name].where(masks[name],0.)
        p[name+'__denominator']=pp.where(masks[name],0.)
    columns=['assignment_exposure',*[x for name in masks for x in [name+'__numerator',name+'__denominator']]]
    out=p.groupby(keys,as_index=False)[columns].sum()
    price_dates=p.groupby(keys,as_index=False).league_evidence_day.max().rename(columns={'league_evidence_day':'price_last_evidence_day'})
    out=out.merge(price_dates,on=keys,how='left',validate='one_to_one')
    shares=p.loc[assignment].copy()
    total=shares.groupby(['game_id','defense_player_id']).partialPossessions.transform('sum')
    shares['share']=shares.partialPossessions/total
    shape=shares.groupby(['game_id','defense_player_id']).agg(primary=('share','max'),concentration=('share',lambda x:float(np.square(x).sum())))
    out=out.merge(shape,on=['game_id','defense_player_id'],how='left',validate='one_to_one')
    for name,column in [('matchup_primary_share','primary'),('matchup_concentration','concentration')]:
        out[name+'__numerator']=out[column]*out.assignment_exposure
        out[name+'__denominator']=out.assignment_exposure;scales[name]=1.
    actual_def=official.groupby(['game_id','player_id']).def_poss.sum()
    index=pd.MultiIndex.from_frame(out[['game_id','defense_player_id']].rename(columns={'defense_player_id':'player_id'}))
    out['official_def_possessions']=actual_def.reindex(index).to_numpy()
    good=out.official_def_possessions.gt(0)
    out['matchup_coverage__numerator']=out.assignment_exposure.where(good,0.)
    out['matchup_coverage__denominator']=out.official_def_possessions.where(good,0.);scales['matchup_coverage']=1.
    out=out.rename(columns={'defense_player_id':'player_id'})
    # Coverage includes official player-games whose matchup data is absent or
    # unusable. A download outage must not make measured coverage look complete.
    eligible=official.loc[official.season.between(2018,2026)&official.season_type.isin(['rs','ps'])&official.def_poss.gt(0)]
    missing=eligible[['game_id','game_date','season','season_type','player_id','def_poss']].merge(
        out[['game_id','player_id']],on=['game_id','player_id'],how='left',indicator=True)
    missing=missing.loc[missing._merge.eq('left_only')].drop(columns='_merge').rename(columns={'def_poss':'official_def_possessions'})
    if len(missing):
        missing['assignment_exposure']=0.
        for name in scales:
            missing[name+'__numerator']=0.;missing[name+'__denominator']=0.
        missing['matchup_coverage__denominator']=missing.official_def_possessions
        aligned=missing.reindex(columns=out.columns)
        for column in out.select_dtypes(include=['datetime']).columns:
            aligned[column]=pd.to_datetime(aligned[column]).astype(out[column].dtype)
        out=pd.concat([out,aligned],ignore_index=True)
    finite=out.price_last_evidence_day.notna()
    if not pd.to_datetime(out.loc[finite,'price_last_evidence_day']).lt(pd.to_datetime(out.loc[finite,'game_date'])).all():
        raise ValueError('Opponent price provenance reaches game day')
    return out,audit,scales


def snapshots(atoms,cuts,scales,*,half_life=550.,prior_exposure=1000.):
    """Hold recipe fixed; shrink toward the pre-cut TD-weighted population."""
    rows=[];provenance=[]
    for value in cuts:
        cut=pd.Timestamp(value).normalize()
        if cut>=pd.Timestamp('2026-10-01'):raise ValueError('2027-season emission is protected')
        d=atoms.loc[pd.to_datetime(atoms.game_date).lt(cut)].copy()
        if d.empty:continue
        w=np.exp2(-(cut-pd.to_datetime(d.game_date)).dt.total_seconds().to_numpy()/(86400*half_life))
        d['weight']=w;d['td_assignment']=w*d.assignment_exposure
        exposure=d.groupby('player_id').td_assignment.sum()
        result=pd.DataFrame({'player_id':exposure.index,'estimate_date':cut,'matchup_exposure':exposure.to_numpy()})
        for feature in FEATURES:
            if feature not in scales:
                result[feature]=np.nan;continue
            nc,dc=feature+'__numerator',feature+'__denominator'
            weighted=d[['player_id',nc,dc]].copy();weighted[nc]*=w;weighted[dc]*=w
            grouped=weighted.groupby('player_id')[[nc,dc]].sum().reindex(exposure.index)
            population=weighted[nc].sum()/weighted[dc].sum() if weighted[dc].sum()>0 else np.nan
            estimate=scales[feature]*(grouped[nc]+prior_exposure*population)/(grouped[dc]+prior_exposure)
            if feature=='matchup_coverage':
                estimate=grouped[nc]/grouped[dc].where(grouped[dc].gt(0))
            estimate=estimate.where(grouped[dc].gt(0));result[feature]=estimate.to_numpy()
            result[feature+'__exposure']=grouped[dc].to_numpy()
            observed=d.loc[d[dc].gt(0)].groupby('player_id').game_date.max()
            last=observed.reindex(exposure.index)
            for player,denom,date in zip(exposure.index,grouped[dc],last):
                provenance.append(dict(player_id=int(player),estimate_date=cut,feature=feature,
                    last_observation_date=date,parameter_fit_end=d.game_date.max(),exposure=float(denom),
                    source_coverage=float(min(denom/exposure.loc[player],1.)) if exposure.loc[player]>0 else 0.,
                    recipe=('rawTDassignment/officialTDdefposs;noK1000;missinglogsdenominatorretained'
                            if feature=='matchup_coverage' else 'TD550;K1000pre-cutpopulation;attackerpricesstrictlyearliergame-days')))
        rows.append(result)
    if not rows:raise ValueError('No matched evidence before requested cuts')
    return pd.concat(rows,ignore_index=True),pd.DataFrame(provenance)


def build(archive=ARCHIVE,out=OUT,cuts=CUTS,*,allow_incomplete=False,metadata_path=None,fga_tolerance=.15):
    archive,out=Path(archive),Path(out);out.mkdir(parents=True,exist_ok=True)
    census=pd.read_parquet(archive/'game_census.parquet');census.game_id=ids(census.game_id)
    census=census.loc[census.season_type.isin(['rs','ps'])&census.season.between(2018,2026)].copy()
    if census.game_id.duplicated().any():raise ValueError('Ambiguous game census')
    chunks=[];sources={str(archive/'game_census.parquet'):sha(archive/'game_census.parquet')};missing=[];empty=[]
    for game in census.itertuples():
        path=archive/'pairs'/f'{game.game_id}.parquet';meta=archive/'metadata'/f'{game.game_id}.json'
        if not meta.exists():missing.append(game.game_id);continue
        m=json.loads(meta.read_text())
        if m.get('status')=='empty':empty.append(game.game_id);continue
        if m.get('status')!='ok' or not path.exists():missing.append(game.game_id);continue
        raw=archive/'raw'/f'{game.game_id}.json'
        if not raw.exists() or sha(raw)!=m['sha256']:raise ValueError('Native source changed '+game.game_id)
        d=pd.read_parquet(path)
        if not ids(d.game_id).eq(game.game_id).all() or not pd.to_datetime(d.game_date).eq(pd.Timestamp(game.game_date)).all():raise ValueError('Pair/census identity conflict')
        chunks.append(d);sources[str(path)]=sha(path);sources[str(meta)]=sha(meta)
    if missing and not allow_incomplete:raise ValueError(f'{len(missing)} RS/PS games not downloaded; explicit --allow-incomplete only for smoke tests')
    if not chunks:raise ValueError('No native per-game matchup rows')
    pairs=pd.concat(chunks,ignore_index=True)
    lines_path=ROOT/'results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet'
    lines=pd.read_parquet(lines_path);sources[str(lines_path)]=sha(lines_path)
    metadata=None
    default_metadata=ROOT/'external/matchup_bpm_v1/player_seasons.parquet'
    if metadata_path is None and default_metadata.exists():metadata_path=default_metadata
    if metadata_path is not None:
        metadata=pd.read_parquet(metadata_path) if str(metadata_path).endswith('.parquet') else pd.read_csv(metadata_path)
        sources[str(metadata_path)]=sha(metadata_path)
    context_path=ROOT/'results/estimated_skills/monthly_vintage_v1/game_context.parquet'
    context=pd.read_parquet(context_path);sources[str(context_path)]=sha(context_path)
    atoms,audit,scales=defender_game_atoms(pairs,lines,metadata=metadata,game_context=context,fga_tolerance=fga_tolerance)
    features,provenance=snapshots(atoms,cuts,scales)
    for name,frame in [('defender_game_atoms.parquet',atoms),('game_coverage.parquet',audit),
                       ('features.parquet',features),('feature_provenance.parquet',provenance)]:
        frame.to_parquet(out/name,index=False)
    artifact_hashes={name:sha(out/name) for name in ['defender_game_atoms.parquet','game_coverage.parquet','features.parquet','feature_provenance.parquet']}
    unavailable=[f for f in FEATURES if features[f].isna().all()]
    manifest=dict(status='partial_download_smoke_only' if missing else 'research_candidate_not_admitted',
        feature_groups=FEATURE_GROUPS,unavailable_features=unavailable,cuts=list(map(str,cuts)),
        source_hashes=sources,code_sha256=sha(__file__),artifact_hashes=artifact_hashes,
        expected_rs_ps_games=len(census),downloaded_games=int(pairs.game_id.nunique()),empty_games=empty,
        missing_games=missing,rows=len(features),provenance_rows=len(provenance),
        ordinary_phases=['rs','ps'],protected_outcomes=0,
        direction='nativeouterattacker_nesteddefender; verifiedagainstboxtotals',
        opponent_price='TD550rawowncounts;fixed2013anchor;entirecurrentdayexcluded;K1000possessions/K100shots;prefixleaguefallback',
        stabilization='fixedTD550andK1000assignmentpossessions;pre-cutpopulationfallback;missingstaysmissing;matchup_coverageexceptionisrawratioandhasnopseudocount',
        feature_scales=scales,
        source_coverage_definition='validlaneTDpartialpossessions/allassignmentTDpartialpossessions;notactioncompleteness;on-courtcoverageisseparateunshrunkmatchup_coverage',
        fg_coverage_tolerance=fga_tolerance,tight_fg_coverage_tolerance=.05,
        bpm_knowledge_floor='2020-03-01;modernBPM2.0backhistoryunavailablebeforepublication',
        limitations=['Assignmentandtrackedactionsarenonexclusiveobservations,notcausalcredit',
            'Historicalsourcepublicationvintagesuncertified;downloadtodayisnotas-ofavailability',
            'BAMisconditionalFGconversionatsourceFGA,notalegacyBAMreplication',
            'Potentialassistzeroswithoutpositivegamecountsareunavailable',
            'Opponentsarepricedfromrawowncountshere;thisisnotcompleteEPMorVPMinput'])
    write_json(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','expected_rs_ps_games','downloaded_games','rows','unavailable_features']}),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ARCHIVE);parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--cuts',nargs='+',default=CUTS);parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--metadata',type=Path);parser.add_argument('--fga-tolerance',type=float,default=.15)
    args=parser.parse_args();build(args.archive,args.out,args.cuts,allow_incomplete=args.allow_incomplete,
                                  metadata_path=args.metadata,fga_tolerance=args.fga_tolerance)

#!/usr/bin/env python3
"""Pair opponent possession-start PPP with own continuation per missed FGA.

Own continuation includes all later own points in the same parsed possession,
with every eligible missed FGA in the denominator (including unrecovered misses).
These overlapping action-level values must not be summed across repeated misses.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from build_possession_start_report import (
    _base_processing, infer_game_team_map, prepare_standard_possession_df,
    RIM_ACTION_TYPES, MISS_DISTANCE_RE, RESULTS_DIR, season_raw_path,
)

OUT = RESULTS_DIR / 'possession_start_report'

def build_miss_rows(year=26):
    raw = _base_processing(season_raw_path(year, 'RS'))
    raw = raw.merge(infer_game_team_map(raw), on='game_id', how='left')
    df, _ = prepare_standard_possession_df(raw, 'POSSESSION_START_CONTINUATION')
    # Exactly the same eligible possession universe as the opponent PPP report.
    detail = pd.read_parquet(OUT / f'possession_start_possessions_{year}_rs.parquet')
    keys = ['game_id', 'poss_group']
    df = df.merge(detail[keys], on=keys, how='inner', validate='many_to_one')
    df['own_points'] = np.where(df.poss_offense.eq('Home'), df.Home_Action_Score, df.Away_Action_Score)
    group = df.groupby(keys, sort=False)
    df['remaining_own_points'] = group.own_points.transform('sum') - group.own_points.cumsum()
    own_text = np.where(df.poss_offense.eq('Home'), df.home_description, df.visitor_description)
    df['own_text'] = pd.Series(own_text, index=df.index).fillna('').astype(str)
    misses = df.loc[df.event_type.eq('MISS') & df.own_text.str.contains(r'\bMISS\b', case=False, regex=True)].copy()
    text = misses.home_description.fillna('') + ' ' + misses.visitor_description.fillna('')
    three = text.str.contains('3PT', case=False, regex=False)
    distance = pd.to_numeric(text.str.extract(MISS_DISTANCE_RE, expand=False), errors='coerce')
    rim = ~three & (pd.to_numeric(misses.event_action_type, errors='coerce').isin(RIM_ACTION_TYPES) | text.str.contains(r'layup|dunk| tip ', case=False, regex=True) | distance.le(4))
    block = text.str.contains('BLOCK', case=False, regex=False)
    misses['At Rim Block'] = rim & block
    misses['At Rim Miss (Unblocked)'] = rim & ~block
    misses['Missed 2PT'] = ~three
    misses['Missed FG'] = True
    misses['Short Mid Range Miss'] = ~three & ~rim & distance.between(5,14)
    misses['Long Mid Range Miss'] = ~three & ~rim & distance.ge(15)
    misses['Missed 3PT'] = three
    assert misses.remaining_own_points.ge(-1e-9).all()
    assert not (misses['At Rim Block'] & misses['At Rim Miss (Unblocked)']).any()
    columns = ['At Rim Block','At Rim Miss (Unblocked)','Missed 2PT','Missed FG','Short Mid Range Miss','Long Mid Range Miss','Missed 3PT']
    rows = []
    for category in columns:
        sub = misses.loc[misses[category]]
        rows.append(dict(category=category, all_misses=len(sub), own_continuation_points=sub.remaining_own_points.sum(), own_points_per_miss=sub.remaining_own_points.mean()))
    result = pd.DataFrame(rows)
    misses[['game_id','poss_group','event_num','own_text','remaining_own_points',*columns]].to_parquet(OUT / f'own_miss_continuation_detail_{year}_rs.parquet',index=False)
    result.to_csv(OUT / f'own_miss_continuation_{year}_rs.csv',index=False)
    return result

def render(year=26):
    league = pd.read_csv(OUT / f'possession_start_league_{year}_rs.csv')
    continuation = pd.read_csv(OUT / f'own_miss_continuation_{year}_rs.csv')
    hidden = {'Corner 3 Miss','Above The Break 3 Miss','Made FG'}
    d = league.loc[~league.category.isin(hidden)].merge(continuation,on='category',how='left').sort_values('ortg',ascending=False).reset_index(drop=True)
    d.to_csv(OUT / f'possession_start_with_continuation_{year}_rs.csv',index=False)
    bg = '#faf7ec'
    fig, ax = plt.subplots(figsize=(15,9.5),dpi=180,facecolor=bg)
    ax.set_facecolor(bg)
    v = d.delta_vs_league/100
    ax.barh(range(len(d)),v,color=np.where(v>=0,'#198e83','#ca6471'),height=.62,zorder=3)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.category,fontsize=11); ax.invert_yaxis()
    for i,r in d.iterrows():
        x=r.delta_vs_league/100
        ax.text(x+(.004 if x>=0 else -.004),i,f'{x:+.3f}',ha='left' if x>=0 else 'right',va='center',fontsize=11,fontweight='bold')
        for pos,label,color,bold in [
            (1.03,f'{r.ortg/100:.3f}','#222222',True),
            (1.21,f'{r.possessions:,.0f}','#666666',False),
            (1.46,f'{r.own_points_per_miss:.3f}' if pd.notna(r.own_points_per_miss) else '—','#985f19',True),
            (1.66,f'{r.all_misses:,.0f}' if pd.notna(r.all_misses) else '—','#666666',False)]:
            ax.text(pos,i,label,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=11,fontweight='bold' if bold else 'normal',color=color)
    for x,t in [(1.03,'OPP. PPP'),(1.21,'POSS.'),(1.46,'OWN PTS / MISS'),(1.66,'ALL MISSES')]:
        ax.text(x,1.035,t,transform=ax.transAxes,ha='right',fontsize=10,fontweight='bold',color='#555555')
    ax.axvline(0,color='#333333',linewidth=1.3)
    ax.set_xlim(-.10,.25)
    ax.set_xticks([-.10,-.05,0,.05,.10,.15,.20])
    ax.xaxis.set_major_formatter(lambda v,_:'Avg' if abs(v)<1e-8 else f'{v:+.2f}')
    ax.set_xlabel('Opponent PPP above / below league average',fontsize=12,labelpad=14)
    ax.tick_params(length=0,pad=8,labelsize=11);ax.set_axisbelow(True);ax.xaxis.grid(True,color='#dedbd2')
    for s in ax.spines.values():s.set_visible(False)
    fig.text(.03,.955,'Misses have two branches: recover it or defend the next possession',fontsize=21,fontweight='bold')
    fig.text(.03,.912,f'2025–26 NBA regular season • Opponent league average: {d.league_avg_ortg.iloc[0]/100:.3f} PPP',fontsize=12,color='#555555')
    fig.text(.03,.065,'Bars / OPP. PPP: full opponent possession after a defensive rebound (for misses).',fontsize=11,color='#555555')
    fig.text(.03,.041,'OWN PTS / MISS: all later own points in the same possession, averaged over all missed FGAs, including zero continuations.',fontsize=11,color='#985f19')
    fig.text(.03,.017,'Different denominators: do not add or subtract these columns. — = no FGA-miss estimate. Categories and repeated misses overlap.',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.225,right=.665,top=.83,bottom=.16)
    for ext in ['png','svg']: fig.savefig(OUT / f'possession_start_with_continuation_{year}_rs.{ext}')
    plt.close(fig)

if __name__ == '__main__':
    build_miss_rows()
    render()

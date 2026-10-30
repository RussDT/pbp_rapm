"""Matched five-year box/Net target calibration; no game-value promotion.

Each window is fitted separately: overlapping windows are sensitivity checks,
not independent validation folds. Missing season/type OFFD coverage excludes
the entire player-window, never silently a portion of its counting events.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
try:
    from . import build_player_game_box_net_points as b
except ImportError:
    import build_player_game_box_net_points as b

OUT=b.ROOT/'results/player_game_box_net_points/five_year_audit'
FEATURES=json.loads(b.CONFIG.read_text())['features']
FEATURES['dTOV']=['STL','OFFD']

def attach_offd(box, source):
    required=['EntityId','Charge Fouls Drawn','Offensive Fouls Drawn']
    if not set(required).issubset(source):
        raise ValueError('Missing OFFD source columns')
    if source.EntityId.duplicated().any():
        raise ValueError('Duplicate OFFD player totals')
    # Provider sparse event columns omit zeros within a present player row.
    source=source.copy()
    source['OFFD']=source[required[1:]].fillna(0).sum(axis=1)
    out=box.merge(source[['EntityId','OFFD']+(['GamesPlayed'] if 'expected_games' in box else [])],left_on='PLAYER_ID',right_on='EntityId',how='left',validate='one_to_one')
    out['offd_covered']=out.EntityId.notna()
    if 'expected_games' in out:
        out['offd_covered'] &= out.expected_games.eq(out.GamesPlayed)
    return out.drop(columns='EntityId')

def fit_window(frame,factor,features):
    side='off_poss' if factor.startswith('o') else 'def_poss'
    frame=frame.loc[frame[side]>=250]
    x=100*frame[features].div(frame[side],axis=0)
    # Count exposure weights, no second regularization of already-ridged targets.
    model=LinearRegression().fit(x,frame[factor],sample_weight=frame[side])
    return dict(factor=factor,features=features,coefficients=model.coef_.tolist(),intercept=float(model.intercept_),alpha=0.,training_rows=len(frame),weight='side possessions',target_penalties='2000 offense / 4000 defense',status='five-year season-impact mapping; not established event valuation')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames=[];coverage=[];sources=[]
    for year in range(2014,2023):
        for kind in ['rs','ps']:
            box,path=b.read_games(year,kind);sources.append(path)
            totals=box.groupby('PLAYER_ID')[b.BOX+['MISSED_FG']].sum()
            totals['expected_games']=box.groupby('PLAYER_ID').size()
            totals=totals.reset_index()
            path=b.ROOT.parent/'player_sheets/gen_totals'/f'{year}{"_ps" if kind=="ps" else ""}_pbp.csv'
            source=pd.read_csv(path);sources.append(path)
            totals=attach_offd(totals,source);totals['year']=year
            coverage.append(dict(year=year,kind=kind,box_players=len(totals),offd_source_players=len(source),missing_offd_players=int((~totals.offd_covered).sum())))
            frames.append(totals)
    annual=pd.concat(frames,ignore_index=True)
    models={};panels=[];coefs=[]
    for end in range(2018,2023):
        start=end-4; window=annual[annual.year.between(start,end)]
        complete=window.groupby('PLAYER_ID').offd_covered.all()
        counts=window.groupby('PLAYER_ID')[b.BOX+['MISSED_FG','OFFD']].sum()
        counts=counts.loc[complete].rename_axis('player_id').reset_index()
        path=b.ROOT/'master_results/net_decomp_rolling'/f'weighted_factors_decomp_net_{start%100:02d}_{end%100:02d}_all_rb_se_a2000_4000.csv'
        targets=b.net_targets(path);sources.append(path)
        panel=counts.merge(targets,on='player_id',validate='one_to_one');panel['window_start']=start;panel['window_end']=end
        panels.append(panel)
        key=f'{start}-{end}';models[key]={}
        for factor,features in FEATURES.items():
            model=fit_window(panel,factor,features);models[key][factor]=model
            for feature,coef in [('INTERCEPT',model['intercept']),*zip(features,model['coefficients'])]:
                coefs.append(dict(window=key,factor=factor,feature=feature,coefficient=coef,training_rows=model['training_rows']))
    pd.concat(panels).to_parquet(OUT/'training_panel.parquet',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(coefs).to_csv(OUT/'coefficients.csv',index=False)
    (OUT/'models.json').write_text(json.dumps(models,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(status='audit only; current game board rejected pending recalibration and game OFFD source',windows=list(models),features=FEATURES,offd_definition='Charge Fouls Drawn + Offensive Fouls Drawn (disjoint provider categories)',missing_rule='exclude entire player-window when any played season/type lacks source row',inputs=[str(p.resolve()) for p in sources]),indent=2)+'\n')
    print(pd.DataFrame(coefs).query("factor == 'dTOV'").to_string(index=False))

if __name__=='__main__':main()

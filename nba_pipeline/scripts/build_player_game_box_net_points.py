#!/usr/bin/env python3
"""Traditional-box coefficient audit and six-factor completed-game credit.

Reuses audited team targets/exposures from build_player_game_net_points; replaces
its historical-player and manually priced action anchors entirely. All scored
terms are displayed in raw event units, including intercept and reconciliation.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from .build_player_game_net_points import FACTORS, project_credit
    from .player_game_playtype import season_difficulty, game_difficulty
except ImportError:
    from build_player_game_net_points import FACTORS, project_credit
    from player_game_playtype import season_difficulty, game_difficulty

ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT.parent/'player_sheets'/'game_report'/'year_files'
CONFIG=ROOT/'config'/'player_game_box_net_points.json'
BOX=['MIN','PTS','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','TOV','STL','BLK','PF']
SOURCE_FIELDS=['PLAYER_ID','PLAYER_NAME','TEAM_ABBREVIATION','date','POSS',*BOX]


def read_games(year, kind):
    path=LOGS/f'{year}{"ps" if kind=="ps" else ""}_games.csv'
    d=pd.read_csv(path,usecols=lambda c:c in SOURCE_FIELDS)
    missing=set(SOURCE_FIELDS)-set(d.columns)
    if missing:
        raise ValueError(f'Missing game-box columns in {path}: {missing}')
    d=d.loc[pd.to_numeric(d.MIN,errors='coerce').gt(0)].copy()
    d['game_date']=pd.to_datetime(d.date.astype(str),format='%Y%m%d').dt.strftime('%Y-%m-%d')
    if d.duplicated(['PLAYER_ID','game_date']).any():
        raise ValueError(f'Duplicate player/date box rows: {path}')
    for c in BOX+['POSS']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    if d[BOX].isna().any().any() or (d[BOX]<0).any().any():
        raise ValueError(f'Invalid traditional box values: {path}')
    d['MISSED_FG']=d.FGA-d.FGM
    if (d.MISSED_FG<0).any() or not np.allclose(d.PTS,2*d.FGM+d.FG3M+d.FTM):
        raise ValueError(f'Invalid shooting identity: {path}')
    return d,path


def net_targets(path):
    d=pd.read_csv(path)
    if not d.DECOMP_PRICING.eq('net_continuation_kappa').all() or d.player_id.duplicated().any():
        raise ValueError(f'Invalid Net DECOMP target: {path}')
    for s in ['o','d']:
        d[s+'TS']=d[[s+'DECOMP_'+a for a in ['RIM_FREQ','RIM_FG','MID_FG','THREE_FREQ','THREE_FG']]].sum(axis=1)+d[s+'DECOMP_FT']
        d[s+'TOV']=d[s+'DECOMP_TOV_VALUE']
        d[s+'SC']=d[s+'DECOMP_SC']
    return d[['player_id','off_poss','def_poss',*FACTORS]]


def training_panel(config):
    parts=[];sources=[];coverage=[]
    for year in config['development_years']+config['historical_report_years']:
        games=[]
        for kind in ['rs','ps']:
            frame,path=read_games(year,kind);games.append(frame);sources.append(path)
        g=pd.concat(games,ignore_index=True)
        if g.duplicated(['PLAYER_ID','game_date']).any():
            raise ValueError(f'RS/PS box overlap in {year}')
        totals=g.groupby('PLAYER_ID')[[*BOX,'MISSED_FG']].sum().reset_index().rename(columns={'PLAYER_ID':'player_id'})
        path=ROOT/'master_results'/'net_decomp_rolling'/f'weighted_factors_decomp_net_{year%100:02d}_all_rb_se_a2000_4000.csv'
        target=net_targets(path);sources.append(path)
        merged=totals.merge(target,on='player_id',how='inner',validate='one_to_one')
        difficulty,prices,pbp_paths=season_difficulty(year)
        sources.extend(pbp_paths)
        merged=merged.merge(difficulty,on='player_id',how='left',validate='one_to_one')
        if merged.loc[merged.FGA.gt(0),'pbp_difficulty_pp'].isna().any():
            raise ValueError(f'Missing season PBP difficulty: {year}')
        merged['PBP_ROLE_POINTS']=2*(merged.FGA+0.44*merged.FTA)*merged.pbp_difficulty_pp.fillna(0)/100
        merged['season']=year
        coverage.append(dict(season=year,box_players=len(totals),target_players=len(target),joined_players=len(merged),box_games=len(g)))
        parts.append(merged)
    return pd.concat(parts,ignore_index=True),sources,pd.DataFrame(coverage)


def fit_linear(frame,factor,features,alpha):
    side='off_poss' if factor.startswith('o') else 'def_poss'
    x=100*frame[features].to_numpy(float)/frame[side].to_numpy(float)[:,None]
    y=frame[factor].to_numpy(float)
    w=np.sqrt(frame[side].to_numpy(float));w/=w.mean()
    scaler=StandardScaler().fit(x,sample_weight=w)
    model=Ridge(alpha=alpha).fit(scaler.transform(x),y,sample_weight=w)
    coef=model.coef_/scaler.scale_
    intercept=float(model.intercept_-np.dot(coef,scaler.mean_))
    np.testing.assert_allclose(intercept+x@coef,model.predict(scaler.transform(x)),rtol=1e-10,atol=1e-10)
    return dict(factor=factor,features=features,coefficients=coef.tolist(),intercept=intercept,alpha=float(alpha),train_first=int(frame.season.min()),train_last=int(frame.season.max()),training_rows=len(frame))


def predict_rate(frame,model):
    side='off_poss' if model['factor'].startswith('o') else 'def_poss'
    x=100*frame[model['features']].to_numpy(float)/frame[side].to_numpy(float)[:,None]
    return model['intercept']+x@np.array(model['coefficients'])


def fit_models(panel,config):
    models={};grids=[];reports=[];coefficients=[]
    for factor,features in config['features'].items():
        side='off_poss' if factor.startswith('o') else 'def_poss'
        eligible=panel.loc[panel[side].ge(config['minimum_season_possessions'])]
        dev=eligible.loc[eligible.season.isin(config['development_years'])]
        years=config['development_years'][config['minimum_training_seasons']:]
        losses=[]
        for alpha in config['alphas']:
            errors=[]
            for year in years:
                train=dev.loc[dev.season.lt(year)];test=dev.loc[dev.season.eq(year)]
                model=fit_linear(train,factor,features,alpha)
                mse=float(np.average((test[factor]-predict_rate(test,model))**2,weights=np.sqrt(test[side])))
                errors.append(mse)
                grids.append(dict(factor=factor,alpha=alpha,test_season=year,mse=mse))
            losses.append(errors)
        losses=np.array(losses);best=int(np.argmin(losses.mean(axis=1)))
        delta=losses-losses[best]
        se=delta.std(axis=1,ddof=1)/np.sqrt(len(years))
        accepted=np.flatnonzero(delta.mean(axis=1)<=se+1e-12)
        selected=int(accepted[-1]);alpha=config['alphas'][selected]
        model=fit_linear(dev,factor,features,alpha);models[factor]=model
        for year in config['historical_report_years']:
            test=eligible.loc[eligible.season.eq(year)]
            pred=predict_rate(test,model);w=np.sqrt(test[side]); y=test[factor]
            mse=float(np.average((y-pred)**2,weights=w))
            zero=float(np.average(y**2,weights=w))
            reports.append(dict(factor=factor,season=year,rows=len(test),rmse=float(np.sqrt(mse)),zero_rmse=float(np.sqrt(zero)),mse_gain_vs_zero=zero-mse,status='historical report only; season-target transport, not game-credit validation'))
        coefficients.append(dict(factor=factor,feature='INTERCEPT',coefficient=model['intercept'],units='points per 100 side possessions',alpha=alpha))
        for feature,coef in zip(features,model['coefficients']):
            coefficients.append(dict(factor=factor,feature=feature,coefficient=coef,units='game points per event',alpha=alpha))
    return models,pd.DataFrame(coefficients),pd.DataFrame(grids),pd.DataFrame(reports)


def score_terms(row,model,exposure):
    terms=[dict(feature='INTERCEPT',raw_value=exposure/100,coefficient=model['intercept'],points=model['intercept']*exposure/100,calculation='intercept × side possessions / 100')]
    for feature,coef in zip(model['features'],model['coefficients']):
        value=float(row[feature])
        if not np.isfinite(value):
            raise ValueError(f'Missing game feature {feature}')
        terms.append(dict(feature=feature,raw_value=value,coefficient=coef,points=value*coef,calculation='box event count × fitted coefficient'))
    return terms


def score_games(base,team_targets,game_boxes,models):
    keys=['game_date','player_id']
    boxes=game_boxes.rename(columns={'PLAYER_ID':'player_id','PLAYER_NAME':'box_name','TEAM_ABBREVIATION':'box_team','POSS':'box_possessions'})
    joined=base.merge(boxes,on=keys,how='left',validate='one_to_one',indicator=True,suffixes=('','_box'))
    if not joined._merge.eq('both').all():
        raise ValueError('Missing game box score for audited player exposure')
    if not joined.team.eq(joined.box_team).all():
        raise ValueError('Game box team differs from raw team')
    output=[];allterms=[];reconciliation=[]
    for (gid,team_name),group in joined.groupby(['game_id','team'],sort=False):
        target=team_targets.loc[team_targets.game_id.eq(gid)&team_targets.team.eq(team_name)]
        if len(target)!=1:
            raise ValueError('Team target must be unique')
        target=target.iloc[0]
        if not np.isclose(group.PTS.sum(),target.points):
            raise ValueError(f'Player boxes do not sum to team score: {gid}/{team_name}')
        records=[]
        for _,row in group.iterrows():
            record={c:row[c] for c in ['game_id','game_date','team','opponent','player_id','off_possessions','def_possessions','box_possessions',*BOX,'MISSED_FG']}
            record['player_name']=row.box_name
            for c in row.index:
                if c.startswith('pbp_') or c in ['TSA','TS_pct','rTS_pp','PBP_zTS_proxy','PBP_ROLE_POINTS','league_TS_pct']:
                    record[c]=row[c]
            for factor,model in models.items():
                exposure=float(row.off_possessions if factor.startswith('o') else row.def_possessions)
                terms=score_terms(row,model,exposure)
                record[factor+'_box_points']=sum(term['points'] for term in terms)
                for term in terms:
                    allterms.append(dict(game_id=gid,team=team_name,player_id=int(row.player_id),player_name=row.box_name,factor=factor,**term))
            records.append(record)
        scored=pd.DataFrame(records)
        for factor in FACTORS:
            exposure=scored.off_possessions if factor.startswith('o') else scored.def_possessions
            box=scored[factor+'_box_points'].to_numpy()
            final,adjustment=project_credit(box,np.zeros(len(box)),exposure,float(target[factor]))
            scored[factor]=final;scored[factor+'_team_adjustment']=adjustment
            for i,row in scored.iterrows():
                allterms.append(dict(game_id=gid,team=team_name,player_id=int(row.player_id),player_name=row.player_name,factor=factor,feature='TEAM_ADJUSTMENT',raw_value=float(exposure.iloc[i]/exposure.sum()),coefficient=float(target[factor]-box.sum()),points=float(adjustment[i]),calculation='side exposure share × (team factor target − team box prior)'))
            reconciliation.append(dict(game_id=gid,team=team_name,factor=factor,box_prior_sum=float(box.sum()),team_target=float(target[factor]),adjustment=float(adjustment.sum()),final_sum=float(final.sum()),gap=float(final.sum()-target[factor])))
        scored['box_net_points']=scored[[f+'_box_points' for f in FACTORS]].sum(axis=1)
        scored['team_adjustment']=scored[[f+'_team_adjustment' for f in FACTORS]].sum(axis=1)
        scored['game_net_points']=scored[FACTORS].sum(axis=1)
        output.append(scored)
    players=pd.concat(output,ignore_index=True);terms=pd.DataFrame(allterms);rec=pd.DataFrame(reconciliation)
    np.testing.assert_allclose(players.box_net_points+players.team_adjustment,players.game_net_points,atol=1e-10)
    return players,terms,rec


def render_review(players,teams,terms,coefficients,reports,out,buckets,candidate_coefficients):
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Game Net Points · Box-score calculation</title><style>body{font:15px system-ui;margin:0;background:#f7f7f3;color:#17212b}main{max-width:1500px;margin:auto;padding:30px}h1{font-size:32px}h2{margin-top:38px}p{max-width:1100px;line-height:1.6}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap;margin:14px 0 25px}td,th{padding:8px 10px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}th{background:#eeefe7}summary{cursor:pointer;padding:13px;background:white;border-bottom:1px solid #ddd}details{margin:8px 0}details>div{padding:0 15px}code{background:#eaece1;padding:3px}a{color:#165c89}.notice{background:#fff1cd;padding:15px;border-left:4px solid #bc8622}</style><main><h1>Six-factor Game Net Points</h1><p><b>Traditional box → fitted factor coefficients → game box prior → explicit team adjustment.</b> No player-history rating, observed tracking, manually priced rebound/steal add-on, or duplicated shooting-residual term enters this version.</p><p class="notice">Research adaptation. Coefficients were newly fitted to 2014–2022 season Net DECOMP targets using the earlier box-to-factor ridge approach. They are not the old gross/native-factor coefficients and are not validated causal values for individual game events. Opponent 3P variance and unequal-possession accounting remain explicit. This is a source-audited game selection; full-season coverage is not established. See the manifest for selected game IDs.</p><p><a href="coefficients.csv">Coefficients</a> · <a href="feature_terms.csv">Every calculation term</a> · <a href="players.csv">Game boxes and values</a> · <a href="reconciliation.csv">Team reconciliation</a> · <a href="manifest.json">Model provenance</a></p>''']
    def table(frame,precision=3):
        return '<div class="scroll">'+frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:,.{precision}f}')+'</div>'
    audit_path=out/'five_year_audit'/'coefficients.csv'
    parts.append('<p class="notice"><b>Game-credit calibration is not approved.</b> The original single-season coefficient mapping understates the steal association and omits offensive fouls drawn. Its player totals below are retained only for calculation inspection. Do not treat them as finalized performance credit.</p>')
    if audit_path.exists():
        audit=pd.read_csv(audit_path)
        comparison=coefficients[['factor','feature','coefficient']].rename(columns={'coefficient':'old_single_season'})
        latest=audit.loc[audit.window.eq('2018-2022'),['factor','feature','coefficient']].rename(columns={'coefficient':'matched_2018_2022'})
        parts.append('<h2>Corrected five-year coefficient audit</h2><p>Matched 2018–2022 RS+PS counting totals and five-year Net targets; possession-weighted least squares with no second ridge. Every played season/type must have OFFD source coverage. OFFD includes charges and other offensive fouls drawn. These still map to regularized player-impact targets; they are not validated event prices and have not replaced the game totals.</p>')
        parts.append(table(comparison.merge(latest,on=['factor','feature'],how='outer'),5))
        parts.append('<p><a href="five_year_audit/coefficients.csv">All five-year windows</a> · <a href="five_year_audit/coverage.csv">OFFD coverage audit</a> · <a href="five_year_audit/manifest.json">Five-year method</a></p>')
    parts.append('<details><summary>All fitted coefficients and feature definitions</summary><div><p>For each factor: <code>box points = intercept × side possessions / 100 + Σ(coefficient × event count)</code>. MISSED_FG = FGA − FGM. Since a season feature is events per 100 and the output is points per 100, converting back to a game cancels the exposure denominator for each event term. These are partial regression coefficients, not universal prices for events.</p>'+table(coefficients,6)+'<p>Headline oTS: PTS, FGA, FTA, FG3A, AST. oTOV: TOV, FGA, FTA, AST. oSC: OREB, missed FG. dTS: BLK, PF. dTOV: STL. dSC: DREB, BLK. Estimated potential assists are not used in this version.</p></div></details>')
    parts.append('<details><summary>PBP-role candidate coefficients — not used in headline</summary><div><p>The PBP difficulty term is shown as a candidate. Its compact season-target fit has a small negative coefficient; this does not support installing a positive difficulty credit by assumption. The existing future-oTS study used a different target and richer creation controls.</p>'+table(candidate_coefficients,6)+'</div></details>')
    parts.append('<details><summary>Historical season-target check — not game attribution validation</summary><div>'+table(reports)+'</div></details>')
    for gid,group in players.groupby('game_id',sort=True):
        parts.append('<h2>'+html.escape(str(group.game_date.iloc[0])+' · '+' / '.join(group.team.unique())+' · '+str(gid))+'</h2>')
        target=teams.loc[teams.game_id.eq(gid)]
        parts.append('<h3>1. Actual box score</h3>'+table(group[['player_name','team',*BOX,'off_possessions','def_possessions','box_possessions']],1))
        parts.append('<h3>PBP playtype difficulty and zTS proxy</h3><p>Six frozen description buckets, priced from the preceding regular season. Game difficulty is blended toward the player’s preceding-season mix with an explicit, provisional prior weight shown below. Actual TS is not shrunk. Positive difficulty means harder shots. PBP zTS proxy = TS% − preceding-season league TS% + stabilized difficulty. This is not the production Synergy zTS. PBP_ROLE_POINTS = 2 × TSA × difficulty / 100; it is an explicit candidate feature, not part of the headline oTS yet. Season training uses full-season mix; game shrinkage is an unvalidated adaptation.</p>'+table(group[['player_name','FGA','TSA','TS_pct','league_TS_pct','rTS_pp','pbp_raw_difficulty_pp','pbp_prior_difficulty_pp','pbp_prior_season','pbp_prior_pseudo_fga','pbp_game_weight','pbp_stabilized_difficulty_pp','PBP_zTS_proxy','PBP_ROLE_POINTS']]))
        parts.append('<h3>2. Six factors and adjustment</h3><p>Box prior is the coefficient calculation. Team adjustment allocates the remaining measured factor result by side possessions. Large adjustments signal that the box model leaves much of the team performance unexplained.</p>'+table(group[['player_name','team',*FACTORS,'box_net_points','team_adjustment','game_net_points']]))
        parts.append('<h3>3. Team accounting</h3>'+table(target[['team','points','opponent_points','actual_margin','game_net_points','opponent_3p_variance','unequal_possessions_baseline','closure_gap']]))
        parts.append('<p><code>sum(player game net points) + unequal-possession baseline − opponent 3P variance = actual margin</code></p><h3>4. Expand a player to inspect every multiplication</h3>')
        for row in group.itertuples():
            parts.append('<details><summary>'+html.escape(row.player_name+' · '+row.team)+f' — box {row.box_net_points:+.2f}, adjustment {row.team_adjustment:+.2f}, final {row.game_net_points:+.2f}</summary><div>')
            ledger=terms.loc[terms.game_id.eq(gid)&terms.player_id.eq(row.player_id)]
            parts.append(table(ledger[['factor','feature','raw_value','coefficient','points','calculation']],6))
            parts.append('<p>Shot-mix calculation (zero-count buckets are included):</p>'+table(buckets.loc[buckets.game_id.eq(gid)&buckets.player_id.eq(row.player_id)].drop(columns=['game_id','player_id']),5))
            parts.append('</div></details>')
    parts.append('</main></html>');(out/'review.html').write_text('\n'.join(parts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=ROOT/'results'/'player_game_net_points')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results'/'player_game_box_net_points')
    parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();config=json.loads(args.config.read_text());out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    panel,sources,coverage=training_panel(config)
    models,coefficients,grid,reports=fit_models(panel,config)
    candidate_config={**config,'features':config['candidate_features']}
    candidate_models,candidate_coefficients,candidate_grid,candidate_reports=fit_models(panel,candidate_config)
    candidate_coefficients.to_csv(out/'playtype_candidate_coefficients.csv',index=False)
    candidate_reports.to_csv(out/'playtype_candidate_historical_report.csv',index=False)
    (out/'playtype_candidate_models.json').write_text(json.dumps(candidate_models,indent=2)+'\n')
    base=pd.read_parquet(args.base_dir/'players.parquet');teams=pd.read_parquet(args.base_dir/'teams.parquet')
    base_manifest=json.loads((args.base_dir/'manifest.json').read_text())
    if base_manifest['year']<=max(config['development_years']):
        raise ValueError('Scored season must follow coefficient training')
    boxes,box_path=read_games(base_manifest['year'],base_manifest['season_type'])
    difficulty,buckets,prices,pbp_paths=game_difficulty(base_manifest['year'],base_manifest['season_type'],base_manifest['selected_game_ids'],config['pbp_prior_pseudo_fga'])
    game_keys=base[['game_id','game_date','player_id']]
    difficulty=game_keys.merge(difficulty,on=['game_id','player_id'],how='left',validate='one_to_one')
    boxes=boxes.merge(difficulty.drop(columns='game_id'),left_on=['PLAYER_ID','game_date'],right_on=['player_id','game_date'],how='left',validate='one_to_one')
    boxes=boxes.loc[boxes.player_id.notna()].drop(columns='player_id').copy()
    if not np.allclose(boxes.pbp_fga.fillna(0),boxes.FGA):
        raise ValueError('PBP and official box FGA counts differ; cannot price game role')
    previous,previous_path=read_games(base_manifest['year']-1,'rs')
    league_ts=100*previous.PTS.sum()/(2*(previous.FGA.sum()+0.44*previous.FTA.sum()))
    boxes['TSA']=boxes.FGA+0.44*boxes.FTA
    boxes['TS_pct']=100*boxes.PTS/(2*boxes.TSA.replace(0,np.nan))
    boxes['league_TS_pct']=league_ts
    boxes['rTS_pp']=boxes.TS_pct-league_ts
    boxes['PBP_zTS_proxy']=boxes.rTS_pp+boxes.pbp_stabilized_difficulty_pp
    boxes['PBP_ROLE_POINTS']=2*boxes.TSA*boxes.pbp_stabilized_difficulty_pp.fillna(0)/100
    # No-FGA games carry no observed playtype mix; do not fabricate difficulty.
    boxes.loc[boxes.FGA.eq(0),'PBP_ROLE_POINTS']=0.
    players,terms,reconciliation=score_games(base,teams,boxes,models)
    buckets.to_csv(out/'playtype_buckets.csv',index=False)
    prices.to_csv(out/'playtype_prices.csv',index=False)
    sources+=pbp_paths+[previous_path,Path(__file__).with_name('player_game_playtype.py')]
    actual=players.groupby(['game_id','team']).game_net_points.sum()
    expected=teams.set_index(['game_id','team']).game_net_points
    np.testing.assert_allclose(actual.sort_index(),expected.sort_index(),atol=1e-9)
    for name,frame in [('players',players),('feature_terms',terms),('reconciliation',reconciliation),('coefficients',coefficients),('development_alpha_grid',grid),('historical_report',reports),('training_coverage',coverage),('teams',teams)]:
        frame.to_csv(out/f'{name}.csv',index=False);frame.to_parquet(out/f'{name}.parquet',index=False)
    (out/'models.json').write_text(json.dumps(models,indent=2)+'\n')
    sources += [box_path,args.config,args.base_dir/'players.parquet',args.base_dir/'teams.parquet',args.base_dir/'manifest.json',Path(__file__)]
    manifest=dict(config=config,status='research prototype, revised to traditional-box coefficient priors',player_games=len(players),games=int(players.game_id.nunique()),selected_games=base_manifest['selected_game_ids'],model_training_last=max(config['development_years']),source_method='existing decomp_prior_harness ridge approach adapted to compact count features and Net DECOMP targets; newly fitted coefficients',base_team_contract=base_manifest,inputs=[dict(path=str(p.resolve()),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    render_review(players,teams,terms,coefficients,reports,out,buckets,candidate_coefficients)
    print(json.dumps(dict(player_games=len(players),games=int(players.game_id.nunique()),max_factor_gap=float(reconciliation.gap.abs().max()),output=str(out/'review.html')),indent=2))

if __name__=='__main__':
    main()

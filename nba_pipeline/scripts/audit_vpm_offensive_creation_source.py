"""Independent raw count, coherent split and chronological state audit."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
from export_vpm_offensive_creation import ROOT,OUT,LINES,OWN,SPECS,CATS,DATES,sha,save
from build_nba_player_game_lines import load_common
from build_nba_estimated_skills import skill_subpanel
from build_nba_estimated_skills_v2 import LayoutV2,run_filter_v2
from build_monthly_own_skills import vintage_cut


def count_audit(data):
    common=load_common();gap=0.;rows=0;shots=0;audits=[]
    for y in range(2014,2027):
        for phase in ['rs','ps']:
            path=ROOT/'raw_data'/f'NBA{y%100:02d}{"_PS" if phase=="ps" else ""}.parquet';raw=pd.read_parquet(path);before=len(raw);raw=raw.drop_duplicates();dups=before-len(raw);raw.game_id=common.normalize_game_id_series(raw.game_id);s=data.loc[data.season.eq(y)&data.season_type.eq(phase)].copy();raw=raw.loc[raw.game_id.isin(set(s.game_id))];t=(raw.home_description.fillna('')+' '+raw.visitor_description.fillna('')).str.lower();pid=pd.to_numeric(raw.player1_id,errors='coerce');made=raw.event_type.eq(1)&pid.between(1,1610612736);fga=raw.event_type.isin([1,2])&pid.between(1,1610612736);three=t.str.contains('3pt',regex=False);rim=pd.to_numeric(raw.event_action_type,errors='coerce').isin(common.RIM_ACTION_TYPES)|t.str.contains(common.RIM_DESC_PATTERN,case=False,regex=True);zone=np.where(three,'three',np.where(rim,'rim','mid'));assist=made&t.str.contains(r'\([^()]+? +\d+ AST\)',case=False,regex=True);d=pd.DataFrame(dict(game_id=raw.game_id,player_id=pid))
            for z in ['rim','mid','three']:d[z+'_ua']=(made&~assist&(zone==z)).astype(int);d[z+'_made']=(made&(zone==z)).astype(int);d[z+'_att']=(fga&(zone==z)).astype(int)
            # Reimplement the six-way neutral-description classification.
            cut=t.str.contains(r'cutting|alley oop');selfmade=t.str.contains(r'pull.?up|step.?back|turnaround|fadeaway|driving|floating|floater|hook');rimtext=t.str.contains(r'dunk|layup|tip');cat=np.select([cut&~three,selfmade&three,three,selfmade&~three,rimtext&~three],CATS[:-1],default=CATS[-1])
            for c in CATS:d[c]=(fga&(cat==c)).astype(int);d[c+'_points']=np.where(made&(cat==c),np.where(three,3,2),0)
            ix=pd.MultiIndex.from_frame(s[['game_id','player_id']]);a=d.groupby(['game_id','player_id']).sum().reindex(ix,fill_value=0).reset_index();s=s.reset_index(drop=True);observed=s.assistedness_measured.to_numpy();rimknown=s.rim_assists.notna().to_numpy()
            for z in ['rim','mid','three']:
                valid=s[z+'_fgm_raw'].notna().to_numpy();gap=max(gap,float(np.max(abs((a.loc[valid,z+'_made']-s.loc[valid,z+'_fgm']).to_numpy()),initial=0)),float(np.max(abs((a.loc[valid,z+'_att']-s.loc[valid,z+'_fga']).to_numpy()),initial=0)),float(np.max(abs((a.loc[observed,z+'_ua']-s.loc[observed,z+'_unassisted_fgm']).to_numpy()),initial=0)))
            valid=s[CATS].notna().all(axis=1)
            for c in CATS:
                for key in [c,c+'_points']:gap=max(gap,float(np.max(abs((a.loc[valid,key]-s.loc[valid,key]).to_numpy()),initial=0)))
            g=s.loc[rimknown].groupby('game_id').rim_assists.sum();expected=pd.DataFrame({'game_id':raw.game_id,'n':(assist&(zone=='rim')).astype(int)}).groupby('game_id').n.sum().reindex(g.index);np.testing.assert_array_equal(g,expected)
            for z in ['rim','mid','three']:assert s.loc[observed,z+'_unassisted_fgm'].between(0,s.loc[observed,z+'_fgm']).all()
            rows+=len(s);shots+=int(fga.sum());audits.append(dict(season=y,phase=phase,rows=len(s),shots=int(fga.sum()),duplicates_removed=dups,assistedness_observed=int(observed.sum()),rim_ast_observed=int(rimknown.sum())));print('Independent offensive source counts',y,phase,flush=True)
    assert gap==0
    return dict(raw_player_games=rows,raw_shots=shots,maximum_count_gap=gap,coverage=audits)


def slow_mixture(d,h,k,pop):
    out=np.empty((len(d),6));vals=d[CATS].to_numpy(float);dates=d.game_date.to_numpy();groups=d.groupby('player_id',sort=False).indices
    for ix in groups.values():
        counts=np.zeros(6);mass=0.;last=None
        for i in ix:
            decay=1. if last is None else 2**(-float((dates[i]-last)/np.timedelta64(1,'D'))/h);counts=counts*decay+vals[i];mass=mass*decay+vals[i].sum();out[i]=(counts+k*pop)/(mass+k);last=dates[i]
    return out


def state_audit(data,f):
    gap=0.;modelgap=0.;mixgap=0.;pricedgap=0.;checked=0;converged=0
    for year in range(2015,2026):
        cuts=[pd.Timestamp(c) for c in DATES if vintage_cut(year)<=pd.Timestamp(c)<vintage_cut(year+1)];cut=vintage_cut(year)
        for name,spec in SPECS.items():
            m=json.loads((OUT/'models'/str(year)/(name+'.json')).read_text());assert pd.Timestamp(m['fit_cut'])==cut and pd.Timestamp(m['fit_end'])<cut;train=data.loc[data.game_date.lt(cut)&data[spec.numer].notna()].copy();sub=skill_subpanel(train,spec);w=sub.exposure.to_numpy();obs=sub.obs_rate.to_numpy();mask=(sub.season.gt(sub.season.min())&sub.step_k.gt(0)).to_numpy();assert len(sub)==m['rows'];assert pd.Timestamp(sub.game_date.max())==pd.Timestamp(m['fit_end']);assert abs(np.average(obs,weights=w)-m['mean'])<1e-13
            pred=run_filter_v2(LayoutV2(sub),obs,m['params'],m['mean'])[0];modelgap=max(modelgap,abs(np.average((pred[mask]-obs[mask])**2,weights=w[mask])-m['objective']));converged+=int(m['converged'])
            # Full-later observations under frozen parameters cannot alter past post-states.
            sub=skill_subpanel(data.loc[data[spec.numer].notna()].copy(),spec);_,_,post,_=run_filter_v2(LayoutV2(sub),sub.obs_rate.to_numpy(),m['params'],m['mean']);bounded=np.clip(post,0,1) if spec.kind=='pct' else np.maximum(post,0)*100;sub['expected']=bounded
            for c in cuts:
                d=sub.loc[sub.game_date.lt(c)].groupby('player_id').tail(1).set_index('player_id');d.loc[(c-d.game_date).dt.days.gt(730),'expected']=np.nan;actual=f.loc[f.estimate_date.eq(c)].set_index('player_id')['own_'+name];ex=d.expected.reindex(actual.index);np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(ex));gap=max(gap,float(np.max(abs((actual-ex).to_numpy()),initial=0)));checked+=int(ex.notna().sum())
        m=json.loads((OUT/'models'/str(year)/'shot_mix.json').read_text());assert pd.Timestamp(m['fit_end'])<cut;training=data.loc[data.game_date.lt(cut)&data.fga.gt(0)&data[CATS].notna().all(axis=1)];pop=training[CATS].sum().to_numpy();pop=pop/pop.sum();np.testing.assert_allclose(pop,m['population'],atol=1e-14);d=data.loc[data.fga.gt(0)&data[CATS].notna().all(axis=1)].sort_values(['player_id','game_date','game_id']).reset_index(drop=True);post=slow_mixture(d,m['selected']['half_life'],m['selected']['prior_fga'],pop)
        for c in cuts:
            ix=d.loc[d.game_date.lt(c)].groupby('player_id').tail(1).index;v=d.loc[ix,['player_id','game_date']].copy();v[CATS]=post[ix];v=v.set_index('player_id');actual=f.loc[f.estimate_date.eq(c)].set_index('player_id');ex=v[CATS].reindex(actual.index);got=actual[['pred_mix_'+z for z in CATS]];np.testing.assert_array_equal(np.isfinite(ex.to_numpy()),np.isfinite(got.to_numpy()));mixgap=max(mixgap,float(np.nanmax(abs(ex.to_numpy()-got.to_numpy()))));priceyear=int(data.loc[data.game_date.lt(c),'season'].max())-1;price=data.loc[data.season.eq(priceyear)&data.season_type.eq('rs')];n=price[CATS].sum().to_numpy();points=price[[z+'_points' for z in CATS]].sum().to_numpy();mean=points.sum()/n.sum();prices=(points+100*mean)/(n+100);expected=50*(mean-ex.to_numpy()@prices);age=(c-v.game_date.reindex(actual.index)).dt.days;expected[age.gt(730)]=np.nan;np.testing.assert_array_equal(np.isfinite(expected),np.isfinite(actual.own_role_difficulty_pp));pricedgap=max(pricedgap,float(np.nanmax(abs(expected-actual.own_role_difficulty_pp))));assert actual.role_parameter_fit_end.dropna().lt(c).all();assert actual.role_last_observation_date.dropna().lt(c).all()
        print('Independent offensive source states',year,flush=True)
    assert max(gap,modelgap,mixgap,pricedgap)<1e-10
    return dict(ou_models=55,ou_converged=converged,ou_finite_search_budget=55-converged,mix_models=11,checked_finite_snapshots=checked,maximum_full_source_state_gap=gap,maximum_training_objective_gap=modelgap,maximum_independent_mix_gap=mixgap,maximum_lag_price_gap=pricedgap)


def main():
    for p,h in json.loads((OUT/'source_lock.json').read_text()).items():assert sha(p)==h,p
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p,h in manifest['artifacts'].items():assert sha(OUT/p)==h,p
    for p,h in manifest['models'].items():assert sha(OUT/p)==h,p
    data=pd.read_parquet(OUT/'measurements.parquet');f=pd.read_parquet(OUT/'features.parquet');p=pd.read_parquet(OUT/'provenance.parquet');assert p.last_observation_date.lt(p.estimate_date).all() and p.parameter_fit_end.lt(p.estimate_date).all();count=count_audit(data);states=state_audit(data,f)
    for z in ['rim','mid','three']:
        a=f['own_'+z+'_assisted_makes100'];u=f['own_'+z+'_unassisted_makes100'];total=f['own_'+z+'_makes100'];np.testing.assert_allclose((a+u).dropna(),total.loc[(a+u).notna()],atol=1e-12);assert a.dropna().ge(-1e-12).all() and u.dropna().ge(0).all()
    x=f[['own_rim_assists100','own_ast_rate']].dropna();assert x.own_rim_assists100.between(0,x.own_ast_rate).all();result=dict(status='passed',**count,**states,feature_rows=len(f),provenance_rows=len(p),source_hashes_verified=True,protected_outcomes=0,limitations=['Passer identity uses exact game/side anchor and same-row roster with official-AST reconciliation; unresolved games missing','Finite DE budgets are disclosed; no convergence-based post-result refits','Historical publication availability and retrospective architecture uncertified']);save(OUT/'independent_source_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='coverage'},indent=2))

if __name__=='__main__':main()

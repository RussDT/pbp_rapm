"""Dated, coherent offensive creation skill candidates; no VPM outcome selection."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from build_nba_player_game_lines import (load_common,norm_name,AST_TAG_RE,REB_ANCHOR_RE,FT_ANCHOR_RE,SUB_ANCHOR_RE,FOUL_ANCHOR_RE,INITIAL_PREFIX_RE)
from analyze_pbp_playtype_difficulty_ots import classify_pbp_shots,PBP_CATEGORIES
from build_nba_estimated_skills import SkillSpec,skill_subpanel
from build_nba_estimated_skills_v2 import bounds_v2,LayoutV2,run_filter_v2
from monthly_ou_filter import FastOU,load_kernel
from build_monthly_own_skills import vintage_cut
from export_vpm_own_skills import snapshot_rows
import build_tracking_next_game as T

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/estimated_skills/vpm_offensive_creation_v1';LINES=ROOT/'results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet';OWN=ROOT/'results/estimated_skills/vpm_monthly_neutral_v1/features.parquet'
DATES=sorted([f'{y}-10-01' for y in range(2015,2026) if y!=2020]+['2020-12-01','2026-09-01']);CATS=PBP_CATEGORIES
SPECS={s.name:s for s in [SkillSpec('potential_assists','potential_assists','off_poss','rate'),SkillSpec('rim_assist_share','rim_assists','ast','pct'),*[SkillSpec(z+'_unassisted_share',z+'_unassisted_fgm',z+'_fgm','pct') for z in ['rim','mid','three']]]}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def resolver_tables(df,side,text,pid,common):
    gid=df.game_id.to_numpy();anchors=defaultdict(set)
    for regex in [REB_ANCHOR_RE,FT_ANCHOR_RE,SUB_ANCHOR_RE,FOUL_ANCHOR_RE]:
        for i,t in enumerate(text):
            m=regex.match(t)
            if m and side[i] and 0<pid[i]<1610612737:anchors[(gid[i],side[i],norm_name(m.group(1)))].add(int(pid[i]))
    names=df.loc[(pid>0)&(pid<1610612737),['player1_id','player1_name']].dropna();names['n']=names.player1_name.map(norm_name);nm=names.groupby('player1_id').n.agg(lambda s:s.mode().iloc[0]);nm={int(k):v for k,v in nm.items()}
    return {k:next(iter(v)) for k,v in anchors.items() if len(v)==1},nm


def resolve_passer(game,side,tag,roster,anchors,names):
    name=norm_name(tag);roster={int(x) for x in roster if 0<int(x)<1610612737};p=anchors.get((game,side,name))
    if p in roster:return p
    m=INITIAL_PREFIX_RE.match(name);surname=m.group(2) if m else name;c=[p for p in roster if names.get(p)==surname]
    if len(c)==1:return c[0]
    if m and len(c)>1:
        taken={v for (g,s,n),v in anchors.items() if g==game and s==side and n!=name and n.endswith(' '+surname)};c=[p for p in c if p not in taken]
        if len(c)==1:return c[0]
    return None


def extract(path,source):
    common=load_common();raw=pd.read_parquet(path);raw.game_id=common.normalize_game_id_series(raw.game_id);before=len(raw);raw=common.dedupe_exact_raw_pbp_rows(raw,path.name);duplicates_removed=before-len(raw);raw=common.sort_raw_pbp_chronologically(raw).reset_index(drop=True)
    allowed=set(source.game_id);assert allowed.issubset(set(raw.game_id));excluded=len(set(raw.game_id)-allowed);raw=raw.loc[raw.game_id.isin(allowed)].reset_index(drop=True)
    h=raw.home_description.fillna('').astype(str);a=raw.visitor_description.fillna('').astype(str);et=pd.to_numeric(raw.event_type).to_numpy();pid=pd.to_numeric(raw.player1_id,errors='coerce').fillna(0).to_numpy(int);text=(h+' '+a).str.strip();person=(pid>0)&(pid<1610612737);fga=np.isin(et,[1,2])&person;made=(et==1)&person;three=text.str.contains('3PT',case=False,regex=False).to_numpy();rim=pd.to_numeric(raw.event_action_type,errors='coerce').isin(common.RIM_ACTION_TYPES).to_numpy()|text.str.contains(common.RIM_DESC_PATTERN,case=False,regex=True).to_numpy();zone=np.where(three,'three',np.where(rim,'rim','mid'))
    side=np.where(h.ne('')&a.eq(''),'H',np.where(a.ne('')&h.eq(''),'A',''));own=np.where(side=='H',h,np.where(side=='A',a,''));hl=raw[[f'home_player{i}' for i in range(1,6)]].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(int);al=raw[[f'away_player{i}' for i in range(1,6)]].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(int)
    # Recover ambiguous shot-side text only from exact same-row shooter membership.
    home_member=(hl==pid[:,None]).any(axis=1);away_member=(al==pid[:,None]).any(axis=1);side[(side=='')&home_member&~away_member]='H';side[(side=='')&away_member&~home_member]='A';own=np.where(side=='H',h,np.where(side=='A',a,''));anchors,names=resolver_tables(raw,side,own,pid,common)
    tags=text.str.extract(AST_TAG_RE,expand=True);assisted=made&tags[0].notna().to_numpy();categories=classify_pbp_shots(raw,pd.Series(three,index=raw.index)).to_numpy();shot=pd.DataFrame(dict(game_id=raw.game_id.to_numpy()[fga],player_id=pid[fga],fga=1,fgm=made[fga].astype(int)))
    for z in ['rim','mid','three']:
        shot[z+'_fga']=(zone[fga]==z).astype(int);shot[z+'_fgm']=(made[fga]&(zone[fga]==z)).astype(int);shot[z+'_unassisted_fgm']=(made[fga]&(zone[fga]==z)&~assisted[fga]).astype(int)
    for c in CATS:shot[c]=(categories[fga]==c).astype(int);shot[c+'_points']=(np.where(made, np.where(three,3,2),0)[fga]*(categories[fga]==c)).astype(int)
    counts=shot.groupby(['game_id','player_id']).sum();ix=pd.MultiIndex.from_frame(source[['game_id','player_id']]);counts=counts.reindex(ix,fill_value=0).reset_index();compare=['fga','fgm','rim_fga','rim_fgm','mid_fga','mid_fgm','three_fga','three_fgm'];matching=(counts[compare].to_numpy()==source[compare].to_numpy()).all(axis=1);assert matching.mean()>.999
    passers=[];unresolved=defaultdict(int);same_shooter=0
    for i in np.flatnonzero(assisted):
        g=raw.game_id.iloc[i];s=side[i];roster=hl[i] if s=='H' else al[i] if s=='A' else []
        p=resolve_passer(g,s,tags.iloc[i,0],roster,anchors,names) if s else None
        if p is None or p==pid[i]:unresolved[g]+=1;same_shooter+=int(p==pid[i]);continue
        passers.append((g,p,int(zone[i]=='rim')))
    pe=pd.DataFrame(passers,columns=['game_id','player_id','rim_assists']);pa=pe.groupby(['game_id','player_id']).agg(parsed_ast=('rim_assists','size'),rim_assists=('rim_assists','sum')).reindex(ix,fill_value=0).reset_index();valid_player=(pa.parsed_ast.to_numpy()==source.ast.to_numpy())&matching;invalid_games=set(source.loc[~valid_player,'game_id'])|set(unresolved);valid_game=~source.game_id.isin(invalid_games)
    counts['rim_assists']=pa.rim_assists.to_numpy();counts.loc[~valid_game,'rim_assists']=np.nan
    fields=[c for c in counts if c not in ['game_id','player_id']];counts.loc[~matching,fields]=np.nan
    # Matched games have exact passer/shooter event accounting and no invented zeros.
    gr=counts.loc[valid_game].groupby('game_id').rim_assists.sum();made_rim=pd.DataFrame(dict(game_id=raw.game_id[assisted&(zone=='rim')])).groupby('game_id').size().reindex(gr.index,fill_value=0);np.testing.assert_array_equal(gr,made_rim)
    audit=dict(rows=len(source),shot_rows=int(fga.sum()),shot_players_matched=int(matching.sum()),rim_ast_observed=int(valid_game.sum()),rim_ast_games_observed=int(source.loc[valid_game,'game_id'].nunique()),games=int(source.game_id.nunique()),unresolved_assist_tags=int(sum(unresolved.values())),same_shooter=same_shooter,duplicates_removed=duplicates_removed,excluded_raw_games=excluded)
    return counts,audit


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);(out/'parts').mkdir(exist_ok=True);lines=pd.read_parquet(LINES);lines.game_id=lines.game_id.astype(str).str.zfill(10);lines.game_date=pd.to_datetime(lines.game_date);lines=lines.loc[lines.season.between(2014,2026)&lines.game_date.lt('2026-09-01')].copy();sources=[LINES,OWN,T.TRACKING,Path(__file__)];parts=[];audits=[]
    for y in range(2014,2027):
        for phase in ['rs','ps']:
            path=ROOT/'raw_data'/f'NBA{y%100:02d}{"_PS" if phase=="ps" else ""}.parquet';sources.append(path);dest=out/'parts'/f'{y}_{phase}.parquet';s=lines.loc[lines.season.eq(y)&lines.season_type.eq(phase)].reset_index(drop=True)
            if not dest.exists():
                counts,a=extract(path,s);d=s.merge(counts,on=['game_id','player_id'],suffixes=('','_raw'),validate='one_to_one');d.to_parquet(dest,index=False);save(dest.with_suffix('.json'),a)
            d=pd.read_parquet(dest);a=json.loads(dest.with_suffix('.json').read_text());a.update(season=y,phase=phase);audits.append(a);parts.append(d);print('Creation source',y,phase,a['rows'],'rimAST observed',a['rim_ast_observed'],flush=True)
    data=pd.concat(parts,ignore_index=True);assert not data.duplicated(['game_id','player_id']).any()
    ua=[z+'_unassisted_fgm' for z in ['rim','mid','three']]
    # A missing official assist tag must not become an unassisted make.
    observed_assists=sum((data[z+'_fgm_raw']-data[z+'_unassisted_fgm'] for z in ['rim','mid','three']))
    parsed=observed_assists.groupby(data.game_id).sum();official=data.groupby('game_id').ast.sum();raw_complete=data.fgm_raw.notna().groupby(data.game_id).all();complete=parsed.eq(official)&raw_complete
    data['assistedness_measured']=data.game_id.map(complete).fillna(False)
    data.loc[~data.assistedness_measured,ua]=np.nan

    T.WARMUP=2014;tracking,_=T.build_panel(pd.read_parquet(T.TRACKING),lines);t=tracking.loc[tracking.potential_assists_valid,['game_id','player_id','potential_assists_value']].copy();t.game_id=t.game_id.astype(str).str.zfill(10);data=data.merge(t.rename(columns={'potential_assists_value':'potential_assists'}),on=['game_id','player_id'],how='left',validate='one_to_one')
    for z in ['rim','mid','three']:
        g=data.dropna(subset=[z+'_unassisted_fgm']);assert g[z+'_unassisted_fgm'].between(0,g[z+'_fgm']).all()
    assert np.allclose(data[CATS].sum(axis=1)[data[CATS].notna().all(axis=1)],data.fga[data[CATS].notna().all(axis=1)])
    data=data.sort_values(['player_id','game_date','game_id'],kind='stable').reset_index(drop=True);data.to_parquet(out/'measurements.parquet',index=False)
    sources += [Path(__file__).with_name(n) for n in ['monthly_ou_filter.py','monthly_ou_kernel.cpp','build_nba_player_game_lines.py','analyze_pbp_playtype_difficulty_ots.py','build_nba_estimated_skills.py','build_nba_estimated_skills_v2.py','export_vpm_own_skills.py','build_tracking_next_game.py']]
    sig={str(p):sha(p) for p in sources};lock=out/'source_lock.json'
    if lock.exists():assert json.loads(lock.read_text())==sig
    save(lock,sig);save(out/'source_audit.json',dict(status='passed',files=audits,rows=len(data),potential_ast_observed=int(data.potential_assists.notna().sum()),rim_ast_observed=int(data.rim_assists.notna().sum()),protected_outcomes=0));load_kernel(out/'kernel')


def fit(task):
    out,year,name=Path(task[0]),task[1],task[2];dest=out/'models'/str(year)/(name+'.json')
    if dest.exists():return dict(year=year,skill=name,cached=True)
    started=time.time();cut=vintage_cut(year);d=pd.read_parquet(out/'measurements.parquet');d=d.loc[d.game_date.lt(cut)].copy();spec=SPECS[name];sub=skill_subpanel(d.loc[d[spec.numer].notna()].copy(),spec);obs=sub.obs_rate.to_numpy();w=sub.exposure.to_numpy();mask=(sub.season.gt(sub.season.min())&sub.step_k.gt(0)).to_numpy();assert mask.sum()>100;mean=float(np.average(obs,weights=w));var=float(np.average((obs-mean)**2,weights=w));engine=FastOU(sub,out/'kernel')
    def loss(p):return float(np.average((engine.run(obs,p,mean)[0][mask]-obs[mask])**2,weights=w[mask]))
    result=differential_evolution(loss,bounds_v2(var,float(w.mean())),seed=20260928,maxiter=60,popsize=15,tol=1e-7,polish=False);fast=engine.run(obs,result.x,mean);slow=run_filter_v2(LayoutV2(sub),obs,result.x,mean);gap=max(float(np.max(abs(a-b))) for a,b in zip(fast,slow));assert gap<1e-10
    save(dest,dict(vintage=year,skill=name,fit_cut=str(cut.date()),fit_end=str(sub.game_date.max().date()),rows=len(sub),params=result.x.tolist(),mean=mean,objective=float(result.fun),converged=bool(result.success),maxiter=60,popsize=15,fast_slow_gap=gap,seconds=time.time()-started));return dict(year=year,skill=name,seconds=time.time()-started)


def mix_states(d,h,k,population):
    # Independent player paths; each step updates each player at most once.
    d=d.sort_values(['player_id','game_date','game_id'],kind='stable').reset_index(drop=True);ids=pd.factorize(d.player_id)[0];n=int(ids.max())+1;counts=np.zeros((n,6));mass=np.zeros(n);gap=d.groupby('player_id').game_date.diff().dt.days.fillna(0).to_numpy();step=d.groupby('player_id').cumcount().to_numpy();order=np.argsort(step,kind='stable');splits=np.flatnonzero(np.diff(step[order]))+1;values=d[CATS].to_numpy(float);pre=np.empty_like(values);post=np.empty_like(values)
    for ix in np.split(order,splits):
        p=ids[ix];decay=2**(-gap[ix]/h);counts[p]*=decay[:,None];mass[p]*=decay;pre[ix]=(counts[p]+k*population)/(mass[p,None]+k);counts[p]+=values[ix];mass[p]+=values[ix].sum(axis=1);post[ix]=(counts[p]+k*population)/(mass[p,None]+k)
    return d,pre,post


def fit_mix(out,year):
    dest=out/'models'/str(year)/'shot_mix.json'
    if dest.exists():return
    cut=vintage_cut(year);d=pd.read_parquet(out/'measurements.parquet');d=d.loc[d.game_date.lt(cut)&d.fga.gt(0)&d[CATS].notna().all(axis=1)].copy();population=d[CATS].sum().to_numpy();population=population/population.sum();best=None;choices=[]
    for h in [180,365,730,1500]:
        for k in [50,200,1000]:
            x,pre,_=mix_states(d,h,k,population);mask=(x.season.gt(x.season.min())&x.groupby('player_id').cumcount().gt(0)).to_numpy();obs=x[CATS].to_numpy()/x.fga.to_numpy()[:,None];loss=float(np.average(((pre[mask]-obs[mask])**2).sum(axis=1),weights=x.fga.to_numpy()[mask]));choices.append(dict(half_life=h,prior_fga=k,loss=loss))
            if best is None or (loss,h,k)<(best['loss'],best['half_life'],best['prior_fga']):best=choices[-1]
    save(dest,dict(vintage=year,fit_cut=str(cut.date()),fit_end=str(d.game_date.max().date()),population=population.tolist(),selected=best,choices=choices,rows=len(d)));print('Creation mixture fit',year,best,flush=True)


def export(out):
    data=pd.read_parquet(out/'measurements.parquet');own=pd.read_parquet(OWN);frames=[];prov=[];checks=[]
    for year in range(2015,2026):
        cuts=[pd.Timestamp(c) for c in DATES if vintage_cut(year)<=pd.Timestamp(c)<vintage_cut(year+1)]
        if not cuts:continue
        latest=max(cuts);x=data.loc[data.game_date.lt(latest)].copy();rows={}
        for name,spec in SPECS.items():
            m=json.loads((out/'models'/str(year)/(name+'.json')).read_text());sub=skill_subpanel(x.loc[x[spec.numer].notna()].copy(),spec);engine=FastOU(sub,out/'kernel');pre,_,post,var=engine.run(sub.obs_rate.to_numpy(),m['params'],m['mean']);values=np.clip(post,0,1) if spec.kind=='pct' else np.maximum(post,0)*100;states=sub[['player_id','game_id','game_date']].copy();states['value']=values;states['state_variance']=var;states['observed_exposure']=sub.groupby('player_id').exposure.cumsum();p=snapshot_rows(states,cuts);p['feature']='own_'+name;p['parameter_fit_end']=pd.Timestamp(m['fit_end']);p['parameter_fit_cut']=pd.Timestamp(m['fit_cut']);p['model_sha256']=sha(out/'models'/str(year)/(name+'.json'));assert p.last_observation_date.lt(p.estimate_date).all() and p.parameter_fit_end.lt(p.estimate_date).all();prov.append(p);rows[name]=p[['player_id','estimate_date','value']].rename(columns={'value':'own_'+name})
        combined=own.loc[own.estimate_date.isin(cuts),['player_id','estimate_date','own_ast_rate',*[f'own_{z}_makes100' for z in ['rim','mid','three']],'own_rim_fga_rate','own_mid_fga_rate','own_three_fga_rate','own_fta_rate']].copy()
        for p in rows.values():combined=combined.merge(p,on=['player_id','estimate_date'],how='left',validate='one_to_one')
        combined['own_rim_assists100']=combined.own_ast_rate*combined.own_rim_assist_share
        for z in ['rim','mid','three']:
            combined['own_'+z+'_unassisted_makes100']=combined['own_'+z+'_makes100']*combined['own_'+z+'_unassisted_share'];combined['own_'+z+'_assisted_makes100']=combined['own_'+z+'_makes100']-combined['own_'+z+'_unassisted_makes100']
        m=json.loads((out/'models'/str(year)/'shot_mix.json').read_text());d=x.loc[x.fga.gt(0)&x[CATS].notna().all(axis=1)].copy();d,pre,post=mix_states(d,m['selected']['half_life'],m['selected']['prior_fga'],np.array(m['population']));all_d,_,all_post=mix_states(data.loc[data.fga.gt(0)&data[CATS].notna().all(axis=1)].copy(),m['selected']['half_life'],m['selected']['prior_fga'],np.array(m['population']));key=d[['player_id','game_id']];full=pd.MultiIndex.from_frame(all_d[['player_id','game_id']]).get_indexer(pd.MultiIndex.from_frame(key));gap=float(np.max(abs(post-all_post[full])));assert gap<1e-12;checks.append(dict(vintage=year,prefix_mix_gap=gap))
        for cut in cuts:
            before=d.loc[d.game_date.lt(cut)];idx=before.groupby('player_id').tail(1).index;tab=d.loc[idx,['player_id','game_date']].copy();p=post[idx];tab['estimate_date']=cut;tab['role_last_observation_date']=tab.pop('game_date');tab.loc[(cut-tab.role_last_observation_date).dt.days.gt(730),'stale']=True
            priceyear=int(data.loc[data.game_date.lt(cut),'season'].max())-1;pricing=data.loc[data.season.eq(priceyear)&data.season_type.eq('rs')];n=pricing[CATS].sum().to_numpy();points=pricing[[c+'_points' for c in CATS]].sum().to_numpy();league=float(points.sum()/n.sum());prices=(points+100*league)/(n+100);tab['own_role_difficulty_pp']=50*(league-p@prices);tab.loc[tab.stale.fillna(False),'own_role_difficulty_pp']=np.nan
            for j,c in enumerate(CATS):tab['pred_mix_'+c]=p[:,j]
            tab['role_price_season']=priceyear;tab['role_parameter_fit_end']=pd.Timestamp(m['fit_end']);assert tab.role_last_observation_date.lt(cut).all() and tab.role_parameter_fit_end.lt(cut).all();combined=combined.merge(tab.drop(columns=['stale']),on=['player_id','estimate_date'],how='left',validate='one_to_one') if len(cuts)==1 else combined
            if len(cuts)>1:
                loc=combined.estimate_date.eq(cut);tmp=combined.loc[loc].merge(tab.drop(columns=['stale']),on=['player_id','estimate_date'],how='left',validate='one_to_one');frames.append(tmp)
        if len(cuts)==1:frames.append(combined)
    f=pd.concat(frames,ignore_index=True);p=pd.concat(prov,ignore_index=True);assert not f.duplicated(['player_id','estimate_date']).any();f.to_parquet(out/'features.parquet',index=False);p.to_parquet(out/'provenance.parquet',index=False)
    for z in ['rim','mid','three']:
        valid=f[[f'own_{z}_unassisted_makes100',f'own_{z}_assisted_makes100',f'own_{z}_makes100']].dropna();np.testing.assert_allclose(valid.iloc[:,0]+valid.iloc[:,1],valid.iloc[:,2],atol=1e-12)
    save(out/'manifest.json',dict(status='passed',features_rows=len(f),cuts=DATES,source_audit=json.loads((out/'source_audit.json').read_text()),prefix_checks=checks,models={str(p.relative_to(out)):sha(p) for p in (out/'models').glob('*/*.json')},artifacts={n:sha(out/n) for n in ['measurements.parquet','features.parquet','provenance.parquet']},source_lock_sha256=sha(out/'source_lock.json'),protected_outcomes=0,limits=['Retrospective architecture/source publication vintages uncertified','Skill parameters annual past-only, no full-history parameter reuse','Raw missing source is not zero; dated states expire after730days']));print('Creation feature export passed',len(f),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--stage',choices=['prepare','fit','export','all'],default='all');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage in ['prepare','all']:prepare(a.out)
    if a.stage in ['fit','all']:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for fut in as_completed([pool.submit(fit,(a.out,y,n)) for y in range(2015,2026) for n in SPECS]):print(json.dumps(fut.result()),flush=True)
        for y in range(2015,2026):fit_mix(a.out,y)
    if a.stage in ['export','all']:export(a.out)

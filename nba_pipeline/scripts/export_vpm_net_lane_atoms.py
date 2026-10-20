"""Price-free Net DECOMP atoms on current Parent's exact possession boundaries."""
from __future__ import annotations
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/vpm_net_lane_atoms_v1'
SLOTS=[f'{s}{i}' for s in ['O','D'] for i in range(1,6)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def build(year,phase,out=OUT):
    from process_rapm_blocks import clean_chance_utils as C
    from process_rapm_blocks.common import _finalize_df
    out=Path(out);out.mkdir(parents=True,exist_ok=True);dest=out/f'atoms_{year}_{phase}.parquet'
    if dest.exists():return json.loads(dest.with_suffix('.json').read_text())
    suffix='_PS' if phase=='PS' else '';raw=ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet';canon=ROOT/f'processed/RAPM{year%100:02d}{suffix}.parquet';capture={};counts=[];oldscore=C._attach_rapm_scoring;oldcomp=C.build_decomp_shot_action_components;oldfetch=C.fetch_player_stats_supabase
    def score(*a,**k):
        d=oldscore(*a,**k);capture['raw']=d;return d
    def component(*a,**k):
        if k.get('col_prefix','FC_DECOMP')=='FC_DECOMP':capture['gross']=k.copy()
        return oldcomp(*a,**k)
    try:
        C._attach_rapm_scoring=score;C.build_decomp_shot_action_components=component;C.fetch_player_stats_supabase=lambda *a:None
        base=C.build_clean_chance_base_py(str(raw),year%100,f'{year-1}-{year%100:02d}','VPM_NET_ATOMS',include_continuation_atoms=True,continuation_game_counts=counts)
    finally:C._attach_rapm_scoring=oldscore;C.build_decomp_shot_action_components=oldcomp;C.fetch_player_stats_supabase=oldfetch
    d=capture['raw'];g=capture['gross'];fc=~d.after_miss.fillna(False).astype(bool);a=pd.DataFrame(index=d.index)
    for z,k in [('rim','is_rim_fga'),('mid','is_mid_fga'),('three','is_three_fga')]:
        mask=g[k].astype(bool)&fc;a[z+'_attempts']=mask.astype(float);a[z+'_points']=g['actual_net'].where(mask,0.).astype(float)
    text=(d.home_description.fillna('')+' '+d.visitor_description.fillna(''));ft=text.str.contains('Free Throw',case=False,regex=False)|d.event_type.eq('FreeThrow');tech=text.str.contains('Technical',case=False,regex=False)&ft;one=text.str.contains(r'\b1 of 1\b',regex=True)&ft;two=text.str.contains(r'\b[12] of 2\b',regex=True)&ft;three=text.str.contains(r'\b[123] of 3\b',regex=True)&ft
    a['ft_points']=d.ExpFT.where(ft&fc,0.);a['ft_reference_weight']=np.where(fc&ft&~tech&~one,np.where(three,1/3,np.where(two,.5,0)),0.)
    values=list(a);cumulative=a.groupby(d.game_id,sort=False).cumsum();e=cumulative.loc[base.index];increments=e-e.groupby(base.game_id).shift(fill_value=0)
    for c in values:base[c]=increments[c]
    np.testing.assert_allclose(base[['rim_points','mid_points','three_points','ft_points']].sum(axis=1),base.First_Chance_Diff,atol=1e-9,rtol=0)
    for z in ['rim','mid','three']:
        np.testing.assert_allclose(base[z+'_attempts'],base[z+'_attempts'].round(),atol=1e-9,rtol=0)
        if base[z+'_attempts'].lt(-1e-9).any():raise ValueError('Negative shot count')
    numer=['Actual_Diff','First_Chance_Diff','Second_Chance_Diff','Is_Turnover',*[f'FC_MISS_{z}' for z in ['RIM','MID','THREE']],*values]
    atoms=_finalize_df(base,numer,year%100);reference=pd.read_parquet(canon)
    if len(atoms)!=len(reference):raise ValueError('Net atom possession count differs')
    for c in ['game_id','event_num',*SLOTS]:np.testing.assert_array_equal(pd.to_numeric(atoms[c]),pd.to_numeric(reference[c]))
    np.testing.assert_array_equal(pd.to_datetime(atoms.game_date),pd.to_datetime(reference.game_date))
    atoms['actual_points']=atoms.pop('Actual_Diff');atoms['source_season_end_year']=year;atoms['season_phase']=phase
    gc=pd.DataFrame(counts);gc.game_id=pd.to_numeric(gc.game_id);dates=reference.groupby('game_id').game_date.first();gc['game_date']=gc.game_id.map(dates);assert gc.game_date.notna().all();gc['source_season_end_year']=year;gc['season_phase']=phase
    for z in ['rim','mid','three']:
        gg=pd.DataFrame(dict(game_id=d.game_id,n=a[z+'_attempts'],p=a[z+'_points'])).groupby('game_id')[['n','p']].sum();gg.index=pd.to_numeric(gg.index);gc[z+'_attempts']=gc.game_id.map(gg.n);gc[z+'_fg_points']=gc.game_id.map(gg.p)
    assert gc[[z+'_attempts' for z in ['rim','mid','three']]].notna().all().all();assert sum(gc[z+'_attempts'].sum() for z in ['rim','mid','three'])>0
    atoms.to_parquet(dest,index=False);gc.to_parquet(out/f'games_{year}_{phase}.parquet',index=False)
    report=dict(season=year,phase=phase,rows=len(atoms),games=int(atoms.game_id.nunique()),actual_FT=True,price_free=True,maximum_first_chance_point_gap=float(abs(atoms[['rim_points','mid_points','three_points','ft_points']].sum(axis=1)-atoms.First_Chance_Diff).max()),inputs={str(p):sha(p) for p in [raw,canon,Path(__file__),Path(C.__file__)]},artifacts={dest.name:sha(dest),f'games_{year}_{phase}.parquet':sha(out/f'games_{year}_{phase}.parquet')},protected_outcomes=0)
    save(dest.with_suffix('.json'),report);print('NET ATOM COMPLETE',year,phase,len(atoms),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,required=True);p.add_argument('--phase',choices=['RS','PS'],default='RS');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not 2015<=a.year<=2026:raise ValueError('Only completed historical source years permitted')
    build(a.year,a.phase,a.out)

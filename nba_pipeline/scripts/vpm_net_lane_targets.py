"""Dated Net DECOMP targets from observed, price-free possession atoms."""
import numpy as np
import pandas as pd
LANES=['rim_freq','rim_conversion','mid_freq','mid_conversion','three_freq','three_conversion','ft','tov','second_chance']
REFERENCES=['mid_reference_on_rim','mid_reference_on_three']

def targets(atoms,games,weights=None):
    """The caller supplies only permitted source games for this target window."""
    d=atoms;w=np.ones(len(d)) if weights is None else np.asarray(weights,float)
    if len(w)!=len(d) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Invalid target weights')
    out=pd.DataFrame(0.,index=d.index,columns=LANES+REFERENCES);pricing=[];bonus=np.zeros(len(d))
    for key,pos in d.groupby(['source_season_end_year','season_phase']).indices.items():
        g=games.loc[games.source_season_end_year.eq(key[0])&games.season_phase.eq(key[1])];part=d.iloc[pos]
        if g.empty or g.game_id.duplicated().any():raise ValueError('Missing/duplicate game pricing population')
        numeric=[z+s for z in ['rim','mid','three'] for s in ['_attempts','_fg_points','_misses','_points']]
        if not np.isfinite(g[numeric]).all().all() or g[numeric].lt(0).any().any():raise ValueError('Nonfinite/negative observed game counts')
        if not set(pd.to_numeric(part.game_id)).issubset(set(pd.to_numeric(g.game_id))):raise ValueError('Pricing misses target games')
        total_attempts=sum(g[z+'_attempts'].sum() for z in ['rim','mid','three']);gross=sum(g[z+'_fg_points'].sum() for z in ['rim','mid','three'])/total_attempts if total_attempts else 0.;means={};ks={}
        for z in ['rim','mid','three']:
            n=float(g[z+'_misses'].sum());k=float(g[z+'_points'].sum()/n) if n else 0.;att=float(g[z+'_attempts'].sum());mean=float((g[z+'_fg_points'].sum()+k*n)/att) if att else 0.;means[z]=mean;ks[z]=k;counts=part[z+'_attempts'].to_numpy();miss=part['FC_MISS_'+z.upper()].to_numpy();points=part[z+'_points'].to_numpy()+k*miss;out.iloc[pos,out.columns.get_loc(z+'_freq')]=mean*counts;out.iloc[pos,out.columns.get_loc(z+'_conversion')]=points-mean*counts;bonus[pos]+=k*miss
        ftref=gross*part.ft_reference_weight.to_numpy();out.iloc[pos,out.columns.get_loc('mid_freq')]+=ftref;out.iloc[pos,out.columns.get_loc('ft')]=part.ft_points.to_numpy()-ftref;out.iloc[pos,out.columns.get_loc('mid_reference_on_rim')]=means['mid']*part.rim_attempts.to_numpy();out.iloc[pos,out.columns.get_loc('mid_reference_on_three')]=means['mid']*part.three_attempts.to_numpy();pricing.append(dict(season=int(key[0]),phase=key[1],last_pricing_game=str(pd.to_datetime(g.game_date).max().date()),gross_fga_reference=gross,net_zone_reference=means,kappa=ks))
    out.second_chance=d.Second_Chance_Diff.to_numpy()-bonus
    np.testing.assert_allclose(out[LANES].sum(axis=1),d.actual_points,atol=1e-9,rtol=0)
    tov=d.Is_Turnover.eq(1).to_numpy();non=~tov
    if not non.any():raise ValueError('No non-turnover price sample')
    # All first-chance children use the identical retained sample/weights.
    for name in LANES[:7]+REFERENCES:
        out.loc[tov,name]=np.average(out.loc[non,name],weights=w[non])
    price=float(np.average((d.First_Chance_Diff.to_numpy()+bonus)[non],weights=w[non]));out.tov=np.where(tov,-price,0.)
    np.testing.assert_allclose(out[LANES].sum(axis=1),d.actual_points,atol=1e-9,rtol=0)
    return out.to_numpy(),dict(files=pricing,net_turnover_price=price,maximum_row_gap=float(np.max(abs(out[LANES].sum(axis=1)-d.actual_points))))


def compact(values):
    """Explicit linear rollup to the existing five-shot tree, no baseline cell."""
    a=dict(zip(LANES+REFERENCES,np.asarray(values).T));return np.column_stack([a['rim_freq']-a['mid_reference_on_rim'],a['rim_conversion'],a['mid_freq']+a['mid_conversion']+a['mid_reference_on_rim']+a['mid_reference_on_three'],a['three_freq']-a['mid_reference_on_three'],a['three_conversion'],a['ft'],a['tov'],a['second_chance']])

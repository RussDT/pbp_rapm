"""Independent literal checks of emitted historical feature artifacts."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from historical_features import ROOT, OUT, QUARANTINE, sha, write_json


def audit(out=OUT):
    out=Path(out);m=json.loads((out/'manifest.json').read_text())
    for path,digest in m['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('Historical source changed: '+path)
    for name,digest in m['artifact_hashes'].items():
        if sha(out/name)!=digest:raise ValueError('Historical artifact changed: '+name)
    d=pd.read_parquet(out/'features.parquet');p=pd.read_parquet(out/'provenance.parquet')
    assert not d.duplicated(['estimate_date','player_id']).any()
    assert not p.duplicated(['estimate_date','player_id','feature']).any()
    assert p.last_observation_date.lt(p.estimate_date).all()
    assert p.parameter_fit_end.isna().all()
    assert set(pd.date_range('2014-01-01','2026-10-01',freq='QS')).issubset(d.estimate_date)
    maximum=0.;cells=0
    for c in [c for c in d if c.endswith('_missing')]:
        assert d[c].eq(d[c[:-8]].isna().astype(int)).all(),c
    for zone,attempt in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),('three','three_fga_rate'),('ft','fta_rate')]:
        good=d[f'own_{zone}_pct'].notna()
        gap=(d.loc[good,f'own_{zone}_makes100']-d.loc[good,f'own_{attempt}']*d.loc[good,f'own_{zone}_pct']).abs().max()
        maximum=max(maximum,float(gap))
        assert d[f'own_{zone}_pct'].dropna().between(0,1).all()
        assert (d[f'own_{zone}_makes100'].le(d[f'own_{attempt}']+1e-12)|d[f'own_{zone}_makes100'].isna()).all()
    good=d.own_rim_saved100.notna()
    gap=(d.loc[good,'own_rim_saved100']-2*d.loc[good,'own_rim_contests100']*d.loc[good,'own_rim_suppression']).abs().max()
    maximum=max(maximum,float(gap));assert maximum<1e-10
    legacy=[pd.read_parquet(f) for f in sorted((out/'lines').glob('*.parquet'))]
    modern=pd.read_parquet(ROOT/'results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet')
    lines=pd.concat([*legacy,modern],ignore_index=True)
    lines['game_date']=pd.to_datetime(lines.game_date)
    lines['game_id']=lines.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    excluded=lines.loc[lines.game_id.isin(QUARANTINE)].groupby(['season','season_type','game_id']).agg(
        player_game_rows=('player_id','size'),raw_player_points=('pts','sum'),
        offensive_possessions=('off_poss','sum'),defensive_possessions=('def_poss','sum'),
        last_observation_date=('game_date','max')).reset_index()
    excluded['reason']=np.where(excluded.game_id.str.startswith('00401'),'confirmed_raw_duplicate',
        np.where(excluded.game_id.eq('0029901052'),'raw_points189_vs_numeric_final188',
                 'previous_pinned_2014_2015_canonical_accounting_quarantine'))
    excluded['last_observation_date']=excluded.last_observation_date.dt.strftime('%Y-%m-%d')
    lines=lines.loc[~lines.game_id.isin(QUARANTINE)]
    source_fields=[]
    for year,g in lines.groupby('season'):
        receipt=out/'lines'/f'{int(year)}_rs.json'
        tags=json.loads(receipt.read_text()) if receipt.exists() else None
        for name in ['ast','pf_committed','off_fouls_committed','dreb','oreb','stl','blk']:
            raw_absent=int(year)<2014 and name in ['stl','blk']
            sums=float(g[name].sum())
            source_fields.append(dict(season=int(year),feature=name,rows=len(g),
                numerator_sum=sums,positive_count_rows=int(g[name].gt(0).sum()),
                admitted_source='unavailable_raw_annual_box_used_separately' if raw_absent else 'PBP_count_or_cumulative_tag',
                raw_unresolved_assist_tags=tags['unresolved_tags'].get('ast',0) if tags and name=='ast' else None))
            if not raw_absent and name in ['ast','pf_committed','dreb','oreb'] and sums<100:
                raise ValueError('Degenerate core source field: '+str((year,name)))
    measured=pd.read_parquet(ROOT/'results/estimated_skills/defense_bank_v51_shooting_pf_repaired/measurements.parquet')
    measured['source_year']=pd.to_datetime(measured.game_date).dt.year
    defensive=[]
    for (name,year),g in measured.groupby(['feature','source_year']):
        defensive.append(dict(feature=name,calendar_year=int(year),rows=len(g),
            numerator_sum=float(g.numerator.sum()),positive_count_rows=int(g.numerator.gt(0).sum()),
            denominator_sum=float(g.denominator.sum()),first_observation=str(pd.to_datetime(g.game_date).min().date()),
            last_observation=str(pd.to_datetime(g.game_date).max().date())))
        if len(g)>100 and g.numerator.sum()==0:
            raise ValueError('Entire observed defensive source family is zero: '+str((name,year)))
    field_report=dict(status='passed',core_fields=source_fields,defensive_fields=defensive,
        early_subtype_policy='Shooting/nonoffensive PF and offensive fouls drawn are unavailable before their measured source, never inferred from raw zero subtype codes.',
        partial_assist_attribution='Unresolved legacy name tags are explicitly counted, not evidence of exact per-player reconciliation.')
    write_json(out/'source_field_audit.json',field_report)
    for cut in pd.to_datetime(['2008-10-01','2014-01-01','2020-04-01','2026-10-01']):
        prefix=lines.loc[lines.game_date.lt(cut)].copy()
        prefix['discount']=2**(-(cut-prefix.game_date).dt.total_seconds()/86400/550)
        result=d.loc[d.estimate_date.eq(cut)].set_index('player_id')
        for field,n,e in [('own_ast_rate','ast','off_poss'),('own_tov_rate','tov','off_poss'),
                           ('own_dreb_rate','dreb','def_poss'),('own_three_fga_rate','three_fga','off_poss')]:
            z=prefix.loc[prefix[e].gt(0)&prefix[n].notna()]
            wn=z[n]*z.discount;we=z[e]*z.discount
            league=wn.sum()/we.sum()
            sums=pd.DataFrame(dict(player_id=z.player_id,n=wn,e=we)).groupby('player_id')[['n','e']].sum()
            expected=100*(sums.n+1000*league)/(sums.e+1000)
            actual=result[field].reindex(expected.index)
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
            maximum=max(maximum,float((actual-expected).abs().max()));cells+=len(expected)
        z=prefix.loc[prefix.game_date.ge(cut-pd.Timedelta(days=365))&prefix.season_type.eq('rs')&prefix.minutes.gt(0)]
        expected=z.groupby('player_id').minutes.mean()
        np.testing.assert_allclose(result.ctx_mpg.reindex(expected.index),expected,atol=1e-10,rtol=0)
        cells+=len(expected)
        # Independently recover the legal full-season STL/POSS observations.
        annual=[]
        for year in range(2000,min(cut.year,2013)+1):
            if pd.Timestamp(f'{year}-07-01')>cut:continue
            q=pd.read_csv(ROOT.parent/'player_sheets/year_totals'/f'{year}.csv',usecols=['PLAYER_ID','POSS','STL'])
            q=q.loc[q.POSS.gt(0)].copy();q['game_date']=pd.Timestamp(f'{year}-06-30')
            q=q.rename(columns={'PLAYER_ID':'player_id','POSS':'e','STL':'n'});annual.append(q)
        q=prefix.loc[prefix.season.ge(2014)&prefix.def_poss.gt(0),['player_id','game_date','stl','def_poss']].rename(columns={'stl':'n','def_poss':'e'})
        z=pd.concat([*annual,q],ignore_index=True)
        w=2**(-(cut-z.game_date).dt.total_seconds()/86400/550)
        sums=z.assign(n=z.n*w,e=z.e*w).groupby('player_id')[['n','e']].sum()
        league=sums.n.sum()/sums.e.sum();expected=100*(sums.n+1000*league)/(sums.e+1000)
        expected=expected.reindex(result.index).dropna()
        np.testing.assert_allclose(result.own_stl_rate.reindex(expected.index),expected,atol=1e-10,rtol=0)
        maximum=max(maximum,float((result.own_stl_rate.reindex(expected.index)-expected).abs().max()));cells+=len(expected)
    report=dict(status='passed',rows=len(d),cuts=int(d.estimate_date.nunique()),
        quarterly_cuts=52,provenance_rows=len(p),literal_source_cells=cells,
        maximum_numeric_gap=maximum,all_feature_missing_flags_exact=True,
        all_observation_dates_strictly_before_cut=True,all_fitted_parameter_dates_null=True,
        all_shooting_and_rim_saved_identities_exact=True,protected_outcomes=0)
    report['quarantined_raw_measurements']=excluded.to_dict('records')
    report['source_field_audit_sha256']=sha(out/'source_field_audit.json')
    write_json(out/'independent_audit.json',report)
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    audit(ap.parse_args().out)

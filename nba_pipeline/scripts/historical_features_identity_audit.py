"""Independent official-season population and numerical audit of clean v2 inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from historical_features import ROOT, sha, write_json
from historical_features_identity import OUT, OU_OUT, SOURCE, OU_SOURCE, ADMISSION


def audit():
    admission=json.loads(ADMISSION.read_text())
    pairs={(int(year),int(pid)) for year,m in admission['seasons'].items() for pid in m['admitted_ids']}
    names={(int(year),int(pid)):name for year,m in admission['seasons'].items() for pid,name in m['names'].items()}
    reports={}
    for root in [OUT,OU_OUT]:
        meta=json.loads((root/'manifest.json').read_text())
        assert meta['identity_admission_sha256']==sha(ADMISSION)
        for group in ['source_hashes','sources','models']:
            for path,h in meta.get(group,{}).items():
                if sha(path)!=h:raise ValueError('Identity-clean source changed: '+path)
        for name,h in meta.get('artifact_hashes',meta.get('artifacts',{})).items():
            if sha(root/name)!=h:raise ValueError('Identity-clean artifact changed: '+name)
        d=pd.read_parquet(root/'features.parquet');p=pd.read_parquet(root/'provenance.parquet')
        assert not d.duplicated(['player_id','estimate_date']).any()
        assert not p.duplicated(['player_id','estimate_date','feature']).any()
        assert p.last_observation_date.lt(p.estimate_date).all()
        fit=p.parameter_fit_end.notna()
        assert p.loc[fit,'parameter_fit_end'].lt(p.loc[fit,'estimate_date']).all()
        if 'parameter_fit_cut' in p:
            assert p.loc[fit,'parameter_fit_cut'].le(p.loc[fit,'estimate_date']).all()
        for c in [c for c in d if c.endswith('_missing')]:
            assert d[c].eq(d[c[:-8]].isna().astype(int)).all()
        for zone,attempt in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),('three','three_fga_rate'),('ft','fta_rate')]:
            good=d[f'own_{zone}_pct'].notna()
            np.testing.assert_allclose(d.loc[good,f'own_{zone}_makes100'],
                d.loc[good,f'own_{attempt}']*d.loc[good,f'own_{zone}_pct'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(d.own_rim_saved100,2*d.own_rim_contests100*d.own_rim_suppression,
            atol=1e-12,rtol=0,equal_nan=True)
        reports[root.name]=dict(rows=len(d),cuts=int(d.estimate_date.nunique()),provenance_rows=len(p),all_dates_missingness_and_identities_pass=True)
    old=pd.read_parquet(SOURCE/'player_game_lines.parquet');clean=pd.read_parquet(OUT/'player_game_lines.parquet')
    expected=old.loc[[(int(y),int(p)) in pairs for y,p in zip(old.season,old.player_id)]].copy()
    keys=['game_id','player_id'];a=expected.sort_values(keys).reset_index(drop=True);b=clean.sort_values(keys).reset_index(drop=True)
    assert len(a)==len(b)
    for c in a:
        if c=='player_name':continue
        pd.testing.assert_series_equal(a[c],b[c],check_names=False)
    assert all(b.player_name.to_numpy()==np.array([names[int(y),int(p)] for y,p in zip(b.season,b.player_id)]))
    assert not b.player_name.str.match(r'^ID_\d+$').any()
    # Every snapshot population must be exactly the admitted prior appearance
    # population; this excludes both future people and administrative actor rows.
    wide=pd.read_parquet(OUT/'features.parquet')
    for cut,d in wide.groupby('estimate_date'):
        wanted=set(clean.loc[clean.game_date.lt(cut),'player_id'])
        assert set(d.player_id)==wanted
    assert 203147 in set(clean.loc[clean.season.eq(2014),'player_id'])
    assert 1630492 in set(clean.loc[clean.season.eq(2022),'player_id'])
    # All original OU own field values survive unchanged on retained identities.
    before=pd.read_parquet(OU_SOURCE/'features.parquet').set_index(['player_id','estimate_date'])
    after=pd.read_parquet(OU_OUT/'features.parquet').set_index(['player_id','estimate_date'])
    fields=[c for c in before if c.startswith('own_')]
    np.testing.assert_allclose(after[fields],before.reindex(after.index)[fields],atol=0,rtol=0,equal_nan=True)
    literal=0;maxgap=0.
    # Independent literal TD counts use the clean combined raw rows, not the
    # feature adapter. Annual STL/BLK remains its separately audited source.
    for cut in pd.to_datetime(['2008-10-01','2014-01-01','2026-10-01']):
        source=clean.loc[clean.game_date.lt(cut)].copy()
        source['w']=2**(-(cut-source.game_date).dt.total_seconds()/86400/550)
        packet=wide.loc[wide.estimate_date.eq(cut)].set_index('player_id')
        for field,n,e in [('own_ast_rate','ast','off_poss'),('own_tov_rate','tov','off_poss'),
                           ('own_dreb_rate','dreb','def_poss'),('own_pf_committed_rate','pf_committed','total_poss')]:
            z=source.loc[source.counts_observed&source[e].gt(0)&source[n].notna()]
            sums=z.assign(n=z[n]*z.w,e=z[e]*z.w).groupby('player_id')[['n','e']].sum()
            league=sums.n.sum()/sums.e.sum();expected=100*(sums.n+1000*league)/(sums.e+1000)
            actual=packet[field].reindex(expected.index)
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
            maxgap=max(maxgap,float((actual-expected).abs().max()));literal+=len(expected)
    result=dict(status='passed',packages=reports,official_seasons=len(admission['seasons']),
        official_source_files=len(admission['source_hashes']),identity_admission_sha256=sha(ADMISSION),
        combined_real_game_rows=len(clean),removed_source_rows=len(old)-len(clean),
        exact_official_season_population=True,exact_official_names=True,Chris_Smith_preserved=True,Luca_Vildoza_preserved=True,
        original_OU_retained_values_max_gap=0.,literal_clean_source_cells=literal,maximum_literal_gap=maxgap,
        protected_outcomes=0)
    write_json(OUT/'independent_audit.json',result);write_json(OU_OUT/'independent_audit.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':audit()

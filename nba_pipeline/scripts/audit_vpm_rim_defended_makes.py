"""Rebuild measured rim counts and independently replay dated OU feature prefixes."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import build_monthly_defensive_skills as source
from build_nba_estimated_skills import SkillSpec,skill_subpanel
from build_nba_estimated_skills_v2 import LayoutV2,run_filter_v2

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/estimated_skills/vpm_rim_defended_makes_v1'

def run(out=OUT):
    meta=json.loads((out/'manifest.json').read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,h in json.loads((out/'source_lock.json').read_text()).items():assert digest(Path(name))==h,name
    for name,h in meta['models'].items():assert digest(out/'models'/f'{name}.json')==h
    for name,h in meta['artifact_hashes'].items():assert digest(out/name)==h
    # Fresh extraction repeats the repaired measurement gates against archive/PBP,
    # independently of the frozen parquet used by the new skill builder.
    fresh=out/'source_rebuild';source.prepare(fresh)
    expected=pd.read_parquet(fresh/'rim_counts.parquet').sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    measured=pd.read_parquet(out/'measurements.parquet').sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,measured)
    features=pd.read_parquet(out/'features.parquet');provenance=pd.read_parquet(out/'provenance.parquet');maxgap=0.;cells=0
    spec=SkillSpec('rim_dfgm','rim_dfgm','def_poss','rate')
    for cut,g in features.groupby('estimate_date'):
        p=provenance.loc[provenance.estimate_date.eq(cut)];v=int(p.vintage.iloc[0]);m=json.loads((out/'models'/f'{v}.json').read_text())
        assert pd.Timestamp(m['fit_end'])<pd.Timestamp(m['fit_cut'])<=cut
        # Truncate raw data BEFORE filtering; full-source and truncated exports must agree.
        sub=skill_subpanel(measured.loc[measured.game_date.lt(cut)].copy(),spec)
        post=run_filter_v2(LayoutV2(sub),sub.obs_rate.to_numpy(),m['params'],m['mean'])[2]
        sub['state']=100*np.maximum(post,0);last=sub.groupby('player_id').tail(1).set_index('player_id')
        values=last.state.where((cut-last.game_date).dt.days.le(730)).reindex(g.player_id).to_numpy()
        np.testing.assert_allclose(values,g.own_rim_dfgm100,atol=1e-10,rtol=0,equal_nan=True)
        pp=p.set_index('player_id').reindex(g.player_id)
        np.testing.assert_array_equal(pp.last_observation_date,last.game_date.reindex(g.player_id))
        finite=np.isfinite(values);maxgap=max(maxgap,float(np.max(abs(values[finite]-g.own_rim_dfgm100.to_numpy()[finite]),initial=0)));cells+=len(g)
    own=pd.read_parquet(ROOT/'results/estimated_skills/vpm_monthly_neutral_v1/features.parquet',columns=['player_id','estimate_date','own_rim_contests100','own_rim_saved100'])
    d=features.merge(own,on=['player_id','estimate_date'],how='left',validate='one_to_one');paired=d[['own_rim_dfgm100','own_rim_contests100']].notna().all(axis=1)
    bad=d.loc[paired&d.own_rim_dfgm100.gt(d.own_rim_contests100)].copy();bad.to_csv(out/'independent_count_bound_exceptions.csv',index=False)
    current=d.loc[d.estimate_date.eq('2026-09-01')].dropna()
    result=dict(status='passed',raw_measurements_rebuilt=len(measured),independently_replayed_cells=cells,maximum_prefix_gap=maxgap,measurement_bounds_valid=True,independent_forecast_bound_exceptions=len(bad),paired_forecasts=int(paired.sum()),current_correlations=current[['own_rim_dfgm100','own_rim_contests100','own_rim_saved100']].corr().to_dict(),source_rebuild_path=str(fresh),no_posthoc_projection=True,protected_outcomes=0)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='current_correlations'},indent=2))

if __name__=='__main__':run()

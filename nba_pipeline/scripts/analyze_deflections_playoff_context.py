"""Frozen-OU challenger: prior-season matched-player playoff observation offset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import build_estimated_deflections as prior
from build_deflections_next_game import OUT as BASE, next_game_cells, compare
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2
OUT = BASE.parent / 'deflections_playoff_context'


def phase_pairs(sub):
    d = sub.assign(count=sub.obs_rate * sub.exposure)
    g = d.groupby(['player_id', 'season', 'season_type'])[['count', 'exposure']].sum().unstack('season_type')
    g = g.dropna()
    # Exact weighted player-season fixed-effect coefficient sufficient statistics.
    weight = g['exposure']['rs'] * g['exposure']['ps'] / (g['exposure']['rs'] + g['exposure']['ps'])
    delta = g['count']['ps']/g['exposure']['ps'] - g['count']['rs']/g['exposure']['rs']
    return pd.DataFrame({'weight': weight, 'delta': delta}).reset_index()


def offsets(sub, pairs):
    coefficients = {}
    for season in sorted(sub.season.unique()):
        past = pairs[pairs.season < season]
        coefficients[int(season)] = float(np.average(past.delta, weights=past.weight)) if len(past) else 0.
    return np.array([coefficients[int(s)] if p == 'ps' else 0. for s, p in zip(sub.season, sub.season_type)]), coefficients


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((BASE/'frozen.json').read_text())
    for path, expected in cfg['dependencies'].items():
        assert prior.sha(Path(path)) == expected, path
    sub = prior.subpanel(pd.read_parquet(prior.OUT/'panel.parquet'))
    pairs = phase_pairs(sub)
    offset, coefficients = offsets(sub, pairs)
    model = cfg['models'][cfg['selected']]
    assert model['engine'] == 'ou'
    layout = LayoutV2(sub)
    control = run_filter_v2(layout, sub.obs_rate.to_numpy(), model['params'], cfg['mean_rate'])[0]
    pred, _, state, _ = run_filter_v2(layout, sub.obs_rate.to_numpy(), model['params'], cfg['mean_rate'], transient=offset)
    rows = next_game_cells(sub, {'control': control, 'playoff_context': pred}, range(2020,2026))
    meta = sub.loc[sub.season.between(2020,2025) & sub.step_k.gt(0)].copy()
    rows['season_type'] = meta.season_type.to_numpy()
    ranks = sub.groupby(['player_id','season','season_type']).cumcount()
    prior_ps = set(zip(sub.loc[sub.season_type.eq('ps'),'player_id'], sub.loc[sub.season_type.eq('ps'),'season']+1))
    early = np.array([(p,s) in prior_ps and t == 'rs' and r < 10 for p,s,t,r in zip(sub.player_id,sub.season,sub.season_type,ranks)])
    rows['early_rs_after_ps'] = early[sub.season.between(2020,2025) & sub.step_k.gt(0)]
    report = {'offsets_per100': {s:100*v for s,v in coefficients.items()}, 'method': 'Prior completed seasons, exposure-weighted player-season fixed effects; frozen next-game OU parameters; additive observation offset.'}
    for name, years in [('development',[2020,2021,2022]),('stress',[2023,2024,2025])]:
        d = rows[rows.season.isin(years)]
        report[name] = {k:compare(x,'playoff_context','control') for k,x in [('all',d),('playoffs',d[d.season_type.eq('ps')]),('regular_season',d[d.season_type.eq('rs')]),('early_rs_after_ps',d[d.early_rs_after_ps])]}
        report[name]['by_season'] = {int(s):compare(x,'playoff_context','control') for s,x in d.groupby('season')}
        if name == 'development':
            c = report[name]['all']
            report['passes_development'] = c['delta_mse'] > c['delta_se']
            # Record gate before opening stress comparisons.
            (OUT/'development_gate.json').write_text(json.dumps(report,indent=2))
    report['dependencies'] = {str(p):prior.sha(p) for p in [Path(__file__),BASE/'frozen.json',prior.OUT/'panel.parquet']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    rows.to_parquet(OUT/'cells.parquet',index=False)
    pairs.to_parquet(OUT/'matched_phase_pairs.parquet',index=False)
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()

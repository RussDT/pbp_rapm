"""Read-only audit of the frozen SQ-parent feature package; outputs are artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def audit(config, source, out):
    spec = json.loads(config.read_text())['own_parent']
    wide = pd.read_parquet(source / 'features.parquet')
    provenance = pd.read_parquet(source / 'feature_provenance.parquet')
    manifest = json.loads((source / 'manifest.json').read_text())
    rows = []
    for cut, frame in wide.groupby('estimate_date'):
        # Core-covered is a historical player-state universe, NOT a target roster.
        core = frame.loc[frame.own_ast_rate.notna()]
        for side, key in [('off', 'off_features'), ('def', 'def_features')]:
            for feature in spec[key]:
                p = provenance.loc[provenance.estimate_date.eq(cut)
                    & provenance.feature.eq(feature) & provenance.player_id.isin(core.player_id)]
                observed = p.loc[~p.is_missing]
                rows.append(dict(cut=str(cut.date()), side=side, feature=feature,
                    exported_players=len(frame), core_covered_players=len(core),
                    available_core_players=int(core[feature].notna().sum()),
                    negative_core_values=int(core[feature].lt(0).sum()),
                    available_states_older_365_days=int(observed.age_days.gt(365).sum()) if len(p) else None,
                    median_available_state_age_days=float(observed.age_days.median()) if len(observed) else None,
                    parameter_fit_not_before_cut=int((p.parameter_fit_end.ge(cut)).sum()) if len(p) else None,
                    provenance_kind='direct' if len(p) else 'derived; inspect input provenance'))
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out/'feature_coverage.csv', index=False)
    summary = dict(config=str(config), source=str(source),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [config,source/'manifest.json',source/'features.parquet',source/'feature_provenance.parquet']},
        off_features=spec['off_features'],def_features=spec['def_features'],
        deflections_in_active_prior=any('deflection' in x for x in spec['all_features']),
        cuts=sorted(table.cut.unique().tolist()),
        observation_cutoff_violations=int((provenance.last_observation_date>=provenance.estimate_date).sum()),
        parameter_cutoff_violations=int((provenance.parameter_fit_end>=provenance.estimate_date).sum()),
        nested_chronological_fit=manifest.get('nested_chronological_fit'),
        caveat='Counts describe the exported historical state universe, not target-season roster or minutes coverage. Derived makes inherit both input clocks; this audit does not validate source vintage or SQ target timing.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT.parent.parent/'vpm/vpm/configs/research_harness_v18_own_skills_sq_parent_joint.json')
    parser.add_argument('--source',type=Path,default=ROOT/'results/estimated_skills/vpm_neutral_v1')
    parser.add_argument('--out',type=Path,default=ROOT/'results/estimated_skills/vpm_feature_audit_20260920')
    args=parser.parse_args()
    audit(args.config,args.source,args.out)

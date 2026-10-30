from build_vpm_component_site_payload import ROOT,build
if __name__=='__main__':
 build(ROOT.parent/'vpm/vpm/results/harness/vpm_own_all_components_v1/player_estimates.csv','owncomponents.json',dict(as_of='2026-09-14',model_id='own_all33_components',model_note='All33 own estimated skills enter every offensive and defensive component prior. Prior scale x2, fixed incumbent component penalties, TD550. Snapshot September14,2026; games through June13. Development challenger; no EPM skill inputs.'))

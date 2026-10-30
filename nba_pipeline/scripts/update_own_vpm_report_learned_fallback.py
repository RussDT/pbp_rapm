"""Attach saved learned-center receipts without changing report identity."""
import json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
root=Path(__file__).resolve().parents[2]
app=root/'nba_pipeline/results/estimated_skills/own_vpm_report'
base=root.parent/'vpm/vpm/results/harness/vpm_own_learned_fallback_v1'
snapshot=json.loads((app/'src/data.json').read_text())
board=pd.read_csv(base/'2026-09-14/current_parent_board.csv')
e=pd.read_csv(base/'2024-10-01/comparison_summary.csv');e=e[e.candidate.eq('learned_minutes')].copy()
labels={'sq_own_defense':'Original SQ','coherent_minutes':'Coherent + zero fallback','coherent':'Coherent, no minutes shrinkage','fixed_minus2':'Fixed −2 (−1.5 O / −0.5 D)'}
e['label']=e.reference.map(labels);e=e.rename(columns={'gain_vs_reference_x1e6':'gain','ci_low_x1e6':'low','ci_high_x1e6':'high'})
centers=[]
for date in ['2024-10-01','2026-09-14']:
 m=json.loads((base/date/'manifest.json').read_text());c=m['learned_fallback']
 centers.append({'as_of':date,'off':c['off_points100'],'def':c['def_points100'],'net':c['net_points100'],'players':c['eligible_players'],'low_players':c['eligible_below500']})
 if date=='2024-10-01':scopes=m['evaluation']['scopes']
e['rows']=e.scope.map(lambda s:scopes[s]['rows'])
def q(rows,files,grain,description):
 return {'rows':rows,'source':{'type':'frozen local research receipts','files':files,'grain':grain,'description':description,'period':'Pre-October 2024 fit; 2024–25 development evaluation; September 14, 2026 current fit','caveats':['No baseline is fitted on following-season evaluation errors. Skill parameters are development-only; 2027 untouched.','Learned center is the zero-experience intercept of the fixed reliability formula, not an observed replacement-player mean.']}}
snapshot['queries']['learned_board']=q(json.loads(board.to_json(orient='records')),['vpm_own_learned_fallback_v1/2026-09-14/current_parent_board.csv'],'one current player','582 players; positive-good O/D. Same coherent inputs, parent/residual alphas and minute weights as v20. Exact prior blend and SQ posterior.')
snapshot['queries']['learned_comparisons']=q(json.loads(e.to_json(orient='records')),['2024-10-01/comparison_summary.csv'],'one comparison and predefined scope','Learned fallback gain relative to the named control: positive favors learned. Actual points, 2,000 paired game bootstrap draws.')
snapshot['queries']['learned_centers']=q(centers,['2024-10-01/manifest.json','2026-09-14/manifest.json'],'one pre-cut joint fit','Two side centers profiled jointly with player coefficients in the weighted SQ EV parent ridge. No averaging of shrunken labels.')
a=board[board.player_id.eq(1630828)].iloc[0];c=centers[-1]
f=lambda x:f'{x:+.2f}'
intro=f'## Learning where low-minute priors should land\n\n**The fitted baseline is negative, but its size is not yet stable.** The current fit learns **{f(c["off"])} offense and {f(c["def"])} defense**, totaling **{f(c["net"])} points/100**. The earlier 2024 fit learned **{f(centers[0]["net"])} total**. These are learned zero-experience intercepts of our fixed shrinkage formula, not measured universal replacement levels.\n\nAlex moves from **{f(a.sq_own_defense_net)} originally**, to **{f(a.coherent_minutes_net)} with a zero fallback**, to **{f(a.learned_minutes_net)} with the learned current fallback**. The illustrative fixed −2 fallback gives **{f(a.fixed_minus2_net)}**. The aggressive learned estimate is a research output, not an adopted rating.'
r=e[(e.scope=='all_returning')&(e.reference=='coherent_minutes')].iloc[0];l=e[(e.scope=='low_minutes_returning')&(e.reference=='coherent_minutes')].iloc[0]
result=f'**Learning a negative center improves the point estimate versus zero, but the test is inconclusive.** Across all returning-player lineups the gain is **{f(r.gain)} [{f(r.low)}, {f(r.high)}] ×10⁻⁶**; on lineups with a below-500-minute player it is **{f(l.gain)} [{f(l.low)}, {f(l.high)}]**. Both intervals include zero.\n\nThe learned version also remains inconclusive versus the original SQ and the fixed −2 sensitivity. The large change in fitted center between 2024 and 2026 warrants caution. Keep this as a research challenger; this result does not choose −6.59 as a validated replacement baseline.'
snapshot['queries']['learned_summary']=q([{'intro':intro,'result':result}],['comparison_summary.csv','manifest.json'],'one interpretation','Separate negative fitted center from predictive confirmation; do not promote from a player-ranking example.')
snapshot['buildStatus']='updating';snapshot['generatedAt']=datetime.now(timezone.utc).isoformat()
(app/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False))
print(result)

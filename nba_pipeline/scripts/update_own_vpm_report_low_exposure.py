"""Attach saved low-minute prior evidence and board to the existing report."""
import json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
root=Path(__file__).resolve().parents[2]
app=root/'nba_pipeline/results/estimated_skills/own_vpm_report'
base=root.parent/'vpm/vpm/results/harness/vpm_own_low_exposure_k500_coherent_v2'
snapshot=json.loads((app/'src/data.json').read_text())
board=pd.read_csv(base/'2026-09-14/current_parent_board.csv')
e=pd.read_csv(base/'2024-10-01/comparison_summary.csv')
m=json.loads((base/'2024-10-01/manifest.json').read_text())
labels={'coherent':'Coherent shooting only','minutes_only':'Minutes shrinkage only','coherent_minutes':'Both changes'}
e=e[e.candidate.isin(labels)].copy();e['label']=e.candidate.map(labels)
e=e.rename(columns={'gain_vs_reference_x1e6':'gain','ci_low_x1e6':'low','ci_high_x1e6':'high'})
for field in ['rows','games']:e[field]=e.scope.map(lambda s:m['evaluation']['scopes'][s][field])
def query(rows,files,grain,description):
 return {'rows':rows,'source':{'type':'local frozen research artifacts','files':files,'grain':grain,'description':description,'period':'2024–25 development; September 14, 2026 estimates','caveats':['Fixed K=500 minutes; no tuning; original SQ remains control; 2027 untouched.']}}
snapshot['queries']['low_board']=query(json.loads(board.to_json(orient='records')),['vpm_own_low_exposure_k500_coherent_v2/2026-09-14/current_parent_board.csv','2026-09-14/manifest.json'],'one current player','Original, coherent-only, minutes-only and combined extended-own SQ estimates; positive defense is good. Minutes from own-skill PBP panel strictly before estimate date.')
snapshot['queries']['low_comparisons']=query(json.loads(e.to_json(orient='records')),['2024-10-01/comparison_summary.csv','2024-10-01/manifest.json'],'one ablation and evaluation scope','Positive gain favors the candidate against original v18 SQ. Actual possession points, paired game-clustered intervals, 2,000 draws. Same predictions evaluated on predefined scopes.')
r=e[(e.scope=='all_returning')&(e.candidate=='coherent_minutes')].iloc[0]
l=e[(e.scope=='low_minutes_returning')&(e.candidate=='coherent_minutes')].iloc[0]
def f(x):return f'{x:+.2f}'
narrative=f'**The adjustment corrects Alex’s optimistic prior, but this replay does not show a predictive improvement.** Across all returning-player lineups, the combined version changes MSE by **{f(r.gain)} [{f(r.low)}, {f(r.high)}] ×10⁻⁶** in gain units; the negative point estimate means slightly worse predictions. For lineups containing a player below 500 pre-cut minutes, it is **{f(l.gain)} [{f(l.low)}, {f(l.high)}]**. Both intervals include zero.\n\nThe imputation-only correction has a small positive, inconclusive gain. Keep the original as the comparison control and this fixed minutes rule as a research challenger. A more plausible tail ranking alone does not establish a better predictive model.'
snapshot['queries']['low_summary']=query([{'narrative':narrative}],['2024-10-01/comparison_summary.csv'],'one interpretation','Development evidence; no candidate selected from these results.')
# Exact ID lookup also removes ambiguous surnames from the preserved old boards.
names=dict(zip(board.player_id,board.player_name))
for key in ['board','sq_board']:
 for row in snapshot['queries'][key]['rows']:row['player_name']=names.get(row['player_id'],row['player_name'])
snapshot['buildStatus']='updating';snapshot['generatedAt']=datetime.now(timezone.utc).isoformat()
(app/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False))
print(narrative)

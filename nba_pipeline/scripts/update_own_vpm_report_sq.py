"""Add the frozen SQ-parent experiment to the existing local report, preserving its identity."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[2]
app=root/'nba_pipeline/results/estimated_skills/own_vpm_report'
base=root.parent/'vpm/vpm/results/harness/vpm_own_skills_sq_parent_joint_v1'
snapshot=json.loads((app/'src/data.json').read_text())

def query(rows,files,grain,description):
    return {'rows':rows,'source':{'type':'local research artifacts','files':files,'grain':grain,'description':description,
      'period':'2024-10-01 replay cut and 2026-09-14 current snapshot','evidenceFlow':[{'title':'Frozen own SQ-parent v18 receipts','detail':'Exact original target hashes; unchanged own priors; joint weighted ridge for controls and challengers. Read saved output, no model rerun.'}],
      'caveats':['Fixed-through-2025 skill parameters: development replay only. Expected-FT scoring convention; 2027 untouched.']}}

board=pd.read_csv(base/'2026-09-14/current_parent_board.csv')
rows=json.loads(board.to_json(orient='records'))
cmp=pd.read_csv(base/'2024-10-01/comparison_summary.csv')
snapshot['queries']['sq_board']=query(rows,['current_parent_board.csv','2026-09-14/manifest.json'],'one current player','582 latest-season PBP participants. Positive O/D are good; SQ change compares exact parent with exact SQ parent.')
snapshot['queries']['sq_comparisons']=query(json.loads(cmp.to_json(orient='records')),['comparison_summary.csv','bootstrap_vs_own_defense.json','bootstrap_vs_epm_benchmark.json'],'one paired comparison on 93,034 common possessions','SQ own-defense gain in MSE ×1e6. All controls share the joint solve except the explicitly saved iterative parent and complete EPM.')
actual=pd.read_csv(base/'2024-10-01/actual_points_sensitivity/comparison_summary.csv')
snapshot['queries']['sq_actual_comparisons']=query(json.loads(actual.to_json(orient='records')),['actual_points_sensitivity/comparison_summary.csv','actual_points_sensitivity/manifest.json'],'one paired comparison on the identical 93,034 possessions','Frozen predictions scored against raw PBP actual field goals and free throws. Positive MSE gain ×1e6 favors SQ own-defense; 2,000 paired game-bootstrap draws.')
snapshot['queries']['sq_actual_comparisons']['source']['caveats']=['Development only: skill hyperparameters selected through 2025; predictions unchanged; 2027 untouched. This is a possession-quality test, not a rotation/minutes/game-score forecast.']
for date,key in [('2024-10-01','sq_replay_receipt'),('2026-09-14','sq_current_receipt')]:
    m=json.loads((base/date/'manifest.json').read_text())
    snapshot['queries'][key]=query([{'as_of':date,'training_rows':m['training_rows'],'current_players':m['current_players'],'row_closure':m['target_row_closure_max'],'equal_penalty_closure_points100':m['matched_penalty_coefficient_closure_points100'],'normal_equation_relative_error':m['normal_equation_max_relative_error']}],[date+'/manifest.json'],'one model build','Numerical/source checks; not independent predictive confirmation.')
snapshot['buildStatus']='updating';snapshot['generatedAt']=datetime.now(timezone.utc).isoformat()
(app/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False))
print('Updated existing report data:',len(board),'SQ player rows')

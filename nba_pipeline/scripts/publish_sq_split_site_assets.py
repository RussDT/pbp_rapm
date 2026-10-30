"""Prepare curated player results for the owner-requested static Sites export."""
from pathlib import Path
import json
import shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sq_shooter_teammate_2024_2026_rs'
SITE=OUT/'site'
DIST=SITE/'dist'
DIST.mkdir(parents=True,exist_ok=True)
(DIST/'downloads').mkdir(exist_ok=True)
for name in ['all_results.csv','improved_hybrid_2024-2026_results.csv','provider_2024-2026_results.csv','coverage.csv']:
    shutil.copyfile(OUT/name,DIST/'downloads'/name)
r=pd.read_csv(OUT/'all_results.csv',dtype={'window':str});a=pd.read_csv(OUT/'coverage.csv')
t=(ROOT/'scripts/templates/sq_shooter_teammate.html').read_text()
t=t.replace('<title>Own shots & teammate shots · 2024–2026 RS</title>', '<title>Own & Teammate Shot Quality | 2024–2026 NBA</title><meta name="description" content="Explore three seasons of own and teammate expected scoring and residual shotmaking. Download player CSVs with additive contributions and conditional rates.">')
t=t.replace('Daniel Myers · Research export · Regular season only','Russell Thomas · Research for Daniel Myers · Regular season')
t=t.replace('<div class="cards">','<p class="scope"><b>Field goals only—not full oTS.</b> Three-year lineup-adjusted shooting impact, excluding free throws. No turnovers or rebound continuation. Units: points per 100 on-court team FGAs.</p><div class="cards">',1)
t=t.replace('<p id="note"></p>', '''<div class="downloads"><button id="download-selected">Download this view · CSV</button><a href="downloads/all_results.csv" download>All seasons & both models · CSV</a><a href="downloads/improved_hybrid_2024-2026_results.csv" download>Three-year improved · CSV</a><a href="downloads/provider_2024-2026_results.csv" download>Three-year provider · CSV</a></div><p class="download-note">Every CSV includes the <code>_conditional</code> rates, the four additive contributions, shot share, and exposure counts. “This view” downloads the current model, period, search and minimum-exposure selection.</p><p id="note"></p>''')
ex=r[(r.source=='improved_hybrid')&(r.window=='2024-2026')&(r.player_name=='Luka Doncic')].iloc[0]
explanation=f'''<section class="explain"><h2>What Daniel can compare</h2>
<p>The starting question is where a player’s shooting impact lands: <b>their own attempts or their teammates’ attempts</b>, and <b>expected value or conversion above expectation</b>. SQ retains shooter talent on purpose. A better shooter can raise expected scoring on their own attempts; changing who gets the attempts can change teammates’ expected scoring too.</p>
<p><b>The teammate SQ column is the closest starting point for the gravity comparison.</b> It estimates the player’s association with expected teammate scoring after adjusting for who shares the floor and who defends. It can capture passing, spacing, defensive attention, screens and usage allocation together. A high number does not tell us how much came from each mechanism. Without potential-assist markers on misses, this cannot separate passer credit from presence credit.</p>
<p><b>Teammate shotmaking is a residual, not a second gravity estimate.</b> Positive means teammates converted above the model’s talent-aware expectation. It may include context the SQ model missed, lineup effects and shooting variation. It is not proof of a repeatable ability to make teammates shoot better.</p>
<h2>Conditional rates versus contributions</h2>
<p>A conditional rate answers “what is the adjusted signal <em>when this player shoots</em>, or <em>when a teammate shoots</em>?” The additive contribution also accounts for how often that situation occurs. A player with a small own-shot share has fewer opportunities to contribute through their own attempts.</p>
<div class="formula">Own contribution = shot share × own conditional rate<br>Teammate contribution = (1 − shot share) × teammate conditional rate</div>
<p>For example, Luka’s pooled improved-hybrid shot share is <b>{ex.shot_share:.1%}</b>. His own SQ conditional rate is <b>{ex.own_sq_conditional:+.2f}</b>, giving an own SQ contribution of <b>{ex.own_sq:+.2f}</b>. His teammate SQ conditional rate is <b>{ex.teammate_sq_conditional:+.2f}</b>, giving <b>{ex.teammate_sq:+.2f}</b> after weighting the other <b>{1-ex.shot_share:.1%}</b> of attempts. Together, those two contributions equal his SQ parent of <b>{ex.sq_parent:+.2f}</b>.</p>
<p>This is the usage pattern Russell flagged: a high-usage player can take difficult attempts while teammates take easier ones. Showing both rates and shares makes that pattern visible. Conditional rates remove the mechanical frequency weight from the display; they do <em>not</em> remove usage’s basketball effects or establish causation.</p>
<h2>Reading the downloads</h2>
<p><code>own_sq</code>, <code>teammate_sq</code>, <code>own_make</code>, and <code>teammate_make</code> are the four additive contributions. Their sum is <code>total_shot_impact</code>. Fields such as <code>own_sq_conditional</code> and <code>teammate_make_conditional</code> are the corresponding unweighted rates. <code>shot_share</code> is a fraction from 0 to 1. <code>self_shot_attempts</code> and <code>teammate_shot_attempts</code> count the covered opportunities.</p>
<p>The <code>window</code> value 2024 means 2023–24; 2024-2026 is the pooled three-year fit, not an average of annual ranks. <code>source</code> distinguishes the consistent provider model from the improved hybrid. The geometry and improved-prediction shares describe coverage on the player’s on-court offensive attempts. CSVs retain every player unless you download a filtered view.</p>
</section>'''
t=t.replace('<details open><summary>How to read this</summary>',explanation+'<details><summary>Units and interpretation</summary>')
t=t.replace('</style>', '''
.scope{border-left:3px solid #308279;padding:10px 14px;background:#e7ece7;font-size:14px}.downloads{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin-top:20px}.downloads a,.downloads button{font:inherit;font-size:14px;padding:10px 14px;border:1px solid #acbdb4;border-radius:6px;color:#17483d;background:white;text-decoration:none;cursor:pointer}.downloads button{background:#17483d;color:white}.download-note{font-size:14px;color:#526b73}.explain{max-width:950px;margin-top:35px}.explain h2{margin-top:30px;font-size:23px}.formula{padding:16px 20px;background:#e7ece7;border-radius:8px;font-size:16px;line-height:1.9}code{font-size:.92em;overflow-wrap:anywhere}.card p,table,#note,#count,label{font-size:14px}.foot{font-size:13px}
</style>''')
t=t.replace("const esc=s=>", "let visibleRows=[];\nconst esc=s=>")
t=t.replace("$('head').innerHTML=", "visibleRows=rows;\n$('head').innerHTML=",1)
t=t.replace("render();\n</script>", '''$('download-selected').addEventListener('click',()=>{
 const columns=Object.keys(DATA[0]);
 const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
 const csv=[columns.map(quote).join(','),...visibleRows.map(r=>columns.map(k=>quote(r[k])).join(','))].join('\\r\\n');
 const blob=new Blob(['\\uFEFF'+csv],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);
 const link=document.createElement('a');link.href=url;link.download='sq_split_'+$('source').value+'_'+$('window').value+'_filtered.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
</script>''')
t=t.replace('__DATA__',r.to_json(orient='records').replace('</','<\\/')).replace('__AUDIT__',a.to_json(orient='records'))
(DIST/'index.html').write_text(t)
(DIST/'_headers').write_text('/downloads/*\n  Content-Type: text/csv; charset=utf-8\n  Content-Disposition: attachment\n')
(SITE/'.gitignore').write_text('.sites-runtime/\n.DS_Store\n')
(SITE/'README.md').write_text('# Own and Teammate Shot Quality\n\nOwner-authorized public Sites export of curated player-level research results. Generated CSVs are intentionally committed here to make the downloadable static release reproducible. No raw provider shots or credentials are included. Rebuild assets with nba_pipeline/scripts/publish_sq_split_site_assets.py in the parent RAPM project.\n')
print('Prepared',DIST)

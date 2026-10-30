"""Local descriptive homegrown-v1 versus cached EPM skill table."""
from pathlib import Path
import unicodedata,re
import json
import pandas as pd
import numpy as np
from compare_nba_estimated_skills_to_epm import SKILLS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/estimated_skills/current_player_table'
s=pd.read_csv(OUT/'current_players_provenance.csv')
e=pd.read_csv(ROOT/'cache/epm/epm_all_2026-06-15_go0.csv')
assert not e.player_id.duplicated().any()
d=s.merge(e,on='player_id',suffixes=('_ours','_epm'),how='left',validate='one_to_one')
rows=[]
for spec in SKILLS:
    for _,r in d.iterrows():
        key='p_'+spec.skill
        if key in e.columns:key+='_ours'
        rows.append({'Player':r['Player'],'ID':r.player_id,'Skill':spec.skill,'Ours':100*r[key], 'EPM':100*spec.epm_scale*r[spec.epm_column], 'Ours date':r[spec.skill+'_last_date'],'EPM effective date':r['date'],'EPM requested date':r['requested_date']})
t=pd.DataFrame(rows);t['Difference']=t.Ours-t.EPM
# Owner-provided DARKO export: normalized exact names only, no fuzzy matches.
darko_path=Path('/Users/russ/Downloads/darko-dpm-leaderboard (2).csv')
darko=pd.read_csv(darko_path)
def normalize(name):
    return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',str(name)).encode('ascii','ignore').decode().lower())
darko['name_key']=darko.Player.map(normalize)
assert not darko.name_key.duplicated().any(), 'Ambiguous DARKO name'
player_keys=s[['player_id','Player']].copy();player_keys['name_key']=player_keys.Player.map(normalize)
assert not player_keys.name_key.duplicated().any(), 'Ambiguous homegrown name'
dm=darko.set_index('name_key')
fields={'pts_rate':'Pts per 100','ast_rate':'Ast per 100','three_pct':'3P%','ft_pct':'FT%'}
t['DARKO']=np.nan
for skill,col in fields.items():
    values=pd.to_numeric(dm[col].astype(str).str.replace('%','',regex=False),errors='coerce')
    mask=t.Skill.eq(skill)
    t.loc[mask,'DARKO']=t.loc[mask,'Player'].map(normalize).map(values)
t['Ours minus DARKO']=t.Ours-t.DARKO
t['EPM minus DARKO']=t.EPM-t.DARKO
t['DARKO snapshot']='Unspecified in uploaded CSV'
unmatched=darko[~darko.name_key.isin(player_keys.name_key)][['Player','Team']]
unmatched.to_csv(OUT/'darko_unmatched.csv',index=False)
print('DARKO matched players',player_keys.name_key.isin(dm.index).sum(),'of',len(darko),'uploaded; direct fields',len(fields))

assert len(t)==len(s)*16
assert ((t.Ours-t.EPM-t.Difference).dropna().abs()<1e-10).all()
t.to_csv(OUT/'epm_comparison.csv',index=False)
summary=t.groupby('Skill').apply(lambda x:pd.Series({'matched':x[['Ours','EPM']].dropna().shape[0],'mean_difference':x.Difference.mean(),'correlation':x.Ours.corr(x.EPM)}))
summary.to_csv(OUT/'epm_comparison_summary.csv')
base=(OUT/'index.html').read_text()
style=base[base.index('<style>'):base.index('</style>')+8]
page='''<!doctype html><meta charset="utf-8"><title>Our skills versus EPM</title>STYLE<h1>Homegrown skills · EPM · DARKO</h1><p>Our saved skill states versus EPM predictive skills requested June 15, 2026. 3P% now uses the volume-informed prior (strength 0.5, research); other categories remain v1. The shooting state is updated through its displayed observation date, not later zero-attempt games. Dates shown per skill and player; states are not advanced to today. Players appeared in our 2025–26 archive, not a verified September roster. This is a descriptive comparison, not a prediction-accuracy test.</p><p>DARKO: owner-uploaded leaderboard CSV, snapshot date unspecified. Four matching fields: points/100, assists/100, 3P%, FT%. Normalized exact-name joins only. Blank DARKO cells mean no matching field or player. Overall FG% is not rim FG%. The column Difference means ours minus EPM.</p><p>Counting rates are per100 offensive or defensive possessions as appropriate. Shooting and usage are percent-scale; differences there are percentage points. Usage is an approximate conceptual match: our possession-event rate and provider usage may differ in definition. Shot-zone and possession definitions may also differ. No level alignment applied. Deflections and fouls have no matching field in this cached EPM schema.</p><a href="epm_comparison.csv">Download all comparisons</a> · <a href="epm_comparison_summary.csv">Category summary CSV</a><br><input id="search" placeholder="Search player"><select id="skill"></select><span id="count"></span><div class="wrap"><table><thead id="head"></thead><tbody id="body"></tbody></table></div><script>const rows=DATA;const cols=['Player','Skill','Ours','EPM','DARKO','Difference','Ours minus DARKO','EPM minus DARKO','Ours date','EPM effective date','EPM requested date','DARKO snapshot'];let sort='Player',dir=1;const q=document.getElementById('search'),sel=document.getElementById('skill');for(const k of ['All',...new Set(rows.map(x=>x.Skill))]){const o=document.createElement('option');o.value=k;o.textContent=k.replaceAll('_',' ');sel.append(o)}sel.value='pts_rate';function render(){const r=rows.filter(x=>x.Player.toLowerCase().includes(q.value.toLowerCase())&&(sel.value==='All'||x.Skill===sel.value)).sort((a,b)=>{let x=a[sort],y=b[sort];if(x==null)return 1;if(y==null)return -1;return dir*(typeof x==='number'?x-y:String(x).localeCompare(String(y)))});document.getElementById('body').replaceChildren(...r.map(x=>{const tr=document.createElement('tr');for(const k of cols){const td=document.createElement('td'),v=x[k];td.textContent=v==null?'—':typeof v==='number'?v.toFixed(2):v;tr.append(td)}return tr}));document.getElementById('count').textContent=' '+r.length+' rows'}const tr=document.createElement('tr');for(const k of cols){const th=document.createElement('th');th.textContent=k;th.onclick=()=>{dir=sort===k?-dir:1;sort=k;render()};tr.append(th)}document.getElementById('head').append(tr);q.oninput=render;sel.onchange=render;render();</script>'''
page=page.replace('STYLE',style).replace('DATA',t.to_json(orient='records').replace('</','<\\/'))
(OUT/'epm_comparison.html').write_text(page)
print('players',len(s),'matched EPM',int(d.requested_date.notna().sum()),'rows',len(t));print(summary.round(3).to_string())

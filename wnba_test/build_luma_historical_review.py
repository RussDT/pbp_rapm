#!/usr/bin/env python3
"""Build a self-contained HTML review table for the LUMA historical bridge."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results" / "luma_historical_our_priors"
METRICS_CSV = RESULTS / "luma_historical_metrics_2003_2026.csv"
COMPARISON_CSV = RESULTS / "comparison_to_event_decomp.csv"
OUTPUT_HTML = RESULTS / "luma_historical_review.html"


def load_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    string_fields = {
        "player_id",
        "display_name",
        "source_scheme",
        "model_variant",
        "scope",
        "metric",
    }
    bool_fields = {"qualified_1000", "has_prior_features"}
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
            elif key in string_fields:
                continue
            elif key in bool_fields:
                row[key] = value.lower() == "true"
            else:
                try:
                    row[key] = int(value) if value.isdigit() else float(value)
                except (TypeError, ValueError):
                    pass
    return rows


COVERAGE = [
    {
        "surface": "5v5 lineup and stint boundaries",
        "luma": "Home five, away five, seconds, points",
        "status": "Exact input",
        "ours": "Directly usable in the sparse RAPM design",
    },
    {
        "surface": "Rim / mid / three volume and points",
        "luma": "FGA and points by zone",
        "status": "Exact aggregate",
        "ours": "Supports aggregate-compatible frequency and FG-value targets",
    },
    {
        "surface": "Free throws / turnovers / offensive rebounds",
        "luma": "FTA, FT points, TOV and OREB counts",
        "status": "Exact aggregate",
        "ours": "Supports FT, TOV and OREB-opportunity value targets",
    },
    {
        "surface": "First-chance possession state",
        "luma": "No event order within the stint",
        "status": "Unavailable",
        "ours": "Cannot reproduce FIRST_CHANCE targets from these tallies",
    },
    {
        "surface": "Second-chance clean / continuation kappa",
        "luma": "OREB count only",
        "status": "Proxy only",
        "ours": "OREB_OPPORTUNITY_VALUE is not SECOND_CHANCE_CLEAN",
    },
    {
        "surface": "ShotQuality expected value residuals",
        "luma": "No shot-level SQ EV join key",
        "status": "Unavailable",
        "ours": "Cannot construct mid/three residual targets",
    },
    {
        "surface": "Box-feature priors",
        "luma": "Finished Box+ / leaffactor boards are published",
        "status": "Our feature model",
        "ours": "Uses our PBP Stats features from 2010 onward, including smoothed assisted/self-created scoring role and foul creation/cost",
    },
    {
        "surface": "2003–2009 priors",
        "luma": "Historical stints exist; our prior features do not",
        "status": "Zero warm-up",
        "ours": "Likelihood-only seasons that seed later history",
    },
    {
        "surface": "Channel-construction source code",
        "luma": "High-level Make/Sel documentation and finished JSON boards",
        "status": "Not published here",
        "ours": "Our target algebra and solver are executable and audited",
    },
]


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUMA Historical RAPM Review</title>
<style>
:root{--bg:#0b0e13;--panel:#121720;--panel2:#171e29;--line:#273140;--text:#eef2f7;--muted:#8995a6;--blue:#56a8ff;--green:#65d58b;--red:#ff7d88;--amber:#ffc766;--purple:#b293ff;--shadow:0 16px 42px rgba(0,0,0,.24)}
*{box-sizing:border-box}html{background:var(--bg)}body{margin:0;background:radial-gradient(circle at 80% -10%,#172b45 0,transparent 34%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px;line-height:1.45}
button,input,select{font:inherit}button{cursor:pointer}.shell{max-width:1760px;margin:0 auto;padding:24px}.top{display:flex;gap:18px;align-items:flex-start;justify-content:space-between;margin-bottom:18px}.eyebrow{color:var(--blue);font-size:10px;letter-spacing:.18em;text-transform:uppercase;font-weight:800}.top h1{font-size:25px;line-height:1.15;margin:5px 0 7px;letter-spacing:-.025em}.top p{margin:0;color:var(--muted);max-width:830px}.stamp{text-align:right;color:var(--muted);font-size:11px;white-space:nowrap}.stamp b{display:block;color:var(--text);font-size:13px;margin-bottom:2px}
.notice{border:1px solid rgba(86,168,255,.35);background:rgba(86,168,255,.08);border-radius:12px;padding:11px 14px;margin-bottom:14px;color:#cbdff6}.notice b{color:white}.tabs{display:flex;gap:6px;margin-bottom:14px}.tab{border:1px solid var(--line);background:var(--panel);color:var(--muted);padding:8px 12px;border-radius:9px}.tab.on{color:var(--text);border-color:#486789;background:#192638;box-shadow:inset 0 0 0 1px rgba(86,168,255,.08)}
.panel{background:rgba(18,23,32,.94);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.controls{display:grid;grid-template-columns:110px 150px 130px 130px minmax(180px,1fr) auto auto;gap:10px;padding:13px;margin-bottom:12px;align-items:end}.field label{display:block;color:var(--muted);font-size:9px;letter-spacing:.13em;text-transform:uppercase;margin:0 0 5px 2px}.field input,.field select{width:100%;border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:8px;padding:8px 9px;outline:none}.field input:focus,.field select:focus{border-color:var(--blue)}.check{height:35px;display:flex;align-items:center;gap:7px;color:var(--muted);white-space:nowrap}.check input{accent-color:var(--blue)}.action{height:35px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);color:var(--text);padding:0 12px}.action:hover{border-color:#53657c}
.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:12px}.card{padding:13px 14px}.card .k{color:var(--muted);font-size:9px;letter-spacing:.13em;text-transform:uppercase}.card .v{font-weight:750;font-size:21px;margin-top:3px;font-variant-numeric:tabular-nums}.card .h{color:var(--muted);font-size:10px;margin-top:1px}.card.accent .v{color:var(--blue)}
.table-wrap{overflow:auto;max-height:680px;border-radius:14px}.data-table{width:100%;border-collapse:separate;border-spacing:0;white-space:nowrap;font-variant-numeric:tabular-nums}.data-table th{position:sticky;top:0;z-index:3;background:#161d27;color:var(--muted);font-size:9px;letter-spacing:.11em;text-transform:uppercase;text-align:right;padding:10px;border-bottom:1px solid var(--line);cursor:pointer}.data-table th:first-child,.data-table th:nth-child(2){text-align:left}.data-table th:first-child{left:0;z-index:5}.data-table th:nth-child(2){position:sticky;left:43px;z-index:5}.data-table td{padding:8px 10px;text-align:right;border-bottom:1px solid rgba(39,49,64,.62)}.data-table td:first-child,.data-table td:nth-child(2){text-align:left}.data-table td:first-child{position:sticky;left:0;background:var(--panel);z-index:2;color:var(--muted);width:43px}.data-table td:nth-child(2){position:sticky;left:43px;background:var(--panel);z-index:2;min-width:180px}.data-table tbody tr:hover td{background:#192230}.player{font-weight:650}.pid{color:var(--muted);font-size:9px;margin-top:1px}.pos{color:var(--green)}.neg{color:var(--red)}.zero{color:var(--muted)}.prior-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--purple);margin-right:6px}.warm-dot{background:var(--amber)}.sort{color:var(--blue);margin-left:4px}.tfoot{display:flex;align-items:center;justify-content:space-between;padding:10px 13px;color:var(--muted)}.pager{display:flex;gap:6px}.pager button{border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:7px;padding:5px 9px}.pager button:disabled{opacity:.35;cursor:default}
.detail{margin-top:12px;padding:15px;display:none}.detail.on{display:block}.detail-head{display:flex;justify-content:space-between;align-items:start;margin-bottom:12px}.detail h2{font-size:17px;margin:0}.detail .meta{color:var(--muted);margin-top:3px}.x{border:0;background:none;color:var(--muted);font-size:19px}.matrix{display:grid;grid-template-columns:180px repeat(6,minmax(72px,1fr));overflow:auto}.matrix>div{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}.matrix>div:nth-child(7n+1){text-align:left;color:var(--muted)}.matrix .mh{color:var(--muted);font-size:9px;letter-spacing:.1em;text-transform:uppercase;background:var(--panel2)}
.section{padding:18px}.section h2{font-size:18px;margin:0 0 5px}.section>p{color:var(--muted);margin:0 0 16px;max-width:920px}.coverage{width:100%;border-collapse:collapse}.coverage th{text-align:left;color:var(--muted);font-size:9px;letter-spacing:.12em;text-transform:uppercase;padding:9px 10px;border-bottom:1px solid var(--line)}.coverage td{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}.coverage td:nth-child(1){font-weight:650;width:22%}.coverage td:nth-child(2){color:#cbd3de;width:28%}.coverage td:nth-child(4){color:var(--muted)}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 7px;font-size:9px;letter-spacing:.08em;text-transform:uppercase}.tag.exact{color:var(--green);border-color:rgba(101,213,139,.4);background:rgba(101,213,139,.08)}.tag.proxy{color:var(--amber);border-color:rgba(255,199,102,.4);background:rgba(255,199,102,.08)}.tag.no{color:var(--red);border-color:rgba(255,125,136,.4);background:rgba(255,125,136,.08)}.tag.other{color:var(--purple);border-color:rgba(178,147,255,.4);background:rgba(178,147,255,.08)}
.compare-tools{display:flex;gap:10px;margin:0 0 12px}.hidden{display:none!important}.footnote{color:var(--muted);font-size:11px;margin-top:13px}.footnote code{color:#cbd3de}@media(max-width:950px){.shell{padding:14px}.top{display:block}.stamp{text-align:left;margin-top:12px}.controls{grid-template-columns:repeat(2,minmax(0,1fr))}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.matrix{grid-template-columns:140px repeat(6,90px)}}
</style>
</head>
<body>
<main class="shell">
  <header class="top"><div><div class="eyebrow">Six-Factor RAPM · Historical bridge</div><h1>LUMA stint data × our prior architecture</h1><p>Interactive review of every player-season produced by the corrected 2003–2026 historical build.</p></div><div class="stamp"><b>3,922 player-seasons</b>Generated from audited local artifacts<br>All values are points per 100 possessions</div></header>
  <div class="notice"><b>Important:</b> this is our model fitted to LUMA’s raw aggregate stints. It is not LUMA’s published leaffactor board. OREB opportunity is a historical proxy—not event-sequenced SECOND_CHANCE_CLEAN.</div>
  <nav class="tabs"><button class="tab on" data-tab="players">Player metrics</button><button class="tab" data-tab="comparison">Event-overlap verification</button><button class="tab" data-tab="coverage">What the data supports</button></nav>

  <section id="players">
    <div class="controls panel">
      <div class="field"><label>Season</label><select id="season"></select></div>
      <div class="field"><label>Estimate layer</label><select id="layer"><option value="fit">Prior-informed fit</option><option value="prior">Prior input only</option></select></div>
      <div class="field"><label>Side</label><select id="side"><option value="net">Net impact</option><option value="off">Offense</option><option value="def">Defense</option></select></div>
      <div class="field"><label>Min possessions</label><input id="minPoss" type="number" value="0" min="0" step="100"></div>
      <div class="field"><label>Player search</label><input id="search" placeholder="Name or LUMA ID"></div>
      <label class="check"><input id="qualified" type="checkbox"> 1,000+ poss</label>
      <button class="action" id="export">Export view</button>
    </div>
    <div class="cards">
      <div class="card panel"><div class="k">Visible players</div><div class="v" id="nPlayers">—</div><div class="h" id="nHint">filtered player-seasons</div></div>
      <div class="card panel accent"><div class="k">Leader · total</div><div class="v" id="leaderVal">—</div><div class="h" id="leaderName">—</div></div>
      <div class="card panel"><div class="k">Mean total</div><div class="v" id="meanVal">—</div><div class="h">current side and layer</div></div>
      <div class="card panel"><div class="k">Standard deviation</div><div class="v" id="sdVal">—</div><div class="h">current filtered population</div></div>
      <div class="card panel"><div class="k">Prior coverage</div><div class="v" id="priorVal">—</div><div class="h">our feature prior available</div></div>
    </div>
    <div class="panel table-wrap"><table class="data-table"><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
    <div class="panel tfoot"><span id="range">—</span><div class="pager"><button id="prev">Previous</button><button id="next">Next</button></div></div>
    <div class="detail panel" id="detail"><div class="detail-head"><div><h2 id="detailName"></h2><div class="meta" id="detailMeta"></div></div><button class="x" id="detailClose">×</button></div><div class="matrix" id="matrix"></div></div>
  </section>

  <section id="comparison" class="hidden panel section"><h2>Comparison with the event DECOMP board</h2><p>This is the stored overlap audit for 2022–2026. Correlation shows ordering agreement; MAD shows level disagreement. Use it to verify the historical bridge without claiming the aggregate targets are identical to event DECOMP.</p><div class="compare-tools field" style="max-width:180px"><label>Season</label><select id="compareSeason"></select></div><div style="overflow:auto"><table class="coverage"><thead><tr><th>Season</th><th>Scope</th><th>Metric</th><th>Players</th><th>Pearson r</th><th>LUMA-stint mean</th><th>Event mean</th><th>MAD</th></tr></thead><tbody id="compareBody"></tbody></table></div></section>
  <section id="coverage" class="hidden panel section"><h2>Recoverability from LUMA’s released data</h2><p>LUMA provides high-quality aggregate lineup stints and finished channel boards. The released stint schema is enough for the fast historical bridge below, but not enough to recreate chronology-dependent active DECOMP targets.</p><div style="overflow:auto"><table class="coverage"><thead><tr><th>Surface</th><th>What LUMA provides</th><th>Status</th><th>Use in our model</th></tr></thead><tbody id="coverageBody"></tbody></table></div><div class="footnote">Fast path used here: construct all compatible targets once, reuse one lineup design, and solve the component matrix as multiple right-hand sides. The source implementation is <code>build_luma_historical_metrics.py::solve_block</code>.</div></section>
</main>
<script>
const DATA=__DATA__;
const COMPARE=__COMPARE__;
const COVERAGE=__COVERAGE__;
const COMPONENTS=[
  ["TOTAL","Total"],["TS","TS"],["TOV","TOV"],["REB","REB"],
  ["RIM_FREQ","Rim freq"],["RIM_FG","Rim FG"],["MID_BASE","Mid base"],
  ["THREE_FREQ","3 freq"],["THREE_FG","3 FG"],["FT_VALUE","FT"],
  ["TOV_VALUE","TOV value"],["OREB_OPPORTUNITY_VALUE","OREB opp."],["PURE_TOTAL","Pure RAPM"]
];
const $=s=>document.querySelector(s);const $$=s=>[...document.querySelectorAll(s)];
const state={season:2026,layer:"fit",side:"net",minPoss:0,qualified:false,search:"",sort:"TOTAL",dir:-1,page:0,size:100,selected:null};
const fmt=(v,d=2)=>v==null||!Number.isFinite(+v)?"—":(+v).toFixed(d);
const cls=v=>v==null||Math.abs(+v)<.005?"zero":+v>0?"pos":"neg";
const poss=r=>Math.min(+(r.off_possessions||0),+(r.def_possessions||0));
function value(r,key){if(key==="PURE_TOTAL")return r[`PURE_TOTAL_${state.side}`]??r.PURE_RAPM;const prefix=state.layer==="prior"?"prior_":"";return r[`${prefix}${key}_${state.side}`];}
function filtered(){const q=state.search.trim().toLowerCase();return DATA.filter(r=>(state.season==="all"||r.season===state.season)&&poss(r)>=state.minPoss&&(!state.qualified||r.qualified_1000)&&(!q||`${r.display_name||""} ${r.player_id}`.toLowerCase().includes(q)));}
function sortedRows(){const rows=filtered();rows.sort((a,b)=>{let av=state.sort==="name"?(a.display_name||a.player_id).toLowerCase():state.sort==="poss"?poss(a):value(a,state.sort);let bv=state.sort==="name"?(b.display_name||b.player_id).toLowerCase():state.sort==="poss"?poss(b):value(b,state.sort);if(av==null)return 1;if(bv==null)return -1;return (av>bv?1:av<bv?-1:0)*state.dir});return rows;}
function makeHead(){const cols=[["rank","#"],["name","Player"],["poss","Poss."],...COMPONENTS];$("#head").innerHTML=cols.map(([k,l])=>`<th data-sort="${k}">${l}${state.sort===k?`<span class="sort">${state.dir<0?"▼":"▲"}</span>`:""}</th>`).join("");$$('[data-sort]').forEach(th=>th.onclick=()=>{const k=th.dataset.sort;if(k==="rank")return;state.dir=state.sort===k?-state.dir:(k==="name"?1:-1);state.sort=k;state.page=0;render();});}
function render(){makeHead();const rows=sortedRows();const start=state.page*state.size;if(start>=rows.length&&state.page){state.page=0;return render()}const shown=rows.slice(start,start+state.size);$("#body").innerHTML=shown.map((r,i)=>`<tr data-id="${r.player_id}" data-season="${r.season}"><td>${start+i+1}</td><td><div class="player"><span class="prior-dot ${r.has_prior_features?"":"warm-dot"}"></span>${r.display_name||r.player_id}</div><div class="pid">${r.season} · ${r.player_id}</div></td><td>${Math.round(poss(r)).toLocaleString()}</td>${COMPONENTS.map(([k])=>{const v=value(r,k);return `<td class="${cls(v)}">${fmt(v)}</td>`}).join("")}</tr>`).join("");
  $$('#body tr').forEach(tr=>tr.onclick=()=>showDetail(DATA.find(r=>r.player_id===tr.dataset.id&&String(r.season)===tr.dataset.season)));
  $("#range").textContent=rows.length?`Showing ${start+1}–${Math.min(start+state.size,rows.length)} of ${rows.length.toLocaleString()}`:"No matching player-seasons";$("#prev").disabled=state.page===0;$("#next").disabled=start+state.size>=rows.length;
  const totals=rows.map(r=>value(r,"TOTAL")).filter(Number.isFinite);const mean=totals.length?totals.reduce((a,b)=>a+b,0)/totals.length:0;const sd=totals.length?Math.sqrt(totals.reduce((a,b)=>a+(b-mean)**2,0)/totals.length):0;const leader=rows.slice().sort((a,b)=>(value(b,"TOTAL")??-999)-(value(a,"TOTAL")??-999))[0];
  $("#nPlayers").textContent=rows.length.toLocaleString();$("#leaderVal").textContent=leader?fmt(value(leader,"TOTAL")):"—";$("#leaderName").textContent=leader?(leader.display_name||leader.player_id):"—";$("#meanVal").textContent=fmt(mean);$("#sdVal").textContent=fmt(sd);$("#priorVal").textContent=rows.length?`${Math.round(100*rows.filter(r=>r.has_prior_features).length/rows.length)}%`:"—";
}
function showDetail(r){state.selected=r;$("#detail").classList.add("on");$("#detailName").textContent=r.display_name||r.player_id;$("#detailMeta").textContent=`${r.season} · ${r.player_id} · ${Math.round(poss(r)).toLocaleString()} possessions · ${r.has_prior_features?"our feature prior":"zero-prior warm-up"}`;const heads=["Component","Fit O","Fit D","Fit net","Prior O","Prior D","Prior net"];let cells=heads.map(x=>`<div class="mh">${x}</div>`).join("");for(const [k,l] of COMPONENTS.filter(x=>x[0]!=="PURE_TOTAL")){const vals=[r[`${k}_off`],r[`${k}_def`],r[`${k}_net`],r[`prior_${k}_off`],r[`prior_${k}_def`],r[`prior_${k}_net`]];cells+=`<div>${l}</div>`+vals.map(v=>`<div class="${cls(v)}">${fmt(v,3)}</div>`).join("")}$("#matrix").innerHTML=cells;$("#detail").scrollIntoView({behavior:"smooth",block:"nearest"});}
function renderCompare(){const y=$("#compareSeason").value;const rows=COMPARE.filter(r=>y==="all"||String(r.season)===y);$("#compareBody").innerHTML=rows.map(r=>`<tr><td>${r.season}</td><td>${String(r.scope).replaceAll("_"," ")}</td><td>${r.metric}</td><td>${r.n_players}</td><td class="${+r.pearson>=.8?"pos":""}">${fmt(r.pearson,3)}</td><td>${fmt(r.mean_luma,3)}</td><td>${fmt(r.mean_event,3)}</td><td>${fmt(r.mean_absolute_difference,3)}</td></tr>`).join("");}
function statusClass(s){s=s.toLowerCase();return s.includes("exact")?"exact":s.includes("proxy")||s.includes("warm")?"proxy":s.includes("unavailable")?"no":"other"}
$("#coverageBody").innerHTML=COVERAGE.map(r=>`<tr><td>${r.surface}</td><td>${r.luma}</td><td><span class="tag ${statusClass(r.status)}">${r.status}</span></td><td>${r.ours}</td></tr>`).join("");
const seasons=[...new Set(DATA.map(r=>r.season))].sort((a,b)=>b-a);$("#season").innerHTML=`<option value="all">All seasons</option>`+seasons.map(y=>`<option value="${y}" ${y===2026?"selected":""}>${y}</option>`).join("");const cy=[...new Set(COMPARE.map(r=>r.season))].sort((a,b)=>b-a);$("#compareSeason").innerHTML=`<option value="all">All seasons</option>`+cy.map(y=>`<option value="${y}">${y}</option>`).join("");
for(const id of ["season","layer","side","minPoss","qualified","search"]){$("#"+id).addEventListener(id==="search"?"input":"change",e=>{state[id]=id==="season"?(e.target.value==="all"?"all":+e.target.value):id==="minPoss"?+(e.target.value||0):id==="qualified"?e.target.checked:e.target.value;state.page=0;render();});}
$("#prev").onclick=()=>{state.page--;render();};$("#next").onclick=()=>{state.page++;render();};$("#detailClose").onclick=()=>$("#detail").classList.remove("on");$("#compareSeason").onchange=renderCompare;
$("#export").onclick=()=>{const rows=sortedRows();const cols=["season","player_id","display_name","possessions",...COMPONENTS.map(x=>x[0])];const lines=[cols.join(","),...rows.map(r=>[r.season,r.player_id,`"${String(r.display_name||"").replaceAll('"','""')}"`,poss(r),...COMPONENTS.map(([k])=>value(r,k)??"")].join(","))];const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([lines.join("\n")],{type:"text/csv"}));a.download=`luma_review_${state.season}_${state.layer}_${state.side}.csv`;a.click();URL.revokeObjectURL(a.href);};
$$('.tab').forEach(b=>b.onclick=()=>{$$('.tab').forEach(x=>x.classList.toggle('on',x===b));for(const id of ['players','comparison','coverage'])$('#'+id).classList.toggle('hidden',id!==b.dataset.tab);});
render();renderCompare();
</script>
</body>
</html>'''


def main() -> None:
    metrics = load_csv(METRICS_CSV)
    comparison = load_csv(COMPARISON_CSV)
    payload = (
        HTML.replace("__DATA__", json.dumps(metrics, separators=(",", ":")))
        .replace("__COMPARE__", json.dumps(comparison, separators=(",", ":")))
        .replace("__COVERAGE__", json.dumps(COVERAGE, separators=(",", ":")))
    )
    OUTPUT_HTML.write_text(payload, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()

"""Read-only exposure-definition investigation; outputs are never admitted states.

Replays the local Databallr engine on historical API rows and compares its
per-person possession counts with the pinned original player_game table. API
response position is retained as an ordering proxy because the API omits the
native event_order column. Agreement cannot certify bad source lineups.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import pandas as pd

from prepare_full_skill_history import read_table


UNSAFE = ["0021500278", "0021701046", "0021800388", "0021900376", "0021900539", "0022100105", "0022400763"]
CONTROLS = ["0021500001", "0021700001", "0021800001", "0021900001", "0022100001", "0022400001"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(game: str, root: Path) -> dict:
    """Never permits a 2027 partition, even for an explicit invocation."""
    if game not in UNSAFE + CONTROLS:
        raise ValueError("Only the registered historical diagnostic games are allowed")
    path = root / f"{game}.json"
    if path.exists():
        return {"game_id": game, "sha256": digest(path), "cached": True}
    url = f"https://api.databallr.com/v1/nba/games/{game}/pbp?include=flags"
    with urllib.request.urlopen(url, timeout=45) as response:
        body = response.read()
        metadata = {"url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                    "headers": {k: v for k, v in response.headers.items() if k.lower() in {"etag", "last-modified", "date"}}}
    data = json.loads(body)
    if int(data["season"]) > 2026 or int(data["game_id"]) != int(game):
        raise ValueError("Historical API identity fence failed")
    path.write_bytes(body)
    path.with_suffix(".receipt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"game_id": game, **metadata}


def engine_source(engine_root: Path) -> str:
    imports = engine_root / "apps/api/src/nba-pbp"
    return f'''import fs from "node:fs";
import {{enrichPlays, NON_PLAY_ACTIONS}} from {json.dumps(str(imports / "possessions.ts"))};
import {{PbpAccumulator}} from {json.dumps(str(imports / "aggregate.ts"))};
const results = [];
for (const filename of process.argv.slice(2)) {{
  const data = JSON.parse(fs.readFileSync(filename, "utf8"));
  const raw = data.plays.map((p, index) => ({{...p, game_id:data.game_id, event_order:index,
    actionNumber:p.action_number, actionType:p.action_type, shotResult:p.shot_result,
    scoreHome:p.score_home, scoreAway:p.score_away, qualifier:p.qualifiers.join("|"),
    playerName:p.player_name, assister_id:p.assister_id,
    ...Object.fromEntries(Object.entries(p.flags).map(([k,v]) => [`_flag_${{k}}`,v]))}}));
  const enriched = enrichPlays(raw);
  const players = [];
  for (const team of data.teams.filter(team=>/^[A-Z]{{3}}$/.test(team))) {{
    const f = {{team, seasons:[data.season], playoffs:data.playoffs, on:[],off:[],opponentOn:[],opponentOff:[],
      players:[],opponents:[],commonGames:false,excludeGarbage:false,startTime:0,endTime:Infinity,
      startDate:null,endDate:null,summaryOnly:false,shotQuality:false}};
    const a = new PbpAccumulator(f); a.addGame(enriched);
    for (const r of raw) if(r.team===team&&Number(r.person_id)>0&&!a.players.has(Number(r.person_id)))
      a.players.set(Number(r.person_id),{{}}); // Source retains actors absent from every floor lineup.
    for (const [id,counts] of a.players) players.push({{person_id:id,team,off_poss:counts.off_poss??0,def_poss:counts.def_poss??0}});
  }}
  const groups = new Map();
  for (const row of enriched) {{if (NON_PLAY_ACTIONS.has(row.action)) continue;
    const group=groups.get(row.possession)??[];group.push(row);groups.set(row.possession,group);}}
  const openers = [...groups.values()].map(group => {{
    const r=group.find(row=>!["substitution","timeout","violation"].includes(row.action)&&
      !(row.action==="foul"&&!/shooting|s\\.foul|technical|t\\.foul|flagrant/.test(row.description)))??group[0];
    return {{possession:r.possession,valid:r.possessionValid,lineup_valid:r.lineupValid,
      period:r.period,clock:r.clock,action:r.action,action_number:r.raw.actionNumber,event_index:r.raw.event_order,offense:r.offense,
      team:r.team,players:r.players,acting:r.actingLineup,opposing:r.opposingLineup}};
  }});
  const ejections=enriched.filter(r=>r.action==="ejection").map(r=>({{person_id:r.raw.person_id,
    action_number:r.raw.actionNumber,event_index:r.raw.event_order,team:r.team,period:r.period,clock:r.clock,description:r.raw.description,
    later_real_rows_with_player:enriched.filter(s=>s.raw.event_order>r.raw.event_order&&
      !NON_PLAY_ACTIONS.has(s.action)&&s.action!=="substitution"&&s.players.includes(Number(r.raw.person_id)))
      .map(s=>({{action_number:s.raw.actionNumber,period:s.period,clock:s.clock,action:s.action}}))}}));
  results.push({{game_id:String(data.game_id).padStart(10,"0"),season:data.season,players,openers,ejections,
    invalid_non_sub_rows:enriched.filter(r=>!r.lineupValid&&r.action!=="substitution"&&!NON_PLAY_ACTIONS.has(r.action)).length}});
}}
console.log(JSON.stringify(results));
'''


def ejection_ownership_bounds(game: dict, players: pd.DataFrame) -> pd.DataFrame:
    """Conditional bounds for one missing replacement slot, never an admission.

    Keeps the source possession partition fixed. A known ejected player may
    lose their slot; each absent same-team player may own it. This deliberately
    permits later entrants as replacements: absence from a suspect source
    lineup does not prove real absence. Other source QA problems are outside
    these narrow conditional bounds.
    """
    rows=[]
    for player in players.to_dict("records"):
        side_ranges={side:[0,0] for side in ["off","def"]}
        affected=[]
        for opener in game["openers"]:
            if not opener["valid"] or not opener["lineup_valid"]:
                continue
            uncertain=[e for e in game["ejections"] if e["event_index"] < opener["event_index"]
                       and e["person_id"] in opener["players"] and e["team"] == player["team"]]
            if not uncertain:
                continue
            side="off" if opener["offense"] == player["team"] else "def"
            if player["person_id"] in [e["person_id"] for e in uncertain]:
                side_ranges[side][0]-=1
            elif player["person_id"] not in opener["players"]:
                side_ranges[side][1]+=1
            affected.append(opener["action_number"])
        row={"game_id":game["game_id"],"person_id":player["person_id"],"team":player["team"],
             "conditional_only":True,"recovery_admitted":False,"affected_opener_actions":"|".join(map(str,affected))}
        for side in ["off","def"]:
            value=player[f"{side}_poss_source"]
            row[f"{side}_source"]=value
            row[f"{side}_lower"]=value+side_ranges[side][0]
            row[f"{side}_upper"]=value+side_ranges[side][1]
        row["career_game_source"]=(row["off_source"]+row["def_source"])/2
        row["career_game_lower"]=(row["off_lower"]+row["def_lower"])/2
        row["career_game_upper"]=(row["off_upper"]+row["def_upper"])/2
        rows.append(row)
    return pd.DataFrame(rows)


def counter_ownership_audit(raw_game: dict, replay: dict, source_wrong_rows: int, source_missing_ten: int) -> dict:
    """Explain whether reproduced ejection flags touch any source count owner.

    Limited to ejection flags: other source-quality failures cannot be waved
    through when their independent explanation is unavailable.
    """
    ejected=[]; wrong=[]; structural=[]; actor=[]
    for index,row in enumerate(raw_game["plays"]):
        action=row["action_type"]
        if action == "ejection":
            ejected.append((row["person_id"],index))
        if action in {"substitution","period","ejection","jumpball","violation","replay","heave","stoppage"}:
            continue
        if any(player in row["players_on"] for player,seen in ejected if seen < index):
            wrong.append(row["action_number"])
        if len(row["players_on"]) != 10 or len(row["off_players_on"]) != 5 or len(row["def_players_on"]) != 5:
            structural.append(row["action_number"])
        # Bench technical actors are permitted to be absent from the floor.
        bench_technical=action == "foul" and row["flags"].get("desc_technical",False)
        if row["person_id"] > 0 and row["person_id"] not in row["players_on"] and action != "timeout" and not bench_technical:
            actor.append(row["action_number"])
    owners=[p for p in replay["openers"] if p["valid"] and p["lineup_valid"]]
    owner_actions={p["action_number"] for p in owners}
    touched=sorted(owner_actions.intersection(wrong+structural+actor))
    explained=(len(wrong) == source_wrong_rows and source_wrong_rows > 0 and source_missing_ten == 0
               and not structural and not actor and replay["invalid_non_sub_rows"] == 0)
    return {"scope":"ejection-counter-ownership diagnostic","source_wrong_rows":source_wrong_rows,
            "source_missing_ten":source_missing_ten,"ejection_wrong_actions":wrong,"structural_wrong_actions":structural,
            "unexpected_actor_actions":actor,"source_flags_explained_by_ejection_rule":explained,
            "affected_counter_owner_actions":touched,"count_ownership_unaffected_by_explained_flags":bool(explained and not touched),
            "count_owners":owners,"recovery_admitted":False}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=Path("nba_pipeline/results/estimated_skills/full_history_v2/source"))
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("nba_pipeline/results/estimated_skills/pregame_exposure_definition_v1"))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw_api"; raw.mkdir(exist_ok=True)
    games = UNSAFE + CONTROLS
    if args.fetch:
        with ThreadPoolExecutor(max_workers=3) as pool:
            fetched = list(pool.map(lambda game: fetch(game, raw), games))
        print(json.dumps({"historical_games_cached": len(fetched)}), flush=True)
    files = [raw / f"{game}.json" for game in games]
    if not all(path.exists() for path in files):
        raise FileNotFoundError("Run with --fetch to retrieve the registered historical games")
    ts = args.out / "replay_engine.ts"; ts.write_text(engine_source(args.engine_root))
    bundle = args.out / "replay_engine.mjs"
    subprocess.run([str(args.engine_root / "node_modules/.bin/esbuild"),str(ts),"--bundle","--platform=node","--format=esm",f"--outfile={bundle}"],check=True,capture_output=True)
    replay = json.loads(subprocess.check_output(["node",str(bundle),*[str(path) for path in files]],text=True))
    (args.out / "engine_replay.json").write_text(json.dumps(replay) + "\n")
    core, _ = read_table(args.source,"player_game")
    core["game_id"] = core.game_id.astype(str).str.zfill(10)
    core = core.loc[core.game_id.isin(games)].copy()
    counts = pd.DataFrame([{**player,"game_id":game["game_id"]} for game in replay for player in game["players"]])
    compare = core[["game_id","person_id","team","off_poss","def_poss","lineups_clean","provably_wrong_rows","rows_without_10_players","known_issue"]].merge(
        counts,on=["game_id","person_id","team"],how="outer",suffixes=("_source","_engine"),indicator=True)
    for side in ["off","def"]:
        compare[f"{side}_gap"] = compare[f"{side}_poss_engine"] - compare[f"{side}_poss_source"]
    compare.to_csv(args.out / "per_person_definition_comparison.csv",index=False)
    bounds=pd.concat([ejection_ownership_bounds(game,compare.loc[compare.game_id.eq(game["game_id"])])
                      for game in replay if game["ejections"]],ignore_index=True)
    bounds.to_csv(args.out / "conditional_ejection_ownership_bounds.csv",index=False)
    latest=next(game for game in replay if game["game_id"] == "0022400763")
    latest_source=core.loc[core.game_id.eq("0022400763")]
    ownership=counter_ownership_audit(json.loads((raw / "0022400763.json").read_text()),latest,
                                    int(latest_source.provably_wrong_rows.iloc[0]),int(latest_source.rows_without_10_players.iloc[0]))
    (args.out / "0022400763_counter_ownership.json").write_text(json.dumps(ownership,indent=2)+"\n")
    single=json.loads(subprocess.check_output(["node",str(bundle),str(raw / "0022400763.json")],text=True))[0]
    prefix_equal=single == latest
    candidate=latest_source.loc[latest_source.person_id.isin([1641725,1641790,1641816]),
                                ["game_id","season","date","person_id","off_poss","def_poss"]].copy()
    candidate["game_exposure"]=(candidate.off_poss+candidate.def_poss)/2
    candidate["count_ownership_unaffected"]=ownership["count_ownership_unaffected_by_explained_flags"]
    candidate["single_game_replay_equals_multi_game_replay"]=prefix_equal
    candidate["recovery_admitted"]=False
    candidate.to_csv(args.out / "candidate_career_only_recovery.csv",index=False)
    summaries=[]
    for game in replay:
        sub=compare[compare.game_id.eq(game["game_id"])]
        summaries.append({"game_id":game["game_id"],"flagged":game["game_id"] in UNSAFE,"players":len(sub),
            "join_complete":bool(sub._merge.eq("both").all()),"off_source":float(sub.off_poss_source.sum()),
            "off_engine":float(sub.off_poss_engine.sum()),"max_off_gap":float(sub.off_gap.abs().max()),
            "max_def_gap":float(sub.def_gap.abs().max()),"invalid_non_sub_rows":game["invalid_non_sub_rows"],
            "ejections":len(game["ejections"]),"ejected_later_real_rows":sum(len(e["later_real_rows_with_player"]) for e in game["ejections"])})
    summary=pd.DataFrame(summaries);summary.to_csv(args.out / "game_summary.csv",index=False)
    receipt={"status":"diagnostic_not_admitted","source_rows_compared":len(core),"games":len(games),
        "all_person_counts_equal":bool(compare._merge.eq("both").all() and compare.off_gap.eq(0).all() and compare.def_gap.eq(0).all()),
        "conditional_bounds":"Fixed source possession partition, allowing an unknown replacement for each still-on-floor ejected player; not a complete QA recovery.",
        "caveats":["Original Python totals builder unavailable; local TypeScript engine is an independently maintained port.",
          "API omits native event_order; response position is an ordering proxy, not the original numeric key.",
          "Count agreement cannot certify source lineups; no career exposure is recovered or admitted by this diagnostic."],
        "latest_source_flags_explained":ownership["source_flags_explained_by_ejection_rule"],
        "latest_affected_count_owners":len(ownership["affected_counter_owner_actions"]),
        "latest_single_game_prefix_equal":prefix_equal,
        "hashes":{str(p):digest(p) for p in [Path(__file__),ts,bundle,*files,*sorted((args.source / "player_game").glob("*.parquet")),
            *[args.engine_root / f"apps/api/src/nba-pbp/{name}.ts" for name in ["events","possessions","aggregate"]],
            args.engine_root / "apps/api/src/nba-game-pbp.ts"]}}
    (args.out / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(summary.to_string(index=False));print(json.dumps({k:v for k,v in receipt.items() if k!="hashes"}))


if __name__ == "__main__":
    main()

"""Publish an exact-ID impact-only comparison; never export provider skill snapshots."""
import argparse
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def build(date):
 site=ROOT.parent/'vpm-sq-board-site/dist'
 vpm=json.loads((site/'fouls.json').read_text())
 stem=ROOT/'nba_pipeline/cache/epm'/f'epm_all_{date}_go0'
 epm=pd.read_csv(stem.with_suffix('.csv'))
 manifest=json.loads(stem.with_suffix('.manifest.json').read_text())
 assert epm.player_id.is_unique
 lookup=epm.set_index('player_id').to_dict('index')
 players=[]
 for p in vpm['players']:
  out={k:p[k] for k in ['id','name','net','off','def','off_possessions','def_possessions']}
  e=lookup.get(p['id'],{})
  for key,vkey,delta in [('epm','net','delta'),('oepm','off','odelta'),('depm','def','ddelta')]:
   val=e.get(key)
   out[key]=float(val) if pd.notna(val) else None
   out[delta]=out[vkey]-out[key] if out[key] is not None else None
  players.append(out)
 meta=dict(vpm_model='Expanded + fouls drawn research',vpm_as_of=vpm['meta']['as_of'],epm_requested_date=manifest['requested_date'],epm_effective_dates=manifest['provider_effective_dates'],epm_latest_game=manifest['provider_latest_game_date'],matched_players=sum(p['epm'] is not None for p in players),source='Dunks & Threes predictive EPM, game_optimized=0')
 (site/'comparison.json').write_text(json.dumps(dict(meta=meta,players=players),ensure_ascii=False,allow_nan=False,separators=(',',':')))
 print(meta)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--date',required=True);build(parser.parse_args().date)

"""Raw next-game attempt-rate comparison on cached EPM snapshot dates."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from compare_nba_estimated_skills_to_epm import SNAPSHOT_DATES,latest_states
from build_deflections_next_game import compare
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/estimated_skills';OUT=BASE/'epm_next_game_attempts'
SPECS={'rim_fga_rate':('rim_fga','p_fga_rim_100'),'mid_fga_rate':('mid_fga','p_fga_mid_100'),'three_fga_rate':('three_fga','p_fg3a_100')}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pred=pd.read_parquet(BASE/'predictions.parquet').sort_values(['skill','player_id','game_date','game_id'])
    lines=pd.read_parquet(BASE/'player_game_lines.parquet');lines=lines[lines.season.eq(2026)&lines.off_poss.gt(0)].copy()
    dup=lines.duplicated(['player_id','game_date'],keep=False);excluded=int(dup.sum());lines=lines[~dup].sort_values(['game_date','game_id'])
    rows=[];paths=[BASE/'predictions.parquet',BASE/'player_game_lines.parquet',Path(__file__)]
    for date_text in SNAPSHOT_DATES:
        date=pd.Timestamp(date_text);path=ROOT/f'cache/epm/epm_all_{date_text}_go0.csv';paths.append(path)
        epm=pd.read_csv(path);assert epm.player_id.is_unique;assert epm.requested_date.eq(date_text).all()
        assert pd.to_datetime(epm.date).le(date).all()
        states=latest_states(pred,date)
        games=lines[lines.game_date.ge(date)&lines.game_date.le(date+pd.Timedelta(days=7))].groupby('player_id',sort=False).head(1)
        for skill,(count,field) in SPECS.items():
            s=states[states.skill.eq(skill)]
            x=games.merge(s[['player_id','ours']],on='player_id',validate='one_to_one').merge(epm[['player_id',field,'date']],on='player_id',validate='one_to_one')
            x=x.dropna(subset=['ours',field]);x['epm']=x[field];x['ours']=100*x.ours;x['target100']=100*x[count]/x.off_poss
            x['exposure']=x.off_poss;x['skill']=skill;x['snapshot']=date_text;x['lag_days']=(x.game_date-date).dt.days
            rows.append(x[['player_id','game_id','game_date','season','skill','snapshot','date','lag_days','ours','epm','target100','exposure']])
    d=pd.concat(rows,ignore_index=True)
    # One outcome per player/game/skill, retain latest eligible snapshot.
    d=d.sort_values('snapshot').drop_duplicates(['player_id','game_id','skill'],keep='last')
    report={'snapshot_dates':list(SNAPSHOT_DATES),'duplicate_source_rows_excluded':excluded,'objective':'raw next-game rate squared error; possession weighted; paired game bootstrap; 2026 spent historical evidence','results':{}}
    for label,frame in [('same_day',d[d.lag_days.eq(0)]),('next_appearance_within7',d)]:
        report['results'][label]={}
        for skill,x in frame.groupby('skill'):
            c=compare(x,'ours','epm',draws=2000)
            c['snapshot_count']=int(x.snapshot.nunique());c['players']=int(x.player_id.nunique())
            c['ours_bias100']=float(np.average(x.ours-x.target100,weights=x.exposure));c['epm_bias100']=float(np.average(x.epm-x.target100,weights=x.exposure))
            report['results'][label][skill]=c
    report['dependencies']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    # Provider values stay only in private ignored artifacts.
    d.to_parquet(OUT/'private_comparison_rows.parquet',index=False)
    print(json.dumps(report['results'],indent=2))
if __name__=='__main__':main()

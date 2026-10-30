import glob, pandas as pd, numpy as np
files = sorted(glob.glob('player_sheets/perc/*.csv'))
rows=[]
for f in files:
    try:
        d = pd.read_csv(f, usecols=['PLAYER_ID','year','AGE','DRAFT_YEAR','DRAFT_NUMBER','PLAYER_HEIGHT_INCHES','PLAYER_WEIGHT'])
    except Exception:
        continue
    rows.append(d)
df = pd.concat(rows, ignore_index=True)
for c in ['AGE','DRAFT_YEAR','DRAFT_NUMBER','PLAYER_HEIGHT_INCHES','PLAYER_WEIGHT']:
    df[c]=pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['PLAYER_ID','year','AGE'])
df['PLAYER_ID']=df['PLAYER_ID'].astype(int); df['year']=df['year'].astype(int)
g = df.groupby(['PLAYER_ID','year'], as_index=False).agg(
    age=('AGE','median'), draft_year=('DRAFT_YEAR','median'),
    draft_number=('DRAFT_NUMBER','median'), height=('PLAYER_HEIGHT_INCHES','median'),
    weight=('PLAYER_WEIGHT','median'))
g = g.rename(columns={'PLAYER_ID':'player_id','year':'season'})
print('age lookup rows', len(g), 'players', g.player_id.nunique(), 'seasons', g.season.min(), g.season.max())
print(g[g.season.between(2019,2026)].groupby('season').size())
g.to_parquet('nba_pipeline/results/estimated_skills/age_lookup.parquet', index=False)

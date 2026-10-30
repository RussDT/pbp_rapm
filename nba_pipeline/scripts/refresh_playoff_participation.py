"""Add participation fields to a saved title-lift cut without refitting title models."""
from pathlib import Path
import argparse
import json
import gzip
import pandas as pd

try:
    from nba_pipeline.scripts.playoff_availability import refresh_participation, career_participation_summary, CAREER_FIELDS, SEASON_FIELDS
    from nba_pipeline.scripts.build_championship_help import rounded_records, write_gzip_json
except ModuleNotFoundError:
    from playoff_availability import refresh_participation, career_participation_summary, CAREER_FIELDS, SEASON_FIELDS
    from build_championship_help import rounded_records, write_gzip_json


def patch_app_participation(app_dir, career, seasons, report):
    """Patch existing rows, preserving externally merged model fields and metadata."""
    def patch(rows, frame, keys, fields):
        lookup={tuple(r[k] for k in keys):r for r in rounded_records(frame, list(keys)+fields)}
        for row in rows:
            additions=lookup.get(tuple(row[k] for k in keys))
            if additions is not None: row.update(additions)
        return rows
    for name,frame,keys,fields in [('players.json.gz',career,('player_id',),CAREER_FIELDS),
                                 ('season-players.json.gz',seasons,('season','player_id'),SEASON_FIELDS)]:
        path=app_dir/name
        with gzip.open(path,'rt') as handle: rows=json.load(handle)
        write_gzip_json(path,patch(rows,frame,keys,fields))
    for path in (app_dir/'players').glob('*.json.gz'):
        with gzip.open(path,'rt') as handle: payload=json.load(handle)
        payload['seasons']=patch(payload['seasons'],seasons,('season','player_id'),SEASON_FIELDS)
        write_gzip_json(path,payload)
    path=app_dir/'metadata.json'; metadata=json.loads(path.read_text())
    metadata['playoff_participation']=report
    path.write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=root/'results/championship_help')
    parser.add_argument('--app-data-dir',type=Path,default=root.parent/'teammate-help-app/public/data/championship-help')
    parser.add_argument('--raw-dir',type=Path,default=root/'raw_data')
    args=parser.parse_args(); directory=args.output_dir
    original=pd.read_csv(directory/'player_season_estimates.csv')
    seasons,cv,report=refresh_participation(original,pd.read_csv(directory/'playoff_games.csv'),
        pd.read_csv(directory/'regular_season_roster.csv'),args.raw_dir,
        root/'config/playoff_participation_roster_overrides.json')
    old_columns=[c for c in original if c not in SEASON_FIELDS]
    pd.testing.assert_frame_equal(original[old_columns].reset_index(drop=True),seasons[old_columns].reset_index(drop=True))
    finite=seasons.playoff_participation_used.dropna()
    if not finite.between(0,1).all(): raise ValueError('Participation outside [0,1]')
    career=pd.read_csv(directory/'career_player_estimates.csv')
    career=career.drop(columns=[c for c in CAREER_FIELDS if c in career]).merge(
        career_participation_summary(seasons),on='player_id',validate='one_to_one')
    metadata=json.loads((directory/'metadata.json').read_text()); metadata['playoff_participation']=report
    seasons.to_csv(directory/'player_season_estimates.csv',index=False)
    career.to_csv(directory/'career_player_estimates.csv',index=False)
    cv.to_csv(directory/'playoff_participation_cv.csv',index=False)
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    patch_app_participation(args.app_data_dir,career,seasons,report)
    print(json.dumps(report,indent=2))
    print(seasons.loc[seasons.player_id.isin([203507,201142,202681,201567,202695]),
        ['season','player_name','playoff_games_played','team_playoff_games','playoff_participation_used','playoff_participation_source']].tail(15).to_string(index=False))

if __name__=='__main__': main()

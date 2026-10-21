"""Official season-specific player admission for historical VPM lineup slots.

Only identity/appearance metadata is exported. Annual statistics are not VPM
features here. An ID must have positive GP in that season's RS or PS official
player sheet; no minimum minutes or rounded possession threshold is applied.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from historical_targets import OUT, ROOT, SLOTS, sha

SHEETS = ROOT.parent / 'player_sheets/year_totals'


def season_identity(rs, ps):
    ids = set(); names = {}; aliases = {}; support = {}
    for phase, frame in [('RS', rs), ('PS', ps)]:
        required = {'PLAYER_ID', 'PLAYER_NAME', 'GP', 'MIN'}
        if not required.issubset(frame):
            raise ValueError('Official annual identity columns missing')
        d = frame.copy()
        d['PLAYER_ID'] = pd.to_numeric(d.PLAYER_ID, errors='raise')
        d['GP'] = pd.to_numeric(d.GP, errors='raise')
        if d.PLAYER_ID.isna().any() or not d.PLAYER_ID.eq(d.PLAYER_ID.round()).all() or d.PLAYER_ID.le(0).any():
            raise ValueError('Invalid official NBA player identity')
        if d.PLAYER_ID.duplicated().any():
            raise ValueError('Duplicate official annual player identity')
        d = d.loc[d.GP.gt(0)]
        for row in d.itertuples():
            key = str(int(row.PLAYER_ID)); name = str(row.PLAYER_NAME).strip()
            if not name or name.lower() == 'nan':
                raise ValueError('Positive appearance has no official player name')
            ids.add(int(key)); names.setdefault(key, name)
            aliases.setdefault(key, set()).add(name)
            support.setdefault(key, []).append(phase)
    return dict(admitted_ids=sorted(ids), names=names,
                aliases={k: sorted(v) for k, v in aliases.items() if len(v) > 1},
                phase_support=support, admission='official_RS_or_PS_GP_gt_0; no_MIN_or_POSS_threshold')


def build_identity(sheets=SHEETS, years=range(2000, 2027)):
    result = dict(schema_version=1, status='official_annual_identity_packet_ready', seasons={}, source_hashes={},
                  identity_metadata_only=True, performance_fields_exported=False, protected_outcomes=0)
    for year in years:
        if not 1997 <= year <= 2026:
            raise ValueError('Only completed historical identity seasons admitted')
        frames=[]
        for suffix in ['', '_ps']:
            p = Path(sheets) / f'{year}{suffix}.csv'
            frames.append(pd.read_csv(p, usecols=['PLAYER_ID', 'PLAYER_NAME', 'GP', 'MIN']))
            result['source_hashes'][str(p.resolve())] = sha(p)
        result['seasons'][str(year)] = season_identity(*frames)
    return result


def complete_rows(a):
    return (np.isfinite(a).all(axis=1) & (a > 0).all(axis=1)
            & (np.diff(np.sort(a[:, :5], axis=1), axis=1) > 0).all(axis=1)
            & (np.diff(np.sort(a[:, 5:], axis=1), axis=1) > 0).all(axis=1)
            & (a[:, :5, None] != a[:, None, 5:]).all(axis=(1, 2)))


def census(packet, root=OUT):
    root = Path(root); files=[]; invalid=[]; all_names={}; appearances={}
    for year, rec in packet['seasons'].items():
        all_names.update(rec['names'])
        for pid in rec['admitted_ids']:
            appearances.setdefault(pid, []).append(int(year))
    for year, rec in packet['seasons'].items():
        allow = np.asarray(rec['admitted_ids'], int)
        for phase in ['RS', 'PS']:
            p = root / f'atoms_{year}_{phase}.parquet'
            meta = json.loads((root / f'historical_{year}_{phase}.json').read_text())
            if sha(p) != meta['artifacts'][p.name]:
                raise ValueError('Historical target changed during identity census')
            d = pd.read_parquet(p, columns=['game_id', 'event_num', 'game_date', *SLOTS])
            d = d.loc[~pd.to_numeric(d.game_id).isin(meta['excluded_game_ids'])].reset_index(drop=True)
            a = d[SLOTS].to_numpy(); complete = complete_rows(a)
            valid = np.isin(a, allow).all(axis=1)
            retained = complete & valid
            for pid in np.unique(a[complete & ~valid]):
                if int(pid) in set(allow):
                    continue
                mask = complete & (a == pid).any(axis=1)
                invalid.append(dict(year=int(year), phase=phase, player_id=int(pid),
                                    name_in_other_official_season=all_names.get(str(int(pid))),
                                    official_seasons=appearances.get(int(pid), []), rows=int(mask.sum()),
                                    games=int(d.loc[mask].game_id.nunique()),
                                    first_game=str(pd.to_datetime(d.loc[mask].game_date).min().date()),
                                    last_game=str(pd.to_datetime(d.loc[mask].game_date).max().date())))
            removed_games=sorted(set(pd.to_numeric(d.game_id)) - set(pd.to_numeric(d.loc[retained].game_id)))
            files.append(dict(year=int(year), phase=phase, scoring_admitted_rows=len(d),
                              complete_positive_5v5_rows=int(complete.sum()),
                              complete_but_unadmitted_identity_rows=int((complete & ~valid).sum()),
                              retained_rows=int(retained.sum()), scoring_admitted_games=int(d.game_id.nunique()),
                              retained_games=int(d.loc[retained].game_id.nunique()),
                              whole_games_removed_by_identity=removed_games))
    packet['census'] = dict(files=files, invalid_id_years=invalid,
                           scoring_admitted_rows=sum(r['scoring_admitted_rows'] for r in files),
                           complete_positive_5v5_rows=sum(r['complete_positive_5v5_rows'] for r in files),
                           identity_rejected_complete_rows=sum(r['complete_but_unadmitted_identity_rows'] for r in files),
                           retained_rows=sum(r['retained_rows'] for r in files),
                           whole_games_removed=sum(len(r['whole_games_removed_by_identity']) for r in files))
    packet['status']='passed'
    return packet


def save(p, value):
    p = Path(p); tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n');tmp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--sheets',type=Path,default=SHEETS);a=p.parse_args()
    packet=build_identity(a.sheets);dest=a.root/'identity_admission.json';save(dest,packet)
    packet=census(packet,a.root);packet['code_sha256']=sha(__file__);save(dest,packet)
    print(json.dumps({k:v for k,v in packet['census'].items() if k not in ['files','invalid_id_years']},indent=2))
    print(json.dumps(packet['census']['invalid_id_years'],indent=2))


if __name__=='__main__':main()

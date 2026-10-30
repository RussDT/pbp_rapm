#!/usr/bin/env python3
"""Validate and publish the full no-prior TD550 Net SQ rolling archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from publish_no_prior_vpm_snapshots import _with_transport_retry

ROOT = Path(__file__).resolve().parents[1]
TABLE = 'nba_no_prior_vpm_td550_rolling'
SOURCE = ROOT / 'master_results/net_sq_decomp_rolling'
OUTPUT = ROOT / 'results/td550_rolling_publication'


def expected_windows(max_end_year):
    """Every rolling 1Y-5Y window from 1997 through max_end_year."""
    return {(start, end) for end in range(1997, max_end_year + 1)
            for start in range(max(1997, end-4), end+1)}


def build_frame(max_end_year=None):
    frames, receipts = [], []
    seen = set()
    alpha_contract = None
    for path in sorted(SOURCE.glob('weighted_factors_vpm_net_sq_resid_*_td550.csv')):
        frame = pd.read_csv(path)
        parquet = pd.read_parquet(path.with_suffix('.parquet'))
        pd.testing.assert_frame_equal(frame, parquet, check_dtype=False, atol=1e-10, rtol=1e-10)
        start, end = int(frame.start_year.iloc[0]), int(frame.end_year.iloc[0])
        assert (start, end) not in seen and set(frame.start_year)=={start} and set(frame.end_year)=={end}
        seen.add((start,end))
        compact = f'{start%100:02d}' if start==end else f'{start%100:02d}_{end%100:02d}'
        manifest = json.loads((ROOT/'results/net_sq_decomp_rolling/net_sq_resid_vpmalpha_td550'/compact/'manifest.json').read_text())
        alphas = manifest['component_alphas']
        if alpha_contract is None: alpha_contract = alphas
        assert alphas == alpha_contract
        assert manifest['half_life_days']==550 and manifest['timedecay']
        assert manifest['alpha_profile']=='vpm'
        assert manifest['rubberband']=='smooth_clock_aware_alpha100'
        frame.columns = frame.columns.str.lower()
        for side in ['o','d']:
            cols = [c for c in frame if c.startswith(side+'_')]
            total = frame[cols].sum(axis=1)
            assert np.allclose(total, frame['estimate_off' if side=='o' else 'estimate_def'],atol=0.00100001,rtol=0)
            assert np.allclose(frame[side+'ts']+frame[side+'tov']+frame[side+'sc'],total,atol=0.00100001,rtol=0)
        assert np.allclose(frame.estimate_net,frame.estimate_off+frame.estimate_def,atol=0.00100001,rtol=0)
        assert not frame.select_dtypes('number').isna().any().any()
        frame['window_years'] = end-start+1
        frame['prior_scale'] = 0.0
        frame['status'] = 'research-only'
        frame['season_type'] = 'ALL'
        frame['rubberband_model'] = manifest['rubberband']
        frame['season_effects'] = manifest['season_effects']
        frame['defense_sign'] = 'positive_is_good'
        frame['component_alphas'] = [alphas]*len(frame)
        frame['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames.append(frame)
        receipts.append({'start_year':start,'end_year':end,'rows':len(frame),'source':str(path),'sha256':frame.source_sha256.iloc[0]})
    # The newest end year is read off the archive rather than hardcoded, so a
    # season rollover does not need a code edit; completeness is still exact,
    # so a partially computed newest season fails instead of publishing half of
    # it.
    assert seen, f'No td550 window CSVs under {SOURCE}'
    if max_end_year is None:
        max_end_year = max(end for _, end in seen)
    expected = expected_windows(max_end_year)
    assert seen == expected, (expected-seen,seen-expected)
    result = pd.concat(frames,ignore_index=True)
    assert not result.duplicated(['player_id','start_year','end_year']).any()
    return result, receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--max-end-year',type=int,default=None,
                        help='Pin the newest end year the archive must cover '
                             '(default: the newest one present on disk).')
    args=parser.parse_args()
    frame, receipts=build_frame(args.max_end_year)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(OUTPUT/'rows.parquet',index=False)
    (OUTPUT/'sources.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(f'Validated {len(frame)} rows across {len(receipts)} windows',flush=True)
    if not args.publish: return
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv(ROOT.parent/'.env')
    client=create_client(os.environ['SUPABASE_URL'],os.environ['SUPABASE_SERVICE_ROLE_KEY'])
    # Standard JSON retains full float precision during upload and verification.
    rows=json.loads(json.dumps(frame.to_dict(orient='records'),allow_nan=False))
    for offset in range(0,len(rows),1000):
        batch=rows[offset:offset+1000]
        _with_transport_retry(lambda:client.table(TABLE).upsert(batch,on_conflict='player_id,start_year,end_year').execute(),f'archive {offset}')
        print(f'Uploaded {min(offset+1000,len(rows))}/{len(rows)}',flush=True)
    count=client.table(TABLE).select('player_id',count='exact',head=True).execute().count
    assert count==len(frame),(count,len(frame))
    # Read every window back and verify every published value, including provenance.
    for source in receipts:
        remote=[]
        for offset in range(0,source['rows'],1000):
            response=_with_transport_retry(lambda:client.table(TABLE).select('*').eq('start_year',source['start_year']).eq('end_year',source['end_year']).order('player_id').range(offset,offset+999).execute(),'archive verification')
            remote.extend(response.data)
        local=frame[(frame.start_year==source['start_year']) & (frame.end_year==source['end_year'])].sort_values('player_id').reset_index(drop=True)
        actual=pd.DataFrame(remote).sort_values('player_id').reset_index(drop=True)[local.columns]
        pd.testing.assert_frame_equal(local,actual,check_dtype=False,atol=1e-10,rtol=1e-10)
    (OUTPUT/'verification.json').write_text(json.dumps({'table':TABLE,'rows':count,'windows':len(receipts),'all_values_verified':True},indent=2)+'\n')
    print('All remote rows verified against local archive',flush=True)

if __name__=='__main__': main()

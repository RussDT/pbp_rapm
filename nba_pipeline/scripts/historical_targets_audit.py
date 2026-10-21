"""Rebuild historical Net targets independently from a freshly dated raw prefix."""
from pathlib import Path
import argparse
import json
import tempfile
import numpy as np
import pandas as pd
from historical_targets import OUT, ROOT, sha, load_before_cut
from audit_vpm_midrange_reference import compare, COLUMNS
from process_rapm_blocks import clean_chance_utils as clean
from process_rapm_blocks.common import _finalize_df


def canonical(path, year):
    old = clean.fetch_player_stats_supabase
    try:
        clean.fetch_player_stats_supabase = lambda *a: None
        d = clean.build_clean_chance_base_py(str(path), year % 100, f'{year-1}-{year%100:02d}',
                                           'HISTORICAL_TARGET_INDEPENDENT_AUDIT')
    finally:
        clean.fetch_player_stats_supabase = old
    return _finalize_df(d, ['Actual_Diff', 'First_Chance_Diff', 'Is_Turnover',
                           'SC_NET_CONT_Diff', 'FC_CONT_KAPPA_Diff', *COLUMNS], year % 100)


def run(year, phase, cut, root=OUT):
    root = Path(root)
    d, g = load_before_cut(root, cut, [year], phases=[phase], sq=False)
    raw = ROOT / f'raw_data/NBA{year%100:02d}{"_PS" if phase=="PS" else ""}.parquet'
    source = pd.read_parquet(raw)
    source = source.loc[pd.to_numeric(source.game_id).isin(set(pd.to_numeric(d.game_id)))]
    with tempfile.TemporaryDirectory(prefix='historical-vpm-target-audit-') as td:
        path = Path(td) / raw.name
        source.to_parquet(path, index=False)
        r = canonical(path, year)
        unit = compare(d, g, r, np.ones(len(d)))
        w = np.exp2(-(d.game_date.max() - d.game_date).dt.total_seconds().to_numpy() / (86400 * 550))
        weighted = compare(d, g, r, w)
    result = dict(status='passed', year=year, phase=phase, cut=cut,
                  admitted_games=int(d.game_id.nunique()), fresh_raw_prefix=unit,
                  fresh_raw_prefix_TD550=weighted, actual_FT=True,
                  scoring_quarantine_applied_to_raw_and_pricing=True,
                  source_hashes={str(p): sha(p) for p in [raw, Path(__file__), Path(clean.__file__)]},
                  no_model_refit=True, protected_outcomes=0)
    output = root / f'independent_target_audit_{year}_{phase}_{cut}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', type=int, default=2000)
    p.add_argument('--phase', choices=['RS', 'PS'], default='PS')
    p.add_argument('--cut', default='2000-06-01')
    p.add_argument('--root', type=Path, default=OUT)
    a = p.parse_args(); run(a.year, a.phase, a.cut, a.root)

#!/usr/bin/env python3
"""
Calculate opponent quality (DPM) faced by a player from RAPM possession data.

Usage:
    python opponent_quality.py "Nikola Jokic"
    python opponent_quality.py 203999
    python opponent_quality.py "Nikola Jokic" --year 26
    python opponent_quality.py "Nikola Jokic" --top 50
    python opponent_quality.py "Nikola Jokic" --csv output.csv
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client


def find_player_id(query, autocomplete_path):
    """Resolve a player name or ID to an nba_id."""
    ac = pd.read_csv(autocomplete_path)
    # Try numeric ID first
    try:
        pid = int(float(query))
        if pid in ac['nba_id'].values:
            name = ac.loc[ac['nba_id'] == pid, 'player_name'].iloc[0]
            return pid, name
        return pid, f"ID {pid}"
    except ValueError:
        pass

    # Fuzzy name match
    matches = ac[ac['player_name'].str.contains(query, case=False, na=False)]
    if len(matches) == 0:
        print(f"No player found matching '{query}'")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Multiple matches for '{query}':")
        for _, row in matches.iterrows():
            print(f"  {row['player_name']} ({int(row['nba_id'])})")
        print("Please be more specific or use the player ID.")
        sys.exit(1)
    return int(matches.iloc[0]['nba_id']), matches.iloc[0]['player_name']


def get_opponent_possessions(df, player_id):
    """Get opponent possession counts and per-possession lineup data."""
    pid = float(player_id)
    o_cols = ['O1', 'O2', 'O3', 'O4', 'O5']
    d_cols = ['D1', 'D2', 'D3', 'D4', 'D5']

    off_mask = (df[o_cols] == pid).any(axis=1)
    def_mask = (df[d_cols] == pid).any(axis=1)

    # Individual opponent counts
    opp_counter = Counter()

    # Per-possession lineup sums (for lineup-level avg)
    lineup_dpms = []  # will be filled after DPM merge

    # When player on offense, opponents are D1-D5
    off_rows = df.loc[off_mask, d_cols]
    for _, row in off_rows.iterrows():
        for col in d_cols:
            opp_counter[int(row[col])] += 1

    # When player on defense, opponents are O1-O5
    def_rows = df.loc[def_mask, o_cols]
    for _, row in def_rows.iterrows():
        for col in o_cols:
            opp_counter[int(row[col])] += 1

    n_off = off_mask.sum()
    n_def = def_mask.sum()
    n_total = (off_mask | def_mask).sum()

    # Store raw possession-level opponent lineups for lineup-level calc
    poss_lineups = []
    for _, row in df.loc[off_mask, d_cols].iterrows():
        poss_lineups.append([int(row[c]) for c in d_cols])
    for _, row in df.loc[def_mask, o_cols].iterrows():
        poss_lineups.append([int(row[c]) for c in o_cols])

    return opp_counter, n_off, n_def, n_total, poss_lineups


def fetch_dpm(year, sb):
    """Fetch all DPM values for a year from Supabase in one query."""
    all_rows = []
    offset = 0
    batch_size = 1000
    while True:
        result = (
            sb.table('player_stats_with_metrics_mat')
            .select('nba_id, Name, dpm, o_dpm, d_dpm, td_rapm, td_orapm, td_drapm')
            .eq('year', year)
            .eq('playoffs', 0)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        all_rows.extend(result.data)
        if len(result.data) < batch_size:
            break
        offset += batch_size
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description='Calculate opponent quality (DPM) faced by a player')
    parser.add_argument('player', help='Player name or NBA ID')
    parser.add_argument('--year', type=int, default=26, help='Season year (default: 26)')
    parser.add_argument('--top', type=int, default=30, help='Show top N opponents (default: 30)')
    parser.add_argument('--csv', help='Save full results to CSV')
    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(pipeline_dir)
    autocomplete_path = os.path.join(project_dir, 'autocomplete_map.csv')
    parquet_path = os.path.join(pipeline_dir, 'processed', f'RAPM{args.year}.parquet')

    # Load env
    load_dotenv(os.path.join(project_dir, '.env'))
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    sb = create_client(url, key)

    # Resolve player
    player_id, player_name = find_player_id(args.player, autocomplete_path)
    print(f"\n{'='*60}")
    print(f"  {player_name} ({player_id}) — 20{args.year} Opponent Quality")
    print(f"{'='*60}")

    # Load parquet
    df = pd.read_parquet(parquet_path)

    # Get opponents
    opp_counter, n_off, n_def, n_total, poss_lineups = get_opponent_possessions(df, player_id)
    print(f"\nPossessions: {n_total:,} total ({n_off:,} off, {n_def:,} def)")
    print(f"Unique opponents: {len(opp_counter)}")

    # Fetch all DPM for the year (single query)
    opp_df = pd.DataFrame(opp_counter.most_common(), columns=['player_id', 'possessions'])
    dpm_df = fetch_dpm(2000 + args.year, sb)
    print(f"Loaded DPM for {len(dpm_df)} players")

    opp_df = opp_df.merge(
        dpm_df[['nba_id', 'Name', 'dpm', 'o_dpm', 'd_dpm', 'td_rapm', 'td_orapm', 'td_drapm']],
        left_on='player_id', right_on='nba_id', how='left'
    )

    missing = opp_df[opp_df['dpm'].isna()]
    if len(missing) > 0:
        print(f"Warning: {len(missing)} opponents missing DPM ({missing['possessions'].sum()} poss)")

    valid = opp_df.dropna(subset=['dpm'])

    # --- Individual-level weighted averages ---
    total_poss = valid['possessions'].sum()
    wavg_dpm = (valid['possessions'] * valid['dpm']).sum() / total_poss
    wavg_odpm = (valid['possessions'] * valid['o_dpm']).sum() / total_poss
    wavg_ddpm = (valid['possessions'] * valid['d_dpm']).sum() / total_poss

    valid_td = valid.dropna(subset=['td_rapm'])
    total_poss_td = valid_td['possessions'].sum()
    if total_poss_td > 0:
        wavg_td = (valid_td['possessions'] * valid_td['td_rapm']).sum() / total_poss_td
        wavg_td_o = (valid_td['possessions'] * valid_td['td_orapm']).sum() / total_poss_td
        wavg_td_d = (valid_td['possessions'] * valid_td['td_drapm']).sum() / total_poss_td
    else:
        wavg_td = wavg_td_o = wavg_td_d = 0

    # --- Lineup-level averages (sum of 5 opponents per possession) ---
    dpm_lookup = dict(zip(valid['player_id'], valid['dpm']))
    odpm_lookup = dict(zip(valid['player_id'], valid['o_dpm']))
    ddpm_lookup = dict(zip(valid['player_id'], valid['d_dpm']))
    td_lookup = dict(zip(valid_td['player_id'], valid_td['td_rapm']))
    td_o_lookup = dict(zip(valid_td['player_id'], valid_td['td_orapm']))
    td_d_lookup = dict(zip(valid_td['player_id'], valid_td['td_drapm']))

    lineup_dpms = []
    lineup_odpms = []
    lineup_ddpms = []
    lineup_tds = []
    lineup_td_os = []
    lineup_td_ds = []
    for lineup in poss_lineups:
        vals = [dpm_lookup.get(p) for p in lineup]
        ovals = [odpm_lookup.get(p) for p in lineup]
        dvals = [ddpm_lookup.get(p) for p in lineup]
        if all(v is not None for v in vals):
            lineup_dpms.append(sum(vals))
            lineup_odpms.append(sum(ovals))
            lineup_ddpms.append(sum(dvals))
        td_vals = [td_lookup.get(p) for p in lineup]
        td_ovals = [td_o_lookup.get(p) for p in lineup]
        td_dvals = [td_d_lookup.get(p) for p in lineup]
        if all(v is not None for v in td_vals):
            lineup_tds.append(sum(td_vals))
            lineup_td_os.append(sum(td_ovals))
            lineup_td_ds.append(sum(td_dvals))

    lineup_avg_dpm = np.mean(lineup_dpms) if lineup_dpms else 0
    lineup_avg_odpm = np.mean(lineup_odpms) if lineup_odpms else 0
    lineup_avg_ddpm = np.mean(lineup_ddpms) if lineup_ddpms else 0
    lineup_avg_td = np.mean(lineup_tds) if lineup_tds else 0
    lineup_avg_td_o = np.mean(lineup_td_os) if lineup_td_os else 0
    lineup_avg_td_d = np.mean(lineup_td_ds) if lineup_td_ds else 0

    print(f"\n--- Individual Opponent Averages (possession-weighted) ---")
    print(f"                    DPM      TD-RAPM")
    print(f"  Overall:        {wavg_dpm:+.2f}      {wavg_td:+.2f}")
    print(f"  Offensive:      {wavg_odpm:+.2f}      {wavg_td_o:+.2f}")
    print(f"  Defensive:      {wavg_ddpm:+.2f}      {wavg_td_d:+.2f}")

    print(f"\n--- Lineup-Level Averages (sum of 5 opponents per possession) ---")
    print(f"                    DPM      TD-RAPM")
    print(f"  Overall:        {lineup_avg_dpm:+.2f}      {lineup_avg_td:+.2f}   ({len(lineup_dpms):,} poss)")
    print(f"  Offensive:      {lineup_avg_odpm:+.2f}      {lineup_avg_td_o:+.2f}")
    print(f"  Defensive:      {lineup_avg_ddpm:+.2f}      {lineup_avg_td_d:+.2f}")

    # Top opponents table
    valid_sorted = valid.sort_values('possessions', ascending=False)
    print(f"\nTop {args.top} opponents by possessions:")
    display_cols = ['Name', 'possessions', 'dpm', 'o_dpm', 'd_dpm', 'td_rapm', 'td_orapm', 'td_drapm']
    display = valid_sorted[display_cols].head(args.top).copy()
    for col in ['dpm', 'o_dpm', 'd_dpm', 'td_rapm', 'td_orapm', 'td_drapm']:
        display[col] = display[col].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "  N/A")
    print(display.to_string(index=False))

    # Save CSV
    if args.csv:
        valid_sorted.to_csv(args.csv, index=False)
        print(f"\nFull data saved to {args.csv}")


if __name__ == '__main__':
    main()

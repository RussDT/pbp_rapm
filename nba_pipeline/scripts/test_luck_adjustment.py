"""
Test script for analyzing the Luck Adjustment implementation in RAPM processing.

Compares:
- LA = 0%: Pure actual scoring (no luck adjustment)
- LA = Current: Your implemented blends (FT=100%, 3PT=20%/40%)
- LA = 100%: Full luck adjustment

Key metric: Average points per possession (Off_Diff, Def_Diff)
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"


def analyze_processed_rapm(rapm_file: str):
    """
    Analyze an already-processed RAPM file to understand the luck adjustment impact.
    Shows distribution of Off_Diff and Def_Diff values.
    """
    print("=" * 70)
    print(f"ANALYZING: {rapm_file}")
    print("=" * 70)
    
    df = pd.read_csv(rapm_file)
    print(f"Loaded {len(df):,} possession rows")
    
    # Basic stats
    print("\n--- Points Per Possession Analysis ---")
    
    for col in ['Net_Diff', 'Off_Diff', 'Def_Diff']:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"\n{col}:")
            print(f"  Mean:   {vals.mean():.4f}")
            print(f"  Median: {vals.median():.4f}")
            print(f"  Std:    {vals.std():.4f}")
            print(f"  Min:    {vals.min():.4f}")
            print(f"  Max:    {vals.max():.4f}")
            print(f"  Total:  {vals.sum():,.2f}")
    
    # Distribution of values
    print("\n--- Off_Diff Distribution ---")
    if 'Off_Diff' in df.columns:
        bins = [-1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 10]
        labels = ['<0', '0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-4', '4+']
        df['Off_Diff_Bin'] = pd.cut(df['Off_Diff'], bins=bins, labels=labels)
        dist = df['Off_Diff_Bin'].value_counts().sort_index()
        for bin_label, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {bin_label:>8}: {count:>8,} ({pct:5.1f}%)")
    
    return df


def compare_la_versions(base_file: str, year: int):
    """
    Run RAPM processing with different luck adjustment weights and compare.
    
    This modifies the processing to output:
    1. RAPM (current LA settings)  
    2. RAPM_NoLA (0% luck adjustment - pure actual)
    """
    print("=" * 70)
    print("COMPARING LUCK ADJUSTMENT VERSIONS")
    print("=" * 70)
    
    # Import the processing module
    import importlib.util
    spec = importlib.util.spec_from_file_location("process_rapm", SCRIPT_DIR / "02_process_rapm.py")
    proc_module = importlib.util.module_from_spec(spec)
    sys.modules["process_rapm"] = proc_module
    spec.loader.exec_module(proc_module)
    
    print(f"\n1. Base processing: {base_file}")
    nba_df = proc_module._base_processing(base_file)
    
    if nba_df is None:
        print("ERROR: Base processing failed!")
        return None
    
    print(f"   Loaded {len(nba_df):,} rows")
    
    # Helper functions from the module
    series_contains = proc_module.series_contains
    case_when = proc_module.case_when
    
    # Identify event types
    is_ft_event = series_contains(nba_df['home_description'], "Free Throw", case=False) | \
                  series_contains(nba_df['visitor_description'], "Free Throw", case=False) | \
                  (nba_df['event_type'] == "FreeThrow")
    
    is_2pt_make = (series_contains(nba_df['home_description'], "PTS", case=False) & 
                   ~series_contains(nba_df['home_description'], "3PT", case=False) & 
                   (nba_df['event_type'] == "MAKE")) | \
                  (series_contains(nba_df['visitor_description'], "PTS", case=False) & 
                   ~series_contains(nba_df['visitor_description'], "3PT", case=False) & 
                   (nba_df['event_type'] == "MAKE"))
    
    is_3pt_event = series_contains(nba_df['home_description'], "3PT", case=False) | \
                   series_contains(nba_df['visitor_description'], "3PT", case=False)
    
    actual_score = (nba_df['Home_Action_Score'] + nba_df['Away_Action_Score']).astype(float)
    
    # Count events
    n_ft = is_ft_event.sum()
    n_3pt = is_3pt_event.sum()
    n_2pt = is_2pt_make.sum()
    
    print(f"\n2. Event counts:")
    print(f"   FT events:  {n_ft:,}")
    print(f"   3PT events: {n_3pt:,}")
    print(f"   2PT makes:  {n_2pt:,}")
    
    # League averages for expected values
    default_ft_perc = 0.77
    default_3p_perc = 0.36
    
    # Calculate totals under different LA schemes
    print("\n3. Calculating point totals under different LA schemes...")
    
    # Scheme 1: No LA (0% expected, 100% actual)
    off_net_no_la = case_when(
        is_ft_event, actual_score,  # Use actual FT results
        is_2pt_make, 2.0,
        is_3pt_event, actual_score,  # Use actual 3PT results
        0.0
    )
    
    # Scheme 2: Current LA (FT=100%, 3PT=20%/40%)
    off_net_current = case_when(
        is_ft_event, default_ft_perc,  # 100% expected
        is_2pt_make, 2.0,
        is_3pt_event, (default_3p_perc * 3.0) * 0.2 + 0.8 * actual_score,
        0.0
    )
    
    def_net_current = case_when(
        is_ft_event, default_ft_perc,  # 100% expected
        is_2pt_make, 2.0,
        is_3pt_event, (default_3p_perc * 3.0) * 0.4 + 0.6 * actual_score,
        0.0
    )
    
    # Scheme 3: Full LA (100% expected for everything)
    exp_3pt = default_3p_perc * 3.0
    off_net_full_la = case_when(
        is_ft_event, default_ft_perc,
        is_2pt_make, 2.0,  # 2PT makes still get 2 (we'd need 2PT% for full adjustment)
        is_3pt_event, exp_3pt,  # 100% expected
        0.0
    )
    
    # Sum totals
    total_actual = actual_score.sum()
    total_no_la = off_net_no_la.sum()
    total_current_off = off_net_current.sum()
    total_current_def = def_net_current.sum()
    total_full_la = off_net_full_la.sum()
    
    print("\n" + "=" * 70)
    print("RESULTS: Total Points Under Different LA Schemes")
    print("=" * 70)
    print(f"\n{'Scheme':<30} {'Total Points':>15} {'vs Actual':>12}")
    print("-" * 60)
    print(f"{'Actual (raw)':<30} {total_actual:>15,.0f} {'+0.0%':>12}")
    print(f"{'No LA (0%)':<30} {total_no_la:>15,.0f} {(total_no_la/total_actual-1)*100:>+11.2f}%")
    print(f"{'Current LA (Off)':<30} {total_current_off:>15,.0f} {(total_current_off/total_actual-1)*100:>+11.2f}%")
    print(f"{'Current LA (Def)':<30} {total_current_def:>15,.0f} {(total_current_def/total_actual-1)*100:>+11.2f}%")
    print(f"{'Full LA (100%)':<30} {total_full_la:>15,.0f} {(total_full_la/total_actual-1)*100:>+11.2f}%")
    
    # Breakdown by shot type
    print("\n" + "=" * 70)
    print("BREAKDOWN BY SHOT TYPE")
    print("=" * 70)
    
    # FT breakdown
    ft_actual = actual_score[is_ft_event].sum()
    ft_la = (is_ft_event.sum() * default_ft_perc)
    print(f"\nFree Throws:")
    print(f"  Actual total:    {ft_actual:>10,.0f}")
    print(f"  LA total:        {ft_la:>10,.0f}")
    print(f"  Difference:      {ft_la - ft_actual:>+10,.0f} ({(ft_la/ft_actual-1)*100 if ft_actual > 0 else 0:+.1f}%)")
    
    # 3PT breakdown  
    pt3_actual = actual_score[is_3pt_event].sum()
    pt3_la_off = ((default_3p_perc * 3.0) * 0.2 + 0.8 * actual_score[is_3pt_event]).sum()
    pt3_la_def = ((default_3p_perc * 3.0) * 0.4 + 0.6 * actual_score[is_3pt_event]).sum()
    pt3_la_full = is_3pt_event.sum() * exp_3pt
    
    print(f"\n3-Pointers:")
    print(f"  Actual total:    {pt3_actual:>10,.0f}")
    print(f"  LA Off (20%):    {pt3_la_off:>10,.0f} ({(pt3_la_off/pt3_actual-1)*100 if pt3_actual > 0 else 0:+.1f}%)")
    print(f"  LA Def (40%):    {pt3_la_def:>10,.0f} ({(pt3_la_def/pt3_actual-1)*100 if pt3_actual > 0 else 0:+.1f}%)")
    print(f"  LA Full (100%):  {pt3_la_full:>10,.0f} ({(pt3_la_full/pt3_actual-1)*100 if pt3_actual > 0 else 0:+.1f}%)")
    
    # 2PT breakdown
    pt2_actual = actual_score[is_2pt_make].sum()
    print(f"\n2-Pointers (makes only - no LA applied):")
    print(f"  Actual total:    {pt2_actual:>10,.0f}")
    
    # Per-possession stats (approximate)
    print("\n" + "=" * 70)
    print("AVERAGE POINTS PER SCORING EVENT")
    print("=" * 70)
    
    n_events = (is_ft_event | is_3pt_event | is_2pt_make).sum()
    print(f"\nTotal scoring events: {n_events:,}")
    print(f"\n{'Scheme':<30} {'Avg Pts/Event':>15}")
    print("-" * 50)
    print(f"{'Actual':<30} {total_actual/n_events:>15.4f}")
    print(f"{'No LA':<30} {total_no_la/n_events:>15.4f}")
    print(f"{'Current LA (Off)':<30} {total_current_off/n_events:>15.4f}")
    print(f"{'Current LA (Def)':<30} {total_current_def/n_events:>15.4f}")
    print(f"{'Full LA':<30} {total_full_la/n_events:>15.4f}")
    
    return {
        "n_events": n_events,
        "total_actual": total_actual,
        "total_no_la": total_no_la,
        "total_current_off": total_current_off,
        "total_current_def": total_current_def,
        "total_full_la": total_full_la,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test luck adjustment in RAPM processing')
    parser.add_argument('input_file', nargs='?', 
                       help='Path to raw input CSV (e.g., raw_data/NBA26.csv) or processed RAPM file')
    parser.add_argument('--year', '-y', type=int, default=2025,
                       help='Season starting year')
    parser.add_argument('--analyze-processed', '-a', action='store_true',
                       help='Analyze an already-processed RAPM file instead of running comparison')
    
    args = parser.parse_args()
    
    if args.input_file is None:
        # Default: try to find a recent RAPM file
        rapm_files = list(PROCESSED_DIR.glob("RAPM*.csv"))
        if rapm_files:
            latest = max(rapm_files, key=lambda p: p.stat().st_mtime)
            print(f"No input specified. Analyzing most recent: {latest}")
            analyze_processed_rapm(str(latest))
        else:
            print("No input file specified and no RAPM files found in processed/")
            print(f"Usage: python {__file__} <input_file>")
            print(f"  For raw data:       python {__file__} ../raw_data/NBA26.csv")
            print(f"  For processed:      python {__file__} -a ../processed/RAPM26.csv")
    elif args.analyze_processed:
        analyze_processed_rapm(args.input_file)
    else:
        # Resolve path
        if not os.path.isabs(args.input_file):
            input_path = SCRIPT_DIR / args.input_file
        else:
            input_path = Path(args.input_file)
        
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            sys.exit(1)
        
        compare_la_versions(str(input_path), args.year)

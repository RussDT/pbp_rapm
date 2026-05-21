#!/usr/bin/env python3
"""
NBA RAPM Data Processing - Main Orchestrator

This script orchestrates the processing of NBA play-by-play data for various RAPM metrics.
Individual processors are modularized in the process_rapm_blocks/ package.

Usage (from nba_pipeline/scripts/):
    python 02_process_rapm.py ../raw_data/NBA26.parquet
    python 02_process_rapm.py ../raw_data/NBA26_PS.parquet

    # With luck adjustment parameters:
    python 02_process_rapm.py ../raw_data/NBA26.parquet --o-luck 1.0 --d-luck 1.0
"""

import argparse
import os
import re

from process_rapm_blocks import (
    PROCESSED_DIR,
    process_rapm_py,
    process_ts_py,
    process_tov_py,
    process_reb_py,
    process_shooter_oreb_py,
    process_rim_freq_py,
    process_rim_fg_pct_py,
    process_midrange_freq_py,
    process_midrange_fg_pct_py,
    process_transition_freq_py,
    process_transition_rim_py,
    process_initial_ev_py,
    process_initial_ev_beta_py,
    process_special_rapm_py,
    process_ev_rapm_py,
    process_sq_poss_py,
    process_ft_premium_py,
    process_contest_py,
    process_second_chance_py,
    process_first_chance_py,
    process_first_chance_clean_py,
    process_second_chance_clean_py,
    process_assist_points_py,
    process_rim_assist_py,
    process_three_freq_py,
    process_three_fg_pct_py,
    process_playtype_ts_mix_py,
    process_playtype_proxy_pts_py,
)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process NBA play-by-play data for RAPM, TS, REB, TOV, and shot metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_process_rapm.py ../raw_data/NBA26.parquet           # Auto-detects 2025-26 season
  python 02_process_rapm.py ../raw_data/NBA23_PS.parquet        # Auto-detects 2022-23 playoffs

  # With luck adjustment parameters:
  python 02_process_rapm.py ../raw_data/NBA26.parquet --o-luck 1.0 --d-luck 1.0   # Full LA for both
  python 02_process_rapm.py ../raw_data/NBA26.parquet --o-luck 0.5 --d-luck 1.0   # 50% off, 100% def
  python 02_process_rapm.py ../raw_data/NBA26.parquet --o-luck 0.0 --d-luck 0.0   # No LA (pure actuals)
        """
    )

    parser.add_argument('input_file',
                       help='Path to the input parquet file (e.g., ../raw_data/NBA26.parquet)')
    parser.add_argument('--year', '-y',
                       type=int,
                       required=False,
                       help='(Optional) Season end year - auto-inferred from filename if not provided')
    parser.add_argument('--season', '-s',
                       required=False,
                       help='(Optional) Season string in format YYYY-YY - auto-inferred from filename if not provided')
    parser.add_argument('--o-luck',
                       type=float,
                       default=1.0,
                       help='Offensive luck adjustment (0.0=no adjustment, 1.0=full adjustment). Default: 1.0')
    parser.add_argument('--d-luck',
                       type=float,
                       default=1.0,
                       help='Defensive luck adjustment (0.0=no adjustment, 1.0=full adjustment). Default: 1.0')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file '{args.input_file}' not found. Please provide the correct path.")
        exit(1)

    # Infer end year and season from filename if not provided
    if args.year is None or args.season is None:
        filename = os.path.basename(args.input_file)
        year_match = re.search(r'NBA(\d{2,4})(?:_PS)?\.parquet', filename, re.IGNORECASE)

        if year_match:
            year_str = year_match.group(1)
            if len(year_str) == 2:
                ending_year = 2000 + int(year_str)
            else:
                ending_year = int(year_str)
            starting_year = ending_year - 1

            if args.year is None:
                args.year = ending_year

            if args.season is None:
                end_year_short = ending_year % 100
                args.season = f"{starting_year}-{end_year_short:02d}"

            print(f"Auto-detected from filename: Year={args.year}, Season={args.season}")
        else:
            print(f"ERROR: Could not auto-detect year/season from filename '{filename}'.")
            print(f"Expected format: NBAXX.parquet or NBAXX_PS.parquet (e.g., NBA26.parquet, NBA23_PS.parquet)")
            print(f"Or provide --year and --season arguments explicitly.")
            exit(1)

    file_to_process = args.input_file
    processing_year = args.year
    processing_season = args.season
    o_luck = args.o_luck
    d_luck = args.d_luck

    print(f"\n--- Starting All Processing for {processing_season} using {file_to_process} ---")
    print(f"--- End Year: {processing_year}, Season: {processing_season} ---")
    print(f"--- Luck Adjustment: o_luck={o_luck}, d_luck={d_luck} ---\n")

    # --- Run Each Process ---
    # RAPM returns a tuple: (no_la_df, la_df)
    rapm_result = process_rapm_py(file_to_process, processing_year, processing_season, o_luck=o_luck, d_luck=d_luck)
    if rapm_result is not None:
        rapm_no_la_df, rapm_la_df = rapm_result
    else:
        rapm_no_la_df, rapm_la_df = None, None

    ts_df = process_ts_py(file_to_process, processing_year, processing_season)
    reb_df = process_reb_py(file_to_process, processing_year, processing_season)
    shooter_oreb_df = process_shooter_oreb_py(file_to_process, processing_year, processing_season)
    tov_df = process_tov_py(file_to_process, processing_year, processing_season)
    rim_freq_df = process_rim_freq_py(file_to_process, processing_year, processing_season)
    rim_fg_pct_df = process_rim_fg_pct_py(file_to_process, processing_year, processing_season)
    midrange_freq_df = process_midrange_freq_py(file_to_process, processing_year, processing_season)
    midrange_fg_pct_df = process_midrange_fg_pct_py(file_to_process, processing_year, processing_season)

    # New metrics. Most ShotQuality-dependent metrics are only available for years 24-26.
    # FT_PREMIUM is historical because it only depends on FT events and FT%.
    transition_freq_df = process_transition_freq_py(file_to_process, processing_year, processing_season)
    transition_rim_df = process_transition_rim_py(file_to_process, processing_year, processing_season)
    initial_ev_df = process_initial_ev_py(file_to_process, processing_year, processing_season)
    initial_ev_beta_df = process_initial_ev_beta_py(file_to_process, processing_year, processing_season)
    special_rapm_df = process_special_rapm_py(file_to_process, processing_year, processing_season)
    ev_rapm_df = process_ev_rapm_py(file_to_process, processing_year, processing_season)
    sq_poss_df = process_sq_poss_py(file_to_process, processing_year, processing_season)
    ft_premium_df = process_ft_premium_py(file_to_process, processing_year, processing_season)
    contest_df = process_contest_py(file_to_process, processing_year, processing_season)
    second_chance_df = process_second_chance_py(file_to_process, processing_year, processing_season)
    first_chance_df = process_first_chance_py(file_to_process, processing_year, processing_season)
    first_chance_clean_df = process_first_chance_clean_py(file_to_process, processing_year, processing_season)
    second_chance_clean_df = process_second_chance_clean_py(file_to_process, processing_year, processing_season)
    assist_points_df = process_assist_points_py(file_to_process, processing_year, processing_season)
    rim_assist_df = process_rim_assist_py(file_to_process, processing_year, processing_season)
    three_freq_df = process_three_freq_py(file_to_process, processing_year, processing_season)
    three_fg_pct_df = process_three_fg_pct_py(file_to_process, processing_year, processing_season)
    playtype_ts_mix_df = process_playtype_ts_mix_py(file_to_process, processing_year, processing_season)
    playtype_proxy_pts_df = process_playtype_proxy_pts_py(file_to_process, processing_year, processing_season)

    # --- Save Results ---
    # Ensure Processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if input file has _PS suffix
    input_basename = os.path.basename(file_to_process)
    input_name_without_ext = os.path.splitext(input_basename)[0]
    ps_suffix = "_PS" if input_name_without_ext.endswith("_PS") else ""

    # Get season suffix for filenames
    season_suffix = processing_season.split('-')[1]  # Get '26' from '2025-26'

    # Save RAPM files (both versions)
    if rapm_no_la_df is not None and not rapm_no_la_df.empty:
        output_filename = PROCESSED_DIR / f"RAPM{season_suffix}{ps_suffix}.parquet"
        try:
            rapm_no_la_df.to_parquet(output_filename, index=False)
            print(f"--- Saved RAPM (No LA) to {output_filename} ({len(rapm_no_la_df)} rows) ---")
        except Exception as e:
            print(f"--- Error saving {output_filename}: {e} ---")
    else:
        print(f"--- No results for RAPM (No LA) {processing_season}. ---")

    if rapm_la_df is not None and not rapm_la_df.empty:
        output_filename = PROCESSED_DIR / f"LA_RAPM{season_suffix}{ps_suffix}.parquet"
        try:
            rapm_la_df.to_parquet(output_filename, index=False)
            print(f"--- Saved LA_RAPM (With LA) to {output_filename} ({len(rapm_la_df)} rows) ---")
        except Exception as e:
            print(f"--- Error saving {output_filename}: {e} ---")
    else:
        print(f"--- No results for LA_RAPM {processing_season}. ---")

    # Save other outputs
    other_outputs = {
        "TS": ts_df,
        "REB": reb_df,
        "SHOOTER_OREB": shooter_oreb_df,
        "TOV": tov_df,
        "RIM_FREQ": rim_freq_df,
        "RIM_FG_PCT": rim_fg_pct_df,
        "MIDRANGE_FREQ": midrange_freq_df,
        "MIDRANGE_FG_PCT": midrange_fg_pct_df,
        "TRANSITION_FREQ": transition_freq_df,
        "TRANSITION_RIM": transition_rim_df,
        "INITIAL_EV": initial_ev_df,
        "INITIAL_EV_BETA": initial_ev_beta_df,
        "SPECIAL_RAPM": special_rapm_df,
        "EV_RAPM": ev_rapm_df,
        "SQ_POSS": sq_poss_df,
        "FT_PREMIUM": ft_premium_df,
        "CONTEST": contest_df,
        "SECOND_CHANCE": second_chance_df,
        "FIRST_CHANCE": first_chance_df,
        "FIRST_CHANCE_CLEAN": first_chance_clean_df,
        "SECOND_CHANCE_CLEAN": second_chance_clean_df,
        "ASSIST_POINTS": assist_points_df,
        "RIM_ASSIST": rim_assist_df,
        "THREE_FREQ": three_freq_df,
        "THREE_FG_PCT": three_fg_pct_df,
        "PLAYTYPE_TS_MIX": playtype_ts_mix_df,
        "PLAYTYPE_PROXY_PTS": playtype_proxy_pts_df,
    }

    for name, df in other_outputs.items():
        if df is not None and not df.empty:
            output_filename = PROCESSED_DIR / f"{name}{season_suffix}{ps_suffix}.parquet"
            try:
                df.to_parquet(output_filename, index=False)
                print(f"--- Saved {name} results to {output_filename} ({len(df)} rows) ---")
            except Exception as e:
                print(f"--- Error saving {output_filename}: {e} ---")
        else:
            print(f"--- No results generated or DataFrame empty for {name} {processing_season}. ---")

    print(f"\n--- Finished All Processing for {processing_season} ---")


if __name__ == '__main__':
    main()

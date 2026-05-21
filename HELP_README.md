# Help Guide: NBA Play-by-Play Data Processing

This guide will help you understand and use the `process_rapm2.py` script to process NBA play-by-play data.

## What Does This Script Do?

The script processes raw NBA play-by-play data and creates structured datasets for basketball analytics. It generates **four different types of analysis files**:

1. **RAPM** (Regularized Adjusted Plus-Minus) - Measures player impact on scoring
2. **TS** (True Shooting) - Measures scoring efficiency 
3. **REB** (Rebounding) - Measures rebounding impact
4. **TOV** (Turnover) - Measures turnover impact

## Quick Start

### Step 1: Install Required Packages

Make sure you have Python 3.x installed, then install the required packages:

```bash
pip install pandas numpy requests nba_api
```

### Step 2: Run the Script

The simplest way to use the script is to provide just the input file. The script will automatically detect the season from the filename:

```bash
python process_rapm2.py NBA26.csv
```

Or if your file has a different name, you can specify the year and season manually:

```bash
python process_rapm2.py your_file.csv --year 2025 --season "2025-26"
```

## Input File Format

Your input CSV file should contain NBA play-by-play data with these key columns:

**Required Columns:**
- `game_id` - Unique identifier for each game
- `event_type` - Numeric code for event type (1=MAKE, 2=MISS, 3=FreeThrow, 4=Rebound, 5=Turnover, etc.)
- `home_description` - Text description of home team action
- `visitor_description` - Text description of visitor team action
- `away_player1` through `away_player5` - Player IDs for away team lineup
- `home_player1` through `home_player5` - Player IDs for home team lineup
- `seconds_remaining_quarter` - Time remaining in the quarter
- `score`, `away_score`, `home_score` - Score information
- `period` - Period/quarter number
- `time_quarter` - Time string (e.g., "12:00")
- `player1_id` - Player ID for the shooter (used for stats lookup)

**Example Input File Structure:**
```
game_id,event_type,home_description,visitor_description,away_player1,...,home_player1,...,score,period,...
0022500001,2,NA,"MISS Sengun 25' 3PT Jump Shot",1630578,...,1631096,...,NA,1,...
0022500001,4,"Wallace REBOUND (Off:0 Def:1)",NA,1630578,...,1631096,...,NA,1,...
```

## Output Files

After running the script, you'll find **4 output files** in the `Processed/` directory:

### 1. RAPM File: `Processed/RAPM{YY}.csv`
Contains possession-level data with:
- `Net_Diff` - Change in raw points scored between possessions
- `Off_Diff` - Change in luck-adjusted offensive points (see note below)
- `Def_Diff` - Change in luck-adjusted defensive points (see note below)
- `O1`-`O5` - Offensive players on court
- `D1`-`D5` - Defensive players on court
- `game_id`, `game_date`, `Season` - Game identifiers
- Game state columns: `score`, `period`, `time_quarter`, etc.

### 2. TS File: `Processed/TS{YY}.csv`
Contains True Shooting analysis with:
- `Net_Diff` - Change in actual points scored (raw, not luck-adjusted)
- Same player and game columns as RAPM

### 3. REB File: `Processed/REB{YY}.csv`
Contains rebounding analysis with:
- `Offensive_Rebound` - Binary flag (1 if offensive rebound, 0 otherwise)
- Same player and game columns

### 4. TOV File: `Processed/TOV{YY}.csv`
Contains turnover analysis with:
- `Is_Turnover` - Binary flag (1 if turnover, 0 otherwise)
- Same player and game columns

**Note:** `{YY}` is the 2-digit ending year (e.g., "26" for 2025-26 season). If processing playoffs, files will have `_PS` suffix (e.g., `RAPM26_PS.csv`).

## Understanding the Output Metrics

### Net_Diff, Off_Diff, Def_Diff Explained

These metrics measure **how much the score changed** between consecutive possessions:

- **Net_Diff**: Raw point difference (actual points scored)
- **Off_Diff**: Luck-adjusted offensive points (attempts to remove shooting luck)
- **Def_Diff**: Luck-adjusted defensive points (attempts to remove shooting luck)

**How it works:**
1. The script calculates cumulative totals (`Net_Total`, `OffTotal`, `DefTotal`) throughout the game
2. At each "End of Possession" event, it calculates the difference from the previous possession
3. This difference (`*_Diff`) represents the impact of that possession

**Example:**
- Possession 1: Team scores 2 points → `Net_Diff = 2`
- Possession 2: Team scores 0 points → `Net_Diff = 0`
- Possession 3: Team scores 3 points → `Net_Diff = 3`

### Luck Adjustment (Important Note)

⚠️ **IMPORTANT:** The luck adjustment feature is currently **not working properly**. The code attempts to adjust for shooting luck by:
- Using player shooting percentages (free throw %, 3-point %)
- Blending expected value with actual results
- For 3-pointers: Using 20% expected + 80% actual for offense, 40% expected + 60% actual for defense

However, **this feature is not functioning correctly** in the current version. The `Off_Diff` and `Def_Diff` columns are generated but may not reflect proper luck adjustment. You should primarily rely on `Net_Diff` for now, or be aware that the luck-adjusted values may not be accurate.

## Command-Line Options

### Basic Usage
```bash
python process_rapm2.py <input_file>
```

### With Manual Year/Season
```bash
python process_rapm2.py <input_file> --year <year> --season "<season>"
```

**Arguments:**
- `input_file` (required): Path to your input CSV file
- `--year` / `-y` (optional): Starting year of season (e.g., `2025` for 2025-26 season)
- `--season` / `-s` (optional): Season string in format `YYYY-YY` (e.g., `"2025-26"`)

**Examples:**
```bash
# Auto-detect from filename
python process_rapm2.py NBA26.csv

# Manual specification
python process_rapm2.py my_data.csv --year 2025 --season "2025-26"

# Playoffs data
python process_rapm2.py NBA26_PS.csv
```

## What Happens During Processing?

The script performs several steps:

1. **Loads and cleans** the input CSV file
2. **Maps event types** from numeric codes to text labels
3. **Calculates scores** and cumulative totals
4. **Identifies possessions** by finding "End of Possession" events
5. **Handles player lineups** during foul sequences (propagation)
6. **Maps players** to offensive/defensive roles
7. **Fetches player stats** from NBA API (for luck adjustment - currently not working properly)
8. **Calculates metrics** (Net_Diff, Off_Diff, Def_Diff, etc.)
9. **Merges game dates** from external data source
10. **Filters and outputs** final files

## Troubleshooting

### Error: "File not found"
- Make sure the input file path is correct
- Use absolute path if file is in a different directory: `python process_rapm2.py /full/path/to/file.csv`

### Error: "Missing required columns"
- Verify your CSV has all required columns (see Input File Format section above)
- Check column names match exactly (case-sensitive)

### Error: "Could not auto-detect year/season"
- Use `--year` and `--season` flags manually
- Filename should match pattern: `NBA{YY}.csv` or `NBA{YY}_PS.csv` (e.g., `NBA26.csv`)

### Warning: "NBA API errors" or "Player stats merge failed"
- This is okay - the script will use default shooting percentages (75% FT, 35% 3PT)
- Since luck adjustment isn't working properly anyway, this won't affect your results significantly

### Warning: "Game dates not found"
- The script tries to merge game dates from an external source
- If this fails, the `game_date` column will be empty but processing continues

### Processing is slow
- Large files (100k+ rows) can take several minutes
- The script prints progress messages - wait for completion
- Processing time depends on file size and internet speed (for API calls)

### Output files are empty or missing columns
- Check that your input file has valid data
- Ensure `End_of_Possession` events are properly identified
- Verify player columns (`away_player1-5`, `home_player1-5`) contain valid player IDs

## Understanding the Output Data

### Each row represents:
- **One possession** (specifically, an "End of Possession" event)
- **The players on court** at that moment (`O1-O5` for offense, `D1-D5` for defense)
- **The change in score** during that possession (`Net_Diff`, `Off_Diff`, `Def_Diff`)

### How to use the output:
- **For RAPM analysis**: Use `Net_Diff`, `Off_Diff`, `Def_Diff` as your outcome variables
- **For factor analysis**: Use `Net_Diff` from TS file, `Offensive_Rebound` from REB file, `Is_Turnover` from TOV file
- **Player identification**: Use `O1-O5` and `D1-D5` columns (these contain player IDs)
- **Game context**: Use `game_id`, `game_date`, `period`, `time_quarter`, `score` for filtering/grouping

## Next Steps

After processing, you typically:
1. Use the output files for RAPM regression analysis
2. Aggregate player statistics across all possessions
3. Calculate player ratings based on their impact on `Net_Diff`, `Off_Diff`, `Def_Diff`

## Questions?

If you encounter issues or have questions:
- Check the error messages - they usually indicate what's wrong
- Verify your input file format matches the requirements
- Make sure all required Python packages are installed
- Remember: luck adjustment is not working properly, so focus on `Net_Diff` for now

---

**Last Updated:** This guide covers the current version of `process_rapm2.py`. The luck adjustment feature is known to not be working properly and should be used with caution.













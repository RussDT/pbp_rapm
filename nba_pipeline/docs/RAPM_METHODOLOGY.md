# RAPM Methodology: Possession Definitions & Numerators

This document defines exactly how each RAPM type identifies possessions (observations) and what value is used as the numerator for the regression, based on `02_process_rapm.py`.

---

## Overview

All RAPM calculations use ridge regression to estimate player impact. Each observation represents a "possession" where we know:
- **O1-O5**: The 5 offensive players on the court
- **D1-D5**: The 5 defensive players on the court
- **Numerator**: The outcome value for that possession (varies by RAPM type)

The regression isolates each player's contribution to the outcome, controlling for teammates and opponents.

### Alternate Clean 3-Factor Layer

The pipeline also has a parallel clean 3-factor build built on a shared first-chance possession universe:

- `FIRST_CHANCE`
- `ALT_TS`
- `ALT_EFG`
- `ALT_SQ`
- `ALT_MAKE`
- `ALT_FT`
- `ALT_TOV`
- `ALT_BADPASS_TOV`
- `ALT_SCORING_TOV`
- `SECOND_CHANCE_CLEAN`

This is documented in [Alternate Clean 3-Factor](./ALTERNATE_CLEAN_3_FACTOR.md). Important caveats:

- `ALT_TS` is not literal TS%. It is a first-chance scoring-surrogate metric defined on all first-chance possessions, including turnover-ended ones.
- `ALT_EFG` and `ALT_FT` are available on all seasons, and satisfy `ALT_TS = ALT_EFG + ALT_FT` on the shared first-chance denominator.
- On ShotQuality seasons, `FIRST_CHANCE` also carries `FC_SQ_Diff` and `FC_MAKE_Diff`, and the solver can expose `ALT_SQ` and `ALT_MAKE` as the richer subcomponents of `ALT_EFG`.
- `ALT_TOV` is a true first-chance turnover-rate metric built from turnovers before the first offensive miss, and `ALT_BADPASS_TOV` / `ALT_SCORING_TOV` are its matching children.

### ShotQuality-Era First-Chance Mode Split

For 2023-24 through 2025-26, `FIRST_CHANCE` also labels whether a standard first-chance possession had at least one first-chance FGA with ShotQuality `is_transition=True`.

- `Is_FC_Transition_Possession = 1` when any field-goal attempt before the first offensive miss is tagged transition.
- `Has_FC_FGA = 1` when the possession has a field-goal attempt before the first offensive miss.
- FT-only, turnover-only, and no-FGA possessions are in the non-transition bucket unless a separate transition-possession detector is added later.

`rapm.py` exposes two same-denominator conditional targets:

- `FC_TRANSITION_SCORING`: actual `FIRST_CHANCE.Net_Diff` on transition possessions, same-sample transition PPP placeholder on non-transition possessions.
- `FC_HALFCOURT_SCORING`: actual `FIRST_CHANCE.Net_Diff` on non-transition possessions, same-sample non-transition PPP placeholder on transition possessions.

It also exposes the additive version of the same split:

- `FC_MODE_MIX = transition * transition_ppp + nontransition * halfcourt_ppp`
- `FC_TRANSITION_VALUE = transition * (FIRST_CHANCE.Net_Diff - transition_ppp)`
- `FC_HALFCOURT_VALUE = nontransition * (FIRST_CHANCE.Net_Diff - halfcourt_ppp)`

At the processed-row level, `FC_MODE_MIX + FC_TRANSITION_VALUE + FC_HALFCOURT_VALUE = FIRST_CHANCE.Net_Diff`. The same-sample PPP baselines are computed inside `rapm.py` on the loaded run window and respect `--timedecay` / `--half-life` when present.

### Optional Season Fixed Effects

`rapm.py` also supports an optional `--season-effects` mode for multi-season runs.

- Adds one season-phase dummy per non-reference `(season, RS/PS)` combo after the player features
- Uses the earliest observed season-phase combo in the run as the omitted reference category, with `RS` preferred over `PS` inside the same season
- Estimates those season coefficients as nuisance fixed effects alongside the player coefficients
- Intended to absorb leaguewide year-to-year baseline shifts, which is especially useful for style/frequency metrics such as rim rate or midrange rate
- Writes a companion `*_season_effects.csv` artifact with the estimated season-phase effects in per-100 units

This mode changes the interpretation of the player coefficients from "relative to the pooled multi-year mean" toward "relative to the season-adjusted baseline inside the run window."

### Optional Age Fixed Effects

`rapm.py` also supports an optional `--age-dummies` mode for `RAPM` only.

- adds signed age-bin nuisance features for ages `18` through `38`
- `38+` is collapsed into `38`
- offensive players contribute `+1` to offense-age features
- defensive players contribute `+1` to defense-age features
- age `27` is omitted as the computational reference bucket on both sides
- player-season age bins come from `dpm_history.csv` via `nba_id`, `season`, and `age`; DARKO DPM values are not used as targets for this mode
- the exported `*_age_effects.csv` file recenters offense and defense curves separately to weighted mean zero across observed age slots, then reports net as `off - def`

This mode estimates separate offense and defense age-effect curves inside the RAPM solve itself, which is distinct from `--age-curve` prior mode.

### Optional DARKO Strength Splits

`rapm.py` supports a `--strength-splits` mode for `RAPM` only.

- Reads DARKO history from `/Users/russellthomas/Docs/2026_NBA_PIPELINE/databallr/darko/dpm_history.csv` by default, or from `--darko-history`
- Buckets each opposing lineup as `strong` or `not_strong` using legacy thresholds
- Offensive player interactions use the opposing five-man defensive DARKO sum: `strong` when summed `d_dpm > 2.0`
- Defensive player interactions use the opposing five-man offensive DARKO sum: `strong` when summed `o_dpm > 2.5`
- Missing DARKO player-season slots default to `0.0`, matching the legacy script behavior
- Adds one player interaction feature for each side and bucket: `player_facing_strong_off`, `player_facing_not_strong_off`, `player_facing_strong_def`, `player_facing_not_strong_def`
- Exports `off_vs_strong`, `off_vs_not_strong`, `def_vs_strong`, `def_vs_not_strong`, exposure counts/percentages, and `overall_*` weighted by each player's actual bucket exposure

This mode keeps the base `off` and `def` columns in the output and makes `net_rapm` equal to the exposure-weighted `overall_net_rapm`; `base_net_rapm` preserves the unsplit base net.

`--strength-legacy-scale` can be added to apply the legacy strong/not-strong post-fit scaling:

- Centers `overall_off` by offensive possessions and `overall_def` by defensive possessions
- Scales centered `overall_off` to a standard deviation of `1.48`
- Scales centered `overall_def` to a standard deviation of `1.36`
- Applies the same player-level delta to `off_vs_strong`, `off_vs_not_strong`, `def_vs_strong`, and `def_vs_not_strong`
- Writes `scaled_overall_*`, `scaled_*_vs_*`, scale factors, centers, and `scaled_overall_net_rapm`
- In scaled exports, `net_rapm` is the scaled net for sorting while unscaled `overall_net_rapm` remains available

### Optional DARKO Strength Sensitivity

`rapm.py` supports a separate `--strength-sensitivity` mode for `RAPM` only.

- Computes continuous five-man DARKO lineup strength for every possession
- Offensive sensitivity uses the opposing defensive lineup's summed `d_dpm`
- Defensive sensitivity uses the opposing offensive lineup's summed `o_dpm`
- Standardizes both strength variables within season, so one slope unit is one same-season standard deviation harder
- Adds separate player slope features: `player_strength_slope_off` and `player_strength_slope_def`
- Penalizes slope features more strongly than base player features by scaling their design columns; default `--strength-sensitivity-alpha-mult 4.0`
- Exports `off_strength_slope`, `def_strength_slope`, `net_strength_slope`, `net_vs_weak_1sd`, `net_vs_avg_strength`, `net_vs_strong_1sd`, `strong_minus_weak`, and `weak_minus_strong_drop`

Interpretation: `strong_minus_weak` is the estimated net RAPM change from a `-1 SD` opponent-strength environment to a `+1 SD` environment. Negative values indicate a projected drop against stronger lineups. Offense and defense are intentionally estimated separately because offensive drop-off against strong defenses and defensive drop-off against strong offenses are different basketball questions.

### Precomputed Season Intercepts

The repo also has a separate parquet-summary harness in `scripts/build_season_intercepts.py`.

- It computes season-level means directly from the processed parquet surface
- It emits both raw column means and solver-facing intercept rows
- It is meant for hardcoded external season variables, not for in-run nuisance estimation

Important implementation detail: the harness treats the processed filename as the canonical season key, because legacy processed parquets do not all store the same `Season` convention internally.

---

## 1. RAPM (Regularized Adjusted Plus-Minus)

**Function**: `process_rapm_py()`

**What it measures**: Net point differential per possession

### End_of_Possession Definition (evaluated in order):

```python
End_of_Possession = True when ANY of these conditions is met:

1. (event_type == "EndOfPeriod") AND (prev_seconds > 0)
   # End of quarter/half/game with time remaining

2. home_description contains "Flagrant (2 of 2|3 of 3)"
   # Final flagrant free throw for home team

3. visitor_description contains "Flagrant (2 of 2|3 of 3)"
   # Final flagrant free throw for away team

4. (event_type == "Turnover")
   # Any turnover ends the possession

5. home_description contains "REBOUND"
   AND Prev_visitor_desc contains "MISS"
   AND Prev_visitor_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Defensive rebound by home after away team miss (excluding mid-FT sequences)

6. visitor_description contains "REBOUND"
   AND Prev_home_desc contains "MISS"
   AND Prev_home_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Defensive rebound by away after home team miss

7. home_description contains "PTS"
   AND (event_type == "MAKE")
   AND Next_visitor_desc does NOT contain "S.FOUL"
   # Made field goal by home without shooting foul continuation

8. visitor_description contains "PTS"
   AND (event_type == "MAKE")
   AND Next_home_desc does NOT contain "S.FOUL"
   # Made field goal by away without shooting foul continuation

9. home_description contains "(1 of 1|2 of 2|3 of 3)"
   AND home_description contains "PTS"
   AND NOT (Prev_visitor_desc2 contains "Transition"
            OR Prev_visitor_desc contains "Transition"
            OR home_description contains "Flagrant")
   # Final FT made by home (excluding transition/flagrant fouls)

10. visitor_description contains "(1 of 1|2 of 2|3 of 3)"
    AND visitor_description contains "PTS"
    AND NOT (Prev_home_desc2 contains "Transition"
             OR Prev_home_desc contains "Transition"
             OR visitor_description contains "Flagrant")
    # Final FT made by away (excluding transition/flagrant fouls)
```

### Numerator Calculation:

**Output columns**: `Off_Diff`, `Def_Diff`, `Net_Diff`

**Scoring logic** (per event):
```python
# For each event, calculate points:
if is_free_throw:
    ActualNet = ExpFT  # Player's expected FT value (from Supabase, default 0.75)
elif is_2pt_make:
    ActualNet = 2.0    # Always exactly 2 points
elif is_3pt_attempt:
    ActualNet = actual_score  # 0 or 3 (actual result)
else:
    ActualNet = 0.0

# Cumulative sum per game, then diff between possessions:
Off_Diff = ActualTotal[current_EOP] - ActualTotal[previous_EOP]
```

**Luck-Adjusted Version (LA_RAPM)**:
```python
# 3PT luck adjustment with o_luck/d_luck parameters:
if is_free_throw:
    LA_OffNet = ExpFT  # FTs ALWAYS 100% luck adjusted
elif is_2pt_make:
    LA_OffNet = 2.0    # No adjustment for 2PT
elif is_3pt_attempt:
    LA_OffNet = o_luck * (Exp3PT * 3.0) + (1 - o_luck) * actual_score
    # Blend of expected 3PT value and actual result

# Default: o_luck=1.0, d_luck=1.0 (full luck adjustment)
```

---

## 2. TS (True Shooting RAPM)

**Function**: `process_ts_py()`

**What it measures**: Points scored per scoring attempt (excludes turnovers in final output)

### End_of_Possession Definition:

```python
End_of_Possession = True when ANY of these conditions is met:

1. (event_type == "EndOfPeriod") AND (prev_seconds > 0)

2. home_description contains "Flagrant (2 of 2|3 of 3)"

3. visitor_description contains "Flagrant (2 of 2|3 of 3)"

4. (event_type == "Turnover")
   # NOTE: Turnovers are included as EOP but FILTERED OUT later

5. visitor_description contains "MISS" AND (event_type == "MISS")
   # Any missed shot by away team

6. home_description contains "MISS" AND (event_type == "MISS")
   # Any missed shot by home team

7. home_description contains "PTS" AND (event_type == "MAKE")
   AND Next_visitor_desc does NOT contain "S.FOUL"

8. visitor_description contains "PTS" AND (event_type == "MAKE")
   AND Next_home_desc does NOT contain "S.FOUL"

9. home_description contains "(1 of 1|2 of 2|3 of 3)"
   AND NOT (Prev_home_desc2 contains "Transition"
            OR Prev_home_desc contains "Transition"
            OR home_description contains "Flagrant")

10. visitor_description contains "(1 of 1|2 of 2|3 of 3)"
    AND NOT (Prev_visitor_desc2 contains "Transition"
             OR Prev_visitor_desc contains "Transition"
             OR visitor_description contains "Flagrant")
```

**Key difference from RAPM**: TS includes missed shots as possession ends, and misses are explicitly captured.

### Post-Processing Filter:
```python
# After identifying EOP rows, remove turnovers:
nba_filt = nba_filt[nba_filt['event_type'] != 'Turnover']
```

### Numerator Calculation:

**Output column**: `Net_Diff`

```python
# FT luck-adjusted scoring:
if is_free_throw:
    LA_Action_Score = ExpFT  # Expected FT value (default 0.75)
else:
    LA_Action_Score = Home_Action_Score + Away_Action_Score  # Actual points

# Cumulative sum, then diff:
LA_Net_Total = cumsum(LA_Action_Score) per game
Net_Diff = LA_Net_Total[current_EOP] - LA_Net_Total[previous_EOP]
```

---

## 3. TOV (Turnover RAPM)

**Function**: `process_tov_py()`

**What it measures**: Turnover rate per possession

### End_of_Possession Definition:

```python
End_of_Possession = True when ANY of these conditions is met:

1. (event_type == "EndOfPeriod") AND (prev_seconds > 0)

2. home_description contains "Flagrant (2 of 2|3 of 3)"

3. visitor_description contains "Flagrant (2 of 2|3 of 3)"

4. (event_type == "Turnover")

5. home_description contains "REBOUND"
   AND Prev_visitor_desc contains "MISS"
   AND (Prev_Event != "FreeThrow")
   # Defensive rebound after FG miss (not FT miss)

6. visitor_description contains "REBOUND"
   AND Prev_home_desc contains "MISS"
   AND (Prev_Event != "FreeThrow")

7. home_description contains "PTS" AND (event_type == "MAKE")
   AND Next_visitor_desc does NOT contain "S.FOUL"

8. visitor_description contains "PTS" AND (event_type == "MAKE")
   AND Next_home_desc does NOT contain "S.FOUL"

9. home_description contains "(1 of 1|2 of 2|3 of 3)"
   AND home_description contains "PTS"
   AND NOT (Prev_visitor_desc2 contains "Transition"
            OR Prev_visitor_desc contains "Transition"
            OR home_description contains "Flagrant")

10. visitor_description contains "(1 of 1|2 of 2|3 of 3)"
    AND visitor_description contains "PTS"
    AND NOT (Prev_home_desc2 contains "Transition"
             OR Prev_home_desc contains "Transition"
             OR visitor_description contains "Flagrant")
```

### Numerator Calculation:

**Output column**: `Is_Turnover`

```python
Is_Turnover = 1 if (event_type == "Turnover") else 0
```

**Sign convention in rapm.py**:
```python
Off_Diff = -Is_Turnover  # Negative for offense (turnovers are bad)
Def_Diff = +Is_Turnover  # Positive for defense (forced turnovers are good)
```

---

## 4. REB (Offensive Rebound RAPM)

**Function**: `process_reb_py()`

**What it measures**: Offensive rebound rate on rebound opportunities

### Offensive_Rebound Definition (numerator):

```python
Offensive_Rebound = 1 when ANY of these conditions is met:

1. home_description contains "REBOUND"
   AND Prev_home_desc contains "MISS"
   AND time_quarter is NOT "0:00"
   AND Next_event is NOT "EndOfPeriod"
   AND Prev_home_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Home offensive rebound after home miss

2. home_description contains "REBOUND"
   AND Prev_home_desc contains "Putback"
   AND time_quarter is NOT "0:00"
   AND Next_event is NOT "EndOfPeriod"
   AND Prev_home_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Home offensive rebound on putback attempt

3. visitor_description contains "REBOUND"
   AND Prev_visitor_desc contains "MISS"
   AND time_quarter is NOT "0:00"
   AND Next_event is NOT "EndOfPeriod"
   AND Prev_visitor_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Away offensive rebound after away miss

4. visitor_description contains "REBOUND"
   AND Prev_visitor_desc contains "Putback"
   AND time_quarter is NOT "0:00"
   AND Next_event is NOT "EndOfPeriod"
   AND Prev_visitor_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Away offensive rebound on putback attempt

Otherwise: Offensive_Rebound = 0 (defensive rebound)
```

### End_of_Possession Definition:

```python
End_of_Possession = True when ANY of these conditions is met:

1. home_description contains "REBOUND"
   AND Prev_visitor_desc contains "MISS"
   AND Prev_visitor_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Home defensive rebound

2. visitor_description contains "REBOUND"
   AND Prev_home_desc contains "MISS"
   AND Prev_home_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Away defensive rebound

3. home_description contains "REBOUND"
   AND Prev_home_desc contains "MISS"
   AND Prev_home_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Home offensive rebound

4. visitor_description contains "REBOUND"
   AND Prev_visitor_desc contains "MISS"
   AND Prev_visitor_desc does NOT contain "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"
   # Away offensive rebound
```

**Key insight**: REB only includes rebound opportunities (plays where a rebound occurred after a miss). Made shots, turnovers, and other events are NOT possessions for REB.

### Numerator:

**Output column**: `Offensive_Rebound` (0 or 1)

**Sign convention in rapm.py**:
```python
Off_Diff = +Offensive_Rebound  # Positive for offense (OREBs are good)
Def_Diff = -Offensive_Rebound  # Negative for defense (allowing OREBs is bad)
```

---

## 4b. SHOOTER_OREB (Shooter Miss Recoverability)

**Function**: `process_shooter_oreb_py()`

**What it measures**: Offensive rebound rate after missed field-goal attempts, with the shooter separated from the four non-shooter offensive players.

### Observation Definition

Each observation is a missed field-goal attempt whose next play-by-play event is a rebound. Made field goals and free throws are excluded.

Rows are built on the missed-shot event rather than the rebound event, so the shooter can be stored directly on the row.

### Player Columns

```python
O1-O4    = the four offensive players on the floor excluding the shooter
Shooter  = player1_id from the missed FGA
D1-D5    = the five defensive players on the floor
```

Rows are dropped if the shooter cannot be matched into the offensive lineup and exactly four non-shooter offensive players cannot be identified.

### Numerators

```python
Offensive_Rebound = 1 if the shooting team gets the next rebound, else 0
Self_Offensive_Rebound = 1 if Offensive_Rebound == 1 and the rebounder is the shooter
```

**Output columns**: `Shooter`, `Rebounder`, `O1-O4`, `D1-D5`, `Offensive_Rebound`, `Self_Offensive_Rebound`

**Intended regression design**:

```python
Offensive_Rebound ~ four_non_shooter_offensive_players + shooter + five_defensive_players
```

This separates a player's impact as a non-shooter offensive rebounder from the recoverability of that player's own misses. The shooter coefficient intentionally includes shot diet unless a later solver adds shot-context controls.

Run the standalone ridge solver with:

```bash
python nba_pipeline/scripts/run_shooter_oreb_rapm.py 26 26 RS --alpha 3000 --min-shooter-misses 150
```

The solver exports `non_shooter_oreb`, `shooter_miss_recoverability`, and `def_oreb_suppression` in per-100 percentage-point units, plus raw shooter miss/team OREB/self OREB context.

---

## 4c. BLOCK_RECOVERY (Defensive Block Recovery Rate)

**Function**: `process_block_recovery_py()`

**What it measures**: Whether the defending team recovers possession after a blocked field-goal attempt.

### Observation Definition

Each observation is a blocked field-goal attempt. The denominator is blocked FGAs, not all missed shots or all possessions.

The processor builds the row on the blocked shot event, maps `O1-O5` to the shooting team and `D1-D5` to the defending team, then scans the next few same-period play-by-play rows for the first non-admin resolution event.

### Numerator

```python
Block_Recovered_By_Defense = 1 if the defending side gets the recovery resolution, else 0
```

The resolution is usually the next rebound row. Jump-ball tips and immediate post-block turnovers/shots are also used when they are the first non-admin event after the block.

**Output column**: `Block_Recovered_By_Defense`

**Internal target convention in rapm.py**:
```python
Off_Diff = -Block_Recovered_By_Defense  # Offensive lineups avoid losing blocked shots
Def_Diff = +Block_Recovered_By_Defense  # Defensive lineups recover their blocks
```

The exported result CSV keeps the repo's standard display convention, where `net_rapm = off - def`; for defensive-side block recovery impact, lower `def` values are better.

Run the standard solver with:

```bash
python nba_pipeline/scripts/rapm.py BLOCK_RECOVERY 21 26 ALL
```

---

## 5. RIM_FREQ (Rim Frequency RAPM)

**Function**: `process_rim_freq_py()`

**What it measures**: How often a lineup attacks the rim vs. takes other shots

### End_of_Possession Definition:

```python
# Every FGA is a possession:
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]
End_of_Possession = True  # For all remaining rows
```

**Included**: All made and missed field goal attempts
**Excluded**: Free throws, turnovers, rebounds, all non-shooting events

### Is_Rim_Attempt Definition (numerator):

```python
# Primary check: event_action_type codes
RIM_ACTION_TYPES = {
    # Layups
    5,    # Layup (generic)
    6,    # Driving Layup
    41,   # Running Layup
    43,   # Alley Oop Layup
    75,   # Driving Finger Roll Layup
    98,   # Cutting Layup Shot
    99,   # Cutting Finger Roll Layup Shot
    100,  # Running Alley Oop Layup Shot
    # Dunks
    7,    # Dunk
    9,    # Driving Dunk
    50,   # Running Dunk
    52,   # Alley Oop Dunk
    87,   # Putback Dunk
    107,  # Tip Dunk Shot
    108,  # Cutting Dunk Shot
    # Tips
    97,   # Tip Layup Shot
}

action_type_rim = event_action_type in RIM_ACTION_TYPES

# Fallback check: description patterns
desc_combined = (home_description + visitor_description).lower()
desc_pattern_rim = "layup" in desc OR "dunk" in desc OR " tip " in desc

# Combined:
Is_Rim_Attempt = 1 if (action_type_rim OR desc_pattern_rim) else 0
```

### Numerator:

**Output column**: `Is_Rim_Attempt` (0 or 1)

**Sign convention in rapm.py**:
```python
Off_Diff = +Is_Rim_Attempt  # Positive for offense (rim attacks are good)
Def_Diff = -Is_Rim_Attempt  # Negative for defense (allowing rim attempts is bad)
```

---

## 6. RIM_FG_PCT (Rim FG Percentage RAPM)

**Function**: `process_rim_fg_pct_py()`

**What it measures**: Efficiency at the rim (made rim attempts / all rim attempts)

### End_of_Possession Definition:

```python
# First filter to FGAs only:
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]

# Then filter to rim attempts only:
is_rim = (action_type_rim OR desc_pattern_rim)  # Same logic as RIM_FREQ
nba_df = nba_df[is_rim]

End_of_Possession = True  # For all remaining rows
```

**Included**: Only rim attempts (layups, dunks, tips)
**Excluded**: All non-rim FGAs (jumpers, 3PTs), free throws, turnovers

### Is_Rim_Make Definition (numerator):

```python
Is_Rim_Make = 1 if (event_type == 'MAKE') else 0
```

### Numerator:

**Output column**: `Is_Rim_Make` (0 or 1)

**Sign convention in rapm.py**:
```python
Off_Diff = +Is_Rim_Make  # Positive for offense (making rims is good)
Def_Diff = -Is_Rim_Make  # Negative for defense (allowing rim makes is bad)
```

---

## 7. ASSIST_POINTS

**Function**: `process_assist_points_py()`

**What it measures**: Assisted field-goal points per possession

### Possession Definition:

Uses the same standard `End_of_Possession` logic as RAPM:
- turnovers
- defensive rebounds
- made FGs without shooting-foul continuation
- final non-transition/non-flagrant made free throws
- end of period with time remaining

### Event Numerator:

```python
shot_text = home_description + " " + visitor_description
is_assisted_make = (event_type == "MAKE") AND shot_text contains "(... N AST)"
is_3pt = shot_text contains "3PT"

shot_points = 3 if (event_type == "MAKE" and is_3pt) else 2 if event_type == "MAKE" else 0
Assist_Points_Event = shot_points if is_assisted_make else 0
```

The possession output is built as a cumulative game total and then differenced at each standard RAPM possession end.

### Numerator:

**Output column**: `Assist_Points`

Most possessions are `0`, `2`, or `3`. Rare `4` / `5` rows can occur when a continued possession contains multiple assisted makes under the standard possession definition (for example: assisted and-one make, missed `1 of 1`, offensive rebound, then another assisted make).

---

## 8. RIM_ASSIST

**Function**: `process_rim_assist_py()`

**What it measures**: Assisted rim makes per possession

### Possession Definition:

Uses the same standard `End_of_Possession` logic as RAPM.

### Event Numerator:

```python
is_assisted_make = (event_type == "MAKE") AND shot_text contains "(... N AST)"
is_rim = (event_action_type in RIM_ACTION_TYPES) OR description contains "layup|dunk| tip "

Rim_Assist_Event = 1 if (is_assisted_make AND is_rim) else 0
```

The possession output is built as a cumulative game total and then differenced at each standard RAPM possession end.

### Numerator:

**Output column**: `Is_Rim_Assist`

Most possessions are `0` or `1`. Rare `2` rows can occur when a continued possession contains two assisted rim makes before the possession ends.

---

## 8a. DUNK

**Function**: `process_dunk_py()`

**What it measures**: Made dunks per possession

### Possession Definition:

Uses the same standard `End_of_Possession` logic as RAPM.

### Event Numerator:

```python
is_dunk = (event_action_type in DUNK_ACTION_TYPES) OR description contains "dunk"

Dunk_Event = 1 if (event_type == "MAKE" AND is_dunk) else 0
```

The possession output is built as a cumulative game total and then differenced at each standard RAPM possession end.

### Numerator:

**Output column**: `Is_Dunk`

Most possessions are `0` or `1`. Rare `2` rows can occur when a continued possession contains two made dunks before the possession ends.

---

## 8b. DUNK_ASSIST

**Function**: `process_dunk_assist_py()`

**What it measures**: Assisted made dunks per possession

### Possession Definition:

Uses the same standard `End_of_Possession` logic as RAPM.

### Event Numerator:

```python
is_assisted_make = (event_type == "MAKE") AND shot_text contains "(... N AST)"
is_dunk = (event_action_type in DUNK_ACTION_TYPES) OR description contains "dunk"

Dunk_Assist_Event = 1 if (is_assisted_make AND is_dunk) else 0
```

The possession output is built as a cumulative game total and then differenced at each standard RAPM possession end.

### Numerator:

**Output column**: `Is_Dunk_Assist`

Most possessions are `0` or `1`. Rare `2` rows can occur when a continued possession contains two assisted made dunks before the possession ends.

---

## 9. THREE_FREQ

**Function**: `process_three_freq_py()`

**What it measures**: 3-point attempt frequency on a shot denominator

### End_of_Possession Definition:

```python
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]
End_of_Possession = True
```

Every FGA is an observation.

### Numerator:

```python
Is_Three_Attempt = 1 if shot_text contains "3PT" else 0
```

**Output column**: `Is_Three_Attempt` (0 or 1)

Interpretation: `3PA / FGA`

---

## 10. THREE_FG_PCT

**Function**: `process_three_fg_pct_py()`

**What it measures**: 3-point field-goal percentage

### End_of_Possession Definition:

```python
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]
nba_df = nba_df[shot_text contains "3PT"]
End_of_Possession = True
```

Only 3PA are observations.

### Numerator:

```python
Is_Three_Make = 1 if (event_type == "MAKE") else 0
```

**Output column**: `Is_Three_Make` (0 or 1)

Interpretation: `3PM / 3PA`

---

## 11. MIDRANGE_FREQ

**Function**: `process_midrange_freq_py()`

**What it measures**: Midrange attempt frequency on a shot denominator

### End_of_Possession Definition:

```python
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]
End_of_Possession = True
```

Every FGA is an observation.

### Numerator:

```python
Is_Midrange_Attempt = 1 if shot is NOT 3PT AND NOT rim else 0
```

**Output column**: `Is_Midrange_Attempt` (0 or 1)

Interpretation: `Midrange FGA / FGA`

---

## 12. MIDRANGE_FG_PCT

**Function**: `process_midrange_fg_pct_py()`

**What it measures**: Midrange field-goal percentage

### End_of_Possession Definition:

```python
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]
nba_df = nba_df[shot is NOT 3PT AND NOT rim]
End_of_Possession = True
```

Only midrange FGA are observations.

### Numerator:

```python
Is_Midrange_Make = 1 if (event_type == "MAKE") else 0
```

**Output column**: `Is_Midrange_Make` (0 or 1)

Interpretation: `Midrange FGM / Midrange FGA`

---

## 13. PLAYTYPE_TS_MIX

**Function**: `process_playtype_ts_mix_py()`

**What it measures**: How a player shifts lineup shot mix toward higher- or lower-value ShotQuality descriptor bundles

### End_of_Possession Definition:

```python
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]
End_of_Possession = True
```

Every FGA is an observation.

### Descriptor Model:

For each FGA, the processor builds a descriptor key from:
- normalized ShotQuality `play_descriptors`
- shot value bucket (`2PT` vs `3PT`)

Then it estimates expected actual points from the empirical mean of that
descriptor bundle, shrunk toward the seasonwide FGA average:

```python
Playtype_Exp_PTS(bundle) =
    (count(bundle) * mean_actual_points(bundle) + k * global_mean) / (count(bundle) + k)
```

with `k = 100`.

### Numerator:

**Output column**: `Playtype_Exp_PTS`

Interpretation: descriptor-based expected points per FGA. Offensively, higher
values mean shifting lineups toward more efficient descriptor families.
Defensively, lower allowed values mean suppressing those families.

---

## 14. PLAYTYPE_PROXY_PTS

**Function**: `process_playtype_proxy_pts_py()`

**What it measures**: How a player shifts lineup FGA mix toward higher- or
lower-value Synergy-proxy shot categories, using actual points per FGA within
each season as the category baseline.

### End_of_Possession Definition:

```python
nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])]
End_of_Possession = True
```

Every FGA is an observation.

### Proxy Taxonomy:

Each FGA is assigned to one mutually exclusive bucket in this order:

1. `TRANSITION`
2. `CUT`
3. `SPOT_UP`
4. `UNASSISTED_3`
5. `UNASSISTED_2`
6. `OTHER_ASSISTED_2`

The bucket rules use ShotQuality descriptor flags plus 2PT/3PT detection:

- `TRANSITION`: `sq_desc_transition`
- `CUT`: `sq_desc_off_cut` or `sq_desc_alley_oop`, excluding prior buckets
- `SPOT_UP`: `sq_desc_catch_and_shoot`, excluding prior buckets
- `UNASSISTED_3`: any remaining 3PA
- `UNASSISTED_2`: any remaining 2PA with self-created tags such as
  `off_drive`, `pull_up`, `post_up`, `turnaround`, `hook`, `floater`,
  or `step_back`
- `OTHER_ASSISTED_2`: residual 2PA bucket

### Category Value Model:

For each season, each category gets a shrunk actual-points mean:

```python
Category_Mean_PTS(bucket) =
    (count(bucket) * mean_actual_points(bucket) + k * global_mean) / (count(bucket) + k)
```

with `k = 100`, where `global_mean` is the seasonwide actual points per FGA.

### Numerator:

**Output column**: `Playtype_Proxy_PTS`

```python
Playtype_Proxy_PTS = Category_Mean_PTS(bucket) - global_mean
```

Interpretation: a positive value means the shot belongs to a proxy category
that is more valuable than the season-average FGA. Offensively, higher values
mean shifting lineups toward better shot-diet buckets; defensively, lower
allowed values mean suppressing those buckets.

---

## Summary Table

| RAPM Type | Possession Definition | Numerator Column | Values | Offense + | Defense - |
|-----------|----------------------|------------------|--------|-----------|-----------|
| **RAPM** | Standard EOP (makes, DREBs, TOVs, final FTs, end of period) | `Off_Diff`/`Def_Diff` | Points | Scores more | Allows fewer |
| **TS** | Scoring attempts only (EOP minus turnovers) | `Net_Diff` | Points (FT luck-adj) | Scores efficiently | Allows fewer |
| **TOV** | Standard EOP | `Is_Turnover` | 0 or 1 | Fewer TOVs* | More forced TOVs* |
| **REB** | Rebound opportunities only | `Offensive_Rebound` | 0 or 1 | More OREBs | Fewer OREBs allowed |
| **SHOOTER_OREB** | Missed FGA with next-event rebound | `Offensive_Rebound`, `Self_Offensive_Rebound` | 0 or 1 | More teammate/self recovery after misses | Fewer OREBs allowed |
| **BLOCK_RECOVERY** | Blocked FGA | `Block_Recovered_By_Defense` | 0 or 1 | Avoids losing blocked shots | More recovered blocks |
| **RIM_FREQ** | Every FGA | `Is_Rim_Attempt` | 0 or 1 | More rim attacks | Fewer rim attempts allowed |
| **RIM_FG_PCT** | Rim attempts only | `Is_Rim_Make` | 0 or 1 | Better rim finishing | Better rim protection |
| **ASSIST_POINTS** | Standard EOP | `Assist_Points` | Usually 0, 2, 3 | More assisted scoring | Allows less assisted scoring |
| **RIM_ASSIST** | Standard EOP | `Is_Rim_Assist` | Usually 0 or 1 | More assisted rim makes | Allows fewer assisted rim makes |
| **DUNK** | Standard EOP | `Is_Dunk` | Usually 0 or 1 | More made dunks | Allows fewer made dunks |
| **DUNK_ASSIST** | Standard EOP | `Is_Dunk_Assist` | Usually 0 or 1 | More assisted made dunks | Allows fewer assisted made dunks |
| **THREE_FREQ** | Every FGA | `Is_Three_Attempt` | 0 or 1 | More 3PA | Allows fewer 3PA |
| **THREE_FG_PCT** | 3PA only | `Is_Three_Make` | 0 or 1 | Better 3PT shooting | Allows lower 3PT% |
| **MIDRANGE_FREQ** | Every FGA | `Is_Midrange_Attempt` | 0 or 1 | More midrange FGA | Allows more midrange FGA |
| **MIDRANGE_FG_PCT** | Midrange attempts only | `Is_Midrange_Make` | 0 or 1 | Better midrange shooting | Allows lower midrange FG% |
| **PLAYTYPE_TS_MIX** | Every FGA | `Playtype_Exp_PTS` | Continuous expected points | Better descriptor mix | Worse descriptor mix allowed |
| **PLAYTYPE_PROXY_PTS** | Every FGA | `Playtype_Proxy_PTS` | Continuous season-relative actual points | Better proxy shot diet | Worse proxy shot diet allowed |

*TOV signs are inverted so positive = good for both offense and defense

---

## Player Propagation & FT Check

For RAPM, TS, and TOV, additional processing ensures correct lineup attribution during free throw sequences:

### PotentialFoul Detection:
```python
# Look ahead 5 events for FTs and substitutions
PotentialFoul = True when:
    - FT events exist in next 5 plays
    - Substitution events exist in next 5 plays
    - Current event is a Foul
    - NOT a Technical, Flagrant, Offensive, or Transition foul
```

### Player Propagation:
When a `PotentialFoul` is detected, player values (a1-h5) are propagated forward through the FT sequence to maintain correct lineup attribution even if substitutions occur during FTs.

### FT Off Check:
Corrects `End_of_Possession` flags when offensive FT rebounds occur (rare edge case where the FT shooter's team gets the rebound on a missed FT).

---

## File Naming Convention

Processed files: `{TYPE}{YEAR}{PS_SUFFIX}.parquet`

Examples:
- `RAPM26.parquet` - 2025-26 regular season
- `RIM_FREQ23_PS.parquet` - 2022-23 playoffs
- `LA_RAPM25.parquet` - 2024-25 luck-adjusted RAPM
- `ASSIST_POINTS26.parquet` - 2025-26 assisted FG points
- `PLAYTYPE_TS_MIX26.parquet` - 2025-26 descriptor-based playtype scoring mix

---

## Output File Columns

All processed files contain:
- `game_id` - NBA game identifier
- `O1` through `O5` - Offensive player IDs
- `D1` through `D5` - Defensive player IDs
- `game_date` - Date of the game
- `Season` - Season year (ending year)
- `{numerator_column}` - The metric-specific value
- Context columns: `score`, `period`, `time_quarter`, `away_score`, `home_score`, `score_margin`, `event_num`

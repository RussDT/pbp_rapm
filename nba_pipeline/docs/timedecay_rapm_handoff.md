## `timedecay_rapm` Table — Supabase Reference

### What It Is
Time-decay weighted RAPM (Regularized Adjusted Plus-Minus) with rubberband adjustment for NBA players. Derived from 6 years of play-by-play data (2021-2026) with a 700-day half-life decay (recent seasons weighted more) and rubberband correction (adjusting for score-margin coasting/pushing effects).

Each row is one player. The table is fully replaced daily by the pipeline — it always reflects the latest data.

### Connection
- **Project**: `databallr` (`bvwahamwfvolchcezxnw`)
- **Table**: `public.timedecay_rapm`
- **RLS**: Enabled. Anon can SELECT. No auth required for reads.
- **Primary Key**: `nba_id` (one row per player)

### Indexes & Common Access Patterns

**Always filter by `year` first.** Most queries should include `.eq('year', 2026)`.

| Index | Columns | Use Case |
|-------|---------|----------|
| PK | `nba_id` | Single player lookup |
| `idx_timedecay_rapm_year` | `year` | **Most common** — filter all players for a year |
| `idx_timedecay_rapm_nba_id_year` | `nba_id, year` | Player + year lookup (composite) |
| `idx_timedecay_rapm_team` | `team_abbreviation` | Filter by team |

**Typical query patterns (in order of frequency):**
1. `.eq('year', 2026)` — get all current players
2. `.eq('nba_id', 203999).eq('year', 2026)` — single player lookup
3. `.eq('team_abbreviation', 'DEN').eq('year', 2026)` — team roster
4. `.eq('year', 2026).not('team_abbreviation', 'is', null)` — active players only

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `nba_id` | bigint (PK) | NBA player ID |
| `year` | integer | **Always filter on this.** Latest year in dataset (currently 2026) |
| `player_name` | text | Full name (e.g., "Nikola Jokic") |
| `team_id` | bigint | NBA team ID (nullable — retired/inactive players won't have this) |
| `team_abbreviation` | text | e.g., "DEN", "BOS" (nullable) |
| `pos2` | text | Position (e.g., "C", "PG", "SF,PF") (nullable) |
| `oTS` | double | Offensive true shooting factor |
| `oTOV` | double | Offensive turnover factor |
| `oREB` | double | Offensive rebounding factor |
| `dTS` | double | Defensive true shooting factor |
| `dTOV` | double | Defensive turnover factor |
| `dREB` | double | Defensive rebounding factor |
| `off` | double | Total offensive RAPM (sum of oTS + oTOV + oREB) |
| `def` | double | Total defensive RAPM (sum of dTS + dTOV + dREB, inverted so positive = good) |
| `net_rapm` | double | Net RAPM (off + def + RESID). The headline number. |
| `RESID` | double | Residual unexplained by the 6 factors |
| `off_poss` | double | Offensive possessions played |
| `pval` | double | "Peripheral value" = oTOV + oREB + dREB + dTOV (non-shooting contributions) |

**For each of these 10 metrics** (`oTS`, `oTOV`, `oREB`, `dTS`, `dTOV`, `dREB`, `off`, `def`, `net_rapm`, `pval`), there are rank and percentile columns:

| Column Pattern | Type | Description |
|---------------|------|-------------|
| `{metric}_rank` | integer | Rank where 1 = best (highest value) |
| `{metric}_pct` | double | Percentile 0-100 where 100 = best |

### Sign Convention
**Higher = better for ALL metrics.** Positive offensive values mean the player helps their team score. Positive defensive values mean the player prevents opponents from scoring. `net_rapm` of +7.58 (Jokic) means his team scores ~7.58 more points per 100 possessions with him on court vs off.

### Data Stats (as of Feb 2026)
- **1,066 players** (anyone with possessions 2021-2026)
- **517 matched** with current team/position (2026 active players)
- **549 unmatched** (retired/inactive — `team_id`, `team_abbreviation`, `pos2` are null)
- `net_rapm` range: -6.34 to +7.58
- Top 5: Jokic (+7.58), SGA (+6.99), Giannis (+5.85), Kawhi (+5.37), Caruso (+5.28)

### Supabase JS Client Usage

```ts
// All active players for current year (MOST COMMON PATTERN)
const { data } = await supabase
  .from('timedecay_rapm')
  .select('*')
  .eq('year', 2026)
  .not('team_abbreviation', 'is', null)
  .order('net_rapm', { ascending: false });

// Single player by nba_id
const { data } = await supabase
  .from('timedecay_rapm')
  .select('*')
  .eq('nba_id', 203999)
  .eq('year', 2026)
  .single();

// Team roster
const { data } = await supabase
  .from('timedecay_rapm')
  .select('player_name, pos2, net_rapm, off, def, net_rapm_rank, net_rapm_pct')
  .eq('year', 2026)
  .eq('team_abbreviation', 'DEN')
  .order('net_rapm', { ascending: false });

// Top 20 defenders
const { data } = await supabase
  .from('timedecay_rapm')
  .select('player_name, team_abbreviation, def, def_rank, def_pct')
  .eq('year', 2026)
  .not('team_abbreviation', 'is', null)
  .order('def', { ascending: false })
  .limit(20);
```

### SQL Examples

```sql
-- All active players for current year
SELECT * FROM timedecay_rapm WHERE year = 2026 AND team_abbreviation IS NOT NULL ORDER BY net_rapm DESC;

-- Single player
SELECT * FROM timedecay_rapm WHERE nba_id = 203999 AND year = 2026;

-- Team roster
SELECT player_name, pos2, net_rapm, off, def FROM timedecay_rapm WHERE year = 2026 AND team_abbreviation = 'DEN' ORDER BY net_rapm DESC;

-- Filter by position
SELECT player_name, net_rapm, pval FROM timedecay_rapm WHERE year = 2026 AND pos2 LIKE '%C%' ORDER BY net_rapm DESC;
```

### Nullable Fields
`team_id`, `team_abbreviation`, and `pos2` are null for ~549 players who aren't active in the current 2026 season. Filter with `.not('team_abbreviation', 'is', null)` or `WHERE team_abbreviation IS NOT NULL` to get only current players.

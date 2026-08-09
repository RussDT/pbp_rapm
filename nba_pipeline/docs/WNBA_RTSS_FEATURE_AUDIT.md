# WNBA RTSS Rich-Tracking Feature Audit

## Status

The local RTSS/Second Spectrum WNBA export is suitable for a new enrichment
pipeline and for season-forward feature research. It is not yet wired into the
live WNBA impact metric.

The July 29, 2026 audit profiled all `761` downloaded games and both markings
variants:

- `2023`: `173` games;
- `2024`: `266` games;
- `2025`: `322` games;
- game-id types: `21` code `101`, `682` code `102`, `57` code `104`, and one
  code `105`;
- `15,864,510,959` bytes profiled across the paired 2D and 3D JSON files;
- `32` rich event tables and `934` distinct table-column pairs.

The source files remain ignored local data under
`downloads/rtss-data/wnba/`. Generated audit CSV/JSON files live under
`wnba_test/Results/rtss_feature_audit/`.

## Shared AWS Bucket Access

The canonical shared mirror is:

```text
s3://rtss-data/wnba/
```

This is a cross-account shared bucket, not one of the buckets returned by
listing the local AWS account. Verify the active identity and access without
printing credentials:

```bash
aws sts get-caller-identity --profile default
aws s3 ls s3://rtss-data/wnba/ --profile default
aws s3 ls s3://rtss-data/wnba/api/portal/schedules.json --profile default
```

Inspect the remote schedule before downloading:

```bash
aws s3 cp s3://rtss-data/wnba/api/portal/schedules.json - --profile default \
  | jq '[.[] | select(.name | contains("2026"))]'
```

Preview and then update the ignored local mirror with:

```bash
aws s3 sync s3://rtss-data/wnba/ downloads/rtss-data/wnba/ \
  --profile default --dryrun
aws s3 sync s3://rtss-data/wnba/ downloads/rtss-data/wnba/ \
  --profile default
```

As of August 6, 2026, the shared manifest exposes 2026 preseason, regular
season, All-Star, and play-in groups. The regular-season group contains 330
scheduled games, 229 marked ended, and ended games already expose both
`basketball-markings.json` and `basketball-markings-3d.json` objects.

The August 6, 2026 sync pulled all pending 2026 data into the local mirror:
`929` objects (`5.15 GB`) covering all `229` ended regular-season games plus
the one ended play-in game, each with paired 2D and 3D markings, along with
`463` refreshed `api/portal` schedule/gamedata objects. A post-sync `--dryrun`
showed zero pending objects, and every one of the `991` local game directories
(2023-2026) carries both markings variants. The 2026 preseason and All-Star
groups expose no games file in the bucket yet. The remaining `101` scheduled
2026 regular-season games and any playoff groups will need follow-up syncs as
games end.

The original sharing note named
`arn:aws:iam::975049904559:user/rtss-data`, but that IAM user no longer exists
in the local account. The current authenticated account session can read the
shared prefix through the bucket policy. For unattended automation, create a
new dedicated least-privilege read profile rather than relying on root
credentials; do not change IAM state merely to run a manual read-only sync.

## Canonical Source Rule

Use `basketball-markings.json` as the canonical base for common event tables.
Join `basketball-markings-3d.json` fields onto those rows by stable event ID.
Treat the 3D-only tables as separate enrichments.

Do not substitute the 3D file for the 2D file:

- four 2023 games and three 2025 games have populated 2D possessions but empty
  3D possessions;
- 2024 common possession/chance/shot tables have exact 2D/3D parity;
- some nonempty screen, pick, pass, rotation, and contest tables differ by a
  small number of rows;
- the 3D side is nearly always a subset or close match, but unmatched rows must
  remain explicit.

The production fact build should therefore fail or warn on unexpected
2D-to-3D match bands rather than dropping unmatched 2D rows.

## Lineup Quality

The embedded `offPlayerIds` / `defPlayerIds` possession arrays are not the best
lineup surface. Only `93.18%`, `93.94%`, and `93.16%` of 2023, 2024, and 2025
possessions respectively contain exact, unique, non-overlapping fives on both
sides.

The `chance_players` table is dramatically better:

| Season | Chance groups | Fully valid | Invalid |
|---|---:|---:|---:|
| 2023 | 36,661 | 99.9891% | 4 |
| 2024 | 56,366 | 100.0000% | 0 |
| 2025 | 68,306 | 100.0000% | 0 |

A valid chance group has exactly ten rows, ten unique player UUIDs, five
offense rows, and five defense rows. Use this table as the preferred rich-data
lineup reconstruction surface, with `player_stints` and the possession arrays
as validation evidence.

This is strong enough to test rich lineups against the repaired Gabriel/CDN
lineups for 2023-2025. It does not authorize silently replacing the current
production lineups before player identity and possession parity close.

## Player Identity

The rich `shots.pbpId` aligns to the normalized WNBA `event_num` within a game.
The conservative shot-only crosswalk matched `108,551` of `111,956` shot-event
records to normalized PBP and produced:

- `248` high-confidence provider-UUID to numeric WNBA-ID mappings out of `320`
  observed provider player UUIDs;
- `28` mappings with no disagreeing shot-event evidence;
- only `3` ambiguous modal mappings;
- `69` UUIDs without sufficient matched-shot evidence.

High confidence means an exact association or at least three matched shot
events with one numeric WNBA ID carrying at least 80% of the evidence.
Disagreeing event rows remain visible in the crosswalk; names are labels and
are not identity keys.

Finish the identity layer by applying game roster, team, lineup, and stint
constraints to unresolved players. Do not use fuzzy names as the primary
crosswalk.

## Shot-Quality Opportunity

The canonical 2D shot table contains `111,956` attempts. Under the current
impact scope—every three plus non-three `jumper`, `post`, or `floater`—there
are `66,886` eligible MID/3P attempts:

| Season | MID | 3P | Eligible total |
|---|---:|---:|---:|
| 2023 | 7,827 | 7,669 | 15,496 |
| 2024 | 11,381 | 12,174 | 23,555 |
| 2025 | 12,108 | 15,727 | 27,835 |

Distance, region, catch-and-shoot, dribbles before the shot, closest-defender
distance and identity, contest level, and contester IDs have full coverage on
the canonical shots. Three-dimensional shot matches cover `97.69%`, `100%`,
and `99.03%` by season.

Shot-clock coverage is the important drift:

- `2023`: `100%`;
- `2024`: `95.40%`;
- `2025`: `89.09%`.

The model must use a documented chance-clock fallback and a missingness flag;
it must not drop the missing-clock attempts.

### Eligible Pre-Shot Features

- shot distance, region, value, clock, shot family, and complex shot type;
- catch-and-shoot, dribbles, touch duration, and shooter movement;
- closest-defender distance, approach, speed, angle, and contest level;
- shot-contest distance, closing speed, rearview state, hand/wrist height, and
  body orientation when their measurement timing is confirmed;
- chance context such as transition, after-timeout, reversals, frontcourt
  clock, bring-up pressure, and pickup distance.

### Quarantined Fields

- `qSQ`, `qSP`, `startQSP`, and `endQSP` are provider benchmarks until their
  definitions and timing contract are documented;
- make/miss, realized points, `complexOutcome`, rim depth, rim left/right,
  bank result, and realized assist/turnover labels are targets or diagnostics;
- arc, release, jump, pose, and fall measurements need their own timing audit
  and belong in a separate shotmaking/mechanics lane when they occur during or
  after release.

WNBA ShotQuality impact use remains MID/THREE only. RIM is unchanged.

## Recommended Feature Families

### 1. Lineup and Exposure

Build canonical game, possession, chance, player-chance, and player-stint
facts. Compare rich chance exposure and stint minutes with repaired WNBA
lineups and box totals. Use rich data as repair evidence before it becomes a
solve input.

### 2. MID/3P Context Expected Points

Train a context-only expected-points model from eligible shots. Use:

- 2023 as partial fit data;
- 2024 as the first forward validation;
- 2025 as untouched confirmation;
- 2026 as final confirmation when acquired.

Group every split by game. Build shooter, defender, team, and opponent
outcome-derived rates inside the fit fold and exclude the current game.
Compare against the existing WNBA PBP context model and use provider `qSQ` /
`qSP` only as external benchmarks.

### 3. Defensive Assignment, Rotation, and Contest

Keep three roles distinct:

- possession matchup assignment from `poss_matchups`;
- physical closest/primary contest from `shots` / `shot_contests`;
- help and recovery activity from `rotations`, `closeouts`,
  `missed_closeouts`, and `defender_stance`.

Estimate contest and rotation value relative to the context-only shot model.
Do not treat provider assignment identity as physical geometry or causal
credit. Any player-facing defensive split must close back to its parent
through exposure-weighted centered contributions.

### 4. Touch and Passing Creation

Use `touches`, `dribbles`, `passes`, `drives`, `isolations`, and `posts` for:

- on-ball time and pressure;
- advantage creation and loss;
- double-team response;
- potential-shot and potential-assist creation;
- pass difficulty, defender gambling, deflections, and turnover pressure.

Realized `ledToShot`, assist, completion, and turnover fields are labels, not
same-row predictors. Player-season summaries must be prior-game-only.

### 5. Screening and Roll Gravity

Use `picks`, `off_ball_screens`, `handoffs`, and `fake_handoffs` to measure:

- screen contact and direct involvement;
- slips, rolls, pops, and short rolls;
- taggers, defensive scheme, and spacer locations;
- handler, screener, cutter, and defender opportunity rates.

Define explicit opportunity denominators before interpreting player effects.

### 6. Rebounding and Boxouts

Use `rebounds`, `boxout_opportunities`, `chance_players`, and `jumps` to
separate:

- crash and rebound-opportunity creation;
- boxout responsibility and win rate;
- rebound collection conditional on opportunity;
- expected second-chance value.

Keep post-shot rebound outcomes outside pre-shot SQ.

### 7. Physical Workload

`speed_load`, jumps, falls, acceleration/deceleration, and vertical load are a
later fatigue/availability research lane. They should not enter the current
impact prior until as-of timing, health interpretation, and forward incremental
value are demonstrated.

## Validation and Promotion Contract

1. Validate primary keys, foreign keys, per-game row parity, clock ranges,
   player/team constraints, and exact lineup counts.
2. Preserve 2D-only and 3D-only rows in explicit audit outputs.
3. Build all outcome-derived encodings inside game-grouped folds.
4. Select models on earlier seasons only.
5. Require forward improvement over both no-prior and the accepted WNBA
   factor-specific box-prior lane.
6. Keep the live `5000/40000/1.5` WNBA solve unchanged until a feature family
   clears that gate.
7. After 2026 data arrives, use it as confirmation rather than retuning the
   2023-2025 winner on the opened test season.

## Reproduction

Run the full audit:

```bash
.venv/bin/python wnba_test/audit_rtss_wnba_features.py --workers 4
```

Build and execute the compact analysis notebook:

```bash
.venv/bin/python wnba_test/build_rtss_wnba_audit_notebook.py
```

Key outputs:

- `wnba_test/Results/rtss_feature_audit/audit_manifest.json`
- `wnba_test/Results/rtss_feature_audit/two_d_three_d_parity.csv`
- `wnba_test/Results/rtss_feature_audit/chance_player_quality.csv`
- `wnba_test/Results/rtss_feature_audit/shot_feature_coverage.csv`
- `wnba_test/Results/rtss_feature_audit/player_crosswalk.csv`
- `wnba_test/Results/rtss_feature_audit/column_modeling_catalog.csv`
- `wnba_test/Results/rtss_feature_audit/feature_catalog.csv`
- `wnba_test/analysis/rtss_wnba_feature_audit.ipynb`

The column modeling catalog conservatively labels all `934` table-column
pairs: `406` identifier/join fields, `251` mechanism candidates, `105` context
candidates, `45` realized targets, `43` post-release/pose diagnostics, four
provider benchmarks, and `80` fields requiring manual semantic review. The
automatic label is a review aid, not a substitute for provider documentation.

# NBA estimated skills: one inventory and daily operational handoff

Owner instruction October20,2026: stay organized across all estimated skills
so new2026–27 game data can feed daily updates. Do not confuse retaining a
stat with admitting it to VPM, or preparing the handoff with a running job.

**Status: inventory, frozen-recipe bundle and daily-output preflight are
implemented. The raw-data-to-state daily adapters and scheduler are not wired.**
No automatic daily-update claim, new source retrieval or2027 outcome scoring.
The inventory/check commands never invoke research builders or fit models.

## One operational map

Curated source: `nba_pipeline/config/estimated_skills_operations.json`.
Entry point: `nba_pipeline/scripts/manage_estimated_skills.py`.
[Readable inventory](http://127.0.0.1:8767/skills-operations.html).
Generated `results/estimated_skills/operations/inventory.json` and CSV provide
machine-readable routes, units, parameter availability and remaining work.

There are16 source/estimator groups and88 **routed outputs**, not88 independent
skills or VPM coefficients:

| Existing profile | Routed outputs | Meaning |
|---|---:|---|
| Current neutral library |59|The actual current skills-page estimators, shooting summaries, context and retained creation outputs;24 distinct numeric mapper inputs |
| Standalone incumbents |28|The older registry's published own-stat estimators; some stats overlap the library but use different recipes |
| Retained rim-makes skill |1|The measured rim defended-makes OU forecast tested separately, outside current VPM |

The existing20 research lanes are listed with live/diagnostic/failed status.
Diagnostics and lab winners do not silently become daily recipes. The existing
standalone registry and its builder remain intact; this operational map routes
both existing profiles and the retained additions without merging their values.
`current_neutral:<key>` and `standalone_incumbent:<key>` explicitly disambiguate
same-stat/different-estimator outputs. Do not name these profiles new VPM versions.

Each family records source requirements, current parameter/code files,
research-builder references, fields, units, mapper membership and remaining
live-adapter work. All16 adapter slots are currently null, deliberately.

## Daily sequence when the source adapters are connected

1. Reconcile a season-aware census of completed games from canonical box/PBP.
   Normalize exact NBA/game IDs, admit rookies/new players and corrected games,
   and keep the source's date/team/exposure. Do not retain the582-player
   historical display census as a live roster rule.
2. Consume measured tracking/hustle/rim sources as they arrive. Missing fields,
   outages and pending source games remain explicit; no inferred zero or filled
   estimate becomes a measurement. Show freshness separately per source family.
3. Update states with the selected frozen parameter bundle. No nightly DE,
   model search, mapper re-selection or new-season error optimization. Context,
   age/debut and season-boundary logic must match the chosen estimator.
4. Compute coherent makes, assisted/unassisted splits and derived summaries.
   Retain approximate TSA=FGA+.44FTA labeling and the exact point-valued role
   transform. Export all retained stats even without current VPM coefficients.
5. Write immutable dated outputs and per-family adapter receipts. Run the
   preflight below. Keep last good output if a family is delayed or fails;
   report its actual source watermark instead of relabeling an old state fresh.
6. Archive predictions before the next target games. Publish a dated skill
   snapshot only after the real adapter/replay checks pass. VPM rating refresh
   is a separate downstream consumer. Activate a monitored schedule only after
   the complete daily path works; the current coordinator installs no jobs.

## Commands available now

From RAPM root:

```sh
.venv/bin/python nba_pipeline/scripts/manage_estimated_skills.py inventory \
  --out nba_pipeline/results/estimated_skills/operations \
  --reader-dir ../vpm/vpm/results/harness/vpm_repaired_forward_expanded_v1

.venv/bin/python nba_pipeline/scripts/manage_estimated_skills.py freeze \
  --season 2027 \
  --out nba_pipeline/results/estimated_skills/operations/bundles/2027_carry_forward_v2

.venv/bin/python nba_pipeline/scripts/manage_estimated_skills.py check \
  --bundle nba_pipeline/results/estimated_skills/operations/bundles/2027_carry_forward_v2 \
  --receipts nba_pipeline/results/estimated_skills/daily_receipts/2027 \
  --as-of 2026-10-22 \
  --out nba_pipeline/results/estimated_skills/operations/readiness/2026-10-22
```

The displayed bundle has already been created; repeating freeze at that path
fails rather than replacing it. Use a new explicit bundle path for a deliberate
recipe update. `v2` here denotes this pinning bundle revision, not a new player
model. It copies the **currently exported** recipes, including their current
2025-vintage files; it is not a newly fitted2026 preseason parameter vintage,
model adoption or a standalone reproducible runtime. Input-source dependencies
and an approved live adapter still have to be supplied.

`inventory` reports all fields even when not operational. `freeze` copies exact
parameter/entry-code objects by SHA256, stores the config and immutable
manifest, and refuses missing pins or replacement. `check` verifies the pinned
objects and upstream adapter receipts. It neither generates new states nor
publishes output. Missing receipts give exit2, not success. No historical
snapshot is automatically accepted as a live refresh.

## Daily adapter output contract

One receipt named `<family_id>.json` per registered family, next to its parquet
artifact. Required metadata:

```json
{
  "family": "creation",
  "season": 2027,
  "as_of": "2026-10-22",
  "parameter_bundle_sha256": "SHA256 of bundle manifest.json",
  "source_status": "complete",
  "completed_game_census_checked": true,
  "identities_checked": true,
  "missingness_checked": true,
  "expected_player_ids": [203999],
  "artifact": "creation.parquet",
  "artifact_sha256": "SHA256 of the parquet artifact"
}
```

Parquet is long, one row per exact `player_id + stat_key`, with `value`,
`last_observation_date`, `missing_reason`. All configured family keys must be
present for every declared player; an unmeasured cell is an explicit missing
row with a reason, while a measured zero remains zero. Observation dates must
be strictly before the as-of cut. The universe is supplied for that day, never
fixed to an archived player count. Source census flags are attestations from
the upstream audited adapter; this coordinator does not independently derive
source completeness from raw games.

The preflight rejects missing/stale receipts, wrong season/bundle, pending
sources, unchecked census/identities/missingness, changed artifacts, wrong
stat keys, duplicate rows, missing per-player fields, wrong player census,
invalid IDs, nonfinite values, missing reasons and future observation dates.
It does not certify individual-stat forecast quality or enforce a full
basketball observation schema; those belong to each live adapter and its audit.
A passing receipt check alone is not end-to-end live readiness.

## Completed verification and remaining work

Sixteen focused tests pass, covering route coverage, immutable pin copies,
tamper detection, absent-feed rejection, valid zero/missing preservation and
stale/wrong/incomplete/future/bad-state rejection. All configured recipe pins
were present. The real readiness check correctly blocks all16 families because
no daily adapter receipts exist. The generated inventory and freeze read only
existing completed2026-and-earlier artifacts, not2027 game outcomes.

The next concrete implementation is the **season-aware canonical observation
adapter plus neutral-core replay**, then measured defense, retained creation,
standalone states, derived exports/reader and a two-run idempotence/partial-feed
replay. Scheduling is last. Do not run the old daily_rapm_update.sh as a shortcut:
it has stale season/path constants, no skills hooks and unrelated publication
steps. Preserve the existing research packages and held-out evaluation design.

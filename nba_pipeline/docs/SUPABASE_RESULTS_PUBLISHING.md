# Supabase Results Publishing

The Databallr Supabase project has a versioned results hub for small, curated
analytics outputs that need to be reusable across sites. It is the default
publication path for reader-facing CSV results that do not already have a
purpose-built typed table.

The no-prior TD550 net-continuation VPM history is the current purpose-built
exception: 34,075 player-date rows live in
`public.nba_no_prior_vpm_td550_snapshots`, with indexed player, player-name,
calendar year/month, and season/checkpoint access. It is research-only and is
not the accepted prior-centred VPM board.

The live datasets are:

| Dataset slug | Rows | Status | As of |
| --- | ---: | --- | --- |
| `vpm_epm12_x2_research_board` | 1,556 | research | 2026-07-24 |
| `historical_scoring_security_1997_2026` | 11,604 | research | 2026-08-25 |
| `nba_box_estimated_rztov_1997_2013` | 6,186 | research | 2026-08-25 |
| `nba_observed_rztov_2014_2026` | 5,370 | research | 2026-08-25 |
| `wnba_box_estimated_rztov_2026` | 229 | research | 2026-08-24 |
| `wnba_box_estimated_rztov_2022_2026` | 890 | research | 2026-08-26 |

All are public derived result surfaces. The VPM board remains the
development-leading EPM12 x2 research board, not untouched confirmation. The
historical board is own scoring plus ball security, not total offense,
playmaking, ORAPM, VPM, or production rZTOV. NBA rZTOV remains two datasets:
the 1997-2013 workload is a retrospective box/PBP estimate, while 2014-2026
retains the observed modern workload. WNBA publishes both the 2026 board and a
2022-2026 descriptive history. The historical rows use one fixed 2023-2025
retrospective model/price fit; they are not observed tracking history,
walk-forward prediction, or live WNBA VPM.

## Database contract

The migration in `supabase/migrations/` creates:

- `public.analytics_datasets` — stable dataset registry and latest-version
  pointer;
- `public.analytics_dataset_versions` — immutable content-addressed publish
  receipts;
- `public.analytics_dataset_rows` — versioned result rows stored as JSONB;
- `public.analytics_current_rows` — site-facing view of the current complete
  version; and
- `public.promote_analytics_dataset_version(...)` — service-role-only finalize
  function that refuses to promote an incomplete upload.

All three tables have RLS enabled. Anonymous and authenticated clients receive
read-only access to rows whose dataset registry entry has `is_public = true`.
Only the service role can write. The view uses PostgreSQL 15
`security_invoker`, so it preserves the underlying RLS policies.

Old versions remain queryable for reproducibility. A content hash plus a
deterministic UUID makes a retry idempotent. A new file does not become current
until the uploaded row count matches the validated source row count.

## Publish a result

Register a CSV in
`nba_pipeline/config/supabase_publications.json`. Every registration must
declare:

- a stable slug, title, description, status, and public/private decision;
- an exact ordered column contract;
- the row-key columns and minimum plausible row count;
- an as-of date or one source column containing a single as-of date; and
- reader-facing provenance and method-status metadata.

Then validate locally:

```bash
.venv/bin/python nba_pipeline/scripts/publish_supabase_results.py \
  --dataset <dataset_slug>
```

The command is dry-run by default. It checks the file, exact schema, row floor,
as-of value, non-null unique row keys, typed JSON conversion, content hash, and
deterministic version ID without touching Supabase.

Publish only after reviewing that receipt:

```bash
.venv/bin/python nba_pipeline/scripts/publish_supabase_results.py \
  --dataset <dataset_slug> \
  --publish
```

The publisher reads `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` from the
repo `.env`, uploads in batches, promotes only the complete version, and then
verifies the current-view count. Transient HTTP transport failures receive at
most three short retries; rerunning the same content is safe because its
version and row keys are deterministic. Never put the service-role key in a
browser, site bundle, committed file, or `NEXT_PUBLIC_` variable.

## Read from a site

Use the project URL plus the current enabled Supabase publishable key. New
sites should name the client variables clearly:

```text
NEXT_PUBLIC_SUPABASE_URL=https://bvwahamwfvolchcezxnw.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=<current sb_publishable_... key>
```

Example with `@supabase/supabase-js`:

```ts
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!,
);

const { data, error, count } = await supabase
  .from("analytics_current_rows")
  .select("row_key,row_ordinal,payload", { count: "exact" })
  .eq("dataset_slug", "vpm_epm12_x2_research_board")
  .order("row_ordinal")
  .range(0, 999);

if (error) throw error;
```

Supabase/PostgREST responses are paginated; fetch subsequent ranges for boards
over 1,000 rows. The JSON object under `payload` preserves native numbers,
booleans, strings, and nulls from the registered CSV. Query
`analytics_datasets` first when a site should discover available public
datasets and their research/production labels.

For the typed no-prior history, query the table directly:

```ts
const byPlayer = await supabase
  .from("nba_no_prior_vpm_td550_snapshots")
  .select("*")
  .eq("player_id", 1628983)
  .order("snapshot_date");

const byMonth = await supabase
  .from("nba_no_prior_vpm_td550_snapshots")
  .select("player_id,player_name,snapshot_date,vpm_off,vpm_def,vpm_net")
  .eq("snapshot_year", 2026)
  .eq("snapshot_month", 8)
  .order("vpm_net", { ascending: false });
```

Use the current enabled `sb_publishable_...` key. The legacy anon keys in an
older local `.env` may be disabled even when the table's public RLS policy is
correct.

## Admission rules

Use the generic hub for curated reader-facing boards, summaries, and small
historical panels. Do not publish:

- raw licensed provider rows, shot/event feeds, private model features, PII,
  auth data, credentials, or large intermediate diagnostics;
- a research result without its exact status and interpretation limits; or
- generated artifacts merely because they exist.

High-traffic or relationally complex products should graduate to a dedicated
typed table and migration. Keep their entry in this publication manifest or
record an explicit product-specific replacement so the result remains
discoverable.

## Verification receipt

On August 25, 2026, the first two registered datasets were published to project
`bvwahamwfvolchcezxnw`. The current publishable key returned exactly 1,556 and
11,604 rows through `analytics_current_rows`. Supabase security advisors found
no result-hub findings. The only immediate performance notice was the expected
"unused index" notice for the newly created latest-version foreign-key index;
the preceding missing-index notice was resolved.

On August 26, 2026, the two NBA rZTOV series and current WNBA rZTOV board were
published. Database receipts and the public publishable-key path both returned
exactly 6,186, 5,370, and 229 rows respectively. The raw all-era role backcast,
modern overlap diagnostics, RTSS training labels, and provider-derived feature
rows remain outside the public hub.

Later on August 26, the 890-row
`wnba_box_estimated_rztov_2022_2026` history was published. The database receipt
and public publishable-key path both returned all 890 rows spanning 2022-2026.
Every row includes a historical-fit status, and the underlying RTSS labels and
price-training records remain private.

On August 31, 2026, the dedicated no-prior VPM snapshot table was created and
all 34,075 rows were uploaded. The database returned 28 distinct dates,
2019-10-01 through 2026-08-30, with no duplicate keys and no nonzero-prior
rows. RLS is enabled, anonymous/authenticated roles have select only, the
service role owns writes, and the current publishable key returned a public
player/date query successfully. No table-specific security advisor finding
remained; the performance advisor reported two expected `unused_index` info
notices immediately after the new indexes were created.

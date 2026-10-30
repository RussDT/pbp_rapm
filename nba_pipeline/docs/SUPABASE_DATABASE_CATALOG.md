# Local Supabase Database Catalog

The local catalog is the owner-facing inventory of the `databallr` Supabase
project. It replaces the need to browse a flat table list in Supabase Studio.

For publishing repo result CSVs into the project for site use, see
[Supabase Results Publishing](./SUPABASE_RESULTS_PUBLISHING.md). The catalog
describes the database; the results publisher owns dataset registration,
versioning, upload, and site-facing access.

The catalog predates the August 31 typed
`public.nba_no_prior_vpm_td550_snapshots` table. Until the next full catalog
refresh, use [Supabase Results Publishing](./SUPABASE_RESULTS_PUBLISHING.md) for
its schema, access contract, and verified row receipt.

Open the generated catalog directly:

- `nba_pipeline/docs/supabase_database_catalog.html`
- raw metadata: `nba_pipeline/docs/supabase_database_catalog.json`

Both generated files are intentionally gitignored. The generator and HTML
template are versioned.

## What the catalog contains

The August 25, 2026 build inventories:

- 15 non-Postgres-internal schemas;
- 412 relations, including 391 tables, 18 views, and 3 materialized views;
- 343 application tables after separating Supabase platform schemas;
- 988 functions and procedures;
- 14 enums, 12 installed extensions, and storage-bucket metadata;
- columns, data types, defaults, comments, primary/foreign keys, constraints,
  indexes, triggers, RLS state and policies, relation sizes, and estimated row
  counts.

Tables and views receive one navigation domain such as NBA data and metrics,
WNBA data and metrics, draft and scouting, games and multiplayer, contracts and
free agency, or pipelines and staging. These labels organize the catalog; they
do not change database schemas or ownership.

The catalog is metadata-only. It does not copy table records, auth identities,
storage object names, database credentials, service keys, or function bodies.
Row counts come from Postgres statistics and are estimates rather than live
`count(*)` scans.

## Refresh

The machine must have an authenticated Supabase CLI session. From the repo
root, run:

```bash
python nba_pipeline/scripts/build_supabase_catalog.py \
  --project-ref bvwahamwfvolchcezxnw \
  --project-name databallr
```

The script initializes and links a temporary Supabase CLI work directory,
queries the remote Postgres catalogs through the Management API, writes the two
local artifacts, and removes the temporary directory. It does not use the
repo's service-role key or write to the database.

## Preview actual rows

Each relation detail view has a **Copy 100-row preview SQL** button. Run the
copied SQL with the CLI through a temporary linked project or an existing linked
Supabase worktree. Do not paste row output into the catalog because the table
may contain user data.

## Security review flag

The current build finds RLS disabled on these application tables:

- `history.hist_awards`
- `history.hist_game_coverage`
- `history.hist_games`
- `history.hist_player_game`
- `history.hist_player_season`
- `history.hist_players`
- `mp.chat_moderators`
- `mp.matchmaking_attempts`

This is an explicit review item, not permission to enable RLS automatically.
Enabling RLS without the necessary policies can block legitimate application
reads and writes. Determine which schemas are exposed through the Data API and
the intended server/client access contract before applying changes.

## Validation receipt

The catalog was regenerated against the live project and browser-checked at
desktop and 390-pixel mobile widths. Search, domain/schema/risk filtering,
table drill-downs, relationship links, and long table-name wrapping were
verified. The generated HTML is standalone and has no runtime network calls.

# TD550 Net DECOMP rolling archive in Supabase

The full no-prior TD550 Net SQ DECOMP archive is published to Databallr's
`public.nba_no_prior_vpm_td550_rolling`. It contains 92,125 player-window rows
across 140 windows: 30 one-year, 29 two-year, 28 three-year, 27 four-year, and
26 five-year windows covering 1997-2026. The first full five-year window is
1997-2001. This is the component-alpha research series, with Russell SQ v2
expected value plus realized residuals, RS+PS, smooth rubberband, season
terms, and 550-day time decay. It has no player priors.

## Consumer contract

The primary key is `(player_id, start_year, end_year)`. Filter on `end_year`
and `window_years` for a leaderboard, or `player_id` and `window_years` for a
player history. `start_year` and `end_year` are season-ending years, not exact
as-of dates. Season windows must not be mixed into the dated snapshot table.

All original CSV fields are retained under lowercase names. `estimate_off`,
`estimate_def`, and `estimate_net` are the decomposed totals; defense is
positive-is-good. `parent_*` is a separate common-alpha comparison, not the
component-alpha headline. Additional fields include `window_years`,
`prior_scale=0`, `status=research-only`, `season_type=ALL`, `component_alphas`,
and the source CSV SHA256. Metric units are points per 100 possessions.
Older source files retain three-decimal rounding; displayed additive sums
can differ by at most 0.001. Publication preserves source values.

Example: filter `end_year=2026&window_years=5` for 2022-2026. The archive has
public SELECT access with RLS, and only the service role can publish writes.

This publication adds the data source; the legacy site's existing dated
snapshot and precomputed profile-factor reads are unchanged. Its current
`nba_no_prior_vpm_td550_snapshots` and `nba_td550_profile_factors` tables remain
separate. Displaying the rolling archive requires the site to query this new
table and offer season/window selection.

## Reproduction and verification

Run `nba_pipeline/scripts/publish_td550_rolling_archive.py` to validate the
complete CSV/parquet grid and create the local publication bundle. Add
`--publish` to upsert the archive, then read every row back and compare every
field with the local source. Authentication uses the existing `.env` service
role; credentials are never included in artifacts.

Sources are the canonical
`master_results/net_sq_decomp_rolling/weighted_factors_vpm_net_sq_resid_*_td550.csv`
and matching parquets/manifests, including the completed FT fallback repair.
Generated source receipts, row bundle, and final verification are under
`results/td550_rolling_publication/` and remain untracked. The schema is recorded
in `supabase/migrations/20260907014025_nba_no_prior_vpm_td550_rolling.sql`.

Publication completed 2026-09-06: all 92,125 remote rows and every field were
read back and matched to the canonical local archive. Database window counts
and public-read/service-write access controls were verified independently.

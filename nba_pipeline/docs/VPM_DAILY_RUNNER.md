# Nightly full-feature VPM runner

Builds the current-season, full-feature Parent VPM from the Cloudflare R2 feed
and publishes it to `nbarapm.com/datasets/VPMPreview`. It only applies frozen
models to new games. It never fits a mapper, a prior or an SQ model.

## Schedule

launchd job `com.russellthomas.vpm-daily`
(`~/Library/LaunchAgents/com.russellthomas.vpm-daily.plist`), 7:30 AM local
time. If the Mac is asleep, launchd runs it on wake. It runs
`scripts/vpm_daily/run_nightly.sh`; logs are in `logs/vpm_daily/{YYYYMMDD}.log`.
A failure shows a macOS notification.

## What one run does

1. The cut is today in America/Los_Angeles; games through yesterday are used.
2. Completeness gate: ESPN's final count for yesterday must equal the R2
   manifest count. It waits up to `--wait-hours` (default 4), then fails. The
   NBA CDN schedule returns 403, so ESPN is the reference.
3. Snapshots R2, renders the Nov 3 stage code for the cut date
   (`templates/*.py.tmpl`, `render.py`), then builds features, tracking, SQ
   atoms and the board.
4. Numeric gates: normal-equation error <= 1e-10, native Net gap <= 1e-9, no
   fits, O + D = total.
5. `publish_vpm_preview.py --publish` inserts one row into
   `public.nba_vpm_preview_boards` and reads it back. It skips a board that is
   not newer than the live one or adds no games, and it fails if the player
   count drops more than 5%.
6. `prune_stages.py --keep 3` removes the bulky base and tracking folders of
   older nightly builds that have a publication receipt. Hand-run builds through
   2026-11-03 and all small `_sq_v1` / `_sq_board_v1` receipts are kept.

The site shows a new board within about 5 minutes on the feed and 30 minutes on
the page (`revalidate = 1800`).

## Manual commands

Run from `nba_pipeline/` with `../.venv/bin/python`:

```bash
../.venv/bin/python scripts/vpm_daily/run_vpm_daily.py --cut 2026-11-09 --wait-hours 0
```

```bash
../.venv/bin/python scripts/vpm_daily/publish_vpm_preview.py results/estimated_skills/current_refresh_20261109_sq_v1/supabase_preview.json
```

Publishing is a dry run unless `--publish` is passed. `--tag NAME --from-saved`
rebuilds into a separate folder from a saved snapshot.

## Evidence

- Replaying Nov 3 from saved inputs reproduced the live board exactly: rankings
  642 x 168, canonical arrays and metadata byte-identical, Supabase players
  equal. Only path-bearing manifest hashes differ.
- First real run: 2026-11-09, 146 games, 641 players, read back from Supabase.

## Known limits

- Runs on the Mac, so a powered-off Mac means no update.
- R2 auth uses the wrangler OAuth login (databallr@gmail.com) unless
  `CLOUDFLARE_API_TOKEN` is set. A read-only R2 token would be sturdier.
- A date with no games the night before has not been exercised end to end.
- SEASON is fixed at 2027; the runner refuses other seasons.
- The frozen Oct-1 mapper, SQ models and `current_model.json` are not refreshed.
- This remains a preview board. It makes no forecast-superiority claim.

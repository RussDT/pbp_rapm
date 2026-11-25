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

Because this checkout is in `~/Documents`, macOS must allow the unattended
launcher to read that protected folder. The LaunchAgent and script use
`/bin/bash`, which already has Full Disk Access on this Mac. The original
LaunchAgent incorrectly used `/bin/zsh`, which lacked that permission. It
exited 127 before opening `run_nightly.sh`, so the script's own logging and
failure notification never ran. This interpreter mismatch caused the missed
November 10-15 scheduled runs; the November 12 publication was a manual run.
The November 15 recovery was also manually published, through November 14.
The corrected bash launch passes an actual unattended file-access and Python
dependency check. Launch-level failures are in `logs/vpm_daily/launchd_stderr.log`.

### November 20-25 failures: node refused under `~/Documents`

Every scheduled run from November 20 through 25 failed in the first second,
when `R2Reader` called `wrangler auth token --json`. The same command worked
from an interactive shell and under `env -i`. The reader threw wrangler's
stderr away, so the cause was invisible. Once stderr was captured, a launchd run
showed node itself was refused:
`Error: EPERM: operation not permitted, open '<databallr>/node_modules/.pnpm/wrangler@4.151.0_.../wrangler/bin/wrangler.js'`.
Under launchd, bash and Python can read the checkout, but node cannot open any
file below `~/Documents`. This is most likely macOS privacy protection on
protected folders (the same class of problem as the zsh launch above), but
that is an inference.

Fix (no privacy-setting change, sign-in or new token):
`databallr_r2_read.py` now copies wrangler's pnpm dependency closure (38 store
entries, about 330 MB) into `~/Library/Caches/rapm-databallr-wrangler/<wrangler
store entry>/`. Python does the copy, and pnpm's relative symlinks are kept.
Node then runs `wrangler.js` from that copy, and it uses the same wrangler OAuth
login in `~/Library/Preferences/.wrangler`. The copy is rebuilt automatically
when databallr's wrangler version changes. If wrangler fails, the
nightly log now shows its exit code and its stderr/stdout, with token-like
values, bearer strings and 32-hex ids redacted.

The same launchd run found a second, unrelated stall. This network has no
working IPv6 egress to `api.cloudflare.com`, and that host resolves IPv6 first.
Python 3.9's `urllib` gave each address the full 120 s timeout before trying
IPv4, so a single R2 GET took about 12 minutes. Every Cloudflare API call in the
reader (object GETs, listings and the catalog lookup) now gives each address
10 s to connect, alternate IPv6 and IPv4, and remember the family that worked.
Reads still have the 120 s timeout. Supabase and ESPN resolve IPv4-first and
were not affected. Tests: `tests/test_databallr_r2_read.py`.

## What one run does

1. The cut is today in America/Los_Angeles; games through yesterday are used.
2. Completeness gate: ESPN's final count for yesterday must equal the R2
   manifest count. It waits up to `--wait-hours` (default 4), then fails. The
   NBA CDN schedule returns 403, so ESPN is the reference.
3. Snapshots R2, renders the Nov 3 stage code for the cut date
   (`templates/*.py.tmpl`, `render.py`), then builds features, tracking, SQ
   atoms and the own-only comparison board. By default, it then restores the
   frozen clean-v61 entrant/background prior before re-solving the same Parent
   likelihood. The own mapper, native geometry, penalties and SQ corrections
   remain unchanged.
4. Numeric gates: normal-equation error <= 1e-10, native Net gap <= 1e-9, no
   fits, O + D = total. The public reporting layer then centers final O/D and
   each final component on the complete active365 player pool, using each
   side's original TD550 possession weights. The dated pool and offsets are
   fixed before any site search/filter. Saved model boards and all priors
   retain their original values; rankings and lineup predictions are unchanged.
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

`--prior-mode own-only` keeps the comparison control. The default is
`--prior-mode entrant`. The restored board, actual background/combined arrays
and same-cut comparison CSV are saved in a separate `_entrant_board_v1` folder;
the original compact own-only board is retained.

## Entrant/background prior

The frozen 2025 clean-v61 component model supplies background ability. It is
blended with the current full-feature own prior using
`background_weight = max(1 - career_NBA_possessions / 2000, 0)` before the lineup
solve. Career exposure uses the original historical counts and exact dated
current native appearances, separately from feature masks and TD weights.
Veterans at 2,000 possessions retain their exact own prior. Unidentified
low-experience players at the archive's 1997 boundary retain own-only priors
with a missing reported background weight; current preview players have
identified weights.

Completed 2026 college counts and birthdates are fetched from Supabase
`draft_ready_data`, joined by unique full name with conflicting NBA IDs
rejected, and transformed by the original frozen estimator. Its `draft_pick`
field is deliberately excluded: the frozen actual-draft source ends in 2025.
Missing backgrounds use the model's learned population/missing-data fallback.
No entrant coefficients, own mapper or SQ model are refitted.

## Public reporting baseline

`preview_centering.py` applies one side-specific weighted mean subtraction per
final atomic component, with matching parent/leaf PPP-intercept adjustments.
Rollup offsets preserve the shot-value, turnover, second-chance and O+D sums.
Metadata `reporting_centering` saves the schema, exact dated player-set digest,
weight totals, every field offset and intercept adjustments. Priors remain
the frozen models' original ability estimates and are not separately centered.

The November 15 board's reporting offsets are +0.1999874085 offense and
+0.0978532657 defense, or +0.2978406742 total. Its 643-player publication was
updated atomically with an exact old-content guard; both its previous payload
and the centered publication/read-back receipt are retained under
`current_refresh_20261115_schedule_recovery_sq_v1/active_centering_v1/`.
The daily builder uses this reporting baseline automatically for future dates.

## Evidence

- Replaying Nov 3 from saved inputs reproduced the live board exactly: rankings
  642 x 168, canonical arrays and metadata byte-identical, Supabase players
  equal. Only path-bearing manifest hashes differ.
- First real run: 2026-11-09, 146 games, 641 players, read back from Supabase.
- Entrant restoration: 2026-11-12, through November 11, 170 games and 641
  preview players. Zero background weight recovers the own-only board within
  5.33e-15; actual restored normal-equation error is 4.68e-15 and native Net
  closure is 1.71e-15. All 20 persisted arrays read back exactly. Same-cut
  changes are retained in `own_only_comparison.csv`; this rerun is not a new
  predictive-performance test.

## Known limits

- Runs on the Mac, so a powered-off Mac means no update.
- R2 auth uses the wrangler OAuth login (databallr@gmail.com) unless
  `CLOUDFLARE_API_TOKEN` is set. A read-only R2 token would be sturdier.
  Under launchd, wrangler runs from the `~/Library/Caches` mirror described
  above. If the OAuth login lapses, the nightly log shows wrangler's redacted
  error. Re-signing in (`wrangler login`) is an owner step.
- A date with no games the night before has not been exercised end to end.
- SEASON is fixed at 2027; the runner refuses other seasons.
- The frozen Oct-1 mapper, SQ models and `current_model.json` are not refreshed.
- This remains a preview board. It makes no forecast-superiority claim.

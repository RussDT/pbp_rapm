# Codex Scratchpad

This file is short-term session memory only.
Durable operating rules belong in root `AGENTS.md` and `CLAUDE.md`.

## How To Use

1. Read the latest entries at session start.
2. Append one short entry at session end.
3. When a lesson becomes durable, move it into `AGENTS.md` / `CLAUDE.md`.
4. Keep only the latest 5 entries in this file.

Never store secrets, tokens, or credentials here.

## Session Log Template

```md
### YYYY-MM-DD HH:MM (local)
- Task:
- What went wrong:
- User correction:
- Fix applied:
- Prevention rule added:
- Outcome:
```

## Session Log

### 2026-06-09 21:30 (local)
- Task: Re-fetch modern 2014-2025 (NBA14..NBA25 RS+PS) raw PBP + rotations from the API after cleanup removed `raw_data`.
- What went wrong: stats.nba.com silently stalled `requests`/plain-curl (Akamai TLS-fingerprint block, looked like an IP ban); separately, `gamerotation` 500s from the home IP and rate-limits hard under concurrency.
- User correction: User wanted everything 2014+ with rotations attached; "try without proxy, be careful about rate limits."
- Fix applied: Added `scripts/_nba_http_curlcffi.py` (monkeypatch nba_api -> curl_cffi chrome120) and imported it in `01_fetch_pbp_data.py`; gave `gamerotation` 10 retries with exponential backoff (cap 45s); added resumable chunked driver `scripts/fetch_modern_seasons.py`. Winning config = curl_cffi + proxy ON + low concurrency.
- Prevention rule added: stats.nba.com needs a browser TLS fingerprint (curl_cffi); gamerotation needs the proxy and gentle pacing. See AGENTS.md / CLAUDE.md.
- Outcome: End-to-end verified (single game = 492 events, 100% 10-man lineups). Full chunked refetch driver in place; tuning concurrency for gamerotation throttle.

### 2026-06-09 02:15 (local)
- Task: Restore Gabe historical raw and processed parquets after cleanup removed generated artifacts.
- What went wrong: The broad generated-artifact cleanup removed local `raw_data` / `processed` parquets that were still needed.
- User correction: User needed the Gabe historical parquets first, before any modern 2014-2026 work.
- Fix applied: Restored Gabe `old_data`, repaired two zero-byte source parquets from the Gabe git object store, rebuilt 1997-2013 RS/PS historical raw and 18-metric processed parquet surfaces, then targeted the missing 2006 pass.
- Prevention rule added: Before deleting generated parquet surfaces, confirm whether the user needs local rebuildability or current artifacts; keep a restoration command/source note for any removed historical data.
- Outcome: Historical Gabe audit is clean: 34 raw parquets and 612 processed parquets, with no missing 1997-2013 RS/PS files.

### 2026-06-08 23:18 (local)
- Task: Clean local generated artifacts after adding cleanup/onboarding docs.
- What went wrong: A blanket `git clean -fdX` would have removed `.env` and local agent/editor config.
- User correction: User asked to actually clean the repo.
- Fix applied: Removed ignored generated data, caches, logs, validation outputs, raw/processed/results/master output dirs, WNBA output CSVs, root scratch CSV/parquet files, and the ignored external `merged_playbyplay` checkout while preserving `.env`, `.claude/`, `.cursor/`, `.gstack/`, and `nba_pipeline/.claude/`.
- Prevention rule added: For cleanup, preview `git clean -ndX`, then use targeted removal when ignored local config should survive.
- Outcome: Checkout dropped to `32M`; repeatable inventory now reports `264` files outside skipped dirs and `2` artifact-like files.

### 2026-06-08 23:12 (local)
- Task: Prepare the repo for a friend/collaborator and start organizing dead-code/artifact cleanup.
- What went wrong: N/A.
- User correction: User wanted action, not just a recommendation.
- Fix applied: Added contributor onboarding, artifact manifest, cleanup plan, and a dry-run repo inventory script; linked the docs from the FreshDocs index and durable agent instructions.
- Prevention rule added: Repo cleanup should start with dry-run inventory and script classification before deleting artifacts or moving legacy code.
- Outcome: `audit_repo_inventory.py` generated ignored reports under `nba_pipeline/validation/repo_inventory/`; the new script compiles successfully.

### 2026-06-03 14:45 (local)
- Task: Add PBP-derived Databallr WOWY team-player presence for 1Y-5Y RAPM DECOMP.
- What went wrong: The existing lineup helper inferred teams from processed `h*/a*` columns, so the new uploader needed raw `home_player*/away_player*` handling instead of importing that helper directly.
- User correction: User required RAPM DECOMP from `player_alt3_efg_factors` and team membership from PBP, not `timedecay_rapm`.
- Fix applied: Added `upload_wowy_team_player_presence.py`, created/uploaded `wowy_team_player_presence` for 2022-2026 RS+PS, wired the daily job, and documented the command/table contract.
- Prevention rule added: Databallr WOWY team RAPM membership should come from raw lineup presence and remain separate from current-strength `timedecay_rapm`.
- Outcome: Supabase has 4,412 presence rows for 2022-2026; OKC/PHX/BKN live counts match local PBP-derived validation with zero duplicate season/type/player keys.


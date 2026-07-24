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

### 2026-07-23 19:33 (local)
- Task: Publish the already validated Help Index production build to its existing Sites project.
- What went wrong: The site's actual access metadata resolved as owner-only custom access, despite the request describing it as public, so the open-world deployment path was not applicable.
- Fix applied: Reused the existing project and source repository, committed and pushed only the nested app repository, packaged the existing `dist/` without rebuilding, saved version 8 from the pushed commit, and deployed through the verified owner-only path without changing access.
- Prevention rule added: Resolve actual Sites access metadata before choosing the deployment tool; preserve that access rather than inferring it from the site's URL.
- Outcome: Deployment succeeded at the existing `nba-help-index.russdt.chatgpt.site` URL.

### 2026-07-23 19:30 (local)
- Task: Rebuild The Help Index as a redesigned editorial data product (new frontend architecture, same data contracts) and redeploy.
- What went wrong: The single-file dashboard mixed state, data, and layout; the dark hero/sidebar design fought the numbers; and in dev the server transparently decodes `.json.gz`, so blind `DecompressionStream` piping aborted every data fetch.
- Fix applied: Split the app into a normalized view-state container (`help-index-app.tsx`), `lib/` data+format layers with cached pair-shard loads, and `components/` (search, zero-baseline career arc with pooled combined seasons, Lineup Company card, rankings); rebuilt the CSS as a light navy/gold/blue system with geometry-stable interaction states; made the fetch helper sniff gzip magic bytes before decompressing.
- Prevention rule added: URL state, view normalization (clamp to seasons a player actually has), and expansion-reset-by-context-key live in the container, not effects; gzip fetch helpers must handle Content-Encoding-decoded responses.
- Outcome: Lint clean, build passes, five tests pass (including a new all-schedule career pair-closure test on a second shard), desktop/mobile browser validation done, and the redesign was republished to the existing Sites project.

### 2026-07-23 18:23 (local)
- Task: Test whether the Embiid-style shooter/zone OREB continuation improves forward shot-release value prediction.
- What went wrong: Immediate `initial_ev` and shot-plus-second-chance targets are not directly comparable, and the first full run retained large parquet buffers and used an allocation-heavy lineup loop.
- Fix applied: Added a strict 2022-24 to 2024-25 first-FGA release benchmark, gave provider and internal SQ models identical contextual continuation lanes, added the shooter-specific nested ablation and role swap, pushed parquet filters/released Arrow buffers, and vectorized lineup parsing.
- Prevention rule added: Judge shooter-specific second-chance value against the same immediate model plus shooter-hidden continuation; never call the automatic gain from adding continuation to a continuation-bearing target an SQ victory.
- Outcome: Shooter × zone gives a small clear combined gain for context+EPM and geometry+identity, but the full geometry model remains tied with provider `initial_ev` given the same continuation.

### 2026-07-23 18:14 (local)
- Task: Remove Help Index layout jank and make both teammate and career-help rankings expandable.
- What went wrong: Yellow top rules felt ornamental, selected leaderboard rows changed padding and shifted their columns, season context was visually buried, six teammate rows and twelve leaderboard rows were hard truncations, and weighted help visually replaced the teammate's own impact.
- Fix applied: Removed the top rules, made leaderboard states geometry-invariant, promoted the season/career label, made as-of DPM impact the primary teammate value, opened teammate rows at 10 with +10 expansion, and opened the league board at 25 with +25 expansion.
- Prevention rule added: Interactive states must never change row box metrics; pair contribution stays secondary when the reader is asking for the teammate's own impact.
- Outcome: The production build and all four app/reconciliation tests pass.

### 2026-07-23 17:58 (local)
- Task: Convert Lineup Company into a coherent share card and clarify teammate value and coverage.
- What went wrong: Orange conflicted with navy/yellow, Avg DPM and help looked duplicative, unloaded cards collapsed, and teammate names were not actionable.
- Fix applied: Unified the navy/yellow palette, embedded focal player and context in a stable-height card, made rows clickable, added context-preserving share links, prioritized rank/percent together/help, and added per-pair rated coverage and definitions.
- Prevention rule added: Label a teammate's own rating separately from his weighted help contribution; never hide pair-level historical DPM coverage.
- Outcome: Four app tests and the production build pass; null career-pair DPM affects 18.62% of RS, 10.35% of PS, and 17.89% of combined rows, while Paul Pierce's top RS teammates have 61.6%-95.7% rated coverage.

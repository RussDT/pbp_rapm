# Saved component VPM in the private reader

The owner requested the other component architecture alongside own-skill SQ
parents. The reader adds the saved incumbent v4 x2 as-of2026-08-21, games
through2026-06-13,582 current-PBP players. Source is sibling
`vpm/results/harness/vpm_offseason_alpha_incumbent_board_v1_2026-08-21/player_estimates_incumbent_v4_x2_2026-08-21.csv`.

It uses EPM predictive-skill component priors x2 and TD550, with no full-EPM
blend. It has not been rebuilt with own tracking/foul-drawing skills and is
not the corrected makes/attempts challenger. Preserve the historical feature
representation label. Do not present it as the September14 own-skill model
or as untouched predictive confirmation.

Export `build_vpm_component_site_payload.py` uses only saved impact outputs
and exact player IDs; no raw provider skill fields. Validate both ten-child
O/D sums and O+D=net. Reader main view shows scoring/turnovers/second chances;
side tabs and player details show ten component contributions each. Snapshot
date and architecture are model-specific. Earlier parent options remain.

Owner follow-up: Site is public; default restored to Expanded + fouls drawn · research. All33 components remains selectable. Source revision f7b96eee75ef35f904b3383619dc5b919558b81b, Site version8.


September 20 comparison update: `build_vpm_epm_site_comparison.py --date 2026-09-20` exports only EPM/O-EPM/D-EPM impact outputs, joined by unique NBA ID to the fixed Expanded + fouls drawn September14 board. All582 players match. The freshly requested September20 game-agnostic predictive snapshot is provider-effective June13; both dates are visible. Differences are VPM minus EPM, not accuracy evidence. The comparison tab is independent of the model selector.

Component display names now resolve by exact NBA ID through the full-name own-skill board. The saved component source used latest boxscore names; nonempty surnames survived the null-only fallback. Both component exports had96 surname-only players and122 total name spelling/format replacements. All numerical fields and IDs remain identical. Keep this identity resolution in the shared exporter for future rebuilds.

Minimum possessions filter defaults to1000 across every model and tab, including EPM comparison. It uses (offensive training possessions + defensive training possessions)/2, not their sum or current-season-only minutes. Set0 to show everyone. Comparison export carries the same fixed v27 possession counts.

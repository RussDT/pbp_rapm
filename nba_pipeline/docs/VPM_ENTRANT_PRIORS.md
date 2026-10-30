# Predictive VPM entrant priors

The September 26 rookie/G League repair is owned by the sibling VPM harness.
Read its [canonical methodology and receipt](../../../vpm/vpm/docs/ENTRANT_PRIORS.md).

The new `entrant-priors` command recovers dated lower-league counting evidence,
audits draft/college/NBA identities, fits population/draft/stats/combined O/D
prior challengers without a minimum NBA-minute label gate, and supports an exact
prior-delta integration into the frozen own-skill SQ Parent. Original v18 is
preserved and has been reproduced with zero rating difference.

Alex Antetokounmpo (`1630828`) now has 98 identified G League games / 1,813
provider-recorded minutes available to the candidate, rather than only his
45 NBA possessions on each side. He is an audit case, never a rating override.

The source audit found incorrect populated NBA IDs in college data and same-name
draft collisions. Never consume these tables as an already-certified crosswalk.
Unknown draft status is not synonymous with undrafted. Missing background data
must remain visible and receive a fitted population fallback, not silent zero.

This remains research. Source vintages, individual uncertainty, live roster
coverage, and prospective 2026–27 confirmation are distinct requirements. No
public board or historical v18 source was overwritten.

Final evidence is in sibling `vpm/results/harness/vpm_entrant_priors_v1_final/`:

The [interactive HTML board](../../../vpm/vpm/results/harness/vpm_entrant_priors_v1_final/entrants/own_parent_2026-09-14/vpm-board.html)
shows original/updated VPM, change, search, exposure filters and prior details.
The comparison reveals broad level movement: 579/582 decline, median −0.640;
established players (20,000+ NBA possessions) average −0.407. It is not a
rookie-only rating adjustment; no display recentering was applied.

- All 582 current-PBP players have finite audited priors; 575 have measured
  draft/college/G League background and seven have explicit fitted fallbacks.
- Alex's background prior is −4.782, combined prior −4.650, and integrated VPM
  −4.604, compared with +1.241 in frozen v18. His 45 NBA possessions receive
  2.20% of the prior mean weight; no player-specific rating override is used.
- Combined draft+stats improves on draft-only in chronological development.
  Inside own-SQ VPM, the 2025 actual-point replay improves over the coherent
  control by +505e-6 MSE [264, 744]; the low-experience group improves +707e-6
  [413, 1042]. These remain spent development results.
- The standalone pure-RAPM primary screen rejects the same prior candidate
  (development −194e-6 [−304, −77]); its positive secondary game-margin check
  does not override that result. No silent promotion.
- The existing source draft table ends in 2025. A 2026 draft and 2026–27 roster
  identity intake is still required before a live next-season launch. The
  inference code supports zero-NBA rookies, but 582 current-PBP players must not
  be described as complete future-roster coverage.

Sixteen focused tests pass; exact reconstruction of both saved parent cuts
has maximum native coefficient difference below 1.8e-15.

# Same-sample regular-season rZTOV correction

## Root cause

Both sources were regular season, but the reader export was a different snapshot.
For Derrick White in 2026, `merged_rs_full.csv.gz` has 64 games, 4,336 OffPoss,
972 FGA, 186 FTA, and 1,049 TSA (24.192804 TSA/100). The raw/processed RAPM source
has 77 White games, 5,157 offensive possessions, 1,108 FGA, 204 FTA, and 1,193
PBP shooting opportunities (23.133605/100).

The raw and processed full-season files each contain 1,230 unique games, all
with normalized `002` IDs. The processed terminal events include all 35,695
raw turnover events. This was not playoff contamination or removal of turnovers
from the possession denominator. It was an invalid cross-snapshot rate/count join.

## Corrected contract

`build_nba_rztov_box_backcast.py` now computes FGA plus one opportunity at the
first nontechnical FT of each two/three-shot trip directly from the RS raw file.
And-one and technical one-shot FTs add no extra attempt. The denominator comes
from the same season's RS processed possessions. Numeric historical IDs are
normalized before requiring the `002` regular-season prefix.

Before scoring, the builder requires source FGA, FTA, and OffPoss to equal the
cached model feature counts/exposure for every player-season. A mismatch fails;
there is no fallback to a reader rate from another snapshot. Model features and
predicted assisted shares/potential assists stay on their existing definition;
only shot counts are partitioned on the corrected sample. The role CSV exposes
`canonical_tsa_count` and an explicit same-sample RS `tsa_source`. Export full
precision so self-created plus assisted closes at floating-point tolerance.

All 14,595 NBA role rows from 1997-2026 were regenerated, along with the rolling
historical prices and turnover ledgers. Existing observed modern references
remain separate and unchanged. This correction does not certify provider-label
game coverage or admit a predictive model; observed role-rate comparisons remain
development diagnostics, with their documented label-source limitations.

The team-page builder also explicitly filters RS, requires the corrected source
contract, and counts each traded player's TSA on the actual team. It applies the
estimated season assisted share to those team-stint counts. Potential-assist
estimates remain a season rate applied to exact player-team exposure. All team
shooting counts must close to the player-season source counts. The separate
team-turnover league baseline remains 0.716850/100.

## Corrected 2026 results

| Quantity | Boston | Portland |
| --- | ---: | ---: |
| Full RS offensive possessions | 7,754 | 8,294 |
| Same-sample shooting opportunities | 8,048 | 8,280 |
| TSA per 100 team possessions | 103.79 | 99.83 |
| Player-workload expected TOV/100 | 14.52 | 13.52 |
| Total expected TOV/100, including team baseline | 15.24 | 14.23 |
| Actual total TOV/100 | 13.08 | 17.08 |
| Fewer than expected TOV/100 | +2.16 | -2.85 |

League player-turnover rate remains 13.830034/100. Boston's previous player
expectation was 14.61; the corrected value is 14.52. The snapshot error mattered,
but does not explain most of the high workload expectation. Boston still has
more actual shooting opportunities per possession and a higher estimated
self-created share. Multiple shooting opportunities within a possession are
possible; a TSA rate above 100 does not imply turnover possessions were omitted.

The team expected-versus-actual comparison remains descriptive and sensitive
to estimated role coherence. Fewer turnovers themselves leave more opportunities
to shoot, so this is not an independent forecast of turnovers.

## Validation and artifacts

All 17 focused tests pass. Tests explicitly cover playoff exclusion, historical integer game IDs, two- and
three-shot FT trips, and-one exclusion, and the fixed-TSA split. Rebuilt
role identity error is at most `7.2e-15`. All-player exposure, raw turnover,
team TSA, and league expectation closure checks pass.

Updated artifacts:

- `results/nba_rztov_box_backcast/`: same-sample role and turnover boards.
- `results/rztov_team_page_accounting/`: corrected full team accounting.
- `results/rztov_team_audit/`: refreshed source/workload diagnostics.
- `results/rztov_team_calibration/`: refreshed nested workload diagnostics.

The older NBA numerical examples in the original team audit/calibration/page
notes are superseded. No public database or player-page application was changed.

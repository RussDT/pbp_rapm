# VPM: the two model families

Use these names in owner-facing discussions. Experiment numbers identify saved
runs; they do not identify new architectures or establish a release order.

| Plain name | Construction | Current role |
|---|---|---|
| **Component VPM / Impact Monster** | Ten component types solved separately on offense and defense; components sum to O/D/total | Existing product architecture; the component launch candidate still needs a correctly scoped comparison |
| **Parent VPM** | One O/D parent with strongly shrunk midrange/three-point residual corrections | Research alternative; this is the family tested in the recent long-SPM/defense/FT runs |

## Latest forward-prior comparison

[Expanded forward priors and ratings](http://127.0.0.1:8767/ratings.html) retains Parent VPM and the previous repaired TD550 lineup solve. It learns the feature mapping from dated skills to future365-day actual lineup outcomes, rather than six-year same-window statistics or shrunken RAPM point labels. The page compares the forward+MPG candidate, feature-only old-label extension, previous repaired and the earlier six-year SPM. [Method, selection distinctions and evidence](VPM_REPAIRED_FORWARD_EXPANDED.md). This is a new prior-training comparison within Parent VPM, not a third architecture or a component launch choice.

[Predictive skills behind this same forward model](http://127.0.0.1:8767/skills.html) shows box-score, shooting, advanced and defensive inputs with per-player imputation/provenance. This is an input view, not another version. It defaults to582 players with recent playing exposure; the inherited585 recent-event flag includes three retired players with zero playing exposure. Full1556 ratings remain available.

## Earlier corrected six-year SPM comparison

The [player priors view](http://127.0.0.1:8766/priors.html) shows saved skill-only SPM and exact blended priors beside final ratings for all three comparisons. It is a view of existing fits, not a new model.

The new private reader [ratings comparison](http://127.0.0.1:8766/ratings.html) shows exactly three main choices: **Fixed fouls + contested DREB**, **Fixed fouls + total DREB**, and **Previous repaired VPM**. Both new choices have seven active numeric defense fields; contested replaces total DREB including its missingness flag. They retain the Parent architecture and actualFT, at4000/8000/TD275. See [completed controlled rerun](VPM_FIXED_FOULS_CONTESTED_DREB.md) for forecasts, TD550 checks, source/numerical audits and current ratings. This does not resolve the large big-player prior-mapping shift or select a component release.

## Earlier pre-repair ratings page

`http://127.0.0.1:8765/ratings.html` contains **Parent VPM only**:

- **Core defense:** blocks, steals and defensive rebounds as the defensive SPM
  prior inputs. The lineup model also learns from opponent points.
- **Core defense + expected FT:** same parent/prior, with dated expected FT in
  lineup deployment. It is a target variant, not a third architecture.
- **Expanded defense:** same parent architecture, with rim savings, rim contest
  volume and non-offensive PF added to the three core fields. Its attempted
  shooting-PF column was incorrectly all zero; corrected counts exist but
  the shown ratings have not been refitted. See
  [the source correction and overlap audit](VPM_DEFENSIVE_FEATURE_OVERLAP.md).
- Previous priors, different decay settings and blends are parent comparisons.

“Selected in development” means lowest 2020–22 game MSE among the registered
parent defense candidates, followed by its parent penalty/decay grid. The winner
uses actual FT and4000/8000/TD275. It does not mean best across architectures,
best on later seasons, component VPM, or launch approval. The component-prior
subexperiment used three broad target groups to inform a parent prior; it did
not emit the ten-component product board.

## Existing component board: verified artifact

The local product bundle is `vpm-explorer/public/data/vpm.csv` with
`vpm-manifest.json`: gross ten-component `epm12_x2`, estimateJuly24,2026,
games throughJune13,2026, TD550, predictive EPM skill priors x2. Its output has
20 O/D component columns and exact O/D/total closure. This identifies the
existing checked-out product bundle; no live deployment was inspected here.
It is distinct from the newer own-skill Parent VPM research. Historical source
and SQ-vintage caveats in the earlier receipts remain applicable.

## Scope correction, October 19

The agent incorrectly framed the parent experiment's core+expected-FT ratings
as the recommendation for launching component VPM. **That recommendation is
withdrawn.** The parent results remain valid for their named experiment; they
do not answer which component VPM to launch. The next launch comparison must
name the component config, dated prior/target inputs, component penalties,
component-sum board and matching forecast evidence explicitly.

Do not open another model family, infer “latest = best,” or make new launch
recommendations by mixing results from the two architectures. Start with this
map; keep research variants secondary to the architecture name.

## Why bigs moved versus the previous repaired parent

The large jumps already occur in the six-season **core** parent, before dated
expected FT, expanded rim/foul features or TD275. On the identical September1
cut, final lineup data,4000/8000 penalties,TD550, repaired SQ and entrant treatment
are held fixed; the prior calibration/mapping and defensive feature set change.
The old map regresses matured short-window RAPM labels on skill forecasts. The
new map profiles a matched six-season actual-lineup likelihood jointly with
player residuals and uses three core defensive fields. These are multiple
changes, not an isolated feature-count experiment.

|Player|Previous repaired D|New core D|Expected-FT increment|Expanded TD275 D|
|---|---:|---:|---:|---:|
|Wembanyama|3.807|6.997|+0.052|6.344|
|Paul Reed|1.741|4.025|+0.009|3.954|
|Robert Williams III|1.917|4.176|−0.054|4.856|
|Clingan|1.229|2.803|+0.168|3.138|
|Queta|1.758|3.125|−0.050|3.578|

Wembanyama's defensive prior mean rises1.742→6.327; Reed's1.671→4.750;
Williams's1.653→4.570. The final solve pulls toward those larger means under
unchanged penalties. Offensive prior mapping changes too: Clingan offense
0.619→2.210. The OREB slope in prior points/100 per extra OREB/100 rises
0.1678→0.5398. The conditional block/DREB slopes also rise substantially, but
the old map contains correlated rim/DFGA/PF fields, so individual slopes are
not directly comparable causal event prices.

This establishes the computational source of the increase, not that bigs are
systematically overvalued or that the old compressed estimates are correct.
Global game-forecast differences remain unresolved; the larger ratings need a
position/lineup calibration audit before they support a launch claim.
The exact saved-rating bridge for all1556 players is private sibling
`vpm/results/harness/vpm_launch_defense_v51/previous_to_parent_prior_changes.csv`
with source hashes and closure in `prior_shift_audit.json`. No refit, new score,
outcome access or rating change was performed for this diagnosis.

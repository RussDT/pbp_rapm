# VPM: the two model families

Use these names in owner-facing discussions. Experiment numbers identify saved
runs; they do not identify new architectures or establish a release order.

| Plain name | Construction | Current role |
|---|---|---|
| **Component VPM / Impact Monster** | Ten component types solved separately on offense and defense; components sum to O/D/total | Existing product architecture; the component launch candidate still needs a correctly scoped comparison |
| **Parent VPM** | One O/D parent with strongly shrunk midrange/three-point residual corrections | Research alternative; this is the family tested in the recent long-SPM/defense/FT runs |

## What the current ratings page contains

`http://127.0.0.1:8765/ratings.html` contains **Parent VPM only**:

- **Core defense:** blocks, steals and defensive rebounds as the defensive SPM
  prior inputs. The lineup model also learns from opponent points.
- **Core defense + expected FT:** same parent/prior, with dated expected FT in
  lineup deployment. It is a target variant, not a third architecture.
- **Expanded defense:** same parent architecture, with rim savings, rim contest
  volume, shooting PF and non-offensive PF added to the three core fields.
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

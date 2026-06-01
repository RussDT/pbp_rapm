# Decomp RAPM Expanded Production Prompt

## Title + Style

Create a 40-second 1920x1080 HyperFrames explainer called `Decomp RAPM`. Use the local design system: background `#10110f`, panel `#181a17`, ink `#f3ead7`, muted ink `#b9ad98`, offense `#d88948`, defense `#5fa9a4`, net `#d8be6f`, risk `#a94f55`, hairline `#3d3b32`. Use `Fraunces` for display equations and `IBM Plex Mono` for labels/data.

## Rhythm Declaration

Pattern: `hook-build-BUILD-PEAK-resolve`.

The video starts with the total identity, opens offense into its channels, zooms into the first-chance scoring layer, mirrors the same logic on defense, then resolves into the full map plus a method caveat.

## Global Rules

- Screen title and file identity must say `Decomp RAPM`.
- Do not use the internal terms `Alt3` or `weighted factor` anywhere on screen.
- Explain that the numbers are point-value, lineup-adjusted RAPM components.
- Keep the caveat concise: lineup signal, not isolated causal proof.
- Use CSS scene transitions: warm wipe, blur-through, and final dip to black.
- Build final layout first; animate into the stable frame with `fromTo()`.

## Scene 1: Total Identity

Concept: A quiet black-brown analytics canvas wakes up with one large equation. The viewer should immediately understand the top-level split.

Content:
- `Decomp RAPM`
- `RAPM = ORAPM + DRAPM`
- `total impact, separated by side of the ball`

Transition out: warm horizontal wipe, 0.72s.

## Scene 2: Offensive Channels

Concept: ORAPM unfolds into three readable lanes: scoring, turnover value, and second chance value. The scene should look like a decomposition tree, not a table.

Content:
- `ORAPM`
- `oTS + oTOV + oSC`
- `oTS: first-chance scoring value`
- `oTOV: value kept or lost before the first miss`
- `oSC: second-chance scoring after an offensive miss`

Transition out: blur-through with thin equation traces.

## Scene 3: First-Chance Scoring Drilldown

Concept: The scoring branch opens further. Shot value separates from free throws; shot value opens into rim, mid, and three, each with mix and make components.

Content:
- `oTS = oEFG + oFT`
- `oEFG = baseline + rim + mid + three`
- `zone value = frequency + make`
- `oFT = trip frequency + trip severity`

Transition out: vertical panel wipe.

## Scene 4: Defensive Mirror

Concept: The defense side mirrors the same architecture, but positive values mean good defense. Use cool teal as the primary accent.

Content:
- `DRAPM`
- `dTS + dTOV + dSC`
- `dTS: suppress shot and free-throw value`
- `dTOV: create first-chance turnover value`
- `dSC: deny or deflate second chances`
- `positive defense is good`

Transition out: gentle blur crossfade.

## Scene 5: Full Map + Caveat

Concept: Everything resolves into one map: total impact, offense, defense, first chance, second chance. The viewer should leave with the accounting structure and the right interpretation.

Content:
- `Decomp RAPM`
- `same possession universe`
- `components add back to total RAPM, up to rounding`
- `lineup-adjusted signal, not box-score attribution`
- `where impact comes from`

Final transition: slow dip to black.

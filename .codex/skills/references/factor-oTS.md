# oTS: Offensive True Shooting RAPM

## Definition

How much a player raises or lowers their team's scoring efficiency (True Shooting %) when they are on the court, isolated via ridge regression controlling for every teammate and opponent.

**Units**: TS% (native), converted to points via reconstruction beta weight.

**Sign convention**: Positive = team shoots better when this player plays. This is the single highest-weighted factor in the reconstruction regression -- shooting efficiency is the biggest lever in basketball.

## What Shows Up Here

oTS is the broadest and richest factor. It captures everything that affects how efficiently your team converts possessions into points via shooting. This includes both direct and indirect effects:

### Direct Shooting Impact
- **Scoring efficiency**: Volume times efficiency. Points are points. A player's individual shot-making directly impacts team TS%.
- Shot selection quality (taking good shots, avoiding bad ones)
- Free-throw value creation: getting the offense to the line, converting those trips, and creating foul pressure that leads to teammate free throws

### Indirect / Team-Level Shooting Impact
These are the things that make oTS powerful and non-obvious:

- **Spacing and gravity**: A corner shooter who never touches the ball but keeps a defender glued to him -- his team shoots better when he plays. No box score stat captures this. Lineup data does.
- **Ball movement and decision-making**: Knowing who's best positioned to attack the defense. Not wasting clock. Making the simple pass that leads to the easy pass. All of this creates better shots for the team.
- **Transition pace**: Pushing in transition before the defense sets creates easy baskets. A point guard who pushes pace will show oTS value from generating transition opportunities.
- **Screen setting quality**: A screener who creates real separation for the ball handler is improving team shooting efficiency, even though screens don't appear in any counting stat.
- **Off-ball movement**: Cutting, relocating, occupying help defenders -- all of this opens up better looks for teammates.
- **Playmaking**: A great playmaker shows up primarily as oTS. Better passing = better shots for teammates. Creating better looks IS what playmaking does to the scoreboard.
- **Free-throw assists / foul creation for teammates**: A pass that leads directly to a shooting foul is part of the team's free-throw value even if the passer never gets the FTA in his own line.
- **Non-spacing centers → negative oTS**: This is a big one. A center who can't shoot and doesn't create for others will often have a high personal TS% -- because they only dunk and lay it in -- but a deeply negative oTS-RAPM. Why? When their defender camps in the paint instead of being pulled to the perimeter, the whole offense clogs. Driving lanes shrink. The ball can't flow through them. Their teammates take harder shots. The lineup data sees this clearly even though the box score says the center is "efficient."

### The oREB Tradeoff
A big who leaks out early in transition might sacrifice offensive rebounding (lower oREB) for easy transition looks -- you'd see the value shifting from oREB into oTS instead. Similarly, a center taking long midrange shots abandons the offensive glass, hurting oREB but potentially generating efficient shots.

## Better Shorthand

The cleanest offensive shorthand is:

`oTS ≈ oEFG + oFT`

Not:

`oTS ≈ oEFG + oFTR`

Why? Because `oFTR` is only a rate. What matters to the offense is free-throw points created relative to scoring opportunities. The FT side of oTS is broader than self foul drawing:

- the player's own trips to the line
- how many of those trips become points
- passes that directly create teammate shooting fouls
- the rim pressure, screening, and defensive distortion that produce foul-heavy possessions

Free-throw assists are especially useful for explaining the playmaking side of `oFT`.

## ShotQuality Sub-Decomposition (Level 2)

oTS decomposes further into three components using ShotQuality data (see [ts-decomposition.md](ts-decomposition.md)):

| Component | What It Isolates | Player Archetype |
|-----------|-----------------|------------------|
| **oSQ** (Shot Quality) | Pre-shot expected value. Spacing, playmaking, shot selection. Value locked in before the ball leaves the hand. | Haliburton (+2.3 SQ), Giannis (+3.0 SQ) |
| **oMake** (Shot Making) | Actual minus expected. Finishing ability, tough-shot conversion. | SGA (+2.8 Make), Jokic (+3.5 Make), Murray (+3.2 Make) |
| **oFT** (Free Throw Value) | Free-throw point generation above the foul-type-aware baseline. Includes own trips, teammate-created trips, and FT conversion. | SGA (+1.0 FT) |

**Key insight**: Two players with similar oTS can generate it through completely different mechanisms. Haliburton is almost entirely shot quality creation. SGA is almost entirely tough-shot making + FT value creation. Same oTS tier, opposite profiles.

## Box-Score Regression Drivers (5Y Model)

How well box-score stats predict oTS-RAPM (what the lineup signal says about a player's shooting impact):

| Metric | Ridge R^2 | XGBoost R^2 | Spearman |
|--------|----------|-------------|----------|
| **OFF_TS** | 0.594 | 0.610 | 0.692 |

**Top predictors** (in order of importance):
1. **Usage** -- Higher-usage players tend to have higher oTS-RAPM, reflecting that efficient volume scorers improve team shooting
2. **pos_TS_added** -- Position-adjusted true shooting added. How much better this player shoots relative to position average.
3. **TS_added** -- Raw true shooting above league average
4. **AssistEFG** -- The efficiency of shots this player assists on. A direct measure of shot creation quality for teammates.
5. **TSA100** -- True shooting attempts per 100 possessions (volume measure)

**What the R^2 = 0.594 means**: Box scores explain ~59% of lineup-level shooting impact. The remaining ~41% is the invisible stuff -- spacing, gravity, off-ball movement, defensive attention drawn, screen quality -- things that affect team shooting but don't appear in any individual stat line.

## Player Examples (2021-2026 Time-Decayed)

### Elite oTS (Top Tier)
| Player | oTS | Context |
|--------|-----|---------|
| Nikola Jokic | +5.76 | Highest in the dataset. Full package: creates AND converts. SQ decomp: +1.9 SQ / +3.5 Make / +0.8 FT |
| Stephen Curry | +4.70 | Gravity monster. Balanced SQ/Make: +2.4 SQ / +2.1 Make. Team shoots dramatically better with him on court. |
| Kevin Durant | +4.48 | Elite individual efficiency translating to team-level impact |
| Giannis Antetokounmpo | +3.49 | SQ-dominant: +3.0 SQ / +0.7 Make. Creates good looks through driving and collapsing the defense. |
| Joel Embiid | +3.23 | Strong oTS despite negative oREB (-1.45) -- his shooting value may partly come at the cost of offensive rebounding |
| Kawhi Leonard | +3.10 | Two-way star with balanced offensive efficiency |
| Luka Doncic | +2.92 | High-usage creator who boosts team shooting |
| SGA | +2.89 | Pure making + FT: -0.1 SQ / +2.8 Make / +1.0 FT |

### Negative oTS (Impact Players Who Hurt Team Shooting)
| Player | oTS | Context |
|--------|-----|---------|
| Clint Capela | -2.54 | Center with no shooting threat. Defenders sag off, clogging lanes. |
| Mitchell Robinson | -2.28 | Same archetype -- elite rebounder (oREB +3.28) but negative spacing |
| Kevon Looney | -1.59 | Again, non-shooting big. oREB +1.97 but oTS -1.59. Classic tradeoff. |
| Amen Thompson | -1.19 | Young player still developing shooting |

**Pattern**: Non-shooting bigs consistently show negative oTS because their defenders help off them, reducing team spacing and shot quality. Their value often shows up in oREB instead.

## Cross-Factor Interactions

- **oTS vs oREB tradeoff**: Very common. Non-shooting bigs sacrifice oTS for oREB. Centers taking long midrange shots sacrifice oREB for oTS. See Steven Adams (oTS -0.92, oREB +4.04) vs Myles Turner (oTS +2.11, oREB -1.31).
- **oTS and oTOV**: High-usage players can generate both (SGA: oTS +2.89, oTOV +2.61) or trade off (Durant: oTS +4.48, oTOV -0.95 -- his possessions sometimes end in turnovers).
- **oTS vs dTS relationship**: Not necessarily correlated. Curry is +4.70 oTS but -1.25 dTS. Gobert is -0.23 oTS but +4.44 dTS. Purely offensive vs purely defensive shootingimpact.

## Caveats

- Lineup-level, not individual. A player's oTS-RAPM includes the effect of teammates who play with them, imperfectly controlled by the regression.
- For the pure series, no box priors. Noisy for low-minute players. Multi-season time-decay helps.
- The ShotQuality SQ/Contest boundary depends on their pre-shot model. The total oTS is model-independent.

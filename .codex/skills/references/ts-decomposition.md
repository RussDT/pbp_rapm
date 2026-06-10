# Inside True Shooting: ShotQuality TS Decomposition

## Table of Contents
- [The Problem](#the-problem)
- [The Three Components](#the-three-components)
- [What Players Look Like Under the Hood](#what-players-look-like-under-the-hood)
- [Defensive Decomposition](#defensive-decomposition)
- [The Nested Decomposition Story](#the-nested-decomposition-story)
- [Credits and Data](#credits-and-data)

## The Problem

The six-factor framework shows that TS factors (oTS and dTS) are the biggest drivers of total RAPM by point-value weight. The efficiency game dominates the possession game. But TS-RAPM is still a black box.

A player with +2.0 oTS-RAPM -- is that because they create great looks for teammates? Because they're an elite finisher who converts tough shots? Because they create free-throw value? TS-RAPM can't tell you. ShotQuality data lets you crack it open.

## The Three Components

Using ShotQuality's pre-shot expected value (based on shot location, closest defender distance, shot type, shooter movement, touch time, and other contextual factors), every non-turnover possession splits into three values that sum exactly:

**Shot Quality (SQ)**: The pre-shot expected value. How good was the look? This captures spacing, playmaking, off-ball movement, shot selection -- everything that determines the quality of the opportunity before the ball leaves the shooter's hand. On FT possessions, SQ captures the "expected shot value" portion that the foul replaced (see foul-type baselines below).

**Shot Making / Contest**: Actual minus expected. On offense, this is finishing ability, touch, degree-of-difficulty conversion. On defense, this is contesting, rim protection, shot alteration -- making opponents miss shots they'd normally make. Zero on FT possessions.

**Free Throw Value (FT)**: The premium above what a normal possession would have produced. Separated because it runs through a completely different mechanical pathway than field goal attempts. On FGA possessions, FT is zero.

For offense, this is broader than self foul drawing. It includes the free-throw point value created by direct trips, teammate-created trips, and the offensive conditions that lead to fouls. Free-throw assists are especially useful for explaining the playmaking side of this channel.

SQ + Contest + FT = TS on every single possession. Not approximately. Exactly. That's the foundation.

### Foul-Type-Aware Baselines

Not all free throw possessions are equal. The SQ and FT components use foul-type-aware baselines to split each FT possession correctly. **avg_pts** = average points scored on FGA possessions (~1.09) -- the counterfactual value of a normal possession if no foul occurred.

| FT Type | SQ Baseline (per FT) | FT Premium (per FT) | Rationale |
|---------|---------------------|--------------------|----|
| **2-shot foul** | avg_pts / 2 (~0.54) | ExpFT - avg_pts/2 | Foul replaced a ~1.09 possession, split across 2 FTs |
| **3-shot foul** | avg_pts / 3 (~0.36) | ExpFT - avg_pts/3 | Same, split across 3 FTs |
| **And-1** | 0 | ExpFT (full value) | Basket already counted; FT is pure bonus |
| **Technical** | 0 | ExpFT (full value) | No possession replaced; pure bonus |

Example: A 90% FT shooter fouled on a 3-pointer. SQ gets 0.36 per FT (1.09 / 3). FT gets +0.54 per FT (0.90 - 0.36). Total across 3 FTs: SQ = 1.09, FT = 1.61. The foul generated +1.61 above what a normal possession would have produced.

### Calibration

ShotQuality's pre-shot EV model can have small year-to-year biases (e.g., 2025-26 shots underrated by ~0.024 points). A per-year additive calibration shifts initial_ev so the mean matches the mean actual points scored on FGAs. This prevents systematic over- or under-crediting of shot quality in any given season. Missing initial_ev values (~6-10% of FGAs where ShotQuality lacks tracking data) are filled with the year's average actual points -- contributing zero signal to both SQ and CONTEST.

**ShotQuality dependency caveat**: The SQ/Contest boundary is defined by ShotQuality's pre-shot model. A different spatial model would draw that boundary slightly differently. The total (TS) and the FT component are unaffected by which pre-shot model you use. The decomposition sharpens as spatial tracking models improve.

## What Players Look Like Under the Hood

All values are points per 100 shots. These are lineup-adjusted RAPM estimates with no box priors, so they represent what the pure lineup signal says about each player's impact.

### The Playmaker vs The Assassin

**Tyrese Haliburton** (oTS +2.59 overall): oSQ +2.3, oMake +0.7, oFT +0.0
Almost entirely a shot quality creator. His value is in generating good looks for the team, not converting tough ones. The playmaker archetype.

**Shai Gilgeous-Alexander** (oTS +2.89 overall): oSQ -0.1, oMake +2.8, oFT +1.0
His SQ is basically zero -- he's not creating good looks at all. He's converting bad ones (+2.8 oMake) and generating free-throw value (+1.0 FT). Pure tough-shot assassin plus FT pressure.

Same tier of total oTS impact. Completely opposite mechanisms.

### The Shot Making Spectrum

**Nikola Jokic** (oTS +5.76): oSQ +1.9, oMake +3.5, oFT +0.8
The full package but primarily a shot making monster -- beating expectations by 3.5 pts/100 shots. SQ is good but not his best dimension.

**Stephen Curry** (oTS +4.70): oSQ +2.4, oMake +2.1, oFT +0.2
Balanced between creating great looks (gravity, spacing) and converting them. Almost no FT value -- doesn't live at the line.

**Giannis Antetokounmpo** (oTS +3.49): oSQ +3.0, oMake +0.7, oFT -0.2
Highest SQ on the board at +3.0 -- his team gets dramatically better looks when he plays. Slightly negative FT despite getting to the line constantly (the lineup signal says his FT channel isn't adding net value once you account for what those possessions would have been worth as FGA opportunities).

**Jamal Murray** (oTS +2.32): oSQ -0.3, oMake +3.2, oFT -0.0
Negative shot quality, but the highest oMake on this list. Taking and making bad shots at an elite rate.

**De'Anthony Melton** (oTS +1.54): oSQ -0.2, oMake +3.1
Nearly identical shot making profile to Murray but nobody thinks of him that way. The decomposition reveals hidden shot-makers.

### What These Profiles Mean

The SQ-dominant players (Haliburton, Giannis) generate value before the ball leaves anyone's hand. Their impact is about creating the opportunity.

The Contest-dominant players (SGA, Murray, Melton, Jokic) generate value by converting. The shot is already defined -- they beat the expectation.

Jokic and Curry are the rare players who are elite at both. Different balance, but positive on both dimensions.

## Defensive Decomposition

Defense is where shot alteration vs shot quality suppression diverges most clearly.

### Rim Protectors -- Two Different Mechanisms

**Rudy Gobert** (dTS +4.44): dSQ +1.2, dContest +2.6, dFT +0.7
Primarily shot alteration -- making guys miss shots they'd normally make (+2.6 Contest). Some SQ too (forcing tough looks). The classic rim protector.

**Victor Wembanyama** (dTS +3.23): dSQ +0.6, dContest +2.7, dFT +0.4
Even more Contest-heavy than Gobert proportionally. Pure shot alteration machine.

**Luke Kornet** (dTS +2.17): dSQ +1.4, dContest +0.6, dFT +0.5
Completely different mechanism -- he's more of a dSQ guy. He's forcing bad shots more than altering good ones. His rim protection works by making opponents take worse looks in the first place, not by contesting makes into misses.

### The Perimeter Defender

**Derrick White** (dTS +2.57): dSQ +0.3, dContest +2.2, dFT +0.3
A guard showing up on dContest like a rim protector -- that's his contest/close-out ability on the perimeter showing up in the data. Elite at making opponents miss their normal shots from the wing and three-point line.

## The Nested Decomposition Story

This is the narrative arc when presenting the full framework:

**Act 1**: RAPM gives you one number. It's a black box. You know Jokic is a +7.67, but not why.

**Act 2**: Six-Factor RAPM opens the box. Three independent regressions, six factors, R^2 = 0.985. Now you know it's mostly shooting efficiency (oTS +5.76) with some defensive rebounding (dREB +1.79). But oTS is still a black box.

**Act 3**: ShotQuality TS decomposition opens the oTS box. Four parallel regressions, three components, R^2 > 0.9999. Now you know the oTS comes from +1.9 shot quality creation AND +3.5 shot making AND +0.8 free throw value.

At every level, independently estimated components reconstruct nearly all of the parent signal. The game keeps reducing to identifiable mechanical channels all the way down.

**The thesis**: Player impact, at a fundamental level, is structurally driven by possession mechanics. It's not mystical team effects. It's not noise. It's efficiency and possession control, measured through independently estimated regression frameworks that converge on the same answer: the game is shooting, turnovers, and rebounds -- and shooting itself is shot creation, finishing, and free throws.

The framework captures where impact lands, not how it's generated. The "how" is still basketball. The six factors (and their sub-components) just tell you which parts of the scoreboard each player is moving.

## Credits and Data

- **Six-Factor RAPM framework**: Original research by Russell Thomas (databallr.com)
- **ShotQuality TS decomposition**: Original research by Russell Thomas using ShotQuality data
- **ShotQuality pre-shot expected values**: Provided by ShotQuality, which uses spatial tracking to compute pre-shot expected values for every NBA FGA based on shot location, closest defender distance, shot type, shooter movement, touch time, and other contextual factors
- **Data**: ~650,000 non-turnover possessions from 2023-24 through 2025-26 NBA seasons
- **Pipeline**: `~/Docs/pbp_rapm/nba_pipeline/`

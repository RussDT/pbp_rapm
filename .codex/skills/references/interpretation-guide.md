# Interpretation Guide: Six-Factor RAPM

## Table of Contents
- [What the Numbers Mean](#what-the-numbers-mean)
- [Where Basketball Actions Land](#where-basketball-actions-land)
- [Player Archetypes](#player-archetypes)
- [The Outputs vs Inputs Distinction](#the-outputs-vs-inputs-distinction)
- [Explaining to Different Audiences](#explaining-to-different-audiences)

## The Core Principle

RAPM measures the change in team point differential when a player is on the floor. It captures how a player alters the environment of the game, not merely what they personally produce.

Every basketball action -- scoring, passing, spacing, contesting, boxing out -- ultimately registers as a change in one of these six channels. The framework captures **where impact lands on the scoreboard**, not how it's generated. The "how" is still basketball.

## What the Numbers Mean

Each factor represents a player's isolated impact on a specific team outcome per 100 possessions, controlling for every teammate and opponent via ridge regression.

| Factor | What It Measures | Positive Means | Basketball Skills That Show Up Here |
|--------|-----------------|----------------|-------------------------------------|
| **oTS** | Team points per shot when player is on court | Team gets more value per shot | Spacing, gravity, shot creation, playmaking, ball movement, transition pace, shot selection |
| **oTOV** | Team turnover rate when player is on court | Team turns it over less | Ball security, passing quality, decision-making, tempo management |
| **oREB** | Team offensive rebound rate when player is on court | Team grabs more OREBs | Glass crashing, positioning, effort, second-chance creation |
| **dTS** | Opponent points per shot when player is on court | Opponent gets less value per shot | Rim protection, contest quality, defensive communication, scheme enforcement, closeout quality |
| **dTOV** | Opponent turnover rate when player is on court | Opponent turns it over more | Active hands, deflections, trapping, defensive pressure, team scheme execution |
| **dREB** | Opponent offensive rebound rate when player is on court | Opponent grabs fewer OREBs | Boxing out, defensive rebounding, denying second chances |

## Where Basketball Actions Land

The framework captures where value lands on the scoreboard. The basketball is still basketball.

### Shooting: oTS / dTS

oTS does not measure a player's personal scoring efficiency. It measures how the player changes the team's points per shot. This includes both their own scoring and the shots they create for teammates.

**Two complementary lenses on oTS**: The ShotQuality decomposition (`oSQ`, `oMAKE`, `oFT`) explains the mechanical channels -- shot quality creation, shot conversion, and free-throw value generation. But oTS can also be read through a **scoring vs playmaking** lens: how much of the team's shot value improvement comes from the player's own scoring vs improving teammates' efficiency? Both views explain the same oTS number from different angles.

**Useful shorthand**: `oTS ≈ oEFG + oFT`. The FT side is broader than personal foul drawing or `oFTR`; it includes the offense's total free-throw point generation, including playmaking that leads to teammate shooting fouls. Free-throw assists are useful evidence for that playmaking side of `oFT`.

**Scoring efficiency → oTS.** Volume times efficiency. Points are points.

**Playmaking → oTS.** Better passing = better shots for teammates. The hockey assist, the simple pass that leads to the easy pass.

**Spacing and gravity → oTS.** Standing in the right spot keeps defenders honest. A corner shooter who never touches the ball but keeps a defender glued to him -- his team shoots better when he plays. No box score stat captures this. Lineup data does.

**Non-spacing centers → negative oTS.** This is a big one. A center who can't shoot and doesn't create for others will often have a high personal TS% -- because they only dunk and lay it in -- but a deeply negative oTS-RAPM. Why? When their defender camps in the paint instead of being pulled to the perimeter, the whole offense clogs. Driving lanes shrink. The ball can't flow through them. Their teammates take harder shots. The lineup data sees this clearly even though the box score says the center is "efficient."

**Pushing pace → oTS.** Fast breaks create easy baskets before the defense sets.

**Ball movement and decision-making → oTS.** Knowing who's best positioned to attack the defense. Not wasting clock. Pushing in transition before the defense sets instead of walking it up.

**Rim protection → dTS.** Contesting and altering shots at the rim.

**Rim deterrence → dTS.** Making opponents avoid the rim entirely. The shot that never gets taken because the rim protector is lurking.

**Contest quality → dTS.** Being well-positioned, making shooters uncomfortable. Arriving on balance without flying past.

**Closeout quality → dTS.** Arriving on balance without flying past the shooter. Controlled closeouts contest the shot; wild ones leave the defense in rotation.

**Getting blown by → negative dTS.** Giving up easy baskets at the rim because the defender can't stay in front.

**Defensive communication → dTS.** Calling out switches, helping teammates get into position, funneling ball handlers into help. A player who makes the whole defense more organized reduces opponent shooting efficiency -- even if his individual defensive stats look ordinary.

### Turnovers: oTOV / dTOV

oTOV measures how a player improves team possession security. Turnovers destroy offensive opportunities, so reducing them increases team points per possession.

**Two types of turnovers reflect two offensive roles.** Scoring turnovers arise from shot creation pressure (charges, strip steals on drives). Bad-pass turnovers arise from playmaking attempts. A player's turnover profile reveals whether their role is primarily as a scorer or creator.

**Ball security → oTOV.** Not forcing passes. Reading the defense. Taking care of it.

**oTOV is role-adjusted by nature.** High-usage players handle the ball more, pass more, and have more chances to turn it over. Raw turnover counts can mislead -- a point guard with 3 turnovers a game might actually be elite relative to their ball handling volume. What matters here is the team's turnover rate when you play vs when you sit. The lineup data handles the role context implicitly.

**Stonehands → bad oTOV.** If a player can't catch the ball, the turnover sometimes gets credited to the passer in the box score. Doesn't matter here -- this is lineup-level data. When you're on the floor and the team turns it over more, that's your oTOV, regardless of whose name ends up in the play-by-play.

**Steals → dTOV.** Active hands in passing lanes.

**Drawing charges → dTOV.** Taking away a possession.

**Drawing offensive fouls → dTOV.** Moving screens, hooks, pushoffs -- opponent loses the ball.

**Gambling for steals → boosts dTOV, but can hurt dTS when you get burned.** The decomposition lets you see whether the dTS cost has actually materialized over time. If a player's dTOV is elite but their dTS is suffering, that's *consistent with* the gambling hypothesis. It's not proof -- there could be other things going on, and this is lineup-level, so teammates are part of the picture too. But it surfaces the pattern in a way that box scores never could.

### Rebounds: oREB / dREB

oREB measures how a player increases the number of offensive possessions the team gets. This decomposes into two mechanisms: extra opportunities (creating more offensive rebounds) and second-chance conversion (turning those extra possessions into points).

**Crashing the glass → oREB.** Second chances.

**Boxing out → dREB.** Denying second chances. A center who boxes out for a guard to grab the board might not get credit in box score rebounding but shows up in team dREB when he plays.

### Cross-Factor Patterns

**Transition decisions (oTS + oREB tradeoff).** A point guard who pushes pace creates easy baskets before the defense sets -- that's oTS. A big who leaks out early in transition might sacrifice offensive rebounding for easy transition looks -- you'd see lower oREB but the value shifting into oTS instead.

**The connective tissue player.** Some guys just make the team smarter when they play. More organized. More in sync. That's hard to quantify, but it's real, and it lands somewhere in these six factors -- even if you'd never see it in a box score. The framework captures outputs, not inputs. The "how" is still basketball.

## Player Archetypes

From the 2021-2026 time-decayed data (top of the leaderboard):

### Elite Two-Way Impact
**Nikola Jokic**: oTS +5.76, oTOV +0.38, oREB +0.45, dTS -0.34, dTOV -0.49, dREB +1.79. Net: +7.67. Offensive impact is almost entirely shooting efficiency. Defensive value comes from rebounding denial (dREB +1.79).

**Shai Gilgeous-Alexander**: oTS +2.89, oTOV +2.61, oREB -0.68, dTS +1.38, dTOV +1.01, dREB -0.04. Net: +7.39. Balanced across efficiency AND possession control. Elite on both axes.

### Defensive Anchors
**Rudy Gobert**: oTS -0.23, dTS +4.44, dREB +0.87. Net: +4.82. Almost all defense, almost all shooting efficiency (dTS). The rim protection archetype.

**Victor Wembanyama**: dTS +3.23, dREB +1.10. Net: +5.22. Similar to Gobert but with meaningful offensive upside (oTS +1.71).

### Possession Control Specialist
**Alex Caruso**: dTOV +3.10 (elite), dTS +1.48. Net: +4.89. The steal/deflection machine -- and his dTS is also positive, suggesting the gambling is controlled or his positioning is good enough that the gambles aren't costing efficiency.

**Steven Adams**: oREB +4.04 (extreme), dREB +1.10. Net: +3.72. Almost entirely a rebounding impact player. Creates and denies possessions.

### Efficiency Creators
**Stephen Curry**: oTS +4.70 (elite), off +4.97. Net: +3.57. Massive offensive efficiency creator, but net RAPM is "only" +3.57 because his defensive factors are negative.

**Draymond Green**: dTS +2.34, dREB +0.82, oTS +1.48. Net: +2.94. Defensive efficiency anchor who also helps offensive shooting. The connective tissue archetype.

### Efficiency vs Possession Control Split
Compare **Curry** (elite Net Efficiency, neutral Possession Value) vs **Caruso** (strong Possession Value through dTOV, solid efficiency through dTS). Both are valuable. Completely different mechanical profiles.

## The Decomposition Hierarchy

The nested structure mirrors the nested structure of basketball impact itself.

```
RAPM (net team impact)
├── ORAPM (team offensive impact)
│   ├── oTS (team points per shot)
│   │   ├── oSQ  (shot quality creation)
│   │   ├── oMAKE (shot conversion / finishing)
│   │   └── oFT  (free-throw value generation)
│   ├── oTOV (possession security)
│   │   ├── scoring turnovers (shot creation pressure)
│   │   └── bad-pass turnovers (playmaking attempts)
│   └── oREB (extra possessions)
│       ├── extra opportunities (creating OREBs)
│       └── second-chance conversion
└── DRAPM (team defensive impact)
    ├── dTS (opponent shot value suppression)
    │   ├── dSQ  (forcing bad shots)
    │   ├── dCONTEST (shot alteration / making them miss)
    │   └── dFT  (opponent FT premium)
    ├── dTOV (forced turnovers)
    └── dREB (OREB denial)
```

Each layer of decomposition moves closer to the underlying causes of team performance.

### Two Lenses on oTS

oTS can also be interpreted through a **scoring vs playmaking** lens:

- **Scoring impact**: How a player improves the team's shot value through their own shooting
- **Playmaking impact**: How a player improves the team's shot value through creating for teammates

Both ultimately improve the same outcome: team points per shot. The SQ/MAKE/FT decomposition tells you the mechanical channel; the scoring/playmaking split tells you who benefits from the value. These are complementary views of the same oTS number.

### Offensive Shot Diet Categories

Separately from the RAPM decomposition, offensive shot attempts can be profiled by how the opportunity arose, ordered from hardest to easiest:

1. **Creation** -- self-created shots (off the dribble, isolation)
2. **Spacing** -- catch-and-shoot, movement shots from passing/screening
3. **Transition** -- fast break opportunities before defense sets
4. **Finishing** -- shots at the rim, putbacks, cuts

The shape of a player's shot diet profile communicates their offensive archetype. This is supporting evidence for understanding a player's role, not a sub-component of any specific RAPM factor.

## The Outputs vs Inputs Distinction

The six factors are **where impact lands**. Playmaking, gravity, screen setting, defensive IQ -- those are **how impact is generated**. The framework doesn't say those things don't exist. It says that whatever a player does on a basketball court, the value ultimately registers in one of these six channels.

- A great playmaker creates better shots for teammates -- that's **oTS**
- A lockdown defender makes opponents miss -- that's **dTS**
- A ball-hawking guard forces turnovers -- that's **dTOV**
- A hustle player who crashes the glass -- that's **oREB**
- A player who takes care of the ball and makes simple passes -- that's **oTOV**
- A center who boxes out and denies second chances -- that's **dREB**

The framework captures **outputs**, not **inputs**. The "how" is still basketball. The six factors just tell you which parts of the scoreboard each player is moving.

## Explaining to Different Audiences

### For basketball fans (no stats background)
"We took the math that measures how much a player helps his team outscore the opponent, and instead of asking one question, we asked three: how did shooting change? How did turnovers change? How did rebounding change? On both ends. Six numbers that tell you how a player helps. And when we check whether these six numbers add back up to the original -- they explain 98.5% of it."

### For analytics-literate audience
"Same ridge regression machinery as standard RAPM, same design matrix, but we swap the dependent variable. Three independent regressions predicting TS%, TOV%, and OREB% instead of point differential. Each produces offense/defense coefficients. A second-stage regression reconstructing full RAPM from the six factors yields R^2 = 0.985. The beta weights serve as unit conversion factors from native units to points per 100 possessions."

### For academic/research audience
"A structural decomposition of Regularized Adjusted Plus-Minus into six independently estimated components corresponding to the four factors of basketball (shooting, turnovers, offensive rebounding, defensive rebounding -- with shooting split into offensive and defensive). Three parallel ridge regressions with identical design matrices and ridge penalties, differing only in the dependent variable. A second-stage OLS reconstruction regression demonstrates that these independently estimated factors account for 98.5% of the variance in full RAPM, establishing that player impact on scoring margin is almost entirely mediated by three mechanical channels: shooting efficiency, turnover rate, and rebounding rate."

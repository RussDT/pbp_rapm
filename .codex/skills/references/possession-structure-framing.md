# Possession-Structure Framing

Use this reference for public copy, methodology intros, SixRings 5Y factor modes, and explanation of active point-valued `oTS` / `oTOV` / `oSC` / `dTS` / `dTOV` / `dSC` artifacts.

## Core Frame

At its core, basketball impact can be described as possession management.

Every possession starts with control of the ball. From there:

1. The offense has to keep the ball.
2. The offense has to turn that possession into a scoring attempt.
3. The shot has to produce points.
4. If the shot misses, the possession either ends with a defensive rebound or continues through offensive rebounding and second-chance value.

The public six-factor frame is built around those stages:

`Before the shot -> The shot -> After the miss`

That becomes:

`Turnover Value -> Shot Value -> Second-Chance Value`

Applied to both sides of the ball:

- `Offense = oTOV + oTS + oSC`
- `Defense = dTOV + dTS + dSC`
- `Total RAPM = oTOV + oTS + oSC + dTOV + dTS + dSC`

Use this point-additive formula only for active point-valued surfaces, such as `weighted_factors_alt3_efg_value_*` and the SixRings factor-game pool. Do not use it to relabel legacy native-rate `oREB` / `dREB` files.

## Public Explanation

One RAPM number says how much a player changes the scoreboard. Six factors say where that scoreboard impact comes from structurally.

The goal is not to invent six categories. The goal is a decomposition that:

- matches basketball's possession structure
- can be summed back to total impact
- lets different kinds of value be compared on one scoreboard-value axis

In point-valued outputs, a player adding `+1` through offensive shot value and a player adding `+1` through defensive turnover creation are creating roughly the same scoreboard value through different possession stages.

## Factor Interpretations

### oTOV: Offensive Shot Creation and Turnover Control

`oTOV` measures whether the possession survives into the shot phase.

The first job of offense is not simply to run actions; it is to create a shot before the defense kills the possession. Pull-up shooting, rim pressure, passing, advantage creation, tempo control, and ball security all matter because they help the team reach an attempt.

Public wording:

- Before a team can score, it has to turn the possession into a shot.
- `oTOV` measures how much a player helps his team reach the shot phase without losing the ball.
- A great offensive engine does not just make shots better once they exist; he makes sure possessions become shots at all.

Example 5-year `oTOV` peaks from the pasted context:

| Rank | Player | Window | Value |
|------|--------|--------|-------|
| 1 | Chris Paul | 2016-2020 | +2.42 |
| 2 | Shai Gilgeous-Alexander | 2022-2026 | +2.28 |
| 3 | Tracy McGrady | 2001-2005 | +2.12 |
| 4 | Baron Davis | 2004-2008 | +2.12 |
| 5 | Nick Van Exel | 1997-2001 | +2.05 |

### oTS: Offensive Scoring Efficiency

`oTS` measures how much a player increases the points his team generates per scoring attempt.

It is broader than self shooting. It includes scoring volume and efficiency, but also passing, spacing, screening, cutting, rim pressure, transition creation, foul drawing, and ball movement because those change the quality and value of team attempts.

Public wording:

- Once the offense reaches a scoring attempt, the next question is how valuable that attempt is.
- `oTS` is the full offensive ability to make scoring attempts more valuable for your team.
- Some players do it through shooting, some through passing, some through pressure, and some through all of the above.

Example 5-year `oTS` peaks from the pasted context:

| Rank | Player | Window | Value |
|------|--------|--------|-------|
| 1 | Steve Nash | 2006-2010 | +9.60 |
| 2 | Stephen Curry | 2014-2018 | +9.20 |
| 3 | Nikola Jokic | 2022-2026 | +8.09 |
| 4 | LeBron James | 2013-2017 | +7.78 |
| 5 | James Harden | 2015-2019 | +7.31 |

### oSC: Offensive Second-Chance Impact

`oSC` measures how much a player helps his team create value after misses.

The main driver is offensive rebounding, but the factor is broader than the rebound itself. It captures possession extension, keeping the ball alive, producing another attempt, and converting after the defense failed to finish the stop.

Public wording:

- A missed shot does not have to end the possession.
- `oSC` is not scoring efficiency or turnover avoidance; it is possession extension.
- Some players create value by making the first shot better. Others create value by making sure the first miss is not the end.

Example 5-year `oSC` peaks from the pasted context:

| Rank | Player | Window | Value |
|------|--------|--------|-------|
| 1 | Mitchell Robinson | 2022-2026 | +4.47 |
| 2 | Steven Adams | 2021-2025 | +4.32 |
| 3 | Enes Kanter | 2018-2022 | +4.06 |
| 4 | Kevin Love | 2009-2013 | +3.47 |
| 5 | Kevon Looney | 2019-2023 | +3.32 |

### dTOV: Defensive Turnover Creation

`dTOV` measures defensive value from ending possessions before a shot exists.

Steals matter, but so do deflections, ball pressure, denial, traps, rotations, passing-lane disruption, drawn charges, moving-screen calls, travels, and other offensive violations.

Public wording:

- The cleanest defensive possession is the one that ends before a shot.
- Shot defense asks how well you defend the attempt. Turnover creation asks whether you can prevent the attempt from happening.
- A defender who forces turnovers is not merely lowering efficiency; he is deleting possessions.

Example 5-year `dTOV` peaks from the pasted context:

| Rank | Player | Window | Value |
|------|--------|--------|-------|
| 1 | Alex Caruso | 2021-2025 | +2.65 |
| 2 | Ricky Rubio | 2011-2015 | +2.26 |
| 3 | Matisse Thybulle | 2019-2023 | +2.10 |
| 4 | Tony Allen | 2014-2018 | +1.94 |
| 5 | Thaddeus Young | 2018-2022 | +1.84 |

### dTS: Defensive Scoring Efficiency Suppression

`dTS` measures how much a player reduces the points opponents generate per scoring attempt.

Players create `dTS` value by lowering opponent shot quality, protecting the rim, contesting jumpers, navigating screens, containing drives, forcing worse locations, and avoiding damaging fouls. Free throws belong in scoring efficiency, so foul discipline is part of the factor.

Public wording:

- If the offense reaches a scoring attempt, the defense still has work to do.
- `dTS` is defensive shot-value impact.
- Rim protection, deterrence, contesting, discipline, and shot suppression show up here.

Example 5-year `dTS` peaks from the pasted context:

| Rank | Player | Window | Value |
|------|--------|--------|-------|
| 1 | Dikembe Mutombo | 1997-2001 | +5.80 |
| 2 | Shawn Bradley | 1998-2002 | +5.54 |
| 3 | Rudy Gobert | 2021-2025 | +4.94 |
| 4 | Tim Duncan | 2001-2005 | +4.83 |
| 5 | Joel Embiid | 2016-2020 | +4.71 |

### dSC: Defensive Second-Chance Prevention

`dSC` measures how much a player prevents opponent second-chance value.

Defensive rebounding is the primary driver, but the factor represents the larger act of ending the defensive possession: boxing out, securing the ball, preventing tap-outs, controlling space, avoiding easy putbacks, and finishing the sequence.

Public wording:

- A defensive possession is not complete when the shot misses; it is complete when the defense gets the ball.
- Second-chance points are painful because they come after the defense already did most of the work.
- `dSC` captures players who help make defensive possessions actually end.

Example 5-year `dSC` peaks from the pasted context:

| Rank | Player | Window | Value |
|------|--------|--------|-------|
| 1 | Nene | 2010-2014 | +2.50 |
| 2 | Jason Collins | 2001-2005 | +2.04 |
| 3 | Jusuf Nurkic | 2020-2024 | +1.99 |
| 4 | Nikola Jokic | 2022-2026 | +1.95 |
| 5 | Domantas Sabonis | 2020-2024 | +1.91 |

## Copy Guardrails

- Keep the structure concrete: possession survives, attempt value, possession extension/completion.
- Use "shot value" or "scoring attempt value" for `TS` when writing for general audiences.
- Use "second-chance value" for `SC`; mention rebounding as the main driver but not the whole factor.
- Avoid making `oTS` sound like personal shooting efficiency only.
- Avoid making `oTOV` sound like only low personal turnovers; it includes shot creation and team possession survival.
- Avoid making `dTOV` sound like steals only.
- Avoid making `dSC` sound like defensive rebounds only.
- Pair the framework with the caveat that RAPM is lineup-adjusted impact signal, not isolated causal proof of a single skill.

## Short Reusable Summary

Six-Factor RAPM reduces basketball impact to three possession questions on both sides of the ball:

`Can you reach a shot? -> How valuable is the shot? -> What happens after a miss?`

On offense, those are `oTOV`, `oTS`, and `oSC`.

On defense, those are `dTOV`, `dTS`, and `dSC`.

One number tells you how much a player changed the scoreboard. The six factors tell you where that impact came from.

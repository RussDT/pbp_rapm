# Deflections playoff context

## Protocol fixed before inspection

Isolated research challenger to the frozen next-game OU. Keep its parameters, source masks, per-100 defensive-possession target and equal-season, exposure-weighted MSE unchanged. Estimate one additive playoff observation effect using only completed prior seasons. Within each player-season with both RS and PS observations, calculate the PS-minus-RS rate difference; pool with weights `E_RS * E_PS / (E_RS + E_PS)`, the exact exposure-weighted player-season fixed-effect coefficient. No sign constraint or threshold search. Warm-up 2019 uses zero offset.

Feed this known offset to the OU observation equation for PS only; subtract it implicitly through the innovation when updating latent skill. RS prediction has no direct offset. This is an association controlling player-season composition, not a causal playoff effect; schedule, opponents and within-season development can still confound it.

Select only if overall 2020–2022 next-game MSE improves by more than one paired game-bootstrap standard error. Report 2023–2025 as already-spent historical stress, including rejected-candidate diagnostics. Report PS, RS, and the first ten observed RS appearances for players observed in the previous postseason; these slices do not select the model. 2026 is unscored, 2027 untouched. Preserve incumbent boards. Runner: `nba_pipeline/scripts/analyze_deflections_playoff_context.py`; ignored receipts: `results/estimated_skills/deflections_playoff_context/`.

## Results — September 12, 2026

Prior-season fitted offsets ranged from −0.39 to −0.18 deflections/100; the 2026 coefficient using 2019–2025 pairs is −0.1824. These are historical observed-rate associations, not proof of a causal reduction in skill.

The candidate passes the registered one-SE development gate by a small margin: overall MSE improvement 0.0288%, game-clustered 95% interval [−0.0154,+0.0749]. In spent 2023–2025 stress, overall improvement is −0.0231% [−0.0553,+0.0072], inconclusive; postseason predictions worsen 0.6862% [0.2513,1.1004]. Regular-season improvement is 0.0228% [0.0123,0.0331], including first-ten-appearance improvement 0.1364% [0.0619,0.2089] among previous postseason participants.

Retain the existing next-game OU board. A blanket additive playoff adjustment does not have consistent predictive support despite a lower matched observed rate and a small following-season benefit. Do not tune a smaller discount on these stress results. No 2026 targets were scored and no production model changed. One regression test verifies composition control, prior-season-only estimation, zero warm-up and no direct RS offset.

## Owner-directed elite-deflector diagnostic

`analyze_deflections_playoff_elite.py` is exploratory, not another selection gate. On 2020–2025, rank players with at least 200 observed RS defensive possessions into quartiles by their last filtered RS skill before the league's first playoff game. Include non-playoff players in the ranking universe; compare only matched RS/PS players. Aggregate phase contrasts with the same harmonic exposure weights as above. No 2026 labels enter.

Lowest through highest quartile matched RS→PS rates are 1.88→1.91, 2.53→2.46, 3.15→2.96, and 4.43→3.99 per100. The top-quartile drop is −0.55 in 2020–2022 and −0.32 in 2023–2025. Even against the updating next-game forecast, the top quartile averages −0.214/100 residual. A season-intercept weighted regression of that residual on pre-playoff skill has slope −0.0955 per additional skill unit/100, player-clustered 95% interval [−0.1451,−0.0370]. This is descriptive evidence of skill-dependent forecast bias; selection, regression to the mean, opponents and role remain potential explanations. It is not validated evidence for a correction.

Alex Caruso drops in each observed postseason (2020,2021,2022,2025); matched weighted phase rates 6.69→5.23. Matisse Thybulle drops in each observed postseason (2020–2022), 7.32→5.75. These are pair-weighted summaries, not simple pooled career rates; Thybulle's sample totals only 852 observed defensive possessions. Full player-season counts and rates live in ignored elite_player_seasons.csv and elite_report.json beside the earlier receipts. Preserve the incumbent and require a separate date-correct predictive gate before admitting any interaction.
